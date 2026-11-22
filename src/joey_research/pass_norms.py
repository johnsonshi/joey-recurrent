"""Physical, source-verified second-visit norm vectors for the earlier-cycle select-20 export."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import tempfile

from . import compact as cp
from .native import LAYER_TYPES, MODEL_REVISION, MODEL_SHA256


WEIGHTS = "second-visit.safetensors"
MANIFEST = "second-visit.json"
WIDTH = 2048
LIMIT = 65536
KEPT = (*range(12), *range(16, 24))
BASE_PLAN = {"format":"joey-compact/v1", "groups":[{"cycles":[i]} for i in (0,1,2,4,5)], "repeat_cycles":[2,3]}
NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight")


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pin(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _directory(p):
    p = Path(p).absolute()
    _check(p.resolve(strict=True) == p and p.is_dir(), "expected a direct local directory")
    return p


def _private(p):
    s = p.stat()
    _check(s.st_uid == os.geteuid() and not s.st_mode & 0o077, "output parent must be owner-only")


def _pairs():
    for physical in range(8, 12):
        for name in NORMS:
            yield (f"visit2.layers.{physical}.{name}", f"{cp._LAYERS}{physical}.{name}",
                   f"{cp._LAYERS}{physical+4}.{name}")


def _vector(t, width):
    _check(t.dtype == "BF16" and t.shape == (width,) and t.nbytes == 2*width, "wrong norm vector layout")
    with t.file.open("rb") as source:
        source.seek(t.offset)
        b = cp._read(source, t.nbytes)
    _check(all(math.isfinite(v) for v in cp._values(b, "BF16")), "nonfinite normalization vector")
    return b


def _read_weight_file(path, width):
    _check(cp._regular(path).stat().st_size <= LIMIT, "excessive normalization sidecar size")
    tensors, metadata = cp._shard(path)
    _check(metadata == {"format":"pt"} and set(tensors) == {k for k,_,_ in _pairs()},
           "unexpected tensors or aliases in normalization sidecar")
    return {k:_vector(t, width) for k,t in tensors.items()}


def _version(path):
    _check(cp._regular(path).resolve(strict=True) == path, "indirect input")
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _base_info(base, pin):
    _check(_pin(pin) and cp._digest(base / "compact.json") == pin, "compact manifest pin mismatch")
    m = cp.verify_compact(base)
    _check(m["plan"] == BASE_PLAN and m["anchor_source_layers"] == list(KEPT) and m["repeat_layers"] == [8,12]
           and m["physical_depth"] == 20 and m["source"]["kind"] == "pinned-joey-1"
           and m["source"]["revision"] == MODEL_REVISION
           and m["source"]["weight_file_sha256"] == {"model.safetensors":MODEL_SHA256["model.safetensors"]},
           "requires the byte-preserving earlier-cycle select-20 payload")
    return {"compact_manifest_sha256":pin, "weight_sha256":m["weights_sha256"],
            "weight_file_bytes":m["storage"]["weight_file_bytes"],
            "stored_elements":m["storage"]["full"]["stored_parameter_elements"]}


def _manifest(base, full_bytes, weight_bytes, vectors):
    return {"format":"joey-second-visit-norms/v1", "source_revision":MODEL_REVISION,
            "source_weights_sha256":MODEL_SHA256["model.safetensors"], "source_weight_file_bytes":full_bytes,
            "base":base, "physical_window":[8,12], "donor_source_layers":[12,13,14,15], "width":WIDTH,
            "dtype":"BF16", "weight_file":WEIGHTS, "weights_sha256":_sha(weight_bytes),
            "vector_sha256":{k:_sha(v) for k,v in vectors.items()},
            "extra_elements":8*WIDTH, "extra_weight_file_bytes":len(weight_bytes),
            "combined_stored_elements":base["stored_elements"] + 8*WIDTH,
            "combined_weight_file_bytes":base["weight_file_bytes"] + len(weight_bytes)}


def read_vectors(bundle, *, manifest_sha256, compact_manifest_sha256):
    """Reopen a sidecar using independent pins. These eight vectors are not a standalone checkpoint."""
    root = _directory(bundle)
    _check({p.name for p in root.iterdir()} == {MANIFEST,WEIGHTS} and _pin(manifest_sha256)
           and _pin(compact_manifest_sha256), "invalid sidecar inventory or missing trusted pins")
    mp, wp = root / MANIFEST, root / WEIGHTS
    before = _version(mp), _version(wp)
    _check(mp.stat().st_size <= 16384 and wp.stat().st_size <= LIMIT and cp._digest(mp) == manifest_sha256,
           "unbound or oversized sidecar")
    m = cp._json_file(mp); base = m.get("base")
    _check(isinstance(base,dict) and set(base) == {"compact_manifest_sha256","weight_sha256","weight_file_bytes","stored_elements"}
           and base["compact_manifest_sha256"] == compact_manifest_sha256 and _pin(base["weight_sha256"])
           and all(type(base[k]) is int and base[k] > 0 for k in ("weight_file_bytes","stored_elements"))
           and type(m.get("source_weight_file_bytes")) is int, "invalid base payload binding")
    data = wp.read_bytes(); vectors = _read_weight_file(wp, WIDTH)
    expected = _manifest(base, m["source_weight_file_bytes"], data, vectors)
    _check(cp._encode(m) == cp._encode(expected) and m["combined_weight_file_bytes"] < m["source_weight_file_bytes"]
           and before == (_version(mp),_version(wp)), "sidecar layout, derived sizes or hashes disagree")
    return m, vectors


def _publish_new(root, files):
    root.mkdir(mode=0o700, exist_ok=False)
    for name in (WEIGHTS, MANIFEST):  # Manifest last; do not treat a partial directory as a release.
        fd = os.open(root/name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(files[name]); f.flush(); os.fsync(f.fileno())
    for directory in (root, root.parent):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)


def source_readback(source_dir, compact_dir, bundle, *, compact_manifest_sha256, create=False):
    """Create a fresh bundle or verify an existing one directly against both physical input artifacts."""
    _check(type(create) is bool, "expected a boolean creation mode")
    source,base = _directory(source_dir),_directory(compact_dir)
    arg = Path(bundle).absolute(); parent = _directory(arg.parent); _private(parent)
    target = parent/arg.name
    _check(arg == target and arg.name not in (".","..","") and source != base and
           all(not a.is_relative_to(b) for a,b in ((source,base),(base,source),(target,source),(source,target),
                                                   (target,base),(base,target))), "source, base and sidecar must be disjoint")
    _check(not os.path.lexists(target) if create else target.resolve(strict=True)==target and target.is_dir(),
           "output must be fresh for creation, or an existing direct directory for verification")
    src = source/"model.safetensors"; bw = base/"model.safetensors"
    inputs = (src,source/"config.json",bw,base/"config.json",base/"compact.json")
    versions = tuple(map(_version,inputs))
    _check(cp._digest(src) == MODEL_SHA256["model.safetensors"]
           and cp._digest(source/"config.json") == MODEL_SHA256["config.json"], "not the original pinned source")
    config = cp._json_file(source/"config.json")["text_config"]
    _check(config["hidden_size"]==WIDTH and config["layer_types"]==list(LAYER_TYPES*6),
           "unexpected source layer family or width")
    info = _base_info(base,compact_manifest_sha256)
    original,_ = cp._shard(src); physical,_ = cp._shard(bw)
    outputs={}; retained={}; donors={}
    for key,anchor,donor in _pairs():
        retained[key]=_vector(physical[anchor],WIDTH)
        _check(retained[key] == _vector(original[anchor],WIDTH), "retained normalization differs from its source")
        donors[key]=_vector(original[donor],WIDTH);outputs[key]=cp.OutputTensor((original[donor],))
    with tempfile.TemporaryDirectory(dir=parent,prefix=".norm-readback-") as tmp:
        wp=Path(tmp)/WEIGHTS
        wh=cp._write_weights(wp,outputs,{})
        _check(_read_weight_file(wp,WIDTH)==donors and wp.stat().st_size<=LIMIT, "generated sidecar differs from donor")
        payload=wp.read_bytes()
    m=_manifest(info,src.stat().st_size,payload,donors); mb=cp._encode(m,pretty=True); pin=_sha(mb)
    _check(_sha(payload)==wh and m["combined_weight_file_bytes"]<m["source_weight_file_bytes"]
           and versions==tuple(map(_version,inputs)), "source changed or combined file size does not shrink")
    if create:_publish_new(target,{WEIGHTS:payload,MANIFEST:mb})
    verified,actual=read_vectors(target,manifest_sha256=pin,compact_manifest_sha256=compact_manifest_sha256)
    _check(verified==m and actual==donors and versions==tuple(map(_version,inputs)),
           "reopened artifact does not reproduce the original donor bytes")
    stats={}
    for k in sorted(actual):
        changes=[b-a for a,b in zip(cp._values(retained[k],"BF16"),cp._values(actual[k],"BF16"),strict=True)]
        stats[k]={"different_elements":sum(v!=0. for v in changes), "max_abs_stored_coefficient_difference":max(map(abs,changes))}
    changed=sum(s["different_elements"] for s in stats.values())
    return {"manifest_sha256":pin, "manifest":m, "vectors":stats, "different_elements":changed,
            "numerically_distinct":changed>0, "source_bytes_reopened":True, "model_inference_performed":False}
