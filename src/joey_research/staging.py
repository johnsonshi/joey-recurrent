"""Verify offline staging evidence without fetching, installing, or executing artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from urllib.parse import urlsplit

from .evaluation import input_sha256
from .native import MODEL_REVISION, MODEL_SHA256, ONEJEV_REVISION, QEV_SHA256

BASE_DIGEST = "sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6"
_MANIFEST = {"application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"}
_CONFIG = {"application/vnd.oci.image.config.v1+json", "application/vnd.docker.container.image.v1+json"}
_LAYERS = {"application/vnd.oci.image.layer.v1.tar", "application/vnd.oci.image.layer.v1.tar+gzip",
           "application/vnd.oci.image.layer.v1.tar+zstd", "application/vnd.docker.image.rootfs.diff.tar.gzip"}
_SHA = re.compile("[0-9a-f]{64}")
_OCI_SHA = re.compile("sha256:[0-9a-f]{64}")


class StageError(ValueError):
    pass


def _check(ok, message):
    if not ok:
        raise StageError(message)


def _unique(pairs):
    out = {}
    for k, v in pairs:
        _check(k not in out, "duplicate field in evidence")
        out[k] = v
    return out


def _document(raw, expected):
    _check(isinstance(raw, bytes) and len(raw) <= 8 * 1024 * 1024, "invalid evidence size")
    _check(hashlib.sha256(raw).hexdigest() == expected, "document does not match trusted pin")
    out = json.loads(raw, object_pairs_hook=_unique)
    _check(isinstance(out, dict), "expected an object")
    return out


def _descriptor(d, media_types):
    _check(isinstance(d, dict) and set(d) == {"digest","mediaType","size"}, "unsupported OCI descriptor or external layer")
    _check(d["mediaType"] in media_types and type(d["size"]) is int and d["size"] > 0
           and isinstance(d["digest"], str) and _OCI_SHA.fullmatch(d["digest"]), "invalid OCI descriptor")
    return d["digest"], d["size"], d["mediaType"]


def _image(manifest, config, expected):
    _check(isinstance(expected, str) and _OCI_SHA.fullmatch(expected), "image digest required")
    doc = _document(manifest, expected[7:])
    _check(doc.get("schemaVersion") == 2 and doc.get("mediaType") in _MANIFEST, "expected single-platform image manifest")
    config_sha, size, _ = _descriptor(doc.get("config"), _CONFIG)
    _check(isinstance(config, bytes) and len(config) == size, "image config length differs")
    cfg = _document(config, config_sha[7:])
    _check((cfg.get("os"), cfg.get("architecture")) == ("linux", "amd64"), "unsupported image platform")
    layers = doc.get("layers")
    _check(isinstance(layers, list) and 1 <= len(layers) <= 128, "invalid image layer set")
    layers = [_descriptor(layer, _LAYERS) for layer in layers]
    rootfs = cfg.get("rootfs", {})
    ids = rootfs.get("diff_ids")
    _check(rootfs.get("type") == "layers" and isinstance(ids, list) and len(ids) == len(layers)
           and all(isinstance(x, str) and _OCI_SHA.fullmatch(x) for x in ids), "unbound uncompressed layer digests")
    return layers, ids


def verify_image_pair(base_manifest, base_config, worker_manifest, worker_config, *, worker_digest):
    """Check same registry-resolved platform and base ancestry. Caller verifies the registry principal and scope."""
    base, base_ids = _image(base_manifest, base_config, BASE_DIGEST)
    worker, ids = _image(worker_manifest, worker_config, worker_digest)
    _check(len(worker) > len(base) and worker[:len(base)] == base and ids[:len(base)] == base_ids,
           "worker does not extend the exact pinned base")
    return {"base": BASE_DIGEST, "worker": worker_digest, "platform": "linux/amd64", "base_layers": len(base),
            "added_layers": len(worker) - len(base)}


def _asset(entry):
    fields = {"path","sha256","bytes","source","revision","license"}
    _check(isinstance(entry, dict) and set(entry) == fields, "invalid artifact receipt fields")
    name = entry["path"]
    _check(isinstance(name, str) and len(name) < 300 and re.fullmatch(r"[A-Za-z0-9_./+-]+", name), "invalid artifact path")
    rel = PurePosixPath(name)
    _check(not rel.is_absolute() and rel.as_posix() == name and len(rel.parts) == 2 and
           not any(p.startswith(".") for p in rel.parts), "artifact escaped its collection")
    scope, filename = rel.parts
    if scope == "model":
        _check(filename in MODEL_SHA256 and entry["sha256"] == MODEL_SHA256[filename], "model asset is not pinned")
    elif scope == "qev":
        _check(filename in QEV_SHA256 and entry["sha256"] == QEV_SHA256[filename], "native source is not pinned")
    elif scope == "notices":
        _check(filename in ("model-LICENSE", "model-NOTICE", "OneJev-LICENSE"), "unexpected notice")
    elif scope == "wheelhouse":
        _check(filename.endswith(".whl"), "source installations are not accepted")
    elif scope == "inputs":
        _check(filename == "discovery.jsonl", "only discovery inputs are admitted")
    else:
        raise StageError("unknown artifact collection")
    _check(type(entry["bytes"]) is int and 0 < entry["bytes"] <= 24 * 2**30 and
           isinstance(entry["sha256"], str) and _SHA.fullmatch(entry["sha256"]), "invalid artifact size or digest")
    _check(isinstance(entry["license"], str) and 1 <= len(entry["license"]) <= 256 and
           isinstance(entry["revision"], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}", entry["revision"])
           and entry["revision"].lower() not in {"head","main","master","latest"}, "license and immutable revision required")
    _check(isinstance(entry["source"], str), "source provenance required")
    if scope == "inputs":
        _check(entry["source"] == "study:generated", "unexpected discovery provenance")
    else:
        u = urlsplit(entry["source"])
        _check(u.scheme == "https" and u.hostname in ("github.com", "huggingface.co", "pypi.org", "files.pythonhosted.org")
               and not (u.username or u.password or u.port or u.query or u.fragment) and u.path.startswith("/"),
               "expected a public origin without credentials or query strings")
        if scope in ("model","qev","notices"):
            is_model = scope == "model" or (scope == "notices" and filename.startswith("model-"))
            origin, revision = (("https://huggingface.co/baselabs/joey-1", MODEL_REVISION) if is_model
                                else ("https://github.com/OmniJev/OneJev", ONEJEV_REVISION))
            _check((entry["source"], entry["revision"]) == (origin, revision), "unexpected pinned project provenance")
    return name


def _verify_file(file, size, digest):
    fd = os.open(file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        st = os.fstat(fd)
        _check(stat.S_ISREG(st.st_mode) and st.st_size == size and st.st_nlink == 1, "artifact is not a materialized regular file")
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
        _check(value.hexdigest() == digest, "artifact checksum mismatch")


def _verify_discovery(file):
    seen = set()
    fd = os.open(file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        _check(os.fstat(fd).st_size <= 64 * 1024 * 1024, "discovery shard is too large")
        for line in stream:
            row = json.loads(line, object_pairs_hook=_unique)
            input_sha256(row)
            _check(row["id"] not in seen and len(seen) < 20000, "duplicate or excessive discoveries")
            seen.add(row["id"])
    _check(seen, "empty discovery shard")
    return len(seen)


def verify_payload(root: Path, receipt_bytes: bytes, *, receipt_sha256: str, discovery_sha256: str) -> dict:
    """Check a materialized, read-only staging tree. Pins come from the coordinator, not the mounted payload."""
    receipt = _document(receipt_bytes, receipt_sha256)
    _check(set(receipt) == {"schema","assets"} and receipt["schema"] == "joey-discovery-bundle/v1"
           and isinstance(receipt["assets"], list) and 1 <= len(receipt["assets"]) <= 512, "invalid payload receipt")
    assets = {}
    for item in receipt["assets"]:
        name = _asset(item)
        _check(name not in assets, "duplicate artifact")
        assets[name] = item
    needed = ({"model/"+p for p in MODEL_SHA256} | {"qev/"+p for p in QEV_SHA256} |
              {"notices/model-LICENSE","notices/model-NOTICE","notices/OneJev-LICENSE","inputs/discovery.jsonl"})
    _check(needed <= assets.keys() and any(f.startswith("wheelhouse/") for f in assets), "incomplete staged bundle")
    _check(assets["inputs/discovery.jsonl"]["sha256"] == discovery_sha256, "unexpected discovery split")
    total = sum(x["bytes"] for x in assets.values())
    _check(total <= 40 * 2**30, "bundle exceeds staging budget")
    root = Path(root).absolute()
    _check(root.resolve(strict=True) == root and root.is_dir(), "bundle must be a real directory")
    expected_dirs = {str(PurePosixPath(name).parent) for name in assets}
    _check({p.name for p in root.iterdir()} == expected_dirs, "unexpected files or directories at staging root")
    for name in expected_dirs:
        directory = root/name
        _check(not directory.is_symlink() and directory.is_dir(), "unapproved staging directory")
        expected = {x for x in assets if x.startswith(name+"/")}
        _check({str(p.relative_to(root)) for p in directory.iterdir()} == expected, "unexpected or missing payload files")
    for name, item in assets.items():
        _verify_file(root/name, item["bytes"], item["sha256"])
    return {"file_count": len(assets), "bytes": total, "decisions": _verify_discovery(root/"inputs/discovery.jsonl"),
            "receipt_sha256": receipt_sha256, "discovery_sha256": discovery_sha256}
