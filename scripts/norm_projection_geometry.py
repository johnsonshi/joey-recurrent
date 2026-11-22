"""Stream direct input projections from pinned artifacts. Only scalar comparisons leave the process."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


FIELDS = ("identity_sse", "donor_sse", "shift_ss", "anchor_ss", "target_ss")


def summarize(items):
    items = list(items)
    if not items:
        raise ValueError("empty geometry inventory")
    r = {k: math.fsum(x[k] for x in items) for k in FIELDS}
    if any(not math.isfinite(v) or v < 0 for v in r.values()):
        raise ValueError("invalid squared geometry")
    return {**r, "entries": sum(x["entries"] for x in items),
            "fraction_gap_removed": 1 - r["donor_sse"] / r["identity_sse"] if r["identity_sse"] else None,
            "relative_shift_norm": math.sqrt(r["shift_ss"] / r["anchor_ss"]) if r["anchor_ss"] else None}


def measure(shape, read, stored_anchor, stored_donor, *, tile=128):
    """read(lo, hi) returns the corresponding anchor/donor matrix row slabs."""
    import numpy as np
    rows, width = shape
    if not (all(type(i) is int for i in (tile, rows, width)) and 0 < tile <= 256
            and 0 < rows <= 16384 and 0 < width <= 4096):
        raise ValueError("invalid or unbounded matrix")
    sa, sb = (1. + np.asarray(s, dtype=np.float64) for s in (stored_anchor, stored_donor))
    if any(s.shape != (width,) or not np.isfinite(s).all() for s in (sa, sb)):
        raise ValueError("invalid norm coefficients")
    values = {k: [] for k in FIELDS}
    aa, bb, ab = np.zeros((3, width), dtype=np.float64)
    for lo in range(0, rows, tile):
        hi = min(lo + tile, rows)
        a, b = (np.asarray(x, dtype=np.float64) for x in read(lo, hi))
        if any(x.shape != (hi-lo, width) or not np.isfinite(x).all() for x in (a, b)):
            raise ValueError("invalid projection slab")
        aa += np.einsum("ij,ij->j", a, a)
        bb += np.einsum("ij,ij->j", b, b)
        ab += np.einsum("ij,ij->j", a, b)
        anchor, target, donor = a * sa, b * sb, a * sb
        for k, m in zip(FIELDS, (anchor-target, donor-target, donor-anchor, anchor, target)):
            values[k].append(float(np.sum(m*m, dtype=np.float64)))
    r = {k: math.fsum(v) for k, v in values.items()}
    # A separately accumulated column Gram matrix avoids reusing the direct error slabs.
    check = (aa*sa*sa + bb*sb*sb - 2*ab*sa*sb, (aa + bb - 2*ab)*sb*sb,
             aa*(sb-sa)**2, aa*sa*sa, bb*sb*sb)
    scale = max(r["anchor_ss"], r["target_ss"], r["shift_ss"], 1e-30)
    for k, col in zip(FIELDS, check):
        if not math.isclose(r[k], math.fsum(col), abs_tol=scale*2e-12, rel_tol=2e-12):
            raise ValueError("independent direct/Gram sums disagree")
    return summarize([{**r, "entries": rows*width}])


def main():
    import resource
    import signal
    import zipfile
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-dir", "compact-dir", "sidecar-dir", "wheel", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("compact-manifest-sha256", "sidecar-manifest-sha256", "protocol-sha256"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    signal.alarm(900); resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); os.umask(0o077)
    for n in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[n] = "1"
    import numpy as np
    from compact_spectrum import check_output, verify_pin, write_new
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root / "src"))
    from joey_research import compact as cp, pass_norms as pn
    from joey_research.native import MODEL_SHA256
    check_output(args.output)
    rulefile = root / "configs/compact/norm-projection-geometry.json"
    if cp._digest(rulefile) != args.protocol_sha256:
        raise ValueError("geometry protocol differs")
    rule = cp._json_file(rulefile)
    if (rule["schema"] != "joey-norm-projection-geometry/v1" or rule["width"] != pn.WIDTH
            or rule["layer_pairs"] != [[j, j+4] for j in range(8, 12)] or rule["stored_norm_offset"] != 1
            or rule["cpu"] != {"threads": 1, "row_tile": 128, "wall_seconds": 900}):
        raise ValueError("unsupported scope")
    dirs = (args.source_dir, args.compact_dir, args.sidecar_dir)
    if any(not p.is_absolute() or p.resolve(strict=True) != p for p in (*dirs, args.wheel)):
        raise ValueError("expected direct local paths")
    if len(set(dirs)) != 3 or any(p in args.output.parents or p in q.parents for p in dirs for q in dirs):
        raise ValueError("input and output areas must be disjoint")
    wl = cp._json_file(root / "locks/linux-inference-candidates.json")
    records = [rec for rec in wl["packages"] if rec["filename"] == args.wheel.name and rec["package"] == "transformers"]
    if len(records) != 1 or cp._digest(args.wheel) != records[0]["sha256"]:
        raise ValueError("unverified decoder wheel")
    with zipfile.ZipFile(args.wheel) as z:
        code = z.read("transformers/models/qwen3_5/modeling_qwen3_5.py")
    if hashlib.sha256(code).hexdigest() != rule["decoder_member_sha256"]:
        raise ValueError("not the reviewed one-plus-weight RMS implementation")
    side, vecs = pn.read_vectors(args.sidecar_dir, manifest_sha256=args.sidecar_manifest_sha256,
                                 compact_manifest_sha256=args.compact_manifest_sha256)
    sm = args.compact_dir / "compact.json"
    if cp._digest(sm) != args.compact_manifest_sha256:
        raise ValueError("compact manifest differs")
    comp = cp._json_file(sm)
    if (comp["plan"] != pn.BASE_PLAN or comp["anchor_source_layers"] != list(pn.KEPT)
            or comp["weights_sha256"] != side["base"]["weight_sha256"] or comp["repeat_layers"] != [8, 12]
            or comp["physical_depth"] != 20 or comp["source"]["weight_file_sha256"] != {"model.safetensors": MODEL_SHA256["model.safetensors"]}):
        raise ValueError("not the pinned earlier-cycle base")
    original, physical = (p / "model.safetensors" for p in dirs[:2])
    paths = (original, physical, sm, args.sidecar_dir/pn.MANIFEST, args.sidecar_dir/pn.WEIGHTS, args.wheel)
    versions = tuple(map(pn._version, paths))
    def decoded(raw):
        words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return (words << 16).view(np.float32).astype(np.float64)
    def bytes_for(f, tensor, offset, count):
        if tensor.dtype != "BF16" or not 0 <= offset <= offset+count <= tensor.nbytes:
            raise ValueError("invalid BF16 extent")
        f.seek(tensor.offset + offset)
        return cp._read(f, count)
    data = []; width = rule["width"]
    with original.open("rb") as src, physical.open("rb") as base:
        verify_pin(src, MODEL_SHA256["model.safetensors"]); verify_pin(base, comp["weights_sha256"])
        source, _ = cp._shard(original); kept, _ = cp._shard(physical)
        for j, k in rule["layer_pairs"]:
            anchor, donor = [cp._LAYERS + str(i) + "." for i in (j, k)]
            kind = "full_attention" if j == 11 else "linear_attention"
            for norm, category in zip(pn.NORMS, (kind, "mlp")):
                tensors = (kept[anchor+norm], source[anchor+norm], source[donor+norm])
                if any(t.shape != (width,) for t in tensors):
                    raise ValueError("unexpected norm layout")
                u, ref, v = (bytes_for(f, t, 0, t.nbytes) for f, t in zip((base, src, src), tensors))
                if u != ref or v != vecs["visit2.layers." + str(j) + "." + norm]:
                    raise ValueError("norm data not equal to physical anchor or source donor")
                for name in rule["projections"][category]:
                    small, early, late = kept[anchor+name], source[anchor+name], source[donor+name]
                    if not small.shape == early.shape == late.shape or len(small.shape) != 2 or small.shape[1] != width:
                        raise ValueError("direct source projections do not match")
                    def load(lo, hi):
                        off = 2*lo*width; count = 2*(hi-lo)*width
                        a = bytes_for(base, small, off, count)
                        if a != bytes_for(src, early, off, count):
                            raise ValueError("compact projection differs from original anchor")
                        return (decoded(a).reshape(-1, width), decoded(bytes_for(src, late, off, count)).reshape(-1, width))
                    r = measure(small.shape, load, decoded(u), decoded(v), tile=rule["cpu"]["row_tile"])
                    data.append({"layers": [j, k], "norm": norm, "projection": name, "shape": list(small.shape), **r})
    if len(data) != 23 or tuple(map(pn._version, paths)) != versions:
        raise ValueError("changed inputs or incomplete geometry coverage")
    out = {"schema": "joey-measured-norm-projection-geometry/v1", "protocol_sha256": args.protocol_sha256,
           "source_sha256": MODEL_SHA256["model.safetensors"], "base_sha256": comp["weights_sha256"],
           "compact_manifest_sha256": args.compact_manifest_sha256, "sidecar_manifest_sha256": args.sidecar_manifest_sha256,
           "sidecar_weights_sha256": side["weights_sha256"], "decoder_member_sha256": rule["decoder_member_sha256"],
           "numpy_version": np.__version__, "independent_gram_check": True, "projections": data, "pooled": summarize(data),
           "by_norm": {n: summarize(r for r in data if r["norm"] == n) for n in pn.NORMS},
           "by_anchor": {str(j): summarize(r for r in data if r["layers"][0] == j) for j in range(8, 12)},
           "inference_performed": False, "weights_created": False, "quality_measured": False}
    raw = json.dumps(out, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    write_new(args.output, raw)
    print("Direct projection geometry verified. SHA-256:", hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError, KeyError, RuntimeError) as exc:
        print("Geometry measurement stopped:", type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
