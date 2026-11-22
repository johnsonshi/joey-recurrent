"""Measure corresponding MLP matrices in a locally staged, pinned checkpoint on CPU.

Write only scalar diagnostics to a new file in a private directory.
measure() and summarize() also work on invented matrices without a checkpoint.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

T0 = time.monotonic()
WALL_SECONDS = 1800
THREAD_VARS = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS",
               "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT")
if __name__ == "__main__":
    import resource
    import signal

    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(WALL_SECONDS)  # Hard stop even inside native BLAS.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    for var in THREAD_VARS:
        os.environ[var] = "1"

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if __name__ == "__main__" and exc.name == "numpy":
        raise SystemExit("The offline geometry diagnostic requires an environment with NumPy.") from None
    raise

LOCK = Path(__file__).resolve().parents[1] / "src/joey_research/native.py"
RANKS = (1, 4, 16, 64)
PROJS = ("gate", "up", "down")
SEED, BLOCK, EXTRA, POWER = 20261121, 128, 16, 2
MEM_LIMIT = 1 << 30


def require(condition, message):
    if not condition:
        raise ValueError(message)


def model_pin():
    # Read the reviewed literal without importing the native stack or touching its sidecars.
    assignments = [node.value for node in ast.parse(LOCK.read_text(encoding="utf-8")).body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                   t.id == "MODEL_SHA256" for t in node.targets)]
    require(len(assignments) == 1, "ambiguous or missing model lock")
    return ast.literal_eval(assignments[0])["model.safetensors"]


def verify_pin(stream, expected):
    require(re.fullmatch("[0-9a-f]{64}", expected), "expected a full SHA-256")
    sha = hashlib.sha256()
    stream.seek(0)
    for chunk in iter(lambda: stream.read(4 << 20), b""):
        sha.update(chunk)
    require(sha.hexdigest() == expected, "checkpoint pin mismatch")
    return sha.hexdigest()


def matched_pairs(tensors, layers, cycle, suffixes):
    suffix = {p: f"mlp.{p}_proj.weight" for p in PROJS}
    require(set(suffix.values()) == suffixes and len(cycle) == 4, "unsupported compact schema")
    first = layers + "8."
    h, = tensors[first + "input_layernorm.weight"].shape
    m, _ = tensors[first + suffix["gate"]].shape
    for pos, kind in enumerate(cycle):
        ids = (8 + pos, 12 + pos)
        prefixes = [layers + str(i) + "." for i in ids]
        for prefix in prefixes:
            linear = any(k.startswith(prefix + "linear_attn.") for k in tensors)
            full = any(k.startswith(prefix + "self_attn.") for k in tensors)
            require(linear != full and kind == ("linear_attention" if linear else "full_attention"),
                    "layer type disagrees with portable cycle")
            require(tensors[prefix + "input_layernorm.weight"].shape == (h,), "hidden size mismatch")
        for p in PROJS:
            a, b = (tensors[k + suffix[p]] for k in prefixes)
            shape = (h, m) if p == "down" else (m, h)
            require(a.shape == b.shape == shape and a.dtype == b.dtype == "BF16",
                    "projection shape/dtype mismatch")
            yield ids, kind, p, a, b


def read_rows(stream, tensor, s):
    n = tensor.shape[1]
    size = (s.stop - s.start) * n * 2
    stream.seek(tensor.offset + s.start * n * 2)
    raw = stream.read(size)
    require(len(raw) == size, "short tensor read")
    words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    words <<= 16
    return words.view(np.float32).astype(np.float64).reshape(-1, n)


def bf16(x):
    """RNE via FP32, returned in FP64 for scoring, never serialized."""
    bits = x.astype(np.float32).view(np.uint32)
    bits += np.uint32(0x7fff) + ((bits >> 16) & 1)
    bits &= np.uint32(0xffff0000)
    out = bits.view(np.float32).astype(np.float64)
    require(np.isfinite(out).all(), "nonfinite BF16 round trip")
    return out


def blocks(m, block):
    return (slice(i, min(i + block, m)) for i in range(0, m, block))


def inner(a, b):
    return float(np.einsum("ij,ij->", a, b))


def factors(d, rank, seed, block):
    m, n = d.shape
    width = min(rank + EXTRA, m, n)

    def right(x):
        y = np.empty((m, width))
        for s in blocks(m, block):
            np.matmul(d[s], x, out=y[s])
        return y

    def left(x):
        z = np.zeros((n, width))
        for s in blocks(m, block):
            z += d[s].T @ x[s]
        return z

    def orth(x):
        return np.linalg.qr(x, mode="reduced")[0]

    rng = np.random.Generator(np.random.PCG64(seed))
    q = orth(right(rng.standard_normal((n, width))))
    for _ in range(POWER):
        q = orth(right(orth(left(q))))
    u, sigma, vt = np.linalg.svd(left(q).T, full_matrices=False)
    scale = np.sqrt(sigma[:rank])
    return (q @ u[:, :rank]) * scale, scale[:, None] * vt[:rank]


def budget(m, n, rank, block):
    k = min(rank + EXTRA, m, n)
    return (128 << 20) + 8 * (m*n + 16*(m+n)*k + 16*min(block, m)*n)


@dataclass
class Score:
    shape: tuple
    a2: float
    b2: float
    ab: float
    d2: float
    midpoint_round2: float
    errors2: dict


def measure(shape, read, ranks=RANKS, *, seed=SEED, block=BLOCK):
    """read(row_slice) supplies two corresponding real blocks. Only scalars survive."""
    m, n = shape
    require(min(m, n, block) > 0 and ranks and all(1 <= r <= min(shape) for r in ranks),
            "rank or dimension out of bounds")
    require(budget(m, n, max(ranks), block) <= MEM_LIMIT, "memory budget exceeded")
    d = np.empty((m, n), dtype=np.float64)
    aa = bb = ab = dd = mm = 0.
    for s in blocks(m, block):
        a, b = (np.asarray(v, dtype=np.float64) for v in read(s))
        require(a.shape == b.shape == d[s].shape and np.isfinite(a).all() and np.isfinite(b).all(),
                "nonfinite or wrong-shaped block")
        aa += inner(a, a)
        bb += inner(b, b)
        ab += inner(a, b)
        np.subtract(b, a, out=d[s])
        d[s] *= 0.5
        dd += inner(d[s], d[s])
        mid = (a + b) * 0.5
        mid -= bf16(mid)
        mm += inner(mid, mid)
    del a, b, mid
    errs = {r: [0., 0.] for r in ranks}
    if dd:
        l, t = factors(d, max(ranks), seed, block)
        for j, (x, y) in enumerate(((l, t), (bf16(l), bf16(t)))):
            for r in ranks:
                for s in blocks(m, block):
                    e = x[s, :r] @ y[:r] - d[s]
                    errs[r][j] += inner(e, e)
    return Score(shape, aa, bb, ab, dd, mm, errs)


def summarize(scores):
    """Concatenation (sums of squares), not a mean of pair percentages."""
    aa, bb, ab, dd, mm = (math.fsum(getattr(s, key) for s in scores)
                          for key in ("a2", "b2", "ab", "d2", "midpoint_round2"))
    base = sum(2 * math.prod(s.shape) for s in scores)
    stride = sum(2 * sum(s.shape) for s in scores)

    def relative(error2, total2):
        return math.sqrt(error2 / total2) if total2 else None

    by_rank = {}
    for r in scores[0].errors2:
        e, eq = (math.fsum(s.errors2[r][j] for s in scores) for j in (0, 1))
        by_rank[str(r)] = {
            "delta_relative_frob_error": relative(e, dd),
            "delta_energy_achieved": 1 - e/dd if dd else None,
            "bf16_delta_energy_achieved": 1 - eq/dd if dd else None,
            "bf16_pair_relative_frob_error": relative(2*(eq + mm), aa + bb),
            "bf16_factor_bytes": r*stride, "bf16_base_plus_factor_bytes": base + r*stride,
        }
    return {
        "pair_count": len(scores), "cosine": ab / (math.sqrt(aa)*math.sqrt(bb)) if aa and bb else None,
        "relative_frob_difference_over_A": relative(4*dd, aa),
        "frobenius_squared": {"A": aa, "B": bb, "half_difference": dd, "midpoint_roundoff": mm},
        "bf16_base_only_pair_relative_frob_error": relative(2*(dd + mm), aa + bb),
        "bf16_base_bytes": base, "bf16_two_originals_bytes": 2*base, "by_rank": by_rank,
    }


def check_output(path):
    require(path.is_absolute() and path.parent.resolve(strict=True) == path.parent
            and not path.is_symlink() and path.name not in ("", ".", ".."), "invalid output path")
    st = path.parent.stat()
    require(path.parent.is_dir() and st.st_uid == os.geteuid() and not st.st_mode & 0o077,
            "use an existing private, owner-only output directory")
    if path.exists():
        raise FileExistsError("refusing to overwrite an existing geometry result")


def write_new(path, payload):
    check_output(path)
    fd, tmp = tempfile.mkstemp(prefix=".spectrum-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(tmp, path)  # Atomic publication; cannot replace a concurrent writer's file.
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        Path(tmp).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True, help="absolute path to the read-only original safetensors file")
    ap.add_argument("--output", type=Path, required=True, help="fresh absolute output filename in a private directory")
    ap.add_argument("--seed", type=int, default=SEED, help="first PCG64 seed; increment per pair")
    ap.add_argument("--block-rows", type=int, default=BLOCK, help="row tile size (1..256)")
    args = ap.parse_args()
    require(0 <= args.seed < 2**63 and 1 <= args.block_rows <= 256, "seed/block out of bounds")
    weights = args.weights
    require(weights.is_absolute() and weights.resolve(strict=True) == weights
            and weights.is_file() and not (weights.stat().st_mode & 0o222), "need a direct, read-only checkpoint file")
    check_output(args.output)
    sys.path.insert(0, str(LOCK.parents[1]))
    from joey_research import compact

    pin = model_pin()
    require(pin == compact._PIN_WEIGHTS, "portable and native pins disagree")
    scores, pairs = [], []
    with weights.open("rb") as f, np.errstate(over="raise", invalid="raise", divide="raise"):
        orig = os.fstat(f.fileno())

        def unchanged():
            now = weights.stat()
            require(os.path.samestat(orig, now) and (orig.st_size, orig.st_mtime_ns, orig.st_ctime_ns)
                    == (now.st_size, now.st_mtime_ns, now.st_ctime_ns), "checkpoint changed")

        digest = verify_pin(f, pin)
        tensors, _ = compact._shard(weights)  # Header only; never open a config, index or sidecar.
        unchanged()
        scope = list(matched_pairs(tensors, compact._LAYERS, compact._CYCLE, compact._MERGE_SUFFIXES))
        for i, (ids, kind, proj, a, b) in enumerate(scope):
            def read(s):
                return read_rows(f, a, s), read_rows(f, b, s)

            score = measure(a.shape, read, seed=args.seed + i, block=args.block_rows)
            scores.append(score)
            pairs.append(dict(layers=ids, layer_type=kind, projection=proj, rows=a.shape[0],
                              columns=a.shape[1], all_finite=True, seed=args.seed + i,
                              **summarize([score])))
        unchanged()
    aggregate = summarize(scores)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "schema": "compact-spectrum/v1", "status": "complete", "upstream_sha256": digest,
        "scope": "Zero-based cycles 2 and 3; same position, attention family and projection only. "
                 "Gate/up=(intermediate,hidden), down=(hidden,intermediate); unpermuted stored axes.",
        "method": {
            "definition": "A=cycle 2, B=cycle 3, M=(A+B)/2, D=(B-A)/2. Ahat=Mhat-LR, Bhat=Mhat+LR.",
            "algorithm": "Gaussian right sketch of D; reduced QR; stabilized subspace iterations "
                         "(QR after each transposed/forward multiply); thin SVD only of Q.T@D. "
                         "Nested truncations of this sketch, not optimal SVD of D.",
            "ranks": RANKS, "rng": "NumPy PCG64 standard_normal; seed+3*position+[gate,up,down].index",
            "seed": args.seed, "oversampling": EXTRA, "power_iterations": POWER,
            "sketch_width": max(RANKS) + EXTRA, "row_block": args.block_rows,
            "precision": "BF16 little-endian source decoded exactly through FP32 to FP64. "
                         "All arithmetic/reductions FP64. BF16 round trips via FP32 then nearest-even. "
                         "sqrt(sigma) split between factors; BF16 variant rounds M,L,R separately, "
                         "scores with FP64 products, no output re-round.",
            "scoring": "Direct row-block residual e=||D-LR||_F^2 at each rank, also with BF16 factors. "
                       "Energy=1-e/||D||_F^2. Two-member BF16 error squared=2*(e_bf16+||M-M_bf16||_F^2). "
                       "Relative difference=||B-A||_F/||A||_F. Undefined ratios=null. Aggregate "
                       "concatenates matrices (sums inner products and squared errors before dividing).",
            "bytes": "Pair: originals 4mn, common BF16 midpoint 2mn, ONE shared BF16 factor pair "
                     "2r(m+n); signs implicit by cycle, no stored singular values. Logical payload only: "
                     "excludes headers, alignment, names, sign/routing metadata, other tensors, "
                     "workspace, activations and caches. Not a measured checkpoint or VRAM saving.",
        },
        "execution": {
            "device": "CPU", "numpy": np.__version__, "python": sys.version.split()[0],
            "blas": getattr(np.__config__, "CONFIG", {}).get("Build Dependencies", {})
                    .get("blas", {}).get("name", "not reported"),
            "thread_caps_before_numpy_import": {k: os.environ[k] for k in THREAD_VARS},
            "pair_concurrency": 1, "hard_wall_limit_s": WALL_SECONDS,
            "elapsed_s_including_hash": time.monotonic() - T0,
            "memory_budget_bytes": MEM_LIMIT,
            "memory_model": "One FP64 D, row tiles and skinny workspaces; admission estimate "
                            "128 MiB + 8*(mn+16*(m+n)*k+16*min(block,m)*n), k=sketch width.",
            "max_admission_estimate_bytes": max(budget(*s.shape, max(RANKS), args.block_rows) for s in scores),
            "peak_process_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        },
        "pairs": pairs, "aggregate": aggregate,
    }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_new(args.output, payload.encode("utf-8"))
    print(f"Wrote geometry for {len(scores)} matched MLP pairs. No weights or factors exported.")


if __name__ == "__main__":
    try:
        main()
    except FileExistsError:
        raise SystemExit("Refusing to overwrite an existing geometry result.") from None
    except OSError as exc:
        raise SystemExit(f"Geometry input/output error (errno {exc.errno}).") from None
