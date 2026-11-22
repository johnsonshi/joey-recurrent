"""CPU-only alignment of one frozen SwiGLU layer pair. Arrays in, statistics out."""

from __future__ import annotations

import hashlib
import heapq
import math
import operator
from collections.abc import Mapping

import numpy as np


PROJECTIONS = ("gate", "up", "down")
_TILE_LIMIT = 4 << 20
_STATE_LIMIT = 128 << 20
_PROBE_WIDTH = 8


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    value = operator.index(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def coordinate_split(hidden_size, *, seed):
    """Return disjoint, equal (fit, audit) index arrays, both in draw order."""
    h = _integer(hidden_size, "hidden_size", 2)
    if h % 2:
        raise ValueError("equal, exhaustive halves require an even hidden size")
    rng = np.random.Generator(np.random.PCG64(_integer(seed, "seed", 0)))
    order = rng.permutation(h)
    return order[:h // 2], order[h // 2:]


def _channel_views(layer):
    if not isinstance(layer, Mapping):
        raise TypeError("expected a mapping of gate, up and down arrays")
    views = {}
    for name in PROJECTIONS:
        a = layer[name]
        if not isinstance(a, np.ndarray) or np.ma.isMaskedArray(a):
            raise TypeError(f"{name} must be an unmasked ndarray or memmap")
        if a.ndim != 2 or a.dtype.kind != "f" or a.dtype.itemsize != 4:
            raise ValueError(f"{name} must be a two-dimensional float32 matrix")
        a = np.asarray(a)  # Strip subclasses without converting the weight buffer.
        views[name] = a.T if name == "down" else a
    shape = views["gate"].shape
    if min(shape) < 1 or any(a.shape != shape for a in views.values()):
        raise ValueError("expected gate/up (M,H) and down (H,M), with M,H > 0")
    return views


def _workspace(m, h, sketch_width, k, probe_width):
    # Even a complete shortlist on a toy layer need not store an M-by-M table.
    page = min(k, 64, max(1, m // 2))
    pair_rows = min(64, max(1, m // 2), _TILE_LIMIT // (8 * m))
    weight_width = max(h, 3 * probe_width)
    weight_rows = min(128, m, _TILE_LIMIT // (8 * weight_width))
    feature_bytes = 2 * m * 3 * sketch_width * 8
    cache_bytes = m * page * (8 + np.dtype(np.intp).itemsize)
    probe_bytes = 2 * m * 3 * probe_width * 8
    if min(pair_rows, weight_rows) < 1:
        raise ValueError("even one row would exceed the tile byte limit")
    if max(feature_bytes + cache_bytes, probe_bytes) > _STATE_LIMIT:
        raise ValueError("feature or shortlist buffers would exceed the state byte limit")
    return {
        "tile_limit_bytes": _TILE_LIMIT, "state_limit_bytes": _STATE_LIMIT,
        "pair_tile_rows": pair_rows, "pair_tile_bytes": pair_rows * m * 8,
        "weight_tile_rows": weight_rows, "weight_tile_bytes": weight_rows * h * 8,
        "candidate_page": page, "candidate_cache_bytes": cache_bytes,
        "sketch_pair_bytes": feature_bytes, "null_probe_pair_bytes": probe_bytes,
    }


def _slices(n, rows):
    for start in range(0, n, rows):
        yield slice(start, min(n, start + rows))


def _block(a, rows, cols):
    # Cartesian gather, not a copy of a[rows, :] followed by column selection.
    if not isinstance(rows, slice) and not isinstance(cols, slice):
        rows = rows[:, None]
    block = np.asarray(a[rows, cols], dtype=np.float64, order="C")
    if not np.isfinite(block).all():
        raise ValueError("nonfinite weight")
    return block


def _features(layer, coords, rows, *, unit):
    m = layer["gate"].shape[0]
    width = len(coords)
    out = np.empty((m, 3 * width), dtype=np.float64)
    for proj, name in enumerate(PROJECTIONS):
        for s in _slices(m, rows):
            b = _block(layer[name], s, coords)
            if unit:
                norm = np.sqrt(np.einsum("ij,ij->i", b, b))
                norm[norm == 0] = 1
                b /= norm[:, None]
            out[s, proj * width:(proj + 1) * width] = b
    return out


def _ranked(row, k):
    if k == row.size:
        return np.argsort(-row, kind="stable")
    threshold = np.partition(row, row.size - k)[row.size - k]
    above = np.flatnonzero(row > threshold)
    ties = np.flatnonzero(row == threshold)[:k - len(above)]
    ids = np.concatenate((above, ties))  # Already ascending within every tie.
    return ids[np.argsort(-row[ids], kind="stable")]


def _match(left, right, k, workspace):
    """Merge ranked row shortlists by (-similarity, source, destination)."""
    m = left.shape[0]
    rows = workspace["pair_tile_rows"]
    page = workspace["candidate_page"]
    scratch = np.empty((rows, m), dtype=np.float64)
    ids = np.empty((m, page), dtype=np.intp)
    scores = np.empty((m, page), dtype=np.float64)
    lengths = np.empty(m, dtype=np.intp)
    more = np.empty(m, dtype=bool)
    used = np.zeros(m, dtype=bool)
    perm = np.full(m, -1, dtype=np.intp)

    def tile(start):
        src = left[start:start + rows]
        out = scratch[:len(src)]
        np.matmul(src, right.T, out=out)
        out /= 3  # Signed mean; a zero projection still has weight 1/3.
        return out

    def fill(i, row):
        ranked = _ranked(row, k)
        live = ranked[~used[ranked]]
        n = min(len(live), page)
        lengths[i], more[i] = n, len(live) > n
        ids[i, :n] = live[:n]
        scores[i, :n] = row[live[:n]]

    def head(i, pos):
        return -float(scores[i, pos]), i, int(ids[i, pos]), pos

    queue = []
    for start in range(0, m, rows):
        for i, row in enumerate(tile(start), start):
            fill(i, row)
            queue.append(head(i, 0))
    heapq.heapify(queue)
    while queue:
        _, i, j, pos = heapq.heappop(queue)
        if not used[j]:
            perm[i], used[j] = j, True
            continue
        pos += 1
        while pos < lengths[i] and used[ids[i, pos]]:
            pos += 1
        if pos == lengths[i] and more[i]:
            # Only unusually wide shortlists need another page. Reproduce the
            # original tile and cutoff, then discard destinations no longer live.
            start = i // rows * rows
            fill(i, tile(start)[i - start])
            pos = 0
        if pos < lengths[i]:
            heapq.heappush(queue, head(i, pos))

    fallback = int(np.count_nonzero(perm < 0))
    for s in _slices(m, rows):
        missing = np.flatnonzero(perm[s] < 0)
        if len(missing):
            block = tile(s.start)
            for r in missing:
                row = block[r]
                row[used] = -np.inf
                j = int(np.argmax(row))  # First live maximum = smallest destination.
                perm[s.start + r], used[j] = j, True
    if not np.array_equal(np.sort(perm), np.arange(m)):
        raise RuntimeError("channel assignment did not produce a bijection")
    return perm, fallback


def _inner(a, b):
    return float(np.einsum("ij,ij->", a, b))


def _sse(a, b):
    diff = a - b  # Not ||a||² + ||b||² - 2<a,b>, which loses small residuals.
    return _inner(diff, diff)


def _projection_stats(a, b, coords, perm, rows):
    def partials():
        for s in _slices(len(perm), rows):
            x = _block(a, s, coords)
            y = _block(b, s, coords)
            z = _block(b, perm[s], coords)
            yield (_inner(x, x), _inner(y, y), _inner(x, y), _inner(x, z),
                   _sse(x, y), _sse(x, z))

    aa, bb, ab, ap, identity, aligned = map(math.fsum, zip(*partials()))
    denom = math.sqrt(aa) * math.sqrt(bb)
    return {
        "count": len(perm) * len(coords), "earlier_ss": aa, "later_ss": bb,
        "identity_dot": ab, "aligned_dot": ap,
        "identity_cosine": ab / denom if denom else None,
        "aligned_cosine": ap / denom if denom else None,
        "identity_sse": identity, "aligned_sse": aligned,
    }


def _score_split(a, b, coords, perm, rows):
    coords = np.sort(coords)
    stats = {p: _projection_stats(a[p], b[p], coords, perm, rows) for p in PROJECTIONS}
    identity = math.fsum(t["identity_sse"] for t in stats.values())
    aligned = math.fsum(t["aligned_sse"] for t in stats.values())
    return {
        "projections": stats,
        "pooled": {"count": 3 * len(perm) * len(coords), "identity_sse": identity,
                   "aligned_sse": aligned,
                   "fraction_sse_removed": 1 - aligned / identity if identity else None},
    }


def _null_probe(a, b, coords, perm, rows, seed, reps):
    x = _features(a, coords, rows, unit=False)
    y = _features(b, coords, rows, unit=False)
    m = len(perm)

    def distance(order):
        return math.fsum(_sse(x[s], y[order[s]]) for s in _slices(m, rows))

    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(1,))))
    return {
        "coordinates": coords.tolist(), "count": 3 * m * len(coords),
        "identity_sse": distance(np.arange(m)), "aligned_sse": distance(perm),
        "shuffled_sse": [distance(rng.permutation(m)) for _ in range(reps)],
    }


def _activations(h):
    rng = np.random.Generator(np.random.PCG64(0))
    signs = (2 * rng.integers(2, size=(h, 2)) - 1) / math.sqrt(h)
    return np.column_stack((signs, signs[:, 0] * 2.0**-24, np.zeros(h)))


def _action(layer, order, x, rows):
    """Full FP64 later-layer output in the given channel order. No model runtime."""
    total = np.zeros_like(x)
    correction = np.zeros_like(x)
    for s in _slices(len(order), rows):
        idx = order[s]
        g = _block(layer["gate"], idx, slice(None)) @ x
        u = _block(layer["up"], idx, slice(None)) @ x
        e = np.exp(-np.abs(g))
        silu = g * (np.where(g >= 0, 1.0, e) / (1.0 + e))
        d = _block(layer["down"], idx, slice(None))
        increment = d.T @ (silu * u) - correction
        updated = total + increment
        correction = (updated - total) - increment
        total = updated
    return total


def _l2(x):
    peak = float(np.max(np.abs(x)))
    return peak * float(np.linalg.norm(x / peak)) if peak else 0.0


def _action_check(layer, perm, rows):
    x = _activations(layer["gate"].shape[1])
    original = _action(layer, np.arange(len(perm)), x, rows)
    reordered = _action(layer, perm, x, rows)
    atol, rtol = 1e-12, 1e-10
    vectors = {}
    for i, name in enumerate(("unit_0", "unit_1", "near_zero", "zero")):
        ref = _l2(original[:, i])
        absolute = _l2(reordered[:, i] - original[:, i])
        vectors[name] = {
            "reference_l2": ref, "absolute_l2": absolute,
            "relative_l2": absolute / ref if ref else None,
            "passed": absolute <= atol + rtol * ref,
        }
    return {"atol": atol, "rtol": rtol, "vectors": vectors,
            "passed": all(v["passed"] for v in vectors.values())}


def analyze_pair(earlier, later, *, seed, sketch=64, shortlist=16, null_reps=32):
    """Return P (earlier j -> later P[j]) and JSON-compatible raw measurements.

    Gate/up have shape (M,H), down (H,M). Inputs remain untouched. The mapping
    depends only on the fit sketch; all scores and the action check are post-fit.
    """
    a, b = _channel_views(earlier), _channel_views(later)
    m, h = a["gate"].shape
    if b["gate"].shape != (m, h):
        raise ValueError("earlier and later layer shapes differ")
    seed = _integer(seed, "seed", 0)
    width = min(_integer(sketch, "sketch", 1), h // 2)
    k = min(_integer(shortlist, "shortlist", 1), m)
    reps = _integer(null_reps, "null_reps", 0)
    q = min(_PROBE_WIDTH, h // 2)
    workspace = _workspace(m, h, width, k, q)
    rows = workspace["weight_tile_rows"]
    fit, audit = coordinate_split(h, seed=seed)
    sketch_coords = np.sort(fit[:width])
    perm, fallback = _match(_features(a, sketch_coords, rows, unit=True),
                            _features(b, sketch_coords, rows, unit=True), k, workspace)
    return perm, {
        "schema_version": 1, "channels": m, "hidden_size": h,
        "permutation_sha256": hashlib.sha256(perm.astype("<i8").tobytes(order="C")).hexdigest(),
        "permutation_encoding": "P in source order, little-endian signed int64, no header",
        "coordinates": {"seed": seed, "generator": "PCG64", "fit": fit.tolist(),
                        "audit": audit.tolist(), "sketch": sketch_coords.tolist()},
        "matching": {"shortlist": k, "shortlist_matches": m - fallback, "fallback": fallback},
        "workspace": workspace,
        "fit": _score_split(a, b, fit, perm, rows),
        "audit": _score_split(a, b, audit, perm, rows),
        "null_probe": _null_probe(a, b, np.sort(audit[:q]), perm, rows, seed, reps),
        "action_check": _action_check(b, perm, rows),
    }
