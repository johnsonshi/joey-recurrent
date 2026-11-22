"""Label-free, batch-one timing analysis for the two-slot A100 crossover.

The host verifies device identity, transport and completion receipts. This module
only checks the supplied worker records and counts each original input row once.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random
import re
import sys

from .evaluation import PROB_SUM_ATOL, input_sha256
from .loop import LoopCounts
from .worker import SELECT20_LAYERS, method_sha256


class _Snapshot:
    """Give a frozen field dict/list value semantics, without coercing its contents."""

    def __set_name__(self, owner, name):
        self.storage = "_" + name

    def __set__(self, frame, value):
        object.__setattr__(frame, self.storage, deepcopy(value))

    def __get__(self, frame, owner=None):
        if frame is None:
            raise AttributeError(self.storage)  # required dataclass constructor argument
        return deepcopy(getattr(frame, self.storage))


@dataclass(frozen=True)
class TimingFrame:
    """One physical execution. Mutable worker fields are copied on input and access."""

    __slots__ = ("slot", "period", "_settings", "packet_sha256", "_predictions", "_trace")
    slot: int
    period: int
    settings: dict = _Snapshot()
    packet_sha256: str
    predictions: list = _Snapshot()
    trace: list = _Snapshot()


# The stock callback includes its own window and head; zero_extra uses the split
# path once. Layer totals alone cannot distinguish these two executions.
_PLANS = {
    "stock": (1, "naive", LoopCounts(1, 0, 0, 0)),
    "split-control": (2, "zero_extra", LoopCounts(0, 1, 1, 1)),
    "naive-2": (2, "naive", LoopCounts(0, 1, 2, 1)),
}
_PAIRS = {
    "equal-work": ("full:split-control", "select-20:naive-2"),
    "same-k-stock": ("full:stock", "select-20:stock"),
    "same-k-naive-2": ("full:naive-2", "select-20:naive-2"),
}
_PRED = {"id", "input_sha256", "probs"}
_TRACE = {"id", "packet_sha256", "tokens", "layer_calls", "callback_counts", "readouts",
          "cuda_event_ms", "wall_ms"}
_PIN = re.compile(r"[0-9a-f]{64}")


def _method(settings: dict, payload: str) -> str:
    if type(settings) is not dict:
        raise ValueError("expected declared worker settings")
    extras = {} if payload == "full" else {
        "compact_recipe": "select-20", "compact_manifest_sha256": settings.get("compact_manifest_sha256"),
    }
    for mode, (k, method, _) in _PLANS.items():
        canonical = {"window": [8, 12], "reverse": False, "loop": {"k": k, "method": method}, **extras}
        # Restrict the schema before dispatching to the worker. Its digest also
        # distinguishes JSON types that Python equality aliases (e.g. 0/false).
        if settings == canonical and method_sha256(settings) == method_sha256(canonical):
            return f"{payload}:{mode}"
    raise ValueError(f"not a canonical {payload} method for this slot and period")


def _roster(frames: Sequence[TimingFrame]) -> dict:
    if not isinstance(frames, Sequence) or len(frames) != 12:
        raise ValueError("expected a sequence of 12 physical frames")
    runs, packets, compact_pins = {}, set(), set()
    for frame in frames:
        if (not isinstance(frame, TimingFrame) or type(frame.slot) is not int or frame.slot not in (0, 1)
                or type(frame.period) is not int or frame.period not in (1, 2)):
            raise ValueError("expected slot 0/1 and period 1/2")
        packet = frame.packet_sha256
        if type(packet) is not str or not _PIN.fullmatch(packet) or packet in packets:
            raise ValueError("expected distinct physical packet SHA-256 pins")
        packets.add(packet)
        payload = "full" if frame.slot == frame.period - 1 else "select-20"
        settings = frame.settings
        key = (frame.slot, _method(settings, payload))
        if key in runs:
            raise ValueError("duplicate method on a physical slot")
        runs[key] = frame
        if payload == "select-20":
            compact_pins.add(settings["compact_manifest_sha256"])
    # Twelve distinct legal cells exhaust the two slots x two payloads x three plans.
    if len(compact_pins) != 1:
        raise ValueError("all compact frames must use the same manifest pin")
    return runs


def _inputs(rows: Iterable[dict], strata: Mapping):
    pairs, ids, owners = [], set(), {}
    members = defaultdict(list)
    for i, row in enumerate(rows):
        digest = input_sha256(row)
        key, group = row["id"], row["group_id"]
        if len(row["choices"]) != 4 or key in ids:
            raise ValueError("expected four choices and distinct input IDs")
        if owners.setdefault(group, row["task"]) != row["task"]:
            raise ValueError("an original graph spans tasks")
        ids.add(key)
        pairs.append((key, digest))
        members[group].append(i)
    if not pairs or not isinstance(strata, Mapping) or set(strata) != set(members):
        raise ValueError("strata must map exactly the nonempty set of original graph IDs")
    graphs = defaultdict(list)
    for group in sorted(members):
        s = strata[group]
        if type(s) is not int and not (type(s) is str and s.strip()):
            raise ValueError("generator seed strata must be integers or nonempty strings")
        graphs[s].append(members[group])
    # Keep integer seeds distinct from their string spellings.
    return pairs, [graphs[s] for s in sorted(graphs, key=lambda s: (isinstance(s, str), s))]


def _work(name: str) -> dict:
    payload, mode = name.split(":")
    depth = 24 if payload == "full" else len(SELECT20_LAYERS)
    calls = _PLANS[mode][2]
    layers = calls.layer_evaluations(8, 4, depth - 12)
    # The select-20 recipe retains whole cycles. Every stage, including [8,12),
    # is made up of four-layer cycles with one full and three GatedDeltaNet layers.
    full = layers // 4
    return {"callback_counts": asdict(calls), "layer_calls": layers, "full_attention_calls": full,
            "gated_deltanet_calls": layers - full, "head_calls": calls.coda_readouts}


def _ms(value) -> float:
    if type(value) not in (int, float) or not 0 < value <= sys.float_info.max:
        raise ValueError("timing must be a finite positive number of milliseconds")
    return float(value)


def _sum(values) -> float:
    try:
        return math.fsum(values)
    except OverflowError as exc:
        raise ValueError("timing sum outside the representable range") from exc


@dataclass
class _Run:
    period: int
    cuda: list[float]
    mean_wall: float
    vectors: list[list]


def _read_run(frame, pairs, tokens, work) -> _Run:
    preds, trace = frame.predictions, frame.trace
    if (not isinstance(preds, list) or not isinstance(trace, list)
            or len(preds) != len(pairs) or len(trace) != len(pairs)):
        raise ValueError("every physical stream must cover every row once")
    cuda, wall, vectors = [], [], []
    for i, ((key, digest), pred, t) in enumerate(zip(pairs, preds, trace, strict=True)):
        if (not isinstance(pred, dict) or set(pred) != _PRED or pred["id"] != key
                or pred["input_sha256"] != digest):
            raise ValueError("prediction schema, row order or input digest mismatch")
        p = pred["probs"]
        if (not isinstance(p, list) or len(p) != 4
                or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in p)
                or abs(math.fsum(p) - 1) > PROB_SUM_ATOL):
            raise ValueError("expected one legal ordered four-way probability vector")
        if (not isinstance(t, dict) or set(t) != _TRACE or t["id"] != key
                or t["packet_sha256"] != frame.packet_sha256 or t["readouts"] != []):
            raise ValueError("trace schema, row order, packet pin or fixed final readout mismatch")
        count, calls = t["tokens"], t["callback_counts"]
        if type(count) is not int or count <= 0 or (tokens[i] is not None and count != tokens[i]):
            raise ValueError("observed token counts must be positive integers paired across all frames")
        tokens[i] = count
        if (type(t["layer_calls"]) is not int or t["layer_calls"] != work["layer_calls"]
                or not isinstance(calls, dict) or calls != work["callback_counts"]
                or any(type(v) is not int for v in calls.values())):
            raise ValueError("actual callback or physical layer counts disagree with the fixed method")
        cuda.append(_ms(t["cuda_event_ms"]))
        wall.append(_ms(t["wall_ms"]))
        vectors.append(p)
    return _Run(frame.period, cuda, _sum(wall) / len(wall), vectors)


def _reproducibility(left, right) -> dict:
    identical = agree = above = 0
    largest = 0.0
    for a, b in zip(left, right, strict=True):
        identical += a == b
        agree += a.index(max(a)) == b.index(max(b))
        gap = max(abs(x - y) for x, y in zip(a, b, strict=True))
        above += gap > 1e-5
        largest = max(largest, gap)
    return {"total_rows": len(left), "identical_vector_rows": identical, "argmax_agreement_rows": agree,
            "argmax_agreement": agree / len(left), "max_abs_probability_difference": largest,
            "rows_exceeding_1e-5": above}


def _resampled_totals(runs, graphs, reps, seed):
    """Draw graphs once per replicate for every slot/method; keep within-graph row sums."""
    blocks = [[{key: _sum(run.cuda[i] for i in group) for key, run in runs.items()}
               for group in stratum] for stratum in graphs]
    rng = random.Random(seed)
    for _ in range(reps):
        sample = [rng.choice(stratum) for stratum in blocks for _ in stratum]
        yield {key: _sum(group[key] for group in sample) for key in runs}


def _ratios(totals, full, compact):
    # The paired row count (including repeated draws of a graph) cancels.
    a, b = (totals[s, compact] / totals[s, full] for s in (0, 1))
    if not all(r > 0 and math.isfinite(r) for r in (a, b)):
        raise ValueError("CUDA ratio outside the representable range")
    return a, b, math.exp((math.log(a) + math.log(b)) / 2)


def _ci95(values):
    ordered = sorted(values)
    if not ordered:
        return None
    bounds = []
    for q in (0.025, 0.975):
        x = (len(ordered) - 1) * q
        lo, hi = math.floor(x), math.ceil(x)
        bounds.append(ordered[lo] + (x - lo) * (ordered[hi] - ordered[lo]))
    return bounds


def crossed_timing(rows, frames: Sequence[TimingFrame], *, strata, reps=2000, seed=4117) -> dict:
    """Summarize 12 host-verified worker streams; never read files, labels or devices.

    In period 1, slot 0 is full and slot 1 is select-20; period 2 swaps them.
    Frame order is immaterial, but predictions and final-only traces must follow
    the input rows exactly. ``strata`` maps every group_id to its original seed.
    ``methods[name].by_slot`` gives period-specific row means. Each comparison's
    ``by_slot`` ratios are compact/full mean CUDA ms; ``geometric_cuda_ratio`` is
    their geometric mean. Percentile CIs use the same whole-graph draws for all
    contrasts; they are null for zero draws or a seed with fewer than two graphs.
    Raw ordered vectors are compared, never renormalized or scored.
    """
    if type(reps) is not int or reps < 0 or type(seed) is not int:
        raise ValueError("expected a nonnegative integer replicate count and integer bootstrap seed")
    roster = _roster(frames)
    pairs, graphs = _inputs(rows, strata)
    tokens = [None] * len(pairs)
    runs, totals, methods, repro = {}, {}, {}, {}
    for payload in ("full", "select-20"):
        for mode in _PLANS:
            name = f"{payload}:{mode}"
            work, by_slot = _work(name), {}
            for slot in (0, 1):
                key = (slot, name)
                run = runs[key] = _read_run(roster[key], pairs, tokens, work)
                totals[key] = _sum(run.cuda)
                by_slot[str(slot)] = {"period": run.period, "mean_cuda_ms": totals[key] / len(pairs),
                                      "mean_wall_ms": run.mean_wall}
            methods[name] = {"work_per_row": work, "by_slot": by_slot}
            repro[name] = _reproducibility(runs[0, name].vectors, runs[1, name].vectors)

    min_groups = min(map(len, graphs))
    status = "not_requested" if reps == 0 else "insufficient_independent_groups" if min_groups < 2 else "ok"
    draws = list(_resampled_totals(runs, graphs, reps, seed)) if status == "ok" else []
    comparisons = {}
    for label, (full, compact) in _PAIRS.items():
        point = _ratios(totals, full, compact)
        scores = [_ratios(draw, full, compact) for draw in draws]
        ci = [_ci95(score[i] for score in scores) for i in range(3)]
        comparisons[label] = {
            "full": full, "compact": compact, "matched_logical_work": label == "equal-work",
            "by_slot": {str(s): {"cuda_ratio": point[s], "ci95": ci[s]} for s in (0, 1)},
            "geometric_cuda_ratio": point[2], "geometric_cuda_ratio_ci95": ci[2],
        }
    return {
        "rows": len(pairs), "groups": sum(map(len, graphs)),
        "methods": methods, "comparisons": comparisons, "reproducibility": repro,
        "bootstrap": {"status": status, "seed": seed, "requested_replicates": reps,
                      "used_replicates": len(draws), "seed_strata": len(graphs),
                      "min_groups_per_stratum": min_groups,
                      "scope": "Whole original graphs paired within generator seed; input-mix uncertainty only."},
        "timing_scope": "Batch-one CUDA-event time is primary; row wall time is illustrative. "
                        "Same-k contrasts do not match logical work. Repeated streams are not independent "
                        "decisions. No between-device significance or batched-serving claim.",
    }
