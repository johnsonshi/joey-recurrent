"""Offline, paired gold-rank statistics for a fixed ordered decision menu."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import chain, zip_longest
import math
import random

from . import evaluation as ev


_CUTOFFS = (1, 3, 5)
_METRICS = tuple(f"top{k}" for k in _CUTOFFS) + ("mrr",)
_BUCKETS = ("1", "2", "3-5", "6+")


def _bound_value(record: ev.Row, key: str, digest: str, column: str, stream: str):
    if not isinstance(record, dict) or record.keys() != {"id", "input_sha256", column}:
        raise ev.EvaluationError(f"{stream}: unexpected row schema")
    if record["id"] != key:
        raise ev.EvaluationError(f"{stream}: ID order differs from inputs")
    if record["input_sha256"] != digest:
        raise ev.EvaluationError(f"{stream}: stale input digest")
    return record[column]


def _hit_count(ranks: Counter[int], k: int) -> int:
    return sum(ranks[r] for r in range(1, k + 1))


def _rr_sum(ranks: Counter[int]) -> float:
    return math.fsum(count / rank for rank, count in ranks.items())


@dataclass(slots=True)
class _Arm:
    stable: Counter[int] = field(default_factory=Counter)
    best: Counter[int] = field(default_factory=Counter)
    worst: Counter[int] = field(default_factory=Counter)
    gold_ties: int = 0

    def observe(self, probs: object, gold: int, width: int) -> int:
        if not isinstance(probs, list) or len(probs) != width:
            raise ev.EvaluationError("expected one probability for every ordered choice")
        if any(type(p) not in (int, float) or not 0 <= p <= 1 for p in probs):
            raise ev.EvaluationError("probabilities must be finite numbers in [0, 1]")
        if abs(math.fsum(probs) - 1) > ev.PROB_SUM_ATOL:
            raise ev.EvaluationError("probabilities do not sum to one within tolerance")

        score = probs[gold]
        best = 1 + sum(p > score for p in probs)
        stable = best + probs[:gold].count(score)
        worst = best + probs.count(score) - 1
        self.stable[stable] += 1
        self.best[best] += 1
        self.worst[worst] += 1
        self.gold_ties += best != worst
        return stable

    def summary(self, n: int) -> dict:
        def hits(ranks):
            return {str(k): _hit_count(ranks, k) for k in _CUTOFFS}

        return {"hits": hits(self.stable), "mrr": _rr_sum(self.stable) / n,
                "gold_ties": self.gold_ties, "possible_hits": hits(self.best),
                "guaranteed_hits": hits(self.worst), "mrr_best": _rr_sum(self.best) / n,
                "mrr_worst": _rr_sum(self.worst) / n}


@dataclass(slots=True)
class _Group:
    task: str
    rows: int = 0
    # Signed rank histogram: candidate counts minus baseline counts.
    balance: Counter[int] = field(default_factory=Counter)

    def observe(self, before: int, after: int) -> None:
        self.rows += 1
        self.balance[before] -= 1
        self.balance[after] += 1

    def totals(self):
        return (self.rows, *(_hit_count(self.balance, k) for k in _CUTOFFS), _rr_sum(self.balance))


def _paired(cells, reps: int, seed: int) -> dict:
    def ratios(sample):
        sizes, *columns = zip(*sample)
        n = sum(sizes)
        return [math.fsum(column) / n for column in columns]

    point = ratios(chain.from_iterable(cells))
    draws = [[] for _ in point]
    if reps and all(len(cell) >= 2 for cell in cells):
        rng = random.Random(seed)
        for _ in range(reps):
            sample = (rng.choice(cell) for cell in cells for _ in cell)
            for values, value in zip(draws, ratios(sample), strict=True):
                values.append(value)
    paired = {}
    for metric, delta, values in zip(_METRICS, point, draws, strict=True):
        values.sort()
        ci95 = [ev._percentile(values, q) for q in (.025, .975)] if values else None
        paired[metric] = {"delta": delta, "ci95": ci95}
    return paired


def _bucket(rank: int) -> str:
    if rank <= 2:
        return str(rank)
    return "3-5" if rank <= 5 else "6+"


def rank_contrast(inputs: Iterable[ev.Row], labels: Iterable[ev.Row], baseline: Iterable[ev.Row],
                  candidate: Iterable[ev.Row], *, strata: Mapping[str, str], reps: int = 2000,
                  seed: int = 4117) -> dict:
    """Compare two pinned ``probs`` streams in input order, against pinned ``gold_index`` labels.

    Hits and MRR use descending raw probabilities, breaking exact ties by option index.
    Possible/best and guaranteed/worst place gold first and last within its tie, respectively.
    ``gold_ties`` counts rows, not tied option pairs. ``rank_buckets[before][after]`` counts
    stable gold ranks; ``change_counts`` compares ranks even within the same bucket.

    Paired deltas are candidate minus baseline, per row. Each replicate pools the rows of
    whole groups drawn with replacement within each (task, stratum), using one common draw
    for all endpoints. ``strata`` in the result counts occupied (task, stratum) cells.
    Pointwise percentile CIs are None for zero draws or any cell with fewer than two groups.
    """
    if type(reps) is not int or reps < 0 or type(seed) is not int:
        raise ev.EvaluationError("expected nonnegative integer reps and an integer seed")
    if not isinstance(strata, Mapping) or not all(map(ev._text, strata.values())):
        raise ev.EvaluationError("strata must map group IDs to nonempty text")

    seen = set()
    groups: dict[str, _Group] = {}
    menu = None
    base, cand = _Arm(), _Arm()
    buckets = {b: dict.fromkeys(_BUCKETS, 0) for b in _BUCKETS}
    changes = dict.fromkeys(("improved", "unchanged", "worsened"), 0)
    for row, label, b, c in zip_longest(inputs, labels, baseline, candidate):
        if row is None or label is None or b is None or c is None:
            raise ev.EvaluationError("all four streams must contain one record per input")
        digest = ev.input_sha256(row)
        key = row["id"]
        if key in seen:
            raise ev.EvaluationError("duplicate input ID")
        seen.add(key)
        if menu is None:
            menu = row["choices"].copy()
        if len(menu) < 5 or row["choices"] != menu:
            raise ev.EvaluationError("expected the same ordered menu of at least five choices")
        gold = _bound_value(label, key, digest, "gold_index", "labels")
        if type(gold) is not int or not 0 <= gold < len(menu):
            raise ev.EvaluationError("gold_index outside the fixed menu")
        before = base.observe(_bound_value(b, key, digest, "probs", "baseline"), gold, len(menu))
        after = cand.observe(_bound_value(c, key, digest, "probs", "candidate"), gold, len(menu))

        group_id = row["group_id"]
        if group_id not in groups:
            groups[group_id] = _Group(row["task"])
        group = groups[group_id]
        if group.task != row["task"]:
            raise ev.EvaluationError("a group spans tasks")
        group.observe(before, after)
        buckets[_bucket(before)][_bucket(after)] += 1
        change = "improved" if after < before else "worsened" if after > before else "unchanged"
        changes[change] += 1

    if not seen:
        raise ev.EvaluationError("empty input set")
    if strata.keys() != groups.keys():
        raise ev.EvaluationError("strata must cover exactly the observed group IDs")
    cells = defaultdict(list)
    for group_id, group in sorted(groups.items()):
        cells[group.task, strata[group_id]].append(group.totals())
    paired = _paired([cells[key] for key in sorted(cells)], reps, seed)
    return {"baseline": base.summary(len(seen)), "candidate": cand.summary(len(seen)),
            "paired": paired, "rank_buckets": buckets, "change_counts": changes,
            "rows": len(seen), "groups": len(groups), "strata": len(cells)}
