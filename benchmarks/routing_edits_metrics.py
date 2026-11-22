"""Score a released same-target edit panel. No file I/O, inference, admission or label release."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from itertools import zip_longest
import math
import random
from typing import NamedTuple

from benchmarks.routing_edits import adjudicate, check_blind_panel
from joey_research import evaluation as ev


_ROLES = ("base", "target_swap", "other_swap")
_METRICS = ("all_three", "switch_on_target", "preserve_on_other",
            *(k + "_accuracy" for k in _ROLES), *(k + "_brier" for k in _ROLES))


class _Cell(NamedTuple):
    entry: str
    baseline: ev._Observation
    candidate: ev._Observation
    tokens: int


def _triples(rows, labels, baseline, candidate, design, baseline_tokens, candidate_tokens):
    rows = list(rows)
    panel = check_blind_panel(rows, design)
    tokens = (baseline_tokens, candidate_tokens)
    if any(not isinstance(t, Mapping) or t.keys() != design.keys() for t in tokens):
        raise ev.EvaluationError("expected exact recorded token coverage in both arms")
    cells = defaultdict(dict)
    semantic = {}
    for r, y, a, b in zip_longest(rows, labels, baseline, candidate):
        if any(x is None for x in (r, y, a, b)):
            raise ev.EvaluationError("different stream lengths")
        key = r["id"]
        expected, info = adjudicate(r)
        if not isinstance(y, Mapping) or type(y.get("gold_index")) is not int or y != expected:
            raise ev.EvaluationError("released label does not match the exact displayed question")
        d = design[key]
        if (d["hop"], d["rendering"]) != (info["hop"], info["rendering"]):
            raise ev.EvaluationError("declared distance or rendering differs from displayed graph")
        group, case, role = r["group_id"], d["case_id"], d["condition"]
        if semantic.setdefault((group, case, role), info["entry"]) != info["entry"]:
            raise ev.EvaluationError("gold entry changed across equivalent renderings")
        observations = []
        for p in (a, b):
            if (not isinstance(p, Mapping) or set(p) != {"id", "input_sha256", "probs"}
                    or p["id"] != key or p["input_sha256"] != y["input_sha256"]):
                raise ev.EvaluationError("prediction schema, order or input digest differs")
            o = ev._observe(r, y["gold_index"], p)
            if o.error:
                raise ev.EvaluationError("invalid or incomplete probability vector")
            observations.append(o)
        n, m = (t[key] for t in tokens)
        if type(n) is not int or type(m) is not int or n < 1 or n != m:
            raise ev.EvaluationError("arms require equal positive measured token counts on this row")
        cells[d["seed"], group, d["hop"], case, d["rendering"]][role] = _Cell(info["entry"], *observations, n)
    grouped = defaultdict(list)
    for (seed, group, hop, _, _), c in sorted(cells.items()):
        base, change, other = triple = tuple(c[k] for k in _ROLES)
        if base.entry == change.entry or base.entry != other.entry:
            raise ev.EvaluationError("the intended changed and preserved entries failed semantic adjudication")
        grouped[seed, group, hop].append(triple)
    return panel, grouped


def _values(obs):
    a, b, c = obs
    return (all(o.correct for o in obs), a.choice != b.choice, a.choice == c.choice,
            *(o.correct for o in obs), *(o.brier for o in obs))


def _panel(grouped, hop, reps, seed):
    totals = [Counter(), Counter()]
    strata = defaultdict(list)
    transitions = Counter()
    lengths = defaultdict(list)
    groups = defaultdict(list)
    n = equal = 0
    for (s, g, h), triples in grouped.items():
        if hop is None or h == hop:
            groups[s, g].extend(triples)
    for (s, _), triples in sorted(groups.items()):
        diffs = []
        for t in triples:
            a = _values(tuple(x.baseline for x in t))
            b = _values(tuple(x.candidate for x in t))
            for counts, vals in zip(totals, (a, b)):
                counts.update(dict(zip(_METRICS, vals, strict=True)))
            diffs.append([v - u for u, v in zip(a, b, strict=True)])
            transitions[a[0], b[0]] += 1
            for k, x in zip(_ROLES, t, strict=True):
                lengths[k].append(x.tokens)
            equal += len({x.tokens for x in t}) == 1
        strata[s].append((len(triples), *(math.fsum(col) for col in zip(*diffs))))
        n += len(triples)

    def mean(sample):
        den, *sums = (math.fsum(col) for col in zip(*sample))
        return [v / den for v in sums]

    blocks = [strata[s] for s in sorted(strata)]
    point = mean([g for block in blocks for g in block])
    draws = [[] for _ in point]
    if all(len(block) >= 2 for block in blocks):
        rng = random.Random(seed)
        for _ in range(reps):
            values = mean(rng.choice(block) for block in blocks for _ in block)
            for target, value in zip(draws, values, strict=True):
                target.append(value)
    paired = {}
    for k, value, ds in zip(_METRICS, point, draws, strict=True):
        ds.sort()
        paired[k] = {"delta": value, "ci95": [ev._percentile(ds, q) for q in (.025, .975)] if ds else None}

    def summary(counts):
        return {"all_three_correct": counts["all_three"], "triple_accuracy": counts["all_three"] / n,
                "switch_on_target": counts["switch_on_target"], "preserve_on_other": counts["preserve_on_other"],
                "conditions": {k: {"correct": counts[k + "_accuracy"], "accuracy": counts[k + "_accuracy"] / n,
                                   "brier": counts[k + "_brier"] / n} for k in _ROLES}}

    return {"triples": n, "graphs": len(groups), "baseline": summary(totals[0]), "candidate": summary(totals[1]),
            "paired": paired, "triple_corrections": transitions[False, True], "triple_corruptions": transitions[True, False],
            "all_three_equal_tokens": equal, "tokens": {k: {"min": min(v), "max": max(v), "mean": sum(v) / n}
                                                         for k, v in lengths.items()}}


def compare_edits(rows, labels, baseline, candidate, *, design_by_id, baseline_tokens, candidate_tokens,
                  reps=2000, seed=4117):
    """Post-release only. Compare two ID-bound streams on every matched triple, not only correct base cases.

    The primary endpoint is all three decisions correct. Switch and preservation alone describe
    behavior, not correctness. All paired intervals share seed-stratified whole-graph draws.
    The caller must verify the cohort's physical receipts and label release before passing real rows.
    """
    if type(reps) is not int or not 0 <= reps <= 10000 or type(seed) is not int:
        raise ev.EvaluationError("invalid bootstrap settings")
    support, grouped = _triples(rows, labels, baseline, candidate, design_by_id, baseline_tokens, candidate_tokens)
    return {**support, "bootstrap": {"reps": reps, "seed": seed},
            "all": _panel(grouped, None, reps, seed),
            "by_hop": {str(h): _panel(grouped, h, reps, seed) for h in (2, 6)}}
