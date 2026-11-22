"""Selective risk and calibration reports on complete, ID-bound legal-slot predictions."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import math
import random

from . import evaluation as ev


def observations(inputs: Iterable[ev.Row], labels: Iterable[ev.Row], predictions: Iterable[ev.Row]):
    ix = ev._index_inputs(inputs)
    gold = ev._bind(labels, ix, ({"id", "input_sha256", "gold_index"},), "labels")
    pred = ev._bind(predictions, ix, ({"id", "input_sha256", "probs"},), "predictions")
    result = []
    for k in sorted(ix):
        row = ix[k][0]
        y = gold[k]["gold_index"]
        if type(y) is not int or not 0 <= y < len(row["choices"]):
            raise ev.EvaluationError("gold index not in this menu")
        o = ev._observe(row, y, pred[k])
        if o.error is not None:
            raise ev.EvaluationError("selective reporting requires complete valid probability distributions")
        result.append(o)
    return result


def _probability(x):
    return type(x) in (float, int) and 0 <= x <= 1


def _point(n, accepted, wrong, groups):
    return {"n": n, "accepted": accepted, "wrong": wrong, "accepted_groups": groups,
            "coverage": accepted / n, "risk": wrong / accepted if accepted else None,
            "accuracy": (accepted - wrong) / accepted if accepted else None}


def _prefixes(obs):
    blocks = defaultdict(list)
    for o in obs:
        blocks[o.confidence].append(o)
    n = len(obs)
    groups = set()
    accepted = wrong = 0
    yield {"cutoff": None, **_point(n, 0, 0, 0)}
    for conf in sorted(blocks, reverse=True):
        block = blocks[conf]
        accepted += len(block)
        wrong += sum(not o.correct for o in block)
        groups.update(o.group for o in block)
        yield {"cutoff": conf, **_point(n, accepted, wrong, len(groups))}


def _targets(values):
    if not isinstance(values, (list, tuple)) or not values or not all(_probability(x) for x in values) or len(set(values)) != len(values):
        raise ev.EvaluationError("provide distinct targets in [0,1]")
    return sorted(values)


def fit_risk_cutoffs(inputs, labels, predictions, *, split, risk_targets=(.01, .05, .10), min_groups=10):
    """Select whole-tie cutoffs on calibration only. Custody and arm identity are external pins."""
    if split != "calibration" or type(min_groups) is not int or min_groups < 1:
        raise ev.EvaluationError("use calibration only and positive minimum independent support")
    targets = _targets(risk_targets)
    obs = observations(inputs, labels, predictions)
    points = list(_prefixes(obs))
    selected = {}
    for target in targets:
        valid = [p for p in points if p["accepted_groups"] >= min_groups and p["risk"] <= target]
        selected[str(target)] = {**(valid[-1] if valid else points[0]), "target": target,
                                 "status": "fitted" if valid else "no_supported_cutoff"}
    return {"fit_split": split, "min_groups": min_groups, "cutoffs": selected,
            "interpretation": "Maximum observed calibration coverage under the empirical risk constraint, not a guarantee."}


def _bin_intervals(obs, reps, seed):
    groups = defaultdict(lambda: defaultdict(dict))
    support = [set() for _ in range(ev.ECE_BINS)]
    for o in obs:
        b = ev._bin(o.confidence)
        d = groups[o.task][o.group]
        n, ok = d.get(b, (0, 0))
        d[b] = n + 1, ok + o.correct
        support[b].add(o.group)
    strata = [[d[g] for g in sorted(d)] for _, d in sorted(groups.items())]
    enough = all(len(s) >= 2 for s in strata)
    draws = [[] for _ in range(ev.ECE_BINS)]
    rng = random.Random(seed)
    for _ in range(reps if enough else 0):
        pop = [0] * ev.ECE_BINS
        correct = [0] * ev.ECE_BINS
        for s in strata:
            for i, mult in Counter(rng.randrange(len(s)) for _ in s).items():
                for b, (n, ok) in s[i].items():
                    pop[b] += n * mult
                    correct[b] += ok * mult
        for b, n in enumerate(pop):
            if n:
                draws[b].append(correct[b] / n)
    out = []
    for b, ds in enumerate(draws):
        groups_n = len(support[b])
        status = ("empty" if not groups_n else "not_requested" if not reps else
                  "insufficient_independent_groups" if groups_n < 2 or not enough else
                  "missing_in_some_resamples" if len(ds) != reps else "ok")
        ds.sort()
        ci = [ev._percentile(ds, p) for p in (.025, .975)] if status == "ok" else None
        out.append({"independent_groups": groups_n, "accuracy_ci95": ci, "ci_status": status,
                    "defined_replicates": len(ds)})
    return out


def describe(inputs, labels, predictions, *, coverage_targets=(.25, .5, .75, .9, 1.0),
             fitted: Mapping | None = None, reps=2000, seed=4117):
    """Descriptive tie-aware coverage, reliability and optional out-of-sample cutoff application."""
    if type(reps) is not int or not 0 <= reps <= 10000 or type(seed) is not int:
        raise ev.EvaluationError("invalid bootstrap settings")
    obs = observations(inputs, labels, predictions)
    targets = _targets(coverage_targets)
    points = list(_prefixes(obs))
    fixed = {str(target): max((p for p in points if p["coverage"] <= target), key=lambda p: p["accepted"])
             for target in targets}
    empirical = ev._summary(obs, .9)
    bins = [{**b, **ci} for b, ci in zip(empirical["ece_bins_valid_only"], _bin_intervals(obs, reps, seed), strict=True)]
    applied = {}
    if fitted is not None:
        cutoffs = fitted.get("cutoffs")
        if (fitted.get("fit_split") != "calibration" or not isinstance(cutoffs, Mapping) or not cutoffs
                or type(fitted.get("min_groups")) is not int or fitted["min_groups"] < 1):
            raise ev.EvaluationError("invalid calibration policy")
        for name, f in cutoffs.items():
            c = f["cutoff"]
            if (not _probability(f["target"]) or str(f["target"]) != name or
                    (c is not None and (not _probability(c) or f["status"] != "fitted"
                     or f["accepted_groups"] < fitted["min_groups"] or not _probability(f["risk"])
                     or f["risk"] > f["target"])) or
                    (c is None and (f["status"] != "no_supported_cutoff" or f["accepted"] != 0))):
                raise ev.EvaluationError("invalid fitted risk operating point")
            selected = [o for o in obs if c is not None and o.confidence >= c]
            applied[name] = {"target": f["target"], "frozen_cutoff": c, **_point(
                len(obs), len(selected), sum(not o.correct for o in selected), len({o.group for o in selected}))}
    by_outcome = {}
    for name, yes in (("correct", True), ("incorrect", False)):
        values = [o.confidence for o in obs if o.correct == yes]
        by_outcome[name] = {"n": len(values), "mean_top_probability": math.fsum(values) / len(values) if values else None}
    return {"n": len(obs), "groups": len({o.group for o in obs}), "reliability": bins, "coverage_points": fixed,
            "risk_coverage": empirical["risk_coverage"], "applied_calibration_cutoffs": applied,
            "confidence_by_outcome": by_outcome, "bootstrap_reps": reps, "seed": seed}
