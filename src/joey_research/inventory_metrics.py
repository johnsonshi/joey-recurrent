"""Paired, typed decision endpoints for the synthetic inventory development experiment."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import random

from . import evaluation as ev


@dataclass(frozen=True)
class Menu:
    seed: int
    length: int
    kind: str
    keys: tuple[str, ...]
    meanings: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    id: str
    digest: str
    menu: Menu
    quality: ev._Observation
    truth: str
    top: str
    declared: str | None
    native_value: float | None
    # Names mirror report units. Squared ordinal error is transformed only AFTER aggregation.
    values: Mapping[str, float | None]


def _require(test, message):
    if not test:
        raise ev.EvaluationError(message)


def _numeric(x, upper):
    return type(x) in (float, int) and 0 <= x <= upper


def decisions(inputs, labels, predictions, native_answers, *, menus: Mapping[str, Menu]):
    """Rows must exactly follow the pinned shard; never infer legal keys from a probability map."""
    rows = list(inputs)
    indexed = ev._index_inputs(rows)
    ids = [r["id"] for r in rows]
    _require(set(ids) == set(menus), "wrong menu coverage")
    gold, pred, typed = [], [], []
    for stream, target, keys in ((labels, gold, {"id", "input_sha256", "gold_index"}),
                                  (predictions, pred, {"id", "input_sha256", "probs"}),
                                  (native_answers, typed, {"id", "input_sha256", "answer"})):
        for x, expected in zip(stream, ids, strict=True):
            _require(isinstance(x, dict) and set(x) == keys and x["id"] == expected
                     and x["input_sha256"] == indexed[expected][1], "changed record, order or input digest")
            target.append(x)
    result = []
    for r, g, p, t in zip(rows, gold, pred, typed, strict=True):
        key = r["id"]
        m, y, a = menus[key], g["gold_index"], t["answer"]
        _require(isinstance(m, Menu) and type(m.seed) is int and type(m.length) is int and m.length > 0
                 and m.kind in {"choice", "noul", "score"} and isinstance(m.keys, tuple) and isinstance(m.meanings, tuple)
                 and len(m.keys) == len(set(m.keys)) == len(r["choices"]) == len(m.meanings) == len(set(m.meanings))
                 and all(isinstance(s, str) and s for s in m.keys + m.meanings)
                 and type(y) is int and 0 <= y < len(m.keys), "invalid frozen item metadata")
        o = ev._observe(r, y, p)
        _require(o.error is None and isinstance(a, dict) and a.get("type") == m.kind, "invalid native or raw decision")
        v = [x / math.fsum(p["probs"]) for x in p["probs"]]
        vals = {"argmax_accuracy": float(o.correct), "summed_brier": o.brier,
                "nll": o.nll if math.isfinite(o.nll) else None}
        declared = None
        value = None
        truth = m.meanings[y]
        if m.kind in ("choice", "score"):
            probs = a.get("probabilities")
            _require(isinstance(probs, dict) and set(probs) == set(m.keys)
                     and all(_numeric(probs[k], 1) and abs(probs[k] - z) <= 1.01e-6
                             for k, z in zip(m.keys, v, strict=True)), "rounded distribution does not match the recorded slots")
        if m.kind == "choice":
            _require(set(a) == {"type", "choice", "probabilities"} and a["choice"] in m.keys
                     and len(m.keys) == 5 and "tie" in m.meanings, "invalid five-way declared choice")
            declared = m.meanings[m.keys.index(a["choice"])]
            vals["native_choice_accuracy"] = float(declared == truth)
        elif m.kind == "noul":
            _require(set(a) == {"type", "noul"} and m.meanings == ("yes", "no") and _numeric(a["noul"], 1)
                     and abs(a["noul"] - v[0]) <= 1.01e-6, "invalid binary p_yes")
            value = a["noul"]
            declared = "yes" if value >= .5 else "no"
            pr_gold = value if truth == "yes" else 1 - value
            vals.update(binary_accuracy=float(declared == truth), binary_brier=(value - int(truth == "yes")) ** 2,
                        binary_log_loss=-math.log(pr_gold) if pr_gold else None)
        else:
            _require(set(a) == {"type", "score", "probabilities"} and m.keys == m.meanings == ("0", "1", "2", "3")
                     and _numeric(a["score"], 3) and abs(a["score"] - math.fsum(i*z for i, z in enumerate(v))) <= 1.01e-6,
                     "invalid ordinal expected score or order")
            value = a["score"]
            vals.update(ordinal_mae=abs(value-y), ordinal_squared_error=(value-y)**2,
                        ordinal_rps=math.fsum((math.fsum(v[:i+1]) - int(y <= i))**2 for i in range(3))/3)
        result.append(Decision(key, g["input_sha256"], m, o, truth, m.meanings[o.choice], declared, value, vals))
    _groups(result)
    return result


def _groups(obs):
    groups = defaultdict(list)
    for x in obs:
        groups[x.quality.group].append(x)
    for pair in groups.values():
        _require(len(pair) == 2, "requires both and only both nuisance renderings")
        a, b = pair
        _require(a.id != b.id and a.digest != b.digest and a.truth == b.truth and set(a.menu.meanings) == set(b.menu.meanings)
                 and (a.menu.seed, a.menu.length, a.menu.kind, a.quality.task) ==
                     (b.menu.seed, b.menu.length, b.menu.kind, b.quality.task), "renderings disagree on group semantics")
    return groups


def _means(obs):
    fields = sorted(obs[0].values)
    _require(all(set(x.values) == set(fields) for x in obs), "multiple native decision types in a task")
    out = {}
    for f in fields:
        z = [x.values[f] for x in obs]
        if f == "ordinal_squared_error":
            out["ordinal_rmse"] = math.sqrt(math.fsum(z)/len(z))
        else:
            out[f] = None if any(x is None for x in z) else math.fsum(z)/len(z)
    return out


def _support(obs):
    groups = _groups(obs)
    kind = obs[0].menu.kind
    support = {"n": len(obs), "groups": len(groups), "truth_counts": dict(sorted(Counter(x.truth for x in obs).items())),
               "argmax_ties": sum(x.quality.tied for x in obs),
               "exact_zero_gold_probability": sum(x.values["nll"] is None for x in obs),
               "top_semantic_agreement_groups": sum(a.top == b.top for a,b in groups.values())}
    if kind != "score":
        support.update(native_declared_agreement_groups=sum(a.declared == b.declared for a,b in groups.values()),
                       native_not_first_argmax=sum(x.declared != x.top for x in obs))
    if kind == "choice":
        support.update(declared_tie=sum(x.declared == "tie" for x in obs),
                       correct_declared_tie=sum(x.declared == x.truth == "tie" for x in obs))
    elif kind == "noul":
        support.update(native_exactly_point_five=sum(x.native_value == .5 for x in obs),
                       native_binary_infinite_loss_count=sum(x.values["binary_log_loss"] is None for x in obs))
    else:
        support["mean_abs_within_group_expected_score_difference"] = math.fsum(
            abs(a.native_value-b.native_value) for a,b in groups.values())/len(groups)
    return support


def summarize(obs: Sequence[Decision], *, thresholds=(.8, .9, .95, .99)):
    _require(bool(obs), "empty decision panel")
    tasks = defaultdict(list)
    for x in obs:
        tasks[x.quality.task].append(x)
    report = {}
    for t, xs in sorted(tasks.items()):
        cells = defaultdict(list)
        for x in xs:
            cells[x.menu.seed, x.menu.length].append(x)
        quality = [x.quality for x in xs]
        probs = ev._summary(quality, .9)
        conf = {}
        for c in thresholds:
            _require(_numeric(c, 1), "invalid confidence threshold")
            accepted = [o for o in quality if o.confidence >= c]
            wrong = sum(not o.correct for o in accepted)
            conf[str(c)] = {"accepted": len(accepted), "wrong": wrong, "coverage": len(accepted)/len(xs),
                            "joint_error": wrong/len(xs), "conditional_error": wrong/len(accepted) if accepted else None}
        report[t] = {"native_kind": xs[0].menu.kind, "means": _means(xs), "support": _support(xs),
                     "full_precision": probs, "confidence": conf,
                     "cells": [{"seed": s, "ledger_length": l, **_support(v), "means": _means(v)}
                               for (s,l), v in sorted(cells.items())]}
    return report


def paired(baseline: Sequence[Decision], candidate: Sequence[Decision], *, reps=2000, seed=4117):
    """Candidate minus baseline, resampling entire rendering pairs within task, seed and difficulty."""
    _require(type(reps) is int and 1 <= reps <= 10000 and type(seed) is int, "invalid resampling rule")
    by_task = defaultdict(list)
    for a, b in zip(baseline, candidate, strict=True):
        _require((a.id, a.digest, a.menu, a.quality.group, a.quality.task, a.truth) ==
                 (b.id, b.digest, b.menu, b.quality.group, b.quality.task, b.truth), "unpaired transfer decisions")
        by_task[a.quality.task].append((a,b))
    _require(bool(by_task), "empty paired comparison")
    report = {}
    for t, pairs in sorted(by_task.items()):
        xa, xb = zip(*pairs)
        ga, gb = _groups(xa), _groups(xb)
        fields = sorted(xa[0].values)
        good = [f for f in fields if all(x.values[f] is not None for arm in (xa,xb) for x in arm)]
        am, bm = _means(xa), _means(xb)
        strata = defaultdict(list)
        for g in sorted(ga):
            aa, bb = ga[g], gb[g]
            values = []
            for f in good:
                u, v = (math.fsum(x.values[f] for x in arm)/2 for arm in (aa,bb))
                values.extend((u,v) if f == "ordinal_squared_error" else (v-u,))
            strata[aa[0].menu.seed,aa[0].menu.length].append(values)
        draws = {f:[] for f in good}
        supported = all(len(v) >= 2 for v in strata.values())
        rng = random.Random(seed)
        for _ in range(reps if supported else 0):
            sample = [pool[rng.randrange(len(pool))] for _, pool in sorted(strata.items()) for _ in pool]
            av = iter(math.fsum(col)/len(sample) for col in zip(*sample))
            for f in good:
                u = next(av)
                draws[f].append(math.sqrt(next(av))-math.sqrt(u) if f == "ordinal_squared_error" else u)
        intervals = {}
        for f in fields:
            metric = "ordinal_rmse" if f == "ordinal_squared_error" else f
            valid = f in good
            ds = sorted(draws[f]) if valid else []
            intervals[metric] = {"delta": bm[metric]-am[metric] if valid else None,
                                 "ci95": [ev._percentile(ds,p) for p in (.025,.975)] if ds else None,
                                 "status": "nonfinite_loss" if not valid else "ok" if ds else "insufficient_independent_groups"}
        report[t] = {"decisions": len(xa), "groups":len(ga), "reps":reps, "pointwise":True, "deltas":intervals,
                     "argmax_corrections":sum(not a.quality.correct and b.quality.correct for a,b in pairs),
                     "argmax_corruptions":sum(a.quality.correct and not b.quality.correct for a,b in pairs)}
    return report
