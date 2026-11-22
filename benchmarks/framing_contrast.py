"""Paired declared-choice contrasts for alternate presentations of one fixed menu."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import math
import random

from joey_research import evaluation as ev


def choice_contrasts(inputs: Iterable[ev.Row], labels: Iterable[ev.Row],
                     arms: Mapping[str, Iterable[ev.Row]], contrasts: Mapping[str, Mapping[str, int]],
                     *, reps: int = 2000, seed: int = 4117) -> dict:
    """Score +1/-1 accuracy and class-macro-F1 contrasts using common whole-group draws.

    Each arm must supply exactly one explicit zero-based choice_index per input.
    The caller decides whether that is a native declared decision or a probability
    projection's argmax. This function never substitutes one for the other.
    """
    ix = ev._index_inputs(inputs)
    first = next(iter(ix.values()))[0]
    if any(r["task"] != first["task"] or r["choices"] != first["choices"] for r, _ in ix.values()):
        raise ev.EvaluationError("choice contrasts require one task and one identical ordered menu")
    if (not arms or not all(ev._text(n) for n in arms) or not contrasts
            or type(reps) is not int or not 0 <= reps <= 10000 or type(seed) is not int):
        raise ev.EvaluationError("invalid contrast panel or resampling settings")
    names = sorted(arms)
    ordered_keys = sorted(ix)
    gold = ev._bind(labels, ix, ({"id", "input_sha256", "gold_index"},), "labels")
    bound = [ev._bind(arms[n], ix, ({"id", "input_sha256", "choice_index"},), n) for n in names]
    width = len(first["choices"])
    group_rows = defaultdict(list)
    for key in ordered_keys:
        y = gold[key]["gold_index"]
        chosen = tuple(a[key]["choice_index"] for a in bound)
        if any(type(j) is not int or not 0 <= j < width for j in (y, *chosen)):
            raise ev.EvaluationError("decision index outside the fixed menu")
        group_rows[ix[key][0]["group_id"]].append((y, chosen))
    groups = [group_rows[k] for k in sorted(group_rows)]
    terms = {}
    for label, coefficients in contrasts.items():
        if (not ev._text(label) or not isinstance(coefficients, Mapping) or len(coefficients) not in (2, 4)
                or not coefficients.keys() <= arms.keys()
                or any(type(w) is not int or w not in (-1, 1) for w in coefficients.values())
                or sum(coefficients.values()) != 0):
            raise ev.EvaluationError("expected a named two-arm difference or four-arm difference-in-differences")
        terms[label] = [(names.index(n), w) for n, w in coefficients.items()]

    def statistics(multiplicities):
        n = 0
        actual = [0] * width
        pred = [[0] * width for _ in names]
        tp = [[0] * width for _ in names]
        for g, times in multiplicities:
            for y, chosen in groups[g]:
                n += times
                actual[y] += times
                for arm, j in enumerate(chosen):
                    pred[arm][j] += times
                    if j == y:
                        tp[arm][y] += times
        values = []
        for pc, tc in zip(pred, tp, strict=True):
            den = [a + b for a, b in zip(actual, pc, strict=True)]
            classes = sum(d > 0 for d in den)
            f1 = math.fsum(2 * t / d for t, d in zip(tc, den, strict=True) if d) / classes
            values.append((sum(tc) / n, f1, classes))
        return values, {k: tuple(math.fsum(w * values[i][j] for i, w in pairs) for j in (0, 1))
                        for k, pairs in terms.items()}

    point, estimates = statistics((g, 1) for g in range(len(groups)))
    draws = {k: ([], []) for k in terms}
    enough = len(groups) >= 2
    rng = random.Random(seed)
    for _ in range(reps if enough else 0):
        counts = Counter(rng.randrange(len(groups)) for _ in groups)
        _, delta = statistics(counts.items())
        for k, (acc, f1) in delta.items():
            draws[k][0].append(acc)
            draws[k][1].append(f1)
    comparisons = {}
    for k, pairs in terms.items():
        comparison = {"coefficients": dict(contrasts[k])}
        for j, metric in enumerate(("accuracy", "observed_class_macro_f1")):
            values = sorted(draws[k][j])
            comparison[metric] = {"delta": estimates[k][j], "ci95": [ev._percentile(values, p) for p in (.025, .975)]
                                  if values else None, "status": "ok" if values else
                                  "not_requested" if not reps else "insufficient_independent_groups"}
        if len(pairs) == 2:
            a = next(i for i, w in pairs if w == -1)
            b = next(i for i, w in pairs if w == 1)
            comparison["corrected"] = sum(c[a] != y and c[b] == y for rows in groups for y, c in rows)
            comparison["corrupted"] = sum(c[a] == y and c[b] != y for rows in groups for y, c in rows)
        comparisons[k] = comparison
    return {"rows": len(ix), "groups": len(groups), "reps": reps, "seed": seed,
            "arms": {n: {"accuracy": v[0], "observed_class_macro_f1": v[1], "classes_scored": v[2]}
                     for n, v in zip(names, point, strict=True)},
            "contrasts": comparisons,
            "intervals": "Pointwise paired percentile intervals. Resample whole groups; recompute each arm's observed-class support."}
