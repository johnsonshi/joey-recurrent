"""Paired, seed-stratified graph contrasts for a frozen routing panel."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import product, zip_longest
import math
import random
from typing import NamedTuple

from joey_research import evaluation as ev


_HOPS = (1, 2, 4, 6)
_EFFECTS = (*_HOPS, "primary", "exact_token")
_THRESHOLDS = (.8, .9, .95, .99)


class _Pair(NamedTuple):
    baseline: ev._Observation
    candidate: ev._Observation
    tokens: int

    def delta(self):
        return (self.candidate.correct - self.baseline.correct,
                self.candidate.brier - self.baseline.brier)


@dataclass
class _Graph:
    seed: int
    task: str
    menus: dict[int, tuple[str, ...]] = field(default_factory=dict)
    cells: dict[tuple[str, int, int], _Pair] = field(default_factory=dict)


def _bind(record, field_name, row_id, digest):
    if not isinstance(record, Mapping) or set(record) != {"id", "input_sha256", field_name}:
        raise ev.EvaluationError(f"expected a bound {field_name} record")
    if record["id"] != row_id or record["input_sha256"] != digest:
        raise ev.EvaluationError(f"{field_name} order or input digest differs")


def _observe(row, gold, prediction):
    # Use the evaluator's actual tolerance, renormalization, tie rule and Brier scale.
    obs = ev._observe(row, gold, prediction)
    if obs.error is not None:
        raise ev.EvaluationError(f"invalid paired prediction: {obs.error}")
    return obs


def _panel(rows, labels, baseline, candidate, design_by_id, baseline_tokens, candidate_tokens):
    maps = (design_by_id, baseline_tokens, candidate_tokens)
    if not all(isinstance(m, Mapping) for m in maps):
        raise ev.EvaluationError("design and recorded token counts must be ID mappings")
    seen, graphs = set(), {}
    missing = object()
    for records in zip_longest(rows, labels, baseline, candidate, fillvalue=missing):
        if any(item is missing for item in records):
            raise ev.EvaluationError("row, label and prediction counts differ")
        row, label, base, cand = records
        digest = ev.input_sha256(row)
        key, group, choices = row["id"], row["group_id"], row["choices"]
        if len(choices) != 4 or key in seen:
            raise ev.EvaluationError("expected four choices and unique row IDs")
        seen.add(key)
        _bind(label, "gold_index", key, digest)
        _bind(base, "probs", key, digest)
        _bind(cand, "probs", key, digest)
        gold = label["gold_index"]
        if type(gold) is not int or not 0 <= gold < 4:
            raise ev.EvaluationError("gold_index must be an integer in the four-choice menu")

        if any(key not in m for m in maps):
            raise ev.EvaluationError("missing design or token count")
        meta = design_by_id[key]
        if not isinstance(meta, Mapping) or set(meta) != {"hop", "entry", "rendering", "seed"}:
            raise ev.EvaluationError("invalid design fields")
        hop, entry, rendering, seed = (meta[k] for k in ("hop", "entry", "rendering", "seed"))
        if (type(hop) is not int or hop not in _HOPS or type(rendering) is not int
                or rendering not in (0, 1) or type(seed) is not int):
            raise ev.EvaluationError("invalid design hop, rendering or seed")
        if not isinstance(entry, str) or not entry.strip() or choices[gold] != f"Enter at {entry}":
            raise ev.EvaluationError("semantic entry does not match the gold choice")
        a, b = baseline_tokens[key], candidate_tokens[key]
        if type(a) is not int or type(b) is not int or a <= 0 or b <= 0 or a != b:
            raise ev.EvaluationError("both recorded token counts must be positive, identical integers")

        if group not in graphs:
            graphs[group] = _Graph(seed, row["task"])
        graph = graphs[group]
        if graph.seed != seed or graph.task != row["task"]:
            raise ev.EvaluationError("a graph spans generator seeds or tasks")
        menu = tuple(choices)
        if graph.menus.setdefault(rendering, menu) != menu:
            raise ev.EvaluationError("ordered menu changes within graph and rendering")
        cell = (entry, rendering, hop)
        if cell in graph.cells:
            raise ev.EvaluationError("duplicate graph design cell")
        graph.cells[cell] = _Pair(_observe(row, gold, base), _observe(row, gold, cand), a)
    if not graphs or any(m.keys() != seen for m in maps):
        raise ev.EvaluationError("expected a nonempty panel and exact design/token coverage")
    return graphs


class _Sums(NamedTuple):
    n: int
    accuracy: int
    brier: float


def _sums(deltas):
    return _Sums(len(deltas), sum(a for a, _ in deltas), math.fsum(b for _, b in deltas))


def _graph_sums(graph, by_hop, token_differences):
    cells = graph.cells
    entries = sorted({e for e, _, _ in cells})
    if len(entries) != 4 or cells.keys() != set(product(entries, (0, 1), _HOPS)):
        raise ev.EvaluationError("expected four entries, two renderings and all hops in each graph")
    tuples = list(product(entries, (0, 1)))
    sums = {}
    for hop in _HOPS:
        pairs = [cells[e, r, hop] for e, r in tuples]
        by_hop[hop].extend(pairs)
        sums[hop] = _sums([p.delta() for p in pairs])

    all_changes, matched_changes = [], []
    for e, r in tuples:
        near, far = cells[e, r, 2], cells[e, r, 6]
        near_acc, near_brier = near.delta()
        far_acc, far_brier = far.delta()
        change = (far_acc - near_acc, far_brier - near_brier)
        all_changes.append(change)
        difference = far.tokens - near.tokens
        token_differences[difference] += 1
        if difference == 0:
            matched_changes.append(change)
    sums["primary"] = _sums(all_changes)
    sums["exact_token"] = _sums(matched_changes)
    return sums


def _ratio(num, den):
    return num / den if den else None


def _pooled(graphs, effect):
    ns, acc, brier = zip(*(graph[effect] for graph in graphs))
    n = sum(ns)
    return _ratio(sum(acc), n), _ratio(math.fsum(brier), n)


def _interval(estimate, draws, reps):
    ci = None
    if reps and len(draws) == reps:
        draws.sort()
        ci = [ev._percentile(draws, q) for q in (.025, .975)]
    return {"estimate": estimate, "ci95": ci, "defined_reps": len(draws)}


def _bootstrap(strata, reps, seed):
    graphs = [g for s in strata for g in s]
    active = [k for k in _EFFECTS if all(sum(g[k].n > 0 for g in s) >= 2 for s in strata)]
    draws = {k: ([], []) for k in _EFFECTS}
    rng = random.Random(seed)
    for _ in range(reps if active else 0):
        # One cluster multiplicity vector, including ineligible graphs, for every effect.
        sample = [rng.choice(s) for s in strata for _ in s]
        for k in active:
            acc, brier = _pooled(sample, k)
            if acc is not None:
                draws[k][0].append(acc)
                draws[k][1].append(brier)
    return {
        k: {name: _interval(point, values, reps)
            for name, point, values in zip(("accuracy", "brier"), _pooled(graphs, k), draws[k])}
        for k in _EFFECTS
    }


def _arm(obs):
    right = [o.confidence for o in obs if o.correct]
    wrong = [o.confidence for o in obs if not o.correct]
    thresholds = {}
    for t in _THRESHOLDS:
        errors = sum(c >= t for c in wrong)
        support = errors + sum(c >= t for c in right)
        thresholds[str(t)] = {"errors": errors, "support": support, "error_rate": _ratio(errors, support)}
    return {
        "accuracy": len(right) / len(obs), "brier": math.fsum(o.brier for o in obs) / len(obs),
        "right": len(right), "wrong": len(wrong),
        "mean_confidence_right": _ratio(math.fsum(right), len(right)),
        "mean_confidence_wrong": _ratio(math.fsum(wrong), len(wrong)),
        "thresholds": thresholds,
    }


def _conditional(count, support):
    return {"count": count, "support": support, "rate": _ratio(count, support)}


def _hop_summary(pairs, n_graphs, effect):
    base = _arm([p.baseline for p in pairs])
    cand = _arm([p.candidate for p in pairs])
    corrections = sum(not p.baseline.correct and p.candidate.correct for p in pairs)
    corruptions = sum(p.baseline.correct and not p.candidate.correct for p in pairs)
    tokens = [p.tokens for p in pairs]
    return {
        "rows": len(pairs), "graphs": n_graphs, "baseline": base, "candidate": cand,
        "paired": effect, "correction": _conditional(corrections, base["wrong"]),
        "corruption": _conditional(corruptions, base["right"]),
        "tokens": {"min": min(tokens), "max": max(tokens), "mean": sum(tokens) / len(tokens)},
    }


def _support(graphs):
    return {
        "total_graphs": len(graphs), "eligible_graphs": sum(g["exact_token"].n > 0 for g in graphs),
        "total_tuples": sum(g["primary"].n for g in graphs),
        "eligible_tuples": sum(g["exact_token"].n for g in graphs),
    }


def compare_horizons(
    rows: Iterable[ev.Row], labels: Iterable[ev.Row], baseline: Iterable[ev.Row],
    candidate: Iterable[ev.Row], *, design_by_id: Mapping[str, dict],
    baseline_tokens: Mapping[str, int], candidate_tokens: Mapping[str, int],
    reps: int = 2000, seed: int = 4117,
) -> dict:
    """Compare two already-frozen arms in memory. See docs/horizon-metrics.md for the estimands."""
    if type(reps) is not int or reps < 0 or type(seed) is not int:
        raise ev.EvaluationError("bootstrap needs a nonnegative integer replicate count and integer seed")
    panel = _panel(rows, labels, baseline, candidate, design_by_id, baseline_tokens, candidate_tokens)
    by_hop = {h: [] for h in _HOPS}
    by_seed = defaultdict(list)
    differences = Counter()
    for group in sorted(panel):
        graph = panel[group]
        by_seed[graph.seed].append(_graph_sums(graph, by_hop, differences))
    strata = [by_seed[s] for s in sorted(by_seed)]
    effects = _bootstrap(strata, reps, seed)
    support = _support([g for s in strata for g in s])
    n = support["total_tuples"]
    return {
        "bootstrap": {"reps": reps, "seed": seed},
        "by_hop": {str(h): _hop_summary(by_hop[h], len(panel), effects[h]) for h in _HOPS},
        "primary": effects["primary"],
        "exact_token": {
            **effects["exact_token"], "support": support,
            "by_seed": {str(s): _support(by_seed[s]) for s in sorted(by_seed)},
        },
        "token_differences": {
            "counts": {str(d): differences[d] for d in sorted(differences)},
            "min": min(differences), "max": max(differences),
            "mean": sum(d * k for d, k in differences.items()) / n,
            "mean_abs": sum(abs(d) * k for d, k in differences.items()) / n,
            "equal_fraction": differences[0] / n,
        },
    }
