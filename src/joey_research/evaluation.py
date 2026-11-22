"""Offline scoring of ordered, finite decision menus. Requires only Python >=3.10."""

from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any
import unicodedata


Row = dict[str, Any]
INPUT_FIELDS = {"id", "group_id", "task", "prompt", "choices"}
SCHEMA = "joey-decision-v1"
PROB_SUM_ATOL = 1e-6
ECE_BINS = 15
_ECE_EDGES = tuple(i / ECE_BINS for i in range(1, ECE_BINS))
DEFAULT_THRESHOLD = 0.8
BOOT_METRICS = (
    "accuracy", "nll", "brier", "ece15", "confident_coverage",
    "confident_error_joint", "confident_error_conditional",
)


class EvaluationError(ValueError):
    """A structural, pairing, or protocol error, not an incorrect decision."""


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def input_sha256(row: Row) -> str:
    """Validate the label-free allowlist and bind IDs, text, and ordered options."""
    if not isinstance(row, Mapping) or set(row) != INPUT_FIELDS:
        raise EvaluationError("input must contain exactly id, group_id, task, prompt, choices")
    if not all(_text(row[key]) for key in ("id", "group_id", "task", "prompt")):
        raise EvaluationError("input IDs, task and prompt must be nonempty strings")
    choices = row["choices"]
    if not isinstance(choices, list) or len(choices) < 2 or not all(map(_text, choices)):
        raise EvaluationError("choices must be an ordered list of at least two nonempty strings")
    if len(set(map(_normalized, choices))) != len(choices):
        raise EvaluationError("menu contains indistinguishable choices")
    return _hash(row)


def label_row(row: Row, gold_index: int) -> Row:
    digest = input_sha256(row)
    if type(gold_index) is not int or not 0 <= gold_index < len(row["choices"]):
        raise EvaluationError("gold_index must be a zero-based integer in this menu")
    return {"id": row["id"], "input_sha256": digest, "gold_index": gold_index}


def _index_inputs(inputs: Iterable[Row]) -> dict[str, tuple[Row, str]]:
    indexed = {}
    groups: dict[str, str] = {}
    states: dict[tuple[str, tuple[str, ...]], str] = {}
    for row in inputs:
        digest = input_sha256(row)
        key, group, task = row["id"], row["group_id"], row["task"]
        if key in indexed:
            raise EvaluationError(f"duplicate input ID: {key}")
        if groups.setdefault(group, task) != task:
            raise EvaluationError("a group spans tasks; it cannot be independently resampled by task")
        state = (_normalized(row["prompt"]), tuple(sorted(map(_normalized, row["choices"]))))
        if states.setdefault(state, group) != group:
            raise EvaluationError("identical state/menu assigned to different groups")
        indexed[key] = (row, digest)
    if not indexed:
        raise EvaluationError("empty input set")
    return indexed


def _sha256_text(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def freeze_split(inputs: Iterable[Row], *, seed: int, weights: Mapping[str, int],
                 config_sha256: str | None = None) -> Row:
    """Allocate WHOLE groups by task, without labels. Remainders break by split name."""
    indexed = _index_inputs(inputs)
    if type(seed) is not int or not isinstance(weights, Mapping) or not weights or not all(
        _text(name) and type(w) is int and w > 0 for name, w in weights.items()
    ):
        raise EvaluationError("split requires an integer seed and positive named integer weights")
    if config_sha256 is not None and not _sha256_text(config_sha256):
        raise EvaluationError("invalid config digest")
    groups_by_task: dict[str, set[str]] = defaultdict(set)
    for row, _ in indexed.values():
        groups_by_task[row["task"]].add(row["group_id"])
    assignment = {}
    names = sorted(weights)
    total = sum(weights.values())
    for task, groups in sorted(groups_by_task.items()):
        order = sorted(groups, key=lambda g: (_hash([SCHEMA, "split", seed, task, g]), g))
        quotients = {n: divmod(len(order) * weights[n], total) for n in names}
        counts = {n: quotients[n][0] for n in names}
        remainder_order = sorted(names, key=lambda n: (-quotients[n][1], n))
        for n in remainder_order[:len(order) - sum(counts.values())]:
            counts[n] += 1
        if not all(counts.values()):
            raise EvaluationError(f"too few independent groups for requested split weights: {task}")
        offset = 0
        for n in names:
            assignment.update(dict.fromkeys(order[offset:offset + counts[n]], n))
            offset += counts[n]
    entries = [
        {"id": key, "group_id": row["group_id"], "task": row["task"],
         "split": assignment[row["group_id"]], "input_sha256": digest}
        for key, (row, digest) in sorted(indexed.items())
    ]
    body = {"schema": SCHEMA, "seed": seed, "weights": dict(weights),
            "config_sha256": config_sha256, "entries": entries}
    return {**body, "sha256": _hash(body)}


def select_frozen_split(inputs: Iterable[Row], manifest: Row, expected_sha256: str,
                        split: str) -> list[Row]:
    """Verify the full universe and a separately pinned digest before subsetting."""
    rows = list(inputs)
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema", "seed", "weights", "config_sha256", "entries", "sha256"
    }:
        raise EvaluationError("invalid split manifest schema")
    rebuilt = freeze_split(rows, seed=manifest["seed"], weights=manifest["weights"],
                           config_sha256=manifest["config_sha256"])
    if rebuilt != manifest or rebuilt["sha256"] != expected_sha256:
        raise EvaluationError("manifest, input universe, or expected freeze digest differs")
    if split not in rebuilt["weights"]:
        raise EvaluationError("unknown split")
    keep = {e["id"] for e in rebuilt["entries"] if e["split"] == split}
    return sorted((row for row in rows if row["id"] in keep), key=lambda r: r["id"])


def _bind(records: Iterable[Row], indexed: dict, shapes: tuple[set[str], ...],
          kind: str) -> dict[str, Row]:
    found = {}
    for record in records:
        if not isinstance(record, dict) or set(record) not in shapes:
            raise EvaluationError(f"{kind}: unexpected row schema")
        key = record["id"]
        if not _text(key) or key not in indexed:
            raise EvaluationError(f"{kind}: ID outside the selected universe")
        if key in found:
            raise EvaluationError(f"{kind}: duplicate ID: {key}")
        if record["input_sha256"] != indexed[key][1]:
            raise EvaluationError(f"{kind}: stale input digest: {key}")
        found[key] = record
    if len(found) != len(indexed):
        raise EvaluationError(f"{kind}: missing {len(indexed) - len(found)} ID(s)")
    return found


@dataclass(frozen=True, slots=True)
class _Observation:
    task: str
    group: str
    choice: int | None = None
    correct: bool = False
    confidence: float = 0.0
    nll: float = 0.0
    brier: float = 0.0
    tied: bool = False
    error: str | None = None


def _observe(row: Row, gold: int, pred: Row) -> _Observation:
    task, group = row["task"], row["group_id"]
    if "error" in pred:
        if not _text(pred["error"]):
            raise EvaluationError("explicit runner error must have a nonempty code")
        return _Observation(task, group, error="runner_error")
    probs = pred["probs"]
    if not isinstance(probs, list) or len(probs) != len(row["choices"]):
        return _Observation(task, group, error="wrong_shape")
    if any(type(p) not in (int, float) or not (0 <= p <= 1) for p in probs):
        return _Observation(task, group, error="nonfinite_or_out_of_range")
    total = math.fsum(probs)
    if abs(total - 1) > PROB_SUM_ATOL:
        return _Observation(task, group, error="not_normalized")
    probs = [p / total for p in probs]
    conf = max(probs)
    choice = probs.index(conf)
    nll = -math.log(probs[gold]) if probs[gold] else math.inf
    brier = math.fsum((p - int(i == gold)) ** 2 for i, p in enumerate(probs))
    return _Observation(task, group, choice, choice == gold, conf, nll, brier,
                        probs.count(conf) > 1)


def _ratio(num: float, den: int) -> float | None:
    return num / den if den else None


def _mean_all(values: Sequence[float | None]) -> float | None:
    if not values or any(x is None for x in values):
        return None
    return math.fsum(values) / len(values)


def _bin(confidence: float) -> int:
    return bisect.bisect_right(_ECE_EDGES, confidence)


@dataclass(slots=True)
class _Totals:
    n: int = 0
    valid: int = 0
    correct: int = 0
    zero_gold: int = 0
    nll: float = 0.0
    brier: float = 0.0
    confident: int = 0
    confident_wrong: int = 0
    bin_gaps: dict[int, float] = field(default_factory=dict)

    def observe(self, o: _Observation, threshold: float) -> None:
        self.n += 1
        if o.error is not None:
            return
        self.valid += 1
        self.correct += o.correct
        if math.isinf(o.nll):
            self.zero_gold += 1
        else:
            self.nll += o.nll
        self.brier += o.brier
        if o.confidence >= threshold:
            self.confident += 1
            self.confident_wrong += not o.correct
        b = _bin(o.confidence)
        self.bin_gaps[b] = self.bin_gaps.get(b, 0.) + (o.correct - o.confidence)

    def merge(self, other: _Totals, times: int = 1) -> None:
        self.n += times * other.n
        self.valid += times * other.valid
        self.correct += times * other.correct
        self.zero_gold += times * other.zero_gold
        self.nll += times * other.nll
        self.brier += times * other.brier
        self.confident += times * other.confident
        self.confident_wrong += times * other.confident_wrong
        for b, gap in other.bin_gaps.items():
            self.bin_gaps[b] = self.bin_gaps.get(b, 0.) + times * gap

    def metrics(self) -> Row:
        complete = self.n == self.valid
        return {
            "accuracy": self.correct / self.n,
            "nll": self.nll / self.n if complete and self.zero_gold == 0 else None,
            "brier": self.brier / self.n if complete else None,
            "ece15": math.fsum(map(abs, self.bin_gaps.values())) / self.n if complete else None,
            "confident_coverage": self.confident / self.n,
            "confident_error_joint": self.confident_wrong / self.n,
            "confident_error_conditional": _ratio(self.confident_wrong, self.confident),
        }


def _risk_coverage(obs: Sequence[_Observation]) -> tuple[list[Row], float, float | None]:
    ties: dict[float, list[int]] = {}
    for o in obs:
        if o.error is None:
            block = ties.setdefault(o.confidence, [0, 0])
            block[0] += 1
            block[1] += not o.correct
    count = wrong = 0
    points = [{"threshold": None, "accepted": 0, "wrong": 0, "coverage": 0., "risk": None}]
    rectangles = []
    for threshold, (size, errors) in sorted(ties.items(), reverse=True):
        count += size
        wrong += errors
        risk = wrong / count
        points.append({"threshold": threshold, "accepted": count, "wrong": wrong,
                       "coverage": count / len(obs), "risk": risk})
        rectangles.append(size / len(obs) * risk)
    area = math.fsum(rectangles)
    return points, area, area if count == len(obs) else None


def _summary(obs: Sequence[_Observation], threshold: float) -> Row:
    t = _Totals()
    for o in obs:
        t.observe(o, threshold)
    counts, successes, conf_sum = [0] * ECE_BINS, [0] * ECE_BINS, [0.] * ECE_BINS
    for o in obs:
        if o.error is None:
            b = _bin(o.confidence)
            counts[b] += 1
            successes[b] += o.correct
            conf_sum[b] += o.confidence
    bins = [
        {"lower": i / ECE_BINS, "upper": (i + 1) / ECE_BINS, "n": k,
         "accuracy": _ratio(successes[i], k), "mean_confidence": _ratio(conf_sum[i], k)}
        for i, k in enumerate(counts)
    ]
    curve, partial, aurc = _risk_coverage(obs)
    return {
        **t.metrics(), "n": t.n, "valid": t.valid, "invalid": t.n - t.valid,
        "invalid_reasons": dict(Counter(o.error for o in obs if o.error is not None)),
        "correct": t.correct, "zero_gold_probability": t.zero_gold,
        "nll_status": ("invalid_distribution" if t.valid != t.n else
                       "infinite" if t.zero_gold else "finite"),
        "argmax_ties": sum(o.tied for o in obs), "confident": t.confident,
        "confident_wrong": t.confident_wrong, "ece_bins_valid_only": bins,
        "mean_top_confidence_valid_only": _ratio(math.fsum(conf_sum), t.valid),
        "risk_coverage": curve, "aurc_observed_area": partial, "aurc": aurc,
    }


def _by_task(obs: Sequence[_Observation]) -> dict[str, list[_Observation]]:
    tasks = defaultdict(list)
    for o in obs:
        tasks[o.task].append(o)
    return dict(sorted(tasks.items()))


def _summarize(obs: Sequence[_Observation], threshold: float) -> Row:
    tasks = {k: _summary(v, threshold) for k, v in _by_task(obs).items()}
    keys = (*BOOT_METRICS, "aurc")
    macro = {m: _mean_all([t[m] for t in tasks.values()]) for m in keys}
    groups = defaultdict(list)
    for o in obs:
        groups[o.group].append(o.correct)
    return {
        "micro": _summary(obs, threshold), "macro": macro, "per_task": tasks,
        "macro_task_count": len(tasks), "group_count": len(groups),
        "equal_group_accuracy": _mean_all([sum(g) / len(g) for g in groups.values()]),
    }


def _transitions(before: Sequence[_Observation], after: Sequence[_Observation]) -> Row:
    pair = [(a, b) for a, b in zip(before, after) if a.error is None and b.error is None]
    n = len(pair)
    before_correct = sum(a.correct for a, _ in pair)
    corrections = sum(not a.correct and b.correct for a, b in pair)
    corruptions = sum(a.correct and not b.correct for a, b in pair)
    persistent = [(a, b) for a, b in pair if not a.correct and not b.correct]
    delta_conf = _ratio(math.fsum(b.confidence - a.confidence for a, b in pair), n)
    delta_acc = _ratio(corrections - corruptions, n)
    return {
        "all": len(before), "valid_pairs": n, "invalid_pairs": len(before) - n,
        "before_correct": before_correct, "before_wrong": n - before_correct,
        "corrections": corrections, "corruptions": corruptions,
        "correction_joint": _ratio(corrections, n),
        "correction_given_wrong": _ratio(corrections, n - before_correct),
        "corruption_joint": _ratio(corruptions, n),
        "corruption_given_correct": _ratio(corruptions, before_correct),
        "changed_choice": sum(a.choice != b.choice for a, b in pair),
        "delta_accuracy_valid_pairs": delta_acc, "delta_top_confidence": delta_conf,
        "confidence_inflation_gap": delta_conf - delta_acc if n else None,
        "persistent_wrong": len(persistent),
        "persistent_wrong_confidence_rises": sum(b.confidence > a.confidence for a, b in persistent),
        "persistent_wrong_delta_top_confidence": _ratio(
            math.fsum(b.confidence - a.confidence for a, b in persistent), len(persistent)),
    }


def _passes(observations: dict[str, list[_Observation]], order: Sequence[str]) -> Row:
    if (isinstance(order, str) or len(order) < 2 or len(set(order)) != len(order)
            or not set(order) <= observations.keys()):
        raise EvaluationError("pass series must name at least two distinct, evaluated arms in order")
    obs = [observations[k] for k in order]
    paths = [p for p in zip(*obs) if all(o.error is None for o in p)]
    once_correct = sum(any(o.correct for o in p[:-1]) for p in paths)
    overthought = sum(any(o.correct for o in p[:-1]) and not p[-1].correct for p in paths)
    final_wrong = sum(not p[-1].correct for p in paths)
    return {
        "order": list(order),
        "adjacent": [{"from": a, "to": b, **_transitions(observations[a], observations[b])}
                     for a, b in zip(order, order[1:])],
        "first_to_last": _transitions(obs[0], obs[-1]),
        "all": len(obs[0]), "valid_paths": len(paths), "invalid_paths": len(obs[0]) - len(paths),
        "previously_correct": once_correct, "final_wrong": final_wrong, "overthought": overthought,
        "overthinking_joint": _ratio(overthought, len(paths)),
        "overthinking_given_previously_correct": _ratio(overthought, once_correct),
        "overthinking_given_final_wrong": _ratio(overthought, final_wrong),
    }


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else b - a


def _percentile(ordered: Sequence[float], q: float) -> float:
    x = (len(ordered) - 1) * q
    lo = int(x)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (x - lo) * (ordered[hi] - ordered[lo])


def _bootstrap(base: Sequence[_Observation], cand: Sequence[_Observation], threshold: float,
               base_summary: Row, cand_summary: Row, reps: int, seed: int) -> Row:
    groups = defaultdict(dict)
    for a, b in zip(base, cand):
        if a.group not in groups[a.task]:
            groups[a.task][a.group] = (_Totals(), _Totals())
        x, y = groups[a.task][a.group]
        x.observe(a, threshold)
        y.observe(b, threshold)
    strata = [[by_group[g] for g in sorted(by_group)] for _, by_group in sorted(groups.items())]
    sizes = {k: len(v) for k, v in sorted(groups.items())}
    enough = all(g >= 2 for g in sizes.values())
    draws = {(scope, m): [] for scope in ("micro", "macro") for m in BOOT_METRICS}
    rng = random.Random(seed)
    for _ in range(reps if enough else 0):
        micro = (_Totals(), _Totals())
        tasks = []
        for stratum in strata:
            totals = (_Totals(), _Totals())
            # One draw per source group, never per row/pass/arm.
            for idx, count in Counter(rng.randrange(len(stratum)) for _ in stratum).items():
                x, y = stratum[idx]
                totals[0].merge(x, count)
                totals[1].merge(y, count)
            micro[0].merge(totals[0])
            micro[1].merge(totals[1])
            tasks.append((totals[0].metrics(), totals[1].metrics()))
        a, b = micro[0].metrics(), micro[1].metrics()
        for m in BOOT_METRICS:
            pooled = _delta(a[m], b[m])
            equal_task = _mean_all([_delta(x[m], y[m]) for x, y in tasks])
            if pooled is not None:
                draws["micro", m].append(pooled)
            if equal_task is not None:
                draws["macro", m].append(equal_task)
    result = {}
    for (scope, m), values in draws.items():
        estimate = _delta(base_summary[scope][m], cand_summary[scope][m])
        status = ("not_requested" if not reps else "insufficient_independent_groups" if not enough
                  else "undefined_estimate" if estimate is None else
                  "undefined_replicates" if len(values) != reps else "ok")
        values.sort()
        result.setdefault(scope, {})[m] = {
            "delta": estimate, "ci95": [_percentile(values, q) for q in (.025, .975)]
            if status == "ok" else None, "status": status, "defined_replicates": len(values),
        }
    return {
        "direction": "candidate_minus_baseline", "seed": seed, "requested_replicates": reps,
        "groups_per_task": sizes, **result,
    }


def _validate_settings(threshold: float, bootstrap_reps: int, bootstrap_seed: int) -> None:
    if type(threshold) not in (float, int) or not 0 <= threshold <= 1:
        raise EvaluationError("confidence threshold must be finite and in [0,1]")
    if type(bootstrap_reps) is not int or bootstrap_reps < 0 or type(bootstrap_seed) is not int:
        raise EvaluationError("bootstrap requires a nonnegative integer draw count and integer seed")


def evaluate(inputs: Iterable[Row], labels: Iterable[Row], arms: Mapping[str, Iterable[Row]],
             *, baseline: str, pass_series: Sequence[str] | None = None,
             threshold: float = DEFAULT_THRESHOLD, bootstrap_reps: int = 2000,
             bootstrap_seed: int = 4117) -> Row:
    """Score one ID-aligned split. In-memory API assumes the caller has label access."""
    _validate_settings(threshold, bootstrap_reps, bootstrap_seed)
    if not isinstance(arms, Mapping) or not all(map(_text, arms)) or baseline not in arms:
        raise EvaluationError("baseline must be one of the named arms")
    indexed = _index_inputs(inputs)
    label_map = _bind(labels, indexed, ({"id", "input_sha256", "gold_index"},), "labels")
    ids = sorted(indexed)
    for key in ids:
        gold = label_map[key]["gold_index"]
        if type(gold) is not int or not 0 <= gold < len(indexed[key][0]["choices"]):
            raise EvaluationError(f"invalid gold_index: {key}")
    observations = {}
    summaries = {}
    for arm, preds in arms.items():
        bound = _bind(preds, indexed, ({"id", "input_sha256", "probs"},
                                       {"id", "input_sha256", "error"}), arm)
        obs = [_observe(indexed[k][0], label_map[k]["gold_index"], bound[k]) for k in ids]
        observations[arm] = obs
        summaries[arm] = _summarize(obs, threshold)
    passes = None if pass_series is None else _passes(observations, pass_series)
    comparisons = {
        arm: _bootstrap(observations[baseline], obs, threshold, summaries[baseline],
                        summaries[arm], bootstrap_reps, bootstrap_seed)
        for arm, obs in observations.items() if arm != baseline
    }
    return {"schema": SCHEMA, "threshold": threshold, "baseline": baseline, "arms": summaries,
            "all_distributions_valid": all(a["micro"]["invalid"] == 0 for a in summaries.values()),
            "paired_deltas": comparisons, "passes": passes}


def _strict_json(text: str) -> Any:
    def object_hook(pairs):
        out = dict(pairs)
        if len(out) != len(pairs):
            raise EvaluationError("duplicate JSON object key")
        return out

    def invalid_constant(_: str):
        raise EvaluationError("nonstandard JSON constant; record an explicit runner error instead")

    return json.loads(text, object_pairs_hook=object_hook, parse_constant=invalid_constant)


def read_jsonl(path: str | Path) -> Iterable[Row]:
    with open(path, encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            try:
                row = _strict_json(line)
            except ValueError as exc:
                raise EvaluationError(f"invalid JSONL at line {n}: {exc}") from exc
            if not isinstance(row, dict):
                raise EvaluationError(f"expected a JSON object at line {n}")
            yield row


def _check_finalists(path: str | None, pinned: str | None, manifest: Row,
                     names: set[str], baseline: str, passes: list[str] | None) -> None:
    if not path or not _sha256_text(pinned):
        raise EvaluationError("final scoring requires the independently pinned finalists receipt")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pinned:
        raise EvaluationError("finalists receipt digest differs")
    receipt = _strict_json(raw.decode("utf-8"))
    if (not isinstance(receipt, dict)
            or set(receipt) != {"schema", "manifest_sha256", "arms", "baseline", "passes", "primary"}
            or receipt["schema"] != "joey-finalists-v1" or receipt["manifest_sha256"] != manifest["sha256"]
            or receipt["baseline"] != baseline or baseline not in names or receipt["passes"] != passes
            or not isinstance(receipt["arms"], dict) or set(receipt["arms"]) != names
            or not all(map(_sha256_text, receipt["arms"].values()))
            or passes is not None and (len(passes) < 2 or len(set(passes)) != len(passes)
                                       or not set(passes) <= names)
            or not isinstance(receipt["primary"], str)
            or receipt["primary"] not in {f"{s}.{m}" for s in ("micro", "macro") for m in BOOT_METRICS}):
        raise EvaluationError("finalists receipt disagrees with the evaluation plan")


def _main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="prepare a label-free group split")
    score = sub.add_parser("score", help="score only the selected IDs after label release")
    for p in (freeze, score):
        p.add_argument("--inputs", required=True, help="complete label-free input JSONL")
        p.add_argument("--config", required=True, help="frozen evaluation protocol JSON")
    score.add_argument("--manifest", required=True)
    score.add_argument("--manifest-sha256", required=True, help="independently pinned digest")
    score.add_argument("--split", required=True)
    score.add_argument("--labels", required=True, help="ONLY labels for the selected split")
    score.add_argument("--arm", action="append", required=True, metavar="NAME=FILE")
    score.add_argument("--baseline", required=True)
    score.add_argument("--passes", nargs="+", help="ordered pass arm names for one method")
    score.add_argument("--finalists", help="required for final: frozen arm/config receipt JSON")
    score.add_argument("--finalists-sha256", help="independently pinned SHA-256 of receipt file bytes")
    args = cli.parse_args()
    try:
        cfg = _strict_json(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(cfg, dict) or set(cfg) != {
            "schema", "split_seed", "group_weights", "threshold", "bootstrap_reps", "bootstrap_seed"
        } or cfg["schema"] != SCHEMA:
            raise EvaluationError("unexpected evaluation config schema")
        _validate_settings(cfg["threshold"], cfg["bootstrap_reps"], cfg["bootstrap_seed"])
        cfg_hash = _hash(cfg)
        inputs = read_jsonl(args.inputs)
        if args.command == "freeze":
            out = freeze_split(inputs, seed=cfg["split_seed"], weights=cfg["group_weights"],
                               config_sha256=cfg_hash)
        else:
            manifest = _strict_json(Path(args.manifest).read_text(encoding="utf-8"))
            rows = select_frozen_split(inputs, manifest, args.manifest_sha256, args.split)
            if ((manifest["seed"], manifest["weights"]) != (cfg["split_seed"], cfg["group_weights"])
                    or manifest["config_sha256"] != cfg_hash):
                raise EvaluationError("config disagrees with frozen protocol")
            arms = {}
            for item in args.arm:
                name, sep, path = item.partition("=")
                if not sep or not _text(name) or not path or name in arms:
                    raise EvaluationError("expected a unique NAME=FILE for each arm")
                arms[name] = read_jsonl(path)
            if args.split == "final" or args.finalists or args.finalists_sha256:
                _check_finalists(args.finalists, args.finalists_sha256, manifest, set(arms),
                                 args.baseline, args.passes)
            out = evaluate(rows, read_jsonl(args.labels), arms, baseline=args.baseline,
                           pass_series=args.passes, threshold=cfg["threshold"],
                           bootstrap_reps=cfg["bootstrap_reps"], bootstrap_seed=cfg["bootstrap_seed"])
            out.update(split=args.split, manifest_sha256=args.manifest_sha256, config_sha256=cfg_hash,
                       finalists_sha256=args.finalists_sha256)
        print(json.dumps(out, allow_nan=False, sort_keys=True))
    except (ValueError, OSError) as exc:
        cli.exit(2, f"evaluation: {exc}\n")


if __name__ == "__main__":
    _main()
