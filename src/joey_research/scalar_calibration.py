"""Posthoc diagnostic only. Fit on verified calibration labels; transform one blind split.

Pinned(data, sha256) uses an independently recorded SHA-256 of exact bytes. The
manifest pin instead uses evaluation.freeze_split's canonical manifest digest.
The frozen roster is JSON: {schema:"joey-temperature-roster/v1", stock:arm_name,
arms:{arm_name:method_sha256}, manifest_sha256, protocol_sha256:PROTOCOL_SHA256}.
Each ArmStream carries the externally verified run binding for its original JSONL.
Neither digest checks nor role names establish authorization or freeze chronology.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Context, Decimal, localcontext
import hashlib
import json
import math

from . import evaluation as ev


PROTOCOL_SHA256 = "e4972603ae4b0fec9fa40423c61d476426a47752fa129fde72f16f009fdba661"
GRID = (.5, 2 / 3, .8, 1., 1.25, 1.5, 2., 3.)


def _check(ok, msg):
    if not ok:
        raise ev.EvaluationError(msg)


@dataclass(frozen=True)
class Pinned:
    data: bytes
    sha256: str

    def verified(self) -> bytes:
        _check(type(self.data) is bytes and hashlib.sha256(self.data).hexdigest() == self.sha256,
               "artifact differs from independent byte pin")
        return self.data


@dataclass(frozen=True)
class ArmStream:
    """Supply fields from independently verified worker receipts, not prediction filenames."""
    split: str
    method_sha256: str
    inputs_sha256: str
    predictions: Pinned


def _jsonl(blob: Pinned):
    def number(text):
        value = float(text)
        _check(math.isfinite(value) and (value != 0 or Decimal(text).is_zero()),
               "source number is not representable as finite binary64")
        return value

    def unique(pairs):
        obj = dict(pairs)
        _check(len(obj) == len(pairs), "duplicate JSON key")
        return obj

    return [json.loads(line.decode("utf-8"), parse_float=number, parse_constant=number, object_pairs_hook=unique)
            for line in blob.verified().splitlines()]


def _manifest_slice(manifest, pin, split):
    _check(isinstance(manifest, dict) and manifest.get("schema") == ev.SCHEMA
           and manifest.get("sha256") == pin == ev._hash({k: v for k, v in manifest.items() if k != "sha256"})
           and isinstance(manifest.get("entries"), list), "invalid frozen split manifest")
    ids, groups, selected = set(), {}, {}
    for entry in manifest["entries"]:
        _check(isinstance(entry, dict) and set(entry) == {"id", "input_sha256", "group_id", "task", "split"}
               and all(ev._text(entry[k]) for k in ("id", "group_id", "task", "split"))
               and ev._sha256_text(entry["input_sha256"]), "invalid frozen state entry")
        key, group, task, part = (entry[k] for k in ("id", "group_id", "task", "split"))
        _check(key not in ids and groups.setdefault(group, (task, part)) == (task, part),
               "duplicate ID or group crossed split/task boundaries")
        ids.add(key)
        if part == split:
            selected[key] = (group, task, entry["input_sha256"])
    _check(selected and any(p == "calibration" for _, p in groups.values()), "missing required split")
    return selected


def _prepare(manifest, manifest_sha256, roster, split, inputs, arms):
    selected = _manifest_slice(manifest, manifest_sha256, split)
    freeze = ev._strict_json(roster.verified().decode())
    _check(isinstance(freeze, dict) and set(freeze) == {
        "schema", "stock", "arms", "manifest_sha256", "protocol_sha256"
    } and freeze["schema"] == "joey-temperature-roster/v1" and freeze["protocol_sha256"] == PROTOCOL_SHA256
           and freeze["manifest_sha256"] == manifest_sha256 and isinstance(freeze["arms"], dict)
           and all(ev._text(a) and ev._sha256_text(h) for a, h in freeze["arms"].items())
           and ev._text(freeze["stock"]) and freeze["stock"] in freeze["arms"]
           and set(arms) == set(freeze["arms"]), "missing stock, changed roster, manifest or protocol")
    rows = _jsonl(inputs)
    indexed = ev._index_inputs(rows)
    _check({k: (r["group_id"], r["task"], h) for k, (r, h) in indexed.items()} == selected,
           "blind inputs do not exactly match the requested frozen partition")
    predictions = {}
    for name, stream in arms.items():
        _check(isinstance(stream, ArmStream) and stream.method_sha256 == freeze["arms"][name]
               and stream.inputs_sha256 == inputs.sha256 and stream.split == split, "arm receipt mismatch")
        bound = ev._bind(_jsonl(stream.predictions), indexed, ({"id", "input_sha256", "probs"},), name)
        for k, p in bound.items():
            prob = p["probs"]
            _check(isinstance(prob, list) and len(prob) == len(indexed[k][0]["choices"])
                   and all(type(v) in (float, int) and 0 <= v <= 1 for v in prob)
                   and abs(math.fsum(prob) - 1) <= ev.PROB_SUM_ATOL, "invalid source distribution")
        predictions[name] = bound
    return indexed, predictions


def _relative_logs(prob):
    # Near the maximum use log1p to retain the separation between adjacent floats.
    m = max(prob)
    return [(-math.inf if not p else math.log1p((p - m) / m) if p > m / 2 else math.log(p / m))
            for p in prob]


def _nll(logs, y, t):
    maximum = logs.index(0.)
    rest = math.fsum(math.exp(x / t) for j, x in enumerate(logs) if j != maximum)
    return math.log1p(rest) - logs[y] / t


def _winner(losses):
    # max(T, 1/T) orders by |log T| without spurious asymmetry for reciprocal grid pairs.
    return min(GRID, key=lambda t: (losses[GRID.index(t)], max(t, 1 / t), t))


def fit_temperatures(*, manifest: ev.Row, manifest_sha256: str, roster: Pinned, split: str,
                     inputs: Pinned, labels: Pinned, arms: Mapping[str, ArmStream]) -> bytes:
    """Return immutable JSON fit-receipt bytes to pin outside the fitter before application."""
    _check(split == "calibration", "fitting requires the calibration partition")
    indexed, pred = _prepare(manifest, manifest_sha256, roster, split, inputs, arms)
    gold = ev._bind(_jsonl(labels), indexed, ({"id", "input_sha256", "gold_index"},), "calibration labels")
    for k, g in gold.items():
        _check(type(g["gold_index"]) is int and 0 <= g["gold_index"] < len(indexed[k][0]["choices"]),
               "invalid calibration gold index")
    fits = {}
    for arm, bound in pred.items():
        tasks = defaultdict(list)
        zero_gold = 0
        for k, (r, _) in indexed.items():
            probs, y = bound[k]["probs"], gold[k]["gold_index"]
            tasks[r["task"]].append((_relative_logs(probs), y))
            zero_gold += probs[y] == 0
        losses = [None] * len(GRID) if zero_gold else [
            math.fsum(math.fsum(_nll(logs, y, t) for logs, y in group) / len(group)
                      for group in tasks.values()) / len(tasks) for t in GRID]
        fits[arm] = {"status": "all_objectives_infinite" if zero_gold else "fitted",
                     "temperature": None if zero_gold else _winner(losses), "zero_gold_count": zero_gold,
                     "macro_nll": losses, "prediction_sha256": arms[arm].predictions.sha256}
    return ev._canonical({"schema": "joey-temperature-fit/v1", "scope": "posthoc_only", "split": split,
                          "protocol_sha256": PROTOCOL_SHA256, "manifest_sha256": manifest_sha256,
                          "roster_sha256": roster.sha256, "inputs_sha256": inputs.sha256,
                          "labels_sha256": labels.sha256, "grid": list(GRID), "arms": fits})


def _preserves_order(prob, scaled):
    levels = {}
    for p, q in zip(prob, scaled):
        if (p == 0) != (q == 0) or levels.setdefault(p, q) != q:
            return False
    values = [levels[p] for p in sorted(levels)]
    return all(a < b for a, b in zip(values, values[1:]))


def _scaled_literals(prob, t):
    if t == 1 and math.fsum(prob) == 1:
        return [str(p) for p in prob], False
    logs = _relative_logs(prob)
    weights = [math.exp(x / t) for x in logs]
    total = math.fsum(weights)
    scaled = [w / total for w in weights]
    if _preserves_order(prob, scaled):
        return [repr(q) for q in scaled], False
    # Binary64 can turn positive values into zeros or merge distinct ranks. Serialize
    # high-precision JSON numbers in that case; never floor, jitter, or fabricate a tie.
    with localcontext(Context(prec=80)):
        values = list(map(Decimal, prob))
        largest = max(values)
        dt = Decimal(t)
        weights = [((p / largest).ln() / dt).exp() if p else Decimal(0) for p in values]
        total = sum(weights)
        out = [w / total for w in weights]
        _check(_preserves_order(prob, out), "temperature transformation lost the input order")
        return [str(q) for q in out], True


def apply_temperatures(fit: Pinned, *, manifest: ev.Row, manifest_sha256: str, roster: Pinned,
                       split: str, inputs: Pinned, arms: Mapping[str, ArmStream]) -> ev.Row:
    """No labels accepted. `raw_jsonl` is unchanged; `calibrated_jsonl` is distinct or None.

    For `requires_decimal_ids`, parse transformed JSON numbers with parse_float=Decimal;
    a binary64-only scorer cannot represent the full calibrated ranking/support there.
    """
    _check(ev._text(split) and split != "calibration", "application requires a disjoint non-calibration split")
    obj = ev._strict_json(fit.verified().decode())
    _check(isinstance(obj, dict) and obj.get("schema") == "joey-temperature-fit/v1"
           and obj.get("scope") == "posthoc_only" and obj.get("split") == "calibration"
           and obj.get("protocol_sha256") == PROTOCOL_SHA256 and obj.get("manifest_sha256") == manifest_sha256
           and obj.get("roster_sha256") == roster.sha256 and obj.get("grid") == list(GRID)
           and obj.get("inputs_sha256") != inputs.sha256 and isinstance(obj.get("arms"), dict)
           and obj["arms"].keys() == arms.keys(), "fit receipt does not bind this application")
    _, predictions = _prepare(manifest, manifest_sha256, roster, split, inputs, arms)
    out = {}
    for arm, bound in predictions.items():
        f = obj["arms"][arm]
        t, losses = f["temperature"], f["macro_nll"]
        _check(isinstance(losses, list) and len(losses) == len(GRID), "missing objective grid")
        if f["status"] == "all_objectives_infinite":
            _check(t is None and type(f["zero_gold_count"]) is int and f["zero_gold_count"] > 0
                   and all(x is None for x in losses), "inconsistent unsupported fit")
            calibrated, exceptional = None, []
        else:
            _check(f["status"] == "fitted" and f["zero_gold_count"] == 0
                   and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in losses)
                   and type(t) in (int, float) and t == _winner(losses), "inconsistent fitted temperature")
            records, exceptional = [], []
            for k, p in bound.items():
                literals, precise = _scaled_literals(p["probs"], t)
                if precise:
                    exceptional.append(k)
                prefix = json.dumps({"id": k, "input_sha256": p["input_sha256"]}, separators=(",", ":"))[:-1]
                records.append(f'{prefix},"probs":[{",".join(literals)}]}}\n'.encode("ascii"))
            calibrated = b"".join(records)
        out[arm] = {"raw_jsonl": arms[arm].predictions.data, "calibrated_jsonl": calibrated,
                    "status": f["status"], "temperature": t, "requires_decimal_ids": exceptional}
    return {"scope": "posthoc_only", "split": split, "inputs_sha256": inputs.sha256,
            "fit_sha256": fit.sha256, "arms": out}
