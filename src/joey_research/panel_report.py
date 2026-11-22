"""Fixed-panel, same-split comparisons. No model execution, selection, or label acquisition."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math

from . import evaluation as ev
from .loop import LoopCounts
from .worker import ensemble_groups, method_sha256


THRESHOLDS = (.8, .9, .95, .99)
REPLICATES = 2000
SEED = 4117
_ARTIFACTS = ("packet.json", "predictions.jsonl", "trace.jsonl", "complete.json")
_CALLS = ("stock_calls", "prelude_calls", "block_calls", "coda_calls")


@dataclass(frozen=True)
class ArmRun:
    """Exact worker bytes and independent host/transport bindings; slot is opaque, not reported."""
    packet: bytes
    predictions: bytes
    trace: bytes
    complete: bytes
    pins: Mapping[str, str]
    slot: str | None
    permit_sha256: str | None = None


@dataclass(frozen=True)
class Pair:
    name: str
    baseline: str
    candidate: str
    kind: str = "contrast"


def _require(condition, message: str) -> None:
    if not condition:
        raise ev.EvaluationError(message)


def _pin(raw: bytes, digest: str, kind: str) -> None:
    _require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == digest, f"{kind}: byte pin differs")


def _object(raw: bytes) -> ev.Row:
    obj = ev._strict_json(raw.decode("utf-8"))
    _require(isinstance(obj, dict), "expected JSON object")
    return obj


def _lines(raw: bytes) -> list[ev.Row]:
    return [_object(line) for line in raw.splitlines()]


def _aligned(records: Sequence[ev.Row], keys: Sequence[tuple[str, str]], kind: str) -> None:
    if len(records) != len(keys):
        raise ev.EvaluationError(f"{kind}: unpaired row count: expected {len(keys)}, got {len(records)}")
    for i, (r, expected) in enumerate(zip(records, keys)):
        if not isinstance(r, Mapping) or (r.get("id"), r.get("input_sha256")) != expected:
            raise ev.EvaluationError(f"{kind}: missing, reordered, or stale ID/digest at position {i}")


def _noninferiority(delta: ev.Row, valid: bool) -> ev.Row:
    acc, brier = (delta["macro"][m] for m in ("accuracy", "brier"))
    out = {"delta_direction": "compact_minus_full_stock", "accuracy_margin": -.02,
           "brier_margin": .01, "macro_accuracy_ci95": acc["ci95"], "macro_brier_ci95": brier["ci95"]}
    if not valid or acc["status"] != "ok" or brier["status"] != "ok":
        return {**out, "status": "not_evaluable", "passed": None,
                "reason": "invalid_distribution" if not valid else
                f"accuracy:{acc['status']};brier:{brier['status']}"}
    passed = acc["ci95"][0] > -.02 and brier["ci95"][1] < .01
    return {**out, "status": "passed" if passed else "not_demonstrated", "passed": passed}


def _confidence(summary: ev.Row) -> ev.Row:
    metrics = ("confident_coverage", "confident_error_joint", "confident_error_conditional")
    def counts(r):
        return {k: r[k] for k in ("n", "valid", "invalid", "confident_wrong", "confident", *metrics)}
    return {"micro": counts(summary["micro"]), "macro": {m: summary["macro"][m] for m in metrics},
            "per_task": {t: counts(r) for t, r in summary["per_task"].items()}}


def _int(x, minimum=0) -> bool:
    return type(x) is int and x >= minimum


def _ms(x) -> bool:
    return type(x) in (int, float) and math.isfinite(x) and x >= 0


def _timing(values: list[float]) -> ev.Row:
    times = sorted(values)
    total = math.fsum(times)
    return {"total": total, "mean": total / len(times),
            **{p: times[math.ceil(len(times) * int(p) / 100) - 1] for p in ("50", "95", "99")}}


def _parse_run(run: ArmRun, rows: list[ev.Row], keys, split: str, inputs_sha256: str):
    _require(set(run.pins) == set(_ARTIFACTS), "incomplete independent arm receipt")
    for name, raw in zip(_ARTIFACTS, (run.packet, run.predictions, run.trace, run.complete)):
        _pin(raw, run.pins[name], name)
    pkt, met = _object(run.packet), _object(run.complete)
    _require(run.slot is None or ev._text(run.slot), "invalid slot attestation")
    _require(type(pkt.get("schema")) is int and type(met.get("schema")) is int
             and pkt["schema"] == met["schema"] == 1 and pkt.get("split") == split
             and met.get("split", "discovery") == split and pkt.get("inputs_sha256") == inputs_sha256
             and met.get("packet_sha256") == run.pins["packet.json"], "mixed split, shard or packet")
    envelope = {"schema", "split", "inputs_sha256", "deadline_utc"}
    if split == "selection":
        envelope |= {"method_sha256", "permit_sha256"}
        _require(ev._sha256_text(run.permit_sha256)
                 and run.permit_sha256 == pkt.get("permit_sha256") == met.get("permit_sha256")
                 and met.get("inputs_sha256") == inputs_sha256
                 and met.get("result_sha256") == {k: run.pins[k] for k in ("trace.jsonl", "predictions.jsonl")},
                 "unbound selection permit, input shard or result streams")
    else:
        _require(run.permit_sha256 is None and not {"method_sha256", "permit_sha256"} & met.keys(),
                 "selection evidence in discovery")
    settings = {k: v for k, v in pkt.items() if k not in envelope}
    method_pin = method_sha256(settings)
    if split == "selection":
        _require(method_pin == pkt.get("method_sha256") == met.get("method_sha256"), "stale method digest")
    preds, traces = _lines(run.predictions), _lines(run.trace)
    _aligned(preds, keys, "predictions")
    _require(len(traces) == len(rows), "unpaired trace count")
    _require(_int(met.get("decisions"), 1) and met["decisions"] == len(rows)
             and type(met.get("batch_size")) is int and met["batch_size"] == 1
             and met.get("hardware") == "A100" and _int(met.get("peak_allocated_bytes"), 1),
             "unusable worker completion counts, hardware or peak allocation")
    compact = pkt.get("compact_manifest_sha256")
    _require(compact == met.get("compact_manifest_sha256"), "compact manifest differs")
    depth = met.get("physical_layers") if compact else 24
    lo, hi = settings["window"]
    _require(_int(depth, 1) and hi <= depth <= 24 and (not compact or depth < 24), "impossible physical window")
    cfg = settings.get("order_ensemble")
    selected = ensemble_groups(((r["task"], r["group_id"]) for r in rows), cfg, split=split) if cfg else set()
    total = Counter()
    wall, cuda, per_decision = [], [], []
    loop = settings["loop"]
    diagnostic = loop.get("readout", "final") == "diagnostic"
    for idx, (row, pred, t) in enumerate(zip(rows, preds, traces)):
        _require(t.get("id") == row["id"], f"unpaired/reordered trace at position {idx}")
        _require("input_sha256" not in t or t["input_sha256"] == keys[idx][1], "stale trace input digest")
        _require((split != "selection" and "packet_sha256" not in t)
                 or t.get("packet_sha256") == run.pins["packet.json"], "stale trace packet digest")
        cc = t.get("callback_counts")
        _require(isinstance(cc, dict) and set(cc) == set(_CALLS) and all(map(_int, cc.values())),
                 "missing or invalid actual callback counts")
        calls = LoopCounts(**cc)
        forwards = calls.stock_calls + calls.prelude_calls
        expected_calls = 1 + int(row["group_id"] in selected)
        _require(forwards == expected_calls and calls.block_evaluations >= forwards
                 and calls.coda_readouts >= forwards and _int(t.get("tokens"), 1)
                 and _int(t.get("layer_calls"), 1)
                 and t["layer_calls"] == calls.layer_evaluations(lo, hi-lo, depth-hi),
                 "inconsistent forward, token or physical layer counts")
        if cfg:
            _require(type(t.get("ensemble_calls")) is int and t["ensemble_calls"] == expected_calls
                     and type(t.get("forward_calls")) is int and t["forward_calls"] == expected_calls
                     and calls == LoopCounts(0, expected_calls, expected_calls, expected_calls),
                     "ensemble spend is not the prespecified whole-group allocation")
        else:
            _require(not {"ensemble_calls", "forward_calls"} & t.keys(), "unexpected ensemble trace")
            if loop.get("k", 1) == 1:
                _require(calls == LoopCounts(1, 0, 0, 0), "stock trace does not contain one actual stock call")
            elif loop.get("method") == "zero_extra":
                _require(calls == LoopCounts(0, 1, 1, 1), "zero-extra trace did not take the split path")
            else:
                _require(calls.stock_calls == 0 and calls.prelude_calls == 1, "recurrence skipped split path")
                budget = loop["k"]
                policy = settings.get("policy")
                if policy is not None and (policy.get("pool", "final") != "final" or policy.get("stop") is not None):
                    _require(1 <= calls.block_calls <= budget and calls.coda_calls == calls.block_calls,
                             "live policy counters disagree with completed readouts")
                else:
                    _require(calls.block_calls == budget and calls.coda_calls == (budget if diagnostic else 1),
                             "recurrence trace does not account for the declared block/coda passes")
        readouts = t.get("readouts")
        _require(isinstance(readouts, list) and bool(readouts) == diagnostic, "unexpected readout availability")
        if readouts:
            _require(len(readouts) == calls.coda_readouts, "unpaired diagnostic coda/readout count")
            for j, rd in enumerate(readouts, 1):
                _require(isinstance(rd, dict) and set(rd) == {"kind", "block_evaluations", "probs"}
                         and type(rd["block_evaluations"]) is int and rd["block_evaluations"] == j
                         and ev._text(rd["kind"]), "reordered or ill-formed diagnostic passes")
            _require(readouts[-1]["block_evaluations"] == calls.block_evaluations
                     and readouts[-1]["probs"] == pred.get("probs"), "diagnostic endpoint differs from prediction")
        _require(_ms(t.get("wall_ms")) and _ms(t.get("cuda_event_ms")), "missing/invalid measured interval")
        wall.append(t["wall_ms"])
        cuda.append(t["cuda_event_ms"])
        per_decision.append({"id": row["id"], "cuda_event_ms": cuda[-1], "wall_ms": wall[-1],
                             "block_evaluations": calls.block_evaluations, "coda_readouts": calls.coda_readouts,
                             "layer_calls": t["layer_calls"]})
        total.update({"forward_calls": forwards, "block_evaluations": calls.block_evaluations,
                      "coda_readouts": calls.coda_readouts, "layer_calls": t["layer_calls"], "tokens": t["tokens"]})
    if cfg:
        spend = {"groups": len({r["group_id"] for r in rows}), "extra_groups": len(selected),
                 "forward_calls": total["forward_calls"], "layer_calls": total["layer_calls"],
                 "coda_calls": total["coda_readouts"]}
        _require(ev._canonical(spend) == ev._canonical(met.get("compute_spend")), "ensemble spend total differs")
    else:
        _require("compute_spend" not in met, "unexpected ensemble summary")
    wall_stats = _timing(wall)
    _require(met.get("warm_latency_ms") == {p: wall_stats[p] for p in ("50", "95", "99")},
             "wall percentiles disagree with trace")
    compute = {"n": len(rows), "totals": dict(total), "wall_ms": wall_stats, "cuda_ms": _timing(cuda),
               "peak_allocated_bytes": met["peak_allocated_bytes"], "physical_layers": depth,
               "whole_group_spend": met.get("compute_spend"), "per_decision": per_decision}
    return preds, traces, settings, compute


def _check_pair(pair: Pair, a: ev.Row, b: ev.Row) -> None:
    def k(s):
        return s["loop"].get("k", 1)
    if pair.kind == "matched_path":
        _require(k(a) > 1 and a["loop"].get("method") == "zero_extra" and k(b) > 1
                 and b["loop"].get("method") != "zero_extra" and "order_ensemble" not in a
                 and "order_ensemble" not in b and all(a.get(f) == b.get(f) for f in (
                     "window", "reverse", "compact_recipe", "compact_manifest_sha256")),
                 "matched_path needs matching zero-extra then recurrence on the same window/model/order")
    elif pair.kind == "order_ensemble":
        _require(("order_ensemble" in a) != ("order_ensemble" in b), "pair needs exactly one order ensemble")
    elif pair.kind == "compact_vs_stock":
        _require(k(a) == 1 and not a["reverse"] and not {"compact_manifest_sha256", "order_ensemble", "policy"} & a.keys()
                 and "compact_manifest_sha256" in b, "noninferiority requires full stock then verified compact")
    else:
        _require(pair.kind == "contrast", "unknown pair kind")


def _cost_pair(a: ev.Row, b: ev.Row, left: ArmRun, right: ArmRun) -> ev.Row:
    base, candidate = a["cuda_ms"]["mean"], b["cuda_ms"]["mean"]
    ratio = candidate / base if base else None
    same = None if left.slot is None or right.slot is None else left.slot == right.slot
    return {"baseline_mean_cuda_ms": base, "candidate_mean_cuda_ms": candidate,
            "candidate_over_baseline": ratio, "same_verified_A100_slot": same,
            "within_10pct_same_slot": None if same is not True or ratio is None else .90 <= ratio <= 1.10,
            "interpretation": "Observed mean CUDA event time on the same A100 slot; not equal hardware efficiency."}


def panel_report(inputs: bytes, labels: bytes, arms: Mapping[str, ArmRun], *, split: str,
                 inputs_sha256: str, labels_sha256: str, pairs: Sequence[Pair],
                 diagnostics: Mapping[str, Sequence[tuple[int, str]]] | None = None) -> ev.Row:
    """Verify one preauthorized split and score only the supplied contrasts, in the supplied direction."""
    _require(split in ("selection", "discovery"), "not a verified worker discovery/selection split")
    _pin(inputs, inputs_sha256, "blind shard")
    _pin(labels, labels_sha256, "label shard")
    rows, gold = _lines(inputs), _lines(labels)
    ix = ev._index_inputs(rows)
    keys = [(rid, digest) for rid, (_, digest) in ix.items()]
    _aligned(gold, keys, "labels")
    pairs = list(pairs)
    _require(arms and all(map(ev._text, arms)) and pairs and all(isinstance(p, Pair) and ev._text(p.name)
             and p.baseline in arms and p.candidate in arms and p.baseline != p.candidate for p in pairs)
             and len({p.name for p in pairs}) == len(pairs), "invalid or missing explicitly named pair")
    diagnostics = {} if diagnostics is None else diagnostics
    _require(isinstance(diagnostics, Mapping) and diagnostics.keys() <= arms.keys(), "unknown diagnostic arm")
    parsed = {arm: _parse_run(run, rows, keys, split, inputs_sha256) for arm, run in arms.items()}
    preds = {arm: p[0] for arm, p in parsed.items()}
    out = {"schema": "joey-panel-report/v1", "split": split, "inputs_sha256": inputs_sha256,
           "labels_sha256": labels_sha256, "arms": {}, "pairs": {}, "retrospective_passes": {},
           "intervals": "pointwise paired 95% percentile intervals, not multiplicity-adjusted",
           "bootstrap_reps": REPLICATES, "bootstrap_seed": SEED}
    for arm, ps in preds.items():
        first = ev.evaluate(rows, gold, {arm: ps}, baseline=arm, bootstrap_reps=0)["arms"][arm]
        bad = {r["id"]: o.error for r, y, p in zip(rows, gold, ps)
               if (o := ev._observe(r, y["gold_index"], p)).error}
        out["arms"][arm] = {"quality_at_0.8": first, "compute": parsed[arm][3],
                            "invalid_ids": bad, "confidence": {}}
        for t in THRESHOLDS:
            q = first if t == .8 else ev.evaluate(rows, gold, {arm: ps}, baseline=arm,
                                                   threshold=t, bootstrap_reps=0)["arms"][arm]
            out["arms"][arm]["confidence"][str(t)] = _confidence(q)
    for pair in pairs:
        a, b = pair.baseline, pair.candidate
        _check_pair(pair, parsed[a][2], parsed[b][2])
        entry = {"baseline": a, "candidate": b, "kind": pair.kind, "confidence_deltas": {},
                 "timing_criterion": _cost_pair(parsed[a][3], parsed[b][3], arms[a], arms[b])}
        for t in THRESHOLDS:
            q = ev.evaluate(rows, gold, {a: preds[a], b: preds[b]}, baseline=a,
                            pass_series=[a, b] if t == .8 else None, threshold=t,
                            bootstrap_reps=REPLICATES, bootstrap_seed=SEED)
            delta = q["paired_deltas"][b]
            if t == .8:
                entry["paired_deltas"] = delta
                entry["transitions"] = q["passes"]["first_to_last"]
                if pair.kind == "compact_vs_stock":
                    entry["compact_noninferiority"] = _noninferiority(delta, q["all_distributions_valid"])
            entry["confidence_deltas"][str(t)] = {
                scope: {k: d for k, d in delta[scope].items() if k.startswith("confident_")}
                for scope in ("micro", "macro")}
        out["pairs"][pair.name] = entry
    for arm, passes in diagnostics.items():
        requested = list(passes)
        _require(len(requested) >= 2 and all(type(n) is int and n >= 1 and ev._text(k) for n, k in requested)
                 and all(a[0] < b[0] for a, b in zip(requested, requested[1:])), "invalid ordered pass list")
        projected = {}
        for n, kind in requested:
            name = f"{n}:{kind}"
            ps = []
            for t, (key, digest) in zip(parsed[arm][1], keys):
                rds = t["readouts"]
                _require(n <= len(rds) and rds[n-1]["kind"] == kind, "diagnostic pass absent or kind mismatch")
                ps.append({"id": key, "input_sha256": digest, "probs": rds[n-1]["probs"]})
            projected[name] = ps
        q = ev.evaluate(rows, gold, projected, baseline=next(iter(projected)),
                        pass_series=list(projected), bootstrap_reps=0)
        out["retrospective_passes"][arm] = {"quality_at_0.8": q["arms"], "transitions": q["passes"],
                                           "cost": None, "independently_timed": False}
    return out
