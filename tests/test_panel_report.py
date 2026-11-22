"""Invented worker artifacts only; no model, hardware, campaign files, or live data."""

import hashlib
import json
from dataclasses import replace
import unittest

from joey_research import evaluation as ev
from joey_research import panel_report as panel
from joey_research.loop import LoopCounts
from joey_research.worker import ensemble_groups, method_sha256


def raw(v):
    return json.dumps(v, allow_nan=False, separators=(",", ":")).encode("ascii")


def stream(rows):
    return b"".join(raw(r) + b"\n" for r in rows)


def pin(data):
    return hashlib.sha256(data).hexdigest()


def inputs(groups=("a", "a", "b", "c", "c", "d")):
    return [{"id": f"fx-{i}", "task": "one" if g <= "b" else "two",
             "group_id": g, "prompt": f"Invented decision {i}", "choices": ["left", "right"]}
            for i, g in enumerate(groups)]


def settings(**kw):
    return {"window": [4, 8], "reverse": False, "loop": {"k": 1}, **kw}


def arm(rows, spec, *, ps=None, cuda=2., split="selection", slot="fixture_slot", prior=None):
    source = pin(stream(rows))
    pkt = {"schema": 1, "split": split, "inputs_sha256": source, "deadline_utc": 1, **spec}
    permit = "e" * 64 if split == "selection" else None
    if permit is not None:
        pkt.update(method_sha256=method_sha256(spec), permit_sha256=permit)
    packet = raw(pkt)
    packet_pin = pin(packet)
    preds, traces = [], []
    ensemble = spec.get("order_ensemble")
    chosen = ensemble_groups(((r["task"], r["group_id"]) for r in rows), ensemble,
                             split=split) if ensemble else set()
    compact = "compact_manifest_sha256" in spec
    depth = 16 if compact else 24
    loop = spec["loop"]
    k = loop["k"]
    diagnostic = loop.get("readout") == "diagnostic"
    for i, r in enumerate(rows):
        p = [.9, .1] if ps is None else ps[i]
        preds.append({"id": r["id"], "input_sha256": ev.input_sha256(r), "probs": p})
        n = 1 + int(r["group_id"] in chosen)
        blocks = 1 if loop.get("method") == "zero_extra" else k
        counts = LoopCounts(1, 0, 0, 0) if k == 1 else LoopCounts(0, n, n * blocks, n * (blocks if diagnostic else 1))
        rd = [{"block_evaluations": j, "kind": "first_pass" if j == 1 else "iterate",
               "probs": prior[i] if j < k and prior is not None else p}
              for j in range(1, blocks+1)] if diagnostic else []
        lo, hi = spec["window"]
        trace = {"id": r["id"], "readouts": rd, "tokens": 7 * n, "wall_ms": float(i + 1),
                 "cuda_event_ms": cuda, "layer_calls": counts.layer_evaluations(lo, hi-lo, depth-hi),
                 "callback_counts": {name: getattr(counts, name) for name in
                                     ("stock_calls", "prelude_calls", "block_calls", "coda_calls")}}
        if ensemble:
            trace.update(ensemble_calls=n, forward_calls=n)
        if permit is not None:
            trace["packet_sha256"] = packet_pin
        traces.append(trace)
    pred, trace = stream(preds), stream(traces)
    n = len(rows)
    met = {"schema": 1, "decisions": n, "hardware": "A100", "batch_size": 1,
           "packet_sha256": packet_pin, "peak_allocated_bytes": 12345,
           "warm_latency_ms": {p: float((n*int(p) + 99)//100) for p in ("50", "95", "99")}}
    if compact:
        met.update(physical_layers=depth, compact_manifest_sha256=spec["compact_manifest_sha256"])
    if permit is not None:
        met.update(split=split, inputs_sha256=source, method_sha256=method_sha256(spec), permit_sha256=permit,
                   result_sha256={"predictions.jsonl": pin(pred), "trace.jsonl": pin(trace)})
    if ensemble:
        fwd = sum(t["forward_calls"] for t in traces)
        met["compute_spend"] = {"groups": len({r["group_id"] for r in rows}), "extra_groups": len(chosen),
                                "forward_calls": fwd, "coda_calls": fwd, "layer_calls": 24*fwd}
    complete = raw(met)
    return panel.ArmRun(packet, pred, trace, complete,
                        {"packet.json": packet_pin, "predictions.jsonl": pin(pred),
                         "trace.jsonl": pin(trace), "complete.json": pin(complete)}, slot, permit)


def rewrite(run, *, pred=None, trace=None, summary=None):
    p, t = run.predictions if pred is None else pred, run.trace if trace is None else trace
    m = json.loads(run.complete)
    if "result_sha256" in m:
        m["result_sha256"] = {"predictions.jsonl": pin(p), "trace.jsonl": pin(t)}
    m.update(summary or {})
    return replace(run, predictions=p, trace=t, complete=raw(m), pins={
        **run.pins, "predictions.jsonl": pin(p), "trace.jsonl": pin(t), "complete.json": pin(raw(m))})


def report(rows, arms, pairs, *, labels=None, **kwargs):
    labels = stream([ev.label_row(r, 0) for r in rows]) if labels is None else labels
    data = stream(rows)
    return panel.panel_report(data, labels, arms, split=kwargs.pop("split", "selection"),
                              inputs_sha256=pin(data), labels_sha256=pin(labels), pairs=pairs, **kwargs)


class PanelTests(unittest.TestCase):
    def test_named_paths_diagnostic_projections_and_actual_compute(self):
        rows = inputs()
        base_p = [[.1, .9]] * 3 + [[.9, .1]] * 3
        after = [[.99, .01]] * 5 + [[.01, .99]]
        a = arm(rows, settings(loop={"k": 2, "method": "zero_extra"}), ps=base_p)
        b = arm(rows, settings(loop={"k": 2, "readout": "diagnostic"}), ps=after, prior=base_p, cuda=2.2)
        out = report(rows, {"zero": a, "loop": b}, [panel.Pair("path", "zero", "loop", "matched_path")],
                     diagnostics={"loop": [(1, "first_pass"), (2, "iterate")]})
        entry = out["pairs"]["path"]
        self.assertEqual(list(out["pairs"]), ["path"])
        self.assertEqual(entry["paired_deltas"]["requested_replicates"], 2000)
        self.assertEqual(entry["paired_deltas"]["groups_per_task"], {"one": 2, "two": 2})
        self.assertEqual((entry["transitions"]["corrections"], entry["transitions"]["corruptions"]), (3, 1))
        self.assertTrue(entry["timing_criterion"]["within_10pct_same_slot"])
        c = out["arms"]["loop"]["compute"]
        self.assertEqual(c["totals"], {"tokens": 42, "forward_calls": 6, "block_evaluations": 12,
                                       "coda_readouts": 12, "layer_calls": 264})
        self.assertEqual((c["wall_ms"]["50"], c["wall_ms"]["95"], c["peak_allocated_bytes"]), (3, 6, 12345))
        self.assertEqual([(x["id"], x["cuda_event_ms"]) for x in c["per_decision"]],
                         [(r["id"], 2.2) for r in rows])
        self.assertEqual(out["arms"]["zero"]["quality_at_0.8"]["micro"]["risk_coverage"][-1]["accepted"], 6)
        self.assertEqual(out["arms"]["loop"]["confidence"]["0.99"]["micro"]["confident"], 6)
        self.assertIsNone(out["arms"]["zero"]["confidence"]["0.95"]["micro"]["confident_error_conditional"])
        self.assertEqual(entry["confidence_deltas"]["0.99"]["macro"]["confident_error_conditional"]["status"],
                         "undefined_estimate")
        view = out["retrospective_passes"]["loop"]
        self.assertFalse(view["independently_timed"])
        self.assertIsNone(view["cost"])
        self.assertEqual(view["transitions"]["overthought"], 1)

    def test_ensemble_is_whole_group_and_stock_counts_include_hidden_block_and_coda(self):
        rows = inputs()
        budget = {"numerator": 1, "denominator": 2, "seed": 81}
        ensemble = arm(rows, settings(loop={"k": 2, "method": "zero_extra"}, order_ensemble=budget))
        stock = arm(rows, settings(), slot="different_fixture_slot")
        pair = panel.Pair("budget", "stock", "ensemble", "order_ensemble")
        out = report(rows, {"stock": stock, "ensemble": ensemble}, [pair])
        t = out["arms"]["stock"]["compute"]["totals"]
        self.assertEqual((t["block_evaluations"], t["coda_readouts"], t["layer_calls"]), (6, 6, 144))
        spend = out["arms"]["ensemble"]["compute"]["whole_group_spend"]
        self.assertEqual(spend["extra_groups"], 2)
        self.assertEqual(spend["forward_calls"], spend["coda_calls"])
        self.assertEqual(spend["layer_calls"], 24*spend["forward_calls"])
        self.assertFalse(out["pairs"]["budget"]["timing_criterion"]["same_verified_A100_slot"])
        self.assertIsNone(out["pairs"]["budget"]["timing_criterion"]["within_10pct_same_slot"])
        ts = [json.loads(l) for l in ensemble.trace.splitlines()]
        ts[1]["forward_calls"] = 3 - ts[1]["forward_calls"]
        with self.assertRaisesRegex(ev.EvaluationError, "whole-group"):
            report(rows, {"stock": stock, "ensemble": rewrite(ensemble, trace=stream(ts))}, [pair])
        with self.assertRaisesRegex(ev.EvaluationError, "spend total"):
            report(rows, {"stock": stock, "ensemble": rewrite(ensemble, summary={"compute_spend": {}})}, [pair])

    def test_compact_strict_bounds_not_quasi_equivalence(self):
        rows = inputs()
        compact = settings(compact_manifest_sha256="a"*64)
        pair = panel.Pair("ni", "stock", "small", "compact_vs_stock")
        out = report(rows, {"stock": arm(rows, settings()), "small": arm(rows, compact, cuda=7.)}, [pair])
        self.assertTrue(out["pairs"]["ni"]["compact_noninferiority"]["passed"])
        self.assertFalse(out["pairs"]["ni"]["timing_criterion"]["within_10pct_same_slot"])
        for lower, upper, ok in [(-.02, .009, False), (-.019, .01, False), (-.019, .009, True)]:
            delta = {"macro": {"accuracy": {"ci95": [lower, .5], "status": "ok"},
                               "brier": {"ci95": [-.03, upper], "status": "ok"}}}
            self.assertEqual(panel._noninferiority(delta, True)["passed"], ok)
        with self.assertRaisesRegex(ev.EvaluationError, "full stock"):
            report(rows, {"stock": arm(rows, compact), "small": arm(rows, settings())}, [pair])
        tiny = inputs(("a", "a"))
        out = report(tiny, {"stock": arm(tiny, settings()), "small": arm(tiny, compact)}, [pair])
        self.assertIsNone(out["pairs"]["ni"]["compact_noninferiority"]["passed"])
        self.assertEqual(out["pairs"]["ni"]["paired_deltas"]["macro"]["brier"]["status"],
                         "insufficient_independent_groups")
        broken = [[.9, .1]] * 5 + [[0, 0]]
        out = report(rows, {"stock": arm(rows, settings()), "small": arm(rows, compact, ps=broken)}, [pair])
        self.assertIn("fx-5", out["arms"]["small"]["invalid_ids"])
        self.assertEqual(out["pairs"]["ni"]["compact_noninferiority"]["reason"], "invalid_distribution")

    def test_correctly_charges_final_only_policy_once(self):
        rows = inputs(("a", "a"))
        run = arm(rows, settings(loop={"k": 3}, policy={"pool": "final"}))
        paired = report(rows, {"policy": run, "stock": arm(rows, settings())},
                        [panel.Pair("policy_check", "stock", "policy")])
        totals = paired["arms"]["policy"]["compute"]["totals"]
        self.assertEqual((totals["block_evaluations"], totals["coda_readouts"]), (6, 2))

    def test_rejects_unpaired_reordered_and_cross_bound_material(self):
        rows = inputs()
        ok = arm(rows, settings())
        ps, ts = [json.loads(l) for l in ok.predictions.splitlines()], [json.loads(l) for l in ok.trace.splitlines()]
        pair = [panel.Pair("fixture", "a", "b")]
        for b in [replace(ok, predictions=ok.predictions+b"\n"), rewrite(ok, pred=stream(ps[:-1])),
                  rewrite(ok, pred=stream(ps[::-1])), rewrite(ok, trace=stream(ts[::-1])),
                  rewrite(ok, trace=stream([{**ts[0], "packet_sha256": "d"*64}, *ts[1:]])),
                  rewrite(ok, pred=stream([{**ps[0], "input_sha256": "0"*64}, *ps[1:]])),
                  rewrite(ok, summary={"result_sha256": {}}), rewrite(ok, summary={"warm_latency_ms": {}}),
                  arm(rows, settings(), split="discovery"), replace(ok, permit_sha256="d"*64)]:
            with self.assertRaises(ev.EvaluationError):
                report(rows, {"a": ok, "b": b}, pair)
        with self.assertRaisesRegex(ev.EvaluationError, "labels:.*reordered"):
            report(rows, {"a": ok, "b": ok}, pair, labels=stream([ev.label_row(r, 0) for r in reversed(rows)]))
        other_window = arm(rows, settings(window=[2, 5], loop={"k": 3}))
        with self.assertRaisesRegex(ev.EvaluationError, "matched_path"):
            report(rows, {"a": arm(rows, settings(loop={"k": 2, "method": "zero_extra"})), "b": other_window},
                   [panel.Pair("p", "a", "b", "matched_path")])
        d = arm(rows, settings(), split="discovery", slot=None)
        out = report(rows, {"a": d, "b": d}, pair, split="discovery")
        self.assertIsNone(out["pairs"]["fixture"]["timing_criterion"]["within_10pct_same_slot"])


if __name__ == "__main__":
    unittest.main()
