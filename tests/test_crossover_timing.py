"""Invented worker-shaped streams only. No files, devices, labels or model execution."""

from dataclasses import FrozenInstanceError, replace
import json
import math
import unittest

from joey_research.crossover_timing import TimingFrame, crossed_timing
from joey_research.evaluation import input_sha256


def fixture():
    rows = [{"id": f"invented-row-{i}", "group_id": group, "task": "left" if group in "ac" else "right",
             "prompt": f"Invented question {i}?", "choices": ["red", "blue", "green", "pink"]}
            for i, group in enumerate(("a", "d", "b", "a", "c", "d"))]
    # Row counts a:2, b:1, c:1, d:2. Both generator seeds cross tasks.
    strata = {"a": 7, "b": 7, "c": "7", "d": "7"}
    primary = {
        (0, "full:split-control"): [1, 2, 4, 6, 10, 7],
        (1, "full:split-control"): [2, 3, 1, 5, 4, 5],
        (0, "select-20:naive-2"): [6, 3, 1, 2, 2, 1],
        (1, "select-20:naive-2"): [8, 6, 2, 4, 12, 8],
    }
    frames = {}
    for slot in (0, 1):
        for payload, period, layers in (("full", (1, 2)[slot], (24, 24, 28)),
                                        ("select-20", (2, 1)[slot], (20, 20, 24))):
            for visits, (name, k, method, callbacks) in zip(layers, (
                    ("stock", 1, "naive", (1, 0, 0, 0)),
                    ("split-control", 2, "zero_extra", (0, 1, 1, 1)),
                    ("naive-2", 2, "naive", (0, 1, 2, 1))), strict=True):
                settings = {"window": [8, 12], "loop": {"k": k, "method": method}, "reverse": False}
                if payload == "select-20":
                    settings.update(compact_recipe="select-20", compact_manifest_sha256="a" * 64)
                key = (slot, f"{payload}:{name}")
                packet = f"{len(frames) + 1:064x}"
                ms = primary.get(key, [(10, 8)[slot] if payload == "full" else (4, 3)[slot]] * len(rows))
                counts = dict(zip(("stock_calls", "prelude_calls", "block_calls", "coda_calls"), callbacks))
                predictions = [{"id": r["id"], "input_sha256": input_sha256(r), "probs": [0.5, 0.5, 0, 0]}
                               for r in rows]
                trace = [{"id": r["id"], "packet_sha256": packet, "tokens": 1 + i, "callback_counts": counts,
                          "layer_calls": visits, "readouts": [], "cuda_event_ms": ms[i],
                          "wall_ms": ms[i] + 3 * period + slot} for i, r in enumerate(rows)]
                frames[key] = TimingFrame(slot, period, settings, packet, predictions, trace)
    return rows, strata, frames


class CrossoverTimingTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.strata, self.frames = fixture()
        self.stock = self.frames[0, "full:stock"]
        self.compact = self.frames[1, "select-20:naive-2"]

    def report(self, **options):
        args = {"rows": self.rows, "frames": tuple(self.frames.values()), "strata": self.strata, "reps": 0}
        return crossed_timing(**{**args, **options})

    def reject_frame(self, frame, **changes):
        with self.assertRaises(ValueError):
            self.report(frames=[replace(f, **changes) if f is frame else f for f in self.frames.values()])

    def test_slot_means_work_and_crossed_ratios_count_inputs_only_once(self):
        report = self.report()
        self.assertEqual((report["rows"], report["groups"]), (6, 4))
        methods, comparisons = report["methods"], report["comparisons"]
        self.assertEqual(set(comparisons), {"equal-work", "same-k-stock", "same-k-naive-2"})
        primary = comparisons["equal-work"]
        full, compact = methods[primary["full"]], methods[primary["compact"]]
        self.assertEqual(full["by_slot"]["0"], {"period": 1, "mean_cuda_ms": 5, "mean_wall_ms": 8})
        self.assertEqual(compact["by_slot"]["0"], {"period": 2, "mean_cuda_ms": 2.5, "mean_wall_ms": 8.5})
        self.assertEqual((full["by_slot"]["1"]["period"], compact["by_slot"]["1"]["period"]), (2, 1))
        self.assertAlmostEqual(full["by_slot"]["1"]["mean_cuda_ms"], 20 / 6)
        self.assertAlmostEqual(compact["by_slot"]["1"]["mean_wall_ms"], 40 / 6 + 4)
        for arm in (full, compact):
            work = arm["work_per_row"]
            self.assertEqual(tuple(work[k] for k in ("layer_calls", "full_attention_calls",
                                                     "gated_deltanet_calls", "head_calls")), (24, 6, 18, 1))
        self.assertTrue(primary["matched_logical_work"])
        self.assertEqual(primary["by_slot"]["0"]["cuda_ratio"], 0.5)
        self.assertEqual(primary["by_slot"]["1"]["cuda_ratio"], 2)
        self.assertAlmostEqual(primary["geometric_cuda_ratio"], 1)
        for name, expected in (("same-k-stock", (4 / 10, 3 / 8)), ("same-k-naive-2", (15 / 60, 40 / 48))):
            same_k = comparisons[name]
            self.assertFalse(same_k["matched_logical_work"])
            for s in (0, 1):
                self.assertAlmostEqual(same_k["by_slot"][str(s)]["cuda_ratio"], expected[s])
        raw = json.dumps(report, allow_nan=False)
        self.assertEqual(json.loads(raw), report)
        for identifier in ("invented-row-", self.stock.packet_sha256, "input_sha256", '"probs"'):
            self.assertNotIn(identifier, raw)

    def test_percentiles_recompute_joint_scores_with_repeated_whole_graphs(self):
        report = self.report(seed=0, reps=2)
        self.assertEqual(report["bootstrap"]["status"], "ok")
        self.assertEqual(report, self.report(frames=tuple(reversed(self.frames.values())), seed=0, reps=2))
        # Seed 0 picks (b,b ; c,d), then (b,b ; d,d). Expanding those graphs
        # gives 5 and 6 rows. These sums are independent of the stats implementation.
        for name, (x0, x1), (y0, y1) in (
                ("equal-work", (8 / 27, 30 / 14), (10 / 26, 32 / 18)),
                ("same-k-naive-2", (8 / 50, 30 / 40), (10 / 60, 32 / 48)),
                ("same-k-stock", (4 / 10, 3 / 8), (4 / 10, 3 / 8))):
            cmp = report["comparisons"][name]
            for interval, first, second in (
                    (cmp["by_slot"]["0"]["ci95"], x0, y0),
                    (cmp["by_slot"]["1"]["ci95"], x1, y1),
                    (cmp["geometric_cuda_ratio_ci95"], math.sqrt(x0 * x1), math.sqrt(y0 * y1))):
                lo, hi = sorted((first, second))
                self.assertAlmostEqual(interval[0], .975 * lo + .025 * hi)
                self.assertAlmostEqual(interval[1], .025 * lo + .975 * hi)

    def test_withhold_all_intervals_without_group_support_or_draws(self):
        for args, status in (({}, "not_requested"), ({"reps": 25, "strata": {**self.strata, "a": 42}},
                                                    "insufficient_independent_groups")):
            with self.subTest(status=status):
                report = self.report(**args)
                self.assertEqual(report["bootstrap"]["status"], status)
                self.assertEqual(report["bootstrap"]["used_replicates"], 0)
                for cmp in report["comparisons"].values():
                    self.assertIsNone(cmp["geometric_cuda_ratio_ci95"])
                    self.assertTrue(all(slot["ci95"] is None for slot in cmp["by_slot"].values()))
                self.assertAlmostEqual(report["comparisons"]["equal-work"]["geometric_cuda_ratio"], 1)

    def test_raw_probability_drift_first_tie_and_strict_threshold(self):
        key = (1, "full:stock")
        preds = self.frames[key].predictions
        preds[0]["probs"] = [.6, .4000004, 0, 0]  # legal, not exactly unit sum: do not normalize
        preds[1]["probs"] = [.5 - 2**-22, .5 + 2**-22, 0, 0]  # sub-threshold argmax flip
        preds[2]["probs"] = [.5 - 5e-6, .5 - 5e-6, 1e-5, 0]  # exactly at the threshold
        self.frames[key] = replace(self.frames[key], predictions=preds)
        repro = self.report()["reproducibility"]
        self.assertEqual(set(repro), {name for _, name in self.frames})
        self.assertEqual(repro["full:stock"], {
            "total_rows": 6, "identical_vector_rows": 3, "argmax_agreement_rows": 5,
            "argmax_agreement": 5 / 6, "rows_exceeding_1e-5": 1, "max_abs_probability_difference": .6 - .5,
        })
        self.assertEqual(repro["select-20:stock"]["identical_vector_rows"], 6)

    def test_frame_owns_settings_predictions_and_traces(self):
        settings, preds, trace = self.stock.settings, self.stock.predictions, self.stock.trace
        f = replace(self.stock, settings=settings, predictions=preds, trace=trace)
        settings["window"].reverse()
        preds[0]["probs"].clear()
        trace[0]["callback_counts"].clear()
        f.trace[0].clear()
        self.assertEqual(f, self.stock)
        with self.assertRaises(FrozenInstanceError):
            f.slot = 1

    def test_roster_roles_canonical_methods_and_distinct_packet_pins(self):
        frames = list(self.frames.values())
        for invalid in (frames[:-1], frames + frames[:1], iter(frames), [None, *frames[1:]]):
            with self.subTest(kind=type(invalid)), self.assertRaises(ValueError):
                self.report(frames=invalid)
        reused = self.compact.packet_sha256
        self.reject_frame(self.stock, packet_sha256=reused, trace=[{**t, "packet_sha256": reused}
                                                                   for t in self.stock.trace])
        split = self.frames[0, "full:split-control"]
        self.reject_frame(self.stock, settings=split.settings,
                          trace=[{**t, "packet_sha256": self.stock.packet_sha256} for t in split.trace])
        for changes in ({"slot": 1}, {"period": 2}, {"slot": False}, {"period": 1.0},
                        {"period": 0}, {"slot": 2}, {"packet_sha256": "z" * 64}):
            with self.subTest(changes=changes):
                self.reject_frame(self.stock, **changes)
        for patch in ({"window": [7, 11]}, {"window": [8, 12.]}, {"reverse": 0},
                      {"loop": {"method": "naive", "k": True}}, {"loop": {"method": "naive", "k": 1.0}},
                      {"loop": {"method": "naive", "k": 1, "readout": "final"}},
                      {"compact_manifest_sha256": "a" * 64}, {"policy": {}}):
            with self.subTest(patch=patch):
                self.reject_frame(self.stock, settings={**self.stock.settings, **patch})
        for patch in ({"compact_recipe": "select-20-late"}, {"compact_manifest_sha256": "b" * 64},
                      {"compact_manifest_sha256": "!" * 64}):
            with self.subTest(patch=patch):
                self.reject_frame(self.compact, settings={**self.compact.settings, **patch})
        self.reject_frame(self.compact, settings={k: v for k, v in self.compact.settings.items()
                                                  if k != "compact_manifest_sha256"})

    def test_exact_graph_strata_four_choices_and_unambiguous_inputs(self):
        for strata in (None, {}, {"a": 7}, {**self.strata, "extra": 7}, {**self.strata, "a": " "},
                       {**self.strata, "a": 7.0}, {**self.strata, "a": True}):
            with self.subTest(strata=strata), self.assertRaises(ValueError):
                self.report(strata=strata)
        for rows in ([], [*self.rows, self.rows[0]], [{**self.rows[0], "task": "other"}, *self.rows[1:]],
                     [{**self.rows[0], "choices": ["yes", "no"]}, *self.rows[1:]],
                     [{**self.rows[0], "unexpected": 0}, *self.rows[1:]]):
            with self.subTest(count=len(rows)), self.assertRaises(ValueError):
                self.report(rows=rows)
        for options in ({"reps": -1}, {"reps": 1.0}, {"seed": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.report(**options)

    def test_predictions_and_traces_must_bind_every_ordered_row(self):
        pred, trace = self.compact.predictions, self.compact.trace
        for changes in ({"predictions": pred[:-1]}, {"trace": trace + trace[:1]},
                        {"predictions": pred[::-1]}, {"trace": trace[::-1]},
                        {"predictions": [pred[1], *pred[1:]]}, {"trace": [trace[1], *trace[1:]]},
                        {"predictions": [{**pred[0], "input_sha256": "0" * 64}, *pred[1:]]},
                        {"predictions": [{**pred[0], "extra_vector": []}, *pred[1:]]},
                        {"trace": [{**trace[0], "packet_sha256": self.stock.packet_sha256}, *trace[1:]]},
                        {"trace": [{**trace[0], "readouts": [{"probs": [.25] * 4}]}, *trace[1:]]},
                        {"trace": [{**trace[0], "ensemble_calls": 1}, *trace[1:]]}):
            with self.subTest(field=next(iter(changes))):
                self.reject_frame(self.compact, **changes)
        for p in ([1, 0], (1, 0, 0, 0), [True, 0, 0, 0], ["1", 0, 0, 0], [0] * 4,
                  [.26] * 4, [1, -0.1, 0.1, 0], [math.nan, 0, 0, 0], [math.inf, 0, 0, 0]):
            with self.subTest(probs=p):
                self.reject_frame(self.compact, predictions=[{**pred[0], "probs": p}, *pred[1:]])

    def test_observed_counts_fixed_callbacks_and_positive_finite_times(self):
        # Both full paths charge 24 layers. Their actual callback counts are not interchangeable.
        split = self.frames[0, "full:split-control"].trace[0]["callback_counts"]
        trace = self.stock.trace
        calls = trace[0]["callback_counts"]
        patches = [{"tokens": v} for v in (0, True, 1.0, 2)]
        patches += [{"layer_calls": 24.0}, {"layer_calls": 20}, {"callback_counts": split},
                    {"callback_counts": {**calls, "block_calls": 0.0}},
                    {"callback_counts": {**calls, "stock_calls": True}},
                    {"callback_counts": {k: v for k, v in calls.items() if k != "coda_calls"}}]
        for field in ("cuda_event_ms", "wall_ms"):
            patches += [{field: v} for v in (0, -1, math.inf, math.nan, True, "1")]
            self.reject_frame(self.stock, trace=[{k: v for k, v in trace[0].items() if k != field}, *trace[1:]])
        for patch in patches:
            with self.subTest(patch=patch):
                self.reject_frame(self.stock, trace=[{**trace[0], **patch}, *trace[1:]])


if __name__ == "__main__":
    unittest.main()
