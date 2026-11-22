"""Invented panels only: no generator, tokenizer, runner or released dataset."""

from copy import deepcopy
import json
import random
import unittest

from benchmarks.horizon_metrics import compare_horizons
from benchmarks.horizon_interaction import ROLES, payload_interaction
from joey_research import evaluation as ev


_ROOTS = ("ASH", "ELM", "OAK", "YEW")
_STREAMS = ("rows", "labels", "baseline", "candidate")
_MAPS = ("design_by_id", "baseline_tokens", "candidate_tokens")


def _probs(gold, right, confidence=1.0):
    top = gold if right else (gold + 1) % 4
    probs = [0.0] * 4
    probs[top], probs[(top + 1) % 4] = confidence, 1 - confidence
    return probs


def _panel(sizes=(2,), decisions=None, lengths=None):
    bundles, design, tokens = [], {}, {}
    for s, size in enumerate(sizes, 1):
        for g in range(size):
            for r, order in enumerate(((0, 1, 2, 3), (2, 0, 3, 1))):
                for h in (1, 2, 4, 6):
                    for e, root in enumerate(_ROOTS):
                        key = f"item:{s}:{g}:{r}:{h}:{e}"
                        row = {"id": key, "group_id": f"graph:{s}:{g}", "task": "toy",
                               "prompt": f"Invented question {key}",
                               "choices": [f"Enter at {_ROOTS[i]}" for i in order]}
                        gold = order.index(e)
                        label = ev.label_row(row, gold)
                        binding = {"id": key, "input_sha256": label["input_sha256"]}
                        a, b = decisions(s, g, e, r, h) if decisions else (False, h == 6)
                        bundles.append((row, label, {**binding, "probs": _probs(gold, a)},
                                        {**binding, "probs": _probs(gold, b)}))
                        design[key] = {"hop": h, "entry": root, "rendering": r, "seed": s}
                        tokens[key] = lengths(s, g, e, r, h) if lengths else 40
    random.Random(91).shuffle(bundles)
    return {**dict(zip(_STREAMS, map(list, zip(*bundles)))),
            "design_by_id": design, "baseline_tokens": tokens, "candidate_tokens": tokens.copy()}


class HorizonMetricsTests(unittest.TestCase):
    def test_signed_interaction_and_transitions(self):
        def decisions(s, g, e, r, h):
            return (e < 2, e % 2 == 0) if h == 1 else (h == 2, h == 6)

        data = _panel((2, 3), decisions)
        result = compare_horizons(**data)
        near, far = (result["by_hop"][h] for h in ("2", "6"))
        self.assertEqual((far["graphs"], far["rows"]), (5, 40))
        self.assertEqual(far["baseline"]["brier"], 2)  # summed over classes, not divided by four
        self.assertEqual(near["paired"]["accuracy"]["estimate"], -1)
        self.assertEqual(far["paired"]["accuracy"]["estimate"], 1)
        self.assertEqual(far["correction"], {"count": 40, "support": 40, "rate": 1})
        self.assertEqual(far["corruption"], {"count": 0, "support": 0, "rate": None})
        self.assertEqual(near["corruption"], far["correction"])
        self.assertIsNone(near["correction"]["rate"])
        balanced = result["by_hop"]["1"]
        self.assertEqual(balanced["correction"], {"count": 10, "support": 20, "rate": .5})
        self.assertEqual(balanced["corruption"], balanced["correction"])
        self.assertIsNone(far["candidate"]["mean_confidence_wrong"])
        self.assertIsNone(far["baseline"]["mean_confidence_right"])
        for metric, value in (("accuracy", 2), ("brier", -4)):
            expected = {"estimate": value, "ci95": [value, value], "defined_reps": 2000}
            self.assertEqual(result["primary"][metric], expected)
            self.assertEqual(result["exact_token"][metric], expected)
        encoded = json.dumps(result, allow_nan=False)
        for private in ("item:", "graph:", "Enter at ", "Invented question", "input_sha256"):
            self.assertNotIn(private, encoded)

    def test_bootstrap_draws_graphs_not_entries_renderings_or_separate_horizons(self):
        def decisions(s, g, e, r, h):
            if h == 1:
                return e < 2, e >= 2
            if h == 4:
                return not r, bool(r)
            return g == 0, g != 0

        result = compare_horizons(**_panel(decisions=decisions), reps=128)
        # The two graphs have opposite gains at BOTH 2 and 6. Marginals vary;
        # their paired interaction cannot. Each graph balances entries at 1 and renderings at 4.
        for h, ci in (("2", [-1, 1]), ("6", [-1, 1]), ("1", [0, 0]), ("4", [0, 0])):
            self.assertEqual(result["by_hop"][h]["paired"]["accuracy"]["ci95"], ci)
        self.assertEqual(result["primary"]["accuracy"]["ci95"], [0, 0])

    def test_strata_are_fixed_but_row_means_pool_unequal_sizes(self):
        data = _panel((2, 4, 3), decisions=lambda s, g, e, r, h: (h == 6 and s == 2,
                                                                 h == 6 and s == 1))
        result = compare_horizons(**data, reps=50)
        expected = (2 - 4) / 9
        self.assertEqual(result["primary"]["accuracy"]["estimate"], expected)
        self.assertEqual(result["primary"]["accuracy"]["ci95"], [expected, expected])
        self.assertEqual(set(result["exact_token"]["by_seed"]), {"1", "2", "3"})

    def test_exact_token_matches_semantic_tuple_and_keeps_both_rows(self):
        # Both renderings permute lengths among roots; every far length occurs somewhere
        # near, but only three of eight tuples match their OWN near row.
        near = ((30, 31, 32, 33), (32, 33, 30, 31))
        far = ((30, 32, 33, 31), (33, 32, 30, 31))

        def lengths(s, g, e, r, h):
            return 5 * g + s + (far if h == 6 else near)[r][e]

        def decisions(s, g, e, r, h):
            up = (h == 6) == ((e, r) in {(0, 0), (2, 1), (3, 1)})
            return not up, up

        data = _panel((2, 3), decisions, lengths)
        result = compare_horizons(**data, reps=12)
        matched = result["exact_token"]
        self.assertEqual(matched["support"], {"eligible_tuples": 15, "total_tuples": 40,
                                              "eligible_graphs": 5, "total_graphs": 5})
        self.assertEqual(matched["by_seed"]["2"]["eligible_tuples"], 9)
        self.assertEqual(matched["accuracy"]["estimate"], 2)
        self.assertEqual(matched["brier"]["estimate"], -4)
        self.assertEqual(result["primary"]["accuracy"]["estimate"], -.5)
        self.assertEqual(result["token_differences"], {
            "counts": {"0": 15, "1": 15, "-1": 5, "-2": 5},
            "min": -2, "max": 1, "mean": 0, "mean_abs": .75, "equal_fraction": 3 / 8,
        })
        for name in _STREAMS:
            data[name] = reversed(data[name])
        self.assertEqual(result, compare_horizons(**data, reps=12))

    def test_matched_ratio_uses_original_graph_multiplicities(self):
        # Contributing tuples per graph: 1, 3, 0. Interaction per tuple: -1, +1, 0.
        data = _panel((3,), decisions=lambda s, g, e, r, h: (h == 6 and g == 0, h == 6 and g == 1),
                      lengths=lambda s, g, e, r, h: 40 + int(h == 6 and 2 * e + r >= (1, 3, 0)[g]))
        result = compare_horizons(**data, reps=2, seed=0)
        self.assertEqual(result["exact_token"]["accuracy"]["estimate"], .5)
        # Seed 0 draws (1,1,0), (1,2,1). Do not redraw only from the two eligible graphs.
        for name, lo, hi in (("primary", 1 / 3, 2 / 3), ("exact_token", 5 / 7, 1)):
            interval = result[name]["accuracy"]["ci95"]
            for endpoint, q in zip(interval, (.025, .975)):
                self.assertAlmostEqual(endpoint, lo + q * (hi - lo))

    def test_sparse_expected_strata_and_absent_matches(self):
        for count in (0, 1):
            with self.subTest(eligible_in_second_seed=count):
                data = _panel((2, 3), lengths=lambda s, g, e, r, h:
                              40 + int(h == 6 and s == 2 and g >= count))
                result = compare_horizons(**data, reps=30)
                matched = result["exact_token"]
                self.assertEqual(matched["by_seed"]["2"]["eligible_graphs"], count)
                self.assertEqual(matched["accuracy"]["estimate"], 1)
                for m in ("accuracy", "brier"):
                    self.assertIsNone(matched[m]["ci95"])
                    self.assertIsNotNone(result["primary"][m]["ci95"])
        none = compare_horizons(**_panel(lengths=lambda s, g, e, r, h: 40 + h), reps=30)["exact_token"]
        self.assertEqual(none["support"]["eligible_tuples"], 0)
        self.assertIsNone(none["accuracy"]["estimate"])
        self.assertIsNone(none["brier"]["ci95"])
        singleton = compare_horizons(**_panel((2, 1)), reps=20)
        self.assertIsNone(singleton["by_hop"]["2"]["paired"]["accuracy"]["ci95"])
        self.assertIsNone(singleton["primary"]["brier"]["ci95"])

    def test_even_one_empty_pooled_replicate_withholds_interval(self):
        data = _panel((4,), lengths=lambda s, g, e, r, h: 40 + int(g >= 2 and h == 6))
        result = compare_horizons(**data, reps=128)
        matched = result["exact_token"]
        self.assertEqual(matched["support"]["eligible_graphs"], 2)
        for m in ("accuracy", "brier"):
            self.assertGreater(matched[m]["defined_reps"], 0)
            self.assertLess(matched[m]["defined_reps"], 128)
            self.assertIsNone(matched[m]["ci95"])
            self.assertEqual(result["primary"][m]["defined_reps"], 128)

    def test_native_normalization_ties_and_threshold_denominators(self):
        data = _panel((1,))
        for label, base, cand in zip(data["labels"], data["baseline"], data["candidate"]):
            meta = data["design_by_id"][label["id"]]
            gold, e = label["gold_index"], _ROOTS.index(meta["entry"])
            base["probs"] = [.25] * 4
            cand["probs"] = _probs(gold, e % 2 == 0, (.8, .9, .95, .99)[e])
            if meta["hop"] == 4:
                cand["probs"] = _probs(gold, True, .8)
                cand["probs"][(gold + 1) % 4] += ev.PROB_SUM_ATOL / 2
        result = compare_horizons(**data, reps=0)
        hop = result["by_hop"]["2"]
        self.assertEqual(hop["baseline"]["accuracy"], .25)  # first legal argmax, not all ties correct
        self.assertEqual(hop["baseline"]["brier"], .75)
        self.assertEqual(hop["corruption"]["count"], 0)  # both first-slot golds remain correct
        arm = hop["candidate"]
        self.assertEqual((arm["right"], arm["wrong"]), (4, 4))
        self.assertAlmostEqual(arm["mean_confidence_right"], (.8 + .95) / 2)
        self.assertAlmostEqual(arm["mean_confidence_wrong"], (.9 + .99) / 2)
        for t, n, errors in ((.8, 8, 4), (.9, 6, 4), (.95, 4, 2), (.99, 2, 2)):
            self.assertEqual(arm["thresholds"][str(t)],
                             {"support": n, "errors": errors, "error_rate": errors / n})
        normalized = result["by_hop"]["4"]["candidate"]
        conf = .8 / (1 + ev.PROB_SUM_ATOL / 2)
        self.assertAlmostEqual(normalized["mean_confidence_right"], conf)
        self.assertAlmostEqual(normalized["brier"], 2 * (1 - conf) ** 2)
        self.assertEqual(normalized["thresholds"]["0.8"], {"errors": 0, "support": 0, "error_rate": None})
        self.assertIsNone(hop["paired"]["accuracy"]["ci95"])

    def test_reject_stale_order_digest_schema_or_coverage(self):
        data = _panel((1,))
        for name in ("labels", "baseline", "candidate"):
            for damage in ("reorder", "stale", "extra_field", "short", "long"):
                with self.subTest(name=name, damage=damage), self.assertRaises(ev.EvaluationError):
                    d = deepcopy(data)
                    stream = d[name]
                    if damage == "reorder":
                        stream[1], stream[0] = stream[0], stream[1]
                    elif damage == "stale":
                        stream[0]["input_sha256"] = stream[1]["input_sha256"]
                    elif damage == "extra_field":
                        stream[0]["note"] = 1
                    elif damage == "short":
                        stream.pop()
                    else:
                        stream.append(stream[0])
                    compare_horizons(**d, reps=0)
        for name in _MAPS:
            for extra in (False, True):
                with self.subTest(name=name, extra=extra), self.assertRaises(ev.EvaluationError):
                    d = deepcopy(data)
                    if extra:
                        d[name]["unexpected"] = d[name][d["rows"][0]["id"]]
                    else:
                        del d[name][d["rows"][0]["id"]]
                    compare_horizons(**d, reps=0)
        with self.assertRaises(ev.EvaluationError):
            compare_horizons(**{k: v + v if k in _STREAMS else v for k, v in data.items()}, reps=0)

    def test_reject_stale_design_or_menu_even_with_current_digests(self):
        data = _panel((1,))
        key = next(k for k, d in data["design_by_id"].items() if d["hop"] == 1 and d["rendering"] == 0)
        for field, value in (("hop", True), ("hop", 3), ("hop", 1.0), ("hop", 2),  # duplicate cell
                             ("entry", "unrelated"), ("rendering", False), ("rendering", 2),
                             ("seed", True), ("seed", 9), ("extra", 1)):
            with self.subTest(field=field, value=value), self.assertRaises(ev.EvaluationError):
                d = deepcopy(data)
                d["design_by_id"][key][field] = value
                compare_horizons(**d, reps=0)
        with self.assertRaises(ev.EvaluationError):
            d = deepcopy(data)
            removed = d["rows"][0]["id"]
            for name in _STREAMS:
                d[name].pop(0)
            for name in _MAPS:
                del d[name][removed]
            compare_horizons(**d, reps=0)
        with self.assertRaisesRegex(ev.EvaluationError, "menu"):
            row = data["rows"][0]
            row["choices"].reverse()
            label = ev.label_row(row, 3 - data["labels"][0]["gold_index"])
            data["labels"][0] = label
            for arm in ("baseline", "candidate"):
                data[arm][0]["input_sha256"] = label["input_sha256"]
                data[arm][0]["probs"].reverse()
            compare_horizons(**data, reps=0)

    def test_reject_token_mismatch_and_non_native_scalars(self):
        data = _panel((1,))
        key = data["rows"][0]["id"]
        for a, b in ((40, 41), (0, 0), (-2, -2), (1, True), (True, 1), (40., 40), (40, 40.)):
            with self.subTest(tokens=(a, b)), self.assertRaises(ev.EvaluationError):
                d = deepcopy(data)
                d["baseline_tokens"][key], d["candidate_tokens"][key] = a, b
                compare_horizons(**d, reps=0)
        for arg, value in (("seed", False), ("reps", True), ("reps", -1)):
            with self.subTest(arg=arg), self.assertRaises(ev.EvaluationError):
                compare_horizons(**data, **{arg: value})
        for gold in (True, 1., -1, 4):
            with self.subTest(gold=gold), self.assertRaises(ev.EvaluationError):
                d = deepcopy(data)
                label = next(label for label in d["labels"] if label["gold_index"] == 1)
                label["gold_index"] = gold
                compare_horizons(**d, reps=0)
        for probs in ([True, 0, 0, 0], [float("nan"), 0, 0, 0], [float("inf"), 0, 0, 0],
                      [1.1, -.1, 0, 0], [.8, .2 + 2 * ev.PROB_SUM_ATOL, 0, 0], [.5, .5], (1, 0, 0, 0)):
            with self.subTest(probs=probs), self.assertRaises(ev.EvaluationError):
                d = deepcopy(data)
                d["candidate"][0]["probs"] = probs
                compare_horizons(**d, reps=0)


class CrossPayloadTests(unittest.TestCase):
    def compare(self, panel, arms=None, tokens=None):
        if arms is None:
            arms = dict(zip(ROLES, (panel["baseline"], panel["candidate"]) * 2))
        if tokens is None:
            tokens = {name: panel["baseline_tokens"] for name in ROLES}
        return payload_interaction(panel["rows"], panel["labels"], arms,
                                  design_by_id=panel["design_by_id"], tokens=tokens, reps=80)

    def test_shared_decisions_cancel_even_when_both_component_intervals_are_wide(self):
        data = _panel((3, 3), decisions=lambda s, g, e, r, h: (False, h == 6 and g == 0))
        marginal = compare_horizons(**data, reps=80)["primary"]["accuracy"]["ci95"]
        self.assertGreater(marginal[1], marginal[0])
        out = self.compare(data)
        for component in ("primary", "exact_token"):
            for metric in ("accuracy", "brier"):
                self.assertEqual(out[component][metric], {"estimate": 0, "ci95": [0, 0], "defined_reps": 80})

    def test_direction_and_ineligible_token_subset(self):
        data = _panel((2, 2), decisions=lambda s, g, e, r, h: (False, h == 6),
                      lengths=lambda s, g, e, r, h: 40 + int(h == 6))
        roles = dict(zip(ROLES, (data["baseline"], data["candidate"], data["candidate"], data["baseline"])))
        out = self.compare(data, roles)
        self.assertEqual(out["primary"]["accuracy"]["estimate"], 2)
        self.assertEqual(out["primary"]["brier"]["estimate"], -4)
        self.assertEqual(out["support"]["eligible_tuples"], 0)
        self.assertIsNone(out["exact_token"]["accuracy"]["ci95"])

    def test_within_payload_length_agreement_is_not_cross_payload_agreement(self):
        data = _panel((2,))
        tokens = {name: data["baseline_tokens"].copy() for name in ROLES}
        key = data["rows"][0]["id"]
        for name in ROLES[2:]:
            tokens[name][key] += 1
        with self.assertRaisesRegex(ev.EvaluationError, "cross-payload"):
            self.compare(data, tokens=tokens)


if __name__ == "__main__":
    unittest.main()
