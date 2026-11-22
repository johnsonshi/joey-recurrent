import unittest

from joey_research.evaluation import EvaluationError, input_sha256, label_row
from joey_research.operating_points import describe, fit_risk_cutoffs


class OperatingPoints(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": str(i), "group_id": str(i // 2), "task": "toy", "prompt": f"Question {i}",
                      "choices": ["east", "west"]} for i in range(8)]
        self.pred = [{"id": r["id"], "input_sha256": input_sha256(r), "probs": [p, 1-p]}
                     for r, p in zip(self.rows, [.95, .95, .85, .85, .7, .7, .6, .6], strict=True)]
        self.gold = [label_row(r, 0 if i < 4 else 1) for i, r in enumerate(self.rows)]

    def test_whole_ties_and_empirical_cutoffs_are_not_test_risk_guarantees(self):
        policy = fit_risk_cutoffs(self.rows, self.gold, self.pred, split="calibration", risk_targets=[.1], min_groups=2)
        self.assertEqual(policy["cutoffs"]["0.1"]["cutoff"], .85)
        opposite = [label_row(r, 1) for r in self.rows]
        final = describe(self.rows, opposite, self.pred, coverage_targets=[.1, .3, .6], fitted=policy, reps=20)
        self.assertIsNone(final["coverage_points"]["0.1"]["risk"])
        self.assertEqual(final["coverage_points"]["0.3"]["coverage"], .25)
        self.assertEqual(final["coverage_points"]["0.6"]["coverage"], .5)
        self.assertEqual(final["applied_calibration_cutoffs"]["0.1"]["risk"], 1.)
        self.assertEqual(final["confidence_by_outcome"]["correct"]["n"], 0)

    def test_undefined_and_group_support(self):
        fit = fit_risk_cutoffs(self.rows, self.gold, self.pred, split="calibration", min_groups=5)
        r = describe(self.rows, self.gold, self.pred, fitted=fit, reps=100)
        self.assertTrue(all(p["risk"] is None and p["accepted"] == 0 for p in r["applied_calibration_cutoffs"].values()))
        self.assertTrue(all(b["accuracy_ci95"] is None for b in r["reliability"]))
        same = [{**p, "probs": [.9, .1]} for p in self.pred]
        d = describe(self.rows, self.gold, same, reps=100)
        b = next(b for b in d["reliability"] if b["n"])
        self.assertEqual((b["independent_groups"], b["n"], b["accuracy"]), (4, 8, .5))
        self.assertEqual(b["accuracy_ci95"], [0., 1.])
        with self.assertRaises(EvaluationError):
            fit_risk_cutoffs(self.rows, self.gold, self.pred, split="final")
        with self.assertRaises(EvaluationError):
            describe(self.rows, self.gold[:-1], self.pred)
