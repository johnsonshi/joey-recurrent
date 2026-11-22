"""Isolated artificial menus only. No model, dataset, filesystem or real labels."""

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import math
import unittest

from joey_research import evaluation as ev
from joey_research import scalar_calibration as cal


def pin(raw):
    return cal.Pinned(raw, hashlib.sha256(raw).hexdigest())


def obj(v):
    return pin(json.dumps(v).encode())


def lines(rows):
    return pin(b"".join(json.dumps(r, ensure_ascii=False).encode() + b"\n" for r in rows))


class TemperatureTests(unittest.TestCase):
    def setUp(self):
        all_rows = [
            {"id": f"fixture-{task}-{g}-{i}", "group_id": f"{task}-{g}", "task": task,
             "prompt": f"Invented state {task}/{g}/{i}", "choices": ["a", "b", "c", "d", "e"]}
            for task, count in (("many", 9), ("few", 1)) for g in range(2) for i in range(count)
        ]
        self.manifest = ev.freeze_split(all_rows, seed=71, weights={"calibration": 1, "assessment": 1})
        self.roster = {"schema": "joey-temperature-roster/v1", "manifest_sha256": self.manifest["sha256"],
                       "protocol_sha256": cal.PROTOCOL_SHA256, "stock": "stock",
                       "arms": {"stock": "a"*64, "variant": "b"*64}}
        self.shared = {"manifest": self.manifest, "manifest_sha256": self.manifest["sha256"],
                       "roster": obj(self.roster)}
        self.parts = {s: ev.select_frozen_split(all_rows, self.manifest, self.manifest["sha256"], s)
                      for s in self.manifest["weights"]}
        self.fit_rows = self.parts["calibration"]
        self.labels = lines([ev.label_row(r, int(r["task"] == "few")) for r in self.fit_rows])

    def args(self, split, fn):
        shard = self.parts[split]
        inputs = lines(shard)
        arms = {}
        for name, method_id in self.roster["arms"].items():
            rows = [{"id": r["id"], "input_sha256": ev.input_sha256(r), "probs": fn(name, r)} for r in shard]
            arms[name] = cal.ArmStream(split, method_id, inputs.sha256, lines(rows[::-1]))
        return {**self.shared, "inputs": inputs, "split": split, "arms": arms}

    @staticmethod
    def probs(name, r):
        correct_or_stock = name == "stock" or r["task"] == "many"
        return [.9, .1, 0, 0, 0] if correct_or_stock else [.1, .9, 0, 0, 0]

    def fit(self, fn=None):
        return pin(cal.fit_temperatures(**self.args("calibration", fn or self.probs), labels=self.labels))

    def test_one_grid_per_arm_equal_task_not_micro_and_immutable_application(self):
        fitted = self.fit()
        f = json.loads(fitted.data)
        self.assertEqual(f["grid"], [.5, 2/3, .8, 1, 1.25, 1.5, 2., 3.])
        self.assertEqual(f["arms"]["stock"]["temperature"], 3.)
        self.assertEqual(f["arms"]["variant"]["temperature"], .5)
        for t, actual in zip(cal.GRID, f["arms"]["stock"]["macro_nll"]):
            success = 9**(1/t) / (9**(1/t) + 1)
            self.assertAlmostEqual(actual, -(math.log(success) + math.log1p(-success)) / 2)
        self.assertEqual(self.fit(), fitted)
        a = self.args("assessment", lambda _, r: [.3, .3, .25, .15, 0.])
        out = cal.apply_temperatures(fitted, **a)
        for name, r in out["arms"].items():
            self.assertEqual(r["raw_jsonl"], a["arms"][name].predictions.data)
            self.assertFalse(r["requires_decimal_ids"])
            for before, after in zip(r["raw_jsonl"].splitlines(), r["calibrated_jsonl"].splitlines()):
                p, q = json.loads(before), json.loads(after)
                self.assertEqual((p["id"], p["input_sha256"]), (q["id"], q["input_sha256"]))
                v = q["probs"]
                self.assertEqual(v[0], v[1])
                self.assertGreater(v[1], v[2])
                self.assertGreater(v[2], v[3])
                self.assertEqual(v[4], 0)
                self.assertAlmostEqual(sum(v), 1.)
                self.assertAlmostEqual(v[0] / v[2], (.3/.25)**(1/r["temperature"]))

    def test_true_zero_is_infinite_but_positive_underflowed_power_is_not(self):
        tiny = math.ulp(0.)
        fits = self.fit(lambda n, _: [1., 0 if n == "stock" else tiny, 0, 0, 0])
        f = json.loads(fits.data)["arms"]
        self.assertEqual(f["stock"]["status"], "all_objectives_infinite")
        self.assertEqual(f["stock"]["zero_gold_count"], 1)
        self.assertEqual(f["stock"]["macro_nll"], [None]*8)
        self.assertIsNone(f["stock"]["temperature"])
        self.assertTrue(all(math.isfinite(x) for x in f["variant"]["macro_nll"]))
        self.assertAlmostEqual(f["variant"]["macro_nll"][0], -math.log(tiny))
        a = self.args("assessment", self.probs)
        out = cal.apply_temperatures(fits, **a)["arms"]["stock"]
        self.assertIsNone(out["calibrated_jsonl"])
        self.assertEqual(out["raw_jsonl"], a["arms"]["stock"].predictions.data)

    def test_numerical_rank_support_and_frozen_tiebreak(self):
        fits = self.fit()
        tiny = math.ulp(0.)
        a = self.args("assessment", lambda _, r: [1., tiny, tiny, 2*tiny, 0.])
        r = cal.apply_temperatures(fits, **a)["arms"]["variant"]
        self.assertEqual(len(r["requires_decimal_ids"]), len(self.parts["assessment"]))
        for line in r["calibrated_jsonl"].splitlines():
            q = json.loads(line, parse_float=Decimal)["probs"]
            self.assertEqual(q[1], q[2])
            self.assertTrue(0 == q[4] < q[1] < q[3] < q[0])
        flat = json.loads(self.fit(lambda *_: [.25, .25, .25, .25, 0.]).data)
        self.assertEqual(flat["arms"]["stock"]["temperature"], 1.)
        for inverse, direct in ((.5, 2.), (2/3, 1.5), (.8, 1.25)):
            losses = [0 if t in (inverse, direct) else 1 for t in cal.GRID]
            self.assertEqual(cal._winner(losses), inverse)

    def test_refuses_wrong_split_crossed_groups_changed_arms_pins_and_labels(self):
        fit_args = self.args("calibration", self.probs)
        fit = self.fit()
        target = self.args("assessment", self.probs)
        with self.assertRaisesRegex(ev.EvaluationError, "calibration partition"):
            cal.fit_temperatures(**target, labels=self.labels)
        with self.assertRaisesRegex(ev.EvaluationError, "non-calibration"):
            cal.apply_temperatures(fit, **fit_args)
        with self.assertRaises(TypeError):
            cal.apply_temperatures(fit, **target, labels=self.labels)
        cases = [
            {"arms": {"variant": fit_args["arms"]["variant"]}},
            {"roster": obj({**self.roster, "stock": "unknown"})},
            {"arms": {**fit_args["arms"], "stock": fit_args["arms"]["variant"]}},
            {"arms": {**fit_args["arms"], "variant": replace(fit_args["arms"]["variant"], split="assessment")}},
            {"inputs": target["inputs"]}, {"manifest_sha256": "0"*64},
            {"labels": lines([ev.label_row(self.fit_rows[0], 0)])},
            {"labels": lines([ev.label_row({**r, "choices": r["choices"][::-1]}, 0) for r in self.fit_rows])},
        ]
        for override in cases:
            with self.assertRaises(ev.EvaluationError):
                cal.fit_temperatures(**{**fit_args, "labels": self.labels, **override})
        changed = json.loads(fit.data)
        changed["arms"]["stock"]["temperature"] = .5
        with self.assertRaisesRegex(ev.EvaluationError, "pin"):
            cal.apply_temperatures(cal.Pinned(obj(changed).data, fit.sha256), **target)
        with self.assertRaisesRegex(ev.EvaluationError, "inconsistent fitted"):
            cal.apply_temperatures(obj(changed), **target)
        bad = json.loads(json.dumps(self.manifest))
        entry = next(e for e in bad["entries"] if e["split"] == "assessment")
        entry["group_id"] = self.fit_rows[0]["group_id"]
        bad["sha256"] = ev._hash({k:v for k, v in bad.items() if k != "sha256"})
        with self.assertRaisesRegex(ev.EvaluationError, "group crossed"):
            cal.fit_temperatures(**{**fit_args, "manifest": bad, "manifest_sha256": bad["sha256"]}, labels=self.labels)

    def test_refuses_partial_or_invalid_vectors_instead_of_normalizing_or_imputing(self):
        for invalid in ([.7, .4, 0., 0., 0.], [1., -1e-15, 0., 0., 0.], [1, False, 0, 0, 0], [1., 0]):
            with self.assertRaisesRegex(ev.EvaluationError, "invalid source"):
                self.fit(lambda *_: invalid)
        a = self.args("assessment", self.probs)
        st = a["arms"]["stock"]
        for blob in (pin(st.predictions.data.splitlines()[0]), replace(st.predictions, sha256="0"*64),
                     pin(st.predictions.data.replace(b"0.1,", b"1e-2000,"))):
            with self.assertRaises(ev.EvaluationError):
                cal.apply_temperatures(self.fit(), **{**a, "arms": {**a["arms"],
                                                                    "stock":replace(st, predictions=blob)}})


if __name__ == "__main__":
    unittest.main()
