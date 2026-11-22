import unittest

from benchmarks.framing_contrast import choice_contrasts
from joey_research.evaluation import EvaluationError, input_sha256, label_row


class FramingContrast(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": str(i), "group_id": str(i // 2), "task": "toy", "prompt": f"Parcel {i}",
                      "choices": ["north", "south", "west"]} for i in range(4)]
        self.gold = [label_row(r, i % 2) for i, r in enumerate(self.rows)]

    def choices(self, *indices):
        return [{"id": r["id"], "input_sha256": input_sha256(r), "choice_index": j}
                for r, j in zip(self.rows, indices, strict=True)]

    def test_common_group_draws_for_framing_interaction(self):
        arms = {"A": self.choices(0, 1, 2, 2), "B": self.choices(0, 1, 0, 1),
                "C": self.choices(0, 1, 2, 2), "D": self.choices(2, 2, 2, 2)}
        contrasts = {"gain": {"B": 1, "A": -1}, "interaction": {"D": 1, "C": -1, "B": -1, "A": 1}}
        result = choice_contrasts(self.rows, self.gold, arms, contrasts, reps=100, seed=2)
        gain, interaction = (result["contrasts"][k] for k in ("gain", "interaction"))
        self.assertEqual(gain["corrected"], 2)
        self.assertEqual(gain["corrupted"], 0)
        self.assertEqual(gain["accuracy"]["ci95"], [0., 1.])
        self.assertAlmostEqual(gain["observed_class_macro_f1"]["delta"], 1 - 4/9)
        # Each graph contributes exactly -1 to the four-arm contrast, despite variable two-arm effects.
        self.assertEqual(interaction["accuracy"], {"delta": -1., "ci95": [-1., -1.], "status": "ok"})
        self.assertEqual(result["arms"]["A"]["classes_scored"], 3)
        self.assertEqual(result["arms"]["B"]["classes_scored"], 2)

    def test_refuses_stale_missing_and_unpaired_choices(self):
        p = self.choices(0, 1, 0, 1)
        for bad in (p[:-1], [dict(p[0], input_sha256="a"*64), *p[1:]], [dict(p[0], choice_index=True), *p[1:]]):
            with self.assertRaises(EvaluationError):
                choice_contrasts(self.rows, self.gold, {"a": p, "b": bad}, {"d": {"a": -1, "b": 1}}, reps=0)
        with self.assertRaises(EvaluationError):
            choice_contrasts(self.rows, self.gold, {"a": p, "b": p}, {"d": {"a": 1, "b": 1}})
        r = choice_contrasts(self.rows[:2], self.gold[:2], {"a": p[:2], "b": p[:2]}, {"d": {"a": -1, "b": 1}})
        self.assertEqual(r["contrasts"]["d"]["accuracy"]["status"], "insufficient_independent_groups")
