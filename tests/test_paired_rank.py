"""Paired rank checks on invented, in-memory decision menus."""
import json
import math
import unittest

from joey_research import evaluation as ev
from joey_research.paired_rank import rank_contrast


def _at_rank(rank, width, gold):
    """A strictly ordered distribution with the gold option at the requested rank."""
    weights = list(range(width, 0, -1))
    weights.insert(gold, weights.pop(rank - 1))
    total = sum(weights)
    return [w / total for w in weights]


def _streams(spec=(("a", "A", 1, 2), ("b", "A", 2, 1)), width=6):
    inputs, labels, baseline, candidate = [], [], [], []
    for i, (group, task, before, after) in enumerate(spec):
        row = {"id": f"fixture-{i}", "group_id": group, "task": task,
               "prompt": f"Choose a tile for invented board {i}.",
               "choices": [f"tile-{j}" for j in range(width)]}
        inputs.append(row)
        gold = i % width
        label = ev.label_row(row, gold)
        labels.append(label)
        pin = {"id": row["id"], "input_sha256": label["input_sha256"]}
        baseline.append({**pin, "probs": _at_rank(before, width, gold)})
        candidate.append({**pin, "probs": _at_rank(after, width, gold)})
    return inputs, labels, baseline, candidate


class PairedRankTests(unittest.TestCase):
    def test_paired_cluster_draws_change_row_support(self):
        spec = [("a", "A", 6, 1), ("b", "A", 1, 6), ("c", "A", 2, 1), ("d", "A", 1, 2),
                ("e", "B", 1, 1), ("f", "B", 3, 6), ("a", "A", 5, 3), ("c", "A", 3, 4),
                ("e", "B", 5, 4), ("a", "A", 4, 5)]
        strata = {"a": "x", "b": "x", "c": "y", "d": "y", "e": "x", "f": "x"}
        out = rank_contrast(*(reversed(s) for s in _streams(spec)), strata=strata, reps=2, seed=5)
        self.assertEqual((out["rows"], out["groups"], out["strata"]), (10, 6, 3))
        self.assertEqual(out["baseline"]["hits"], {"1": 3, "3": 6, "5": 9})
        self.assertEqual(out["candidate"]["hits"], {"1": 3, "3": 5, "5": 8})

        # Seed 5 draws (bb, cd, ee), then (aa, dd, ef): 9 and 11 rows, not 10.
        # Some gold options disappear (e.g. option 5 occurs only in omitted group f in draw 1).
        # Reference values come from expanding those rows, not averaging group rates.
        for metric, point, low, high in [
            ("top1", 0, -2/9, 0), ("top3", -1/10, -1/3, 3/11),
            ("top5", -1/10, -2/9, 1/11), ("mrr", -7/600, -11/60, 43/660),
        ]:
            with self.subTest(metric=metric):
                self.assertAlmostEqual(out["paired"][metric]["delta"], point)
                left, right = out["paired"][metric]["ci95"]
                self.assertAlmostEqual(left, (39 * low + high) / 40)
                self.assertAlmostEqual(right, (low + 39 * high) / 40)

        self.assertEqual(out["rank_buckets"], {
            "1": {"1": 1, "2": 1, "3-5": 0, "6+": 1},
            "2": {"1": 1, "2": 0, "3-5": 0, "6+": 0},
            "3-5": {"1": 0, "2": 0, "3-5": 4, "6+": 1},
            "6+": {"1": 1, "2": 0, "3-5": 0, "6+": 0},
        })
        self.assertEqual(out["change_counts"], {"improved": 4, "unchanged": 1, "worsened": 5})
        encoded = json.dumps(out, allow_nan=False)
        self.assertNotIn("fixture-", encoded)
        self.assertEqual(json.loads(encoded), out)

    def test_gold_tie_bounds_and_stable_mrr(self):
        inputs, labels, base, cand = _streams()
        labels[0]["gold_index"], labels[1]["gold_index"] = 2, 4
        base[0]["probs"] = [1/6] * 6                         # gold: third of six tied
        base[1]["probs"] = [.4, .3, .15, .05, .05, .05]     # gold: fifth, tied at 4-6
        cand[0]["probs"] = [.2, .1, .4, .1, .1, .1]          # ties exclude gold
        cand[1]["probs"] = [.4, .25, .15, .1, .075, .025]
        out = rank_contrast(inputs, labels, base, cand, strata={"a": "x", "b": "x"}, reps=0)
        arm = out["baseline"]
        self.assertEqual(arm["gold_ties"], 2)
        self.assertEqual(arm["hits"], {"1": 0, "3": 1, "5": 2})
        self.assertEqual(arm["possible_hits"], {"1": 1, "3": 1, "5": 2})
        self.assertEqual(arm["guaranteed_hits"], {"1": 0, "3": 0, "5": 0})
        self.assertAlmostEqual(arm["mrr"], 4/15)
        self.assertAlmostEqual(arm["mrr_best"], 5/8)
        self.assertAlmostEqual(arm["mrr_worst"], 1/6)
        arm = out["candidate"]
        self.assertEqual(arm["gold_ties"], 0)
        self.assertEqual(arm["guaranteed_hits"], arm["hits"])
        self.assertEqual(arm["possible_hits"], arm["hits"])
        self.assertAlmostEqual(arm["mrr"], 3/5)
        self.assertTrue(all(m["ci95"] is None for m in out["paired"].values()))

    def test_five_choices_and_raw_adjacent_probabilities(self):
        inputs, labels, base, cand = _streams([("a", "A", 1, 1)], width=5)
        labels[0]["gold_index"] = 1
        base[0]["probs"] = [.25, .25, .1875, .1875, .125]
        # These two leading floats are distinct but become equal if divided by the total.
        cand[0]["probs"] = [math.nextafter(.25, 0), .25, .1875, .1875, .125 - 2**-52]
        out = rank_contrast(inputs, labels, base, cand, strata={"a": "x"}, reps=0)
        self.assertEqual(out["baseline"]["gold_ties"], 1)
        self.assertEqual(out["candidate"]["gold_ties"], 0)
        self.assertEqual(out["rank_buckets"]["2"]["1"], 1)
        self.assertEqual(out["paired"]["mrr"]["delta"], .5)

    def test_one_singleton_cell_withholds_every_interval(self):
        spec = [("a", "A", 2, 1), ("b", "A", 2, 1), ("c", "B", 2, 1), ("c", "B", 2, 1)]
        out = rank_contrast(*_streams(spec), strata=dict.fromkeys("abc", "same"), reps=9)
        self.assertEqual(out["strata"], 2)
        self.assertEqual(out["paired"]["top1"]["delta"], 1)
        for metric in ("top1", "top3", "top5", "mrr"):
            self.assertIsNone(out["paired"][metric]["ci95"])

    def test_exact_order_length_uniqueness_and_pins(self):
        strata = {"a": "x", "b": "x"}
        for i in range(4):
            for change in ("reverse", "shorten"):
                with self.subTest(stream=i, change=change), self.assertRaises(ev.EvaluationError):
                    streams = _streams()
                    if change == "reverse":
                        streams[i].reverse()
                    else:
                        streams[i].pop()
                    rank_contrast(*streams, strata=strata)
        for i in (1, 2, 3):
            with self.subTest(stale=i), self.assertRaises(ev.EvaluationError):
                streams = _streams()
                streams[i][0]["input_sha256"] = streams[i][1]["input_sha256"]
                rank_contrast(*streams, strata=strata)
        duplicated = ([s[0], s[0]] for s in _streams())
        with self.assertRaises(ev.EvaluationError):
            rank_contrast(*duplicated, strata={"a": "x"})

    def test_only_exact_schemas_and_valid_gold(self):
        strata = {"a": "x", "b": "x"}
        for i in range(4):
            with self.subTest(extra_key=i), self.assertRaises(ev.EvaluationError):
                streams = _streams()
                streams[i][0]["choice_index"] = 0
                rank_contrast(*streams, strata=strata)
        for i in (2, 3):
            with self.subTest(projected_rank=i), self.assertRaises(ev.EvaluationError):
                streams = _streams()
                del streams[i][0]["probs"]
                streams[i][0]["ranks"] = list(range(1, 7))
                rank_contrast(*streams, strata=strata)
        for gold in (-1, 6, True, 1.0):
            with self.subTest(gold=gold), self.assertRaises(ev.EvaluationError):
                streams = _streams()
                streams[1][0]["gold_index"] = gold
                rank_contrast(*streams, strata=strata)

    def test_whole_menu_distribution_and_tolerance(self):
        strata = {"a": "x", "b": "x"}
        for i in (2, 3):
            for probs in ([1, 0], [0] * 5, [True, 0, 0, 0, 0], ["1", 0, 0, 0, 0],
                          [math.nan, 0, 0, 0, 0], [math.inf, 0, 0, 0, 0], [1, -1e-9, 0, 0, 0],
                          [1 + 1e-9, 0, 0, 0, 0], [1 - 2 * ev.PROB_SUM_ATOL, 0, 0, 0, 0]):
                with self.subTest(arm=i, probs=probs), self.assertRaises(ev.EvaluationError):
                    streams = _streams(width=5)
                    streams[i][0]["probs"] = probs
                    rank_contrast(*streams, strata=strata, reps=0)
        streams = _streams(width=5)
        streams[3][0]["probs"] = [1 - ev.PROB_SUM_ATOL / 2, 0, 0, 0, 0]
        out = rank_contrast(*streams, strata=strata, reps=0)
        self.assertEqual(out["candidate"]["hits"]["1"], 2)

    def test_fixed_menu_and_exact_strata(self):
        streams = _streams()
        for strata in ({"a": "x"}, {"a": "x", "b": "x", "c": "x"}, {"a": 1, "b": "x"},
                       {"a": " ", "b": "x"}, ["x", "x"]):
            with self.subTest(strata=strata), self.assertRaises(ev.EvaluationError):
                rank_contrast(*streams, strata=strata)
        with self.assertRaises(ev.EvaluationError):
            rank_contrast(*_streams([("a", "A", 1, 2), ("a", "B", 2, 1)]), strata={"a": "x"})
        with self.assertRaises(ev.EvaluationError):
            rank_contrast(*_streams(width=4), strata={"a": "x", "b": "x"})
        streams[0][1]["choices"].reverse()
        for stream in streams[1:]:
            stream[1]["input_sha256"] = ev.input_sha256(streams[0][1])
        with self.assertRaises(ev.EvaluationError):
            rank_contrast(*streams, strata={"a": "x", "b": "x"})


if __name__ == "__main__":
    unittest.main()
