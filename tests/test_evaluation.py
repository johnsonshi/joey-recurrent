"""Hand-built or namespace-isolated fixtures only. No external or campaign examples."""

import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unittest

from benchmarks import mind2web, routing
from joey_research import evaluation as ev


def row(i, group=None, task="fixture", k=2):
    return {"id": f"fx-{i}", "group_id": group or f"fg-{i}", "task": task,
            "prompt": f"Fixture state {i}", "choices": [f"option {j}" for j in range(k)]}


def pred(r, p):
    data = {"id": r["id"], "input_sha256": ev.input_sha256(r)}
    return {**data, **({"error": "fixture_error"} if p is None else {"probs": p})}


def report(rows, gold, arms, **options):
    options.setdefault("bootstrap_reps", 0)
    labels = [ev.label_row(r, g) for r, g in zip(rows, gold)]
    return ev.evaluate(rows, labels, {k: [pred(r, p) for r, p in zip(rows, ps)]
                                      for k, ps in arms.items()}, baseline=next(iter(arms)), **options)


class MetricTests(unittest.TestCase):
    def test_scoring_denominators_bins_macro_and_risk_blocks(self):
        rows = [row(i, task="one" if i < 3 else "two", k=3) for i in range(4)]
        ps = [[.8, .1, .1], [.8, .1, .1], [.1, .8, .1], [.3, .4, .3]]
        arm = report(rows, [0, 2, 1, 0], {"p1": ps})["arms"]["p1"]
        micro, macro = arm["micro"], arm["macro"]
        self.assertEqual(micro["correct"], 2)
        self.assertEqual(micro["accuracy"], .5)
        self.assertAlmostEqual(macro["accuracy"], 1 / 3)
        self.assertAlmostEqual(micro["nll"], -(2*math.log(.8)+math.log(.1)+math.log(.3))/4)
        self.assertAlmostEqual(micro["brier"], (.06 + 1.46 + .06 + .74) / 4)
        self.assertEqual([b["n"] for b in micro["ece_bins_valid_only"]],
                         [0]*6 + [1] + [0]*5 + [3, 0, 0])
        self.assertAlmostEqual(micro["ece15"], .2)
        self.assertAlmostEqual(macro["ece15"], (abs(2/3-.8) + .4) / 2)
        self.assertEqual((micro["confident_wrong"], micro["confident"], micro["n"]), (1, 3, 4))
        self.assertEqual(micro["confident_error_joint"], .25)
        self.assertAlmostEqual(micro["confident_error_conditional"], 1/3)
        self.assertIsNone(macro["confident_error_conditional"])
        self.assertEqual([(p["accepted"], p["coverage"], p["risk"]) for p in micro["risk_coverage"]],
                         [(0, 0, None), (3, .75, 1/3), (4, 1, .5)])
        self.assertAlmostEqual(micro["aurc"], .375)

    def test_argmax_tie_and_true_zero_gold_probability(self):
        rows = [row(i) for i in range(2)]
        micro = report(rows, [0, 0], {"p": [[.5, .5], [0., 1.]]})["arms"]["p"]["micro"]
        self.assertEqual((micro["argmax_ties"], micro["correct"]), (1, 1))
        self.assertEqual(micro["zero_gold_probability"], 1)
        self.assertIsNone(micro["nll"])
        self.assertEqual(micro["nll_status"], "infinite")
        self.assertEqual(micro["brier"], (0.5 + 2) / 2)
        self.assertEqual(micro["ece_bins_valid_only"][-1]["n"], 1)
        json.dumps(micro, allow_nan=False)

    def test_near_boundary_stays_in_its_bin(self):
        bound = 8/15
        lo = math.nextafter(bound, -math.inf)
        probs = [[lo, 1-lo], [bound, 1-bound], [.60000001, .4]]
        stats = report([row(i) for i in range(3)], [0, 0, 0], {"p": probs})["arms"]["p"]["micro"]
        bins = stats["ece_bins_valid_only"]
        self.assertEqual((bins[7]["n"], bins[8]["n"], bins[9]["n"]), (1, 1, 1))
        self.assertAlmostEqual(bins[9]["mean_confidence"], .60000001 / 1.00000001)
        self.assertEqual(stats["invalid"], 0)

    def test_invalid_outputs_not_dropped_or_replaced_by_uniform(self):
        probs = [None, [math.nan, 0], [math.inf, 0], [-.1, 1.1], [1], [False, 1],
                 ["0", 1], [1, 1], [0, 0], [10**3000, 0], [.8, .2]]
        result = report([row(i) for i in range(len(probs))], [1]*len(probs), {"p": probs})
        m = result["arms"]["p"]["micro"]
        self.assertFalse(result["all_distributions_valid"])
        self.assertEqual((m["invalid"], m["valid"], m["correct"]), (10, 1, 0))
        self.assertEqual(m["invalid_reasons"], {"runner_error": 1, "wrong_shape": 1,
                                                "not_normalized": 2, "nonfinite_or_out_of_range": 6})
        self.assertIsNone(m["nll"])
        self.assertEqual(m["nll_status"], "invalid_distribution")
        self.assertIsNone(m["brier"])
        self.assertIsNone(m["ece15"])
        self.assertEqual(m["confident_error_joint"], 1/11)
        self.assertEqual(m["confident_error_conditional"], 1.)
        self.assertEqual(m["risk_coverage"][-1]["coverage"], 1/11)
        self.assertEqual(m["aurc_observed_area"], 1/11)
        self.assertIsNone(m["aurc"])
        json.dumps(result, allow_nan=False)

    def test_paired_id_hash_strictness(self):
        rows = [row(1), row(2)]
        gold = [ev.label_row(r, 0) for r in rows]
        probs = [pred(rows[0], [.8, .2]), pred(rows[1], [.3, .7])]
        def evaluate(rs, gs, ps):
            return ev.evaluate(rs, gs, {"p": ps}, baseline="p", bootstrap_reps=0)
        self.assertEqual(evaluate(rows, gold, probs), evaluate(rows[::-1], gold[::-1], probs[::-1]))
        for rs, gs, ps in [
            (rows, gold, probs[:1]), (rows, gold, probs + probs[:1]), (rows, gold[:1], probs),
            (rows, gold, [{**probs[0], "id": "other"}, probs[1]]),
            (rows, [gold[1], gold[1]], probs),
            (rows, [{**gold[0], "gold_index": True}, gold[1]], probs),
            (rows, [{**gold[0], "gold_index": 2}, gold[1]], probs),
            (rows, [{**gold[0], "input_sha256": "old"}, gold[1]], probs),
            (rows, gold, [dict(probs[0], input_sha256="old"), probs[1]]),
            (rows, gold, [dict(probs[0], error="ambiguous"), probs[1]]),
            ([dict(rows[0], choices=rows[0]["choices"][::-1]), rows[1]], gold, probs),
            ([dict(rows[0], gold_index=0), rows[1]], gold, probs),
            ([rows[0], rows[0]], gold, probs),
        ]:
            with self.subTest(rs=rs, gs=gs, ps=ps), self.assertRaises(ev.EvaluationError):
                evaluate(rs, gs, ps)

    def test_transitions_do_not_count_invalid_recovery_as_correction(self):
        rows = [row(i) for i in range(5)]
        p1 = [[.8, .2], [.4, .6], [.45, .55], [.4, .6], None]
        p2 = [[.3, .7], [.7, .3], [.25, .75], [.2, .8], [.9, .1]]
        p3 = [[.1, .9], [.2, .8], [.6, .4], [.05, .95], [.9, .1]]
        path = report(rows, [0]*5, {"p1": p1, "p2": p2, "p3": p3},
                      pass_series=["p1", "p2", "p3"])["passes"]
        t = path["first_to_last"]
        self.assertEqual((t["valid_pairs"], t["corrections"], t["corruptions"]), (4, 1, 1))
        self.assertEqual(t["correction_given_wrong"], 1/3)
        self.assertEqual(t["corruption_given_correct"], 1)
        self.assertEqual(t["delta_accuracy_valid_pairs"], 0)
        self.assertAlmostEqual(t["confidence_inflation_gap"], .175)
        self.assertAlmostEqual(t["persistent_wrong_delta_top_confidence"], .275)
        self.assertEqual(t["persistent_wrong_confidence_rises"], 2)
        self.assertEqual((path["invalid_paths"], path["overthought"], path["previously_correct"],
                          path["final_wrong"]), (1, 2, 2, 3))
        self.assertEqual(path["overthinking_given_final_wrong"], 2/3)
        self.assertEqual(path["adjacent"][1]["valid_pairs"], 5)

    def test_group_bootstrap_keeps_whole_clusters_and_pairs_within_task(self):
        rows = [row(i, "a0" if i < 3 else "a1", "a") for i in range(4)]
        rows += [row(4, "b0", "b"), row(5, "b1", "b")]
        yes, no = [.8, .2], [.3, .7]
        a = [no, no, no, yes, yes, no]
        b = [yes, yes, yes, no, yes, no]
        result = report(rows, [0]*6, {"a": a, "b": b, "same": a}, bootstrap_reps=300)
        delta = result["paired_deltas"]["b"]
        self.assertEqual(delta["groups_per_task"], {"a": 2, "b": 2})
        self.assertAlmostEqual(delta["micro"]["accuracy"]["delta"], 1/3)
        self.assertEqual(delta["micro"]["accuracy"]["ci95"], [-.5, .75])
        self.assertEqual(delta["macro"]["accuracy"]["delta"], .25)
        self.assertEqual(delta["macro"]["accuracy"]["ci95"], [-.5, .5])
        same = result["paired_deltas"]["same"]
        for name in ["accuracy", "nll", "brier", "ece15"]:
            self.assertEqual(same["micro"][name]["ci95"], [0., 0.])
        self.assertIsNone(delta["macro"]["confident_error_conditional"]["ci95"])
        reverse = report(rows[::-1], [0]*6, {"a": a[::-1], "b": b[::-1]}, bootstrap_reps=300)
        self.assertEqual(delta, reverse["paired_deltas"]["b"])

    def test_bootstrap_recomputes_pooled_nonlinear_metrics_from_same_draws(self):
        rows = [row(0, "a0", "a"), row(1, "a0", "a"), row(2, "a1", "a"), row(3, "a2", "a")]
        left = [[.7, .3], [.2, .8], [.1, .9], [.98, .02]]
        right = [[.3, .7], [.1, .9], [.99, .01], [.05, .95]]
        labels = [0, 1, 1, 1]
        comp = report(rows, labels, {"l": left, "r": right}, bootstrap_reps=99,
                      bootstrap_seed=12345)["paired_deltas"]["r"]["micro"]
        rng = random.Random(12345)
        reference = collections.defaultdict(list)
        for _ in range(99):
            indices = [i for _ in range(3) for i in [[0, 1], [2], [3]][rng.randrange(3)]]
            for key in ("nll", "brier", "ece15"):
                pair = []
                for probs in (left, right):
                    bins = collections.defaultdict(float)
                    val = 0
                    for i in indices:
                        p, y = probs[i], labels[i]
                        if key == "nll":
                            val -= math.log(p[y])
                        elif key == "brier":
                            val += sum((j == y) * (1-q)**2 + (j != y) * q**2
                                       for j, q in enumerate(p))
                        else:
                            h = p.index(max(p))
                            bins[min(14, math.floor(15*p[h]))] += (h == y) - p[h]
                    pair.append((val + sum(map(abs, bins.values()))) / len(indices))
                reference[key].append(pair[1] - pair[0])
        for key, values in reference.items():
            values.sort()
            for actual, q in zip(comp[key]["ci95"], (.025, .975)):
                ix = (len(values)-1)*q
                expected = values[int(ix)] + (ix % 1)*(values[math.ceil(ix)] - values[int(ix)])
                self.assertAlmostEqual(actual, expected)

    def test_confidence_and_bootstrap_support_fail_closed(self):
        rows = [row(0, "one"), row(1, "one")]
        ps = [[.55, .45], [.45, .55]]
        comp = report(rows, [0, 0], {"a": ps, "b": ps}, bootstrap_reps=10)
        self.assertIsNone(comp["arms"]["a"]["micro"]["confident_error_conditional"])
        self.assertEqual(comp["paired_deltas"]["b"]["micro"]["accuracy"]["status"],
                         "insufficient_independent_groups")
        with self.assertRaises(ev.EvaluationError):
            report(rows, [0, 0], {"a": ps}, threshold=math.nan)


class SplitTests(unittest.TestCase):
    def test_split_is_label_blind_group_atomic_and_digest_pinned(self):
        rows = [row(f"{t}-{g}-{v}", f"{t}-{g}", t) for t in ("x", "y")
                for g in range(8) for v in range(1 + int(g % 3 == 1))]
        weights = {"selection": 1, "sealed": 1, "discovery": 2}
        m = ev.freeze_split(rows, seed=9, weights=weights)
        self.assertEqual(m, ev.freeze_split(rows[::-1], seed=9, weights=dict(reversed(weights.items()))))
        seen = {}
        for entry in m["entries"]:
            split = entry["split"]
            self.assertEqual(seen.setdefault(entry["group_id"], split), split)
            self.assertNotIn("gold_index", entry)
        for t in ("x", "y"):
            self.assertEqual(collections.Counter(s for g, s in seen.items() if g.startswith(t)),
                             {"discovery": 4, "selection": 2, "sealed": 2})
        selected = ev.select_frozen_split(rows, m, m["sha256"], "selection")
        self.assertEqual({r["id"] for r in selected}, {e["id"] for e in m["entries"]
                                                      if e["split"] == "selection"})
        modified = copy.deepcopy(m)
        modified["entries"][0]["split"] = "modified"
        for universe, manifest, pin in [
            (rows, modified, m["sha256"]), (rows, m, "wrong"), (rows[:-1], m, m["sha256"]),
            ([dict(rows[0], choices=rows[0]["choices"][::-1]), *rows[1:]], m, m["sha256"]),
        ]:
            with self.subTest(pin=pin), self.assertRaises(ev.EvaluationError):
                ev.select_frozen_split(universe, manifest, pin, "selection")

    def test_group_leak_and_undersized_strata_rejected(self):
        r = row(0)
        for rs in [
            [r, {**r, "id": "other", "group_id": "other", "choices": r["choices"][::-1]}],
            [r, {**r, "id": "other", "task": "different"}],
            [{**r, "gold_index": 0}],
            [r, {**r, "id": "other", "choices": ["one  option", "one option"]}],
        ]:
            with self.subTest(rs=rs), self.assertRaises(ev.EvaluationError):
                ev.freeze_split(rs, seed=1, weights={"x": 1})
        with self.assertRaises(ev.EvaluationError):
            ev.freeze_split([r], seed=1, weights={"x": 1, "y": 1})


class RoutingTests(unittest.TestCase):
    def test_forward_oracle_inverse_verifier_and_within_group_counterbalance(self):
        kwargs = dict(namespace="fixture-routing-only", seed=708041, depths=[2, 4], groups_per_depth=4)
        items = list(routing.iter_inputs(**kwargs))
        self.assertEqual(items, list(routing.iter_inputs(**kwargs)))
        balances = collections.defaultdict(collections.Counter)
        for r in items:
            pred_label = routing.inverse_label(r)
            header, _, *lines = r["prompt"].splitlines()
            dest = re.search(r"(?:finish at|destination:) ([A-Z]{3})", header).group(1)
            successor = {}
            for line in lines:
                words = line.split()
                if len(words) == 3:
                    successor[words[0]] = words[-1]
                else:
                    successor[words[-1]] = words[0]
            ends = []
            for option in r["choices"]:
                loc = option.split()[-1]
                length = 0
                while loc in successor:
                    loc = successor[loc]
                    length += 1
                self.assertEqual(r["task"], f"routing_{length}hop")
                ends.append(loc)
            i = pred_label["gold_index"]
            self.assertEqual(ends.index(dest), i)
            self.assertEqual(pred_label["input_sha256"], ev.input_sha256(r))
            self.assertEqual(routing.inverse_label(dict(r, choices=r["choices"][::-1]))["gold_index"], 3-i)
            balances[r["group_id"], header.startswith("Required")][i] += 1
        self.assertEqual(len(balances), 16)
        self.assertTrue(all(c == {0: 1, 1: 1, 2: 1, 3: 1} for c in balances.values()))
        new_items = routing.iter_inputs(**{**kwargs, "namespace": "other-fixture-only"})
        self.assertFalse({r["id"] for r in items} & {r["id"] for r in new_items})

    def test_verifier_changes_answer_when_only_final_handoffs_swap(self):
        r = row("oracle")
        header = ("A parcel must finish at CCC. Choose the entry station that sends it there.\n"
                  "Each line is one directed handoff. Line order does not set travel order.")
        r.update(prompt=header + "\nBBB -> CCC\nXXX -> YYY\nAAA -> BBB\nYYY -> ZZZ",
                 choices=["Send to XXX", "Send to AAA"])
        self.assertEqual(routing.inverse_label(r)["gold_index"], 1)
        r["prompt"] = header + "\nBBB -> ZZZ\nXXX -> YYY\nAAA -> BBB\nYYY -> CCC"
        self.assertEqual(routing.inverse_label(r)["gold_index"], 0)
        for bad in [r["prompt"] + "\nCCC -> KKK", r["prompt"] + "\nAAA -> KKK",
                    r["prompt"] + "\nMMM -> NNN\nNNN -> MMM", header + "\nAAA -> CCC\nXXX -> ZZZ"]:
            with self.subTest(bad=bad), self.assertRaises(ev.EvaluationError):
                routing.inverse_label(dict(r, prompt=bad))


class SourceProjectionTests(unittest.TestCase):
    def setUp(self):
        self.step = {
            "action_uid": "fixture-step",
            "cleaned_html": '<html><button backend_node_id="91"><text>Find</text></button>'
                '<input backend_node_id="15" aria-label="Start" value="do-not-expose"/>'
                '<a backend_node_id="17">Duplicate</a><a backend_node_id="18">Duplicate</a>'
                '<div hidden=""><button backend_node_id="33">secret</button></div></html>',
            "pos_candidates": [{"backend_node_id": "91"}], "neg_candidates": [],
            "operation": {"op": "CLICK", "value": "do-not-expose"},
        }
        self.trajectory = {"annotation_id": "fixture-trajectory", "confirmed_task": "Find an item",
                           "action_reprs": ["do-not-expose"], "actions": [self.step]}

    def test_blind_menu_works_without_annotation_or_future_fields(self):
        full = mind2web.project_step(self.trajectory, 0)
        stripped = {"annotation_id": self.trajectory["annotation_id"],
                    "confirmed_task": self.trajectory["confirmed_task"],
                    "actions": [{"cleaned_html": self.step["cleaned_html"],
                                 "action_uid": self.step["action_uid"]}]}
        self.assertEqual(full, mind2web.project_step(stripped, 0))
        self.assertEqual(set(full.input["choices"]), {"button: Find", "input: Start",
                                                      "None of the listed elements"})
        self.assertEqual(full.discarded_ambiguous_elements, 2)
        self.assertEqual(full.unique_named_elements, 2)
        label = mind2web.adjudicate_step(self.trajectory, 0)
        self.assertEqual(full.node_ids[label["gold_index"]], "91")
        self.assertEqual(label["input_sha256"], ev.input_sha256(full.input))

    def test_source_ambiguity_is_not_picked_away_or_retrieved_away(self):
        original = mind2web.project_step(self.trajectory, 0)
        self.step["pos_candidates"] = [{"backend_node_id": "91"}, {"backend_node_id": "15"}]
        with self.assertRaises(mind2web.UnscorableStep):
            mind2web.adjudicate_step(self.trajectory, 0)
        self.step["pos_candidates"] = []
        with self.assertRaises(mind2web.UnscorableStep):
            mind2web.adjudicate_step(self.trajectory, 0)
        self.step["pos_candidates"] = [{"backend_node_id": "17"}]
        not_listed = mind2web.adjudicate_step(self.trajectory, 0)
        self.assertIsNone(original.node_ids[not_listed["gold_index"]])
        self.assertEqual(original, mind2web.project_step(self.trajectory, 0))
        self.step["cleaned_html"] = "<html/>"
        with self.assertRaises(mind2web.UnscorableStep):
            mind2web.project_step(self.trajectory, 0)


class CliTests(unittest.TestCase):
    def test_score_selection_only_with_strict_file_joins(self):
        kwargs = dict(namespace="fixture-cli-only", seed=408106, groups_per_depth=8, depths=[2])
        all_rows = list(routing.iter_inputs(**kwargs))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            tmp = Path(tmp)
            def dump(name, rows):
                path = tmp / name
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                return str(path)
            common = ["--inputs", dump("inputs", all_rows), "--config", "configs/evaluation/protocol.json"]
            def cli(*args):
                return subprocess.run([sys.executable, "-B", "-m", "joey_research.evaluation", *args],
                                      text=True, capture_output=True,
                                      env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            frozen = cli("freeze", *common)
            self.assertEqual(frozen.returncode, 0, frozen.stderr)
            m = json.loads(frozen.stdout)
            manifest = tmp / "manifest"
            manifest.write_text(frozen.stdout, encoding="utf-8")
            sel = ev.select_frozen_split(all_rows, m, m["sha256"], "selection")
            labels = [routing.inverse_label(r) for r in sel]
            uniform = [pred(r, [.25]*4) for r in sel]
            score_args = ["score", *common, "--manifest", str(manifest), "--manifest-sha256", m["sha256"],
                          "--split", "selection", "--labels", dump("labels", labels),
                          "--arm", "uniform=" + dump("uniform", uniform), "--baseline", "uniform"]
            scored = cli(*score_args)
            self.assertEqual(scored.returncode, 0, scored.stderr)
            self.assertEqual(json.loads(scored.stdout)["arms"]["uniform"]["micro"]["accuracy"], .25)
            changed = tmp / "changed.json"
            cfg = json.loads(Path(common[-1]).read_text(encoding="utf-8"))
            changed.write_text(json.dumps({**cfg, "threshold": .9}), encoding="utf-8")
            incompatible = cli(*score_args, "--config", str(changed))
            self.assertNotEqual(incompatible.returncode, 0)
            self.assertIn("frozen protocol", incompatible.stderr)
            dump("labels", labels[:-1])
            self.assertNotEqual(cli(*score_args).returncode, 0)
            # No final labels are generated or opened, even in this fixture.
            gate = cli(*score_args, "--split", "final", "--labels", str(tmp / "never-open"))
            self.assertNotEqual(gate.returncode, 0)
            self.assertIn("pinned finalists receipt", gate.stderr)

    def test_finalists_receipt_bound_before_label_access(self):
        manifest = {"sha256": "1" * 64}
        receipt = {"schema": "joey-finalists-v1", "baseline": "p1", "passes": ["p1", "p2"],
                   "arms": {"p1": "2" * 64, "p2": "3" * 64}, "primary": "macro.accuracy",
                   "manifest_sha256": manifest["sha256"]}
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            p = Path(tmp) / "fixture-receipt"
            p.write_text(json.dumps(receipt), encoding="utf-8")
            pin = hashlib.sha256(p.read_bytes()).hexdigest()
            args = (str(p), pin, manifest, {"p1", "p2"}, "p1", ["p1", "p2"])
            ev._check_finalists(*args)
            for bad in [
                (str(p), "4" * 64, *args[2:]),
                (*args[:3], {"p1", "p3"}, *args[4:]),
                (*args[:4], "p2", args[5]),
                (*args[:5], ["p2", "p1"]),
            ]:
                with self.subTest(bad=bad), self.assertRaises(ev.EvaluationError):
                    ev._check_finalists(*bad)

    def test_duplicate_json_fields_and_nonstandard_constants_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            p = Path(tmp) / "malformed"
            for data in ['{"probs": [NaN, 0]}', '{"id": "a", "id": "b"}', "[]", "\n"]:
                p.write_text(data, encoding="utf-8")
                with self.subTest(data=data), self.assertRaises(ev.EvaluationError):
                    list(ev.read_jsonl(p))


if __name__ == "__main__":
    unittest.main()
