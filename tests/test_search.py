"""Synthetic branch-decision fixtures, not estimates of any model's quality."""

import hashlib
import json
import math
import unittest
from dataclasses import asdict, replace
from fractions import Fraction

from joey_research.search import (
    Arm, Band, Candidate, DevRun, Item, MeasuredCost, PPM, Paired, Plan, PolicyError,
    Rates, Rung, Search,
)
from joey_research.scheduler import TrialKey


def h(s):
    return hashlib.sha256(s.encode("ascii")).hexdigest()


IDS = tuple(h(f"member{i}") for i in range(12))
STOCK = Arm(h("stock"), "stock", h("full"), 1)


def make_plan(n=1, rungs=None):
    candidates = []
    for i in range(n):
        loop = Arm(h(f"loop{i}"), "loop", STOCK.payload_sha, i + 2)
        compact = Arm(h(f"compact{i}"), "compact", h(f"payload{i}"), 2)
        control = Arm(h(f"k1-{i}"), "compact", compact.payload_sha, 1)
        candidates.extend((Candidate(loop, STOCK, priority=i), Candidate(compact, STOCK, control, i)))
    return Plan(h("study"), h("interval-protocol"), tuple(candidates),
                (Rung(IDS[:8], 2),) if rungs is None else rungs, 800_000, "layer_millicalls", 0)


def item(member, correct, high):
    return Item(member, correct, 900_000 if high else 600_000,
                (20_000 if correct else 1_620_000) if high else (320_000 if correct else 720_000))


def exact(a, b):
    # Test-only boundary fixtures; no uncertainty estimator is simulated by production code.
    if a is None or b is None:
        return None
    value = (a - b) * PPM
    return Band(math.floor(value), math.ceil(value))


def inputs(search, r=0, modify=None):
    plan, arms = search.plan, {}
    for c in plan.candidates:
        if c.arm.id in search.active:
            for a in (c.arm, *c.controls):
                arms[a.id] = a
    rung = plan.rungs[r]
    runs = {}
    for a in arms.values():
        rows = tuple(item(i, IDS.index(i) % 4 != 0, IDS.index(i) % 4 in (1, 2)) for i in rung.ids)
        cost = MeasuredCost(100, 1000, 2000, 50 if a.kind == "compact" else 100, h("cost" + a.id))
        run = DevRun("dev", plan.study_sha, a, h(f"trial:{r}:{a.id}"), h(f"result:{r}:{a.id}"),
                     h(f"validated:{r}:{a.id}"), cost, rows)
        runs[a.id] = modify(run) if modify else run
    pairs = {}
    for c in plan.candidates:
        if c.arm.id not in search.active:
            continue
        for control in c.controls:
            a, b = runs[c.arm.id], runs[control.id]
            ar, br = (Rates.from_run(x, plan.confidence_ppm) for x in (a, b))
            values = {k: exact(getattr(ar, k), getattr(br, k))
                      for k in ("accuracy", "brier", "joint_error", "conditional_error", "coverage")}
            pairs[(a.arm.id, b.arm.id)] = Paired("dev", plan.study_sha, plan.method_sha, rung.membership_sha,
                                                 a.result_sha, b.result_sha, h(a.arm.id + b.arm.id), **values)
    return runs, pairs


def candidates(plan, branch):
    return [c for c in plan.candidates if c.arm.kind == branch]


class SearchTests(unittest.TestCase):
    def test_two_distinct_initial_lanes_and_predeclared_limits(self):
        p = make_plan(n=3)
        q = Search(p)
        self.assertEqual(q.initial_slots(), (h("loop0"), h("compact0")))
        self.assertEqual(q.branch_wave(completed=[h("loop0")]), (h("loop1"), h("compact0")))
        self.assertEqual(q.branch_wave(completed=[c.arm.id for c in candidates(p, "loop")]),
                         (None, h("compact0")))
        with self.assertRaises(PolicyError):
            replace(p, candidates=tuple(candidates(p, "loop")))
        with self.assertRaises(PolicyError):
            replace(p, rungs=(Rung(IDS[:8], 3),))
        with self.assertRaises(PolicyError):
            replace(p, rungs=(Rung(IDS[:4], 1), Rung(IDS[:8], 2)))
        c = candidates(p, "compact")[0]
        for broken in (None, replace(c.recovery, k=2), replace(c.recovery, payload_sha=h("different"))):
            with self.assertRaises(PolicyError):
                replace(c, recovery=broken)

    def test_only_successful_validated_dev_runs_and_dev_intervals(self):
        q = Search(make_plan())
        runs, pairs = inputs(q)
        run, pair = next(iter(runs.values())), next(iter(pairs.values()))
        args = dict(study_sha=run.study_sha, arm=run.arm, cost=run.cost, items=run.items,
                    validation_sha=run.validation_sha)
        key = TrialKey(h("model"), h("dev-members"), h("code"), run.arm.id, 1, h("scoring"), "p32")
        row = dict(id=key.trial_id, key=asdict(key), state="succeeded", result_sha=run.result_sha, split="dev")
        self.assertEqual(DevRun.from_registry(row, **args), replace(run, dev_id=key.trial_id))
        with self.assertRaises(PolicyError):
            DevRun.from_registry({**row, "id": h("different-trial")}, **args)
        with self.assertRaises(PolicyError):
            DevRun.from_registry(row, **{**args, "arm": replace(run.arm, id=h("substituted-arm"))})
        for split in ("test", "calibration", "", "train"):
            with self.assertRaises(PolicyError):
                replace(run, split=split)
            with self.assertRaises(PolicyError):
                replace(pair, split=split)
            with self.assertRaises(PolicyError):
                DevRun.from_registry({**row, "split": split}, **args)
        with self.assertRaises(PolicyError):
            DevRun.from_registry({**row, "state": "expired"}, **args)
        with self.assertRaises(PolicyError):
            replace(run, validation_sha="")

    def test_exact_pairing_and_complete_matched_references_required(self):
        q = Search(make_plan())
        runs, pairs = inputs(q)
        c = candidates(q.plan, "compact")[0]
        r = runs[c.recovery.id]
        other = replace(r.items[0], id=IDS[-1])
        for invalid in ([v for v in runs.values() if v != r],
                        [v if v != r else replace(r, items=(other, *r.items[1:])) for v in runs.values()]):
            with self.assertRaises(PolicyError):
                q.advance(invalid, pairs.values())
        with self.assertRaises(PolicyError):
            replace(r, items=(r.items[0], r.items[0]))
        p = pairs[(c.arm.id, c.recovery.id)]
        for bad in (replace(p, candidate_result=p.reference_result, reference_result=p.candidate_result),
                    replace(p, membership_sha=h("other-members")), replace(p, method_sha=h("unregistered")),
                    replace(p, accuracy=Band(1, 3))):
            with self.assertRaises(PolicyError):
                q.advance(runs.values(), [bad if v == p else v for v in pairs.values()])
        with self.assertRaises(PolicyError):
            q.advance(runs.values(), [v for v in pairs.values() if v != p])
        self.assertTrue(q.advance(runs.values(), pairs.values()).final)  # No partial promotion on error.

    def test_noninferiority_requires_both_bounds_inside_margins(self):
        plan = make_plan()
        arm = candidates(plan, "compact")[0].arm
        for acc, brier, expected in (
                (Band(-19_999, 100), Band(-10, 9_999), True),
                (Band(-20_000, 100), Band(-10, 9_999), False),
                (Band(-19_999, 100), Band(-10, 10_000), False),
                (Band(-120_000, 100), Band(-10, 95_000), False)):
            q = Search(plan)
            runs, pairs = inputs(q)
            k = (arm.id, STOCK.id)
            pairs[k] = replace(pairs[k], accuracy=acc, brier=brier)
            decision = q.advance(runs.values(), pairs.values())
            rec = next(r for r in decision.assessments if r.arm == arm)
            self.assertEqual(rec.compact_noninferiority, "within_both_margins" if expected else "inconclusive")
            self.assertEqual(runs[arm.id].dev_id in decision.nominee_dev_ids, expected)
            self.assertIs(rec.recovery_gain, False)  # Noninferiority is not k>1 recovery.

    def test_positive_compact_brier_loss_over_limit_is_not_recovery(self):
        q = Search(make_plan())
        def worse(run):
            return replace(run, items=tuple(replace(i, brier_ppm=i.brier_ppm + 11_000) for i in run.items)
                           ) if run.arm.kind == "compact" and run.arm.k > 1 else run
        runs, pairs = inputs(q, modify=worse)
        d = q.advance(runs.values(), pairs.values())
        c = next(a for a in d.assessments if a.arm.kind == "compact")
        self.assertIn("compact.loss_exceeds_margin", c.eliminate)
        self.assertIn("k1.quality_regression", c.eliminate)
        self.assertEqual(c.compact_noninferiority, "outside_margin")
        self.assertNotIn(c.dev_id, d.nominee_dev_ids)

    def test_accuracy_gain_cannot_hide_conditional_harm_or_coverage_change(self):
        q = Search(make_plan())
        def pattern(run):
            is_loop = run.arm.kind == "loop"
            return replace(run, items=tuple(
                item(i.id, IDS.index(i.id) not in ({0, 3} if is_loop else {0, 1, 2, 3}),
                     IDS.index(i.id) in ({0} if is_loop else {0, 4, 5, 6})) for i in run.items))
        runs, pairs = inputs(q, modify=pattern)
        decision = q.advance(runs.values(), pairs.values())
        loop = next(r for r in decision.assessments if r.arm.kind == "loop")
        stock = next(c.rates for c in decision.controls if c.arm == STOCK)
        self.assertGreater(loop.rates.accuracy, stock.accuracy)
        self.assertEqual((loop.rates.joint_error, stock.joint_error), (Fraction(1, 8), Fraction(1, 8)))
        self.assertEqual((loop.rates.conditional_error, stock.conditional_error), (1, Fraction(1, 4)))
        self.assertEqual((loop.rates.coverage, stock.coverage), (Fraction(1, 8), Fraction(1, 2)))
        self.assertIn("stock.conditional_error.regression", loop.eliminate)
        self.assertNotIn(loop.dev_id, decision.nominee_dev_ids)
        report = json.loads(json.dumps(decision.report()))
        output = next(a["rates"] for a in report["assessments"] if a["arm"]["kind"] == "loop")
        self.assertEqual(output["conditional_high_confidence_error"], {"numerator": 1, "denominator": 1})
        self.assertEqual(output["joint_high_confidence_error"], {"numerator": 1, "denominator": 8})

    def test_joint_harm_and_undefined_conditional_risk_are_flagged(self):
        for mode in ("worse", "zero"):
            q = Search(make_plan())
            def pattern(run):
                if run.arm.kind != "loop":
                    return run
                return replace(run, items=tuple(item(i.id, i.correct, mode == "worse") for i in run.items))
            runs, pairs = inputs(q, modify=pattern)
            d = q.advance(runs.values(), pairs.values())
            r = next(a for a in d.assessments if a.arm.kind == "loop")
            if mode == "worse":
                self.assertIn("stock.joint_error.regression", r.eliminate)
            else:
                self.assertIsNone(r.rates.conditional_error)
                self.assertEqual(r.rates.coverage, 0)
                self.assertIn("stock.conditional_error.undefined_interval", r.review)
            self.assertNotIn(r.dev_id, d.nominee_dev_ids)

    def test_finite_measurements_and_measured_cost_match_not_nominal_match(self):
        q = Search(make_plan())
        for value in (math.inf, math.nan, -1, 0, True):
            with self.assertRaises(PolicyError):
                MeasuredCost(value, 1, 1, 1, h("cost"))
        for value in (math.inf, math.nan, None, True):
            with self.assertRaises(PolicyError):
                Band(value, 10)
        def mismatch(r):
            return replace(r, cost=replace(r.cost, layer_millicalls=1010)) if r.arm.kind == "loop" else r
        runs, pairs = inputs(q, modify=mismatch)
        d = q.advance(runs.values(), pairs.values())
        loop = next(r for r in d.assessments if r.arm.kind == "loop")
        self.assertIn("stock.measured_cost_unmatched", loop.eliminate)
        self.assertNotIn(loop.dev_id, d.nominee_dev_ids)

    def test_pareto_tradeoffs_caps_shared_controls_and_replay(self):
        plan = make_plan(3)
        def costs(run):
            by_arm = {h("loop0"): (100, 200), h("loop1"): (200, 100), h("loop2"): (300, 300)}
            if run.arm.id not in by_arm:
                return run
            latency, peak = by_arm[run.arm.id]
            return replace(run, cost=replace(run.cost, latency_us=latency, peak_bytes=peak))
        q = Search(plan)
        runs, pairs = inputs(q, modify=costs)
        d = q.advance(runs.values(), pairs.values())
        self.assertEqual({i for i in d.kept if runs[i].arm.kind == "loop"}, {h("loop0"), h("loop1")})
        self.assertNotIn(h("loop2"), d.frontier)
        self.assertEqual(len(d.kept), 4)
        required = {runs[a.id].dev_id for c in plan.candidates if c.arm.id in d.kept for a in c.controls}
        self.assertEqual(set(d.control_dev_ids), required)
        self.assertEqual(len(d.control_dev_ids), 3)  # Shared stock and two distinct k=1 payloads.
        self.assertFalse(set(d.control_dev_ids) & set(d.nominee_dev_ids))
        replay = Search(replace(plan, candidates=tuple(reversed(plan.candidates))))
        d2 = replay.advance(reversed(list(runs.values())), reversed(list(pairs.values())))
        self.assertEqual(d2, d)

    def test_successive_elimination_no_reentry_no_finalists_before_last_rung(self):
        plan = make_plan(3, (Rung(IDS[:4], 1), Rung(IDS[:8], 1)))
        q = Search(plan)
        first, pairs = inputs(q)
        d0 = q.advance(first.values(), pairs.values())
        self.assertFalse(d0.final)
        self.assertEqual(d0.nominee_dev_ids, ())
        self.assertEqual(len(q.active), 2)
        runs, pairs = inputs(q, r=1)
        with self.assertRaises(PolicyError):
            q.advance([*runs.values(), next(r for i, r in first.items() if i not in runs)], pairs.values())
        i = q.active[0]
        run = runs[i]
        changed = replace(run, items=tuple(replace(x, brier_ppm=x.brier_ppm + 1)
                                           if x.id == IDS[0] else x for x in run.items))
        with self.assertRaises(PolicyError):
            q.advance([changed if x == run else x for x in runs.values()], pairs.values())
        d1 = q.advance(runs.values(), pairs.values())
        self.assertEqual(set(d1.kept), set(d0.kept))
        self.assertEqual(len(d1.nominee_dev_ids), 2)
        self.assertNotEqual(d1.decision_sha, d0.decision_sha)
        self.assertEqual(q.branch_wave(), (None, None))
        with self.assertRaises(PolicyError):
            q.advance(runs.values(), pairs.values())


if __name__ == "__main__":
    unittest.main()
