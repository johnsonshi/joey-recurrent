"""CPU-only witnesses for decisions and call accounting; no checkpoint or labels."""

import math
import unittest
from contextlib import contextmanager
from dataclasses import replace

from joey_research.loop import LoopCounts, LoopPlan
from joey_research.loop_analysis import (
    PassTrace, Policy, SlotOrder, Stop, actuate, fit_threshold, record, replay,
)


class Witness:
    """Supply listed scores by depth, including when only the final coda is requested."""

    def __init__(self, rows, *, slots=("right", "left"), key="sample", split="discovery"):
        self.rows = rows
        self.order = SlotOrder(split, key, slots)
        self.active = False
        self.blocks, self.codas = [], []
        self.buffer = []

    def native_order(self):
        return self.order

    @contextmanager
    def inference_context(self):
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def prelude(self):
        return 0.0

    def fresh_state(self):
        return {}

    def block(self, h, state):
        assert self.active
        self.blocks.append(h + 1)
        return h + 1

    def coda(self, h, state):
        depth = int(h)
        assert self.active and h == depth
        self.codas.append(depth)
        self.buffer[:] = self.rows[depth - 1]
        return self.buffer

    def stock(self):
        assert self.active
        return self.rows[0]


ROWS = ((0, math.log(1.5)), (math.log(4), 0), (0, math.log(9)))
PLAN = LoopPlan(k=3)


def capture(rows=ROWS, **kwargs):
    q = Witness(rows, **kwargs)
    return record(LoopPlan(k=len(rows)), q, q.native_order)


class PassAnalysisTest(unittest.TestCase):
    def test_live_exit_never_reads_late_coda_or_charges_replay_as_savings(self):
        policy = Policy(stop=Stop("margin", 0.5))
        trace = capture()
        sim = replay(trace, policy)
        q = Witness(ROWS[:2])  # A third coda would fail.
        live = actuate(PLAN, q, q.native_order, policy)
        self.assertEqual(q.blocks, [1, 2])
        self.assertEqual(q.codas, [1, 2])
        self.assertEqual(live.probabilities, sim.probabilities)
        self.assertEqual((live.decision_pass, live.reason, live.slot), (2, "margin", "right"))
        self.assertEqual(live.executed, LoopCounts(0, 1, 2, 2))
        self.assertIsNone(live.simulated)
        self.assertIsNone(live.recorded)
        self.assertEqual(sim.recorded, LoopCounts(0, 1, 3, 3))
        self.assertEqual(sim.executed, LoopCounts(0, 0, 0, 0))
        self.assertEqual(sim.simulated, live.executed)
        self.assertIsNone(live.fallback_used)

    def test_first_and_final_policies_schedule_only_the_required_coda(self):
        trace = capture()
        for pool, depth, expected_slot in (("first", 1, "left"), ("final", 3, "left")):
            with self.subTest(pool=pool):
                q = Witness(ROWS)
                live = actuate(PLAN, q, q.native_order, Policy(pool))
                sim = replay(trace, Policy(pool))
                self.assertEqual((live.slot, live.decision_pass), (expected_slot, depth))
                self.assertEqual(q.codas, [depth])
                self.assertEqual(q.blocks, list(range(1, depth + 1)))
                self.assertEqual(live.executed, sim.simulated)
                self.assertEqual(sim.recorded.coda_readouts, 3)
        one = Witness(ROWS[:1])
        d = actuate(LoopPlan(k=1), one, one.native_order, Policy(stop=Stop("margin", 0.1)))
        self.assertEqual(d.reason, "margin")
        self.assertEqual(d.executed, LoopCounts(1, 0, 0, 0))
        self.assertEqual((d.executed.coda_readouts, one.codas, one.blocks), (1, [], []))

    def test_distribution_rules_have_distinct_scale_and_anchor_semantics(self):
        rows = ((40.0, 0.0), (0.0, 1.0), (0.0, 1.0))
        trace = capture(rows)
        avg = replay(trace, Policy("probability_mean"))
        norm = replay(trace, Policy("normalized_logit_mean"))
        anchor = replay(trace, Policy("first_anchor", first_weight=0.8))
        self.assertEqual((avg.slot, norm.slot, anchor.slot), ("right", "left", "right"))
        first_p = 1 / (1 + math.exp(-40))
        last_p = 1 / (1 + math.e)
        self.assertAlmostEqual(avg.probabilities[0], (first_p + 2 * last_p) / 3)
        self.assertAlmostEqual(norm.probabilities[0], 1 / (1 + math.exp(2 / 3)))
        self.assertAlmostEqual(anchor.probabilities[0], 0.8 * first_p + 0.2 * last_p)
        self.assertEqual(anchor.simulated.coda_readouts, 3)
        shifted = capture(((3, 2), (1000, 1500), (-7, -6)))
        self.assertEqual(norm.probabilities, replay(shifted, Policy("normalized_logit_mean")).probabilities)
        uniform = capture(((2, 2, 2), (0, 0, 0)), slots=(9, 4, 7))
        self.assertEqual(replay(uniform, Policy("normalized_logit_mean")).probabilities, (1 / 3,) * 3)
        tiny = capture(((0.0, 1e-20),))
        self.assertEqual(replay(tiny, Policy()).slot_index, 1)

    def test_entropy_and_stability_are_causal_and_ties_reset_the_streak(self):
        rows = ((1, 0, 0), (4, 4, 0), (0, 4, 1), (0, 5, 0))
        q = Witness(rows, slots=(71, 11, 31))
        d = actuate(LoopPlan(k=4), q, q.native_order, Policy(stop=Stop("stability", 2)))
        self.assertEqual((d.slot, d.decision_pass, d.reason, d.fallback_used), (11, 4, "stability", None))
        other = Witness(rows, slots=(71, 11, 31))
        e = actuate(LoopPlan(k=4), other, other.native_order, Policy(stop=Stop("entropy", 0.4)))
        self.assertEqual((e.slot, e.decision_pass, other.codas), (11, 3, [1, 2, 3]))
        tie = replay(capture(((4, 4), (0, 1))), Policy(stop=Stop("entropy", 1)))
        self.assertEqual(tie.decision_pass, 2)

    def test_fallback_to_first_does_not_hide_later_cost(self):
        rows = ((0, 1), (1, 0), (1, 0))
        q = Witness(rows)
        policy = Policy(stop=Stop("margin", 0.99), fallback="first")
        d = actuate(PLAN, q, q.native_order, policy)
        self.assertEqual((d.reason, d.fallback_used, d.slot), ("no_trigger", "first", "left"))
        self.assertEqual((d.decision_pass, d.executed.coda_readouts, q.codas), (3, 3, [1, 2, 3]))
        latest = replay(capture(rows), replace(policy, fallback="aggregate"))
        self.assertEqual((latest.fallback_used, latest.slot), ("aggregate", "right"))
        self.assertEqual(d.executed, latest.simulated)

    def test_same_width_is_not_same_native_order_and_reused_buffers_are_snapshotted(self):
        class SwapsKeys(Witness):
            def coda(self, h, state):
                raw = super().coda(h, state)
                if h == 2:
                    self.order = replace(self.order, slots=self.order.slots[::-1])
                return raw

        q = SwapsKeys(ROWS)
        with self.assertRaisesRegex(ValueError, "slot order changed"):
            record(PLAN, q, q.native_order)
        self.assertEqual(q.codas, [1, 2])
        trace = capture()
        self.assertEqual([p.logits.logits for p in trace.result.readouts], list(ROWS))
        r = trace.result
        foreign = capture(key="other").result.readouts[1]
        for bad in (
            replace(r, readouts=(r.readouts[0], foreign, r.readouts[2])),
            replace(r, readouts=(r.readouts[0], r.readouts[0], r.readouts[2])),
            replace(r, readouts=(replace(r.readouts[0], kind="stage_output"), *r.readouts[1:])),
            replace(r, counts=LoopCounts(0, 1, 3, 1)),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                PassTrace(trace.plan, bad)

    def test_rejects_stages_and_malformed_scores_before_analysis(self):
        for plan in (LoopPlan(k=2, method="rk", beta=0.5), LoopPlan(k=2, method="euler")):
            q = Witness(ROWS)
            with self.assertRaisesRegex(ValueError, "completed"):
                actuate(plan, q, q.native_order, Policy())
            self.assertFalse(q.blocks)
        for rows in ((((1,), (2,)),), ((1, math.inf),), ((1, math.nan),), ((1, 2, 3),)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                capture(rows)
        with self.assertRaises(ValueError):
            capture(slots=(7, 7))

    def test_fit_is_paired_label_free_and_excludes_reserved_splits(self):
        for split in ("discovery", "selection"):
            a = capture(split=split)
            b = capture(((math.log(39), 0),) * 3, key="another", split=split)
            manifest = (a.order, b.order)
            policy = Policy(stop=Stop("margin", 0.5))
            fit = fit_threshold(manifest, (b, a), policy, (0.5, 0.9))
            self.assertEqual((fit.policy.stop.threshold, fit.agreements_with_final), (0.9, 2))
            self.assertEqual((fit.simulated_blocks, fit.simulated_codas, fit.recorded_codas), (4, 4, 6))
            with self.assertRaises(ValueError):
                fit_threshold(manifest, (a, b), policy, (0.5,))
            for expected, traces in (
                (manifest, (a,)), (manifest, (a, a)), ((a.order, a.order), (a, a)),
                ((replace(a.order, slots=a.order.slots[::-1]), b.order), (a, b)),
                (manifest, (a, capture(key="another", split="calibration"))),
            ):
                with self.subTest(expected=expected), self.assertRaises(ValueError):
                    fit_threshold(expected, traces, policy, (0.9,))
        for split in ("calibration", "final"):
            order = SlotOrder(split, "reserved", ("right", "left"))
            with self.assertRaisesRegex(ValueError, "discovery or selection"):
                fit_threshold((order,), (), Policy(stop=Stop("margin", 1.0)), (1.0,))


if __name__ == "__main__":
    unittest.main()
