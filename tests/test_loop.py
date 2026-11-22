"""Analytic CPU fixtures for the integrators, not checkpoint evaluations."""

import math
import unittest
from array import array
from contextlib import contextmanager
from importlib.util import find_spec

from joey_research.loop import LoopCounts, LoopPlan, RKTableau, run_loop


class AnalyticHandle:
    """Two noncommuting layers with composite g(y) = y + y**2/2, y0 = 1."""

    def __init__(self, slots=3):
        self.slots = slots
        self.positions = (4, 5)
        self.mask = ((True, False), (True, True))
        self.active = False
        self.calls = []
        self.inputs = []
        self.states = []
        self.stock_logits = self._slots(1.5)

    def _slots(self, y):
        return array("f", (y * (i + 1) for i in range(self.slots)))

    @contextmanager
    def inference_context(self):
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def _call(self, name):
        assert self.active, "call outside inference context"
        self.calls.append(name)

    def stock(self):
        self._call("stock")
        return self.stock_logits

    def prelude(self):
        self._call("prelude")
        return 1.0

    def fresh_state(self):
        self._call("fresh")
        state = {"positions": self.positions, "mask": self.mask, "visits": []}
        self.states.append(state)
        return state

    def _use(self, state, visits):
        assert state["positions"] is self.positions and state["mask"] is self.mask
        assert not state["visits"], "state reused across calls"
        state["visits"].extend(visits)

    def block(self, hidden, state):
        self._call("block")
        self._use(state, ("double", "nonlinear"))
        self.inputs.append(hidden)
        u = 2 * hidden
        return u / 2 + u**2 / 8

    def coda(self, hidden, state):
        self._call("coda")
        self._use(state, ("read",))
        return self._slots(hidden)


class LoopTest(unittest.TestCase):
    def test_k_one_returns_actual_stock_object_without_split_or_state_factory(self):
        for method, kwargs in (
            ("naive", {}),
            ("zero_extra", {}),
            ("damped_extra", {"alpha": 0.0}),
            ("euler", {}),
            ("rk", {"beta": 0.5}),
            ("prelude_anchor", {"beta": 1.0}),
        ):
            with self.subTest(method=method):
                handle = AnalyticHandle()
                r = run_loop(LoopPlan(k=1, method=method, readout="diagnostic", **kwargs), handle)
                self.assertIs(r.logits, handle.stock_logits)
                self.assertEqual(handle.calls, ["stock"])
                self.assertEqual(r.counts, LoopCounts(1, 0, 0, 0))
                self.assertEqual(r.counts.layer_evaluations(2, 3, 1), 6)
                self.assertEqual(r.counts.coda_readouts, 1)
                self.assertEqual(r.readouts[0].kind, "stock")

    def test_zero_extra_is_an_independent_full_split_reconstruction(self):
        for slots in (2, 5):
            with self.subTest(slots=slots):
                handle = AnalyticHandle(slots)
                r = run_loop(LoopPlan(k=4, method="zero_extra"), handle)
                self.assertEqual(handle.calls, ["prelude", "fresh", "block", "fresh", "coda"])
                self.assertEqual(r.counts, LoopCounts(0, 1, 1, 1))
                self.assertEqual(r.logits.tobytes(), handle.stock_logits.tobytes())
                self.assertIsNot(r.logits, handle.stock_logits)
                self.assertEqual(len(r.logits), slots)
                self.assertEqual(r.readouts, ())

    def test_two_evaluation_nonlinear_trajectories_and_distinct_anchors(self):
        for method, kwargs, second_input, endpoint in (
            ("naive", {}, 3 / 2, 21 / 8),
            ("damped_extra", {"alpha": 0.25}, 3 / 2, 57 / 32),
            ("damped_extra", {"alpha": 0.0}, 3 / 2, 3 / 2),
            ("euler", {}, 5 / 4, 105 / 64),
            ("rk", {"beta": 0.5}, 5 / 4, 201 / 128),
            ("prelude_anchor", {"beta": 0.5}, 5 / 4, 169 / 128),
            ("prelude_anchor", {"beta": 1.0}, 5 / 4, 1.0),
        ):
            with self.subTest(method=method, kwargs=kwargs):
                handle = AnalyticHandle()
                r = run_loop(LoopPlan(k=2, method=method, **kwargs), handle)
                self.assertEqual(handle.inputs, [1.0, second_input])
                self.assertEqual(r.logits[0], endpoint)
                self.assertEqual(r.counts, LoopCounts(0, 1, 2, 1))

    def test_exact_stage_budget_including_anchor_at_both_beta_endpoints(self):
        k = 5
        for beta in (0.0, 0.25, 1.0):
            with self.subTest(beta=beta):
                handle = AnalyticHandle()
                built_in = run_loop(LoopPlan(k=k, method="rk", beta=beta), handle)
                w = (1.0 - beta) / k
                explicit = RKTableau(tuple((1 / k,) * i for i in range(k)), (beta + w,) + (w,) * (k - 1))
                general = run_loop(LoopPlan(k=k, method="rk", tableau=explicit), AnalyticHandle())
                self.assertEqual(built_in.logits, general.logits)
                self.assertEqual(built_in.counts.block_evaluations, k)
                self.assertEqual(len(handle.inputs), k)
                self.assertEqual(built_in.counts.coda_readouts, 1)
                if beta == 0:
                    euler = run_loop(LoopPlan(k=k, method="euler"), AnalyticHandle())
                    self.assertEqual(built_in.logits, euler.logits)
                if beta == 1:
                    self.assertEqual(built_in.logits, handle.stock_logits)

    def test_generic_rk_uses_full_block_residual_at_each_stage_not_at_origin(self):
        tableau = RKTableau(((), (0.5,), (-1, 2)), (1 / 6, 2 / 3, 1 / 6))
        handle = AnalyticHandle()
        r = run_loop(LoopPlan(k=3, method="rk", tableau=tableau, readout="diagnostic"), handle)
        self.assertEqual(handle.inputs, [1.0, 5 / 4, 33 / 16])
        self.assertEqual([probe.kind for probe in r.readouts], ["stage_output", "stage_output", "endpoint"])
        self.assertEqual([probe.logits[0] for probe in r.readouts[:-1]], [3 / 2, 65 / 32])
        self.assertAlmostEqual(r.logits[0], 6017 / 3072, places=6)
        self.assertEqual(r.counts, LoopCounts(0, 1, 3, 3))

    def test_final_and_diagnostic_costs_and_fresh_state_at_every_use(self):
        final = run_loop(LoopPlan(k=3, method="euler"), AnalyticHandle())
        handle = AnalyticHandle()
        diag = run_loop(LoopPlan(k=3, method="euler", readout="diagnostic"), handle)
        self.assertEqual(final.logits, diag.logits)
        self.assertIs(diag.logits, diag.readouts[-1].logits)
        self.assertEqual([r.block_evaluations for r in diag.readouts], [1, 2, 3])
        self.assertEqual([r.kind for r in diag.readouts], ["substep", "substep", "endpoint"])
        self.assertAlmostEqual(diag.readouts[0].logits[0], 1 + 1 / 6)
        self.assertEqual(final.counts.layer_evaluations(2, 3, 1), 12)
        self.assertEqual(diag.counts.layer_evaluations(2, 3, 1), 14)
        self.assertEqual(diag.counts.layer_evaluations(0, 3, 0), 9)
        self.assertEqual(diag.counts.coda_readouts, 3)
        self.assertEqual([s["visits"] for s in handle.states], [["double", "nonlinear"], ["read"]] * 3)
        self.assertEqual(len({id(s) for s in handle.states}), 6)
        self.assertFalse(handle.active)

    def test_adaptive_exit_uses_live_readouts_and_skips_later_evaluations(self):
        handle = AnalyticHandle(5)
        checked = []

        def stop(probe):
            self.assertTrue(handle.active)
            checked.append(probe.block_evaluations)
            return probe.logits[0] > 2.0

        r = run_loop(LoopPlan(k=6, readout="diagnostic"), handle, stop_when=stop)
        self.assertEqual(checked, [1, 2])
        self.assertTrue(r.stopped_early)
        self.assertEqual(handle.inputs, [1, 3 / 2])
        self.assertEqual(r.counts, LoopCounts(0, 1, 2, 2))
        self.assertIs(r.logits, r.readouts[-1].logits)
        self.assertEqual(r.logits[0], 21 / 8)

    def test_reject_ambiguous_plans_and_non_actuating_exit_requests(self):
        for kwargs in (
            {"k": True}, {"k": 0}, {"k": 1.5}, {"method": "unknown"}, {"readout": "unknown"},
            {"method": "euler", "alpha": 0.1}, {"method": "damped_extra", "alpha": math.nan},
            {"method": "rk", "beta": 1.5}, {"method": "prelude_anchor"}, {"beta": 0.5},
            {"k": 2, "method": "rk", "tableau": RKTableau(((),), (1,))},
            {"method": "rk", "beta": 0.5, "tableau": RKTableau(((),), (1,))},
        ):
            with self.subTest(plan=kwargs), self.assertRaises(ValueError):
                LoopPlan(**kwargs)
        for a, b in (((), ()), (((1,),), (1,)), (((),), (0.5,)), (((), (math.inf,)), (0, 1))):
            with self.subTest(tableau=(a, b)), self.assertRaises(ValueError):
                RKTableau(a, b)
        for plan in (LoopPlan(k=3), LoopPlan(k=3, method="rk", beta=0.5, readout="diagnostic")):
            handle = AnalyticHandle()
            with self.assertRaises(ValueError):
                run_loop(plan, handle, stop_when=lambda _: True)
            self.assertEqual(handle.calls, [])

    def test_callback_error_propagates_and_restores_inference_context(self):
        handle = AnalyticHandle()

        def fail(hidden, state):
            raise RuntimeError("fixture block failed")

        handle.block = fail
        with self.assertRaisesRegex(RuntimeError, "fixture block failed"):
            run_loop(LoopPlan(k=2), handle)
        self.assertFalse(handle.active)
        self.assertNotIn("coda", handle.calls)

    def test_actuation_rejects_ambiguous_truth_values(self):
        handle = AnalyticHandle()
        with self.assertRaisesRegex(TypeError, "boolean"):
            run_loop(LoopPlan(k=3, readout="diagnostic"), handle, stop_when=lambda _: "false")
        self.assertFalse(handle.active)
        self.assertEqual(len(handle.inputs), 1)

    def test_vector_stages_do_not_write_to_original_buffers(self):
        class Vec:
            def __init__(self, values):
                self.buffer = array("f", values)

            def __add__(self, other):
                return Vec(a + b for a, b in zip(self.buffer, other.buffer, strict=True))

            def __sub__(self, other):
                return Vec(a - b for a, b in zip(self.buffer, other.buffer, strict=True))

            def __mul__(self, scale):
                return Vec(a * scale for a in self.buffer)

            def __iadd__(self, other):
                raise AssertionError("in-place update")

            __isub__ = __imul__ = __iadd__

        original, bias = Vec([1, -2]), Vec([0.5, 1])
        before = original.buffer.tobytes(), bias.buffer.tobytes()

        class VectorHandle(AnalyticHandle):
            def prelude(self):
                return original

            def block(self, hidden, state):
                return hidden + bias

            def coda(self, hidden, state):
                return hidden.buffer

        for plan in (
            LoopPlan(k=4, method="rk", beta=0.5),
            LoopPlan(k=2, method="rk", tableau=RKTableau(((), (0.5,)), (0, 1))),
        ):
            r = run_loop(plan, VectorHandle())
            self.assertEqual(list(r.logits), [1.5, -1.0])
            self.assertEqual((original.buffer.tobytes(), bias.buffer.tobytes()), before)

    @unittest.skipUnless(find_spec("torch") is not None, "optional tensor backend not installed")
    def test_tensor_inputs_and_weights_unchanged_no_autograd(self):
        import torch

        hidden = torch.tensor([0.25, 0.75], requires_grad=True)
        weight = torch.tensor([0.5, 0.125], requires_grad=True)
        inputs_before = (hidden.detach().clone(), weight.detach().clone())

        class TensorHandle:
            inference_context = staticmethod(torch.inference_mode)

            def stock(self):
                return self.coda(self.block(self.prelude(), self.fresh_state()), self.fresh_state())

            def prelude(self):
                return hidden

            def fresh_state(self):
                return {"positions": (0, 1), "mask": (True, True)}

            def block(self, h, state):
                assert torch.is_inference_mode_enabled()
                return h + h.square() * weight

            def coda(self, h, state):
                return torch.stack((h.sum(), h[0] - h[1], h[1])).float()

        with torch.enable_grad():
            r = run_loop(LoopPlan(k=3, method="rk", beta=0.5, readout="diagnostic"), TensorHandle())
            self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(tuple(r.logits.shape), (3,))
        self.assertEqual(r.logits.dtype, torch.float32)
        self.assertTrue(all(not p.logits.requires_grad and p.logits.grad_fn is None for p in r.readouts))
        for tensor, old in zip((hidden, weight), inputs_before):
            self.assertTrue(torch.equal(tensor, old))
            self.assertIsNone(tensor.grad)


if __name__ == "__main__":
    unittest.main()
