"""Invented CPU split-graph witnesses; no model, inputs or target labels."""

import math
import unittest
from array import array
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from joey_research.loop import LoopCounts
from joey_research.one_probe import run_one_probe


class Hidden:
    def __init__(self, branch, values):
        self.branch, self.positions = branch, branch.positions
        self.values = list(values)

    def clone(self):
        self.branch.note("clone")
        copied = Hidden(self.branch, self.values)
        self.branch.copies.append((self, copied))
        return copied


class SplitSpy:
    def __init__(self, probabilities=(0.125, 0.5625, 0.3125)):
        self.labels = ["z", "a", "m"]
        self.distribution = probabilities
        self.positions = (4, 5)
        self.active = False
        self.calls, self.states, self.seeds = [], [], []
        self.blocks, self.codas, self.copies = [], [], []

    def note(self, name):
        assert self.active, "outside inference context"
        self.calls.append(name)

    @contextmanager
    def inference_context(self):
        assert not self.active
        self.active = True
        try:
            self.note("enter")
            yield
        finally:
            self.note("exit")
            self.active = False

    def prelude(self):
        self.note("prelude")
        seed = Hidden(self, (1, 3))
        self.seeds.append(seed)
        return seed

    def fresh_state(self):
        self.note("fresh")
        state = {"positions": self.positions, "used": False}
        self.states.append(state)
        return state

    def use(self, name, hidden, state):
        self.note(name)
        assert hidden.branch is self and state["positions"] is hidden.positions is self.positions
        assert state is self.states[-1] and not state["used"]
        state["used"] = True

    def block(self, hidden, state):
        self.use("block", hidden, state)
        out = Hidden(self, (2 * x + 1 for x in hidden.values))
        self.blocks.append((hidden, out))
        return out

    def coda(self, hidden, state):
        self.use("coda", hidden, state)
        x, y = hidden.values
        logits = array("d", (y - i * x for i in range(len(self.labels))))
        self.codas.append((hidden, logits))
        return logits

    def probabilities(self, logits):
        self.note("probabilities")
        assert logits is self.codas[-1][1], "calibrate the actual readout"
        return self.distribution


FIRST_CHECK = ["enter", "prelude", "fresh", "block", "clone", "fresh", "coda", "probabilities"]


class OneProbeTest(unittest.TestCase):
    def test_exact_callback_and_hidden_lineage_at_either_exit(self):
        for cap in (2, 3, 4, 6):
            for threshold, depth, reason in ((0.25, 1, "margin"), (0.5, cap, "cap")):
                with self.subTest(cap=cap, threshold=threshold):
                    q = SplitSpy()
                    r = run_one_probe(q, cap=cap, threshold=threshold)
                    tail = [] if depth == 1 else ["fresh", "block"] * (cap - 1) + ["fresh", "coda"]
                    self.assertEqual(q.calls, FIRST_CHECK + tail + ["exit"])
                    self.assertEqual((r.exit_depth, r.reason), (depth, reason))
                    self.assertEqual(r.counts, LoopCounts(0, 1, depth, 1 if depth == 1 else 2))
                    self.assertEqual(r.first_margin, 0.25)
                    self.assertEqual((r.labels, r.first_probabilities),
                                     (("z", "a", "m"), (0.125, 0.5625, 0.3125)))
                    self.assertIs(q.blocks[0][0], q.seeds[0])
                    for previous, current in zip(q.blocks, q.blocks[1:]):
                        self.assertIs(current[0], previous[1])
                    ((source, copied),) = q.copies
                    self.assertIs(source, q.blocks[0][1])
                    self.assertIs(q.codas[0][0], copied)
                    self.assertIsNot(copied, source)
                    if depth > 1:
                        self.assertIs(q.codas[-1][0], q.blocks[-1][1])
                    self.assertIs(r.logits, q.codas[-1][1])
                    self.assertEqual(len({id(s) for s in q.states}), depth + len(q.codas))
                    self.assertTrue(all(s["used"] for s in q.states))

    def test_input_mutating_coda_cannot_change_continuation_or_first_evidence(self):
        class ClobbersInputs(SplitSpy):
            def coda(self, hidden, state):
                logits = super().coda(hidden, state)
                if len(self.codas) == 1:
                    hidden.values[:] = [-99, -99]
                else:
                    self.distribution[:] = [0, 1, 0]
                return logits

        q = ClobbersInputs([0.125, 0.5625, 0.3125])
        r = run_one_probe(q, cap=3, threshold=0.75)
        self.assertEqual(q.blocks[1][0].values, [3, 7])
        self.assertEqual(list(r.logits), [31, 16, 1])
        self.assertEqual(r.first_probabilities, (0.125, 0.5625, 0.3125))
        self.assertEqual(r.first_margin, 0.25)
        q.labels.reverse()
        self.assertEqual(r.labels, ("z", "a", "m"))
        with self.assertRaises(FrozenInstanceError):
            r.exit_depth = 1

    def test_tied_slots_and_unrounded_inclusive_boundary(self):
        for p, threshold, depth in (
            ((0.5, 0.5), 0, 1),
            ((0.4375, 0.125, 0.4375), 0, 1),  # Two equal maxima, not two distinct values.
            ((0.4375, 0.125, 0.4375), 0.10, 4),
            ((0.0625, 0.0625, 0, 0.5625, 0.3125), 0.25, 1),
            ((0.125, 0.5625, 0.3125), 0.25, 1),
            ((0.125, 0.5624999999, 0.3125000001), 0.25, 4),
            ((0.1250005, 0.5625, 0.3125), 0.25, 1),  # Within mass tolerance; don't renormalize.
            ((0, 1, 0), 1.0, 1),
        ):
            with self.subTest(p=p, threshold=threshold):
                q = SplitSpy(p)
                q.labels = ("violet", "red", "yellow", "blue", "green")[:len(p)]
                r = run_one_probe(q, cap=4, threshold=threshold)
                self.assertEqual(r.labels, q.labels)
                self.assertEqual((r.first_probabilities, r.exit_depth), (p, depth))
                largest, runner_up = sorted(p, reverse=True)[:2]
                self.assertEqual(r.first_margin, largest - runner_up)

    def test_one_new_decision_and_fresh_split_per_invocation(self):
        q = SplitSpy()
        first = run_one_probe(q, cap=4, threshold=0.5)
        q.distribution = (0, 0.875, 0.125)
        second = run_one_probe(q, cap=4, threshold=0.5)
        self.assertEqual(first.counts, LoopCounts(0, 1, 4, 2))
        self.assertEqual(second.counts, LoopCounts(0, 1, 1, 1))
        self.assertEqual((first.first_margin, second.first_margin), (0.25, 0.75))
        self.assertIs(q.blocks[4][0], q.seeds[1])
        self.assertIsNot(q.seeds[0], q.seeds[1])
        self.assertEqual(len({id(s) for s in q.states}), 8)

    def test_invalid_policies_and_initial_binding_do_no_execution(self):
        for cap, threshold in (
            (1, 0), (5, 0), (2.0, 0), (True, 0), (4, False), (4, math.inf),
            (4, math.nan), (4, -0.1), (4, "0.25"), (4, 0.2500000001),
        ):
            q = SplitSpy()
            with self.subTest(cap=cap, threshold=threshold), self.assertRaises(ValueError):
                run_one_probe(q, cap=cap, threshold=threshold)
            self.assertEqual(q.calls, [])
        for labels in ((), ("a",), ("z", "z"), {"a", "b"}, "ab", (True, 2), ("z", ["a"])):
            q = SplitSpy()
            q.labels = labels
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                run_one_probe(q, cap=2, threshold=0)
            self.assertEqual(q.calls, [])

    def test_invalid_probabilities_never_trigger_or_continue(self):
        for p in (0.5, (0.5, 0.5), ((0, 1, 0),), "010", {"z": 0, "a": 1, "m": 0},
                  (0, True, 0), (math.nan, 1, 0), (0, math.inf, 0), (-0.25, 0.75, 0.5),
                  (0, 1.0000001, 0), (0.25, 0.25, 0.499998)):
            q = SplitSpy(p)
            with self.subTest(p=p), self.assertRaisesRegex(ValueError, "probabilities"):
                run_one_probe(q, cap=6, threshold=0)
            self.assertEqual(q.calls, FIRST_CHECK + ["exit"])
            self.assertFalse(q.active)

    def test_invalid_logits_rejected_at_first_or_final_coda(self):
        for at in (1, 2):
            for bad in ((1, 2), ((1, 2, 3),), (4, math.nan, 2), (False, 1, 2)):
                q = SplitSpy()
                coda = q.coda

                def sometimes_bad(hidden, state):
                    logits = coda(hidden, state)
                    return bad if len(q.codas) == at else logits

                with self.subTest(at=at, bad=bad), patch.object(q, "coda", sometimes_bad):
                    with self.assertRaisesRegex(ValueError, "logits"):
                        run_one_probe(q, cap=3, threshold=1)
                self.assertEqual(len(q.codas), at)
                self.assertEqual(q.calls.count("probabilities"), at - 1)
                self.assertFalse(q.active)

    def test_same_width_slot_mutation_or_rebinding_is_not_a_valid_decision(self):
        for name, occurrence in (("block", 1), ("coda", 1), ("probabilities", 1),
                                 ("block", 2), ("coda", 2), ("exit", 1)):
            q = SplitSpy()
            note = q.note

            def swap(name_seen):
                note(name_seen)
                if name_seen == name and q.calls.count(name) == occurrence:
                    if name == "coda":
                        q.labels = tuple(reversed(q.labels))
                    else:
                        q.labels.reverse()  # Same list object, same width.

            with self.subTest(name=name, occurrence=occurrence), patch.object(q, "note", swap):
                with self.assertRaisesRegex(ValueError, "slot order changed"):
                    run_one_probe(q, cap=4, threshold=1)
            self.assertFalse(q.active)

    def test_no_optimistic_fallback_for_missing_or_identity_clone(self):
        for clone, error in ((None, TypeError), (lambda h: h, ValueError)):
            q = SplitSpy()
            with patch.object(Hidden, "clone", clone), self.assertRaises(error):
                run_one_probe(q, cap=2, threshold=0)
            self.assertFalse(q.codas)
            self.assertFalse(q.active)


if __name__ == "__main__":
    unittest.main()
