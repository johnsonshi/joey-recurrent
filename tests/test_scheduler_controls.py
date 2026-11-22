"""Offline nomination tests: synthetic method preimages, receipts, payloads and clocks."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from joey_research.scheduler import (
    GPU_CAP_S, Gate, Manifest, MethodWitness, Observation, Outcome, Pick, PolicyError,
    Queue, Role, Split, TrialKey,
)
from test_scheduler import Clock, RUN, START, digest, receipt


FULL, COMPACT, PIN = map(digest, ("full weights", "compact weights", "compact receipt"))
PINS = {FULL: None, COMPACT: PIN}
COMPACT_SETTINGS = {"compact_recipe": "select-20", "compact_manifest_sha256": PIN}
DECISION = digest("synthetic dev decision")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def method(k=3, name="euler", **fields):
    return {"loop": {"k": k, "method": name}, "window": [8, 12], "reverse": False, **fields}


def zero(**fields):
    return method(2, "zero_extra", **fields)


def ensemble(n=2, d=6, seed=91473, **fields):
    return zero(order_ensemble={"numerator": n, "denominator": d, "seed": seed}, **fields)


class ControlNominations(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(tmp.cleanup)
        self.db = Path(tmp.name) / "synthetic.sqlite"
        self.clock = Clock()
        # Two free slots, but never two simultaneous 120s test envelopes.
        self.q = Queue.create(self.db, run_id=RUN, start_utc=START, gpu_classes=("cpu", "cpu"),
                              prior_gpu_s=GPU_CAP_S - 128, clock=self.clock)
        self.addCleanup(lambda: self.q.close())
        self.witnesses = {}

    def restart(self):
        self.q.close()
        self.q = Queue.resume(self.db, run_id=RUN, clock=self.clock)

    def pair(self, role, settings, *, parent=None, model=FULL, dev_data=digest("dev"),
             test_data=digest("held out"), complete=True, **key_changes):
        raw = canonical(settings)
        dev = replace(TrialKey(model, dev_data, digest("code"), sha(raw), 23, digest("score"), "bf16"),
                      **key_changes)
        test = replace(dev, dataset=test_data)
        for key, split, budget in ((dev, Split.DEV, 1), (test, Split.TEST, 120)):
            self.q.register(Manifest(digest("manifest:" + key.trial_id), key, split, "cpu", budget))
        if complete and not self.q.history(dev.trial_id):
            for gate in Gate:
                self.q.attest(dev.trial_id, gate, digest(gate.value + dev.trial_id))
            ticket = self.q.reserve(trial_id=dev.trial_id)
            self.assertIsNotNone(ticket)
            self.q.record(receipt(ticket, self.clock))
        p = Pick(dev.trial_id, test.trial_id, role)
        self.witnesses[p] = MethodWitness(raw, parent)
        return p

    def nominate(self, *picks, pins=None):
        self.q.nominate(picks, decision_sha=DECISION, witnesses={p: self.witnesses[p] for p in picks},
                        payloads={FULL: None} if pins is None else pins)

    def test_full_and_compact_controls_freeze_exact_reproducible_plan_across_restart(self):
        picks = []
        for model, fields in ((FULL, {}), (COMPACT, COMPACT_SETTINGS)):
            selected = self.pair(Role.SELECTED, method(**fields), model=model)
            picks.extend([self.pair(Role.STOCK, method(1, "naive", **fields), model=model), selected,
                          self.pair(Role.ZERO_EXTRA, zero(**fields), model=model, parent=selected)])
            if model == FULL:
                picks.append(self.pair(Role.ORDER_ENSEMBLE, ensemble(), parent=selected))
        self.nominate(*reversed(picks), pins=PINS)
        self.restart()
        self.nominate(*picks, pins=PINS)  # Same decision, not a second nomination event.
        self.assertEqual(sum(e["kind"] == "nominated" for e in self.q.events()), 1)
        for p in picks:
            self.assertIsNone(self.q.reserve(trial_id=p.test_id))
        self.clock.now = self.q.freeze_at
        with self.assertRaisesRegex(PolicyError, "frozen"):
            self.q.nominate([], decision_sha=digest("late replacement"))
        freeze = self.q.seal("freeze")
        self.assertEqual(sha(canonical(freeze["payload"])), freeze["sha256"])
        self.assertEqual(freeze["effective_at"], self.clock.now)
        self.assertEqual(freeze["payload"]["payloads"], PINS)
        rows = {row["test_id"]: row for row in freeze["payload"]["picks"]}
        self.assertEqual(set(rows), {p.test_id for p in picks})
        for p in picks:
            row, w = rows[p.test_id], self.witnesses[p]
            self.assertEqual((row["dev_id"], row["role"], row["method_sha256"]),
                             (p.dev_id, p.role.value, sha(w.settings_json)))
            self.assertEqual(canonical(row["settings"]), w.settings_json)
            if w.parent is None:
                self.assertNotIn("selected_parent", row)
            else:
                self.assertEqual(row["selected_parent"],
                                 {**asdict(w.parent), "method_sha256": rows[w.parent.test_id]["method_sha256"]})
        self.restart()
        self.assertEqual(self.q.seal("freeze"), freeze)

    def test_gate_budget_and_one_shot_for_each_new_role(self):
        p = self.pair(Role.SELECTED, method())
        z = self.pair(Role.ZERO_EXTRA, zero(), parent=p)
        e = self.pair(Role.ORDER_ENSEMBLE, ensemble(), parent=p)
        self.nominate(p, z, e)
        self.q.freeze_early(reason_sha=digest("synthetic freeze"))
        for control in (z, e):
            self.assertIsNone(self.q.reserve(trial_id=control.test_id))
            self.q.attest(control.test_id, Gate.OFFLINE_STAGING, digest("stage:" + control.test_id))
            self.assertIsNone(self.q.reserve(trial_id=control.test_id))
            self.q.attest(control.test_id, Gate.STOCK_PARITY, digest("parity:" + control.test_id))
        t = self.q.reserve(trial_id=z.test_id)
        self.assertEqual((t.split, t.attempt), (Split.TEST, 1))
        self.q.record(Observation(t.token, Outcome.UNKNOWN))
        self.restart()
        self.assertEqual(self.q.outstanding(), [t])
        self.assertEqual(self.q.status()["gpu_budget"]["reserved_s"], 120)
        self.assertIsNone(self.q.reserve(trial_id=e.test_id))  # Other slot free, insufficient balance.
        self.q.record(receipt(t, self.clock, Outcome.RETRYABLE))
        self.assertIsNone(self.q.reserve(trial_id=z.test_id))
        second = self.q.reserve(trial_id=e.test_id)
        self.q.record(receipt(second, self.clock))
        self.restart()
        for control in (z, e):
            self.assertIsNone(self.q.reserve(trial_id=control.test_id))
            self.assertEqual(len(self.q.history(control.test_id)), 1)
        self.clock.now = self.q.finalize_at
        final = self.q.seal("finalize")["payload"]
        self.assertEqual(final["freeze_sha"], self.q.seal("freeze")["sha256"])
        self.assertEqual({r["test_id"]: r["state"] for r in final["results"]},
                         {z.test_id: "failed", e.test_id: "succeeded", p.test_id: "expired"})

    def test_incomplete_wrong_id_noncanonical_or_stale_witness_cannot_replace_plan(self):
        p = self.pair(Role.SELECTED, method())
        z = self.pair(Role.ZERO_EXTRA, zero(), parent=p)
        self.nominate(p, z)
        valid = self.witnesses.copy()
        wrong = replace(p, dev_id=z.dev_id)
        inputs = [None, {}, {z: valid[z]}, {p: valid[p]}, {**valid, wrong: valid[p]},
                  {wrong: valid[p], z: valid[z]}, {**valid, z: valid[p]}]
        # Hash-equivalent after permissive JSON parsing is not canonical byte evidence.
        for raw in (valid[z].settings_json + b"\n", b'{"reverse":false,' + valid[z].settings_json[1:],
                    b" " * 8193, b"not JSON", canonical(zero(loop={"method": "zero_extra", "k": 2,
                                                                   "readout": "final"}))):
            inputs.append({**valid, z: replace(valid[z], settings_json=raw)})
        for i, witnesses in enumerate(inputs):
            with self.subTest(i=i), self.assertRaises(PolicyError):
                self.q.nominate([p, z], decision_sha=digest("substitution"), witnesses=witnesses,
                                payloads={FULL: None})
        for pins in (None, {}, {COMPACT: None}, PINS, {FULL: "not a digest"}):
            with self.subTest(pins=pins), self.assertRaises(PolicyError):
                self.q.nominate([p, z], decision_sha=digest("substitution"), witnesses=valid, payloads=pins)
        self.q.freeze_early(reason_sha=digest("keep earlier proposal"))
        self.assertEqual(self.q.seal("freeze")["payload"]["decision_sha"], DECISION)
        self.assertEqual(sum(e["kind"] == "nominated" for e in self.q.events()), 1)

    def test_role_is_checked_even_if_both_registered_config_hashes_agree(self):
        p = self.pair(Role.SELECTED, method())
        cases = [(Role.STOCK, method(2, "naive")), (Role.ZERO_EXTRA, method(2, "naive")),
                 (Role.ZERO_EXTRA, method(1, "zero_extra")), (Role.ZERO_EXTRA, ensemble()),
                 (Role.ORDER_ENSEMBLE, zero()), (Role.ORDER_ENSEMBLE, ensemble(loop={"k": 2})),
                 (Role.SELECTED, zero()), (Role.SELECTED, method(1, "naive")),
                 (Role.SELECTED, method(loop={"k": 2, "readout": "diagnostic"})),
                 (Role.SELECTED, method(2, "naive", policy={"pool": "first"}))]
        for role, spec in cases:
            with self.subTest(role=role, spec=spec):
                if role == Role.SELECTED:
                    bad = self.pair(role, spec)
                    picks = [bad, self.pair(Role.ZERO_EXTRA, zero(), parent=bad)]
                else:
                    bad = self.pair(role, spec, parent=None if role == Role.STOCK else p)
                    picks = [p, bad]
                with self.assertRaises(PolicyError):
                    self.nominate(*picks)

    def test_controls_require_success_and_same_method_on_both_splits(self):
        p = self.pair(Role.SELECTED, method())
        pending = self.pair(Role.ZERO_EXTRA, zero(), parent=p, complete=False)
        with self.assertRaisesRegex(PolicyError, "completed dev"):
            self.nominate(p, pending)
        z = self.pair(Role.ZERO_EXTRA, zero(), parent=p)
        splice = replace(z, test_id=p.test_id)
        self.witnesses[splice] = self.witnesses[z]
        with self.assertRaisesRegex(PolicyError, "only the split-specific dataset"):
            self.nominate(p, splice)

    def test_parent_must_be_the_selected_pair_on_both_views_and_settings(self):
        p = self.pair(Role.SELECTED, method())
        other = self.pair(Role.SELECTED, method(2, "naive"))  # Successful but never selected.
        for parent in (None, other, replace(p, role=Role.STOCK), replace(p, test_id=other.test_id)):
            z = self.pair(Role.ZERO_EXTRA, zero(), parent=parent)
            with self.subTest(parent=parent), self.assertRaisesRegex(PolicyError, "exact selected pair"):
                self.nominate(p, z)
        for key_change in ({"seed": 24}, {"scorer": digest("different scorer")}, {"precision": "fp32"},
                           {"code": digest("different code")}, {"dev_data": digest("other dev")},
                           {"test_data": digest("other test")}, {"model": COMPACT}):
            fields = COMPACT_SETTINGS if "model" in key_change else {}
            z = self.pair(Role.ZERO_EXTRA, zero(**fields), parent=p, **key_change)
            with self.subTest(key_change=key_change), self.assertRaisesRegex(PolicyError, "trial identity"):
                self.nominate(p, z, pins=PINS if fields else None)
        for role, spec in ((Role.ZERO_EXTRA, zero(window=[4, 8])), (Role.ZERO_EXTRA, zero(reverse=True)),
                           (Role.ORDER_ENSEMBLE, ensemble(window=[4, 8]))):
            c = self.pair(role, spec, parent=p)
            with self.subTest(role=role, spec=spec), self.assertRaisesRegex(PolicyError, "window and option order"):
                self.nominate(p, c)

    def test_compact_preimages_and_independent_payload_pins_must_agree(self):
        pins = {COMPACT: PIN}
        p = self.pair(Role.SELECTED, method(**COMPACT_SETTINGS), model=COMPACT)
        for spec in (zero(), zero(compact_recipe="select-20"),
                     zero(**{**COMPACT_SETTINGS, "compact_manifest_sha256": digest("other receipt")}),
                     zero(compact_manifest_sha256=PIN),
                     zero(**{**COMPACT_SETTINGS, "compact_recipe": "select-16"})):
            z = self.pair(Role.ZERO_EXTRA, spec, model=COMPACT, parent=p)
            with self.subTest(spec=spec), self.assertRaises(PolicyError):
                self.nominate(p, z, pins=pins)
        for spec in (ensemble(), ensemble(**COMPACT_SETTINGS)):
            e = self.pair(Role.ORDER_ENSEMBLE, spec, model=COMPACT, parent=p)
            with self.subTest(spec=spec), self.assertRaises(PolicyError):
                self.nominate(p, e, pins=pins)
        # Comparing two sets of settings alone would accept these joint omissions/substitutions.
        for model, fields, expected in ((COMPACT, {}, PIN), (FULL, COMPACT_SETTINGS, None)):
            candidate = self.pair(Role.SELECTED, method(**fields), model=model)
            z = self.pair(Role.ZERO_EXTRA, zero(**fields), model=model, parent=candidate)
            with self.subTest(model=model), self.assertRaisesRegex(PolicyError, "independently pinned"):
                self.nominate(candidate, z, pins={model: expected})

    def test_spend_fraction_must_match_without_rounding_or_clipping(self):
        p = self.pair(Role.SELECTED, method())
        for spec in (ensemble(1, 6), ensemble(3333, 10000), ensemble(1, 1), ensemble(True, 3),
                     ensemble(1, 0), ensemble(seed=7)):
            e = self.pair(Role.ORDER_ENSEMBLE, spec, parent=p)
            with self.subTest(spec=spec), self.assertRaises(PolicyError):
                self.nominate(p, e)
        p = self.pair(Role.SELECTED, method(4, window=[4, 16]))
        e = self.pair(Role.ORDER_ENSEMBLE, ensemble(1, 1, window=[4, 16]), parent=p)
        with self.assertRaisesRegex(PolicyError, "extra layer visits"):
            self.nominate(p, e)  # 36 extra visits cannot be matched by even 100% of groups.

    def test_no_extra_winner_stock_or_control_even_via_a_second_view(self):
        p = self.pair(Role.SELECTED, method())
        for role, first, second in (
            (Role.STOCK, method(1, "naive"), method(loop={"k": 1})),
            (Role.SELECTED, method(), method(2, "naive")),
            (Role.ZERO_EXTRA, zero(), method(3, "zero_extra")),
            (Role.ORDER_ENSEMBLE, ensemble(), ensemble(1, 3)),
        ):
            parent = p if role in (Role.ZERO_EXTRA, Role.ORDER_ENSEMBLE) else None
            one = self.pair(role, first, parent=parent)
            for view in (digest("held out"), digest("another held out")):
                extra = self.pair(role, second, parent=parent, test_data=view)
                with self.subTest(role=role, view=view), self.assertRaisesRegex(PolicyError, "one .*role"):
                    self.nominate(*([p] if role != Role.SELECTED else []), one, extra)

    def test_opaque_legacy_pair_keeps_original_seal_shape_without_witnesses(self):
        p = self.pair(Role.SELECTED, {}, config=digest("opaque legacy config"))
        stock = replace(p, role=Role.STOCK)
        self.q.nominate([stock, p], decision_sha=DECISION)
        self.restart()
        self.clock.now = self.q.freeze_at
        seal = self.q.seal("freeze")
        rows = [{**asdict(pick), "dev_result_sha": digest("result:" + p.dev_id),
                 "dev_manifest_sha": digest("manifest:" + p.dev_id),
                 "test_manifest_sha": digest("manifest:" + p.test_id)} for pick in (p, stock)]
        original = {"name": "freeze", "run_id": RUN, "effective_at": self.clock.now,
                    "decision_sha": DECISION, "picks": rows}
        self.assertEqual(seal["payload"], original)
        self.assertEqual(seal["sha256"], sha(canonical(original)))


if __name__ == "__main__":
    unittest.main()
