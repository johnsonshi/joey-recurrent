"""CPU-only queue tests: synthetic digests, simulated time and in-memory workers."""

import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from joey_research.scheduler import (
    Admission, BackendUnavailable, ClockRegression, Gate, HOUR, Manifest, Observation,
    Outcome, Pick, PolicyError, Queue, Role, Split, TrialKey,
)


def digest(name):
    return hashlib.sha256(name.encode("ascii")).hexdigest()


START = 1_000_000
RUN = digest("synthetic-run")


class Clock:
    def __init__(self, now=START):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def manifest(config, *, split=Split.DEV, gpu="class-a", budget=120):
    key = TrialKey(digest("model"), digest(split.value), digest("code"), digest(config),
                   23, digest("scorer"), "fp32")
    return Manifest(digest("manifest:" + key.trial_id), key, split, gpu, budget)


def receipt(ticket, clock, outcome=Outcome.SUCCEEDED):
    return Observation(ticket.token, outcome, clock(), digest("fence:" + ticket.token),
                       digest("result:" + ticket.trial_id) if outcome == Outcome.SUCCEEDED else None)


class PhysicalDouble:
    """Independent occupancy and tombstones; no process, evaluator or label data."""

    def __init__(self, clock):
        self.clock = clock
        self.workers = {}
        self.tombstones = {}
        self.sent = []
        self.observed = []
        self.ambiguous = False
        self.starts = 0
        self.peak = 0

    def admit(self, ticket: Admission):
        self.sent.append(ticket.token)
        if ticket.token in self.tombstones:
            return self.tombstones[ticket.token]
        existing = self.workers.get(ticket.slot)
        if existing and existing.token == ticket.token:
            return Observation(ticket.token, Outcome.LIVE, self.clock())
        if self.clock() >= ticket.admit_before:
            self.tombstones[ticket.token] = receipt(ticket, self.clock, Outcome.FENCED_ABSENT)
            return self.tombstones[ticket.token]
        if existing:
            raise AssertionError("a second physical worker would occupy the same slot")
        self.workers[ticket.slot] = ticket
        self.starts += 1
        self.peak = max(self.peak, len(self.workers))
        if self.ambiguous:
            raise BackendUnavailable("simulated lost reply after a start")
        return Observation(ticket.token, Outcome.LIVE, self.clock())

    def observe(self, ticket: Admission):
        self.observed.append(ticket.token)
        if ticket.token in self.tombstones:
            return self.tombstones[ticket.token]
        if not self.ambiguous and self.workers.get(ticket.slot) == ticket:
            return Observation(ticket.token, Outcome.LIVE, self.clock())
        return Observation(ticket.token, Outcome.UNKNOWN)  # Ordinary absence is not a fence.

    def finish(self, ticket, outcome=Outcome.SUCCEEDED):
        if self.workers.pop(ticket.slot) != ticket:
            raise AssertionError("incorrect worker identity")
        self.tombstones[ticket.token] = receipt(ticket, self.clock, outcome)
        return self.tombstones[ticket.token]


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "queue.sqlite"
        self.clock = Clock()
        self.q = Queue.create(self.db, run_id=RUN, start_utc=START, gpu_classes=("class-a", "class-a"),
                              lease_s=10, clock=self.clock)
        self.addCleanup(lambda: self.q.close())

    def restart(self):
        self.q.close()
        self.q = Queue.resume(self.db, run_id=RUN, clock=self.clock)

    def add(self, m, ready=True, **options):
        tid = self.q.register(m, **options)
        if ready:
            for gate in Gate:
                self.q.attest(tid, gate, digest(gate.value + m.sha256))
        return tid

    def pair(self, name):
        dev = self.add(manifest(name))
        test = self.add(manifest(name, split=Split.TEST))
        ticket = self.q.reserve()
        self.assertEqual(ticket.trial_id, dev)
        self.q.record(receipt(ticket, self.clock))
        return Pick(dev, test)

    def test_seven_field_dedup_and_immutable_descriptor(self):
        m = manifest("one")
        tid = self.add(m)
        self.assertEqual(self.q.register(m), tid)
        self.assertEqual(len(self.q.registry()), 1)
        for field in ("model", "dataset", "code", "config", "seed", "scorer", "precision"):
            change = {"seed": 24, "precision": "bf16"}.get(field, digest("other"))
            key = replace(m.key, **{field: change})
            self.assertNotEqual(self.q.register(replace(m, key=key, sha256=digest(field))), tid)
        with self.assertRaises(PolicyError):
            self.q.register(replace(m, sha256=digest("repackaged")))
        with self.assertRaises(PolicyError):
            self.q.register(replace(m, budget_s=121))
        with self.assertRaises(PolicyError):
            self.q.attest(tid, Gate.STOCK_PARITY, digest("changed-proof"))
        with self.assertRaises(PolicyError):
            self.q.register(replace(m, split=Split.TEST, sha256=digest("contaminated"),
                                    key=replace(m.key, config=digest("new-config"))))
        with self.assertRaises(PolicyError):
            self.q.register(manifest("no-test-retry", split=Split.TEST), max_attempts=2)

    def test_native_measurement_needs_real_staging_but_not_a_fabricated_parity_gate(self):
        import json
        from joey_research.scheduler import _hash
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            q=Queue.create(Path(d)/"native.sqlite",run_id=RUN,start_utc=START,
                           gpu_classes=("A100-loop","A100-compact"),clock=self.clock)
            try:
                def register(k):
                    settings={"family":"additive","loop":{"k":k,"method":"naive"},"window":[8,12]}
                    payload=(json.dumps({"mode":"screen","split":"discovery","branch":"loop","settings":settings},
                                        separators=(",",":"),sort_keys=True)+"\n").encode()
                    m=manifest("native"+str(k),gpu="A100-loop")
                    m=replace(m,key=replace(m.key,config=_hash(settings)),sha256=hashlib.sha256(payload).hexdigest())
                    return q.register(m),payload
                native,payload=register(1)
                other,recurrent=register(2)
                self.assertIsNone(q.reserve_native_control(native,payload))
                for tid in (native,other):q.attest(tid,Gate.OFFLINE_STAGING,digest("actual-stage"))
                self.assertIsNone(q.reserve())
                with self.assertRaises(PolicyError):q.reserve_native_control(other,recurrent)
                with self.assertRaises(PolicyError):q.reserve_native_control(other,payload)
                t=q.reserve_native_control(native,payload)
                self.assertEqual((t.slot,t.trial_id),(0,native))
                self.assertIsNone(q.reserve_native_control(native,payload))
                self.assertEqual(q.status()["gpu_budget"]["reserved_s"],120)
                self.assertIsNone(next(r for r in q.registry() if r["id"]==native)["parity_sha"])
            finally:q.close()

    def test_gates_dependency_and_gpu_class_are_independent(self):
        parent = self.add(manifest("prerequisite"), ready=False, priority=100)
        child = self.add(manifest("child"), requires=[parent], priority=0)
        wrong = self.add(manifest("wrong-slot", gpu="class-b"), priority=0)
        held = self.add(manifest("held", split=Split.TEST))
        with self.assertRaises(PolicyError):
            self.q.register(manifest("leak"), requires=[held])
        self.assertIsNone(self.q.reserve())
        snapshot = self.q.snapshot()
        waiting = {r["id"]: r["waiting_for"] for r in snapshot["registry"]}
        self.assertEqual(waiting[parent], ["offline_staging", "stock_parity"])
        self.assertEqual(waiting[child], ["prerequisite"])
        self.assertEqual(waiting[wrong], ["gpu_class"])
        self.assertEqual(waiting[held], ["freeze"])
        self.q.attest(parent, Gate.OFFLINE_STAGING, digest("stage"))
        self.assertIsNone(self.q.reserve())
        self.q.attest(parent, Gate.STOCK_PARITY, digest("parity"))
        ticket = self.q.reserve()
        self.assertEqual(ticket.trial_id, parent)
        self.assertIsNone(self.q.reserve())
        self.q.record(receipt(ticket, self.clock))
        self.assertEqual(self.q.reserve().trial_id, child)
        self.assertIsNone(self.q.reserve())
        rows = {r["id"]: r for r in self.q.registry()}
        self.assertEqual(rows[wrong]["attempts"], 0)
        self.assertEqual(rows[child]["requires"], [parent])
        self.assertEqual(rows[parent]["offline_sha"], digest("stage"))

    def test_distinct_physical_classes_and_stable_priority_tie(self):
        path = self.db.with_name("class-pair.sqlite")
        q = Queue.create(path, run_id=RUN, start_utc=START, gpu_classes=("b", "a"), clock=self.clock)
        self.addCleanup(q.close)
        a = sorted([manifest("x", gpu="a"), manifest("y", gpu="a")],
                   key=lambda m: m.key.trial_id, reverse=True)
        b = manifest("z", gpu="b")
        for m in [*a, b]:
            tid = q.register(m, priority=0 if m == b else 1)
            for gate in Gate:
                q.attest(tid, gate, digest(gate.value))
        first, second = q.reserve(), q.reserve()
        self.assertEqual((first.slot, first.trial_id), (0, b.key.trial_id))
        self.assertEqual((second.slot, second.trial_id), (1, a[-1].key.trial_id))
        self.assertIsNone(q.reserve())

    def test_targeted_reservation_never_returns_another_trial_or_bypasses_either_gate(self):
        ready = self.add(manifest("ready"))
        held = self.add(manifest("not-yet-verified"), ready=False)
        self.assertIsNone(self.q.reserve(trial_id=held))
        self.q.attest(held, Gate.OFFLINE_STAGING, digest("stage-only"))
        self.assertIsNone(self.q.reserve(trial_id=held))
        t = self.q.reserve(trial_id=ready)
        self.assertEqual(t.trial_id, ready)
        self.assertIsNone(self.q.reserve(trial_id=ready))

    def test_independent_connections_race_for_two_slots(self):
        for index in range(8):
            self.add(manifest(str(index)))
        barrier = threading.Barrier(8)

        def contend(_):
            q = Queue.resume(self.db, run_id=RUN, clock=self.clock)
            try:
                barrier.wait(timeout=10)
                return q.reserve()
            finally:
                q.close()

        with ThreadPoolExecutor(max_workers=8) as pool:
            winners = [t for t in pool.map(contend, range(8)) if t]
        self.assertEqual(sorted(t.slot for t in winners), [0, 1])
        self.assertEqual(len({t.trial_id for t in winners}), 2)
        self.assertEqual({t.token for t in winners}, {t.token for t in self.q.outstanding()})

    def test_ambiguous_starts_survive_restart_without_third_worker(self):
        for name in ("first", "second", "third"):
            self.add(manifest(name), max_attempts=3)
        backend = PhysicalDouble(self.clock)
        backend.ambiguous = True
        self.q.tick(backend)
        original = list(backend.workers.values())
        self.assertEqual(backend.starts, 2)
        self.clock.advance(11)
        self.restart()
        for _ in range(3):
            status = self.q.tick(backend)
            self.assertEqual([s["state"] for s in status["slots"]], ["reconciling", "reconciling"])
        self.assertEqual(len(backend.sent), 2)  # No replay on restart.
        done = backend.finish(original[0])
        self.q.tick(backend)
        self.assertEqual((backend.starts, backend.peak), (3, 2))
        self.assertEqual(backend.workers[original[0].slot].generation, original[0].generation + 1)
        self.assertEqual(backend.admit(original[0]), done)  # A delayed RPC cannot resurrect it.
        self.q.record(done)
        self.q.record(Observation(original[0].token, Outcome.LIVE, self.clock()))
        self.assertIsNone(self.q.reserve())

    def test_crash_before_send_needs_irreversible_absence_and_consumes_retry(self):
        tid = self.add(manifest("lost"), max_attempts=2)
        old = self.q.reserve()
        self.restart()
        backend = PhysicalDouble(self.clock)
        self.clock.advance(10)
        self.q.tick(backend)
        self.assertEqual((backend.sent, backend.starts), ([], 0))
        self.assertEqual(self.q.outstanding(), [old])
        backend.tombstones[old.token] = receipt(old, self.clock, Outcome.FENCED_ABSENT)
        self.q.tick(backend)
        new, = self.q.outstanding()
        self.assertEqual((new.trial_id, new.attempt), (tid, 2))
        self.assertEqual(backend.admit(old).outcome, Outcome.FENCED_ABSENT)
        self.q.record(backend.finish(new, Outcome.RETRYABLE))
        self.assertIsNone(self.q.reserve())
        self.assertEqual([x["observation"]["outcome"] for x in self.q.history(tid)],
                         ["fenced_absent", "retryable_failure"])
        self.assertEqual(self.q.registry()[0]["state"], "failed")

    def test_delayed_still_valid_request_cannot_resurrect_a_fenced_generation(self):
        self.add(manifest("delayed"), max_attempts=2)
        old = self.q.reserve()
        backend = PhysicalDouble(self.clock)
        backend.tombstones[old.token] = receipt(old, self.clock, Outcome.FENCED_ABSENT)
        self.q.record(backend.observe(old))
        self.q.tick(backend)
        new, = self.q.outstanding()
        self.assertLess(self.clock(), old.admit_before)
        self.assertEqual((new.slot, new.generation), (old.slot, old.generation + 1))
        self.assertEqual(backend.admit(old).outcome, Outcome.FENCED_ABSENT)
        self.assertEqual((backend.starts, backend.workers[old.slot]), (1, new))

    def test_stale_pulses_cannot_renew_or_release_a_lease(self):
        self.add(manifest("pulse", budget=25))
        t = self.q.reserve()
        spare = self.add(manifest("spare"), priority=200)
        self.clock.advance(7)
        pulse = Observation(t.token, Outcome.LIVE, self.clock())
        self.q.record(pulse)
        self.assertEqual(self.q.status()["slots"][0]["lease_until"], START + 17)
        self.clock.advance(10)
        self.q.record(pulse)
        self.assertEqual(self.q.status()["slots"][0]["state"], "reconciling")
        with self.assertRaises(PolicyError):
            self.q.record(receipt(t, self.clock, Outcome.FENCED_ABSENT))
        with self.assertRaises(PolicyError):
            self.q.record(Observation(t.token, Outcome.LIVE, self.clock() + 1))
        self.q.record(Observation(t.token, Outcome.LIVE, self.clock()))
        self.assertEqual(self.q.status()["slots"][0]["lease_until"], t.hard_stop)
        self.clock.now = t.hard_stop + 1
        self.q.record(Observation(t.token, Outcome.LIVE, self.clock()))
        self.assertEqual(self.q.status()["slots"][0]["state"], "reconciling")
        self.assertIsNone(self.q.reserve())  # Unfenced post-stop activity withholds all headroom.
        self.q.record(receipt(t, self.clock))
        self.assertGreater(self.q.status()["gpu_budget"]["charged_s"], 25)
        self.assertEqual(self.q.reserve().trial_id, spare)
        self.assertEqual(self.q.registry()[0]["state"], "expired")
        self.assertIsNone(self.q.registry()[0]["result_sha"])

    def test_restart_uses_original_post_setup_origin_and_exact_budget_boundaries(self):
        self.clock.now = START + 23 * HOUR - 30
        path = self.db.with_name("after-setup.sqlite")
        q = Queue.create(path, run_id=RUN, start_utc=START, gpu_classes=("a", "a"),
                         prior_gpu_s=15, clock=self.clock)
        try:
            for duration in (31, 30):
                tid = q.register(manifest(str(duration), gpu="a", budget=duration), priority=100-duration)
                for gate in Gate:
                    q.attest(tid, gate, digest(gate.value))
            t = q.reserve()
            self.assertEqual(t.hard_stop, START + 23 * HOUR)
            self.assertEqual(t.trial_id, manifest("30", gpu="a", budget=30).key.trial_id)
            self.assertIsNone(q.reserve())
        finally:
            q.close()
        self.clock.advance(30)
        q = Queue.resume(path, run_id=RUN, clock=self.clock)
        self.addCleanup(q.close)
        self.assertEqual(q.status()["phase"], "test")
        self.assertEqual(q.status()["gpu_budget"]["remaining_s"], 48 * HOUR - 15 - 30)
        self.assertEqual(q.seal("freeze")["payload"]["picks"], [])
        q.record(receipt(t, self.clock))
        self.assertFalse(q.history(t.trial_id)[0]["accepted"])  # Cannot affect an already-frozen plan.
        self.clock.now = START + 26 * HOUR
        self.assertEqual(q.status()["phase"], "closed")
        self.assertEqual(q.seal("finalize")["effective_at"], START + 25 * HOUR)
        self.assertIsNone(q.reserve())
        with self.assertRaises(FileExistsError):
            Queue.create(path, run_id=RUN, start_utc=self.clock(), gpu_classes=("a", "a"), clock=self.clock)
        with self.assertRaises(PolicyError):
            Queue.resume(path, run_id=digest("wrong-run"), clock=self.clock)
        self.clock.advance(-1)
        with self.assertRaises(ClockRegression):
            q.reserve()

    def test_once_only_freeze_shared_stock_selected_and_test_receipt(self):
        p = self.pair("winner")
        other = self.pair("other")
        with self.assertRaises(PolicyError):
            self.q.nominate([Pick(p.dev_id, other.test_id)], decision_sha=digest("bad-config"))
        with self.assertRaises(PolicyError):
            self.q.nominate([p, other], decision_sha=digest("two-winners"))
        self.q.nominate([other], decision_sha=digest("earlier"))
        self.q.nominate([replace(p, role=Role.STOCK), p], decision_sha=digest("validated-selection"))
        self.assertIsNone(self.q.reserve())  # No held-out execution before freeze.
        self.clock.now = self.q.freeze_at
        with self.assertRaises(PolicyError):
            self.q.nominate([other], decision_sha=digest("too-late"))
        seal = self.q.seal("freeze")
        self.assertEqual(seal["payload"]["decision_sha"], digest("validated-selection"))
        self.assertEqual({r["role"] for r in seal["payload"]["picks"]}, {"stock", "selected"})
        self.restart()
        self.assertEqual(self.q.seal("freeze"), seal)
        t = self.q.reserve()
        self.assertEqual((t.trial_id, t.split), (p.test_id, Split.TEST))
        self.assertIsNone(self.q.reserve())
        evidence = receipt(t, self.clock)
        self.q.record(evidence)
        self.restart()
        self.q.record(evidence)
        with self.assertRaises(PolicyError):
            self.q.record(replace(evidence, result_sha=digest("alternate-result")))
        self.assertIsNone(self.q.reserve())
        self.clock.now = self.q.finalize_at
        final = self.q.seal("finalize")["payload"]
        self.assertEqual(final["freeze_sha"], seal["sha256"])
        self.assertEqual([r["result_sha"] for r in final["results"]], [evidence.result_sha])
        self.assertIs(final["results"][0]["accepted"], True)
        self.assertEqual(self.q.history(other.test_id), [])
        events = self.q.events()
        self.assertEqual(sum(e["kind"] == "freeze" for e in events), 1)
        self.assertEqual(sum(e["kind"] == "terminal" and e["trial_id"] == p.test_id for e in events), 1)
        self.assertEqual(self.q.events(after=events[-1]["seq"]), [])
        json.dumps(self.q.registry())
        json.dumps(self.q.status())

    def test_holdout_not_retried_after_maybe_admitted_or_retryable_error(self):
        p = self.pair("final")
        self.q.nominate([p], decision_sha=digest("selection"))
        self.clock.now = self.q.freeze_at
        backend = PhysicalDouble(self.clock)
        backend.ambiguous = True
        self.q.tick(backend)
        self.clock.advance(11)
        self.restart()
        self.q.tick(backend)
        t, = self.q.outstanding()
        backend.finish(t, Outcome.RETRYABLE)
        self.q.tick(backend)
        self.restart()
        self.q.tick(backend)
        self.assertEqual((backend.sent, backend.starts), ([t.token], 1))
        self.assertEqual(len(self.q.history(p.test_id)), 1)
        self.assertFalse(self.q.history(p.test_id)[0]["accepted"])

    def test_freeze_does_not_evict_stale_workers_to_make_room_for_test(self):
        p = self.pair("selected")
        self.q.nominate([p], decision_sha=digest("decision"))
        for name in ("left", "right"):
            self.add(manifest(name, budget=30), max_attempts=2)
        self.clock.now = self.q.freeze_at - 30
        backend = PhysicalDouble(self.clock)
        backend.ambiguous = True
        self.q.tick(backend)
        devs = list(backend.workers.values())
        self.clock.now = self.q.freeze_at
        self.restart()
        status = self.q.tick(backend)
        self.assertEqual([s["state"] for s in status["slots"]], ["reconciling", "reconciling"])
        self.assertEqual(backend.starts, 2)
        backend.finish(devs[0], Outcome.RETRYABLE)
        self.q.tick(backend)
        self.assertEqual((backend.workers[devs[0].slot].trial_id, backend.peak), (p.test_id, 2))
        self.assertEqual(len(self.q.history(devs[0].trial_id)), 1)

    def test_test_must_fit_before_finalize_and_nonadmission_is_explicit(self):
        p = self.pair("insufficient-time")
        self.q.nominate([p], decision_sha=digest("decision"))
        self.clock.now = self.q.finalize_at - 119
        self.assertIsNone(self.q.reserve())
        pending = next(r for r in self.q.registry() if r["id"] == p.test_id)
        self.assertEqual(pending["waiting_for"], ["time_budget"])
        self.clock.now = self.q.finalize_at
        final, = self.q.seal("finalize")["payload"]["results"]
        self.assertEqual((final["state"], final["result_sha"], final["accepted"], final["receipt"]),
                         ("expired", None, None, None))

    def test_no_subsequent_backend_calls_when_observation_spans_deadline(self):
        for name in ("slow", "unobserved"):
            self.add(manifest(name))
        backend = PhysicalDouble(self.clock)
        self.q.tick(backend)
        self.clock.now = self.q.deadline - 1

        def crossing(ticket):
            backend.observed.append(ticket.token)
            self.clock.advance(1)
            return Observation(ticket.token, Outcome.UNKNOWN)

        backend.observe = crossing
        status = self.q.tick(backend)
        self.assertEqual((len(backend.observed), len(backend.sent)), (1, 2))
        self.assertEqual(status["phase"], "closed")
        self.assertTrue(all(s["state"] == "reconciling" for s in status["slots"]))

    def test_late_receipt_cannot_amend_finalized_evidence_or_rerun_test(self):
        p = self.pair("late")
        self.q.nominate([p], decision_sha=digest("decision"))
        self.clock.now = self.q.finalize_at - 121
        t = self.q.reserve()
        self.clock.now = t.hard_stop
        done_before_cutoff = receipt(t, self.clock)
        self.clock.now = self.q.deadline + 50
        self.restart()
        final = self.q.seal("finalize")
        self.assertEqual(final["payload"]["results"][0]["state"], "active")
        backend = PhysicalDouble(self.clock)
        self.q.tick(backend)
        self.assertEqual((backend.sent, backend.observed), ([], []))
        self.q.record(done_before_cutoff)  # A late physical audit, not a scientific result.
        self.assertEqual(self.q.seal("finalize"), final)
        self.assertFalse(self.q.history(p.test_id)[0]["accepted"])
        self.assertIsNone(self.q.reserve())

    def test_driver_runs_fake_day_and_preserves_finalize_reserve(self):
        # Large fake poll interval; no worker is ever created.
        path = self.db.with_name("driver.sqlite")
        q = Queue.create(path, run_id=RUN, start_utc=START, gpu_classes=("a", "b"), lease_s=HOUR,
                         clock=self.clock)
        self.addCleanup(q.close)
        backend = PhysicalDouble(self.clock)
        status = q.run(backend, poll_s=HOUR, sleep=self.clock.advance)
        self.assertEqual((status["phase"], status["as_of"]), ("closed", START + 26 * HOUR))
        self.assertEqual([(q.seal(k)["effective_at"], q.seal(k)["recorded_at"]) for k in ("freeze", "finalize")],
                         [(START + 23 * HOUR,) * 2, (START + 25 * HOUR,) * 2])
        self.assertEqual((backend.sent, backend.observed), ([], []))

    def test_last_gpu_seconds_cannot_be_reserved_twice_or_regained_on_restart(self):
        # Dev uses both slots for 23h. The 2h test window has room for 4 GPU-hours
        # on the clock, but only 2 in the cumulative budget.
        picks = []
        for name, role, budget, priority in (("stock", Role.STOCK, 2 * HOUR, 0), ("pick", Role.SELECTED, 1, 1)):
            dev = self.add(manifest(name, budget=23 * HOUR))
            test = self.add(manifest(name, split=Split.TEST, budget=budget), priority=priority)
            picks.append(Pick(dev, test, role))
        devs = [self.q.reserve(), self.q.reserve()]
        self.clock.now = self.q.freeze_at - 1
        for ticket in devs:
            self.q.record(receipt(ticket, self.clock))
        self.q.nominate(picks, decision_sha=digest("dev evidence only"))
        self.clock.advance(1)
        barrier = threading.Barrier(2)

        def contend(_):
            q = Queue.resume(self.db, run_id=RUN, clock=self.clock)
            try:
                barrier.wait(timeout=10)
                return q.reserve()
            finally:
                q.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            ticket, = [t for t in pool.map(contend, range(2)) if t]
        self.assertEqual(ticket.trial_id, picks[0].test_id)
        self.assertEqual(ticket.hard_stop, self.q.finalize_at)  # Exact fit on both limits.
        self.q.record(Observation(ticket.token, Outcome.UNKNOWN))
        self.clock.advance(11)
        self.restart()
        snapshot = self.q.snapshot()
        self.assertEqual(snapshot["status"]["gpu_budget"],
                         {"cap_s": 48 * HOUR, "charged_s": 46 * HOUR, "reserved_s": 2 * HOUR, "remaining_s": 0})
        waiting = next(t for t in snapshot["registry"] if t["id"] == picks[1].test_id)
        self.assertEqual(waiting["waiting_for"], ["gpu_budget"])  # A physically free slot is not sufficient.
        self.assertIsNone(self.q.reserve())
        self.clock.now = self.q.finalize_at - 1
        self.q.record(receipt(ticket, self.clock))
        self.restart()
        self.assertEqual(self.q.status()["gpu_budget"]["charged_s"], 48 * HOUR)
        self.assertIsNone(self.q.reserve())  # The 1s pick still fits on the wall clock.
        self.q.finish_early(reason_sha=digest("cap exhausted"))
        skipped = next(r for r in self.q.seal("finalize")["payload"]["results"] if r["test_id"] == picks[1].test_id)
        self.assertEqual((skipped["state"], skipped["receipt"]), ("expired", None))

    def test_fenced_actual_time_replaces_full_reservation_exactly_once_across_retries(self):
        self.add(manifest("retry", budget=60), max_attempts=2)
        first = self.q.reserve()
        self.clock.advance(9)
        ended = receipt(first, self.clock, Outcome.RETRYABLE)
        self.clock.advance(90)
        self.q.record(Observation(first.token, Outcome.UNKNOWN))
        self.restart()
        self.assertEqual(self.q.status()["gpu_budget"]["reserved_s"], 60)  # Even past hard_stop.
        self.q.record(ended)  # Actual terminal second, not the late poll.
        self.restart()
        self.q.record(ended)
        second = self.q.reserve()
        self.assertEqual(self.q.status()["gpu_budget"], {"cap_s": 172800, "charged_s": 10,
                                                        "reserved_s": 60, "remaining_s": 172730})
        self.clock.advance(7)
        self.q.record(receipt(second, self.clock, Outcome.FENCED_ABSENT))
        self.restart()
        self.assertEqual(self.q.status()["gpu_budget"]["remaining_s"], 172790)
        self.assertIsNone(self.q.reserve())  # Never-started consumed its attempt, not GPU time.

    def test_early_freeze_and_finish_keep_real_times_and_all_feasible_picks(self):
        pick = self.pair("early")
        self.q.nominate([pick], decision_sha=digest("dev decision"))
        peer = Queue.resume(self.db, run_id=RUN, clock=self.clock)
        self.addCleanup(peer.close)
        self.add(manifest("other dev"))
        active = self.q.reserve()
        self.q.record(Observation(active.token, Outcome.UNKNOWN))
        reason = digest("dev stopping rule")
        with self.assertRaises(PolicyError):
            peer.freeze_early(reason_sha=reason)
        self.q.record(receipt(active, self.clock, Outcome.FENCED_ABSENT))
        with self.assertRaises(PolicyError):
            peer.finish_early(reason_sha=reason)
        self.assertIsNone(self.q.seal("freeze"))  # The rejected close was atomic.
        self.clock.advance(32)
        peer.freeze_early(reason_sha=reason)
        freeze = self.q.seal("freeze")
        self.assertEqual((freeze["effective_at"], freeze["recorded_at"], freeze["payload"]["reason_sha"]),
                         (self.clock(), self.clock(), reason))
        self.assertEqual(self.q.finalize_at, self.clock() + 2 * HOUR)
        with self.assertRaises(PolicyError):
            self.q.nominate([], decision_sha=digest("change after freeze"))
        with self.assertRaises(PolicyError):
            self.q.extend_once(hours=1, evidence_sha=reason)
        t = self.q.reserve()
        self.assertEqual(t.trial_id, pick.test_id)
        with self.assertRaises(PolicyError):
            peer.finish_early(reason_sha=reason)
        self.clock.advance(5)
        evidence = receipt(t, self.clock)
        self.q.record(evidence)
        self.restart()
        status = self.q.finish_early(reason_sha=digest("work complete"))
        final = peer.seal("finalize")
        self.assertEqual((status["phase"], final["recorded_at"], final["effective_at"]),
                         ("closed", self.clock(), self.clock()))
        self.assertEqual(final["payload"]["results"][0]["result_sha"], evidence.result_sha)
        self.assertEqual(peer.run(PhysicalDouble(self.clock), sleep=self.fail)["as_of"], self.clock())
        self.clock.now = START + 30 * HOUR  # CPU verification/reporting overrun, no additional GPU work.
        self.restart()
        self.assertEqual(self.q.snapshot()["status"]["event_seq"], status["event_seq"])
        self.assertIsNone(self.q.reserve())
        self.assertEqual((self.q.seal("freeze"), self.q.seal("finalize")), (freeze, final))

    def test_v1_resume_accounts_for_existing_evidence_without_extending_old_cutoffs(self):
        self.add(manifest("settled"))
        closed = self.q.reserve()
        self.clock.advance(6)
        self.q.record(receipt(closed, self.clock, Outcome.FAILED))
        self.add(manifest("unknown", budget=80))
        outstanding = self.q.reserve()
        self.q.close()
        # A v1 DB has the same durable tickets/receipts, but no run_budget table.
        with sqlite3.connect(self.db) as db:
            db.execute("DROP TABLE run_budget")
            db.execute("PRAGMA user_version=1")
        self.clock.now = START
        with self.assertRaises(ClockRegression):
            Queue.resume(self.db, run_id=RUN, clock=self.clock)
        with sqlite3.connect(self.db) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 1)  # Entire upgrade rolled back.
        self.clock.now = START + 22 * HOUR
        self.q = Queue.resume(self.db, run_id=RUN, clock=self.clock)
        status = self.q.status()
        self.assertEqual((status["phase"], self.q.freeze_at, self.q.finalize_at, self.q.deadline),
                         ("test", START + 21 * HOUR, START + 23 * HOUR, START + 24 * HOUR))
        self.assertEqual((status["gpu_budget"]["charged_s"], status["gpu_budget"]["reserved_s"]), (7, 80))
        self.assertEqual(self.q.outstanding(), [replace(outstanding, run_deadline=START + 24 * HOUR)])
        self.restart()
        self.assertEqual(self.q.status(), status)

    def test_explicit_extension_shifts_new_work_only_and_survives_restart(self):
        self.add(manifest("long", budget=27 * HOUR), priority=0)
        self.add(manifest("short"))
        old = self.q.reserve()
        self.assertIsNone(self.q.reserve())  # The long envelope needs an explicit extension.
        budget = self.q.status()["gpu_budget"]
        peer = Queue.resume(self.db, run_id=RUN, clock=self.clock)
        self.addCleanup(peer.close)
        evidence = digest("approved dev-only timing decision")
        status = peer.extend_once(hours=4, evidence_sha=evidence)
        self.assertEqual((status["start_utc"], status["nominal_deadline"], status["gpu_budget"]),
                         (START, START + 26 * HOUR, budget))
        extension = {"hours": 4, "evidence_sha": evidence, "recorded_at": START, "freeze_at": START + 27 * HOUR,
                     "finalize_at": START + 29 * HOUR, "deadline": START + 30 * HOUR}
        self.assertEqual(status["extension"], extension)
        new = self.q.reserve()  # Same UTC second, different ticket deadlines.
        self.assertEqual((new.hard_stop, new.run_deadline), (extension["freeze_at"], extension["deadline"]))
        self.restart()
        self.assertEqual(self.q.outstanding(), [old, new])
        self.q.extend_once(hours=4, evidence_sha=evidence)  # Idempotent, not +4h again.
        for hours, proof in ((1, evidence), (4, digest("replacement evidence"))):
            with self.assertRaises(PolicyError):
                peer.extend_once(hours=hours, evidence_sha=proof)
        events = [e for e in peer.events() if e["kind"] == "timing_extended"]
        self.assertEqual([{**e["detail"], "recorded_at": e["utc"]} for e in events], [extension])
        self.clock.now = old.run_deadline
        backend = PhysicalDouble(self.clock)
        self.assertEqual(self.q.tick(backend)["phase"], "search")  # At the nominal wall target.
        self.assertEqual(backend.observed, [new.token])  # No extended RPC authority for the old token.
        self.clock.now = extension["freeze_at"]
        self.assertEqual(self.q.status()["phase"], "test")
        self.clock.now = extension["finalize_at"]
        self.assertEqual(self.q.status()["phase"], "finalize")
        self.assertIsNone(self.q.reserve())
        self.clock.now = extension["deadline"] + HOUR
        self.restart()
        self.assertEqual(self.q.status()["phase"], "closed")
        self.assertEqual((self.q.seal("freeze")["effective_at"], self.q.seal("finalize")["effective_at"]),
                         (extension["freeze_at"], extension["finalize_at"]))

    def test_invalid_or_boundary_extension_cannot_reopen_or_buy_time(self):
        proof = digest("justification")
        self.clock.now = self.q.freeze_at - 1
        for hours, evidence in ((0, proof), (5, proof), (True, proof), (1.5, proof), (1, "unbound")):
            with self.assertRaises(PolicyError):
                self.q.extend_once(hours=hours, evidence_sha=evidence)
        self.clock.advance(1)
        with self.assertRaises(PolicyError):
            self.q.extend_once(hours=1, evidence_sha=proof)
        self.restart()
        status = self.q.status()
        self.assertIsNone(status["extension"])
        self.assertEqual((status["phase"], status["deadline"]), ("test", START + 26 * HOUR))
        self.assertEqual(self.q.seal("freeze")["effective_at"], self.clock())

    def test_worker_outer_deadline_guard_remains_bounded(self):
        from joey_research import worker

        args = SimpleNamespace(packet=None, packet_sha256=None)
        obj = {"schema": 1, "split": "discovery", "inputs_sha256": digest("input"), "window": [4, 8],
               "loop": {"k": 1}, "reverse": False}
        flags = {k: "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")}
        with (patch.dict(worker.os.environ, flags), patch.object(worker.time, "time", return_value=START),
              patch.object(worker, "_load") as load, patch.object(worker.signal, "signal"),
              patch.object(worker.signal, "setitimer", side_effect=StopIteration) as timer):
            for remaining in (0, 30 * HOUR + 1):
                load.return_value = json.dumps({**obj, "deadline_utc": START + remaining}).encode()
                with self.assertRaisesRegex(ValueError, "deadline"):
                    worker.run(args)
            timer.assert_not_called()
            load.return_value = json.dumps({**obj, "deadline_utc": START + 30 * HOUR}).encode()
            with self.assertRaises(StopIteration):  # Stop before even reading rows; no real alarm or device.
                worker.run(args)
            timer.assert_called_once_with(worker.signal.ITIMER_REAL, 30 * HOUR)

    def test_rejects_mismatched_observations_and_immutable_database_writes(self):
        self.add(manifest("response"))

        class Misbound:
            @staticmethod
            def admit(_):
                return Observation(digest("wrong-token"), Outcome.UNKNOWN)

        with self.assertRaises(PolicyError):
            self.q.tick(Misbound())
        t, = self.q.outstanding()
        self.q.record(receipt(t, self.clock))
        self.clock.now = self.q.freeze_at
        self.q.seal("freeze")
        with sqlite3.connect(self.db) as raw:
            for statement in ("UPDATE receipts SET accepted=0", "DELETE FROM seals", "DELETE FROM events",
                              "UPDATE run_budget SET cap_s=cap_s+3600", "UPDATE run_budget SET charged_s=0"):
                with self.assertRaises(sqlite3.IntegrityError):
                    raw.execute(statement)
            self.assertEqual(raw.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(raw.execute("PRAGMA foreign_key_check").fetchall(), [])


if __name__ == "__main__":
    unittest.main()
