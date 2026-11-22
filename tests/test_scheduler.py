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
        self.q.record(receipt(t, self.clock))
        self.assertEqual(self.q.registry()[0]["state"], "expired")
        self.assertIsNone(self.q.registry()[0]["result_sha"])

    def test_restart_uses_original_setup_origin_and_exact_budget_boundaries(self):
        self.clock.now = START + 21 * HOUR - 30
        path = self.db.with_name("after-setup.sqlite")
        q = Queue.create(path, run_id=RUN, start_utc=START, gpu_classes=("a", "a"), clock=self.clock)
        try:
            for duration in (31, 30):
                tid = q.register(manifest(str(duration), gpu="a", budget=duration), priority=100-duration)
                for gate in Gate:
                    q.attest(tid, gate, digest(gate.value))
            t = q.reserve()
            self.assertEqual(t.hard_stop, START + 21 * HOUR)
            self.assertEqual(t.trial_id, manifest("30", gpu="a", budget=30).key.trial_id)
            self.assertIsNone(q.reserve())
        finally:
            q.close()
        self.clock.advance(30)
        q = Queue.resume(path, run_id=RUN, clock=self.clock)
        self.addCleanup(q.close)
        self.assertEqual(q.status()["phase"], "test")
        self.assertEqual(q.seal("freeze")["payload"]["picks"], [])
        q.record(receipt(t, self.clock))
        self.assertFalse(q.history(t.trial_id)[0]["accepted"])  # Cannot affect an already-frozen plan.
        self.clock.now = START + 24 * HOUR
        self.assertEqual(q.status()["phase"], "closed")
        self.assertEqual(q.seal("finalize")["effective_at"], START + 23 * HOUR)
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
        self.assertEqual((status["phase"], status["as_of"]), ("closed", START + 24 * HOUR))
        self.assertEqual([(q.seal(k)["effective_at"], q.seal(k)["recorded_at"]) for k in ("freeze", "finalize")],
                         [(START + 21 * HOUR,) * 2, (START + 23 * HOUR,) * 2])
        self.assertEqual((backend.sent, backend.observed), ([], []))

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
            for statement in ("UPDATE receipts SET accepted=0", "DELETE FROM seals", "DELETE FROM events"):
                with self.assertRaises(sqlite3.IntegrityError):
                    raw.execute(statement)
            self.assertEqual(raw.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(raw.execute("PRAGMA foreign_key_check").fetchall(), [])


if __name__ == "__main__":
    unittest.main()
