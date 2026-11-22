# Two-slot experiment queue

`joey_research.scheduler.Queue` is a standard-library SQLite controller, not a GPU launcher. It can reserve a prevalidated manifest on one of two configured slots and consume observations about that reservation. It has no shell, model, provider, or credential interface.

## Time is part of the run identity

Infrastructure setup is **outside** the experimental budget. Before the private artifact path, durable result storage, workload namespace and both GPU slots are verified end to end, keep proposals as setup records: there is no experimental start, deadline or timed queue. Once infrastructure is ready, persist that first verified UTC timestamp and create the database **once** with `Queue.create(path, run_id=..., start_utc=..., gpu_classes=(class0, class1))`. Use the readiness timestamp, not the setup launch, later stock-parity completion or next controller wake. Resume the same epoch after a restart. Use one durable database with reliable SQLite locking; never resume a stale copy against live workers.

| Time from the original start | What the queue permits |
| --- | --- |
| Before 21h | Register manifests; admit dev work only if its complete time budget fits before 21h. Propose frozen evaluations using completed dev evidence. |
| 21h to before 23h | The last committed proposal is immutable. Only its distinct held-out trials may start; each must fit before 23h. No proposal means an empty freeze, not a fallback selection. |
| 23h to before 24h | Seal the final evidence and admit nothing. Reconcile outstanding workers and use the remaining hour for reporting. |
| At or after 24h | The foreground driver exits. No new backend calls, admissions, or scientific evidence. A later, explicit `record` can audit a physical fence without changing a sealed result. |

Transitions are committed on the first queue operation at or after each boundary. Each seal records both its scheduled instant and its actual write time. Downtime never extends a cutoff. An observed backward UTC jump below the persisted high-water mark rejects operations; use a trustworthy UTC clock at both ends.

## Register and select

1. Construct a `TrialKey(model, dataset, code, config, seed, scorer, precision)`. The first four fields and `scorer` are lowercase SHA-256 content digests. `dataset` identifies the **actual split**, not just a benchmark name; a digest cannot be both dev and test. `config` must bind every behavior-changing option. `seed` is a nonnegative integer, `precision` a short identifier. The trial ID is SHA-256 of compact, key-sorted JSON of these seven fields.
2. Register `Manifest(sha256, key, split, gpu_class, budget_s)` via `q.register(manifest, priority=100, max_attempts=1, requires=())`. Use `Split.DEV` or `Split.TEST`. `budget_s` includes provider handling and all worker time. `requires` can reference only previously registered dev trials. An identical registration returns the same ID; reusing a key with a different descriptor or policy fails. Within each eligible set the order is lowest priority number, then trial ID, then slot number. Class matching is exact.
3. Attach verified, immutable gate evidence for **each exact manifest and hardware class**, using `q.attest(trial_id, Gate.OFFLINE_STAGING, proof_sha)` and `q.attest(trial_id, Gate.STOCK_PARITY, proof_sha)`. Both gates and successful dependencies are required before any admission. The caller verifies the artifacts, split disjointness, stock parity and staging; a digest alone does not verify their contents.
4. Before hour 21, call `q.nominate([Pick(dev_id, test_id, Role.SELECTED), ...], decision_sha=...)` with the complete plan from the declared dev-only selection policy. Include `Role.STOCK` if a frozen stock comparison is required. A pick needs accepted dev evidence, a pre-registered test manifest and an identical key apart from the dataset. At most one pick is allowed per model, held-out split and role. A later **pre-freeze** proposal replaces the whole prior proposal. Two roles that select the same held-out key use one physical trial.

No registration, reprioritization or selection from held-out results is possible. The controller accepts result artifact digests, not scores, answers or labels. Keep actual evidence in the coordinator's artifact store.

## Adapter contract: what makes a slot safe to reuse

Implement two bounded methods: `admit(ticket: Admission) -> Observation` and `observe(ticket: Admission) -> Observation`. **Do not enable admission if the adapter cannot meet the following contract.**

* An `Admission` binds the run and trial IDs, validated manifest digest, split, attempt number, slot and generation, class, unique token, issue time, exclusive `admit_before`, `hard_stop`, and absolute `run_deadline`. Verify it against the approved artifact, actual slot mapping, class and time. Reject arbitrary command strings or manifest substitutions. Enforce at most one physical worker per bound slot, including pending starts.
* Admission must be atomic and idempotent by token. Reject stale slot generations. Never recreate a fenced token, even if a delayed request for it is still inside its original admission window. The provider/worker must reject late starts and enforce `hard_stop` independently of the controller, including if the controller disappears. Bound all adapter I/O by the applicable absolute cutoff.
* Use `Observation(token, Outcome.LIVE, at=last_worker_pulse_utc)` for a real heartbeat. Repeated polls of the same pulse do not renew the lease. `Outcome.UNKNOWN`, ordinary absence, transport errors, and lease expiry **never free a slot**. `BackendUnavailable`, `TimeoutError`, and `ConnectionError` are treated as unknown; malformed or misbound observations raise an error.
* A terminal receipt is `Observation(token, outcome, at=finished_or_fenced_utc, fence_sha=..., result_sha=...)`. Its immutable proof must establish **both** that the token no longer owns running/pending workers **and** that no delayed admission can start one. `FENCED_ABSENT` means verified *never started* plus that irreversible fence. A not-found response, stale pulse, timeout, missing result, or vanished control process is not such proof. If provenance is unavailable, return `UNKNOWN` and leave the slot pinned.
* `SUCCEEDED` needs a result digest. `FAILED`, `RETRYABLE`, and `FENCED_ABSENT` must not carry one. Only the latter two can requeue a dev trial, with a new token and up to three total admissions including the original. A held-out trial consumes its one admission **before** external I/O; even a crash in the send gap does not authorize a resend. A physical receipt seen late is audited, but cannot amend a freeze or final evidence.

`q.tick(adapter)` is one watchdog cycle: reconcile outstanding tokens, then reserve/admit at most two new tickets. No database write lock spans a backend call. The alternate low-level API is `reserve()` followed by **one** send by that caller and `record(observation)`. Tickets returned by `outstanding()` are for observation only. After a crash, use `Queue.resume(path, run_id=...)` and `tick`; do not replay your previous sends. `run(adapter, poll_s=...)` is an optional foreground driver for an already-prepared queue; use `tick` within the coordinator loop if it also updates dev-only nominations or publishes per-trial projections.

## Evidence and verification

`q.snapshot()` returns JSON-serializable `status` and `registry` from one transaction. Status includes the latest event sequence, watchdog timestamp, both occupied/free slots, cutoffs, counts and seal digests. The registry includes seven-field keys, manifest/gate/result digests, attempts, dependencies and reasons queued work cannot start. The underlying `trial_registry` and `status_counts` views update in the terminal receipt's transaction; there is no projection file to repair after a crash.

Use `q.events(after=cursor)` for incremental publication, `q.history(trial_id)` for every slot generation and physical receipt, and `q.seal("freeze" | "finalize")` for immutable decisions and final evidence. A finalized `active` entry means the queue lacked a terminal receipt by cutoff; it is not a success or proof the worker is still physically running.

Run the isolated CPU-only verification from the package root:

```sh
PYTHONPATH=src python3 -B -m unittest discover -s tests -p test_scheduler.py -v
```
