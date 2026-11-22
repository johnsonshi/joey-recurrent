# Two-slot experiment queue

`joey_research.scheduler.Queue` is a standard-library SQLite controller, not a GPU launcher. It can reserve a prevalidated manifest on one of two configured slots and consume observations about that reservation. It has no shell, model, provider, or credential interface.

## Time is part of the run identity

Infrastructure setup is **outside the experimental clock**. Before the private artifact path, durable result storage, workload namespace and both GPU slots are verified end to end, keep proposals as setup records: there is no experimental start, deadline or timed queue. Once infrastructure is ready, persist that first verified UTC timestamp and create the database **once** with `Queue.create(path, run_id=..., start_utc=..., gpu_classes=(class0, class1))`. Use the readiness timestamp, not the setup launch, later stock-parity completion or next controller wake. Use one durable database with reliable SQLite locking; never resume a stale copy against live workers.

| Default time from the original start | What the queue permits |
| --- | --- |
| Before 23h | Register manifests. Admit dev work only if its entire envelope ends by the freeze and fits the GPU balance. Propose held-out evaluations using completed dev evidence only. |
| 23h to before 25h | Freeze the last committed proposal. Only its distinct held-out trials may start; each must fit by 25h and within the same GPU balance. No proposal means an empty freeze, not a fallback selection. |
| 25h to before 26h | Stop scientific GPU work, seal final evidence and begin final reporting. The watchdog may reconcile outstanding tokens, but **no new admissions** or scientific results are accepted. |
| At or after 26h | The foreground watchdog exits; `tick` makes no further backend calls. **26h is a wall target, not a reporting deadline.** Non-GPU verification, report writing, projections and explicit `record` of physical fences may continue as needed. They cannot change either seal or grant more GPU time. |

The new-run offsets are `FREEZE_AFTER_S=82800`, `FINALIZE_AFTER_S=90000`, `WALL_TARGET_S=93600`. `q.freeze_at`, `q.finalize_at` and `q.deadline` expose effective absolute UTC seconds; `q.nominal_deadline` always identifies the original wall target. A ticket may end **at** its cutoff, but a result must be recorded **before** its scientific cutoff to count. The first operation at or after a scheduled boundary commits the transition, recording the effective instant separately from the actual write time. Use a trustworthy UTC clock at both ends; regression below the persisted high-water mark fails closed.

`Queue.resume(path, run_id=...)` preserves the original start, cutoffs, balance, outstanding tickets and prior seals. Stop older controllers before upgrading: opening a v1 database pins its **original 21/23/24h** schedule and accounts for its existing receipts and reservations in the upgrade transaction. It does not gain the new run's two hours.

### One explicit extension for scientific work

Before the freeze, the coordinator may call `q.extend_once(hours=n, evidence_sha=sha256)` **once**, with an integer `n` from **1 through 4** and the lowercase SHA-256 of a verified, dev-only justification. Do not call it on a timer. It returns status after the durable decision. From the **unchanged** post-setup start, the new-run offsets become `23+n`, `25+n`, `26+n` hours (maximum **27/29/30h**). No GPU-seconds are added. New tickets still have to fit both limits. Already issued tickets retain their original `admit_before`, `hard_stop` **and** `run_deadline`, even across restart.

`status["extension"]` is `None` or `{hours, evidence_sha, recorded_at, freeze_at, finalize_at, deadline}`; `q.events()` exposes the immutable `timing_extended` event with the decision and its UTC. Use top-level `status["freeze_at"]`, `["finalize_at"]` and `["deadline"]` for current projections, including an early seal; `["nominal_deadline"]` never shifts. Replaying the exact request before freeze does not add time. A different second request, or **any** request at/after an early or scheduled freeze, is rejected. Non-GPU reporting may overrun either wall target **without requesting this extension**.

## GPU time is a separate ceiling

**Exactly two one-A100 slots share 48 GPU-hours (`GPU_CAP_S=172800` GPU-seconds), not 52.** If setup or earlier work already consumed part of this ceiling, pass that conservative debit in `Queue.create(..., prior_gpu_s=...)` (0–172800; default 0). This reduces available GPU time without backdating the experimental clock. Use this queue for every subsequent chargeable attempt; it does not meter other launch paths. `q.status()["gpu_budget"]` (also in `q.snapshot()["status"]`) exposes:

| Field | Meaning in GPU-seconds |
| --- | --- |
| `cap_s` | Fixed cumulative ceiling across both slots, both splits and all attempts. |
| `charged_s` | Initial debit plus settled charges from terminal receipts, including failures, retries and scientifically rejected results. |
| `reserved_s` | Full `hard_stop - issued_at` for every outstanding token, even across lease expiry, `UNKNOWN`, a crash before sending, or downtime past its stop. A live pulse **after** the hard stop withholds all remaining headroom until fenced. |
| `remaining_s` | `max(0, cap_s - charged_s - reserved_s)`, available for a **whole** new attempt. A queued trial that cannot fit has `gpu_budget` in `waiting_for`. |

`reserve()` checks the balance and writes the reservation under the same SQLite write lock. A matching irreversible terminal receipt replaces that reservation with a charge **once**, in the same transaction that frees the slot. Verified `FENCED_ABSENT` costs zero GPU time but still consumes an attempt. Otherwise charge from issue through the worker's terminal/fence second, **not** the poll time. Count a partial terminal second conservatively: issue at `100`, end at `109`, hard stop `160` costs `10` seconds; an end at `160` costs the full `60`. Do not truncate evidence *later* than the hard stop to make the ledger look compliant. Independent physical enforcement by the adapter remains mandatory.

For example, after 46 GPU-hours are charged, a 2h reservation consumes all remaining headroom even if the other slot is free. If it fences early, only the unused balance becomes available; a time overrun for reporting never changes the balance or the scientific cutoff.

## Register and select

1. Construct a `TrialKey(model, dataset, code, config, seed, scorer, precision)`. The first four fields and `scorer` are lowercase SHA-256 content digests. `dataset` identifies the **actual split**, not just a benchmark name; a digest cannot be both dev and test. `config` must bind every behavior-changing option. `seed` is a nonnegative integer, `precision` a short identifier. The trial ID is SHA-256 of compact, key-sorted JSON of these seven fields.
2. Register `Manifest(sha256, key, split, gpu_class, budget_s)` via `q.register(manifest, priority=100, max_attempts=1, requires=())`. Use `Split.DEV` or `Split.TEST`. `budget_s` is 1–108,000 seconds including admission latency and all worker time; **actual admissions** are limited further by the phase cutoff, remaining GPU budget and any stricter adapter policy. `requires` can reference only previously registered dev trials. An identical registration returns the same ID; reusing a key with a different descriptor or policy fails. Within each eligible set the order is lowest priority number, then trial ID, then slot number. Class matching is exact.
3. Attach verified, immutable gate evidence for **each exact manifest and hardware class**, using `q.attest(trial_id, Gate.OFFLINE_STAGING, proof_sha)` and `q.attest(trial_id, Gate.STOCK_PARITY, proof_sha)`. Both gates and successful dependencies are required before any admission. The caller verifies the artifacts, split disjointness, stock parity and staging; a digest alone does not verify their contents.
4. Before the freeze (default: hour 23), call `q.nominate([Pick(dev_id, test_id, Role.SELECTED), ...], decision_sha=...)` with the complete plan from the declared dev-only selection policy. Include `Role.STOCK` if a frozen stock comparison is required. A pick needs accepted dev evidence, a pre-registered test manifest and an identical key apart from the dataset. At most one pick is allowed per model, held-out split and role. A later **pre-freeze** proposal replaces the whole prior proposal. Two roles that select the same held-out key use one physical trial.

No registration, reprioritization or selection from held-out results is possible. The controller accepts result artifact digests, not scores, answers or labels. Keep actual evidence in the coordinator's artifact store.

### Finishing early

An empty queue between batches is not evidence of completion. To finish deliberately, pass a lowercase SHA-256 digest of the coordinator's justification; do **not** advance the clock or synthesize worker receipts.

* `q.freeze_early(reason_sha=...)` seals the current dev-only proposal at the actual UTC when no token is outstanding. It moves the **same two-hour** held-out window earlier; it cannot widen it. Other connections and restarts see the effective cutoffs. Registration and nomination cannot reopen.
* `q.finish_early(reason_sha=...)` returns closed status at the actual UTC. It requires no outstanding token and cannot skip a frozen pick that still fits the time and GPU budgets (even if a gate is temporarily missing). It can close an empty plan in one transaction, or mark provably unaffordable/unfinishable picks `expired` with no receipt. A refused close leaves the plan unchanged. Both seals are immutable; `run()` returns on close rather than waiting for the nominal wall target.

## Adapter contract: what makes a slot safe to reuse

Implement two bounded methods: `admit(ticket: Admission) -> Observation` and `observe(ticket: Admission) -> Observation`. **Do not enable admission if the adapter cannot meet the following contract.**

* An `Admission` binds the run and trial IDs, validated manifest digest, split, attempt number, slot and generation, class, unique token, issue time, exclusive `admit_before`, `hard_stop`, and absolute `run_deadline`. `admit_before` is the earlier of `hard_stop` and `issued_at + lease_s` (lease 1–3600s; default 90s). Bind the worker packet's `deadline_utc` to **`hard_stop`, not `run_deadline`**. The discovery worker's 30h outer sanity bound (`MAX_WALL_S`) is not permission to spend 30h. The existing `compile_discovery` path is stricter: at most **1h** remaining per job. Verify the approved artifact, actual slot mapping, class and time. Reject arbitrary command strings or manifest substitutions. Enforce at most one physical worker per bound slot, including pending starts.
* Admission must be atomic and idempotent by token. Reject stale slot generations. Never recreate a fenced token, even if a delayed request for it is still inside its original admission window. The provider/worker must reject late starts and enforce `hard_stop` independently of the controller, including if the controller disappears. Bound all adapter I/O by the applicable absolute cutoff.
* Use `Observation(token, Outcome.LIVE, at=last_worker_pulse_utc)` for a real heartbeat. Repeated polls of the same pulse do not renew the lease. `Outcome.UNKNOWN`, ordinary absence, transport errors, and lease expiry **never free a slot**. `BackendUnavailable`, `TimeoutError`, and `ConnectionError` are treated as unknown; malformed or misbound observations raise an error.
* A terminal receipt is `Observation(token, outcome, at=finished_or_fenced_utc, fence_sha=..., result_sha=...)`. Its immutable proof must establish **both** that the token no longer owns running/pending workers **and** that no delayed admission can start one. `FENCED_ABSENT` means verified *never started* plus that irreversible fence. A not-found response, stale pulse, timeout, missing result, or vanished control process is not such proof. If provenance is unavailable, return `UNKNOWN` and leave the slot pinned.
* `SUCCEEDED` needs a result digest. `FAILED`, `RETRYABLE`, and `FENCED_ABSENT` must not carry one. Only the latter two can requeue a dev trial, with a new token and up to three total admissions including the original. A held-out trial consumes its one admission **before** external I/O; even a crash in the send gap does not authorize a resend. A physical receipt seen late is audited, but cannot amend a freeze or final evidence.

`q.tick(adapter)` is one watchdog cycle: reconcile outstanding tokens, then reserve/admit at most two new tickets. No database write lock spans a backend call. The alternate low-level API is `reserve()` followed by **one** send by that caller and `record(observation)`. Tickets returned by `outstanding()` are for observation only. After a crash, use `Queue.resume(path, run_id=...)` and `tick`; do not replay your previous sends. `run(adapter, poll_s=...)` is an optional foreground driver for an already-prepared queue; use `tick` within the coordinator loop if it also updates dev-only nominations or publishes per-trial projections.

## Evidence and verification

`q.snapshot()` returns JSON-serializable `status` and `registry` from one transaction. Status includes the latest event sequence, watchdog timestamp, both occupied/free slots, cutoffs, counts and seal digests. The registry includes seven-field keys, manifest/gate/result digests, attempts, dependencies and reasons queued work cannot start. The underlying `trial_registry` and `status_counts` views update in the terminal receipt's transaction; there is no projection file to repair after a crash.

Use `q.events(after=cursor)` for incremental publication, `q.history(trial_id)` for every slot generation and physical receipt, and `q.seal("freeze" | "finalize")` for immutable decisions and final evidence. A finalized `active` entry means the queue lacked a terminal receipt by cutoff; it is not a success or proof the worker is still physically running.

Run the isolated CPU-only verification from the package root with the project interpreter:

```sh
PYTHONPATH=src:tests python3 -B -m unittest test_scheduler test_worker test_workload test_search
```
