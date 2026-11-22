# Typed transfer: one synthetic inventory stage

`typed-transfer` consumes a frozen, label-free development view and returns Joey's native answers. It is not a release, a final test, a generator, or a scorer. The separate coordinator establishes synthetic provenance and holds the approval and pins. Neither worker nor exporter interprets the ledger to decide what the answer should be.

## Admission

Use immutable `/admission/packet.json`, `/admission/typed-transfer-permit.json`, and **only** the read-only shard `/staged/inputs/typed-transfer.jsonl`. Both processes open the fixed files for themselves and check all rows before native loading; the exporter also verifies the staged runtime lock before starting a child.

Each line has exactly `id`, `group_id`, `task`, `prompt`, `choices`. Keep `evaluation.input_sha256(row)` unchanged: it hashes the generic row, including the supplied menu order, **not** the rendered request. `inputs_sha256` below hashes the exact shard bytes including newlines. No wrapper, menu digest, labels, or native request fields belong in the shard or view.

```python
mode = "typed-transfer"
view = {"schema": "joey-label-free-view/v1", "split": mode,
        "projection": "joey-native-inventory-development/v1",
        "rows": count, "bytes": size, "inputs_sha256": S, "split_manifest_sha256": F}
V = worker.view_sha256(view)
M = worker.method_sha256(settings)
permit = {"schema": "joey-typed-transfer-permit/v1", "split": mode, "view": view,
          "not_before": T, "admit_before": A, "hard_stop": H, "run_sha256": U,
          "approval_sha256": J, "methods": {M: settings}}  # or multiple admitted methods
# N = sha256(exact permit file bytes).hexdigest()
packet = {"schema": 1, "split": mode, "inputs_sha256": S, "deadline_utc": H,
          "permit_sha256": N, "view_sha256": V, "method_sha256": M, **settings}
# P = sha256(exact packet file bytes).hexdigest()
```

`worker.PhaseGate(mode, A, H, P, N, U, V, approval_sha256=J)` is supplied independently of those files. No nomination or final-freeze pin. Times are integer UTC seconds: `0 < T < A <= H`, `H - T <= 3600`. Admission is in `[T, min(A, H - 15))`; computation stops by `H - 15`, retrieval by `H`. Existing limits apply: 1–20,000 distinct IDs, at most 64 MiB of shard, 8 KiB of packet and 128 KiB of permit.

For byte-level integration, `worker.phase_packet(packet_bytes, permit_bytes, gate)` returns `(packet, window, plan, view)`; `worker.phase_rows(shard_bytes, view)` returns the validated generic rows in file order. The fixed-file admission path calls `gate.check_start(now)` before and after reading and validating those bytes.

The transport CLI contract is positional mode `typed-transfer` with `--admit-before A --hard-stop H --packet-sha256 P --requirements-sha256 R --phase-permit-sha256 N --run-sha256 U --view-sha256 V --approval-sha256 J`. Its worker receives `--phase typed-transfer`, the same admission and time pins (not the transport's runtime-lock pin `R`), `--packet /admission/packet.json`, `--model-dir /staged/model`, `--inputs /staged/inputs/typed-transfer.jsonl`, and `--output /work/run/out`. There is no path override, alternate role or shard fallback for this stage.

## Native request and result

`worker.typed_transfer_payload(row)` returns `{"state": row["prompt"], "questions": {"audit": {"type": kind, "instructions": instruction, "criteria": criteria}}}`. State is text, not a dictionary. The model sees no `id`, `group_id`, or `task`.

| `task` | `type` and exact instruction | Menu → criteria | Required `branch.labels` in order |
|---|---|---|---|
| `stock-eligibility` | `noul` — After posted entries only, does the focus depot have at least five units? | `["Yes","No"]` → `{"true":"Yes","false":"No"}` | `("yes","no")`, **not** the criteria keys |
| `stock-level` | `score` — After posted entries only, rate the final stock of the focus depot using the ordered levels. | `["Empty (0 units)","Low (1 to 3 units)","Medium (4 to 6 units)","High (7 or more units)"]` → that list | `("0","1","2","3")`, **not** the descriptions |
| `stock-choice` | `choice` — After posted entries only, which depot has the most units? If there is no unique maximum, choose No unique depot. | Five distinct options including `No unique depot` exactly once → `{option_0: choices[0], …, option_4: choices[4]}` in supplied order | `("option_0",…,"option_4")` |

`worker.measure_choice(native, row, window, plan, split="typed-transfer")` returns `(prediction, trace)`. It uses `native.branches(state, questions, window=window)["audit"]`. One `run_loop(plan, branch)` supplies the logits for both `branch.probabilities(result.logits)` and `branch.decode(result.logits)`. No text generation, second forward, or study-row warmup.

* `predictions.jsonl`: exactly `{id,input_sha256,probs}`. These are the full-precision probabilities in the authoritative order; they are never rounded to make the native answer.
* `trace.jsonl`: exactly `id,tokens,layer_calls,callback_counts,readouts,wall_ms,cuda_event_ms,packet_sha256,transfer_answer`. `readouts` is `[]`. No `native_answer`, prompt, menu text, logits, or diagnostics. Other modes forbid `transfer_answer`.
* `transfer_answer` is the **returned native object**, not a reconstruction:
  * `{"type":"noul","noul":p_yes}` (no probability map).
  * `{"type":"score","score":expected_level,"probabilities":{"0":p0,"1":p1,"2":p2,"3":p3}}`; `expected_level` is the zero-based `sum(i*p_i)`, not a class.
  * `{"type":"choice","choice":declared_label,"probabilities":{"option_0":p0,…,"option_4":p4}}`. Use the declared label, even if it is not the first probability maximum.

`worker.validate_transfer_answer(task, answer, probs)` raises `ValueError` on a mismatch. Probabilities are finite numbers in `[0,1]` (never booleans); raw mass tolerance is `1e-6`, rounded map mass tolerance `1e-5`. Public native numbers have six-decimal precision and must agree with the unrounded value to half a last decimal (plus floating-point tie slack). An expected level must be in `[0,3]`.

## Methods and accounting

Settings are exactly `window:[8,12]`, `reverse:false`, and `loop` with only `k`, `method`, `readout` (defaults `1`, `"naive"`, `"final"`). The only optional pair is `compact_recipe:"select-20"` **and** `compact_manifest_sha256:C`. All compact methods in a permit must name the same `C`. The worker derives the existing recipe from pinned source and uses the physically verified reload, retaining source layers 0–11 and 16–23 with physical repeat window `[8,12)`. A bare manifest, staged compact override, another topology, duplicate arm, or even an **unused** out-of-scope method invalidates admission. No ensemble, policy, nonfinal readout, or ignored loop parameters (including `alpha:null`).

| Loop | Actual callbacks `(stock, prelude, block, coda)` | Layers full / select-20 |
|---|---|---|
| k1 naive | `(1,0,0,0)` | 24 / 20 |
| k2 zero_extra | `(0,1,1,1)` | 24 / 20 |
| k2 naive | `(0,1,2,1)` | 28 / 24 |
| k4 naive | `(0,1,4,1)` | 36 / 32 |
| k6 naive | `(0,1,6,1)` | 44 / 40 |

The exporter checks the actual counts against the packet, not just a maximum cost. It verifies each row's ID, input digest, task, width, vector and typed answer together. Tokens are positive branch lengths up to 16,384; wall and CUDA latencies are finite and nonnegative. The `50`, `95`, `99` warm latencies must be the worker's nearest ranks of the individual wall times.

## Verified collection and recovery

`complete.json` binds the split, input, method, packet, permit, run, view and approval, and seals the **exact** `predictions.jsonl` and `trace.jsonl` bytes in `result_sha256`. It retains the existing parity, timing, device and (for select-20) physical mapping and reload metrics. Failure or timeout has no result files.

```python
args = dict(mode="typed-transfer", nonce=nonce, admit_before=A, hard_stop=H,
            packet_sha256=P, requirements_sha256=R, permit_sha256=N,
            run_sha256=U, view_sha256=V, approval_sha256=J)
attempt = transport.Attempt(**args)
D = transport.pin_receipt(port, attempt)  # coordinator journals this independently
transport.collect(port, destination, **args, receipt_sha256=D)
# After ACK loss, without running inference:
transport.recover(destination, attempt=attempt, receipt_sha256=D)
```

The destination contains `manifest.json` plus the three successful result files. Offline recovery checks custody even after expiry; it doesn't consult staged inputs or deliver an ACK. The authenticated receipt binds the exported rows; their distinct widths also let collection/recovery recheck the answer schemas against the raw vectors, and the sealed method ID identifies the exact callback budget. An optional `port=` on recovery requires the identical still-live manifest before re-ACK.

The socket-free CPU contract tests use invented cases and branch doubles only. From this isolated copy:

```sh
PYTHONPATH=src:tests python3 -B -m unittest test_worker.TypedTransferTest \
  test_transport.TransportTest.test_transfer_refuses_replaced_bytes_wrong_targets_and_bad_rows_before_loading \
  test_transport.TransportTest.test_transfer_export_collection_and_offline_recovery_without_sockets \
  test_transport.TransportTest.test_transfer_resealed_outputs_still_need_the_right_primitive_vector_and_cost
```
