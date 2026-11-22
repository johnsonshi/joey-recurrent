# Run-only loopback export

Run one attempt per Pod. Use the independently verified Linux/amd64 Python + torch image and reviewed study source, a read-only root, a private writable `/work`, read-only verified staged assets, and an immutable packet. No host networking, Service, sidecars, service links, mounted credentials, service account token, or egress. Drop all capabilities; use `allowPrivilegeEscalation: false`. The worker does **not** contact Kubernetes. The coordinator alone owns admission, the forward to **this exact Pod UID**, and terminal-state checks.

```
python -m joey_research.transport probe --admit-before A --hard-stop H
python -m joey_research.transport discovery --admit-before A --hard-stop H \
  --packet-sha256 P --requirements-sha256 R
```

Times are integer Unix UTC seconds. `H` is at most one hour away. Set `JOEY_EXPORT_NONCE` to a **fresh 32 random bytes encoded as 64 lowercase hex digits** in the attempt's environment, not the command line. `P` pins `/admission/packet.json`; its discovery split digest and `deadline_utc == H` are mandatory. `R` independently pins `/staged/requirements-linux.txt`. Mount the capsule's verified `/staged/wheelhouse`, `/staged/model`, `/staged/inputs/discovery.jsonl`; compact packets additionally require verified `/staged/compact`. The image supplies the reviewed importable `joey_research` and `qev` sources. Use the staged wheel set as-is; no dependency resolution, build, download, or fallback to a host install.

The supervisor refuses late starts, cuts off computation at `H − 15s` to leave a retrieval window, and ends **all** service by `H`. Leave realistic time for startup, installation, and collection. A missed cutoff, missing file, or unacknowledged transfer is not evidence of a successful run. The probe uses the base torch, not the model or discovery inputs. The HTTP listener is IPv4 **127.0.0.1:18765 in the Pod**, only **after** the supervised child has exited. Retry the exact-Pod forward until publication; there is no command, upload, list, or progress endpoint. Send **`X-Joey-Nonce: <64 hex digits>` on every request**. Use one request per connection; no redirects, proxies, ranges or chunked encoding.

| Request | Response |
|---|---|
| `GET /manifest` | Immutable JSON, schema `joey-loopback/v1`. Hash its **exact bytes** for the receipt digest. |
| `GET /file/NAME` | Exact published bytes. Each successful GET has `Content-Length` and lowercase hex `X-SHA256`. |
| `POST /ack` | JSON object with exactly `nonce` and `receipt_sha256`, with a bounded `Content-Length`. `204` on a match, then exit. Never put the nonce in a URL. |

The manifest binds `mode`, `status`, `admit_before`, `hard_stop`, `nonce_sha256` (SHA-256 of the **ASCII hex string**), `packet_sha256`, `requirements_sha256` (both null for probes), and `files` (exact filename → `{bytes,sha256}`).

* `probe/setup_probe`: `probe.json` has exactly `nonce_sha256`, `device_count`, `devices` (one `{a100, vram_bytes, capability:[major,minor]}` per visible device), `python:[major,minor]`, `torch_version`, `cxx11_abi` (boolean), `cuda` (numeric dotted version or null). Independently reproduce `challenge.bin` with `hashlib.shake_256(b"joey-loopback/challenge/v1\x00" + bytes.fromhex(nonce)).digest(65536)`. No UUIDs, host names, raw device names, model imports, predictions, or claim of parity.
* `discovery/complete`: only the genuine worker's `predictions.jsonl`, `trace.jsonl`, and `complete.json` (metrics). A nonzero child exit, partial files, or invalid bindings cannot publish these.
* Either mode: `failed` or `timeout`, empty file map. The admission pins identify **which attempt** failed; they don't attest that its inputs passed verification.

After establishing the authorized forward, call `joey_research.transport.collect(local_port, new_destination, mode=..., nonce=..., admit_before=A, hard_stop=H, packet_sha256=P, requirements_sha256=R)` (omit pins for probes). Use a fresh result directory in a private, single-writer coordinator directory. It validates the manifest and files, fsyncs them, rereads and verifies them, atomically publishes the directory and fsyncs its parent **before** sending the ACK. A lost ACK does not erase durable files; reconcile the retained bytes and exact Pod terminal evidence, never substitute a new run's receipt. ACK asserts coordinator persistence, not success of inference or independent validation of study metrics. Exit after ACK: `0` for setup/complete, `2` for failed, `124` for timeout. Hard expiry without ACK is non-success. Keep the nonce, raw outputs, paths and trace out of shared logs.

## Promotion selection (new image only)

```
python -m joey_research.transport selection --admit-before A --hard-stop H \
  --packet-sha256 P --requirements-sha256 R --nomination-sha256 N
```

Mount immutable, read-only `/admission/packet.json`, `/admission/nomination.json`, and **only the approved selection shard** at `/staged/inputs/selection.jsonl`. No labels, calibration or final data. `P` and `N` are SHA-256 of the **exact file bytes**, supplied independently by the coordinator. `nomination.json` is the worker permit, not a trial proposal or a list of scores. The coordinator must first verify that the selection shard came from the frozen split manifest and the methods came from the immutable nomination; a content hash alone does not establish either provenance.

Example construction (all digest strings are lowercase hex):

```python
from joey_research.worker import method_sha256

settings = {"window": [4, 8], "loop": {"k": 2}, "reverse": False}
M = method_sha256(settings)  # 0061ab40681a47fd974e520e842a12f7c83569ac583939d88e3975c51665161b
permit = {"schema": "joey-promotion-permit/v1", "split": "selection",
          "inputs_sha256": S, "split_manifest_sha256": F, "methods": {M: settings}}
# Freeze permit bytes first; N is their digest, not a field inside the permit.
packet = {"schema": 1, "split": "selection", "inputs_sha256": S, "deadline_utc": H,
          "permit_sha256": N, "method_sha256": M, **settings}
```

`M` hashes `json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")` **without a newline**. `settings` must contain exactly `window`, `loop`, `reverse`, plus only the optional worker fields `policy`, `compact_recipe`, `compact_manifest_sha256`, `order_ensemble` when applicable. No default expansion, field dropping, list sorting, family aliases, or int/float coercion. The permit contains exactly the five keys above, 1–256 methods, and at most 128 KiB; the packet limit remains 8192 bytes. Every method ID is rehashed, including unselected entries.

The exporter verifies before installation. The child uses the existing `python -m joey_research.worker` flags with `--inputs /staged/inputs/selection.jsonl --selection --nomination-sha256 N` and independently rereads both admission files and the shard before native import. The verifier is also callable as `worker.selection_packet(packet_bytes, permit_bytes, packet_sha256=P, nomination_sha256=N)`, returning `(packet, window, loop_plan)`. A selection packet is never accepted by the discovery entry point or vice versa.

Collect with `collect(port, fresh_destination, mode="selection", nonce=nonce, admit_before=A, hard_stop=H, packet_sha256=P, requirements_sha256=R, permit_sha256=N)`. Only selection manifests include `permit_sha256`, including failures/timeouts. Successful selection uses the same three filenames as discovery, but `complete.json` also records `split`, `inputs_sha256`, `permit_sha256`, `method_sha256`, and `result_sha256` (hashes of the two actual JSONL streams). Each selection trace row binds `packet_sha256`. Mixing files, receipts, or destinations between attempts is not supported. Discovery/probe manifests and the ACK protocol are unchanged.

### Bounded order-ensemble control

On discovery or selection, declare `order_ensemble: {"numerator": 1, "denominator": 4, "seed": 42}` in the settings. All three values are JSON integers, not booleans: `0 ≤ numerator ≤ denominator ≤ 10000`, denominator positive, `0 ≤ seed < 2³²`. Requires the **full model**, `reverse:false`, `loop:{"k":2,"method":"zero_extra"}` (optional `readout:"final"`); no compact, policy, or diagnostic mode.

For `G` distinct groups, spend a second pass on exactly `floor(G × numerator / denominator)` groups, **not rows**. Each task gets its floor quota; remaining places go to largest remainders, ties by task string ascending. Within a task, rank groups by SHA-256 of the same canonical JSON encoding of `["joey-order-ensemble/v1", split, seed, task, group_id]`, ascending; tie by group ID. Groups spanning tasks are rejected. Row order, labels, logits, group size and latency do not choose the budget.

Every row gets one genuine split forward (prelude, one block, coda). Only selected groups get a fresh forward on reversed criteria; average the two probability vectors **after** restoring original option order. No extra study-row warmup on selection or ensemble runs; fixed public parity cases remain separate setup work. Trace `ensemble_calls` and `forward_calls` are 1 or 2. `tokens`, `layer_calls` (24 or 48), and `callback_counts` sum the actual work across orders; wall and CUDA intervals enclose both calls. `complete.json.compute_spend` totals groups, extra groups, forward calls, coda calls and layer calls for measured study rows. These are compute counters, not evidence of improved accuracy or equal latency.

## Frozen phase transition — isolated implementation, not a release

These modes are **siblings** of selection. No real phase, including final, is authorized by this code or its fixtures. The coordinator must verify run identity, provenance of the label-free shard, and immutable role-specific approval (finalists/protocol freeze for final) under the timed queue **before** releasing the permit and starting an attempt. Digest agreement alone does not establish that authority. Do not stage any labels. Calibration here means collecting predictions on a frozen view, **not** fitting or loading a calibration model.

| `MODE` | Required `view.projection` | Release flag (other flag forbidden) |
|---|---|---|
| `public-development` | `banking77-public-train-research-77/v1` | `--approval-sha256 J` (public train research approval) |
| `calibration` | `joey-calibration-research/v1` | `--approval-sha256 J` (calibration approval) |
| `final` | `joey-final-research/v1` | `--final-freeze-sha256 J` (final freeze) |

```
python -m joey_research.transport MODE --admit-before A --hard-stop H \
  --packet-sha256 P --requirements-sha256 R --phase-permit-sha256 N \
  --run-sha256 U --view-sha256 V <role-specific release flag above>
```

Same fresh `JOEY_EXPORT_NONCE`, exact-Pod forward, quiet supervised child, verified runtime lock and private scratch as before. Data reads are fixed at `/admission/packet.json`, `/admission/MODE-permit.json` and `/staged/inputs/MODE.jsonl` on immutable read-only mounts. No alternate path or discovery/selection fallback. The supervised worker receives `--phase MODE --phase-permit-sha256 N --run-sha256 U --view-sha256 V --admit-before A --hard-stop H` and the same release flag, in addition to its usual fixed packet/model/input/output arguments. `--selection`/`--nomination-sha256` cannot admit a phase. Both processes independently re-open and verify the permit, packet and shard before native import.

Exact schemas, using the selection section's canonical settings encoding:

```python
view = {"schema": "joey-label-free-view/v1", "split": MODE, "projection": PROJECTION,
        "inputs_sha256": S, "split_manifest_sha256": F, "rows": ROWS, "bytes": SIZE}
# Public-development ONLY: also set view["menu_sha256"] to SHA256(canonical JSON bytes
# of the approved ordered list of all 77 choice strings). Every row must match it.
V = worker.view_sha256(view)
M = worker.method_sha256(settings)
release_key = "final_freeze_sha256" if MODE == "final" else "approval_sha256"
permit = {"schema": f"joey-{MODE}-permit/v1", "split": MODE, "not_before": T,
          "admit_before": A, "hard_stop": H, "run_sha256": U, release_key: J,
          "view": view, "methods": {M: settings}}
# N = SHA256(exact permit bytes). S = SHA256(exact shard bytes, including newlines).
packet = {"schema": 1, "split": MODE, "deadline_utc": H, "inputs_sha256": S,
          "view_sha256": V, "permit_sha256": N, "method_sha256": M, **settings}
# P = SHA256(exact packet bytes); retain P, R, N, U, V, J independently of these mounts.
```

`T, A, H` are integer UTC seconds; `0 < T < A ≤ H`, `H − T ≤ 3600`. A phase may start only in `[T, min(A,H−15))`. All methods obey the existing settings allowlist and exact method digests. Views have exactly the shown keys (plus `menu_sha256` only for public-development), 1–20000 distinct decision IDs, and a nonempty shard no larger than 64 MiB. Public-development is the full-menu **research projection**, not a native index score or an official test split. Row validation rejects extra fields instead of removing them.

The manifest and `complete.json` bind `MODE` (manifest `mode`, metrics `split`), `permit_sha256`, `run_sha256`, `view_sha256`, and **only** the corresponding release digest. A phase completion seals the two actual JSONL digests, method, input digest and packet as for selection. Failures/timeouts have no result files. No default-discovery, probe or selection manifest acquires these new fields.

### Verified phase collection and ACK-loss recovery

Use the fresh nonce and all launch pins **from the coordinator's own journal**, not from the saved manifest. In addition to the normal private, single-writer destination, retain the exact Pod UID and terminal-state evidence separately:

```python
args = dict(mode=MODE, nonce=NONCE, admit_before=A, hard_stop=H,
            packet_sha256=P, requirements_sha256=R, permit_sha256=N,
            run_sha256=U, view_sha256=V, **{release_key: J})
attempt = transport.Attempt(**args)
D = transport.pin_receipt(port, attempt)  # authenticated GET through the authorized forward
# Durably journal D independently BEFORE collection, even for a failed or timeout receipt.
manifest = transport.collect(port, new_destination, receipt_sha256=D, **args)
```

If collection published the directory but ACK was lost, `transport.recover(destination, attempt=attempt, receipt_sha256=D)` verifies the original local receipt, complete file inventory, byte lengths, SHA-256, and sealed worker result bindings, and fsyncs files and directories again. It may run offline after expiry; it proves local custody, **not** delivery of an ACK or successful inference. With an independently re-established forward to the **same UID before H**, pass `port=port` to also require byte-identical live manifest evidence before re-ACK. Never infer `D` from the suspect directory. Missing/partial custody, mismatched pins/nonce, or changed files fail closed; recovery never creates a release, reruns inference, or overwrites results.
