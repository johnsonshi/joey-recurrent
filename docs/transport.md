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
