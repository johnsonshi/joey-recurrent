# After staging: readback, then an offline local build

These are separate coordinator gates. `prepare_capsule.py` performs none of them. Use the **existing, approved ACR** and its established transfer process; do not provision resources, run a remote build, change access or give the worker registry credentials.

1. **Read back the bytes.** Preserve the independently held handoff, discovery, receipt, wheel-lock, runtime and `pins.json` hashes. Mirror the sealed payload/build inputs through the approved channel. Retrieve the exact capsule artifact and the selected **single-platform** Linux/amd64 base manifest, config and layers **by digest** from that ACR; verify the blob digests/lengths. The base digest is in `locks/worker-base.json`. A tag, an image index or a cached public pull is not a readback.
2. **Verify before use.** On the materialized read-only copy, check the capsule pins, runtime archive and per-member allowlist against coordinator-held values. Use the **trusted coordinator** verifier, not one imported from the transfer:

   ```python
   result = verify_payload(
       capsule / "payload", (capsule / "receipt.json").read_bytes(),
       receipt_sha256=approved_receipt_sha256,
       discovery_sha256=approved_study_discovery_sha256,
       wheel_lock_bytes=trusted_linux_wheel_lock_bytes,
       wheel_lock_sha256="503b666e08a919fefcd01fb98e23a3916200c3d4f1cba4cf148726c4247e3eea",
   )
   ```

3. **Build locally without egress.** Confirm the read-back base's CPython 3.11, torch 2.8.0, C++ ABI, CUDA 12 and system libraries match the locked wheel tags. In a credential-free, network-disabled local builder using those base bytes (no pulls), use only verified `runtime.tar` members, their notices and the staged wheelhouse. From `payload/`, the install step is `python -m pip --isolated --disable-pip-version-check install --no-index --no-deps --only-binary=:all: --require-hashes -r ../requirements-linux.txt`. Keep the approved `src/` importable and the checkpoint path local. Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1` and `DO_NOT_TRACK=1` before imports. Enforce no egress independently of those flags. Run offline dependency/import checks; an ABI failure stops the build, not the offline policy.
4. **Read the worker back by digest.** Independently retain the resulting worker digest, transfer via the existing mirror process, and obtain both raw manifest/config pairs from ACR. Run `joey_research.staging.verify_image_pair(base_manifest, base_config, worker_manifest, worker_config, worker_digest=approved_worker_digest)` and verify the real layer pull. Ancestry, registry provenance, payload identity and runtime behavior are different checks.

Stop here. No Kubernetes changes or GPU execution are part of this procedure. Neither an offline build nor fixture tests establish stock parity, kernel compatibility on target hardware, or experiment results.
