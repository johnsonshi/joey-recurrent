# Running a physical compact counterpart

`CompactJoey.load` is a separate offline entry point for the [physical compact export](compact.md). It **does not** relax the [stock load path](native-runtime.md): stock still requires the exact original weight, config, tokenizer, template and readout digests. A compact checkpoint, even a valid one, is not stock Joey. Its own **k=1** path is the comparator for recurrence.

```python
from joey_research.native_compact import CompactJoey

model = CompactJoey.load(
    "/staged/compact-16",
    source_model_dir="/staged/joey-1",
    expected_manifest_sha256=trusted_export_digest,
    device="cuda:0",
)
answers = model.counterpart(state, questions)  # native shared-prefix decisions, compact weights
one = model.score_physical(state, questions, physical_ids=model.physical_ids(1))
two = model.score_physical(state, questions, physical_ids=model.physical_ids(2))
```

Use immutable staged directories, the stock runtime, and `HF_HUB_OFFLINE=1` before any HF import. Pass a previously recorded SHA-256 for `compact.json`, **not** a digest calculated from an unknown bundle and trusted on sight. Loading does the following before accepting a schedule:

1. Verify the manifest fingerprint and reopen the export with `verify_compact`. Require the pinned Joey source revision, native sidecar hashes, correct source tensor hash and a real export (not `generated-fixture`). Reject unrelated loader inputs.
2. Load the *original* Joey model using the unchanged stock verifier, including the patched `qev` source closure. Check its full-depth real-model parity. Then independently compare **every exported tensor** against the pinned source. A selection is an exact byte comparison. For a merged cycle, compute only `mlp.{gate,up,down}_proj.weight` from the contributing source phases in CPU fp32 (separate multiply and add in source order, then one cast to the output dtype); compare every tile bit-for-bit. Vision, head, embeddings, attention, delta parameters and norms remain exact copies of their specified sources.
3. Compute reference k=1 slot logits by running the source model's mapped anchor layers. For a merge, temporarily substitute the **independently reconstructed** MLP projections in the private reference instance, not just its anchor MLP. Restore every parameter on exit; no checkpoint is written. Release the source model before loading the physical export.
4. Compare the newly loaded export's full uncached forward **and** explicit one-pass split to that reference on the mandatory calibration branches and any `extra_requests`. Require identical token layouts, same top-1, and maximum slot-logit and legal-probability errors <= 1e-5. `reload_report` records these errors. Separately run the stock adapter's cached-vs-uncached gate on the compact repeat window. A source-construction disagreement, failed reload, or failed native parity gate stops loading.

`physical_ids(k)` returns the manifest schedule. `score_physical` accepts only that exact sequence (including its prefix and suffix), with `1 <= k <= 32`. A compact 16-layer model has physical IDs `0..15`; these are not the original anchor IDs. Each repeat invokes the same real modules with fresh per-sequence state and unchanged positions. `branches(..., window=model.repeat_window)` exposes the same early-exit and mixing callbacks as the stock adapter. Its `stock()` is the **compact** cached native reference; `one_pass()` runs the explicit compact prelude, block once and coda. Neither is the original 24-layer baseline.

The verifier reads the full source and export and temporarily holds an independent reference model. Exclude verification/startup from steady-state scoring measurements, and reset measurement counters after loading before attributing a peak to the compact deployment. Structural verification and dependency-free tests do not establish model parity, quality, or speed. The real test requires `JOEY_TEST_MODEL_DIR`, `JOEY_TEST_COMPACT_DIR` and `JOEY_TEST_COMPACT_MANIFEST_SHA256`; run `python -m unittest discover -s tests -p test_native_compact.py` in the approved staged runtime. Check selection and merged bundles separately.
