# Discovery worker

The stock/Loop worker consumes one digest-bound packet, a materialized Joey checkpoint, and a label-free input shard. It runs only in an already-staged, offline runtime with exactly one A100. It has no cloud client, image builder, dependency installer, or permissions to create workloads.

```sh
PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m joey_research.worker --model-dir "$MODEL_DIR" \
  --inputs "$STAGED_INPUTS" --packet "$STAGED_PACKET" \
  --packet-sha256 "$VERIFIED_PACKET_DIGEST" --output "$NEW_OUTPUT_DIRECTORY"
```

The packet fields are `schema` (1), `split` (`discovery`), `inputs_sha256`, `window` (half-open), `loop` (a `LoopPlan`), `reverse` and `deadline_utc` (an integer POSIX timestamp). The admission controller, not the worker, must verify its provenance and bind its cutoff to the remaining trial and campaign time. An OS alarm is a worker-side backstop, not a replacement for the job deadline. The directory must be new. Do not score partial files; a missing `complete.json` means an incomplete trial.

All timed cases use one question per request. Probabilities remain in the original menu order even for reversed-order controls. Traces include the actual coda and block invocations. Wall latency includes formatting, synchronization and raw probability extraction; file persistence is counted in end-to-end throughput. CUDA event intervals are elapsed stream time, not a sum of kernel execution time. The reference readout uses native diagnostics; it is not a measurement of an optimized production endpoint or shared-prefix multi-question serving.
