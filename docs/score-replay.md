# Reproducing a released score without rerunning Joey

Run this from the audited source revision using Python 3.10 or later. Use a local copy of the **released** score bundle and the manifest SHA-256 from the independently held delivery receipt:

```sh
PYTHONPATH=src python3 -B -m joey_research.score_replay \
  --bundle "$RELEASED_BUNDLE" --manifest-sha256 "$TRUSTED_MANIFEST_SHA256"
```

The command is read-only and needs no model, GPU, cloud credentials, network, scheduler, or dataset generator. It checks the manifest hash before opening any data. Its `objects/` directory must contain only the content-addressed copies of the exact input, authorized label and scored prediction streams named by that manifest. Do not add unrelated files. Keep the bundle private. A hash supplied by the same untrusted copy is not independent evidence of provenance or permission to access labels.

Replay requires exact row order and input fingerprints. It recalculates every declared arm's primary summary (including per-task calibration and selective-risk metrics) and every declared paired interval and correctness-transition summary. The result must match the released summary hashes. Selection uses the original 0.8 threshold for paired intervals; the independent test uses 0.9. Both use 2,000 whole-group draws within task and seed 4117. Posthoc predictions are distinct from raw streams; a transformation withheld from binary64 scoring must remain omitted. Replay does **not** fit a temperature, choose a method, remeasure latency, or create a new inference.
