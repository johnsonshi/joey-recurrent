# Public-native: original q1 framing on public training inputs

This compares two ways to ask Joey the same question on the [train-derived diagnostic](public-development.md). The complaints, menu order, six methods and scorer are fixed. Only the native `q1` request framing changes. This is not an official test score; workers receive neither source labels nor permission to select finalists.

```
python -m joey_research.transport public-native --admit-before A --hard-stop H \
  --packet-sha256 P --requirements-sha256 R --phase-permit-sha256 N \
  --run-sha256 U --view-sha256 V --approval-sha256 J
```

Use the existing phase nonce, immutable `/admission/packet.json`, and a read-only `/admission/public-native-permit.json` (`schema:"joey-public-native-permit/v1"`, `split:"public-native"`). Same `not_before`, admit/stop checks, result paths and loopback/ACK protocol. Both child and exporter revalidate the **single** `/staged/inputs/public-native.jsonl` shard; there is no fallback to the generic shard. The view has exactly:

```
schema = "joey-label-free-view/v1"       split = "public-native"
projection = "banking77-original-train-native77/v1"
inputs_sha256, split_manifest_sha256, rows, bytes,
menu_sha256, research_inputs_sha256, native_requests_sha256
```

`inputs_sha256` and `bytes` bind the entire paired shard, with `rows` equal to the authorized count (at most 512). `menu_sha256` retains the existing canonical ordered-list hash. Each line is exactly `{"input":ROW,"native":{"id":ID,"input_sha256":ROW_DIGEST,"payload":PAYLOAD}}`. `ROW_DIGEST` is `evaluation.input_sha256(ROW)`, not a hash of the JSONL bytes. To recover each source stream, concatenate `json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",",":"), allow_nan=False).encode("utf-8") + b"\n"` in shard order, over the original `input` or entire `native` object respectively. `research_inputs_sha256` and `native_requests_sha256` must match those byte hashes. **Do not sort keys, including nested criteria**, when building the paired shard. Pin the complete descriptor with `worker.view_sha256(view)`.

The exact verified payload is `{state:{}, questions:{q1:{type:"choice", instructions:ROW.prompt, criteria:{option_0:ROW.choices[0], …, option_76:ROW.choices[76]}}}}`. Legal labels and their numeric insertion order are fixed. `worker.public_native_payload(row)` reconstructs that payload after admission has proved equivalence; it never repairs a request. Only the payload reaches the model.

Use existing `worker.method_sha256(settings)` IDs, exact window `[8,12]`, forward order, final readout, no policy/ensemble: full `(k=1,naive)`, `(2,zero_extra)`, `(2,naive)`, `(3,euler)`; compact `(1,naive)` or `(2,naive)` with `compact_recipe:"select-20"` and its independent `compact_manifest_sha256`. An unused out-of-scope method also invalidates the permit.

`trace.jsonl` adds exactly `native_answer:{type,choice,probabilities}`. It is the genuine result of `branch.decode(result.logits)`; read its **declared** `choice`, never replace it with argmax. The typed map must have all 77 legal keys, finite values in `[0,1]` and mass within `0.01` of one. Separately, unchanged `predictions.jsonl` has `id,input_sha256,probs` in numeric menu order at full precision (mass tolerance `1e-6`). One `run_loop` supplies both; the timed interval includes the additional native CPU decode/softmax, but no additional model forward or coda. The exporter checks the physical callback/layer counts. Use `collect(..., mode="public-native", packet_sha256=P, requirements_sha256=R, permit_sha256=N, run_sha256=U, view_sha256=V, approval_sha256=J, nonce=nonce, admit_before=A, hard_stop=H)`; `pin_receipt` and `recover` are unchanged.

For paired accuracy and observed-class macro-F1, use `benchmarks.framing_contrast.choice_contrasts(inputs, labels, arms, contrasts)`. Each arm contains `{id, input_sha256, choice_index}`. Use the declared choice for the native arm and the first probability maximum for the generic research arm. The scorer requires matching input hashes and one common ordered menu. It resamples whole complaint groups, using the same draws for every arm. Coefficients `+1, -1, -1, +1` on native-repeat, native-stock, generic-repeat and generic-stock estimate how much the repeat-minus-stock effect changes with framing. These are pointwise intervals on the same sample, not a second benchmark.

To exercise the projection on invented data, run `PYTHONPATH=src:tests:. python -B -m unittest -q test_worker test_transport test_framing_contrast` from the repository root.
