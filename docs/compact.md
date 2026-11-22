# Physical Joey-1 compaction

`joey_research.compact` exports a smaller **full checkpoint** by removing whole four-layer text cycles. It keeps the vision encoder, embedding, final norm and language head. The one-pass control and repeated-layer experiment load **one** weight file with the same SHA-256; only the call order changes. The exporter uses Python 3.10+ and the standard library. It does not load model code, run inference, or download anything.

## Export and reopen

Start with an immutable, materialized local copy of [baselabs/joey-1 at the supported revision](https://huggingface.co/baselabs/joey-1/tree/db5c2a631f2da93c4173378438d8246966e25159). Use a new destination outside it. The actual export requires a matching canonical `config.json` and the published SHA-256 of `model.safetensors`, not just matching layer names.

```sh
export PYTHONPATH=src
python -m joey_research.compact export "$JOEY_SOURCE" "$COMPACT_OUT" configs/compact/select-16.json
python -m joey_research.compact verify "$COMPACT_OUT" --k 1
python -m joey_research.compact verify "$COMPACT_OUT" --k 2
```

The result contains a standalone `model.safetensors`, the physical-depth `config.json`, and `compact.json` with the plan, source/compact sizes, hashes, tie observations and anchor map. If present, `readout.json`, tokenizer/processor configs, `tokenizer.json`, chat template, generation config, `LICENSE` and `NOTICE` are copied unchanged and hashed. They are not fetched or inferred. Executables, setup scripts and patches are **not** bundled; supply the pinned Joey runtime separately. Verification checks the export against its manifest; independent source lineage can be checked by exporting again from the pinned source and comparing digests. A manifest is not a signature.

```python
from joey_research.compact import export_compact, layer_schedule, verify_compact

export_compact(source_dir, new_output_dir, "configs/compact/select-16.json")
m = verify_compact(new_output_dir)
physical_ids = layer_schedule(m, k=2)  # immutable tuple; k is an integer in [1, 32]
```

Use the compact **text config** in the same Joey text-prefill runner as the baseline. At virtual step `j`, dispatch the existing physical module `physical_ids[j]`; do not construct a second set of parameters or a new checkpoint per pass. Preserve the original hidden-state flow, slot readout, full-attention block-bidirectional mask and causal linear-attention semantics. Use cache-free prefill, or validate independent KV/linear state per virtual occurrence; indexing mutable state only by the repeated physical ID is incorrect. This module provides the schedule, **not** that inference adapter.

## What changes

The pinned text stack has 24 layers: six `[linear, linear, linear, full]` cycles. A plan uses **zero-based source cycles**, in increasing, disjoint order. A group emits one physical cycle, preserving the four phases. The half-open `repeat_cycles` range instead indexes the **new compact cycles**.

| Plan | Physical anchor cycles | Body at `k=2` | Transformation |
| --- | --- | --- | --- |
| `select-16.json` | 0, 2, 3, 5 | Physical layers 4 through 11, twice | All survivor tensors are byte copies. |
| `merge-16.json` | 0, 2, 3, 5 | Same | For cycles (1, 2) use weights (0.25, 0.75); for (3, 4) use (0.75, 0.25). |

Both yield 16 calls at `k=1` and 24 at `k=2`. Only `text_config.num_hidden_layers` and `text_config.layer_types` change. Unselected `model.language_model.layers.*` records are physically absent rather than ignored at runtime.

The three 20-layer recipes vary the representative for middle cycles 2 and 3. `select-20.json` keeps the earlier cycle. `select-20-late.json` keeps the later cycle. `merge-20-mid.json` keeps the earlier anchor but uses the midpoint of the corresponding MLP projections. All three repeat physical layers `[8,12)`, giving 20 layer visits at one pass and 24 at two passes. Each needs a native one-pass control on its own exported bytes.

Merging applies **only** to the same phase's dense `mlp.{gate,up,down}_proj.weight`. Each coefficient must be positive; coefficients sum to one. Products and additions round separately to FP32 in source order. The result is cast once to the anchor dtype (round-to-nearest-even, including BF16). All attention projections, gated-delta state parameters (`A_log`, `dt_bias`), convolution, gates inside attention, norms, heads, embeddings and vision are **never averaged**. The selected anchor supplies every excluded layer tensor. The merge loop is a bounded-memory scalar reference, not a fast multi-billion-element kernel; use selection for a copy-only first run.

## Storage contract and measurements

- **Accepted:** dense, little-endian PyTorch safetensors. The pinned release is one BF16 file. The reader also supports F16/F32 and a conventional `model.safetensors.index.json` for marked generated fixtures; a re-sharded or re-quantized real checkpoint does not match the release digest. The output is always one file. Payload copying is streamed, and merge memory is bounded independently of matrix size.
- **Validated:** expected text namespaces and shapes (including attention-specific projections), same-phase dtypes, complete cycles, tensor extents, index membership and tied-head consistency. Unknown layouts, missing vision, unverifiable aliases, duplicate keys, gaps/overlaps, symlinked input files, pickle, and quantization are rejected. Only `format=pt` (or absent) and the supported omitted-head alias are accepted as safetensors metadata.
- **Disk vs. parameters:** `storage.weight_file_bytes` is the actual sum of safetensors file lengths, including headers; `storage.full`, `.vision` and `.text` count stored parameter elements and payload bytes independently. The corresponding source figures are in `source.storage`. Index JSON, tokenizer and runtime code are not part of the weight-file metric.
- **Ties:** the pinned header stores separate `lm_head.weight` and `model.language_model.embed_tokens.weight` records despite the configured tie. Preserve both and compare their bytes; equality does not imply shared runtime storage. A genuinely omitted tied head in a generated fixture stays omitted. `text.parameter_elements_assuming_configured_ties` is a conditional count, not measured runtime allocation.

For the actual **deployed text subset**, call `measure_text_storage` on the loaded runner's named text parameters and any separately kept slot-readout tensor:

```python
from joey_research.compact import measure_text_storage

resident = dict(text_model.named_parameters(remove_duplicate=False))
resident["slot_readout"] = actual_slot_readout  # only if retained by the runner
footprint = measure_text_storage(resident)
```

This counts shared byte ranges once and includes a real FP32 head clone when supplied. It reports unique referenced elements/bytes and reachable backing-storage bytes. Supply resident contiguous tensors, not meta/sharded placeholders; mixed-dtype views of one storage are unsupported. The result excludes activations, caches, allocator reserve and other tensors not passed in. Measure peak device allocation and latency separately.

## Weight-only residual diagnostic

On a POSIX host with NumPy already available, measure the same 12 gate/up/down projection pairs without running Joey. Supply the absolute path to the read-only original safetensors file and a **fresh** absolute output filename in an existing owner-only directory outside version control:

```sh
python -B scripts/compact_spectrum.py --weights "$READ_ONLY_WEIGHTS" --output "$PRIVATE_REPORT"
```

The diagnostic checks the full source digest and compares only corresponding positions of the same attention family. For matrices `A` and `B`, it measures reconstruction from the shared midpoint and one sign-shared factorization of `(B-A)/2`. It scores ranks 1, 4, 16 and 64 by directly measuring residual error, including after rounding the factors to BF16. The result is **achieved** energy for a seeded sketch, not an optimal-SVD bound. The JSON contains scalar measurements, algorithm settings and local resource measurements; no factors or weights are written. A lower reconstruction error or a smaller logical factor payload does not establish better task accuracy or lower GPU memory use.

## Evidence and limits

`joey_research.pass_norms.source_readback(source, base, new_bundle, compact_manifest_sha256=trusted_base_pin, create=True)` is a separate CPU-only feasibility check. It copies the removed cycle's eight BF16 pre-attention and pre-MLP norm vectors for a possible **second visit only** to physical layers `[8,12)`. An independent second call with `create=False` reconstructs the expected file from the original source and checks every donor and retained vector against the verified earlier-cycle select-20 export. `read_vectors(bundle, manifest_sha256=trusted_bundle_pin, compact_manifest_sha256=trusted_base_pin)` checks the actual bundle without loading a model. Use only direct local inputs and an owner-only parent for the new bundle.

The sidecar is **not a checkpoint** and is not consumed by any existing experiment worker. Count the entire base weight file plus the entire extra safetensors file, including its header. This preparation does not change the base, original masks, MLPs, or attention parameters; it does not establish parity, latency, memory savings, numerical benefit, or decision quality for a pass-specific runtime. A tensor-level difference is a feasibility signal, not evidence to promote a finalist.

```sh
PYTHONPATH=src python -B -m unittest discover -s tests -p 'test_compact*.py' -v
```

To budget scalar merge time on a target CPU, run `PYTHONPATH=src python -B tests/bench_compact_merge.py`. The generated BF16 benchmark reads two input matrices, merges them, writes one safetensors file and hashes it. Divide the number of merged MLP **output elements** by `sample_floor_outputs_per_s` for a conservative sample-based time estimate. A planning rule is to allow twice that estimate for the merge stage alone; otherwise use selection. The reported peak is Python-traced temporary memory, not peak device memory. The benchmark excludes source verification, copies of untouched tensors, cache effects at checkpoint scale and inference.

On the generated BF16 fixture, both 16-layer exports reduced stored parameters from **1,502 to 1,020**, payload from **3,004 to 2,040 bytes**, and the actual safetensors file from **39,740 to 26,584 bytes**. A separate reader compared every survivor or merged tensor against the source-based expectation. For `k=1` and `k=2`, a small nonlinear fixture readout matched the corresponding source-weight schedule after reload. This checks serialization and dispatch, **not** parity with the unmodified 24-layer model or official Joey logits. Tiny-file percentages do not predict real savings.

The published [config](https://huggingface.co/baselabs/joey-1/blob/db5c2a631f2da93c4173378438d8246966e25159/config.json) and [weight file metadata](https://huggingface.co/baselabs/joey-1/blob/db5c2a631f2da93c4173378438d8246966e25159/model.safetensors) describe 618 records, with 2,721,801,024 stored BF16 elements including vision and the separately stored head. All 618 JSON header records passed the layout validator; no real tensor payload was used in the fixture tests. Base/compact quality, real runner reload parity, latency and GPU memory remain unmeasured here.
