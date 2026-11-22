# Native Joey text decisions

`NativeJoey.stock` calls the patched OneJev `DecisionEngine.decide` in process. It does not generate a response and parse it as an answer. `NativeBranch` is a separate, inference-only path for repeating a window of real decoder layers. Its results are **not stock results**, even if it chooses the same answer.

## Stage the reference runtime

Use a clean, offline process with a local checkpoint directory and reviewed, importable Python packages. The loader verifies SHA-256 for the required checkpoint files and the eight Python files in the `qev.engine` import closure before loading the engine. It rejects competing model files, including calibration overrides. The digest tables are in [`native.py`](../src/joey_research/native.py).

| Component | Required input |
| --- | --- |
| Checkpoint | [`baselabs/joey-1` at `db5c2a6`](https://huggingface.co/baselabs/joey-1/tree/db5c2a631f2da93c4173378438d8246966e25159): `model.safetensors`, `config.json`, `readout.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`. Preserve `LICENSE` and `NOTICE` with the mirror. |
| Native decision code | [OneJev `81ce62f`](https://github.com/OmniJev/OneJev/tree/81ce62f1597c91e46767d4d02d6ac2e18534fe94), with the checkpoint's [`serve.patch`](https://huggingface.co/baselabs/joey-1/blob/db5c2a631f2da93c4173378438d8246966e25159/serve.patch) applied. Preserve the upstream license. Only `qev/{__init__,client,engine,prompt,schema,readout,answers,calibrate}.py` is needed on the Python path. The adapter never uses the HTTP client. |
| Linux inference stack | [Publisher setup](https://huggingface.co/baselabs/joey-1/blob/db5c2a631f2da93c4173378438d8246966e25159/setup.sh): one CUDA GPU, `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, its matched torch/torchvision, and [`constraints.txt`](https://huggingface.co/baselabs/joey-1/blob/db5c2a631f2da93c4173378438d8246966e25159/constraints.txt). Python 3.10+, Transformers 5.19.0, Accelerate 1.15.0, tokenizers 0.23.3, safetensors 0.8.0, huggingface_hub 2.2.0, Pydantic 2.14.1 and Jinja2 3.1+ with their resolved dependencies. The reference setup includes flash-linear-attention/fla-core 0.5.2 and causal-conv1d 1.7.0. |

Resolve and lock the Linux/Python/CUDA-specific wheel closure when preparing the mirror; `constraints.txt` alone is not a complete install manifest. Keep the CUDA extensions matched to the image. Do not run `setup.sh` at inference time. The Decision Index kit, server, training extras, images and vision processor are not used by this adapter. Set `HF_HUB_OFFLINE=1` **before any HF import**. Use a process without credentials or outbound network. The only model argument is an existing absolute or relative local directory. The verified engine uses `local_files_only=True`, `trust_remote_code=False`, and the built-in `Qwen3_5ForCausalLM` with the checkpoint's *text* config.

```python
from joey_research.native import NativeJoey

native = NativeJoey.load("/staged/joey-1", device="cuda:0")
questions = {
    "route": {"type": "choice", "instructions": "Which queue fits?",
              "criteria": {"billing": "Charges and refunds", "tech": "Application faults"}},
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
}
state = "A duplicate charge appeared; please reverse it."
answers = native.stock(state, questions)  # {question_id: typed answer}
```

## What stock computes

1. [OneJev's renderer](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/prompt.py) embeds the state once in the system/user chat template and renders a separate question suffix with labeled option lines. `enable_thinking=False` and `add_generation_prompt=True` are part of the scoring template; no token is sampled.
2. The [engine](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/engine.py) tokenizes each full branch without additional special tokens. A sentinel locates the state boundary; the shared token prefix is shortened until it agrees with all branches. The patch verifies that `P`, the shared-cache length, equals the last token boundary at or before the start of the suffix. Token intervals for option lines are half-open and include tokens overlapping the lines. Layout errors are refused rather than approximated.
3. Stock computes one fresh prefix per request. Sequential mode deep-copies the prefix cache for each suffix; batched mode forks full-attention keys/values **and** the linear layers' convolution/recurrent states. Native suffix positions start at `P`. With padding, the score is taken at the final **real** suffix token in each row, not at the batch width. Full-attention block IDs are `0` for prefix, `1` for suffix, `-1` for padding: prefix tokens see one another, suffix tokens see the prefix and their own suffix, and prefix tokens never see a suffix. The linear layers stay causal.
4. The [checkpoint](https://huggingface.co/baselabs/joey-1/blob/db5c2a631f2da93c4173378438d8246966e25159/config.json) has 24 text layers of width 2,048: three GatedDeltaNet layers followed by one full-attention layer, six times. Stock uses a bf16 backbone, final norm, then a cloned fp32 LM head at the last prompt position. `slot_lm` is **not** a separate classifier head, a pooled option embedding, or a generated letter. Scores are LM logits for exactly the legal labels. The tokenizer is round-trip checked: `A` through `Z` are IDs 32 through 57; `AA` is 5840. The ordered 256-ID vector is verified at load.

There is no `calibration.json` at this revision. The native temperature lookup (`kind:K`, then `kind`, then `1`) therefore returns 1. The legal slots are softmaxed together; the diagnostic full-vocabulary allowed-token mass is not a class probability. `noul` returns P(yes), with slots `[yes, no]`; `choice` returns the first maximum and probabilities keyed in criteria insertion order; `score` returns the expected **zero-based** level and probabilities keyed by level index. Public answers use [OneJev's six-decimal rounding](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/answers.py). The parity checks use unrounded logits and probabilities.

## Hidden-state interface

```python
window = (8, 12)  # zero-based, start inclusive, stop exclusive
report = native.verify_parity(window=window)
b = native.branches(state, questions, window=window)["route"]
official_logits = b.stock()       # actually runs the whole official request
one_pass_logits = b.one_pass()    # prelude + block once + coda, not a stock shortcut
with b.inference_context():
    hidden = b.prelude()          # [1, P + suffix_length, 2048]
    hidden = b.block(hidden, b.fresh_state())
    hidden = b.block(hidden, b.fresh_state())
    logits = b.coda(hidden, b.fresh_state())  # fp32 [number of legal slots]
answer = b.decode(logits)
```

Each handle is one complete, unpadded, text-only branch. `b.layout` exposes the native token IDs, absolute option spans, prefix length, legal slot IDs and readout position. A `fresh_state()` belongs to one handle and is consumed once. Each block invocation processes **all original token positions** at fixed absolute positions, with the same layer weights and native masks, and without KV, convolution or recurrent state from any prior invocation. The coda runs the remaining layers, final norm and native head. All real model calls force inference mode. There are no trainable adapters, image paths, or autoregressive decoding here.

The prelude calls the actual token embedding. Each nonempty stage is a shallow view of the native text decoder with the selected original layer modules, a matching slice of the layer-type list and no stage-final norm. The native decoder forward receives `inputs_embeds`, the full branch validity mask, `past_key_values=None` and `use_cache=False` under the patched block-ID context. It builds its own rotary positions and attention masks on each call. Views share parameters, not mutable sequence state. Profile this mask/rotary setup on every invocation; it is part of the split path's cost.

The hook path deliberately recomputes the entire prefix+suffix for **each** question; a reused stock prefix cache would be stale after a hidden intervention. This does not preserve the stock serving cost. Compare both `b.stock()` and `b.one_pass()`; for multi-question throughput time `native.stock(state, questions)` once for the request, not `b.stock()` once per question.

`verify_parity` runs five mandatory synthetic branches from two independent requests (all three decision types), plus any `(state, questions)` in `extra_requests`. For every branch it checks typed answer reconstruction from the official diagnostics, then compares the official cached path against both the whole uncached model and the split uncached one-pass coda. Both comparisons require max absolute slot-logit error <= 0.05, max absolute legal-probability error <= 0.005 and identical top-1. The whole vs split uncached comparison requires both errors <= 1e-5 and identical top-1. Thresholds may be tightened, not loosened. Exact ties use the native first maximum; near-tied examples are not omitted. A mismatch raises `ParityError` with a report and disables this window on this instance. A passing window is not a claim about other hardware, inputs, builds, or repeated-block quality.

## Scope and limits

The adapter defaults to OneJev's **official sequential** shared-prefix path for reproducibility. `fork_mode="auto"` requests the publisher's batched-vs-sequential selection. Upstream's 0.05 *probability* fork check is a bf16 structural sanity check, not this adapter's parity gate. The publisher's [engine wrapper](https://huggingface.co/baselabs/joey-1/blob/db5c2a631f2da93c4173378438d8246966e25159/engine.py) documents a 32-bit indexing error on large batched forks; the adapter preserves its sequential retry for this error only. The underlying engine also has its own OOM handling.

The card specifies a trained length of **16,384** tokens, whereas the pinned engine constructor defaults to 32,768 per branch and 65,536 per request. This adapter enforces the stricter 16,384 per branch and 65,536 per request, counting the shared prefix once toward the request limit. It never truncates. The native schema limits choices to 255 options and scores to ten levels. All valid choices fit this checkpoint's 256 verified slots. Hidden-state comparisons support one native ordering per question, not chunking or permutation averaging.

To run the optional real-model test, supply `JOEY_TEST_MODEL_DIR` in an approved offline runtime and run `python -m unittest discover -s tests -p test_native.py` with `src` on `PYTHONPATH`. Without that variable, only the dependency-free interface fixtures run. Those fixtures do not establish model parity.
