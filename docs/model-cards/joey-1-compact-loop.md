---
license: other
license_name: mit-and-apache-2.0
license_link: LICENSE
base_model: baselabs/joey-1
language:
  - en
tags:
  - decision-model
  - routing
  - layer-reuse
---

# Joey-1-Compact-Loop

**Joey-1-Compact-Loop** is an independent compact variant of [Joey-1](https://huggingface.co/baselabs/joey-1), the 2B-parameter **decision model** Base Labs at Baseten post-trained from Qwen3.5-2B. It is not an official Base Labs or Baseten release. Instead of writing chat replies word by word, a decision model reads context, a question, and named options, returning a winning choice and a probability per option. Deleting four layers shrinks the weight file by **8.4%**, and `Joey-1-Compact-Loop` uses **direct (naive) feedback**: it runs the same four frozen middle layers on their own updated output a second time, with no new learned cross-pass gate.

By contrast, [Thinking Machines' Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/) and [Hugging Face's Inkling 2 Mini explainer](https://huggingface.co/blog/inkling2mini) describe a trained loop that inspired this variant, not a replicated architecture or accuracy comparison. Each **layer** is a processing step and each **weight** is a learned number. In Inkling 2 Mini, the first six layers (`0–5`) run once, and each of the three visits to its 12 middle layers starts from that same initial output. On visits 2 and 3, learned attention and gates mix a message from the immediately previous visit, without rerunning the opening layers or reading all prior passes.

The evaluation tests how much lost accuracy direct feedback alone recovers on the compact checkpoint without training for loops. Repeating those four middle layers raised held-out routing accuracy on that compact file from **229/480 (47.7%)** to **328/480 (68.3%)**, versus **365/480 (76.0%)** for full `Joey-1`.

## How Joey-1-Compact-Loop works

```mermaid
flowchart LR
    A["Original layers 0–7"] --> B["Original layers 8–11"]
    B -->|"Updated values, same four layers"| B
    B -->|"After visit 2; layers 12–15 are absent"| C["Original layers 16–23"]
    C --> D["Score the choices"]
```

**Inference runtime:** The tested and released path is a custom text-only Python inference runner using the patched OneJev/`qev` `DecisionEngine` and Hugging Face Transformers Qwen3.5 layers on PyTorch with CUDA (evaluated on NVIDIA A100 GPUs), with `run_joey.py` as its included entry point. Loading the checkpoint in stock `transformers` alone runs the 20 stored layers once without enabling the loop, and these experiments were not benchmarked or validated in vLLM, SGLang, `llama.cpp`, or MLX.

- **`model.safetensors`** is a standalone **20-layer** weight file. Original `Joey-1` text layers `12–15` are physically deleted; the remaining 20 layers are exact copies from [Base Labs' `baselabs/joey-1` checkpoint](https://huggingface.co/baselabs/joey-1/tree/db5c2a631f2da93c4173378438d8246966e25159), reindexed `0–19` (source `16–23` become physical `12–19`).
- **`run_joey.py`** executes the two-pass schedule (`k=2`, `method="naive"` on slice `[8, 12)`, zero-based layers `8–11`). One forward pass splits into `prelude`, `block`, and `coda`. `Prelude` runs `0–7` once. `Block` runs `8–11` twice, feeding pass one's numeric output straight into pass two as input. `Coda` is the existing layers after the loop (source `16–23`, physical `12–19`), followed by normal normalization and one choice score per option, not an extra module or pass.

| Layer numbering (zero-based, inclusive) | Execution schedule |
|---|---|
| Original `Joey-1` layers (`12–15` deleted) | `0–7 → 8–11 → 8–11 → 16–23` |
| Reindexed layers inside the compact file | `0–7 → 8–11 → 8–11 → 12–19` |

That gives **24 layer visits** from **20 stored text layers** without duplicating weights. Layers `8–11` have three pretrained causal `GatedDeltaNet` layers and one block-masked full-attention layer; `Joey-1`'s original internal gates remain intact, with no learned cross-pass gate. The full model never restarts, repeats the prompt, or feeds back a first answer. Prompt tokens, positions, masks, and weights stay fixed, while sequence cache and attention, short-convolution, and recurrent state start fresh each visit.

## Held-out routing results and confidence

All five schedules answered the same **480 synthetic routing questions across 60 new graphs**. Each question asked which candidate station could reach a destination. The first three rows use the original 24-layer weights; the last two use the same compact 20-layer file.

<!-- routing-results:begin -->

| Model / run | What ran | Layer visits | Correct of 480 | Accuracy | Net vs. one pass on the same weights |
| --- | --- | ---: | ---: | ---: | ---: |
| `Joey-1` (one pass, loop off) | All 24 layers once | 24 | 365 | 76.0% | Baseline |
| `Joey-1` (two full runs, loop off) | Run all 24 layers twice independently; reverse the choices in run two and average each choice's probabilities | 48 | 366 | 76.25% | +1 |
| **`Joey-1-Loop` (loop on)** | Same 24 layers; repeat only `8–11` before one decision | 28 | **412** | **85.8%** | **+47** |
| `Joey-1-Compact-Loop` (one pass, loop off) | All 20 retained layers once | 20 | 229 | 47.7% | Baseline |
| **`Joey-1-Compact-Loop` (loop on)** | Same 20-layer file; repeat only `8–11` before one decision | 24 | **328** | **68.3%** | **+99** |

<!-- routing-results:end -->

```mermaid
xychart-beta horizontal
    title "On 480 route questions: repeating four middle layers added 47 correct choices"
    x-axis ["Joey-1: one full run (365)", "Joey-1: two full runs averaged (366)", "Joey-1-Loop: four layers repeated (412)", "Compact Joey-1: one run (229)", "Joey-1-Compact-Loop: four layers repeated (328)"]
    y-axis "Correct choices out of 480" 0 --> 480
    bar [365, 366, 412, 229, 328]
```

The compact one-pass row is the **same 20-layer file** with its loop off, not a separate model release. The 99-answer gain is relative to that file's 229 correct answers.

Looping yielded **99 more correct overall** (**105** wrong-to-right and **6** right-to-wrong, net **+99**). That **+20.6 percentage point** gain (paired whole-graph 95% CI **+16.9 to +24.6 points** over 2,000 graph resamples) finished **7.7 points** behind full `Joey-1`.

Raw Brier score (squared error across option probabilities, lower is better) went from **0.6575** to **0.5412**, while top-choice ECE worsened from **0.1088** to **0.2613**. Distinct from aggregate calibration, wrong-choice confidence rose on **74/146** shared errors.

Synthetic inventory tasks and **512 BANKING77 training complaints** (`423/512` loop off vs. `457/512` loop on) were separate development checks, not confirmatory datasets or official held-out BANKING77 or Decision Index tests.

## Checkpoint size and CUDA execution time

Deleting four layers shrinks `model.safetensors` by **8.4%** (**5,443,677,680** to **4,985,916,160 bytes**). At **24 layer visits** each, batch-one inference across **2,304 development inputs** on **two NVIDIA A100 GPUs** took **3.4% more** CUDA execution time than staged one-pass `Joey-1` (GPU execution time, not serving latency).

## Quickstart and usage

Download the complete [Joey-1-Compact-Loop release on Hugging Face](https://huggingface.co/models?search=Joey-1-Compact-Loop) so `run_joey.py` and the compact loop metadata sit alongside `model.safetensors`. The checkpoint is **standalone** and requires no separate `Joey-1` download. On any CPU machine with Python 3.10+, verify the snapshot checksums, reindexed layer layout, and schedule from the downloaded directory:

```bash
python run_joey.py --verify-only
```

Running actual decisions requires Linux with an NVIDIA CUDA GPU; follow the [Linux/CUDA setup and example request](../../scripts/hub/USAGE.md) to install the pinned dependencies. Then run the built-in support-ticket example or pass a JSON decision request:

```bash
python run_joey.py
python run_joey.py --request request.json
```

The runner outputs the winning `choice` and per-option `probabilities` for each question. Loading the file without `run_joey.py` will not enable the loop, and only text decisions were tested.

## License and attribution

Base Labs at Baseten post-trained [`baselabs/joey-1`](https://huggingface.co/baselabs/joey-1/tree/db5c2a631f2da93c4173378438d8246966e25159) (`db5c2a631f2da93c4173378438d8246966e25159`) from Alibaba Cloud's [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B). Baseten's modifications use MIT and Qwen3.5-2B uses Apache-2.0; both licenses apply. The snapshot includes [LICENSE](https://huggingface.co/baselabs/joey-1/blob/db5c2a631f2da93c4173378438d8246966e25159/LICENSE), [NOTICE](https://huggingface.co/baselabs/joey-1/blob/db5c2a631f2da93c4173378438d8246966e25159/NOTICE), `LICENSE-APACHE-2.0`, and `THIRD_PARTY.md`.

## Strengths and limitations

On this held-out routing task, repeating four middle layers lifted a pruned checkpoint of an existing decision model from **47.7%** to **68.3%** accuracy without retraining. Because a frozen schedule tested identical **8.4% smaller** weights on paired prompts, that recovery is meaningful despite trailing full `Joey-1` and worsening calibration. Whether this trade-off holds across broader datasets or other models is not yet established.
