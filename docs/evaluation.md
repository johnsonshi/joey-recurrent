# Scoring structured decisions

The evaluator compares complete distributions over the **same ordered menu** on the **same states**. It does not call a model, generate labels, calibrate a head, or decide a winner. Python 3.10+ and the standard library suffice.

## Interchange

Use one UTF-8 JSON object per line, with exactly the fields below. Do not put scores, labels, rationales, source annotations, or pass history in the inputs.

| File | Fields |
| --- | --- |
| Inputs | `id`, `group_id`, `task`, `prompt` (nonempty strings), `choices` (ordered list of distinct nonempty strings) |
| Labels, separately for each split | `id`, `input_sha256`, `gold_index` (zero-based integer) |
| Predictions, separately for each arm/pass and split | `id`, `input_sha256`, `probs` (one probability per choice), **or** `id`, `input_sha256`, `error` (nonempty failure code) |

`input_sha256(row)` binds all five input fields, including choice order: SHA-256 over UTF-8 `json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)`. These hashes are for alignment, not secrecy. Only `prompt` and `choices` reach the model. Do not expose IDs, group metadata, labels, the verifier, or neighboring examples through conversational memory.

`id` identifies a state/rendering. `group_id` identifies the independent source graph, trajectory, or acquisition unit; put every correlated rendering, order permutation, and step in the same group. `task` is a preregistered task family/stratum, not a unique row ID. A group may not span strata. The evaluator detects duplicate IDs and identical prompt/menu sets in different groups, not arbitrary paraphrases or shared source sessions.

## Freeze and score

From the package root:

```sh
PYTHONPATH=src python3 -B -m unittest discover -s tests -p test_evaluation.py
```

Only the coordinator should create a campaign freeze. The `freeze` command takes a **complete label-free** universe and writes a manifest to stdout:

```sh
PYTHONPATH=src python3 -m joey_research.evaluation freeze \
  --inputs "$ALL_INPUTS" --config configs/evaluation/protocol.json > "$MANIFEST"
```

Pin the output's `sha256` independently. The CLI manifest binds the canonical config digest as well as every input digest. Within each task, it ranks groups by a seeded SHA-256 key and assigns integer group counts by largest remainders (ties by split name). A task too small to populate a requested split fails. Allocations are fractions of **groups**, not rows. `select_frozen_split(inputs, manifest, pinned_digest, split)` verifies the full universe and returns only the requested rows. Never silently rerun the split on a filtered universe.

Keep labels physically separated by split. Give the inference runner only the public projections; do not give it a combined labeled source file and rely on dropping columns. On an already released selection split:

```sh
PYTHONPATH=src python3 -m joey_research.evaluation score \
  --config configs/evaluation/protocol.json --inputs "$ALL_INPUTS" \
  --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_PIN" --split selection \
  --labels "$SELECTION_LABELS" --arm "p1=$P1_PROBS" --arm "p2=$P2_PROBS" \
  --baseline p1 --passes p1 p2
```

The reserved split `final` additionally requires `--finalists FILE --finalists-sha256 DIGEST`. Before opening any label or prediction file, the CLI checks the SHA-256 of the **raw receipt file bytes** and its plan. The receipt is a JSON object with `schema: "joey-finalists-v1"`, `manifest_sha256`, `baseline`, `arms` (name to frozen method-spec SHA-256), `passes` (ordered names or `null`), and `primary` (e.g. `"macro.accuracy"`). Obtain the pin from a separate trusted freeze record. The scorer checks declared names and digest format; it cannot attest that a model runner actually used those method specs or that a previous process never read labels. Those are external custody and run-ledger requirements.

The in-memory API is `evaluate(inputs, labels, arms, baseline=..., pass_series=..., threshold=..., bootstrap_reps=..., bootstrap_seed=...)`. It assumes the caller already checked custody, the split and release. Its result has `arms[name].micro`, `.macro`, `.per_task`, `paired_deltas[name]` vs the baseline and optional `passes`. Output is strict JSON. Unknown, duplicate, missing, or stale IDs/digests are fatal; do not inner-join or zip files by row order.

## Reading a report

| Quantity | Convention |
| --- | --- |
| Accuracy | First maximum in frozen choice order; ties counted. Micro uses all states. Macro is the unweighted mean over task families, not over rows or classes. `equal_group_accuracy` is a separate descriptive equal-source-group measure. |
| NLL / Brier | Natural logarithm of the gold probability, negated; no epsilon floor. Brier is `sum_j (p[j] - 1[j=gold])**2`, not a mean over classes. |
| Top-label ECE | 15 fixed equal-width confidence bins `[i/15,(i+1)/15)`, last bin includes 1. Sum of bin frequency times absolute accuracy/confidence gap. Micro pools decisions before binning; macro averages per-task ECE. Empty bins contribute zero. |
| Confident errors | Fixed threshold comparison `max(p) >= threshold`. Joint = confident wrong / **all states**; conditional = confident wrong / **confident states**. Report `confident_wrong`, `confident`, `n` and coverage. A zero conditional denominator is `null`, not zero. A macro with an undefined constituent is `null`. |
| Risk/coverage | Accept whole equal-confidence blocks, high to low. Coverage uses all states; risk uses accepted states. No risk defined at zero coverage. AURC uses post-block risk over each coverage increment; do not interpolate through tied examples or claim an unattainable target coverage. |
| Passes | Corrections are wrong-to-right, corruptions right-to-wrong; joint rates use valid pairs and conditional rates use the corresponding initial status. Overthinking means correct at **any earlier** pass and wrong at the last (complete, valid paths only). Inflation gap = change in mean top confidence minus change in accuracy on identical valid pairs. Persistent errors also get a separate confidence-change count and mean. |
| Paired uncertainty | Draw the original number of **groups** with replacement within each task. Reuse the multiplicities across both arms and all rows in each group. Recompute each micro and equal-task macro metric; report candidate minus baseline, with linearly interpolated 2.5th/97.5th percentiles. The interval is withheld if any task has fewer than two groups, an estimate is undefined, or any replicate is undefined. These are pointwise, not multiplicity-adjusted, intervals; pass/risk curves are descriptive. |

Negative, nonfinite, nonnumeric, wrong-size or unnormalized vectors are invalid. Finite vectors with absolute sum error at most `1e-6` are normalized once; logits are not accepted. Explicit runner errors and invalid vectors count as incorrect for accuracy and as no accepted prediction for coverage. Full-denominator proper scores, ECE and full AURC are `null` for an incomplete distribution set. Partial risk area and valid-only bin supports are explicitly labeled. A valid zero probability on gold gives mathematical `+infinity` NLL, serialized as `null` with `nll_status="infinite"` and a zero-gold count. Use `all_distributions_valid`, support counts and finite status before interpreting any gain. Encode nonfinite runtime output as an explicit failure, not a nonstandard JSON `NaN` token.

## Diagnostic sources

`benchmarks.routing.iter_inputs(namespace=..., seed=..., groups_per_depth=...)` yields **inputs only**. Each graph has independent random bijections between layers, opaque equal-length station names, unordered rules, and a menu shuffled independently of the requested terminal. Every terminal is queried under both renderings, so each option position is correct equally often *within a graph and template*. `inverse_label(row)` is a label-side verifier: it parses the actual text, checks disjoint equal-length multi-hop paths, and walks backward from the goal. Never invoke it on campaign final rows before the finalists freeze. This is synthetic controlled reasoning, not web task success.

`benchmarks.mind2web.project_step(trajectory, index)` takes only the annotation ID, task text, step ID, and pre-action cleaned markup. It returns `Projection.input` plus a **private** joining map. It constructs a bounded menu from named interactive elements without consulting positives, negatives, operation, current/future action text, or a learned retriever. `adjudicate_step` is separate: an existing positive outside the listed set maps to "none"; empty positive annotations or multiple listed positive elements cannot be scored with a unique gold index. Report these as blockers/coverage losses; do not select a favorable labeled subset after freeze. This is restricted element selection, not native retrieval, operation/value prediction, trajectory success, or a leaderboard score.
