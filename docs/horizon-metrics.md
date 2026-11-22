# Frozen routing panel: paired metrics

`compare_horizons` compares **one prespecified candidate with one prespecified baseline** on the same rows. The primary question is whether the candidate's accuracy gain is larger at six hops than at two. It is not a depth search or a finalist selection rule.

```python
from benchmarks.horizon_metrics import compare_horizons

report = compare_horizons(
    rows, labels, baseline, candidate,
    design_by_id=design_by_id, baseline_tokens=baseline_tokens,
    candidate_tokens=candidate_tokens, reps=2000, seed=4117,
)
```

Arguments after `candidate` are keyword-only. The return type is `dict`. The function consumes in-memory inputs after label release; it does not load a model, tokenize, derive labels, read files or write results.

## Input contract

| Argument | Required form |
|---|---|
| `rows` | Nonempty iterable of distinct, blinded evaluator rows: `id`, `group_id`, `task`, `prompt`, `choices`. Exactly four ordered legal choices; `group_id` identifies the independent graph. |
| `labels` | In **exactly** `rows` order: `{id, input_sha256, gold_index}`. Integer `gold_index` is zero-based in this row's menu. |
| `baseline`, `candidate` | In the same order: `{id, input_sha256, probs}`. `probs` is a list of four probabilities in the **original** legal menu order, not sorted by probability or by entry. |
| `design_by_id` | Exact ID coverage. Each value has exactly `{hop, entry, rendering, seed}`. `hop` is one of `1, 2, 4, 6`; `rendering` is `0` or `1`; generator `seed` is an integer. `entry` is a nonblank semantic root string; the choice at `gold_index` must equal `"Enter at " + entry`. |
| `baseline_tokens`, `candidate_tokens` | Exact ID coverage. Each value is a **recorded**, positive integer. Counts for the two arms must agree on **every** row, even outside the matched subset. |
| `reps`, `seed` | Nonnegative integer replicate count and integer **bootstrap** seed. `reps=0` computes points and supports only. |

Every graph belongs to one generator seed and one task. It must have exactly one row in each of 4 semantic entries × 2 renderings × 4 hops. The ordered menu must stay fixed across all targets in a graph/rendering; a different rendering may permute it. Panel size and generator seed strata come from the full input, **before** matching.

All three ordered streams must bind to `evaluation.input_sha256(row)`. Invalid bindings, coverage, design, counts or scores raise `evaluation.EvaluationError`; no rows are dropped to repair a comparison. Booleans are not integers or probabilities here. Native scoring accepts only finite Python `int`/`float` probabilities in `[0,1]` with `abs(fsum(probs) - 1) <= evaluation.PROB_SUM_ATOL`. It divides by the accepted sum before computing confidence, argmax or Brier. The **first** maximum in menu order wins a tie. A runner error is not a prediction. The caller is responsible for the frozen universe, graph independence, released design and original probability ordering; a digest binds the displayed menu, not the provenance of the numbers.

## What is averaged

For each row, `accuracy` is 0 or 1 and `brier` is the **sum** over four choices of `(p[i] - 1[i == gold])²` (range 0–2, lower is better). For either metric let `d(h)` be the candidate score minus the baseline score on a row at hop `h`.

* **Per hop:** the mean of `d(h)` over all rows at that hop. Corrections are baseline wrong → candidate right; their support is all baseline-wrong rows. Corruptions are baseline right → candidate wrong; their support is all baseline-right rows. Confidence is the normalized top probability of the indicated arm. Threshold support includes all its predictions with confidence **at least** the threshold; errors are its wrong answers among those predictions.
* **Primary:** for each `(graph, semantic entry, rendering)` tuple take `d(6) - d(2)`, then average over tuples. With the complete panel this equals the six-hop paired mean minus the two-hop paired mean. A positive accuracy interaction means more accuracy gain at six; a negative Brier interaction means a larger Brier reduction at six.
* **Exact-token secondary:** average the **same two-row difference** only over tuples whose recorded six- and two-hop token counts are equal. There is no nearest-length matching, cross-rendering substitution or imputation. `token_differences` describes `tokens(6) - tokens(2)` on **all** tuples, before filtering. Each eligible tuple contributes two paired rows, not two independent graphs.

Means are pooled ratios of sums across generator seeds, **not** equal-seed averages. Each observed row (per hop) or tuple (interaction) gets one vote. Each graph therefore has equal weight in the complete panel; in the matched panel its weight is its number of eligible tuples. A matched point describes only that subset, even if a whole seed has no matches. No matches means no point. Neither contrast establishes that one recurrent visit implements one graph hop.

## Uncertainty

In each replicate, sample as many graphs **with replacement within each generator seed** as that seed originally contained. Seeds are visited in numeric order, graphs in `group_id` order, using a local `random.Random(seed)`. One set of graph multiplicities weights every arm, entry, rendering, horizon and contrast, including the exact-token subset. Only counts and sums per graph enter the resampling loop. A graph with no matches remains in the draw, contributing zero matched numerator **and** denominator.

`ci95` is the pointwise 2.5th/97.5th percentile interval, linearly interpolated at `q × (reps - 1)` in the sorted replicates. It is not multiplicity-adjusted. An effect gets no interval if `reps=0`, any original seed has fewer than two graphs contributing to that effect, or **any** replicate has zero pooled denominator. `defined_reps` counts defined draws for that effect (zero if not attempted for support or `reps=0`); it is never used to quietly discard undefined draws and compute a shorter interval.

## Exact return keys

All dynamic keys are strings, including hops, thresholds, generator seeds and signed token differences. Leaves are numbers or `None` (`null` in JSON); `ci95`, when present, is a two-number list. Counts remain zero at zero support; conditional means and rates are `None`.

```text
report
  bootstrap: {reps, seed}
  by_hop["1" | "2" | "4" | "6"]:
    rows, graphs
    baseline: Arm
    candidate: Arm
    paired: Contrast
    correction: {count, support, rate}
    corruption: {count, support, rate}
    tokens: {min, max, mean}
  primary: Contrast
  exact_token:
    accuracy: Effect
    brier: Effect
    support: Support
    by_seed[generator seed]: Support
  token_differences: {counts, min, max, mean, mean_abs, equal_fraction}

Arm      = {accuracy, brier, right, wrong, mean_confidence_right,
            mean_confidence_wrong, thresholds}
thresholds["0.8" | "0.9" | "0.95" | "0.99"] = {errors, support, error_rate}
Contrast = {accuracy: Effect, brier: Effect}
Effect   = {estimate, ci95, defined_reps}
Support  = {total_graphs, eligible_graphs, total_tuples, eligible_tuples}
counts[signed token difference] = number of tuples at that difference
```

For the primary read `report["primary"]["accuracy"]`. For the sensitivity analysis read `report["exact_token"]["accuracy"]` **alongside** `report["exact_token"]["by_seed"]` and `report["token_differences"]["equal_fraction"]`. An eligible graph has at least one eligible tuple.
