# Decisions across completed passes

Use `loop_analysis` with the [frozen-block adapter](loop.md) to compare aggregation and stopping rules on the **same question and ordered choices**. It accepts stock at `k=1` and successive `naive` or `damped_extra` passes. It does not interpret an RK stage, Euler substep, missing probe, or final-only trace as a sequence of completed predictions.

```python
from joey_research.loop import LoopPlan
from joey_research.loop_analysis import Policy, SlotOrder, Stop, actuate, record, replay

plan = LoopPlan(k=3, method="naive")
policy = Policy(pool="final", stop=Stop("margin", 0.5), fallback="first")

def native_order():
    return SlotOrder(split, input_key, tuple(ordered_slot_keys()))

trace = record(plan, question, native_order)       # full diagnostic run
sim = replay(trace, policy)                      # no additional model calls
live = actuate(plan, question, native_order, policy)  # independent live run
```

`ordered_slot_keys()` stands for the adapter's authoritative slot metadata for the vector it emits. Do not sort it, derive it from tensor width, or label previously unbound vectors after the run. Each callback result is snapshotted at emission; mismatched identities, changed slot orders, duplicate keys, wrong widths, and nonfinite scores are rejected. The returned `slot` is an input-local key, not a decoded answer label. Slot probabilities are copied to the host for these reference policies; logical call counts are not GPU timings.

| `pool` | Distribution at decision pass `d` |
| --- | --- |
| `first`, `final` | Softmax of the first or current logits. Winner uses the original logits to avoid a tie introduced by rounding softmax. |
| `probability_mean` | Arithmetic mean of the `d` separate softmax distributions. |
| `normalized_logit_mean` | Center each logit vector across its slots and divide by its population standard deviation (a constant vector becomes zero). Average those standardized vectors, then apply softmax once. |
| `first_anchor` | `first_weight * p_first + (1-first_weight) * p_current`; not a mean of all intervening passes. |

Stopping uses the **current unpooled** scores. Margin is the difference between the top two probabilities; stop at or above the threshold. Entropy is Shannon entropy divided by `log(number_of_slots)`; stop at or below the threshold. Stability requires the same unique top slot for the specified number of consecutive passes, including this one. A tied pass cannot trigger and breaks a streak. An output tie is resolved in native slot order. If no rule matches *including on the last pass*, `reason="no_trigger"` and `fallback_used` names the selected action (`first` or `aggregate`). Falling back to the first prediction does not refund later calls.

**Costs.** `actuate` returns `executed` counts from the actual callbacks. Fixed first uses one block and one coda; fixed final uses `k` blocks and one coda. All other policies in this implementation run coda on every visited pass, including the stop pass. `record` always pays for `k` blocks and `k` codas. `replay` reports that recording under `recorded`, zero new execution under `executed`, and the hypothetical live cost under `simulated`. Do not add the same recording cost again for each policy tried.

**Choosing a threshold.** `fit_threshold(expected_inputs, traces, policy, thresholds, min_agreement=...)` takes a predeclared `SlotOrder` manifest for one discovery or selection split. It rejects incomplete, duplicate, wrong-order, or extra inputs and mixed repeat plans. Use one frozen model and partition for the whole comparison. Eligible candidates must meet the requested fraction of paired agreements with the full-budget **prediction**, not ground truth. Among those, it minimizes simulated block calls, then coda reads, then favors more agreements and earlier candidate order. Freeze the result before evaluating another split. No answer labels are used, and calibration and final splits are ineligible for fitting.

**Synthetic CPU witness, not a model evaluation:** with ordered slots `(right, left)`, the three pass distributions are `(.4,.6)`, `(.8,.2)`, and `(.1,.9)`. Fixed final picks `left` using 3 blocks and 1 coda. A live margin threshold of `0.5` picks `right` at pass 2 using 2 blocks and **2 codas**. Replay makes the same decision but executes no model calls; its source already cost 3 blocks and 3 codas. This example establishes a decision change, not which answer is better.
