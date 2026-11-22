# One-probe split execution

`joey_research.one_probe.run_one_probe(branch, *, cap, threshold) -> OneProbeResult`
answers one question: after **one** physical visit to the middle block, is the
native probability margin large enough to return that readout?

The caller freezes `cap` (integer `2`, `3`, `4` or `6`) and `threshold` (`0`,
`0.10`, `0.25`, `0.50`, `0.75` or `1.0`) before the call. Neither accepts a
boolean. There is no fit, target label, replay or worker admission in this module.

## Execution and accounting

Run the prelude once, traverse the bound block once, and give the first coda a
native `clone()` of that block's output. Read `branch.probabilities(first_logits)`
once. Subtract the second-largest slot probability from the largest, **without
rounding or renormalizing**. The first maximum in native order wins a tie; a
second slot at that maximum gives margin zero, which passes threshold zero.

| Condition | `reason` | `exit_depth` | Returned logits | Actual `(stock_calls, prelude_calls, block_calls, coda_calls)` |
|---|---|---:|---|---|
| `first_margin >= threshold` | `"margin"` | 1 | First coda's object | `(0, 1, 1, 1)` |
| `first_margin < threshold` | `"cap"` | `cap` | Last coda's object | `(0, 1, cap, 2)` |

On a miss, feed the **retained original** first block output to the next block.
Perform all `cap - 1` remaining visits on that trajectory, with no coda between
them. Read out the last block's output directly. Neither path calls `stock()`,
appends a token, re-runs the prelude, or does a forward to recover logits.

Time the whole call, including the first-hidden clone on **both** paths, vector
validation and native probability extraction. `LoopCounts` counts callbacks,
not copies or latency; a miss costs one more coda than a final-only naive loop.

## Coordinator contract

Pass one already-bound branch. The host owns the input, slot mapping, physical
window, model graph, full-sequence representation, masks and positions. Keep
these fixed until the result is consumed. `labels` are native slot keys, **not**
ground truth: an ordered tuple or list of at least two distinct nonempty strings
or integers. The primitive snapshots their values and checks the binding across
callbacks; mutating a list in place is not a way to change the result's order.

The split calls run inside `branch.inference_context()`. `branch.prelude()`
supplies the initial hidden; each `branch.block(hidden, state)` and
`branch.coda(hidden, state)` receives its own `branch.fresh_state()`. The hidden's
`clone()` must preserve its representation, device and dtype with **independent
storage**. There is no shallow-copy or assumed-immutable fallback.
`branch.probabilities(logits)` must read those logits without modifying them and
return the unrounded, calibrated native vector in the bound order.

`OneProbeResult` is a frozen dataclass with these fields:

| Field | Meaning |
|---|---|
| `logits` | Exact native coda output object at `exit_depth`, not a converted vector. Treat its storage as read-only. |
| `labels` | Immutable tuple of native keys indexing both vectors. |
| `first_probabilities` | `tuple[float, ...]` snapshot of the **first** readout, even on a miss. |
| `first_margin` | `float` used in the single inclusive comparison. |
| `exit_depth` | `int`: number of physical block calls, not the number of extra visits. |
| `reason` | Literal `"margin"` or `"cap"`, as above. |
| `counts` | The public `joey_research.loop.LoopCounts` for this execution. |

Both coda outputs, when present, must be flat finite numeric vectors of native
width. First probabilities must also lie in `[0, 1]` with absolute mass error
at most `1e-6`. Booleans are not scores. Invalid vectors, policies or slot order
raise `ValueError` rather than selecting a fallback. If the coordinator needs
the exit probabilities, reuse the first vector on a hit and the last logits on
a miss; neither needs another coda:

```python
result = run_one_probe(branch, cap=4, threshold=0.25)
p_exit = result.first_probabilities if result.exit_depth == 1 else branch.probabilities(result.logits)
```

## CPU contract test

From the repository root, run the invented fixture; no install is needed.

```sh
PYTHONPATH=src:tests:. python3 -B -m unittest -q test_one_probe
```

These tests establish callback and hidden-state lineage, not model quality or speed.
