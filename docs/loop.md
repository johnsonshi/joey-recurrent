# Frozen-block inference

`run_loop` changes how often a whole contiguous layer window runs. It does not choose the window, modify weights, or decode predictions. Start by comparing **ordered logits** from the real stock call (`k=1`) with the `zero_extra` split control (`k>1`) on the same input. The first path alone cannot test whether the split reassembles the model correctly.

```python
from joey_research.loop import LoopPlan, run_loop

stock = run_loop(LoopPlan(k=1), question)
split = run_loop(LoopPlan(k=3, method="zero_extra"), question)
# Check stock.logits against split.logits with the adapter's stated tolerance.
result = run_loop(LoopPlan(k=3, method="rk", beta=0.5), question)
prediction = question.decode(result.logits)
```

## Adapter contract

Bind `question` to one immutable full-prefix input and one partition. `stock()` calls the actual unmodified forward path. `prelude()` returns the full-sequence hidden state just before the window. `block(hidden, state)` runs **all** window layers in their original order, exactly once, and returns a hidden state of the same shape. `coda(hidden, state)` runs the remaining layers and normal readout. For choice scoring, stock and coda return 1-D fp32 logits in the same slot order; their length is specific to this question.

`fresh_state()` supplies the identical positions and attention mask but isolated mutable execution state for **each** block or coda call. Stock and prelude must also start clean. This is full-sequence recomputation, not a token-decoding cache policy. Do not advance positions with loop depth, share a writable cache across stages, or let later calls overwrite returned hidden states or logits.

The engine enters `question.inference_context()` before calling the model or doing arithmetic. For PyTorch, return `torch.inference_mode()` and supply an already-frozen model in evaluation mode. The engine does not switch model modes, cast tensors, or update tensors in place. Math uses the caller's hidden-state arithmetic; algebraic identities need not be bitwise identical in reduced precision.

## Updates and budgets

Write `x` for the prelude output, `g` for the complete block, and `f(y) = g(y) - y`. For every valid plan, `k=1` bypasses **all** split callbacks. For `k>1`, `k` counts total block evaluations, not additional repeats:

| Method | Result before coda | Block calls |
| --- | --- | ---: |
| `zero_extra` | `g(x)`: explicit prelude, full first traversal, coda | 1 |
| `naive` | Apply `g` successively starting at `x` | `k` |
| `damped_extra` | Start at `g(x)`; apply `y + alpha*f(y)` for the remaining evaluations | `k` |
| `euler` | Start at `x`; take `k` substeps `y + f(y)/k` over **one** unit interval, ending at `E` | `k` |
| `rk`, with `beta` | `beta*g(x) + (1-beta)*E`; the first stage supplies the **output** anchor | `k` |
| `prelude_anchor` | `beta*x + (1-beta)*E`; a separate **input**-anchor control, not the paper's output rule | `k` |
| `rk`, with `tableau` | One explicit unit-step RK update with the specified stage and output weights | `k` |

`alpha` and `beta` lie in `[0,1]`. `damped_extra(alpha=0)` is a compute-matched no-update control: it still evaluates each extra block. `zero_extra` actually skips that work. Likewise, `rk(beta=1)` returns the original one-pass hidden output but still pays for `k` evaluations.

[Chen et al., Algorithm 1 and equations 13-16 (v1)](https://arxiv.org/html/2605.23872v1#alg1) define each RK slope at its **own** stage input. The built-in tableau gives every earlier slope weight `1/k` when constructing stages. Its output slope weights are `beta + (1-beta)/k` for the first and `(1-beta)/k` for each other slope. The terminal blend in the table computes this update without a separate anchor evaluation or a list of `k` hidden-sized slopes. Stage count does not imply order of accuracy. For a different tableau, pass `RKTableau(a, b)` instead of `beta`; for example, `RKTableau(((), (0.5,)), (0, 1))` is the two-evaluation midpoint rule for a **single** unit step.

## Readouts and cost

`readout="final"` runs coda once and returns no intermediate trace. `readout="diagnostic"` records one `Readout(block_evaluations, kind, logits)` per evaluated stage, including the returned endpoint; it never rereads the final logits. For naive or extra-damped repeats, probes are the actual iterates. Euler probes are completed substeps. For RK and the prelude-anchor control, nonfinal probes are `g(stage_input)` passed through coda, **not** completed lower-budget solutions.

For an actuated early exit, pass `stop_when=predicate` with diagnostic naive or extra-damped repeats. The predicate receives the current nonfinal `Readout`, returns a boolean, and must use only information available during inference. A true result skips all later block and coda calls and sets `stopped_early`. Euler and RK reject early-exit predicates because stopping mid-step would change the integration rule.

`result.counts` distinguishes actual stock, prelude, block, and coda calls. `block_evaluations` and `coda_readouts` include one implicit evaluation/readout per stock call. `layer_evaluations(p,b,c)` weights those counts by the actual prelude, window, and coda depths. For partition `(2,3,1)` and `k=3`, a fixed loop costs `2 + 3*3 + 1 = 12` layer calls in final mode, or `2 + 3*3 + 3*1 = 14` in diagnostic mode. Head readouts still count when `c=0`. These are per-question logical calls, not FLOPs or elapsed time; time a stock implementation with shared prefill separately.

[Example plans](../configs/loop/examples.json) are test settings, not recommended hyperparameters. From the source tree, run `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_loop.py'`. The fixtures check discrete trajectories, fresh state, and accounting. They do not measure any checkpoint's accuracy or speed.
