# Accounting for one measured decision

`joey_research.trace_cost.account_trace(trace, *, text_config, window, method_id, weight_file_bytes=None)`
is a pure, standard-library function. Pass mappings already in memory. It does not load a model,
open a trace, run inference, or measure hardware.

| Result | Meaning |
| --- | --- |
| `weight_file_bytes` | Optional host-supplied size of the serialized weight files (e.g. `compact._storage`'s field of the same name). Count once per artifact, **not** per decision, loop or readout. Not resident or peak memory. |
| `cuda_event_ms` | The recorded CUDA-event interval for the entire row, including both orders when present. Not wall time or an estimate from shapes. Absent timing or size is `None`, not zero. |
| `logical_layer_calls`, `logical_token_layer_visits` | Schedule counts by physical layer type, not kernel launches. `linear_attention` denotes causal GatedDeltaNet; `full_attention` denotes the native block-masked attention layers. |
| `split_projection_flops_approx`, `full_attention_dense_envelope` | Selected shape products for the **split path only**. Neither is total model FLOPs, a GPU counter, a memory measurement or a speed prediction. |

## Inputs and grouping

For memory, `allocator_observation(complete)` takes the corresponding **verified worker completion**, not an individual trace row. It returns raw bytes for the PyTorch allocator: `steady_allocated_bytes` and `steady_reserved_bytes` are instantaneous readings at the end of the timed loop, not time averages. `peak_allocated_bytes` and `peak_reserved_bytes` are maxima after the worker reset peak statistics following loading and parity checks. Resetting statistics does not clear the allocator's pool. Allocated and reserved peaks need not occur at the same moment. Device capacity is a bound, not measured consumption. These counters exclude memory outside this process's PyTorch allocator; none is an estimate of serialized weight bytes. Do not sum them over decisions or posthoc projections.

Pass the physical **text** config of the measured method, not a multimodal wrapper or the
original unpruned config for a compact model. Supply `num_hidden_layers`, ordered `layer_types`,
and the dimensions below. The supported shape schema is `qwen3_5_text` with `attn_output_gate`
true, no MLP-only layers and no attention bias. Layers are not inferred from a cycle.
The physical window `[lo, hi)` must satisfy `0 <= lo < hi <= num_hidden_layers`.

A worker row supplies integer `tokens`, `layer_calls`, and the four `callback_counts`
(`stock_calls`, `prelude_calls`, `block_calls`, `coda_calls`). No method is inferred from `k`,
readouts, probabilities, or how many layers ran. For example, a host reporter can keep rows apart:

```python
from collections import defaultdict
from joey_research.trace_cost import account_trace

by_method = defaultdict(list)
for method_id, text, window, file_bytes, row in measured_rows:
    r = account_trace(row, text_config=text, window=window, method_id=method_id,
                      weight_file_bytes=file_bytes)
    by_method[r["method_id"]].append(r)
```

Use the host's measured method key (e.g. its bound `method_sha256`) and matching config/artifact.
Sum counts or envelope endpoints **after** accounting for each row; never pool token lengths
first. Report the coverage of non-`None` estimates rather than substituting zero for withheld work.

## What is counted

- **Native cached stock:** one opaque stock callback is one *logical* visit to each physical
  layer for the recorded branch. Its token-layer visits are nominal, not a reconstruction of
  shared-prefix, suffix, cache-fork, sibling, padding or retry work. Neither split projection
  FLOPs (including the head) nor dense pair envelopes are reported for stock.
- **Fresh-state split:** for each type, weight its counts in `[:lo]`, `[lo:hi]`, `[hi:]` by
  the observed prelude, block and coda calls. Their sum across types must match `layer_calls`.
  Every nonempty stage revisits all branch positions; no block or coda inherits sequence cache
  from its predecessor. Extra blocks and extra codas are counted separately relative to one
  of each per split forward. Even when `hi` is the depth, every split coda still runs the head.
- **Order ensemble:** the worker admits one or two orders, exactly one prelude/block/coda
  per order. `forward_calls` / `ensemble_calls` must agree if both present; an unmarked row is
  one forward. With `T = sum(n_i)` recorded tokens, linear terms use `T` times **one order's**
  layer counts. Heads use the **observed** coda count. For a dense pair envelope we need
  `sum(n_i²)`, not `T²`. Using only the sum and a minimum of two tokens per branch
  (`native._layout` requires nonempty prefix and suffix), the lower bound balances the integer
  lengths; the upper bound puts all spare tokens in one order. The limits coincide for one order.
  Other aggregate schedules and mixed
  stock/split callbacks are rejected, not divided into guessed per-order work.

## Shape products, not execution counts

One multiply-add counts as approximately two FLOPs; the shape arithmetic itself is integer.
Let `H = hidden_size`, `M = intermediate_size`, `Q = num_attention_heads`, `B = num_key_value_heads`,
`d = head_dim`, `K = linear_num_key_heads × linear_key_head_dim`, `L = linear_num_value_heads`,
and `V = L × linear_value_head_dim`. From `compact._schemas` and `native.NativeBranch._coda`:

| Component | Approximate FLOPs per indicated unit |
| --- | --- |
| MLP, every token-layer visit | `2H × 3M` (gate, up, down) |
| GatedDeltaNet dense projections, per linear token-layer visit | `2H × [(2K+V) + V + L + L + V]` (qkv, z, a, b, out) |
| Full-attention dense projections, per full token-layer visit | `2H × (2Qd + Bd + Bd + Qd)` (gated Q, K, V, O) |
| Split LM head, per coda | `2H × vocab_size`. Final norm sees all positions; the head sees **only the last position**, projects the full vocabulary, then legal slots are gathered. |
| Full-attention score and value, per dense token pair | **Separately** `2Qd` for QK and `2Qd` for AV, not KV-head count. `token_pairs` sums one unmasked square per full-layer visit before multiplying by heads. |

Envelope pairs `(lower, upper)` are **inclusive bounds on the dense shape formula** over possible
order lengths, not bounds on actual executed operations. They are not causal triangles, counts
of native mask entries, or padded/kernel-exact shapes. No estimate is made for GatedDeltaNet
convolution, recurrence or scan, elementwise gates, norms, embeddings, rotary/mask setup, softmax,
loop arithmetic, cache traffic, fusion or kernel padding.

Run the invented fixtures from this folder: `PYTHONPATH=src python3 -B -S -m unittest discover -s tests -p test_trace_cost.py`.
