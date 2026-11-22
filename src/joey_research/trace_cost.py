"""Shape-based accounting for one worker trace row. No model, tensor, file or device access."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import NamedTuple


_LINEAR = "linear_attention"  # causal GatedDeltaNet in the Joey text decoder
_FULL = "full_attention"
_TYPES = (_LINEAR, _FULL)
_CALLS = ("stock_calls", "prelude_calls", "block_calls", "coda_calls")
_MIN_TOKENS = 2  # native._layout requires a nonempty prefix AND suffix


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def allocator_observation(completion: Mapping) -> dict[str, int]:
    """Copy one batch-one A100 run's PyTorch byte counters, without inferring GPU use from storage."""
    if not isinstance(completion, Mapping) or completion.get("hardware") != "A100":
        raise ValueError("expected an A100 worker completion")
    if _integer("batch_size", completion.get("batch_size"), 1) != 1:
        raise ValueError("allocator reporting requires the measured batch-one protocol")
    capacity = _integer("total_device_bytes", completion.get("total_device_bytes"), 1)
    fields = ("steady_allocated_bytes", "peak_allocated_bytes", "steady_reserved_bytes", "peak_reserved_bytes")
    observed = {name: _integer(name, completion.get(name)) for name in fields}
    end, peak = observed["steady_allocated_bytes"], observed["peak_allocated_bytes"]
    retained, pool_peak = observed["steady_reserved_bytes"], observed["peak_reserved_bytes"]
    if not (end <= peak <= pool_peak <= capacity and end <= retained <= pool_peak):
        raise ValueError("inconsistent end, peak, reservation or device capacity")
    return {"total_device_bytes": capacity, **observed}


class _Matmuls(NamedTuple):
    mlp: int           # per token-layer visit
    linear: int
    full: int
    score_or_value: int  # per full-attention token pair, across query heads
    head: int            # per split coda, not per prompt token


def _matmuls(text: Mapping) -> _Matmuls:
    """Two FLOPs per multiply-add in the dense weight shapes in compact._schemas."""
    if (text.get("model_type") != "qwen3_5_text" or text.get("attn_output_gate") is not True
            or text.get("mlp_only_layers", []) != [] or text.get("attention_bias", False) is not False):
        raise ValueError("expected Joey's gated-Q, dense-MLP text configuration")

    def dim(key: str) -> int:
        return _integer(key, text.get(key), 1)

    h, m = dim("hidden_size"), dim("intermediate_size")
    q, kv, d = dim("num_attention_heads"), dim("num_key_value_heads"), dim("head_dim")
    lk, lv = dim("linear_num_key_heads"), dim("linear_num_value_heads")
    if q % kv or lv % lk:
        raise ValueError("incompatible query/KV or linear value/key head groups")
    k_width = lk * dim("linear_key_head_dim")
    v_width = lv * dim("linear_value_head_dim")
    return _Matmuls(
        2 * h * (3 * m),                             # gate, up, down
        2 * h * (2 * k_width + 3 * v_width + 2 * lv),  # qkv, z, out, a, b
        2 * h * (3 * q + 2 * kv) * d,               # doubled Q, K, V, O
        2 * q * d,
        2 * h * dim("vocab_size"),
    )


def _schedule(trace: Mapping) -> tuple[int, tuple[int, ...]]:
    counts = trace.get("callback_counts")
    if not isinstance(counts, Mapping) or set(counts) != set(_CALLS):
        raise ValueError(f"callback_counts must contain exactly {_CALLS}")
    calls = tuple(_integer(key, counts[key]) for key in _CALLS)
    stock, prelude, block, coda = calls
    forwards = _integer("forward_calls", trace.get("forward_calls", trace.get("ensemble_calls", 1)), 1)
    ensemble = _integer("ensemble_calls", trace.get("ensemble_calls", forwards), 1)
    if ensemble != forwards or forwards not in (1, 2):
        raise ValueError("expected one or two orders with matching forward/ensemble counts")
    if "ensemble_calls" in trace or forwards > 1:
        # worker._settings admits only zero-extra split passes to _measure_ensemble.
        if calls != (0, forwards, forwards, forwards):
            raise ValueError("order ensemble requires one split pass per order; cannot apportion other totals")
    elif stock:
        if calls != (1, 0, 0, 0):
            raise ValueError("expected one opaque stock call without split callbacks")
    elif prelude != 1 or not 1 <= coda <= block:
        raise ValueError("split forward requires one prelude and 1 <= coda_calls <= block_calls")
    return forwards, calls


def _dense_envelope(tokens: int, forwards: int, full_per_order: int, rate: int) -> dict:
    # Extremize sum(n_i**2) at fixed sum: balance the lengths, or put all slack in one order.
    n, r = divmod(tokens, forwards)
    smallest = (forwards - r) * n**2 + r * (n + 1)**2
    rest = forwards - 1
    largest = (tokens - rest * _MIN_TOKENS)**2 + rest * _MIN_TOKENS**2
    pairs = (full_per_order * smallest, full_per_order * largest)
    flops = (rate * pairs[0], rate * pairs[1])
    return {"token_pairs": pairs, "score_flops_approx": flops, "value_flops_approx": flops}


def account_trace(trace: Mapping, *, text_config: Mapping, window: tuple[int, int] | list[int],
                  method_id: str, weight_file_bytes: int | None = None) -> dict:
    """Account for one batch-one worker row using its measured method's physical text config.

    Returns logical counts, split-only projection estimates and an inclusive dense-pair
    envelope. Cached stock estimates are None. Bytes and CUDA-event ms are observations,
    never inputs to the estimates. Raises ValueError on unsupported or inconsistent input.
    """
    if not isinstance(trace, Mapping) or not isinstance(text_config, Mapping):
        raise ValueError("pass the trace row and text config as mappings")
    if not isinstance(method_id, str) or not method_id.strip():
        raise ValueError("pass the measured method's identifier")
    if weight_file_bytes is not None:
        _integer("weight_file_bytes", weight_file_bytes)
    ms = trace.get("cuda_event_ms")
    if "cuda_event_ms" in trace and (type(ms) not in (float, int) or ms < 0
                                     or (type(ms) is float and not isfinite(ms))):
        raise ValueError("cuda_event_ms must be finite and nonnegative")

    depth = _integer("num_hidden_layers", text_config.get("num_hidden_layers"), 1)
    types = text_config.get("layer_types")
    if (not isinstance(types, (list, tuple)) or len(types) != depth
            or any(kind not in _TYPES for kind in types)):
        raise ValueError("expected one supported layer type per physical text layer")
    if (not isinstance(window, (list, tuple)) or len(window) != 2
            or any(type(i) is not int for i in window) or not 0 <= window[0] < window[1] <= depth):
        raise ValueError("window must be a nonempty half-open range in the physical decoder")
    lo, hi = window
    rates = _matmuls(text_config)
    forwards, calls = _schedule(trace)
    stock, prelude, block, coda = calls
    tokens = _integer("tokens", trace.get("tokens"), _MIN_TOKENS * forwards)

    physical = {kind: types.count(kind) for kind in _TYPES}
    stages = (types[:lo], types[lo:hi], types[hi:])
    layer_calls = {
        kind: stock * physical[kind] + sum(n * stage.count(kind) for n, stage in zip(calls[1:], stages))
        for kind in _TYPES
    }
    if sum(layer_calls.values()) != _integer("layer_calls", trace.get("layer_calls"), 1):
        raise ValueError("layer_calls disagrees with callbacks, physical depth or window")
    # Only the validated one-pass ensemble has aggregate counts. Each order visits the
    # same layers, even when its length differs. Never apply the aggregate count to T.
    per_order = {kind: count // forwards for kind, count in layer_calls.items()}
    visits = {kind: tokens * count for kind, count in per_order.items()}

    projections = envelope = None
    if not stock:
        projections = {
            "mlp": rates.mlp * sum(visits.values()),
            "gated_deltanet": rates.linear * visits[_LINEAR],
            "full_attention": rates.full * visits[_FULL],
            "last_position_vocab_head": rates.head * coda,
        }
        envelope = _dense_envelope(tokens, forwards, per_order[_FULL], rates.score_or_value)

    return {
        "method_id": method_id, "window": (lo, hi),
        "weight_file_bytes": weight_file_bytes, "cuda_event_ms": ms,
        "path": "opaque_native_cached_stock" if stock else "fresh_state_split",
        "forward_calls": forwards, "tokens_sum": tokens, "callback_counts": dict(zip(_CALLS, calls)),
        "extra_split_block_calls": block - prelude, "extra_split_coda_calls": coda - prelude,
        "physical_layers": physical, "logical_layer_calls": layer_calls,
        "logical_token_layer_visits": visits,
        "split_projection_flops_approx": projections, "full_attention_dense_envelope": envelope,
    }
