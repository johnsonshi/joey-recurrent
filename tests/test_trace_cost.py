import unittest

from joey_research.trace_cost import account_trace, allocator_observation


LINEAR, FULL = "linear_attention", "full_attention"
TEXT = {
    "model_type": "qwen3_5_text", "attn_output_gate": True,
    "num_hidden_layers": 5, "layer_types": [FULL, LINEAR, LINEAR, FULL, LINEAR],
    "hidden_size": 6, "intermediate_size": 7, "vocab_size": 11,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 2,
    "linear_num_key_heads": 1, "linear_key_head_dim": 3,
    "linear_num_value_heads": 2, "linear_value_head_dim": 4,
}
WINDOW = (1, 4)


def row(tokens, layer_calls, calls, **fields):
    names = ("stock_calls", "prelude_calls", "block_calls", "coda_calls")
    return {"tokens": tokens, "layer_calls": layer_calls, "callback_counts": dict(zip(names, calls)), **fields}


def account(trace, *, text=TEXT, window=WINDOW, method="invented", size=None):
    return account_trace(trace, text_config=text, window=window, method_id=method, weight_file_bytes=size)


class TraceCostTests(unittest.TestCase):
    def test_end_and_peak_allocator_counters_are_not_weight_or_timing_estimates(self):
        complete = {"hardware": "A100", "batch_size": 1, "total_device_bytes": 1000,
                    "steady_allocated_bytes": 200, "peak_allocated_bytes": 700,
                    "steady_reserved_bytes": 450, "peak_reserved_bytes": 800,
                    "weight_file_bytes": 2000, "cuda_event_ms": 11}
        observed = allocator_observation(complete)
        self.assertEqual((observed["steady_reserved_bytes"], observed["peak_allocated_bytes"]), (450, 700))
        self.assertNotIn("weight_file_bytes", observed)
        for change in ({"steady_allocated_bytes": None}, {"peak_reserved_bytes": True},
                       {"peak_allocated_bytes": 199}, {"steady_reserved_bytes": 199},
                       {"peak_reserved_bytes": 699}, {"total_device_bytes": 799}, {"batch_size": 2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                allocator_observation({**complete, **change})

    def test_repeated_window_and_diagnostic_coda(self):
        result = account(row(5, 9, (0, 1, 2, 2), cuda_event_ms=0.625), size=2041)
        self.assertEqual((result["weight_file_bytes"], result["cuda_event_ms"]), (2041, 0.625))
        self.assertEqual(result["path"], "fresh_state_split")
        self.assertEqual((result["extra_split_block_calls"], result["extra_split_coda_calls"]), (1, 1))
        self.assertEqual(result["physical_layers"], {LINEAR: 3, FULL: 2})
        self.assertEqual(result["logical_layer_calls"], {LINEAR: 6, FULL: 3})
        self.assertEqual(result["logical_token_layer_visits"], {LINEAR: 30, FULL: 15})
        # Dense widths: MLP 7+7+7; delta 14+8+2+2+8; full 16+4+4+8. All have H=6.
        self.assertEqual(result["split_projection_flops_approx"], {
            "mlp": 11340, "gated_deltanet": 12240, "full_attention": 5760,
            "last_position_vocab_head": 264,
        })
        self.assertEqual(result["full_attention_dense_envelope"], {
            "token_pairs": (75, 75), "score_flops_approx": (1200, 1200),
            "value_flops_approx": (1200, 1200),
        })
        # Same two block calls, final readout only: one fewer linear tail and one fewer head.
        final = account(row(5, 8, (0, 1, 2, 1)))
        self.assertEqual(final["logical_layer_calls"], {LINEAR: 5, FULL: 3})
        self.assertEqual(final["split_projection_flops_approx"]["last_position_vocab_head"], 132)
        self.assertEqual(final["full_attention_dense_envelope"], result["full_attention_dense_envelope"])

    def test_stock_and_one_split_pass_have_different_coverage(self):
        # Identical nominal schedule, even with no decoder layers after the split block.
        stock = account(row(4, 5, (1, 0, 0, 0)), window=(0, 5), method="cached")
        split = account(row(4, 5, (0, 1, 1, 1)), window=(0, 5), method="split-once")
        self.assertEqual(stock["logical_token_layer_visits"], split["logical_token_layer_visits"])
        self.assertEqual(stock["logical_layer_calls"], {LINEAR: 3, FULL: 2})
        self.assertEqual((stock["method_id"], split["method_id"]), ("cached", "split-once"))
        self.assertEqual(stock["path"], "opaque_native_cached_stock")
        self.assertIsNone(stock["split_projection_flops_approx"])
        self.assertIsNone(stock["full_attention_dense_envelope"])
        self.assertIsNone(stock["cuda_event_ms"])
        self.assertIsNone(stock["weight_file_bytes"])
        self.assertEqual(split["split_projection_flops_approx"]["last_position_vocab_head"], 132)

    def test_aggregate_orders_bound_separately_shaped_forwards(self):
        merged = account(row(11, 10, (0, 2, 2, 2), forward_calls=2, ensemble_calls=2, cuda_event_ms=1.5))
        self.assertEqual(merged["logical_layer_calls"], {LINEAR: 6, FULL: 4})
        self.assertEqual(merged["logical_token_layer_visits"], {LINEAR: 33, FULL: 22})
        self.assertEqual(merged["cuda_event_ms"], 1.5)
        self.assertEqual(merged["extra_split_coda_calls"], 0)
        possible = {key: [] for key in merged["full_attention_dense_envelope"]}
        for first in range(2, 10):  # every split of 11 into two native branch lengths >= 2
            orders = [account(row(n, 5, (0, 1, 1, 1))) for n in (first, 11 - first)]
            for key, value in merged["split_projection_flops_approx"].items():
                self.assertEqual(value, sum(o["split_projection_flops_approx"][key] for o in orders))
            for key in possible:
                possible[key].append(sum(o["full_attention_dense_envelope"][key][0] for o in orders))
        self.assertEqual(merged["full_attention_dense_envelope"],
                         {key: (min(values), max(values)) for key, values in possible.items()})

    def test_inconsistent_or_unsupported_inputs_fail_closed(self):
        valid = row(5, 5, (0, 1, 1, 1))
        for bad in (
            {**valid, "tokens": False}, {**valid, "layer_calls": 5.0}, {**valid, "layer_calls": 6},
            {**valid, "callback_counts": {"block_calls": 1}}, row(5, 5, (0, 1, -1, 1)),
            row(5, 10, (1, 1, 1, 1)), row(5, 6, (0, 1, 1, 2)),
            row(3, 10, (0, 2, 2, 2), forward_calls=2),
            row(8, 10, (0, 2, 2, 2), forward_calls=2, ensemble_calls=1),
            row(8, 16, (0, 2, 4, 2), forward_calls=2),  # divisible totals don't establish a schedule
            {**valid, "cuda_event_ms": float("nan")},
        ):
            with self.subTest(trace=bad), self.assertRaises(ValueError):
                account(bad)
        for bad_args in (
            {"window": (2, 2)}, {"window": (0, 6)}, {"window": (False, 4)},
            {"text": {**TEXT, "num_hidden_layers": 6}}, {"text": {**TEXT, "head_dim": 0}},
            {"text": {**TEXT, "attn_output_gate": False}},
            {"text": {**TEXT, "layer_types": [LINEAR] * 4 + ["unknown"]}},
        ):
            with self.subTest(args=bad_args), self.assertRaises(ValueError):
                account(valid, **bad_args)


if __name__ == "__main__":
    unittest.main()
