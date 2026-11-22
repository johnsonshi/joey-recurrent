"""Contract fixtures use no torch. Only StagedModelIntegration exercises model parity."""

import hashlib
import math
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from joey_research import native as n


def fixture_softmax(logits, temperature):
    top = max(logits)
    exps = [math.exp((x - top) / temperature) for x in logits]
    return [x / sum(exps) for x in exps]


class NativeContracts(unittest.TestCase):
    def test_full_hidden_seam_visits_actual_window_in_order_without_mutating_backbone(self):
        # Tiny scalar-valued decoder protocol, *not* an implementation of Qwen or a parity claim.
        calls, masks, norms = [], [], []

        class Hidden:
            dtype, device = "bf16", "cpu"

            def __init__(self, value, length=5):
                self.value, self.shape = value, (1, length, 1)

            def __getitem__(self, key):
                self_test.assertEqual(key, (slice(None), slice(-1, None), slice(None)))
                return Hidden(self.value, 1)

        class Embedding:
            weight = Hidden(0)

            def __call__(self, ids):
                return Hidden(sum(ids))

        def norm(h):
            norms.append(h.value)
            return Hidden(h.value + 100)

        class Decoder:
            def __init__(self):
                self._modules = {"layers": [0, 1, 2, 3], "norm": norm}
                self.config = SimpleNamespace(hidden_size=1, num_hidden_layers=4, layer_types=["L", "L", "F", "L"])
                self.embed_tokens = Embedding()

            layers = property(lambda d: d._modules["layers"], lambda d, v: d._modules.update(layers=v))
            norm = property(lambda d: d._modules["norm"], lambda d, v: d._modules.update(norm=v))

            def __call__(self, *, inputs_embeds, attention_mask, past_key_values, use_cache, return_dict):
                self_test.assertEqual(masks[-1], ([2], [5], 5))
                self_test.assertEqual((attention_mask, past_key_values, use_cache, return_dict), ((1,) * 5, None, False, True))
                self_test.assertEqual(self.config.num_hidden_layers, len(self.layers))
                h = inputs_embeds
                for i, kind in zip(self.layers, self.config.layer_types, strict=True):
                    calls.append((i, kind))
                    h = Hidden(h.value * 2 + i + 1)
                return SimpleNamespace(last_hidden_state=self.norm(h), past_key_values=None)

        class Head:
            def __call__(self, h):
                self_test.assertEqual(h.shape, (1, 1, 1))
                self.value = h.value
                return self

            def __getitem__(self, key):
                self_test.assertEqual(key[:2], (0, -1))
                return SimpleNamespace(float=lambda: [self.value + i for i in key[2]])

        @contextmanager
        def blocks(prefixes, lengths, width):
            masks.append((prefixes, lengths, width))
            try:
                yield
            finally:
                masks.pop()

        self_test = self
        decoder = Decoder()
        engine = SimpleNamespace(model=SimpleNamespace(model=decoder, get_output_embeddings=Head), _blocks=blocks)
        owner = n.NativeJoey(engine)
        owner._gates[(1, 3)] = object()
        q = n.BranchLayout("q", "choice", ("z", "a"), (2, 0), 2, (1, 2, 3, 4, 5), ((2, 3), (3, 4)))
        branch = n.NativeBranch(owner, None, None, q, (1, 3))
        fake_torch = SimpleNamespace(nn=SimpleNamespace(ModuleList=list, Identity=lambda: (lambda h: h)))
        with (patch.dict(sys.modules, torch=fake_torch), patch.object(owner, "inference_context", nullcontext),
              patch.object(branch, "_tokens", return_value=(q.input_ids, (1,) * 5))):
            self.assertEqual(branch.one_pass(), [368, 366])
            self.assertEqual(calls, [(0, "L"), (1, "L"), (2, "F"), (3, "L")])
            calls.clear()
            h = branch.prelude()
            for _ in range(2):
                h = branch.block(h, branch.fresh_state())
            self.assertEqual(branch.coda(h, branch.fresh_state()), [1168, 1166])
            self.assertEqual(calls, [(0, "L"), (1, "L"), (2, "F"), (1, "L"), (2, "F"), (3, "L")])
        self.assertEqual((norms, masks), ([266, 1066], []))
        self.assertEqual((decoder.layers, decoder.norm, decoder.config.num_hidden_layers), ([0, 1, 2, 3], norm, 4))

    def test_local_only_and_integrity(self):
        with self.assertRaisesRegex(ValueError, "local model directory"):
            n._checkpoint_dir("https://invalid.example/not-a-staged-path")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hashes = {"config.json": hashlib.sha256(b"reviewed").hexdigest()}
            with patch.object(n, "MODEL_SHA256", hashes):
                with self.assertRaisesRegex(ValueError, "missing staged"):
                    n._checkpoint_dir(root)
                (root / "config.json").write_bytes(b"reviewed")
                self.assertEqual(n._checkpoint_dir(root), root.resolve())
                (root / "config.json").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    n._checkpoint_dir(root)
                (root / "adapter_config.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "unexpected files"):
                    n._checkpoint_dir(root)

    def test_layout_keeps_native_order_and_absolute_offsets(self):
        r = SimpleNamespace(qid="q2", kind="choice", labels=["z", "a"], n_slots=2)
        layout = n._layout(r, [11, 22, 33], [44, 55, 66, 77, 88], [(0, 2), (2, 4)], [32, 33, 34])
        self.assertEqual((layout.input_ids, layout.answer_position), ((11, 22, 33, 44, 55, 66, 77, 88), 7))
        self.assertEqual((layout.slot_ids, layout.labels, layout.option_spans),
                         ((32, 33), ("z", "a"), ((3, 5), (5, 7))))
        with self.assertRaises(ValueError):
            n._layout(r, [11], [22], [(-1, 1), (1, 2)], [32, 33])

    def test_bad_windows_and_gate_do_not_load_a_model(self):
        owner = n.NativeJoey(None)
        for window in ((2, 2), (-1, 2), (0, 25), (True, 2), (1, 2, 3)):
            with self.subTest(window=window), self.assertRaises(ValueError):
                owner.branches("", {}, window=window)
        with self.assertRaisesRegex(RuntimeError, "verify_parity"):
            owner.branches("", {}, window=(8, 12))

    def test_native_indexing_fallback_is_narrow_and_restores_original_mode(self):
        engine = SimpleNamespace(info=SimpleNamespace(fork_mode="batched"))
        owner = n.NativeJoey(engine)
        modes = []

        def decide(*args, **kwargs):
            modes.append(engine.info.fork_mode)
            if modes[-1] == "batched":
                raise RuntimeError("32BitIndexMath")
            return "sequential", kwargs

        engine.decide = decide
        with patch.object(owner, "inference_context", nullcontext):
            self.assertEqual(owner._reference("request", debug=True),
                             ("sequential", {"debias": 1, "debug": True}))
            self.assertEqual(modes, ["batched", "sequential"])
            self.assertEqual(engine.info.fork_mode, "batched")
            engine.decide = Mock(side_effect=RuntimeError("invalid mask"))
            with self.assertRaisesRegex(RuntimeError, "invalid mask"):
                owner._reference("request", debug=False)
            engine.decide.assert_called_once()

    def test_tie_probability_and_nonfinite_fail_closed(self):
        _, p, top = n._comparison([2, 2], [2, 2 + 1e-9], fixture_softmax, 1)
        self.assertFalse(top)
        self.assertLess(p, 1e-8)  # tiny differences are not a reason to omit the row
        for values in ((math.nan, 1), (math.inf, 1), ((0, 1), (1, 2))):
            with self.assertRaises(ValueError):
                n._comparison([0, 1], values, fixture_softmax, 1)

    def test_parity_compares_the_real_split_route_not_only_stock_and_full(self):
        # Control-flow fixture: the broken split must revoke the gate even when stock == full.
        owner = n.NativeJoey(None)
        window = (4, 8)
        owner._gates[window] = object()
        branch = SimpleNamespace(_official_logits=lambda meta: [1., 2.], _full_logits=lambda: [1., 2.],
                                 _one_pass=Mock(return_value=[1., 2.002]), temperature=1.,
                                 decode=lambda z: {"type": "noul", "noul": 0.268941})
        ref = SimpleNamespace(answers={"q": SimpleNamespace(type="noul", model_dump=lambda **kw: branch.decode(None))})
        mods = {"qev.calibrate": SimpleNamespace(softmax=fixture_softmax)}
        with (patch.dict(sys.modules, mods), patch.object(owner, "inference_context", nullcontext),
              patch.object(owner, "_request", return_value=object()),
              patch.object(owner, "_reference", return_value=(ref, {"fork_mode": "sequential"})),
              patch.object(owner, "_plan", return_value={"q": branch})):
            with self.assertRaises(n.ParityError) as exc:
                owner.verify_parity(window=window)
        self.assertFalse(exc.exception.report.passed)
        self.assertGreater(exc.exception.report.split_max_abs_logit, exc.exception.report.split_atol)
        self.assertTrue(branch._one_pass.called)
        with self.assertRaisesRegex(RuntimeError, "verify_parity"):
            owner._require_gate(window)

    def test_state_is_branch_specific_and_single_use(self):
        a, b = object(), object()
        s = n._FreshSequence(a)
        with self.assertRaises(ValueError):
            s.consume(b)
        s.consume(a)
        with self.assertRaises(ValueError):
            s.consume(a)

    def test_rejects_noncanonical_diagnostics_instead_of_treating_chunks_as_slots(self):
        layout = n.BranchLayout("q", "choice", ("z", "a"), (32, 33), 3, (1, 2, 3, 4, 5), ((3, 4), (4, 5)))
        b = n.NativeBranch(None, None, None, layout, (8, 12))
        meta = {"prefix_tokens": 3, "diagnostics": {"q": {"slot_logits": [[7, 3]]}}}
        self.assertEqual(b._official_logits(meta), [7, 3])
        meta["diagnostics"]["q"]["chunks"] = 1
        with self.assertRaisesRegex(ValueError, "chunking"):
            b._official_logits(meta)
        meta["diagnostics"]["q"] = {"slot_logits": [[7, 3], [3, 7]]}
        with self.assertRaisesRegex(ValueError, "order-averaging"):
            b._official_logits(meta)
        meta["diagnostics"]["q"] = {"slot_logits": [[7, 3]]}
        meta["prefix_tokens"] = 2
        with self.assertRaisesRegex(ValueError, "different native prefix"):
            b._official_logits(meta)


@unittest.skipUnless(os.environ.get("JOEY_TEST_MODEL_DIR"), "real, verified local checkpoint not requested")
class StagedModelIntegration(unittest.TestCase):
    def test_stock_split_and_two_actual_block_calls(self):
        import torch
        owner = n.NativeJoey.load(os.environ["JOEY_TEST_MODEL_DIR"],
                                  device=os.environ.get("JOEY_TEST_DEVICE", "cuda:0"))
        window = (8, 12)
        report = owner.verify_parity(window=window)
        self.assertTrue(report.passed)
        self.assertGreaterEqual(report.branch_count, 5)
        state, questions = n._parity_requests()[0]
        branch = owner.branches(state, questions, window=window)["team"]
        self.assertEqual(branch.decode(branch.stock()), owner.stock(state, questions)["team"])
        with torch.enable_grad():  # the adapter enforces inference mode even for such a caller
            h = branch.prelude()
            self.assertTrue(h.is_inference())
            once = branch.block(h, branch.fresh_state())
            z1 = branch.coda(once, branch.fresh_state())
            torch.testing.assert_close(z1, branch.one_pass(), atol=1e-5, rtol=0)
            twice = branch.block(once, branch.fresh_state())
            z2 = branch.coda(twice, branch.fresh_state())
            self.assertTrue(z2.is_inference())
            self.assertEqual(tuple(z2.shape), (3,))
            self.assertTrue(torch.isfinite(z2).all().item())


if __name__ == "__main__":
    unittest.main()
