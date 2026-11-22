"""Invented in-memory reload evidence and decoder. No checkpoint, sidecar I/O or parity claim."""

import copy
import importlib.util
import struct
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from threading import Event
from types import SimpleNamespace as NS
from unittest.mock import patch

from joey_research import native, pass_norms
from joey_research.loop import LoopCounts
from joey_research.native_compact import CompactJoey, ReloadReport
from joey_research.norm_overlay import NormOverlay


WINDOW = (8, 12)
COMPACT_PIN, SIDE_PIN, WEIGHT_PIN = "a" * 64, "b" * 64, "c" * 64
NAMES = ("input_layernorm", "post_attention_layernorm")
KEYS = tuple(f"visit2.layers.{i}.{name}.weight" for i in range(8, 12) for name in NAMES)


def raw(value):
    word = struct.pack("<f", value)[2:]
    return word * (pass_norms.WIDTH - 1) + b"\x00\x80"  # include BF16 negative zero


class Tensor:
    dtype, device, layout, is_meta = "bf16", "cpu", "strided", False

    def __init__(self, data):
        self.data = bytearray(data)
        self.shape = (len(data) // 2,)
        self.requires_grad = False

    def is_contiguous(self):
        return True

    def copy_(self, source):
        self.data[:] = source.data

    def first(self):
        return struct.unpack("<f", b"\0\0" + self.data[:2])[0]


class Module:
    """Only the shallow-copy/registry/call semantics used by these fixtures."""

    def __init__(self):
        self._modules, self._parameters = {}, {}
        self._forward_hooks, self._forward_hooks_with_kwargs = {}, {}
        self.training = False

    def __getattr__(self, key):
        for table in ("_modules", "_parameters"):
            if key in vars(self).get(table, {}):
                return vars(self)[table][key]
        raise AttributeError(key)

    def __setattr__(self, key, value):
        if isinstance(value, Tensor):
            self._parameters[key] = value
        elif isinstance(value, Module):
            self._modules[key] = value
        else:
            object.__setattr__(self, key, value)

    def modules(self):
        yield self
        for child in self._modules.values():
            yield from child.modules()

    def parameters(self):
        return (p for m in self.modules() for p in m._parameters.values())

    def register_forward_hook(self, fn):
        self._forward_hooks[fn] = fn
        return NS(remove=lambda: self._forward_hooks.pop(fn))

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        for hook in self._forward_hooks.values():
            hook(self, args, out)
        return out


class ModuleList(Module):
    def __init__(self, modules):
        super().__init__()
        self._modules.update(enumerate(modules))

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, key):
        return self._modules[key]


class Scale(Module):
    def __init__(self, value):
        super().__init__()
        self.weight = Tensor(raw(value))

    def forward(self, h):
        return replace(h, value=h.value + 1 + self.weight.first())


class Identity(Module):
    def forward(self, h):
        return h


class Projection(Module):
    def __init__(self):
        super().__init__()
        self.weight = Tensor(b"\x80\x3f")  # a 1x1 BF16 identity matrix
        self.weight.shape = (1, 1)

    def forward(self, h):
        return replace(h, value=h.value * self.weight.first())


class Layer(Module):
    def __init__(self, i, trace):
        super().__init__()
        self.i, self.trace = i, trace
        self.input_layernorm, self.post_attention_layernorm = Scale(2*i), Scale(2*i + 1)
        self.attention, self.mlp = Projection(), Projection()
        self.post_calls = 1

    def forward(self, h):
        self.trace.layers.append(self)
        h = self.attention(self.input_layernorm(h))
        for _ in range(self.post_calls):
            h = self.post_attention_layernorm(h)
        return self.mlp(h)


@dataclass(frozen=True)
class Hidden:
    value: float
    length: int
    dtype, device = Tensor.dtype, Tensor.device

    @property
    def shape(self):
        return (1, self.length, pass_norms.WIDTH)

    def __getitem__(self, key):
        assert key == (slice(None), slice(-1, None), slice(None))
        return replace(self, length=1)


class Decoder(Module):
    def __init__(self, trace):
        super().__init__()
        self.trace = trace
        self.layers = ModuleList(Layer(i, trace) for i in range(20))
        self.config = NS(layer_types=native.LAYER_TYPES * 5, hidden_size=pass_norms.WIDTH, num_hidden_layers=20)
        self.norm = Scale(100)
        self.embed_tokens = Module()
        self.embed_tokens.weight = Tensor(raw(0))
        self.embed_tokens.forward = lambda ids: Hidden(sum(ids), len(ids))

    def forward(self, *, inputs_embeds, attention_mask, past_key_values, use_cache, return_dict):
        h = inputs_embeds
        assert (past_key_values, use_cache, return_dict) == (None, False, True)
        assert attention_mask == (1,) * h.length
        assert self.trace.scope == ([self.trace.prefix], [h.length], h.length)
        assert self.config.num_hidden_layers == len(self.layers)
        assert self.config.layer_types == [native.LAYER_TYPES[layer.i % 4] for layer in self.layers]
        for layer in self.layers:
            h = layer(h)
        return NS(last_hidden_state=self.norm(h), past_key_values=None)


class Scores:
    def __init__(self, h):
        assert h.length == 1
        self.value = h.value

    def __getitem__(self, key):
        assert key[:2] == (0, -1)
        return NS(float=lambda: tuple(self.value + i for i in key[2]))


def evidence(engine):
    owner = CompactJoey(engine)
    owner.repeat_window, owner.physical_to_source = WINDOW, pass_norms.KEPT
    owner.reload_report = ReloadReport(COMPACT_PIN, pass_norms.KEPT, 1, 0., 0., True)
    owner._gates[WINDOW] = native.ParityReport(WINDOW, 1, ("sequential",), 0., 0., 0., 0., True, .05, .005, 1e-5)
    owner._manifest = {
        "physical_depth": 20, "plan": copy.deepcopy(pass_norms.BASE_PLAN),
        "repeat_layers": list(WINDOW), "anchor_source_layers": list(pass_norms.KEPT),
        "source": {"kind": "pinned-joey-1", "model": "baselabs/joey-1", "revision": native.MODEL_REVISION,
                   "depth": 24, "weight_file_sha256": {"model.safetensors": native.MODEL_SHA256["model.safetensors"]}},
        "weights_sha256": WEIGHT_PIN, "storage": {"full": {"stored_parameter_elements": 40000},
                                                 "weight_file_bytes": 81234},
    }
    return owner


class OverlayFixtures(unittest.TestCase):
    def setUp(self):
        def parameter(data, *, requires_grad):
            self.assertIs(requires_grad, False)
            return data

        torch = NS(bfloat16=Tensor.dtype, strided=Tensor.layout, empty_like=lambda w: Tensor(bytes(len(w.data))),
                   frombuffer=lambda buf, *, dtype: Tensor(buf),
                   nn=NS(ModuleList=ModuleList, Identity=Identity, Parameter=parameter))
        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(patch.dict(sys.modules, torch=torch))
        self.trace = NS(layers=[], states=[], scope=None, prefix=2)
        model = Module()
        model.model = Decoder(self.trace)
        model.get_output_embeddings = lambda: Scores

        @contextmanager
        def blocks(*scope):
            self.assertIsNone(self.trace.scope)
            self.trace.scope = scope
            try:
                yield
            finally:
                self.trace.scope = None

        self.owner = evidence(NS(model=model, _blocks=blocks))
        self.owner.inference_context = nullcontext
        layout = native.BranchLayout("toy", "choice", ("x", "y"), (5, 2), 2, (9, 7, 3, 1), ((2, 3), (3, 4)))
        self.branch = native.NativeBranch(self.owner, None, None, layout, WINDOW)
        self.original = tuple(model.model.layers)
        patches.enter_context(patch.object(native.NativeBranch, "_tokens",
                                          lambda b: (b.layout.input_ids, (1,) * len(b.layout.input_ids))))
        consume = native._FreshSequence.consume

        def observe_state(state, owner):
            consume(state, owner)
            self.trace.states.append(state)

        patches.enter_context(patch.object(native._FreshSequence, "consume", observe_state))
        self.bundle = object()
        self.side = {"base": {"compact_manifest_sha256": COMPACT_PIN, "weight_sha256": WEIGHT_PIN,
                              "stored_elements": 40000, "weight_file_bytes": 81234}, "physical_window": [8, 12]}
        self.raw = {k: raw(i + 24) for i, k in enumerate(KEYS)}  # four distinct source-12:16 pairs
        self.reader = patches.enter_context(patch.object(pass_norms, "read_vectors", return_value=(self.side, self.raw)))

    def load(self, kind="donor", owner=None):
        return NormOverlay.load(self.owner if owner is None else owner, self.bundle, kind=kind,
                                manifest_sha256=SIDE_PIN, compact_manifest_sha256=COMPACT_PIN)

    def snapshot(self):
        return [(m, m._modules.copy(), m._parameters.copy(), m._forward_hooks.copy(),
                 [(p, bytes(p.data), p.requires_grad) for p in m._parameters.values()])
                for m in self.owner._engine.model.modules()]

    def test_first_unmodified_second_only_scales_shared_across_rows(self):
        before = self.snapshot()
        for kind in ("identity", "donor"):
            overlay = self.load(kind)
            self.reader.assert_called_with(self.bundle, manifest_sha256=SIDE_PIN, compact_manifest_sha256=COMPACT_PIN)
            previous = None
            for extra in ((), (4, 2)):
                layout = replace(self.branch.layout, input_ids=self.branch.layout.input_ids + extra)
                b = native.NativeBranch(self.owner, None, None, layout, WINDOW)
                expected = b.one_pass()
                h = b.prelude()
                for _ in range(2):
                    h = b.block(h, b.fresh_state())
                base_twice = b.coda(h, b.fresh_state())

                self.trace.layers.clear()
                self.trace.states.clear()
                once = overlay.run(b, visits=1)
                self.assertEqual((once.logits, once.norm_calls, once.counts),
                                 (expected, dict.fromkeys(KEYS, 0), LoopCounts(0, 1, 1, 1)))
                self.assertEqual(tuple(self.trace.layers), self.original)
                self.assertEqual(len(self.trace.states), 2)

                self.trace.layers.clear()
                self.trace.states.clear()
                twice = overlay.run(b)
                delta = 8 * 8 if kind == "donor" else 0
                self.assertEqual((twice.logits, twice.norm_calls, twice.counts),
                                 (tuple(z + delta for z in base_twice), dict.fromkeys(KEYS, 1), LoopCounts(0, 1, 2, 1)))
                self.assertEqual(len({id(s) for s in self.trace.states}), 3)
                invoked = self.trace.layers
                self.assertEqual(tuple(invoked[:12] + invoked[16:]), self.original)
                scales = []
                self.assertEqual(len(invoked), 24)
                for i, (base, view) in enumerate(zip(self.original[8:12], invoked[12:16])):
                    self.assertIsNot(base, view)
                    self.assertIsNot(base._modules, view._modules)
                    self.assertIsNot(base._parameters, view._parameters)
                    for shared in ("mlp", "attention"):
                        self.assertIs(getattr(base, shared), getattr(view, shared))
                        self.assertIs(getattr(base, shared).weight, getattr(view, shared).weight)
                    for j, name in enumerate(NAMES):
                        old, new = getattr(base, name), getattr(view, name)
                        self.assertIsNot(old, new)
                        for table in ("_modules", "_parameters", "_forward_hooks", "_forward_hooks_with_kwargs"):
                            self.assertIsNot(getattr(old, table), getattr(new, table))
                        self.assertIsNot(old.weight, new.weight)
                        self.assertEqual(new.weight.data, old.weight.data if kind == "identity" else self.raw[KEYS[2*i+j]])
                        scales.append(new.weight)
                self.assertEqual(len(set(scales)), 8)
                if previous is not None:
                    self.assertEqual(scales, previous)  # same eight device objects for the next row
                previous = scales
                self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.reader.call_count, 2)  # once per overlay, not per row

    def test_partial_or_duplicate_visit_and_exceptions_do_not_leak(self):
        before = self.snapshot()
        overlay = self.load()
        layer = overlay._views[2]
        for repeats in (0, 2):
            with patch.object(layer, "post_calls", repeats), self.assertRaisesRegex(RuntimeError, "norm calls"):
                overlay.run(self.branch)
        with (patch.object(layer, "forward", side_effect=ArithmeticError("injected")),
              self.assertRaisesRegex(ArithmeticError, "injected")):
            overlay.run(self.branch)
        self.assertTrue(all(not leaf._forward_hooks for leaf in overlay._leaves.values()))
        self.assertIsNone(self.trace.scope)
        self.assertEqual(overlay.run(self.branch).norm_calls, dict.fromkeys(KEYS, 1))
        self.assertEqual(self.branch._stages, {})
        self.assertEqual(self.snapshot(), before)

    def test_partial_stage_and_native_cache_or_hidden_layout_errors_propagate(self):
        overlay = self.load()
        stage = native.NativeBranch._stage

        def truncated(b, start, stop):
            view = stage(b, start, stop)
            if (start, stop) == WINDOW:
                view.layers = ModuleList(list(view.layers)[:-1])
            return view

        with patch.object(native.NativeBranch, "_stage", truncated), self.assertRaisesRegex(ValueError, "partial"):
            overlay.run(self.branch, visits=1)
        forward = Decoder.forward
        for invalid in ("cache", "layout"):
            def broken(decoder, **kwargs):
                out = forward(decoder, **kwargs)
                if decoder.layers is overlay._views:
                    if invalid == "cache":
                        out.past_key_values = object()
                    else:
                        out.last_hidden_state = replace(out.last_hidden_state, length=1)
                return out

            with patch.object(Decoder, "forward", broken), self.assertRaisesRegex(RuntimeError, invalid):
                overlay.run(self.branch)
        self.assertEqual(overlay.run(self.branch).norm_calls, dict.fromkeys(KEYS, 1))

    def test_scope_pins_witness_and_physical_payload_are_independent(self):
        with self.assertRaises(TypeError):
            self.load(owner=self.owner._engine.model)
        with self.assertRaises(ValueError):
            self.load(kind="first")
        r = self.owner.reload_report
        for report in (None, NS(passed=True), replace(r, same_top1=False), replace(r, manifest_sha256="d"*64),
                       replace(r, physical_to_source=tuple(range(20)))):
            with patch.object(self.owner, "reload_report", report), self.assertRaises(ValueError):
                self.load()
        for key, val in (("physical_to_source", tuple(range(20))), ("repeat_window", (12, 16))):
            with patch.object(self.owner, key, val), self.assertRaises(ValueError):
                self.load()
        for key, val in (("plan", {}), ("anchor_source_layers", list(range(20))),
                         ("physical_depth", 24), ("repeat_layers", [7, 11])):
            with patch.dict(self.owner._manifest, {key: val}), self.assertRaises(ValueError):
                self.load()
        self.reader.assert_not_called()
        for key in self.side["base"]:
            with patch.dict(self.side["base"], {key: "other"}), self.assertRaisesRegex(ValueError, "payload"):
                self.load("identity")
        with patch.dict(self.side, physical_window=[9, 13]), self.assertRaisesRegex(ValueError, "window"):
            self.load()
        with patch.dict(self.owner._gates, clear=True), self.assertRaises(RuntimeError):
            self.load()
        failed = replace(self.owner._gates[WINDOW], same_top1=False)
        with patch.dict(self.owner._gates, {WINDOW: failed}), self.assertRaises(ValueError):
            self.load()
        with patch.object(self.reader, "side_effect", ValueError("bad sidecar pin")), self.assertRaises(ValueError):
            self.load("identity")

    def test_bad_visits_branch_model_and_training_refused_before_execution(self):
        overlay = self.load()
        for k in (0, 3, True, False, 1., "2"):
            with self.subTest(k=k), self.assertRaises(ValueError):
                overlay.run(self.branch, visits=k)
        for owner, window in ((evidence(self.owner._engine), WINDOW), (self.owner, (8, 11)),
                              (self.owner, (8., 12))):
            with self.assertRaises(ValueError):
                overlay.run(native.NativeBranch(owner, None, None, self.branch.layout, window))
        with self.assertRaises(ValueError):
            overlay.run(NS(_native=self.owner, _window=WINDOW))
        with patch.object(self.owner._engine, "model", copy.copy(self.owner._engine.model)), self.assertRaises(ValueError):
            overlay.run(self.branch)
        with patch.object(self.original[1].mlp.weight, "requires_grad", True), self.assertRaises(ValueError):
            overlay.run(self.branch)
        with patch.object(self.original[8], "training", True), self.assertRaises(ValueError):
            self.load()
        self.assertEqual(self.trace.layers, [])

    def test_overlapping_observation_rejected_without_disturbing_first(self):
        overlay = self.load()
        entered, release = Event(), Event()
        layer = overlay._views[0]
        forward = layer.forward

        def paused(h):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("fixture did not release the first visit")
            return forward(h)

        with patch.object(layer, "forward", paused), ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(overlay.run, self.branch)
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(RuntimeError, "concurrent"):
                    overlay.run(self.branch, visits=1)
            finally:
                release.set()
            self.assertEqual(first.result(timeout=5).norm_calls, dict.fromkeys(KEYS, 1))


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional real torch CPU toy; torch not installed")
class TorchStorageToy(unittest.TestCase):
    def test_private_bf16_storage_real_module_registries_and_unmodified_forward(self):
        import torch
        from joey_research.norm_overlay import _prepare

        class Norm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(pass_norms.WIDTH, dtype=torch.bfloat16, device="cpu"),
                                                 requires_grad=False)

            def forward(self, x):
                return (self.weight + 1) * x

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm, self.post_attention_layernorm = Norm(), Norm()
                self.mlp = torch.nn.Linear(2, 2, bias=False, device="cpu").requires_grad_(False)

            def forward(self, x):
                return self.post_attention_layernorm(self.input_layernorm(x))

        base = [Layer().eval() for _ in range(20)]
        data = dict.fromkeys(KEYS, raw(2))
        with torch.inference_mode():
            x = torch.ones_like(base[8].input_layernorm.weight)
            for kind in ("identity", "donor"):
                views, leaves = _prepare(base, data, kind)
                self.assertEqual(len({m.weight.data_ptr() for m in leaves.values()}), 8)
                for i, layer in enumerate(views, 8):
                    self.assertIs(layer.mlp.weight, base[i].mlp.weight)
                    for name in NAMES:
                        p, q = getattr(layer, name).weight, getattr(base[i], name).weight
                        self.assertNotEqual(p.data_ptr(), q.data_ptr())
                        self.assertEqual((p.device, p.dtype, p.stride(), p.requires_grad),
                                         (q.device, q.dtype, q.stride(), False))
                        if kind == "donor":
                            bits = torch.frombuffer(bytearray(raw(2)), dtype=torch.uint8)
                            self.assertTrue(torch.equal(p.view(torch.uint8), bits))
                    self.assertTrue(torch.equal(base[i](x), x))
                    self.assertEqual(layer(x)[0].item(), 1 if kind == "identity" else 9)


if __name__ == "__main__":
    unittest.main()
