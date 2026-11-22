"""Inference-only, second-visit norm controls for an already reloaded select-20 CompactJoey."""

from __future__ import annotations

import copy
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from threading import Lock
from typing import Any, Literal

from . import pass_norms
from .loop import LoopCounts
from .native import MODEL_REVISION, MODEL_SHA256, NativeBranch, ParityReport, _check_window
from .native_compact import CompactJoey, ReloadReport, _kept_ids


_WINDOW = (8, 12)
_LEAVES = tuple(name.removesuffix(".weight") for name in pass_norms.NORMS)


@dataclass(frozen=True)
class OverlayResult:
    logits: Any
    counts: LoopCounts
    # Completed forwards on the isolated norm leaves, keyed as in the sidecar.
    norm_calls: dict[str, int]


def _base_binding(native, pin):
    if type(native) is not CompactJoey:
        raise TypeError("expected the already reloaded CompactJoey, not a module or adapter")
    report = getattr(native, "reload_report", None)
    if (type(report) is not ReloadReport or not report.passed or report.manifest_sha256 != pin
            or _kept_ids(report.physical_to_source) != pass_norms.KEPT):
        raise ValueError("missing, failed or mismatched compact reload witness")
    manifest = native._manifest
    source = manifest["source"]
    if (_kept_ids(native.physical_to_source) != pass_norms.KEPT
            or _kept_ids(manifest["anchor_source_layers"]) != pass_norms.KEPT
            or _check_window(native.repeat_window) != _WINDOW
            or _check_window(tuple(manifest["repeat_layers"])) != _WINDOW
            or manifest["physical_depth"] != len(pass_norms.KEPT) or manifest["plan"] != pass_norms.BASE_PLAN
            or source["kind"] != "pinned-joey-1" or source["model"] != "baselabs/joey-1"
            or source["depth"] != 24 or source["revision"] != MODEL_REVISION
            or source["weight_file_sha256"] != {"model.safetensors": MODEL_SHA256["model.safetensors"]}):
        raise ValueError("requires the verified earlier-cycle select-20 base and physical window 8:12")
    native._require_gate(_WINDOW)
    gate = native._gates[_WINDOW]
    if type(gate) is not ParityReport or gate.window != _WINDOW or not gate.passed:
        raise ValueError("missing successful native split gate for physical window 8:12")
    return {
        "compact_manifest_sha256": report.manifest_sha256,
        "weight_sha256": manifest["weights_sha256"],
        "weight_file_bytes": manifest["storage"]["weight_file_bytes"],
        "stored_elements": manifest["storage"]["full"]["stored_parameter_elements"],
    }


def _require_frozen(model):
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("the compact model must remain frozen and in eval mode")


def _shell(module):
    """Keep native methods and shared children/tensors, but not their registration tables."""
    view = copy.copy(module)
    for name, value in vars(module).items():
        if (name in ("_modules", "_parameters", "_buffers", "_non_persistent_buffers_set")
                or "_hooks" in name and isinstance(value, (dict, set))):
            vars(view)[name] = value.copy()
    return view


def _prepare(layers, vectors, kind):
    import torch

    if sys.byteorder != "little":
        raise ValueError("the BF16 sidecar requires a little-endian host")
    views, leaves = [], {}
    for index in range(*_WINDOW):
        original = layers[index]
        view = _shell(original)
        for name in _LEAVES:
            key = f"visit2.layers.{index}.{name}.weight"
            raw = vectors[key]
            norm = original._modules[name]
            weight = norm.weight
            if (norm._modules or norm._parameters.get("weight") is not weight
                    or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (pass_norms.WIDTH,)
                    or weight.is_meta or weight.layout != torch.strided or not weight.is_contiguous()
                    or len(raw) != 2 * pass_norms.WIDTH):
                raise ValueError(f"expected a native BF16 norm scale and matching vector: {key}")
            leaf = _shell(norm)
            # One owned allocation per leaf, even on CPU and even for identity. No FP conversion.
            data = torch.empty_like(weight)
            source = weight if kind == "identity" else torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
            data.copy_(source)
            leaf.weight = torch.nn.Parameter(data, requires_grad=False)
            setattr(view, name, leaf)
            leaves[key] = leaf
        views.append(view)
    if vectors.keys() != leaves.keys():
        raise ValueError("unexpected second-visit vectors")
    window = torch.nn.ModuleList(views)
    window.training = False  # Do not call eval() recursively through the shared children.
    return window, leaves


class NormOverlay:
    """Use load(); identity and donor use the same eight private parameters and split path."""

    def __init__(self):
        raise TypeError("use NormOverlay.load()")

    @classmethod
    def load(cls, compact_native: CompactJoey, bundle, *, manifest_sha256: str,
             compact_manifest_sha256: str, kind: Literal["identity", "donor"]) -> NormOverlay:
        """Bind the pinned sidecar to the existing reload witness and physical manifest.

        The caller must keep the verified model unchanged. This does not rehash live
        tensors, load a checkpoint, or establish parity for the overlay.
        """
        if kind not in ("identity", "donor"):
            raise ValueError("kind must be identity or donor")
        base = _base_binding(compact_native, compact_manifest_sha256)
        manifest, vectors = pass_norms.read_vectors(
            bundle, manifest_sha256=manifest_sha256, compact_manifest_sha256=compact_manifest_sha256
        )
        if manifest["base"] != base or tuple(manifest["physical_window"]) != compact_native.repeat_window:
            raise ValueError("sidecar does not bind this compact physical payload and window")
        engine = compact_native._engine
        model = engine.model
        decoder = model.model
        layers = tuple(decoder.layers)
        if len(layers) != len(pass_norms.KEPT) or decoder.config.hidden_size != pass_norms.WIDTH:
            raise ValueError("physical decoder does not match the select-20 norm overlay")
        _require_frozen(model)
        with compact_native.inference_context():
            views, leaves = _prepare(layers, vectors, kind)
        overlay = object.__new__(cls)
        overlay._native, overlay._engine = compact_native, engine
        overlay._model, overlay._decoder = model, decoder
        overlay._base, overlay._layers = base, layers
        overlay._views, overlay._leaves, overlay._lock = views, leaves, Lock()
        return overlay

    def _check_branch(self, branch):
        if (type(branch) is not NativeBranch or branch._native is not self._native
                or _check_window(branch._window) != _WINDOW):
            raise ValueError("expected a branch of this compact model at physical window 8:12")
        if (self._native._engine is not self._engine or self._engine.model is not self._model
                or self._model.model is not self._decoder
                or len(self._decoder.layers) != len(self._layers)
                or any(a is not b for a, b in zip(self._decoder.layers, self._layers))
                or _base_binding(self._native, self._base["compact_manifest_sha256"]) != self._base):
            raise ValueError("compact model or its reload binding changed")
        _require_frozen(self._model)

    def run(self, branch: NativeBranch, *, visits: int = 2) -> OverlayResult:
        """Always split. norm_calls contains all eight sidecar keys, including zeros at visits=1."""
        if type(visits) is not int or visits not in (1, 2):
            raise ValueError("visits must be the integer 1 or 2")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent use of a norm overlay observer")
        try:
            self._check_branch(branch)
            return self._split(branch, visits)
        finally:
            self._lock.release()

    def _split(self, branch, visits):
        # Never inherit or write a caller's cached stage table. Only module views (not
        # sequence state) are involved; NativeBranch still owns tokens, masks, RoPE and head.
        plain = copy.copy(branch)
        plain._stages = {}
        calls = dict.fromkeys(self._leaves, 0)
        observing = False

        def record(key, module, args, output):
            if not observing or module is not self._leaves[key]:
                raise RuntimeError(f"second-visit norm called outside its scope: {key}")
            calls[key] += 1

        with plain.inference_context(), ExitStack() as hooks:
            for key, leaf in self._leaves.items():
                hooks.callback(leaf.register_forward_hook(partial(record, key)).remove)
            stage = plain._stage(*_WINDOW)
            originals = self._layers[slice(*_WINDOW)]
            if (len(stage.layers) != len(originals)
                    or any(a is not b for a, b in zip(stage.layers, originals))):
                raise ValueError("partial or reordered native window stage")

            prelude_calls = block_calls = coda_calls = 0
            hidden = plain.prelude()
            prelude_calls += 1
            hidden = plain.block(hidden, plain.fresh_state())
            block_calls += 1
            if visits == 2:
                stage_view = _shell(stage)
                stage_view.layers = self._views
                second = copy.copy(plain)
                second._stages = {_WINDOW: stage_view}
                observing = True
                hidden = second.block(hidden, second.fresh_state())
                observing = False
                block_calls += 1
            if any(n != visits - 1 for n in calls.values()):
                raise RuntimeError(f"incomplete or duplicated second-visit norm calls: {calls}")
            logits = plain.coda(hidden, plain.fresh_state())
            coda_calls += 1
        return OverlayResult(logits, LoopCounts(0, prelude_calls, block_calls, coda_calls), calls)
