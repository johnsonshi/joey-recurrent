"""Offline execution of a verified physical Joey layer subset, never a stock-weight override."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import native as stock


def _kept_ids(values) -> tuple[int, ...]:
    if (not isinstance(values, (list, tuple)) or not 0 < len(values) < 24
            or any(type(i) is not int or not 0 <= i < 24 for i in values)
            or list(values) != sorted(set(values))):
        raise ValueError("expected a strictly increasing proper subset of original layer IDs 0..23")
    return tuple(values)


def _physical_window(window, depth):
    start, stop = stock._check_window(window)
    if stop > depth:
        raise ValueError(f"window exceeds compact physical depth {depth}; do not use source layer IDs")
    return start, stop


def _verified_export(model_dir, source_dir, expected_digest):
    from .compact import verify_compact
    root, source = Path(model_dir).resolve(strict=True), Path(source_dir).resolve(strict=True)
    if not root.is_dir() or not source.is_dir() or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("compact and pinned source must be distinct, disjoint staged directories")
    if not isinstance(expected_digest, str) or not re.fullmatch("[0-9a-f]{64}", expected_digest):
        raise ValueError("supply a trusted SHA-256 of compact.json from outside the export")
    stock._verify_hashes(root, {"compact.json": expected_digest})
    m = verify_compact(root)
    src = m["source"]
    if (src["kind"] != "pinned-joey-1" or src["model"] != "baselabs/joey-1"
            or src["revision"] != stock.MODEL_REVISION or src["depth"] != 24
            or src["weight_file_sha256"] != {"model.safetensors": stock.MODEL_SHA256["model.safetensors"]}):
        raise ValueError("compact native scoring requires pinned Joey-1 lineage, never a generated fixture")
    sidecars = m["sidecar_sha256"]
    required = {k: h for k, h in stock.MODEL_SHA256.items() if k not in ("model.safetensors", "config.json")}
    if not required.items() <= sidecars.items():
        raise ValueError("compact tokenizer, template, readout or generation config differs from pinned source")
    if {p.name for p in root.iterdir()} != {"compact.json", "config.json", "model.safetensors", *sidecars}:
        raise ValueError("unexpected loader inputs inside compact model directory")
    kept = _kept_ids(m["anchor_source_layers"])
    return root, m, kept


_LAYER_KEY = re.compile(r"model\.language_model\.layers\.(\d+)\.(.+)")
_PROJECTIONS = {"mlp.up_proj.weight", "mlp.gate_proj.weight", "mlp.down_proj.weight"}


def _tensor_plan(source_keys, kept, groups):
    """Map exported physical keys to anchors or independently weighted original MLP tensors."""
    physical = {original: i for i, original in enumerate(kept)}
    for key in source_keys:
        match = _LAYER_KEY.fullmatch(key)
        if not match:
            yield key, (key,), (), None
            continue
        original, suffix = int(match[1]), match[2]
        if original not in physical:
            continue
        index = physical[original]
        target = f"model.language_model.layers.{index}.{suffix}"
        group = groups[index // 4]
        if suffix in _PROJECTIONS and len(group["cycles"]) > 1:
            names = tuple(f"model.language_model.layers.{4*c + original%4}.{suffix}" for c in group["cycles"])
            yield target, names, tuple(group["weights"]), (original, suffix.split(".")[1])
        else:
            yield target, (key,), (), None


@contextmanager
def _independent_construction(source, source_dir, root, manifest, kept):
    """Verify every raw output byte; temporarily install independent MLPs on the private reference."""
    import torch
    from safetensors import safe_open
    with (source.inference_context(), ExitStack() as restore,
          safe_open(str(Path(source_dir) / "model.safetensors"), framework="pt", device="cpu") as raw,
          safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as exported):
        touched = set()
        for key, names, coefficients, replacement in _tensor_plan(raw.keys(), kept, manifest["plan"]["groups"]):
            output = exported.get_tensor(key)
            inputs = [raw.get_tensor(name) for name in names]
            if any(t.dtype != output.dtype or t.shape != output.shape for t in inputs):
                raise ValueError(f"compact tensor layout differs from pinned source: {key}")
            flat = output.reshape(-1)
            inputs = [t.reshape(-1) for t in inputs]
            if replacement:
                anchor, attr = replacement
                proj = getattr(source._engine.model.model.layers[anchor].mlp, attr)
                if proj.weight.shape != output.shape or proj.weight.dtype != output.dtype:
                    raise ValueError(f"source model projection differs from safetensors: {key}")
                merged = torch.empty_like(proj.weight)
            for i in range(0, flat.numel(), 1 << 20):
                sl = slice(i, i + (1 << 20))
                if coefficients:
                    # Independent CPU FP32 product and add in source order, then one BF16 rounding.
                    total = torch.zeros_like(inputs[0][sl], dtype=torch.float32)
                    for t, w in zip(inputs, coefficients, strict=True):
                        total.add_(t[sl].float().mul(w))
                    expected = total.to(output.dtype)
                else:
                    expected = inputs[0][sl]
                if not torch.equal(expected.view(torch.uint8), flat[sl].view(torch.uint8)):
                    raise ValueError(f"compact payload disagrees with pinned source construction: {key}")
                if replacement:
                    merged.view(-1)[sl].copy_(expected)
            if replacement:
                # Restore on every exit; the source checkpoint and original parameter are never written.
                restore.callback(setattr, proj, "weight", proj.weight)
                proj.weight = torch.nn.Parameter(merged, requires_grad=False)
            touched.add(key)
        if touched != set(exported.keys()):
            raise ValueError("compact checkpoint contains unexpected or missing parameters")
        yield


@dataclass(frozen=True)
class ReloadReport:
    manifest_sha256: str
    physical_to_source: tuple[int, ...]
    branches: int
    max_abs_logit: float
    max_abs_probability: float
    same_top1: bool

    @property
    def passed(self):
        return (self.branches > 0 and self.same_top1
                and max(self.max_abs_logit, self.max_abs_probability) <= 1e-5)


class ReloadParityError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"compact export disagrees with independent source-kept pass: {report}")


def _source_kept_logits(branch, kept):
    """Original modules, original layer IDs, no cache, no copied or reconstructed weights."""
    if branch._window != (0, 24):
        raise ValueError("source witness must start at the embedding and end at the original head")
    with branch.inference_context():
        h = branch._prelude()  # no decoder layers before window start 0
        for original_id in kept:
            h = branch._layers(h, original_id, original_id + 1)
        z = branch._coda(h)  # no decoder layers after window stop 24
        return stock._finite_vector(z, len(branch.labels))


def _source_witness(source_dir, device, root, manifest, kept, cases):
    source = stock.NativeJoey.load(source_dir, device=device, fork_mode="sequential")
    source.verify_parity(window=(0, 24))
    witness = []
    with _independent_construction(source, source_dir, root, manifest, kept):
        for state, questions in cases:
            branches = source._plan(source._request(state, questions), (0, 24))
            witness.append({qid: (b.layout, _source_kept_logits(b, kept)) for qid, b in branches.items()})
    return witness  # no tensors, modules or caches escape the full-model lifetime


class CompactJoey(stock.NativeJoey):
    """Physical decoder stack. Its native counterpart is this compact model at k=1, not 24-layer Joey."""

    @classmethod
    def load(cls, model_dir, *, source_model_dir, expected_manifest_sha256, device="cuda:0", extra_requests=()):
        """Verify source and export, then compare independent source-kept vs freshly reloaded compact k=1."""
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            raise RuntimeError("set HF_HUB_OFFLINE=1 before starting the process")
        root, manifest, kept = _verified_export(model_dir, source_model_dir, expected_manifest_sha256)
        extra = tuple(extra_requests)
        cases = tuple(stock._parity_requests()) + extra
        expected = _source_witness(source_model_dir, device, root, manifest, kept, cases)
        gc.collect()
        import torch
        from qev.engine import DecisionEngine
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            e = DecisionEngine(str(root), device=device, dtype="bfloat16", head_dtype="float32",
                               max_branch_tokens=stock.TRAINED_BRANCH_LIMIT, max_request_tokens=stock.REQUEST_LIMIT,
                               fork_mode="sequential")
        types = tuple((stock.LAYER_TYPES * 6)[i] for i in kept)
        if (type(e.model).__name__ != "Qwen3_5ForCausalLM" or e.model.config.model_type != "qwen3_5_text"
                or tuple(e.model.config.layer_types) != types or len(e.model.model.layers) != len(kept)
                or e.model.config.num_hidden_layers != len(kept) or e.model.config._attn_implementation != "sdpa"
                or e.model.training or not e.readout.config.noncausal or e.readout.config.readout != "slot_lm"
                or e.readout.head is not None or e.calibration.temperatures
                or e.model.get_output_embeddings().weight.dtype != torch.float32
                or e.model.model.embed_tokens.weight.dtype != torch.bfloat16
                or hashlib.sha256(json.dumps(list(e.slot_ids), separators=(",", ":")).encode()).hexdigest()
                != stock.SLOT_IDS_SHA256):
            raise RuntimeError("reloaded compact graph, precision, mask, slots or readout differ from manifest")
        e.model.requires_grad_(False)
        result = cls(e)
        result.physical_to_source = kept
        result.repeat_window = tuple(manifest["repeat_layers"])
        result._manifest = manifest
        result.reload_report = result._compare_reload(cases, expected, expected_manifest_sha256)
        if not result.reload_report.passed:
            raise ReloadParityError(result.reload_report)
        # The cached qev counterpart must also agree, not just two single-sequence paths.
        result.verify_parity(window=result.repeat_window, extra_requests=extra)
        return result

    def _compare_reload(self, cases, expected, digest):
        from qev.calibrate import softmax
        n, dz, dp, same = 0, 0., 0., True
        full = (0, len(self.physical_to_source))
        with self.inference_context():
            for (state, questions), ref in zip(cases, expected, strict=True):
                planned = self._plan(self._request(state, questions), full)
                if planned.keys() != ref.keys():
                    raise ValueError("reloaded question set differs from source-kept witness")
                for qid, b in planned.items():
                    layout, z0 = ref[qid]
                    if b.layout != layout:
                        raise ValueError("compact k=1 uses different prefix, suffix, readout or slot IDs")
                    for got in (b._full_logits(), b._one_pass()):
                        z, p, top = stock._comparison(z0, got, softmax, b.temperature)
                        dz, dp, same = max(dz, z), max(dp, p), same and top
                    n += 1
        return ReloadReport(digest, self.physical_to_source, n, dz, dp, same)

    def verify_parity(self, *, window, **kwargs):
        if not self.reload_report.passed:
            raise ReloadParityError(self.reload_report)
        return super().verify_parity(window=_physical_window(window, len(self.physical_to_source)), **kwargs)

    def branches(self, state, questions, *, window):
        return super().branches(state, questions, window=_physical_window(window, len(self.physical_to_source)))

    def counterpart(self, state, questions):
        """Native decisions from the exact physical export, k=1, with shared-prefix qev serving."""
        return self.stock(state, questions)

    def physical_ids(self, k: int) -> tuple[int, ...]:
        from .compact import layer_schedule
        return layer_schedule(self._manifest, k)

    def score_physical(self, state, questions, *, physical_ids) -> dict[str, dict]:
        ids = tuple(physical_ids)
        start, stop = self.repeat_window
        k, rem = divmod(len(ids) - (len(self.physical_to_source) - (stop - start)), stop - start)
        if (rem or not 1 <= k <= 32 or any(type(i) is not int for i in ids)
                or ids != self.physical_ids(k)):
            raise ValueError("expected this verified manifest's exact physical layer schedule")
        answers = {}
        for qid, b in self.branches(state, questions, window=self.repeat_window).items():
            h = b.prelude()
            for _ in range(k):
                h = b.block(h, b.fresh_state())
            answers[qid] = b.decode(b.coda(h, b.fresh_state()))
        return answers
