"""Offline, text-only Joey decisions. Importing this module needs only Python."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_REVISION = "db5c2a631f2da93c4173378438d8246966e25159"
ONEJEV_REVISION = "81ce62f1597c91e46767d4d02d6ac2e18534fe94"
TRANSFORMERS_VERSION = "5.19.0"
TRAINED_BRANCH_LIMIT = 16_384
REQUEST_LIMIT = 65_536

# Model LFS OIDs are SHA-256 of the payload, not of the pointer or Xet object.
MODEL_SHA256 = {
    "model.safetensors": "4c503aded23bae311dff80a63326503fb720572c178f07a2142332d55f5f3afc",
    "config.json": "7c7ebc3322308391787aac279c7cbd4c3a38bba85e270b5f5568ad07467673e1",
    "readout.json": "96ea8a5c6ecfc30c20cdd369762db5997c5f4fb9821ce0a0318941343eb5e8ba",
    "tokenizer.json": "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f",
    "tokenizer_config.json": "2984c03425a28d242744e234a5003786fd289b3713db79fe0958c65eb59d85b2",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "generation_config.json": "290866a6d7d4a4442fe83ba2434a31517446f40a01de500ff97b4354e7e5d082",
}
# Import closure of qev.engine after applying the checkpoint's serve.patch.
QEV_SHA256 = {
    "__init__.py": "6446cd0006940630b28180d275efe39972b871c2fb2ae502c848607c940e8376",
    "client.py": "560ca00f7ecffee72e6490afdd69a34cf4f55a6e1765a0493d4beebf8b6b4278",
    "answers.py": "09b06dba258c80a6a9afdaa5aacfd0f7523f9d2a087a7b6402ca03553ea6fba6",
    "calibrate.py": "3fa6013a81c7e79a702afcdcc0ebd089bee7047590c18cc44708850f5aeda1ee",
    "engine.py": "dc3eb95e0118cb056357bd3e68c0ed9f18b4656492b02087e9ca53773af8d3f0",
    "prompt.py": "f0763b5f747039c2919bc2a6d51e4ed88308fe3497e53498e6f298286b6169b2",
    "readout.py": "e084edead750ddeb39384b9f659182df52720d96294d9a95c9259a8001e3c20d",
    "schema.py": "bce4b7d353ac115a9e50aef284975ddeed9a3bb452e3fd928a556668705bc2f3",
}
SLOT_IDS_SHA256 = "35c8f2e0a581bd39341f0e58a6d0aee317e27436daf052d614d777b57b4e374e"
LAYER_TYPES = ("linear_attention",) * 3 + ("full_attention",)
ANSWER_FIELDS = {
    "noul": {"type", "noul"},
    "choice": {"type", "choice", "probabilities"},
    "score": {"type", "score", "probabilities"},
}
Window = tuple[int, int]


def _verify_hashes(root: Path, hashes: Mapping[str, str]) -> None:
    for name, expected in hashes.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing staged file: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")


def _checkpoint_dir(model_dir: str | Path) -> Path:
    path = Path(model_dir)
    if not path.is_dir():
        raise ValueError("a staged local model directory is required, not a Hub ID or URL")
    path = path.resolve(strict=True)
    # Reject e.g. an adapter pointer, calibration override, competing tokenizer template or weights.
    allowed = MODEL_SHA256.keys() | {
        "README.md", "LICENSE", "NOTICE", "assets", "constraints.txt", "engine.py",
        "processor_config.json", "serve.patch", "setup.sh",
    }
    extra = sorted(p.name for p in path.iterdir() if not p.name.startswith(".") and p.name not in allowed)
    if extra:
        raise ValueError(f"unexpected files in staged checkpoint: {extra}")
    _verify_hashes(path, MODEL_SHA256)
    return path


def _check_window(window: Window) -> Window:
    if (not isinstance(window, tuple) or len(window) != 2 or any(type(i) is not int for i in window)
            or not 0 <= window[0] < window[1] <= 24):
        raise ValueError("window must be an integer pair (start, stop), 0 <= start < stop <= 24")
    return window


def _public(answer) -> dict:
    return answer.model_dump(include=ANSWER_FIELDS[answer.type])


@dataclass(frozen=True)
class BranchLayout:
    qid: str
    kind: str
    labels: tuple[str, ...]
    slot_ids: tuple[int, ...]
    prefix_tokens: int
    input_ids: tuple[int, ...]
    option_spans: tuple[tuple[int, int], ...]  # absolute, half-open token intervals

    @property
    def answer_position(self) -> int:
        return len(self.input_ids) - 1


def _layout(rendered, prefix, suffix, relative_spans, slots) -> BranchLayout:
    p, n, k = len(prefix), len(prefix) + len(suffix), rendered.n_slots
    if not (p and p < n and 1 <= k <= len(slots)) or len(relative_spans) != k:
        raise ValueError("unsupported or empty native branch layout")
    spans = tuple((p + start, p + stop) for start, stop in relative_spans)
    if any(not p <= a < b <= n for a, b in spans):
        raise ValueError("invalid option offset in native suffix")
    return BranchLayout(rendered.qid, rendered.kind, tuple(rendered.labels), tuple(slots[:k]),
                        p, tuple(prefix) + tuple(suffix), spans)


def _finite_vector(values: Any, k: int) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().tolist()
    if not isinstance(values, (list, tuple)) or len(values) != k or k < 1:
        raise ValueError("expected one flat logit per native option")
    if any(not isinstance(x, (float, int)) or not math.isfinite(x) for x in values):
        raise ValueError("non-finite or invalid native option logit")
    return [float(x) for x in values]


def _comparison(ref: Sequence[float], got: Sequence[float], softmax, temp: float):
    a, b = _finite_vector(ref, len(ref)), _finite_vector(got, len(ref))
    pa, pb = softmax(a, temp), softmax(b, temp)
    return (max(abs(x - y) for x, y in zip(a, b)),
            max(abs(x - y) for x, y in zip(pa, pb)),
            max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__))


@dataclass(frozen=True)
class ParityReport:
    window: Window
    branch_count: int
    fork_modes: tuple[str, ...]
    cached_max_abs_logit: float
    cached_max_abs_probability: float
    split_max_abs_logit: float
    split_max_abs_probability: float
    same_top1: bool
    cached_logit_atol: float
    cached_probability_atol: float
    split_atol: float

    @property
    def passed(self) -> bool:
        return (self.branch_count > 0 and self.same_top1
                and self.cached_max_abs_logit <= self.cached_logit_atol
                and self.cached_max_abs_probability <= self.cached_probability_atol
                and max(self.split_max_abs_logit, self.split_max_abs_probability) <= self.split_atol)


class ParityError(RuntimeError):
    def __init__(self, report: ParityReport):
        self.report = report
        super().__init__(f"native block parity failed: {report}")


def _parity_requests():
    # Public synthetic canaries, not evaluation data. Exercise fresh requests and unequal branches.
    return [
        ("Receipt: two charges for one ticket. Please refund the duplicate.", {
            "intent": {"type": "noul", "instructions": "Is a refund requested?"},
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "Payments", "technical": "App failures", "sales": "Pricing"}},
            "time": {"type": "score", "instructions": "Rate urgency.",
                     "criteria": ["Can wait", "Today", "Immediate intervention required"]},
        }),
        ({"route": "Zürich to Roma", "status": "Arrived as scheduled"}, {
            "status": {"type": "choice", "instructions": "Select the status.",
                       "criteria": {"delayed": "Late arrival", "on_time": "No delay"}},
            "late": {"type": "noul", "instructions": "Was the train late?"},
        }),
    ]


class NativeJoey:
    """One frozen model per process. Use load(); stock scoring does not require a parity pass."""

    def __init__(self, engine):
        self._engine = engine
        self._gates: dict[Window, ParityReport] = {}

    @classmethod
    def load(cls, model_dir: str | Path, device: str = "cuda:0", *, fork_mode: str = "sequential"):
        """Load only the pinned, staged checkpoint and reviewed, already-available qev package."""
        if fork_mode not in ("sequential", "auto"):
            raise ValueError("fork_mode must be sequential or auto")
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            raise RuntimeError("start the process with HF_HUB_OFFLINE=1 before importing HF libraries")
        path = _checkpoint_dir(model_dir)
        spec = importlib.util.find_spec("qev")
        if spec is None or not spec.origin:
            raise RuntimeError("make the reviewed, patched OneJev qev source importable offline")
        _verify_hashes(Path(spec.origin).parent, QEV_SHA256)
        import torch
        import transformers
        from huggingface_hub import constants
        from qev.engine import DecisionEngine

        if not constants.HF_HUB_OFFLINE:
            raise RuntimeError("huggingface_hub was imported before offline mode was set; restart the process")
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError(f"transformers=={TRANSFORMERS_VERSION} is required for the native graph")
        if torch.__version__.split("+")[0] != "2.8.0":
            raise RuntimeError("the verified publisher stack requires torch==2.8.0")
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            engine = DecisionEngine(str(path), device=device, dtype="bfloat16", head_dtype="float32",
                                    max_branch_tokens=TRAINED_BRANCH_LIMIT, max_request_tokens=REQUEST_LIMIT,
                                    fork_mode=fork_mode)
        cfg = engine.model.config
        slot_digest = hashlib.sha256(json.dumps(list(engine.slot_ids), separators=(",", ":")).encode()).hexdigest()
        if (type(engine.model).__name__ != "Qwen3_5ForCausalLM" or cfg.model_type != "qwen3_5_text"
                or tuple(cfg.layer_types) != LAYER_TYPES * 6 or len(engine.model.model.layers) != 24
                or cfg._attn_implementation != "sdpa" or engine.model.training
                or not engine.readout.config.noncausal or engine.readout.config.readout != "slot_lm"
                or engine.readout.head is not None or engine.calibration.temperatures
                or engine.model.get_output_embeddings().weight.dtype != torch.float32
                or engine.model.model.embed_tokens.weight.dtype != torch.bfloat16
                or slot_digest != SLOT_IDS_SHA256):
            raise RuntimeError("loaded model, masks, dtype, readout, calibration or slot IDs differ from pinned Joey")
        engine.model.requires_grad_(False)
        return cls(engine)

    @staticmethod
    def _request(state, questions):
        from qev.schema import SystemOneRequest
        return SystemOneRequest(state=state, questions=questions)

    @contextmanager
    def inference_context(self):
        import torch
        # An outer autocast context would otherwise downcast the native fp32 LM head.
        with torch.inference_mode(), torch.autocast(device_type=self._engine.device.type, enabled=False):
            yield

    def _reference(self, request, *, debug):
        e = self._engine
        with self.inference_context():
            try:
                return e.decide(request, debias=1, debug=debug)
            except RuntimeError as exc:
                if e.info.fork_mode != "batched" or "32BitIndexMath" not in str(exc):
                    raise
            # Same fallback as the checkpoint's public engine.py.
            previous, e.info.fork_mode = e.info.fork_mode, "sequential"
            try:
                return e.decide(request, debias=1, debug=debug)
            finally:
                e.info.fork_mode = previous

    def stock(self, state, questions) -> dict[str, dict]:
        response, _ = self._reference(self._request(state, questions), debug=False)
        return {qid: _public(answer) for qid, answer in response.answers.items()}

    def _plan(self, request, window: Window):
        from qev.prompt import render_question
        e = self._engine
        rendered = [render_question(qid, q, style=e.style) for qid, q in request.questions.items()]
        prefix, suffixes, spans = e._split(request.state, rendered)
        if (len(prefix) + max(map(len, suffixes)) > e.max_branch_tokens
                or len(prefix) + sum(map(len, suffixes)) > e.max_request_tokens):
            raise ValueError("request exceeds native token limits; truncation is not supported")
        return {r.qid: NativeBranch(self, request, r, _layout(r, prefix, s, sp, e.slot_ids), window)
                for r, s, sp in zip(rendered, suffixes, spans, strict=True)}

    def branches(self, state, questions, *, window: Window) -> dict[str, NativeBranch]:
        self._require_gate(_check_window(window))
        return self._plan(self._request(state, questions), window)

    def _require_gate(self, window) -> None:
        if window not in self._gates:
            raise RuntimeError(f"run verify_parity(window={window!r}) successfully on this model first")

    def verify_parity(self, *, window: Window, extra_requests=(),
                      logit_atol: float = 0.05, probability_atol: float = 0.005) -> ParityReport:
        """Real cached qev vs complete uncached forward vs real contiguous split, before interventions."""
        from qev.calibrate import softmax
        window = _check_window(window)
        self._gates.pop(window, None)
        if not (0 <= logit_atol <= 0.05 and 0 <= probability_atol <= 0.005):
            raise ValueError("parity thresholds may only tighten 0.05 logit and 0.005 probability")
        n, same, cached_z, cached_p, split_z, split_p = 0, True, 0., 0., 0., 0.
        modes = []
        with self.inference_context():
            for state, questions in (*_parity_requests(), *extra_requests):
                req = self._request(state, questions)
                ref, meta = self._reference(req, debug=True)
                modes.append(meta["fork_mode"])
                for qid, branch in self._plan(req, window).items():
                    z0 = branch._official_logits(meta)
                    if branch.decode(z0) != _public(ref.answers[qid]):
                        raise RuntimeError("official ordered logits did not reconstruct the typed stock answer")
                    z1 = branch._full_logits()
                    z2 = branch._one_pass()
                    z, p, top = _comparison(z1, z2, softmax, branch.temperature)
                    split_z, split_p, same = max(split_z, z), max(split_p, p), same and top
                    for candidate in (z1, z2):
                        z, p, top = _comparison(z0, candidate, softmax, branch.temperature)
                        cached_z, cached_p, same = max(cached_z, z), max(cached_p, p), same and top
                    n += 1
        report = ParityReport(window, n, tuple(modes), cached_z, cached_p, split_z, split_p, same,
                              logit_atol, probability_atol, 1e-5)
        if not report.passed:
            raise ParityError(report)
        self._gates[window] = report
        return report


@dataclass
class _FreshSequence:
    owner: object
    used: bool = False

    def consume(self, owner) -> None:
        if self.owner is not owner or self.used:
            raise ValueError("a new fresh_state() from this branch is required for each call")
        self.used = True


class NativeBranch:
    """Inference-only callbacks for one ordered question. All token positions stay fixed."""

    def __init__(self, native, request, rendered, layout, window):
        self._native = native
        self._request = request
        self._rendered = rendered
        self._layout = layout
        self._window = window
        self._context = None

    @property
    def layout(self) -> BranchLayout:
        return self._layout

    @property
    def qid(self) -> str:
        return self.layout.qid

    @property
    def labels(self) -> tuple[str, ...]:
        return self.layout.labels

    @property
    def temperature(self) -> float:
        return self._native._engine.calibration.temperature(self.layout.kind, len(self.labels))

    def inference_context(self):
        return self._native.inference_context()

    def fresh_state(self):
        return _FreshSequence(self)

    def _official_logits(self, meta):
        diag = meta["diagnostics"][self.qid]
        zs = diag["slot_logits"]
        if "chunks" in diag or len(zs) != 1 or meta["prefix_tokens"] != self.layout.prefix_tokens:
            raise ValueError("chunking, order-averaging or a different native prefix is not a single branch")
        return _finite_vector(zs[0], len(self.labels))

    def stock(self):
        """Fresh official request (including sibling branches), returning this question's [K] tensor."""
        import torch
        with self.inference_context():
            _, meta = self._native._reference(self._request, debug=True)
            return torch.tensor(self._official_logits(meta), dtype=torch.float32, device=self._native._engine.device)

    def probabilities(self, logits) -> tuple[float, ...]:
        from qev.calibrate import softmax
        return tuple(softmax(_finite_vector(logits, len(self.labels)), self.temperature))

    def decode(self, logits) -> dict:
        from qev.answers import make_choice, make_noul, make_score
        p = self.probabilities(logits)
        if self.layout.kind == "noul":
            return _public(make_noul(p[0]))
        if self.layout.kind == "choice":
            return _public(make_choice(self.labels, p))
        if self.layout.kind == "score":
            return _public(make_score(self._rendered.descriptions, p))
        raise ValueError("unsupported decision type")

    def _tokens(self):
        import torch
        ids = torch.tensor([self.layout.input_ids], dtype=torch.long, device=self._native._engine.device)
        return ids, torch.ones_like(ids)

    def _full_logits(self):
        e = self._native._engine
        ids, mask = self._tokens()
        n = ids.shape[1]
        with self.inference_context(), e._blocks([self.layout.prefix_tokens], [n], n):
            out = e.model(input_ids=ids, attention_mask=mask, past_key_values=None, use_cache=False,
                          return_dict=True, logits_to_keep=1)
            if out.past_key_values is not None:
                raise RuntimeError("uncached forward unexpectedly produced a sequence cache")
            return out.logits[0, -1, list(self.layout.slot_ids)].float()

    def prelude(self):
        self._native._require_gate(self._window)
        with self.inference_context():
            return self._prelude()

    def block(self, hidden, state):
        self._native._require_gate(self._window)
        state.consume(self)
        with self.inference_context():
            return self._layers(hidden, *self._window)

    def coda(self, hidden, state):
        self._native._require_gate(self._window)
        state.consume(self)
        with self.inference_context():
            return self._coda(hidden)

    def _one_pass(self):
        hidden = self._prelude()
        hidden = self._layers(hidden, *self._window)
        return self._coda(hidden)

    def one_pass(self):
        """Explicit split prelude + selected block once + coda; never a stock fast path."""
        self._native._require_gate(self._window)
        with self.inference_context():
            return self._one_pass()

    def _coda(self, hidden):
        model = self._native._engine.model
        hidden = self._layers(hidden, self._window[1], len(model.model.layers))
        normed = model.model.norm(hidden)
        logits = model.get_output_embeddings()(normed[:, -1:, :])
        return logits[0, -1, list(self.layout.slot_ids)].float()
