"""Fixed-canary source versus disk-reload parity for an optional second-visit normalization study."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import gc
import hashlib
import math
import struct

from . import native, native_compact, pass_norms
from .loop import LoopCounts


WINDOW = (8, 12)
ATOL = 1e-5
_NAMES = ("input_layernorm", "post_attention_layernorm")


@contextmanager
def _source_donor(layers):
    """Only for the verifier's private full model, never the reloaded or shared scientific instance."""
    if len(layers) != 24:
        raise ValueError("reference needs the full 24 original layers")
    pairs = []
    for i in range(*WINDOW):
        for name in _NAMES:
            a, b = (getattr(layers[j], name) for j in (i, i+4))
            if (a is b or type(a) is not type(b) or type(a).__name__ != "Qwen3_5RMSNorm"
                    or a.weight.shape != b.weight.shape or tuple(a.weight.shape) != (pass_norms.WIDTH,)
                    or a.weight.device != b.weight.device or str(a.weight.dtype) != "torch.bfloat16"
                    or a.weight.dtype != b.weight.dtype or a.weight.requires_grad or b.weight.requires_grad):
                raise ValueError("unexpected original source norm")
            pairs.append((a, a.weight, b.weight))
    if len({id(w) for _, a, b in pairs for w in (a,b)}) != 16:
        raise ValueError("source normalization parameters are unexpectedly aliased")
    with ExitStack() as restore:
        for norm, first, second in pairs:
            restore.callback(setattr, norm, "weight", first)
            norm.weight = second
        yield


def _hidden_pin(h):
    import torch
    dims = tuple(h.shape)
    if h.dtype != torch.bfloat16 or len(dims) != 3 or dims[0] != 1 or not torch.isfinite(h).all().item():
        raise ValueError("invalid first-visit canary hidden state")
    raw = h.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(struct.pack("<3Q", *dims) + raw).hexdigest()


def _endpoint(branch, logits):
    n = len(branch.labels)
    z = tuple(native._finite_vector(logits, n))
    p = tuple(native._finite_vector(branch.probabilities(z), n))
    if any(not 0 <= v <= 1 for v in p) or abs(math.fsum(p)-1) > 1e-6:
        raise ValueError("invalid legal-branch canary probabilities")
    return z, p


def _source_case(b, layers):
    if b._window != (0, 24):
        raise ValueError("source canary needs the entire original span")
    h = b._layers(b._prelude(), 0, 8)
    first = b._layers(h, 8, 12)
    def finish(hidden):
        return _endpoint(b, b._coda(b._layers(hidden, 16, 24)))
    one = finish(first.clone())
    ordinary = finish(b._layers(first.clone(), 8, 12))
    with _source_donor(layers):
        alternate = b._layers(first.clone(), 8, 12)
    return {"layout": b.layout, "first_hidden": _hidden_pin(first),
            "one": one, "identity_two": ordinary, "donor_two": finish(alternate)}


def _source_reference(source_dir, device, requests):
    full = native.NativeJoey.load(source_dir, device=device, fork_mode="sequential")
    full.verify_parity(window=(0,24))
    refs = {}
    with full.inference_context():
        for i, (state, questions) in enumerate(requests):
            for qid, b in full.branches(state, questions, window=(0,24)).items():
                refs[i,qid] = _source_case(b, full._engine.model.model.layers)
    return refs  # Only public layouts, hashes and small legal-slot vectors survive the full model.


def _compare(ref, got):
    if any(len(a) != len(b) or not a for a,b in zip(ref, got, strict=True)):
        raise ValueError("incompatible canary endpoints")
    ref_z, ref_p = ref; z, p = got
    max_z = max(abs(a-b) for a,b in zip(ref_z,z,strict=True))
    max_p = max(abs(a-b) for a,b in zip(ref_p,p,strict=True))
    if (max_z > ATOL or max_p > ATOL or z.index(max(z)) != ref_z.index(max(ref_z))
            or p.index(max(p)) != ref_p.index(max(ref_p))):
        raise ValueError("reloaded canary disagrees with independent source layers")
    return max_z, max_p


def _check_counts(result, visits):
    keys = {key for key,_,_ in pass_norms._pairs()}
    if (type(result.counts) is not LoopCounts or any(type(n) is not int for n in vars(result.counts).values())
            or result.counts != LoopCounts(0, 1, visits, 1) or set(result.norm_calls) != keys
            or any(type(x) is not int or x != visits-1 for x in result.norm_calls.values())):
        raise ValueError("canary did not execute the required blocks and normalization calls")


def verify_reload(source_dir, compact_dir, bundle, *, compact_manifest_sha256, sidecar_manifest_sha256, device="cuda:0"):
    """Run only the built-in public canaries on a separately admitted, staged offline runtime.

    No dataset/label argument, worker permission, timing or scoring. Fixtures cannot establish this gate.
    """
    from .norm_overlay import NormOverlay
    side, _ = pass_norms.read_vectors(bundle, manifest_sha256=sidecar_manifest_sha256,
                                      compact_manifest_sha256=compact_manifest_sha256)
    _, m, kept = native_compact._verified_export(compact_dir, source_dir, compact_manifest_sha256)
    if kept != pass_norms.KEPT or m["plan"] != pass_norms.BASE_PLAN or m["weights_sha256"] != side["base"]["weight_sha256"]:
        raise ValueError("source witness requires the exact earlier-cycle base")
    requests = tuple(native._parity_requests())
    refs = _source_reference(source_dir, device, requests)
    gc.collect()
    import torch
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    small = native_compact.CompactJoey.load(compact_dir, source_model_dir=source_dir,
                                            expected_manifest_sha256=compact_manifest_sha256, device=device)
    overlays = {kind: NormOverlay.load(small, bundle, manifest_sha256=sidecar_manifest_sha256,
                                       compact_manifest_sha256=compact_manifest_sha256, kind=kind)
                for kind in ("identity","donor")}
    maximum = {n: [0.,0.] for n in ("one", "identity_two", "donor_two")}
    observed = set(); distinct = 0
    def check(ref, got, role):
        delta = _compare(ref[role], got)
        maximum[role] = [max(x, y) for x,y in zip(maximum[role],delta,strict=True)]
    with small.inference_context():
        for i,(state,questions) in enumerate(requests):
            for qid,b in small.branches(state, questions, window=WINDOW).items():
                key = i,qid
                if key not in refs or key in observed or b.layout != refs[key]["layout"]:
                    raise ValueError("canary input layout, branch or legal slots changed")
                observed.add(key); ref = refs[key]
                def first_pin():
                    return _hidden_pin(b.block(b.prelude(), b.fresh_state()))
                if first_pin() != ref["first_hidden"]:
                    raise ValueError("first-visit hidden differs from the original kept layers")
                measured = {}
                for kind, overlay in overlays.items():
                    for visits in (1, 2):
                        result = overlay.run(b, visits=visits); _check_counts(result, visits)
                        role = "one" if visits == 1 else kind+"_two"
                        got = _endpoint(b, result.logits)
                        check(ref, got, role)
                        if visits == 2:
                            measured[role] = got
                if first_pin() != ref["first_hidden"]:
                    raise ValueError("using the second-visit overlay changed the first path")
                def gap(results):
                    return max(abs(x-y) for x,y in zip(results["identity_two"][0],results["donor_two"][0],strict=True))
                distinct += min(gap(ref), gap(measured)) > 2*ATOL
    if not refs or observed != refs.keys() or not distinct:
        raise ValueError("canaries lack full coverage or cannot distinguish the second-visit substitution")
    return {"schema": "joey-pass-norm-reload-canary/v1", "source_weights_sha256": native.MODEL_SHA256["model.safetensors"],
            "compact_manifest_sha256": compact_manifest_sha256, "sidecar_manifest_sha256": sidecar_manifest_sha256,
            "canary_branches": len(refs), "distinguishing_branches": distinct, "first_hidden_byte_identical": True,
            "max_abs_logit": {k:v[0] for k,v in maximum.items()}, "max_abs_probability": {k:v[1] for k,v in maximum.items()},
            "atol": ATOL, "same_top1": True, "study_inputs_used": False, "quality_measured": False}
