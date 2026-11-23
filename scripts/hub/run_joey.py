"""Run a local Joey-1-Loop or Joey-1-Compact-Loop Hub snapshot (Python 3.10+)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


VARIANTS = {
    "Joey-1-Loop": (24, "4c503aded23bae311dff80a63326503fb720572c178f07a2142332d55f5f3afc",
                   "7c7ebc3322308391787aac279c7cbd4c3a38bba85e270b5f5568ad07467673e1"),
    "Joey-1-Compact-Loop": (20, "88474794adb9c87089a8baf8e73ce919b0ca2f052608c1ed96e30ccf2ff45d5c",
                           "9d0074091a2fd9075ae2577dac0feffb5465b03731ead2e66d21ac28e8006a08"),
}
WINDOW = (8, 12)
EXAMPLE = {
    "state": "Please replace my card. The chip has stopped working.",
    "questions": {"department": {"type": "choice", "instructions": "Who should handle the request?",
                                 "criteria": {"cards": "Card replacement", "transfers": "Money transfers",
                                              "login": "Trouble signing in"}}},
}
QEV_NOTICE = b"# OneJev source distribution with Baseten's Joey serve.patch; see THIRD_PARTY.md.\n"


def check_snapshot(root, native):
    spec = json.loads((root / "loop_config.json").read_bytes())
    name = spec["variant"]
    if name not in VARIANTS:
        raise ValueError("not a supported Joey loop release")
    depth, weights, config = VARIANTS[name]
    expected = {"variant": name, "source_revision": native.MODEL_REVISION, "stored_text_layers": depth,
                "physical_layer_visits": list(range(12)) + list(range(8, depth)),
                "middle_passes": 2, "readout": "slot_lm"}
    if spec != expected:
        raise ValueError("the declared schedule is not the published two-pass schedule")
    pins = {**native.MODEL_SHA256, "model.safetensors": weights, "config.json": config}
    native._verify_hashes(root, pins)
    for filename, pin in native.QEV_SHA256.items():
        data = (root / "qev" / filename).read_bytes()
        if not data.startswith(QEV_NOTICE) or hashlib.sha256(data[len(QEV_NOTICE):]).hexdigest() != pin:
            raise ValueError(f"OneJev file differs from the pinned Baseten runtime: {filename}")
    return spec, tuple(pins)


def run_two_pass(native_model, state, questions):
    answers = {}
    for qid, branch in native_model.branches(state, questions, window=WINDOW).items():
        with branch.inference_context():
            hidden = branch.prelude()
            for _ in range(2):
                hidden = branch.block(hidden, branch.fresh_state())
            answers[qid] = branch.decode(branch.coda(hidden, branch.fresh_state()))
    return answers


def decide(root, request, *, device="cuda:0"):
    from native_joey import NativeJoey, LAYER_TYPES, SLOT_IDS_SHA256, REQUEST_LIMIT, TRAINED_BRANCH_LIMIT
    import native_joey

    spec, files = check_snapshot(root, native_joey)
    import torch
    import transformers
    from huggingface_hub.constants import HF_HUB_OFFLINE
    from qev.engine import DecisionEngine

    if not HF_HUB_OFFLINE or transformers.__version__ != native_joey.TRANSFORMERS_VERSION:
        raise RuntimeError("start a fresh offline process with the pinned Transformers version")
    if torch.__version__.split("+")[0] != "2.8.0" or torch.device(device).type != "cuda":
        raise RuntimeError("the supported runtime uses CUDA and torch==2.8.0")
    depth = spec["stored_text_layers"]
    # Give Transformers just the verified loader inputs, never unrelated Hub checkout files.
    with tempfile.TemporaryDirectory(prefix="joey-model-") as tmp:
        for filename in files:
            (Path(tmp) / filename).symlink_to((root / filename).resolve(strict=True))
        with torch.autocast(device_type="cuda", enabled=False):
            e = DecisionEngine(tmp, device=device, dtype="bfloat16", head_dtype="float32",
                               max_branch_tokens=TRAINED_BRANCH_LIMIT, max_request_tokens=REQUEST_LIMIT,
                               fork_mode="sequential")
        cfg = e.model.config
        slots = hashlib.sha256(json.dumps(list(e.slot_ids), separators=(",", ":")).encode()).hexdigest()
        if (type(e.model).__name__ != "Qwen3_5ForCausalLM" or cfg.model_type != "qwen3_5_text"
                or tuple(cfg.layer_types) != LAYER_TYPES * (depth // 4) or cfg.num_hidden_layers != depth
                or len(e.model.model.layers) != depth or cfg._attn_implementation != "sdpa"
                or e.model.training or not e.readout.config.noncausal or e.readout.config.readout != "slot_lm"
                or e.readout.head is not None or e.calibration.temperatures or slots != SLOT_IDS_SHA256
                or e.model.get_output_embeddings().weight.dtype != torch.float32
                or e.model.model.embed_tokens.weight.dtype != torch.bfloat16):
            raise RuntimeError("unexpected model depth, attention, head, or slot mapping")
        e.model.requires_grad_(False)
        model = NativeJoey(e)
        model.verify_parity(window=WINDOW)
        return run_two_pass(model, request["state"], request["questions"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only", action="store_true", help="check the actual checkpoint without GPU dependencies")
    p.add_argument("--request", type=Path, help="text decision JSON containing state and questions")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    root = Path(__file__).resolve().parent
    if args.verify_only:
        import native_joey
        spec, _ = check_snapshot(root, native_joey)
        print(f'{spec["variant"]}: verified {spec["stored_text_layers"]} stored layers, '
              f'{len(spec["physical_layer_visits"])} layer visits; no inference run.')
    else:
        request = json.loads(args.request.read_bytes()) if args.request is not None else EXAMPLE
        print(json.dumps(decide(root, request, device=args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
