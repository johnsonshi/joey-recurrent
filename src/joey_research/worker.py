"""Offline, single-question decision execution. The worker never loads labels or cloud clients."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict

from .evaluation import input_sha256
from .loop import LoopPlan, run_loop
from .loop_analysis import Policy, SlotOrder, Stop, actuate


def compact_recipe(name: str) -> dict:
    if name == "select-20":
        groups, repeat = [{"cycles":[g]} for g in (0,1,2,4,5)], [2,3]
    elif name == "select-16":
        groups, repeat = [{"cycles":[g]} for g in (0,2,3,5)], [1,3]
    elif name == "merge-16":
        groups = [{"cycles":[0]}, {"cycles":[1,2],"weights":[0.25,0.75],"anchor":2},
                  {"cycles":[3,4],"weights":[0.75,0.25]}, {"cycles":[5]}]
        repeat = [1,3]
    else:
        raise ValueError("unknown deterministic compact topology")
    return {"format":"joey-compact/v1","groups":groups,"repeat_cycles":repeat}


def choice_request(row: dict, order: tuple[int, ...]) -> tuple[str, dict]:
    input_sha256(row)  # Validate the label-free projection.
    choices = row["choices"]
    if len(choices) > 255 or sorted(order) != list(range(len(choices))):
        raise ValueError("unsupported choice count or ordering")
    criteria = {f"opt_{i}": choices[i] for i in order}
    return row["prompt"], {"decision": {"type": "choice", "instructions": "Select the best option.", "criteria": criteria}}


def measure_choice(native, row: dict, window: tuple[int, int], plan: LoopPlan, *, reverse=False, policy=None) -> tuple[dict, dict]:
    order = tuple(range(len(row["choices"])))
    if reverse:
        order = order[::-1]
    state, questions = choice_request(row, order)
    branch = native.branches(state, questions, window=window)["decision"]
    if tuple(branch.labels) != tuple(questions["decision"]["criteria"]):
        raise ValueError("native option order differs from the request")

    def original_order(p):
        if len(p) != len(order) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) or abs(math.fsum(p)-1) > 1e-6:
            raise ValueError("invalid native distribution")
        return [v for _, v in sorted(zip(order, p, strict=True))]
    lo, hi = window
    depth = len(native.physical_to_source) if hasattr(native, "physical_to_source") else 24
    digest = input_sha256(row)
    more = {}
    if policy is None:
        result = run_loop(plan, branch)
        counts = result.counts
        p = original_order(branch.probabilities(result.logits))
        trace = [{"block_evaluations": r.block_evaluations, "kind": r.kind,
                  "probs": original_order(branch.probabilities(r.logits))} for r in result.readouts]
    else:
        if branch.temperature != 1:
            raise ValueError("raw-logit policy requires the pinned uncalibrated readout")
        decision = actuate(plan, branch, lambda: SlotOrder("discovery",digest,tuple(branch.labels)), policy)
        counts = decision.executed
        p = original_order(decision.probabilities)
        trace = []
        more = {"policy_decision_pass":decision.decision_pass,"policy_reason":decision.reason,
                "policy_fallback":decision.fallback_used, "mode":"live-actuated"}
    return (
        {"id": row["id"], "input_sha256": digest, "probs": p},
        {"id": row["id"], "tokens": len(branch.layout.input_ids), "layer_calls": counts.layer_evaluations(lo, hi-lo, depth-hi),
         "callback_counts": asdict(counts), "readouts": trace, **more},
    )


def _load(path: Path, digest: str) -> bytes:
    data = path.read_bytes()
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("staged input digest differs")
    return data


def _policy(obj):
    if not isinstance(obj,dict) or set(obj)-{"pool","stop","fallback","first_weight"}:
        raise ValueError("invalid discovery policy")
    cfg=dict(obj)
    if "stop" in cfg:
        if not isinstance(cfg["stop"],dict) or set(cfg["stop"])!={"kind","threshold"}:
            raise ValueError("invalid stopping rule")
        cfg["stop"]=Stop(**cfg["stop"])
    return Policy(**cfg)


def _packet(raw: bytes) -> tuple[dict, tuple[int, int], LoopPlan]:
    obj = json.loads(raw)
    required = {"schema", "split", "inputs_sha256", "window", "loop", "reverse", "deadline_utc"}
    if not required <= set(obj) <= (required | {"compact_manifest_sha256","compact_recipe","policy"}) or obj["schema"] != 1:
        raise ValueError("unexpected worker packet")
    if "compact_manifest_sha256" in obj and not re.fullmatch(r"[0-9a-f]{64}", obj["compact_manifest_sha256"]):
        raise ValueError("invalid compact receipt")
    if "compact_recipe" in obj:
        compact_recipe(obj["compact_recipe"])
        if "compact_manifest_sha256" not in obj:
            raise ValueError("derived compact payload needs an independent coordinator pin")
    if obj["split"] != "discovery" or type(obj["reverse"]) is not bool:
        raise ValueError("worker only accepts explicit discovery packets")
    w = obj["window"]
    if not isinstance(w, list) or len(w) != 2 or not all(type(i) is int for i in w) or not 0 <= w[0] < w[1] < 24:
        raise ValueError("invalid stock model window")
    if not isinstance(obj["loop"], dict) or set(obj["loop"]) - {"k", "method", "alpha", "beta", "readout"}:
        raise ValueError("unrecognized loop configuration")
    plan = LoopPlan(**obj["loop"])
    if plan.k > 8:
        raise ValueError("pass budget exceeded")
    if "policy" in obj:
        _policy(obj["policy"])
        if plan.method not in ("naive","damped_extra") or plan.readout != "final":
            raise ValueError("live pooling needs completed repeats; use separate diagnostic packets for recording")
    return obj, (w[0], w[1]), plan


def run(args) -> None:
    if any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("offline loader flags are required")
    obj, window, plan = _packet(_load(args.packet, args.packet_sha256))
    if type(obj["deadline_utc"]) is not int or not 0 < obj["deadline_utc"] - time.time() <= 30 * 3600:
        raise ValueError("expired or unbounded worker deadline")
    # The coordinator binds a stricter absolute cutoff to this exact packet.
    # The OS alarm also covers initialization and time spent inside native code.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    remaining = obj["deadline_utc"] - time.time()
    if remaining <= 0:
        raise ValueError("worker deadline elapsed before start")
    signal.setitimer(signal.ITIMER_REAL, remaining)
    rows = [json.loads(s) for s in _load(args.inputs, obj["inputs_sha256"]).splitlines()]
    if not 0 < len(rows) <= 20000 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("invalid tier or duplicate decision IDs")
    for row in rows:
        choice_request(row, tuple(range(len(row["choices"]))))
    if os.environ.get("TMPDIR") == "/work/tmp":
        Path("/work/tmp").mkdir(mode=0o700, exist_ok=True)
    import torch
    from .native import NativeJoey
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("one BF16-capable CUDA device is required")
    props = torch.cuda.get_device_properties(0)
    if not re.search(r"\bA100\b", props.name) or (props.major, props.minor) != (8, 0):
        raise ValueError("device class mismatch")
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    begin = time.perf_counter()
    if "compact_manifest_sha256" in obj:
        if "compact_recipe" in obj:
            from .compact import export_compact
            if args.compact_dir is not None:
                raise ValueError("choose a staged or a derived compact directory, not both")
            compact_dir = args.output.parent / "derived-compact"
            export_compact(args.model_dir, compact_dir, compact_recipe(obj["compact_recipe"]))
            _load(compact_dir / "compact.json", obj["compact_manifest_sha256"])
        elif args.compact_dir is None:
            raise ValueError("verified compact directory required")
        else:
            compact_dir = args.compact_dir
        from .native_compact import CompactJoey
        native = CompactJoey.load(compact_dir, source_model_dir=args.model_dir,
                                  expected_manifest_sha256=obj["compact_manifest_sha256"])
        if native.repeat_window != window:
            raise ValueError("compact window differs from the verified physical mapping")
    else:
        if args.compact_dir is not None:
            raise ValueError("compact directory without compact manifest pin")
        native = NativeJoey.load(args.model_dir)
    report = native.verify_parity(window=window)
    policy = _policy(obj["policy"]) if "policy" in obj else None
    measure_choice(native, rows[0], window, plan, reverse=obj["reverse"], policy=policy)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - begin
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    times = []
    outputs = [args.output / "predictions.partial", args.output / "trace.partial"]
    with outputs[0].open("x") as predictions, outputs[1].open("x") as traces:
        for row in rows:
            torch.cuda.synchronize()
            wall = time.perf_counter_ns()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            pred, diagnostic = measure_choice(native, row, window, plan, reverse=obj["reverse"], policy=policy)
            end.record()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - wall) / 1e6
            diagnostic.update(wall_ms=elapsed, cuda_event_ms=start.elapsed_time(end))
            times.append(elapsed)
            for f, rec in ((predictions, pred), (traces, diagnostic)):
                f.write(json.dumps(rec, allow_nan=False, separators=(",", ":")) + "\n")
                f.flush()
                os.fsync(f.fileno())
    duration = time.perf_counter() - t0
    ordered_times = sorted(times)
    info = {"schema": 1, "packet_sha256": args.packet_sha256, "batch_size": 1, "decisions": len(rows),
            "parity": asdict(report), "hardware": "A100", "total_device_bytes": props.total_memory,
            "warmup_load_and_parity_s": warmup_s, "wall_including_persistence_s": duration,
            "decisions_per_wall_s": len(rows)/duration,
            "warm_latency_ms": {str(p): ordered_times[math.ceil(len(times)*p/100)-1] for p in (50,95,99)},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "steady_allocated_bytes": torch.cuda.memory_allocated(), "steady_reserved_bytes": torch.cuda.memory_reserved()}
    if "compact_manifest_sha256" in obj:
        info.update(compact_manifest_sha256=obj["compact_manifest_sha256"], reload_parity=asdict(native.reload_report),
                    physical_layers=len(native.physical_to_source))
    for f, name in zip(outputs, ("predictions.jsonl", "trace.jsonl"), strict=True):
        f.rename(args.output/name)
    with (args.output/"complete.json").open("x") as f:
        json.dump(info, f, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def _quiet_native():
    # Native libraries can print outside Python's sys.stderr. Jobs return results
    # through their authorized private volume, not through shared container logs.
    sys.stdout.flush()
    sys.stderr.flush()
    backup = os.dup(1), os.dup(2)
    try:
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        for destination, original in enumerate(backup, start=1):
            os.dup2(original, destination)
            os.close(original)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "inputs", "model-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--compact-dir", type=Path)
    p.add_argument("--packet-sha256", required=True)
    args = p.parse_args()
    try:
        with _quiet_native():
            run(args)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError, TypeError) as e:
        # Never log the prompt, provider message, hostname, path or local bindings.
        p.exit(2, f"worker refused trial ({type(e).__name__}); incomplete output is not scoreable\n")


if __name__ == "__main__":
    main()
