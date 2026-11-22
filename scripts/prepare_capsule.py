#!/usr/bin/env -S python3 -IBS
"""Coordinator-operated public ingress. The default command is offline."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
if sys.version_info < (3, 11):
    raise SystemExit("Use Python 3.11+ (on the development host: python3.14).")
SCRIPTS = Path(__file__).resolve().parent
SURFACE = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
from capsule_build import assemble, native_files, read_handoff, receipt_row, wheels
from capsule_io import Refused, canonical, copy_cached_tensor, directory, document, download, require, run_directory, sha, small_file
from capsule_net import HF, Network
from capsule_policy import PINS, WHEEL_LOCK, Policy, review_refs


def checkpoint(net, policy, work, partials, *, model_file=None):
    payload = directory(work / "payload")
    model = directory(payload / "model")
    rows = []
    tensor = "model.safetensors"
    # Fail closed on an explicit local file before any GET. Otherwise download the big object last.
    sidecars = sorted(policy.model.keys() - {tensor})
    for name in ([tensor, *sidecars] if model_file is not None else [*sidecars, tensor]):
        pin = policy.model[name]
        limit = 24 * 2**30 if name == tensor else 128 * 2**20
        if name == tensor and model_file is not None:
            size = copy_cached_tensor(model_file, model / name, partials, pin, limit)
        else:
            size = download(net, policy.hf(name), model / name, partials, pin, limit)
        rows.append(receipt_row("model/" + name, pin, size, HF, policy.model_rev, policy.model_license))
    return rows


class Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse normally echoes unknown arguments (which might contain a URL).
        self.exit(2, "invalid arguments; see --help\n")


def main(argv=None):
    parser = Parser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("plan", help="offline: check trust roots and print public counts")
    check = commands.add_parser("check-input", help="offline: validate the private, coordinator-pinned handoff")
    model = commands.add_parser("model", help="authorized host only: stage the checkpoint (not a capsule)")
    refs = commands.add_parser("review-refs", help="authorized host only: materialize four pinned public review texts")
    prefetch = commands.add_parser("prefetch", help="stage locked wheels, runtime, and sidecars while an independently verified weight transfer runs")
    prefetch.add_argument("--reference-pins-sha256", required=True)
    prepare = commands.add_parser("prepare", help="authorized host only: assemble and seal a complete capsule")
    for p in (model, refs, prefetch, prepare):
        p.add_argument("run", help="new lowercase run name, not a path")
        p.add_argument("--resume", action="store_true", help="explicitly continue this incomplete run")
    for p in (model, prepare):
        group = p.add_mutually_exclusive_group()
        group.add_argument("--model-file", type=Path, help="use this private staged model.safetensors; no weight GET")
        group.add_argument("--model-cache", type=Path, help="use only DIR/model.safetensors; never scan DIR")
    for p in (check, prepare):
        p.add_argument("--handoff-sha256", required=True, help="independent digest of the approved handoff")
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        policy = Policy(SURFACE)
        if args.command in (None, "plan"):
            print(canonical({"model_files": len(policy.model), "qev_files": len(policy.qev),
                             "wheels": len(policy.wheels), "base_digest": policy.base_digest,
                             "wheel_lock_sha256": PINS[WHEEL_LOCK]}).decode().strip())
            return 0
        handoff = None
        if args.command in ("check-input", "prepare"):
            handoff = read_handoff(SURFACE, args.handoff_sha256)
            if args.command == "check-input":
                print("Pinned handoff checked; no network or capsule writes.")
                return 0
        with run_directory(SURFACE, args.run, args.resume) as run:
            model_file = getattr(args, "model_file", None)
            if getattr(args, "model_cache", None) is not None:
                model_file = args.model_cache / "model.safetensors"
            if model_file is not None:
                policy.sources.discard(policy.hf("model.safetensors"))
            net = Network(policy.sources)
            if args.command == "review-refs":
                print(canonical(review_refs(net, policy, run)).decode().strip())
            elif args.command == "prefetch":
                pins = small_file(run/"review"/"reference-pins.json", 8192, pin=args.reference_pins_sha256, private=True)
                reviewed = document(pins)
                require(set(reviewed) == {"notices", "serve_patch_sha256"}, "unexpected reviewed pins")
                policy.sources.discard(policy.hf("model.safetensors"))
                work = directory(run/"work")
                partials = directory(run/"partials")
                native = native_files(net, policy, work, partials, reviewed)
                model_path = directory(directory(work/"payload")/"model")
                for name,pin in policy.model.items():
                    if name != "model.safetensors":
                        download(net, policy.hf(name), model_path/name, partials, pin, 128*2**20)
                checked = wheels(net, policy, work, partials)
                print(canonical({"wheel_count":len(checked),"native_and_notice_count":len(native),
                                 "weight_downloads":0,"review_sha256":sha(pins)}).decode().strip())
            elif args.command == "model":
                checkpoint(net, policy, directory(run / "work"), directory(run / "partials"), model_file=model_file)
                print("Checkpoint verified. Incomplete run; no capsule published.")
            else:
                pins = assemble(net, policy, run, handoff, args.handoff_sha256, checkpoint, model_file=model_file)
                print(canonical(pins).decode().strip())
        return 0
    except Refused as e:
        print(f"staging refused: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("interrupted; use --resume for an incomplete run", file=sys.stderr)
    except Exception:
        # HTTP errors can contain signed redirects; OS/parse errors can contain host paths or row text.
        print("staging stopped: transport, input or local filesystem error", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
