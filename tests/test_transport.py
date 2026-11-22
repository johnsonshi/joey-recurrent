"""Synthetic transport fixtures only: no installed wheels, torch, model, or cluster."""
import contextlib
import errno
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import sys
import threading
import time
import unittest
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import patch

from joey_research import transport as tr
from joey_research import worker
from joey_research.evaluation import input_sha256


class TransportTest(unittest.TestCase):
    def setUp(self):
        self.root = Path("tests", ".transport-fixture-" + secrets.token_hex(8)).absolute()
        self.root.mkdir(mode=0o700)
        self.addCleanup(shutil.rmtree, self.root)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for name in ("WORK", "STAGED", "ADMISSION"):
            path = self.root / name.lower()
            path.mkdir()
            stack.enter_context(patch.object(tr, name, path))
        t = int(time.time())
        self.probe = tr.Attempt("probe", secrets.token_hex(32), t + 45, t + 90)
        self.row = {"id": "one", "group_id": "g", "task": "choice", "prompt": "Which?", "choices": ["A", "B"]}

    def discovery(self, *, row=None, requirements=None, **updates):
        wheel = "sample-0.1-py3-none-any.whl"
        wheels = tr.STAGED / "wheelhouse"
        wheels.mkdir(exist_ok=True)
        (wheels / wheel).write_bytes(b"fixture, not an installable wheel")
        if requirements is None:
            requirements = f"./wheelhouse/{wheel} --hash=sha256:{tr._sha((wheels / wheel).read_bytes())}\n".encode()
        (tr.STAGED / "requirements-linux.txt").write_bytes(requirements)
        raw = json.dumps(row or self.row).encode() + b"\n"
        inputs = tr.STAGED / "inputs"
        inputs.mkdir(exist_ok=True)
        (inputs / "discovery.jsonl").write_bytes(raw)
        packet = {"schema": 1, "split": "discovery", "inputs_sha256": tr._sha(raw), "reverse": False,
                  "window": [4, 8], "loop": {"k": 1}, "deadline_utc": self.probe.hard_stop, **updates}
        packet_raw = json.dumps(packet).encode()
        (tr.ADMISSION / "packet.json").write_bytes(packet_raw)
        return replace(self.probe, mode="discovery", packet_sha256=tr._sha(packet_raw),
                       requirements_sha256=tr._sha(requirements))

    def selection(self):
        a = self.discovery()  # Leave a different discovery.jsonl present throughout.
        groups = ("alpha", "alpha", "beta", "gamma", "delta")
        rows = [dict(self.row, id=str(i), prompt=f"Menu {i}?", group_id=g, task="A" if i < 3 else "B")
                for i, g in enumerate(groups)]
        raw = b"".join(tr._encoded(r) + b"\n" for r in rows)
        (tr.STAGED / "inputs/selection.jsonl").write_bytes(raw)
        settings = {"window": [8, 12], "loop": {"k": 2, "method": "zero_extra"}, "reverse": False,
                    "order_ensemble": {"numerator": 1, "denominator": 2, "seed": 42}}
        mid = worker.method_sha256(settings)
        nom = tr._encoded({"schema": "joey-promotion-permit/v1", "split": "selection", "inputs_sha256": tr._sha(raw),
                           "split_manifest_sha256": "c"*64, "methods": {mid: settings}})
        (tr.ADMISSION / "nomination.json").write_bytes(nom)
        packet = tr._encoded(dict(settings, schema=1, split="selection", permit_sha256=tr._sha(nom), method_sha256=mid,
                                  inputs_sha256=tr._sha(raw), deadline_utc=a.hard_stop))
        (tr.ADMISSION / "packet.json").write_bytes(packet)
        return replace(a, mode="selection", permit_sha256=tr._sha(nom), packet_sha256=tr._sha(packet)), rows

    @staticmethod
    def probe_doc():
        # Deliberately include identifying attributes in the fake CUDA object;
        # none belong in the wire schema, including the full GPU name.
        props = SimpleNamespace(name="private vendor A100 board", total_memory=42_000_000_000,
                                uuid="private-uuid", major=8, minor=0)
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1, get_device_properties=lambda _: props),
                                compiled_with_cxx11_abi=lambda: True, __version__="2.8.0+cu129",
                                version=SimpleNamespace(cuda="12.9"))
        return tr._probe(torch)

    def probe_files(self):
        return {"challenge.bin": tr._challenge(self.probe),
                "probe.json": tr._encoded(dict(self.probe_doc(), nonce_sha256=self.probe.nonce_sha256))}

    @staticmethod
    def worker_files(attempt, row):
        prediction = {"id": row["id"], "input_sha256": input_sha256(row), "probs": [0.4, 0.6]}
        trace = {"id": row["id"], "tokens": 4, "layer_calls": 24, "readouts": [],
                 "wall_ms": 2, "cuda_event_ms": 1, "callback_counts": {}}
        metrics = {"decisions": 1, "schema": 1, "batch_size": 1, "packet_sha256": attempt.packet_sha256,
                   "hardware": "A100", "total_device_bytes": 42000000000, "parity": {},
                   "warmup_load_and_parity_s": 2.0, "wall_including_persistence_s": 0.1,
                   "decisions_per_wall_s": 10, "warm_latency_ms": {"50": 2, "95": 2, "99": 2}}
        return {"predictions.jsonl": tr._encoded(prediction) + b"\n", "trace.jsonl": tr._encoded(trace) + b"\n",
                "complete.json": tr._encoded(metrics)}

    def collect(self, port, *, destination=None):
        a = self.probe
        return tr.collect(port, destination or self.root / "saved", mode=a.mode, nonce=a.nonce,
                          admit_before=a.admit_before, hard_stop=a.hard_stop)

    def test_probe_roundtrip_real_loopback_and_durable_ack(self):
        def probe_phase(*args, **kwargs):
            (tr.WORK / "run/base.json").write_bytes(tr._encoded(self.probe_doc()))

        with patch.object(tr, "_phase", side_effect=probe_phase):
            status, files = tr._result(self.probe)
        routes = tr._publication(self.probe, status, files)
        independent = hashlib.shake_256(b"joey-loopback/challenge/v1\x00" +
                                        bytes.fromhex(self.probe.nonce)).digest(65536)
        self.assertEqual((tr.WORK / "run/challenge.bin").read_bytes(), independent)
        saved = self.root / "saved"
        exits = []
        did_sync = []
        real_fsync = os.fsync
        real_exchange = tr._exchange

        def sync(fd):
            real_fsync(fd)
            did_sync.append(os.fstat(fd).st_ino)

        def exchange(port, attempt, target, limit, **kwargs):
            if target == "/ack":
                self.assertEqual((saved / "challenge.bin").read_bytes(), independent)
                self.assertEqual(set(os.listdir(saved)), {"challenge.bin", "manifest.json", "probe.json"})
                for path in (saved, self.root, *saved.iterdir()):
                    self.assertIn(path.stat().st_ino, did_sync)
                    self.assertEqual(path.stat().st_mode & 0o077, 0)
                self.assertEqual(json.loads(kwargs["ack"])["receipt_sha256"],
                                 hashlib.sha256((saved / "manifest.json").read_bytes()).hexdigest())
            return real_exchange(port, attempt, target, limit, **kwargs)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            thread = threading.Thread(target=lambda: exits.append(tr._serve(listener, self.probe, routes, status)),
                                      daemon=True)
            thread.start()
            with patch.object(os, "fsync", side_effect=sync), patch.object(tr, "_exchange", side_effect=exchange):
                doc = self.collect(listener.getsockname()[1])
            thread.join(2)
        self.assertEqual(exits, [0])
        self.assertEqual(doc["status"], "setup_probe")
        self.assertEqual(set(doc["files"]), {"challenge.bin", "probe.json"})
        metadata = json.loads((saved / "probe.json").read_bytes())
        self.assertEqual(metadata["devices"], [{"a100": True, "vram_bytes": 42000000000, "capability": [8, 0]}])
        self.assertEqual(metadata["nonce_sha256"], hashlib.sha256(self.probe.nonce.encode()).hexdigest())
        self.assertNotIn("private", json.dumps(metadata))
        self.assertNotIn(self.probe.nonce, json.dumps(doc))

    def wire(self, routes, line, *, body=b"", nonce=True, extra=b""):
        a, b = socket.socketpair()
        with a, b:
            headers = b"X-Joey-Nonce: " + self.probe.nonce.encode() + b"\r\n" if nonce else b""
            a.sendall(line + b" HTTP/1.1\r\n" + headers + extra + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            acked = tr._request(b, self.probe, routes)
            b.shutdown(socket.SHUT_WR)
            with a.makefile("rb") as stream:
                response = http.client.HTTPResponse(SimpleNamespace(makefile=lambda *a, **kw: stream))
                response.begin()
                return response.status, response.read(), acked

    def test_only_exact_authenticated_routes_and_bound_ack(self):
        routes = tr._publication(self.probe, "setup_probe", self.probe_files())
        self.assertEqual(self.wire(routes, b"GET /manifest", nonce=False), (403, b"", False))
        for line in (b"POST /manifest", b"GET //manifest", b"GET /manifest?q=1", b"GET /file/%70robe.json",
                     b"GET /file/../missing", b"GET /file/trace.jsonl", b"GET /file/", b"HEAD /manifest"):
            with self.subTest(line=line):
                self.assertEqual(self.wire(routes, line), (404, b"", False))
        self.assertEqual(self.wire(routes, b"GET /manifest", extra=b"Range: bytes=0-2\r\n"), (400, b"", False))
        self.assertEqual(self.wire(routes, b"GET /file/probe.json")[1], self.probe_files()["probe.json"])
        for nonce, digest in ((self.probe.nonce, "1"*64), ("2"*64, routes[b"/manifest"][1])):
            self.assertEqual(self.wire(routes, b"POST /ack", body=tr._encoded({"nonce": nonce, "receipt_sha256": digest})),
                             (403, b"", False))

    def test_bad_digest_challenge_or_fsync_never_ack(self):
        for case in ("bad_bytes", "wrong_challenge", "stale_manifest", "fsync"):
            with self.subTest(case=case):
                files = self.probe_files()
                attempt = self.probe
                if case == "wrong_challenge":
                    files["challenge.bin"] = bytes(65536)
                if case == "stale_manifest":
                    attempt = replace(attempt, nonce="2"*64)
                routes = tr._publication(attempt, "setup_probe", files)
                calls = []

                def exchange(port, a, target, limit):
                    calls.append(target)
                    if target == "/file/probe.json" and case == "bad_bytes":
                        return b"wrong"
                    return routes[target.encode()][0]

                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(tr, "_exchange", side_effect=exchange))
                    if case == "fsync":
                        stack.enter_context(patch.object(os, "fsync", side_effect=OSError))
                    with self.assertRaises(tr.TransportError):
                        self.collect(1)
                self.assertNotIn("/ack", calls)
                self.assertFalse((self.root / "saved").exists())
                self.assertFalse((self.root / ".saved.receiving").exists())

    def test_genuine_file_bytes_frozen_after_supervision(self):
        attempt = self.discovery()
        expected = self.worker_files(attempt, self.row)
        seen = []

        def phase(a, argv, *, cwd, installed):
            seen.append(argv)
            self.assertNotIn(a.nonce, " ".join(argv))
            if not installed:
                self.assertEqual(cwd, tr.STAGED)
                for flag in ("pip", "--isolated", "--require-hashes", "--no-deps", "--no-index", "--only-binary=:all:"):
                    self.assertIn(flag, argv)
                return
            self.assertIn("joey_research.worker", argv)
            for path in (tr.ADMISSION / "packet.json", tr.STAGED / "inputs/discovery.jsonl", tr.STAGED / "model"):
                self.assertIn(str(path), argv)
            out = tr.WORK / "run/out"
            out.mkdir()
            for name, data in expected.items():
                (out / name).write_bytes(data)

        with patch.object(tr, "_phase", side_effect=phase):
            status, files = tr._result(attempt)
        self.assertEqual((status, files, len(seen)), ("complete", expected, 2))
        routes = tr._publication(attempt, status, files)
        (tr.WORK / "run/out/trace.jsonl").write_bytes(b"changed")
        self.assertEqual(routes[b"/file/trace.jsonl"], (expected["trace.jsonl"], tr._sha(expected["trace.jsonl"])))

    def test_selection_cpu_worker_through_export_and_local_collection(self):
        a, rows = self.selection()
        calls = []
        callbacks = []

        @dataclass
        class FixtureParity:
            fixture: bool = True

        def branches(state, questions, *, window):
            labels = tuple(questions["decision"]["criteria"])
            calls.append((state, labels))
            self.assertEqual(window, (8, 12))
            # Always favor the first POSITION. Only a second native call on
            # genuinely reversed criteria will average to a tie after remapping.
            def observed(kind):
                callbacks.append((state, labels, kind))
                return 1
            branch = SimpleNamespace(labels=labels, layout=SimpleNamespace(input_ids=(1, 2)),
                                     prelude=lambda: observed("prelude"), block=lambda h, s: observed("block"),
                                     coda=lambda h, s: observed("coda"),
                                     probabilities=lambda _: (0.75, 0.25), fresh_state=object,
                                     inference_context=contextlib.nullcontext)
            return {"decision": branch}

        native = SimpleNamespace(branches=branches, verify_parity=lambda **kw: FixtureParity())
        prop = SimpleNamespace(name="Fixture A100", major=8, minor=0, total_memory=100)
        cuda = SimpleNamespace(get_device_properties=lambda i: prop, device_count=lambda: 1, is_bf16_supported=lambda: True,
                               Event=lambda **kw: SimpleNamespace(record=lambda: None, elapsed_time=lambda end: 1),
                               synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                               **{k: lambda: 1 for k in ("memory_allocated", "memory_reserved",
                                                         "max_memory_allocated", "max_memory_reserved")})

        def phase(a, argv, *, cwd, installed):
            if installed:
                self.assertNotIn(str(tr.STAGED / "inputs/discovery.jsonl"), argv)
                self.assertIn(str(tr.STAGED / "inputs/selection.jsonl"), argv)
                self.assertEqual(argv[argv.index("--nomination-sha256") + 1], a.permit_sha256)
                worker.main(argv[argv.index("joey_research.worker") + 1:])
            else:
                self.assertIn("pip", argv)  # Never install, even fixture wheels.

        with contextlib.ExitStack() as s:
            s.enter_context(patch.dict(sys.modules, torch=SimpleNamespace(cuda=cuda)))
            s.enter_context(patch("joey_research.native.NativeJoey.load", return_value=native))
            s.enter_context(patch.dict(os.environ, {k: "1" for k in (
                "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")}, clear=True))
            for module, name in ((signal, "signal"), (signal, "setitimer"), (os, "umask")):
                s.enter_context(patch.object(module, name))
            s.enter_context(patch.object(tr, "_phase", side_effect=phase))
            status, files = tr._result(a)
        routes = tr._publication(a, status, files)
        dest = self.root / "promotion-result"
        exits = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            server = threading.Thread(target=lambda: exits.append(tr._serve(listener, a, routes, status)), daemon=True)
            server.start()
            receipt = tr.collect(listener.getsockname()[1], dest, mode="selection", nonce=a.nonce,
                                 admit_before=a.admit_before, hard_stop=a.hard_stop, packet_sha256=a.packet_sha256,
                                 requirements_sha256=a.requirements_sha256, permit_sha256=a.permit_sha256)
            server.join(2)
        self.assertEqual(exits, [0])
        self.assertEqual((receipt["mode"], receipt["status"], receipt["permit_sha256"]), ("selection", "complete", a.permit_sha256))
        preds, traces = ([json.loads(line) for line in (dest / name).read_bytes().splitlines()]
                         for name in ("predictions.jsonl", "trace.jsonl"))
        selected, forwards = set(), 0
        for row, pred, trace in zip(rows, preds, traces, strict=True):
            observed = [labels for state, labels in calls if state == row["prompt"]]
            n = len(observed)
            self.assertIn(n, (1, 2))
            self.assertEqual(observed, [("opt_0", "opt_1"), ("opt_1", "opt_0")][:n])
            self.assertEqual([(order, kind) for state, order, kind in callbacks if state == row["prompt"]],
                             [(order, kind) for order in observed for kind in ("prelude", "block", "coda")])
            self.assertEqual((trace["ensemble_calls"], trace["forward_calls"], trace["layer_calls"],
                              trace["tokens"], trace["readouts"]), (n, n, 24*n, 2*n, []))
            self.assertEqual(trace["callback_counts"], {"prelude_calls": n, "block_calls": n, "coda_calls": n, "stock_calls": 0})
            self.assertEqual(pred, {"id": row["id"], "input_sha256": input_sha256(row),
                                    "probs": [0.5, 0.5] if n == 2 else [0.75, 0.25]})
            if n == 2:
                selected.add(row["group_id"])
            forwards += n
        self.assertEqual(len(selected), 2)
        self.assertEqual([x["ensemble_calls"] for x in traces], [1 + (r["group_id"] in selected) for r in rows])
        metrics = json.loads((dest / "complete.json").read_bytes())
        self.assertEqual(metrics["compute_spend"], {"groups": 4, "extra_groups": 2, "forward_calls": forwards,
                                                    "layer_calls": forwards*24, "coda_calls": forwards})
        self.assertEqual(metrics["method_sha256"], json.loads(tr._packet_bytes())["method_sha256"])
        # Even a correctly re-indexed manifest cannot substitute another valid
        # prediction stream for the bytes sealed by this selection worker.
        swapped = dict(files)
        swapped["predictions.jsonl"] += b"\n"
        mixed = tr._publication(a, "complete", swapped)
        with patch.object(tr, "_exchange", side_effect=lambda p, a, target, lim: mixed[target.encode()][0]) as get:
            with self.assertRaises(tr.TransportError):
                tr.collect(1, self.root / "spliced", mode=a.mode, nonce=a.nonce, admit_before=a.admit_before,
                           hard_stop=a.hard_stop, packet_sha256=a.packet_sha256,
                           requirements_sha256=a.requirements_sha256, permit_sha256=a.permit_sha256)
        self.assertNotIn("/ack", [c.args[2] for c in get.call_args_list])

    def test_selection_child_rechecks_pins_shard_and_fixed_paths_after_parent(self):
        with patch.dict(os.environ, {k: "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")}), \
                patch.object(signal, "signal"), patch.object(signal, "setitimer"), patch.dict(sys.modules, torch=None):
            for changed in ("permit", "packet", "shard", "discovery_path", "no_pin", "no_mode", "label"):
                with self.subTest(changed=changed):
                    a, _ = self.selection()
                    tr._selection(a)
                    args = SimpleNamespace(selection=True, nomination_sha256=a.permit_sha256, packet=tr.ADMISSION / "packet.json",
                                           packet_sha256=a.packet_sha256, inputs=tr.STAGED / "inputs/selection.jsonl",
                                           model_dir=tr.STAGED / "model", compact_dir=None)
                    if changed == "permit":
                        with (tr.ADMISSION / "nomination.json").open("ab") as f:
                            f.write(b"\n")
                    elif changed == "packet":
                        args.packet.write_bytes(args.packet.read_bytes() + b"\n")
                    elif changed == "shard":
                        args.inputs.write_bytes((tr.STAGED / "inputs/discovery.jsonl").read_bytes())
                    elif changed == "discovery_path":
                        args.inputs = tr.STAGED / "inputs/discovery.jsonl"
                    elif changed == "no_pin":
                        args.nomination_sha256 = None
                    elif changed == "no_mode":
                        args.selection, args.nomination_sha256 = False, None
                    else:
                        # Even a hash-bound synthetic row with an extra field
                        # must not reach native import in either process.
                        raw = tr._encoded(dict(self.row, gold_index=1)) + b"\n"
                        args.inputs.write_bytes(raw)
                        path = tr.ADMISSION / "nomination.json"
                        path.write_bytes(tr._encoded(dict(json.loads(path.read_bytes()), inputs_sha256=tr._sha(raw))))
                        args.nomination_sha256 = tr._sha(path.read_bytes())
                        pkt = dict(json.loads(args.packet.read_bytes()), inputs_sha256=tr._sha(raw),
                                   permit_sha256=args.nomination_sha256)
                        args.packet.write_bytes(tr._encoded(pkt))
                        args.packet_sha256 = tr._sha(args.packet.read_bytes())
                        with self.assertRaises(ValueError):
                            tr._selection(replace(a, packet_sha256=args.packet_sha256, permit_sha256=args.nomination_sha256))
                    with self.assertRaises(ValueError):
                        worker.run(args)

    def test_no_mixed_mode_or_permit_receipts(self):
        a, _ = self.selection()
        discovery = replace(a, mode="discovery", permit_sha256=None)
        with self.assertRaises(ValueError):
            tr._discovery(discovery)
        for bad in (lambda: replace(discovery, permit_sha256=a.permit_sha256),
                    lambda: replace(self.probe, permit_sha256=a.permit_sha256), lambda: replace(a, permit_sha256=None)):
            with self.assertRaises(ValueError):
                bad()
        for old in (discovery, self.probe):
            self.assertNotIn("permit_sha256", old.bindings())  # No new null field on the old wire.
        doc = json.loads(tr._publication(a, "failed", {})[b"/manifest"][0])
        for wrong in (discovery, replace(a, permit_sha256="e"*64)):
            with self.assertRaises(ValueError):
                tr._validate_manifest(doc, wrong)
        # Correct authenticated outer manifest but discovery completion bytes: no durable ACK.
        routes = tr._publication(a, "complete", self.worker_files(a, self.row))
        with patch.object(tr, "_exchange", side_effect=lambda p, a, target, lim: routes[target.encode()][0]) as get:
            with self.assertRaises(tr.TransportError):
                tr.collect(1, self.root / "mixed", mode=a.mode, nonce=a.nonce, admit_before=a.admit_before, hard_stop=a.hard_stop,
                           packet_sha256=a.packet_sha256, requirements_sha256=a.requirements_sha256, permit_sha256=a.permit_sha256)
        self.assertNotIn("/ack", [c.args[2] for c in get.call_args_list])
        self.assertFalse((self.root / "mixed").exists())

    def test_refuses_unbound_input_labels_and_expiry_before_child(self):
        for bad in ("label", "split", "deadline", "input_pin", "packet_pin"):
            with self.subTest(bad=bad):
                changes = {"split": {"split": "final"}, "deadline": {"deadline_utc": self.probe.hard_stop + 1},
                           "input_pin": {"inputs_sha256": "0"*64}}
                row = dict(self.row, gold_index=0) if bad == "label" else self.row
                a = self.discovery(row=row, **changes.get(bad, {}))
                if bad == "packet_pin":
                    a = replace(a, packet_sha256="0"*64)
                with self.assertRaises(ValueError):
                    tr._discovery(a)
        with patch.object(tr, "_phase") as child, self.assertRaises(TimeoutError):
            tr._result(replace(self.probe, admit_before=int(time.time()) - 1))
        child.assert_not_called()
        expired = int(time.time()) - 1
        with patch.dict(os.environ, JOEY_EXPORT_NONCE=self.probe.nonce), patch.object(tr, "_linux") as native:
            self.assertEqual(tr.main(["probe", "--admit-before", str(expired), "--hard-stop", str(expired)]), 124)
        native.assert_not_called()

    def test_packet_projection_stays_under_fixed_mount(self):
        a = self.discovery()
        raw = (tr.ADMISSION / "packet.json").read_bytes()
        projected = tr.ADMISSION / "..2026_01_01.123"
        projected.mkdir()
        (tr.ADMISSION / "packet.json").rename(projected / "packet.json")
        (tr.ADMISSION / "..data").symlink_to(projected.name)
        (tr.ADMISSION / "packet.json").symlink_to("..data/packet.json")
        self.assertEqual(tr._discovery(a)[0]["deadline_utc"], a.hard_stop)
        (tr.ADMISSION / "..data").unlink()
        (tr.ADMISSION / "..data").symlink_to("../staged")
        (tr.STAGED / "packet.json").write_bytes(raw)
        with self.assertRaises(ValueError):
            tr._packet_bytes()

    def test_lock_is_independent_pinned_and_not_a_general_pip_language(self):
        a = self.discovery()
        tr._verify_lock(a)
        with self.assertRaises(ValueError):
            tr._verify_lock(replace(a, requirements_sha256="f"*64))
        for lock in (b"-r ../other\n", b"--find-links=https://invalid.example/\n",
                     b"../sample-0.1-py3-none-any.whl --hash=sha256:" + b"0"*64 + b"\n"):
            with self.assertRaises(ValueError):
                tr._verify_lock(self.discovery(requirements=lock))
        a = self.discovery()
        wheel = next((tr.STAGED / "wheelhouse").iterdir())
        wheel.unlink()
        wheel.symlink_to(tr.STAGED / "requirements-linux.txt")
        with self.assertRaises(ValueError):
            tr._verify_lock(a)

    def test_only_complete_bound_regular_files_in_pinned_directory(self):
        a = self.discovery()
        packet, rows = tr._discovery(a)
        out = tr.WORK / "out"
        out.mkdir()
        files = self.worker_files(a, self.row)
        for name, data in files.items():
            (out / name).write_bytes(data)
        other = self.root / "other"
        other.write_bytes(files["predictions.jsonl"])
        with tr._directory(tr.WORK) as run:
            for bad in ("symlink", "hardlink", "fifo", "missing", "partial", "digest", "empty"):
                with self.subTest(bad=bad):
                    pred = out / "predictions.jsonl"
                    pred.unlink()
                    if bad == "symlink":
                        pred.symlink_to(other)
                    elif bad == "hardlink":
                        os.link(other, pred)
                    elif bad == "fifo":
                        os.mkfifo(pred)
                    elif bad == "partial":
                        pred.write_bytes(other.read_bytes())
                        (out / "trace.partial").touch()
                    elif bad == "empty":
                        pred.touch()
                    elif bad == "digest":
                        pred.write_text(json.dumps(dict(json.loads(other.read_bytes()), input_sha256="1"*64)))
                    with self.assertRaises((ValueError, OSError)):
                        tr._outputs(run, packet, rows, a)
                    pred.unlink(missing_ok=True)
                    (out / "trace.partial").unlink(missing_ok=True)
                    pred.write_bytes(other.read_bytes())
            out.rename(self.root / "different-output")
            out.symlink_to(self.root / "different-output", target_is_directory=True)
            with self.assertRaises(OSError):
                tr._outputs(run, packet, rows, a)

    def test_failure_and_timeout_receipts_are_quiet_not_predictions(self):
        a = self.discovery()
        for error, status in ((RuntimeError("private fixture trace/path"), "failed"), (TimeoutError(), "timeout")):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                    patch.object(tr, "_result", side_effect=error), patch.object(tr.socket, "socket") as sock, \
                    patch.object(tr, "_serve", return_value=2) as serve:
                tr._export(a)
                sock.return_value.__enter__.return_value.bind.assert_called_once_with(("127.0.0.1", 18765))
                routes = serve.call_args.args[2]
                self.assertEqual(set(routes), {b"/manifest"})
                receipt = json.loads(routes[b"/manifest"][0])
                self.assertEqual((receipt["status"], receipt["files"]), (status, {}))
                self.assertNotIn("private", json.dumps(receipt))
                self.assertEqual(out.getvalue() + err.getvalue(), "")
        with patch.dict(os.environ, {"JOEY_EXPORT_NONCE": a.nonce, "PRIVATE_AUTH": "not for a child"}):
            env = tr._child_env(True)
            self.assertNotIn("JOEY_EXPORT_NONCE", env)
            self.assertNotIn("PRIVATE_AUTH", env)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(tr.main(["probe", "--unrecognized", "private/path"]), 2)
            self.assertEqual(err.getvalue(), "")

    def test_expiry_reaps_child_group_and_closes_unacked_export(self):
        events = []
        child = SimpleNamespace(pid=10101, wait=lambda: events.append("reaped leader") or 0)
        # No real process IDs or signals are used in this supervisor fixture.
        with contextlib.ExitStack() as stack:
            for constant in ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT"):
                stack.enter_context(patch.object(os, constant, 1, create=True))
            stack.enter_context(patch.object(os, "waitid", return_value=None, create=True))
            killed = stack.enter_context(patch.object(os, "killpg", side_effect=lambda *args: events.append("killed group")))
            drained = stack.enter_context(patch.object(os, "waitpid", side_effect=ChildProcessError))
            with self.assertRaises(TimeoutError):
                tr._supervise(child, time.time() - 1)
        killed.assert_called_once_with(child.pid, signal.SIGKILL)
        drained.assert_called_once_with(-child.pid, 0)
        self.assertEqual(events, ["killed group", "reaped leader"])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            expired = int(time.time()) - 1
            a = replace(self.probe, admit_before=expired, hard_stop=expired)
            self.assertEqual(tr._serve(listener, a, tr._publication(a, "timeout", {}), "timeout"), 124)

    @unittest.skipUnless(sys.platform == "linux" and os.uname().machine == "x86_64", "verified Linux/amd64 only")
    def test_native_guard_denies_inet_in_fixture_child(self):
        parent = os.getpid()
        pid = os.fork()
        if pid == 0:
            try:
                os.setsid()
                tr._child_guard(tr._linux(), parent, self.probe)
                try:
                    socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
                except OSError as e:
                    os._exit(0 if e.errno == errno.EPERM else 2)
            except BaseException:
                pass
            os._exit(1)
        _, status = os.waitpid(pid, 0)
        self.assertEqual(os.waitstatus_to_exitcode(status), 0)
