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
from dataclasses import asdict, dataclass, replace
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

    def phase_attempt(self, mode, *, choices=None, rows=None, settings=None, view_changes=None, permit_changes=None, packet_changes=None):
        a = self.discovery()
        if settings is None:
            settings = {"window": [8, 12], "loop": {"k": 1}, "reverse": False}
            if mode in ("one-probe","routing-horizon","routing-edits"):settings["loop"]["method"]="naive"
        mid = worker.method_sha256(settings)
        public = mode in ("public-development", "public-native")
        if choices is None:
            choices = [f"synthetic option {i}" for i in range(77)] if public else ["A", "B"]
            if mode == "typed-transfer":
                choices = ["Yes", "No"]
            if mode == "one-probe":
                choices = ["North","South","East","West"]
            if mode in ("routing-horizon","routing-edits"):
                choices = ["Enter at "+s for s in ("AAA","BBB","CCC","DDD")]
        row = dict(self.row, id="toy-" + mode, prompt="Invented café case?", choices=choices)
        if mode == "typed-transfer":
            row.update(task="stock-eligibility", prompt="Focus: Fern. Fern opens with 2 units; +4 posted, +1 pending.")
        if mode=="one-probe":row["task"]="routing_2hop"
        if mode=="routing-horizon":row["task"]="fixed_evidence_route"
        if mode=="routing-edits":row["task"]="parcel_entry_route"
        records = [row] if rows is None else rows
        shard = b"".join(tr._encoded(r) + b"\n" for r in records)
        if mode == "public-native":
            q1 = {"type": "choice", "instructions": row["prompt"], "criteria": {f"option_{i}": s for i,s in enumerate(choices)}}
            wrapper = {"id": row["id"], "input_sha256": input_sha256(row), "payload": {"state": {}, "questions": {"q1": q1}}}
            def line(obj):
                return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
            shard = line({"input": row, "native": wrapper})
        (tr.STAGED / "inputs" / (mode + ".jsonl")).write_bytes(shard)
        view = {"schema": "joey-label-free-view/v1", "split": mode, "projection": worker.PHASES[mode],
                "inputs_sha256": tr._sha(shard), "split_manifest_sha256": "e"*64, "rows": len(records), "bytes": len(shard)}
        if public:
            view["menu_sha256"] = tr._sha(tr._encoded(choices))
        if mode == "public-native":
            view.update(research_inputs_sha256=tr._sha(line(row)), native_requests_sha256=tr._sha(line(wrapper)))
        view.update(view_changes or {})
        vp = worker.view_sha256(view)
        release = ({"final_freeze_sha256": "d"*64} if mode == "final" else {"approval_sha256": "c"*64})
        permit = {"schema": f"joey-{mode}-permit/v1", "split": mode, "not_before": a.hard_stop-100, "admit_before": a.admit_before,
                  "hard_stop": a.hard_stop, "run_sha256": "b"*64, **release, "view": view, "methods": {mid: settings}}
        permit_bytes = tr._encoded(dict(permit, **(permit_changes or {})))
        (tr.ADMISSION / (mode + "-permit.json")).write_bytes(permit_bytes)
        arguments = {"schema": 1, "split": mode, "deadline_utc": a.hard_stop, "inputs_sha256": view["inputs_sha256"],
                     "permit_sha256": tr._sha(permit_bytes), "method_sha256": mid, "view_sha256": vp, **settings}
        packet = tr._encoded(dict(arguments, **(packet_changes or {})))
        (tr.ADMISSION / "packet.json").write_bytes(packet)
        return replace(a, mode=mode, packet_sha256=tr._sha(packet), permit_sha256=tr._sha(permit_bytes),
                       view_sha256=vp, run_sha256="b"*64, **release)

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

    def test_horizon_catalog_fences_deep_fixed_repeats_and_original_input_role(self):
        mode="routing-horizon"
        base={"window":[8,12],"loop":{"k":8,"method":"naive"},"reverse":False}
        for small in (False,True):
            m={**base,**({"compact_recipe":"select-20","compact_manifest_sha256":"a"*64} if small else {})}
            a=self.phase_attempt(mode,settings=m);packet,_,plan,rows=tr._phase_inputs(a.phase_gate())
            self.assertEqual((plan.k,rows[0]["task"]),(8,"fixed_evidence_route"))
            metrics={"method_sha256":packet["method_sha256"],**({"compact_manifest_sha256":"a"*64} if small else {})}
            calls=tr._sealed_transfer_calls(metrics,horizon=True);self.assertEqual(calls,(0,1,8,1))
            trace={"id":rows[0]["id"],"tokens":33,"readouts":[],"layer_calls":48 if small else 52,
                   "packet_sha256":a.packet_sha256,"wall_ms":1,"cuda_event_ms":1,
                   "callback_counts":dict(zip(("stock_calls","prelude_calls","block_calls","coda_calls"),calls))}
            tr._fixed_trace(trace,a.packet_sha256,20 if small else 24,calls,typed=False)
            with self.assertRaises(ValueError):tr._fixed_trace(dict(trace,transfer_answer={}),a.packet_sha256,20 if small else 24,calls,typed=False)
            with self.assertRaises(ValueError):tr._sealed_transfer_calls(metrics)
        for changed in (dict(base,reverse=True),dict(base,loop={"method":"naive","k":3}),
                        dict(base,loop={"method":"naive","k":8,"readout":"diagnostic"}),dict(base,window=[4,8])):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                tr._phase_inputs(self.phase_attempt(mode,settings=changed).phase_gate())
        with self.assertRaises(ValueError):
            tr._phase_inputs(self.phase_attempt(mode,choices=["Yes","No","East","West"]).phase_gate())

    def test_phase_permits_are_role_view_release_and_time_specific(self):
        for mode in worker.PHASES:
            with self.subTest(mode=mode):
                a = self.phase_attempt(mode)
                self.assertEqual(tr._phase_inputs(a.phase_gate())[0]["split"], mode)
                wrong = dict(final_freeze_sha256=None) if mode == "final" else dict(approval_sha256=None)
                with self.assertRaises(ValueError):
                    replace(a, **wrong)
                for change in ({"run_sha256": "a"*64}, {"view_sha256": "a"*64}, {"permit_sha256": "a"*64},
                               {"hard_stop": a.hard_stop+1}, {"admit_before": a.admit_before-1}):
                    with self.assertRaises(ValueError):
                        tr._phase_inputs(replace(a, **change).phase_gate())
                with patch.object(time, "time", return_value=a.admit_before), self.assertRaises(TimeoutError):
                    tr._phase_inputs(a.phase_gate())
                with patch.object(time, "time", return_value=a.hard_stop-101), self.assertRaises(TimeoutError):
                    tr._phase_inputs(a.phase_gate())
                with self.assertRaises(ValueError):
                    worker._packet(tr._packet_bytes())
                with self.assertRaises(ValueError):
                    worker.selection_packet(tr._packet_bytes(), tr._admission_bytes(mode+"-permit.json"),
                                            packet_sha256=a.packet_sha256, nomination_sha256=a.permit_sha256)
                other = "final" if mode != "final" else "calibration"
                for changes in ({"packet_changes": {"split": "selection"}}, {"packet_changes": {"reverse": True}},
                                {"view_changes": {"projection": worker.PHASES[other]}},
                                {"view_changes": {"split": other}}, {"view_changes": {"rows": 2}},
                                {"permit_changes": {"schema": f"joey-{other}-permit/v1"}},
                                {"permit_changes": {"methods": {}}}):
                    with self.assertRaises(ValueError):
                        tr._phase_inputs(self.phase_attempt(mode, **changes).phase_gate())
        for args in ({"choices": ["A", "B"]}, {"view_changes": {"menu_sha256": "1"*64}}):
            with self.assertRaises(ValueError):
                tr._phase_inputs(self.phase_attempt("public-development", **args).phase_gate())

    def test_phase_worker_independently_refuses_replaced_shard(self):
        a = self.phase_attempt("calibration")
        tr._phase_inputs(a.phase_gate())
        shard = tr.STAGED / "inputs/calibration.jsonl"
        shard.write_bytes((tr.STAGED / "inputs/discovery.jsonl").read_bytes())
        args = SimpleNamespace(phase=a.mode, selection=False, nomination_sha256=None, packet=tr.ADMISSION / "packet.json",
                               inputs=shard, model_dir=tr.STAGED / "model", compact_dir=None, packet_sha256=a.packet_sha256,
                               phase_permit_sha256=a.permit_sha256, admit_before=a.admit_before, hard_stop=a.hard_stop,
                               run_sha256=a.run_sha256, view_sha256=a.view_sha256, approval_sha256=a.approval_sha256,
                               final_freeze_sha256=None)
        with patch.dict(os.environ, {k: "1" for k in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")}), \
                patch.dict(sys.modules, torch=None):
            with self.assertRaisesRegex(ValueError, "shard"):
                worker.run(args)
            args.inputs = tr.STAGED / "inputs/discovery.jsonl"
            with self.assertRaisesRegex(ValueError, "launch"):
                worker.run(args)

    def test_transfer_refuses_replaced_bytes_wrong_targets_and_bad_rows_before_loading(self):
        mode = "typed-transfer"
        def arguments(a):
            return SimpleNamespace(phase=mode, selection=False, nomination_sha256=None,
                packet=tr.ADMISSION / "packet.json", inputs=tr.STAGED / "inputs/typed-transfer.jsonl",
                model_dir=tr.STAGED / "model", output=tr.WORK / "run/out", compact_dir=None,
                packet_sha256=a.packet_sha256, phase_permit_sha256=a.permit_sha256,
                admit_before=a.admit_before, hard_stop=a.hard_stop, final_freeze_sha256=None, **a.phase_gate().evidence())
        model_flags = {k: "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}
        with patch.dict(os.environ, model_flags, clear=True), patch.dict(sys.modules, torch=None):
            self.assertEqual(tr._SHARDS[mode], "typed-transfer.jsonl")
            for target in ("packet", "permit", "shard"):
                a = self.phase_attempt(mode)
                args = arguments(a)
                _, _, _, rows = tr._phase_inputs(a.phase_gate())  # Parent accepted the original, not the replacement.
                path = {"packet": args.packet, "permit": tr.ADMISSION / (mode + "-permit.json"), "shard": args.inputs}[target]
                path.write_bytes(path.read_bytes() + b" ")
                with self.subTest(replaced=target), self.assertRaises(ValueError):
                    worker.run(args)
            a = self.phase_attempt(mode)
            for change in ({"inputs": tr.STAGED / "inputs/discovery.jsonl"}, {"packet": tr.ADMISSION / "other"},
                           {"output": tr.WORK / "other"}, {"model_dir": self.root}, {"compact_dir": tr.STAGED / "compact"},
                           {"selection": True}, {"nomination_sha256": "0"*64}, {"phase": None}, {"phase": "refinement"},
                           {"hard_stop": a.hard_stop + 1}, {"admit_before": a.admit_before - 1},
                           {"run_sha256": "0"*64}, {"approval_sha256": "0"*64}, {"view_sha256": "0"*64}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    worker.run(SimpleNamespace(**(vars(arguments(a)) | change)))
            for other in ("public-native", "calibration", "refinement", "final"):
                # Even with the right hashes and a role-appropriate release, the documents are role-specific.
                gate = replace(a.phase_gate(), mode=other, **(
                    {"final_freeze_sha256": a.approval_sha256, "approval_sha256": None} if other == "final" else {}))
                with self.assertRaises(ValueError):
                    worker.phase_packet(tr._packet_bytes(), tr._admission_bytes(mode + "-permit.json"), gate)
            for extra in ("menu_sha256", "native_requests_sha256", "criteria", "tasks"):
                a = self.phase_attempt(mode, view_changes={extra: "e"*64})
                with self.assertRaises(ValueError):
                    tr._phase_inputs(a.phase_gate())
            valid, = rows
            for change in ({"task": "unknown"}, {"answer": 0}, {"native": {}}, {"prompt": {}}, {"choices": ["No", "Yes"]},
                           {"task": "stock-level", "choices": ["0", "1", "2", "3"]},
                           {"task": "stock-choice", "choices": ["A", "B", "C", "D", "E"]},
                           {"task": "stock-choice", "choices": ["A", "B", "C", "No unique depot"]},
                           {"task": "stock-choice", "choices": ["A", "B", "C", "C ", "No unique depot"]}):
                a = self.phase_attempt(mode, rows=[valid, dict(valid, **change, id="second")])
                with self.subTest(change=change):
                    with self.assertRaises(ValueError):
                        tr._phase_inputs(a.phase_gate())
                    with self.assertRaises(ValueError):
                        worker.run(arguments(a))

    def transfer_fixture(self):
        """In-process native callbacks -> real loop -> export files. No child, runtime install or listener."""
        from joey_research.native import ParityReport
        from joey_research.native_compact import ReloadReport
        choices = ["Mauve", "Fern", "No unique depot", "Teal", "Lime"]
        levels = ["Empty (0 units)", "Low (1 to 3 units)", "Medium (4 to 6 units)", "High (7 or more units)"]
        states = ("Focus: Fern. Opening units: Fern 1, Mauve 2, Lime 3, Teal 1. Fern +7 posted; Teal +2 pending.",
                  "Focus: Lime. Opening units: Fern 6, Mauve 4, Lime 2, Teal 7. Lime −1 posted; Mauve +3 posted.")
        rows = [dict(id=f"{i}:{task}", group_id=f"toy:{i}", prompt=prompt, task=task, choices=menu)
                for i, prompt in enumerate(states)
                for task, menu in (("stock-eligibility", ["Yes", "No"]), ("stock-level", levels),
                                   ("stock-choice", choices if i else choices[::-1]))]
        settings = {"window": [8, 12], "loop": {"k": 4}, "reverse": False,
                    "compact_recipe": "select-20", "compact_manifest_sha256": "9"*64}
        a = self.phase_attempt("typed-transfer", rows=rows, settings=settings)
        packet, window, plan, admitted = tr._phase_inputs(a.phase_gate())
        self.assertEqual(admitted, rows)
        kept = (*range(12), *range(16, 24))
        def branches(state, qs, *, window):
            kind = qs["audit"]["type"]
            labels, p = {"noul": (("yes", "no"), (.12345671, .87654329)),
                         "score": (("0", "1", "2", "3"), (.33333349, .33333349, .33333302, 0.)),
                         "choice": (tuple(f"option_{i}" for i in range(5)), (.49999971, .2, .2, .1, .00000029))}[kind]
            public = {"type": kind}
            if kind == "noul":
                public[kind] = round(p[0], 6)
            else:
                public[kind] = "option_3" if kind == "choice" else round(sum(i*v for i, v in enumerate(p)), 6)
                public["probabilities"] = {k: round(v, 6) for k, v in zip(labels, p, strict=True)}
            return {"audit": SimpleNamespace(labels=labels, layout=SimpleNamespace(input_ids=(1, 2)),
                inference_context=contextlib.nullcontext, prelude=lambda: 0, block=lambda h, s: h+1, coda=lambda h, s: h,
                fresh_state=object, probabilities=lambda z: p, decode=lambda z: public)}
        native = SimpleNamespace(branches=branches, physical_to_source=kept)
        predictions, traces = [], []
        for i, r in enumerate(admitted):
            p, t = worker.measure_choice(native, r, window, plan, split=a.mode)
            predictions.append(p)
            traces.append(dict(t, packet_sha256=a.packet_sha256, wall_ms=i+1, cuda_event_ms=0.25))
        bound = {r["id"]: tr._Decision(input_sha256(r), len(r["choices"]), r["task"], r["group_id"]) for r in admitted}
        metrics = dict(json.loads(self.worker_files(a, rows[0])["complete.json"]), decisions=len(bound),
            split=a.mode, inputs_sha256=packet["inputs_sha256"], permit_sha256=a.permit_sha256,
            method_sha256=packet["method_sha256"], warm_latency_ms={"50": 3, "95": 6, "99": 6},
            parity=asdict(ParityReport(window, 5, ("sequential", "sequential"), 0., 0., 0., 0., True, .05, .005, 1e-5)),
            physical_layers=20, compact_manifest_sha256=settings["compact_manifest_sha256"],
            reload_parity=asdict(ReloadReport(settings["compact_manifest_sha256"], kept, 5, 0., 0., True)),
            **{k: 1 for k in ("peak_allocated_bytes", "peak_reserved_bytes", "steady_allocated_bytes", "steady_reserved_bytes")},
            **a.phase_gate().evidence())
        (tr.WORK / "out").mkdir()
        return a, packet, bound, predictions, traces, metrics

    @staticmethod
    def sealed_outputs(predictions, traces, metrics):
        files = {name: b"".join(tr._encoded(row) + b"\n" for row in rows) for name, rows in (
            ("predictions.jsonl", predictions), ("trace.jsonl", traces))}
        files["complete.json"] = tr._encoded(dict(metrics, result_sha256={n: tr._sha(data) for n, data in files.items()}))
        for name, data in files.items():
            (tr.WORK / "out" / name).write_bytes(data)
        return files

    def test_transfer_export_collection_and_offline_recovery_without_sockets(self):
        a, packet, bound, preds, traces, metrics = self.transfer_fixture()
        self.assertEqual([t["transfer_answer"]["type"] for t in traces], ["noul", "score", "choice"]*2)
        self.assertEqual([t["transfer_answer"]["choice"] for t in traces if t["transfer_answer"]["type"] == "choice"],
                         ["option_3", "option_3"])
        self.assertTrue(all(t["layer_calls"] == 32 and t["callback_counts"]["block_calls"] == 4 for t in traces))
        files = self.sealed_outputs(preds, traces, metrics)
        with tr._directory(tr.WORK) as run:
            self.assertEqual(tr._outputs(run, packet, bound, a), files)
        routes = tr._publication(a, "complete", files)
        dest = self.root / "collected"
        def exchange(port, attempt, target, limit, **kw):
            if target == "/ack":
                self.assertEqual(json.loads(kw["ack"])["receipt_sha256"], tr._sha((dest / "manifest.json").read_bytes()))
                raise ConnectionError("invented ACK loss")
            return routes[target.encode()][0]
        with patch.object(tr, "_exchange", side_effect=exchange):
            pin = tr.pin_receipt(1, a)
            with self.assertRaises(tr.TransportError):
                tr.collect(1, dest, **asdict(a), receipt_sha256=pin)
        with patch.object(tr, "_exchange", side_effect=AssertionError), patch.object(time, "time", return_value=a.hard_stop + 1):
            self.assertEqual(tr.recover(dest, attempt=a, receipt_sha256=pin)["status"], "complete")
            self.assertEqual({p.name for p in dest.iterdir()}, {"manifest.json", "complete.json", "predictions.jsonl", "trace.jsonl"})
            for wrong in (replace(a, mode="calibration"), replace(a, view_sha256="0"*64), replace(a, hard_stop=a.hard_stop+1)):
                with self.assertRaises(tr.TransportError):
                    tr.recover(dest, attempt=wrong, receipt_sha256=pin)
            (dest / "trace.jsonl").write_bytes(files["trace.jsonl"] + b"\n")
            with self.assertRaises(tr.TransportError):
                tr.recover(dest, attempt=a, receipt_sha256=pin)

    def test_transfer_resealed_outputs_still_need_the_right_primitive_vector_and_cost(self):
        a, packet, bound, preds, traces, metrics = self.transfer_fixture()
        with tr._directory(tr.WORK) as run:
            def refused(p=preds, t=traces, m=metrics, rows=bound):
                self.sealed_outputs(p, t, m)
                with self.assertRaises(ValueError):
                    tr._outputs(run, packet, rows, a)
            n, s, c = (t["transfer_answer"] for t in traces[:3])
            for i, changes in (
                (0, {"request": "text"}), (0, {"native_answer": n}), (0, {"transfer_answer": s}),
                (0, {"transfer_answer": dict(n, probabilities={"yes": 1})}), (0, {"transfer_answer": dict(n, noul=True)}),
                (0, {"transfer_answer": dict(n, noul=.123456)}),  # Within 1e-5 but outside rounding allowance.
                (1, {"transfer_answer": dict(s, score=2)}), (1, {"transfer_answer": dict(s, probabilities={"Low": 1})}),
                (2, {"transfer_answer": dict(c, choice="Mauve")}), (2, {"transfer_answer": dict(c, debug="prompt")}),
                (2, {"transfer_answer": dict(c, probabilities=dict(c["probabilities"], option_0=.4, option_4=.1))}),
                (0, {"tokens": True}), (0, {"tokens": 16385}), (1, {"wall_ms": -1}), (2, {"cuda_event_ms": "slow"}),
                (0, {"layer_calls": 36}), (1, {"callback_counts": dict(traces[1]["callback_counts"], coda_calls=True)}),
                (0, {"layer_calls": 24, "callback_counts": dict(traces[0]["callback_counts"], block_calls=2)}),
                (0, {"packet_sha256": "0"*64}), (0, {"readouts": [0]}),
            ):
                with self.subTest(index=i, fields=changes):
                    refused(t=[*traces[:i], dict(traces[i], **changes), *traces[i+1:]])
            for changes in ({"probs": [.1, .8]}, {"probs": [True, False]}, {"probs": [-1, 2]}, {"probs": [.5, .3, .2]},
                            {"input_sha256": "0"*64}, {"prompt": "text"}):
                with self.subTest(pred=changes):
                    refused(p=[dict(preds[0], **changes), *preds[1:]])
            key = preds[0]["id"]
            refused(rows=bound | {key: replace(bound[key], task="stock-choice")})
            for changes in ({"split": "public-native"}, {"note": "raw request"}, {"warm_latency_ms": {"50": 4, "95": 6, "99": 6}},
                            {"parity": dict(metrics["parity"], fork_modes=["text"])}, {"physical_layers": 24},
                            {"reload_parity": dict(metrics["reload_parity"], physical_to_source=list(range(20)))}):
                with self.subTest(metrics=changes):
                    refused(m=dict(metrics, **changes))
            wrong_budget = [dict(t, callback_counts=dict(t["callback_counts"], block_calls=2), layer_calls=24) for t in traces]
            for wrong_traces in ([dict(traces[0], native_answer=n), *traces[1:]], wrong_budget):
                files = self.sealed_outputs(preds, wrong_traces, metrics)
                routes = tr._publication(a, "complete", files)
                with patch.object(tr, "_exchange", side_effect=lambda p, att, target, lim: routes[target.encode()][0]) as get:
                    with self.assertRaises(tr.TransportError):
                        tr.collect(1, self.root / "rejected", **asdict(a), receipt_sha256=routes[b"/manifest"][1])
                    self.assertNotIn("/ack", [call.args[2] for call in get.call_args_list])
            # A legacy envelope must not acquire a new nested answer channel either.
            old = self.discovery()
            old_packet, old_rows = tr._discovery(old)
            legacy = self.worker_files(old, self.row)
            legacy["trace.jsonl"] = tr._encoded(dict(json.loads(legacy["trace.jsonl"]), transfer_answer=n)) + b"\n"
            for name, data in legacy.items():
                (tr.WORK / "out" / name).write_bytes(data)
            with self.assertRaises(ValueError):
                tr._outputs(run, old_packet, old_rows, old)

    def test_public_native_pairs_bind_inner_byte_streams_and_ordered_q1(self):
        a = self.phase_attempt("public-native")
        raw = tr._shard_bytes(a.mode)
        view = json.loads(tr._admission_bytes(a.mode + "-permit.json"))["view"]
        line = lambda obj: (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        self.assertEqual(worker.phase_rows(raw, view), [json.loads(raw)["input"]])
        # Inner ordering (and non-ASCII UTF-8) is byte-significant, not just the JSON value.
        for part, pin in (("input","research_inputs_sha256"), ("native","native_requests_sha256")):
            obj = json.loads(raw)
            obj[part] = dict(reversed(obj[part].items()))
            cooked = line(obj)
            v = dict(view, bytes=len(cooked), inputs_sha256=tr._sha(cooked))
            with self.assertRaisesRegex(ValueError, "original JSONL digests"):
                worker.phase_rows(cooked, v)
            v[pin] = tr._sha(line(obj[part]))
            self.assertEqual(worker.phase_rows(cooked, v), [obj["input"]])
        for bad in ("order", "description", "prompt", "id", "digest", "state", "other_question",
                    "type", "wrapper_extra", "pair_extra", "q1_extra", "payload_extra", "missing_option"):
            with self.subTest(bad=bad):
                pair = json.loads(raw)
                wrapper = pair["native"]
                payload = wrapper["payload"]
                q1 = payload["questions"]["q1"]
                if bad == "order": q1["criteria"] = dict(sorted(q1["criteria"].items()))
                elif bad == "description": q1["criteria"]["option_3"] = "different"
                elif bad == "prompt": q1["instructions"] = "different"
                elif bad == "id": wrapper["id"] = "different"
                elif bad == "digest": wrapper["input_sha256"] = "d"*64
                elif bad == "state": payload["state"] = {"unexpected": True}
                elif bad == "other_question": payload["questions"]["q2"] = q1
                elif bad == "type": q1["type"] = "score"
                elif bad == "wrapper_extra": wrapper["extra"] = True
                elif bad == "pair_extra": pair["extra"] = True
                elif bad == "q1_extra": q1["extra"] = True
                elif bad == "payload_extra": payload["model"] = "different"
                else: del q1["criteria"]["option_76"]
                cooked = line(pair)
                v = dict(view, bytes=len(cooked), inputs_sha256=tr._sha(cooked), native_requests_sha256=tr._sha(line(wrapper)))
                with self.assertRaises(ValueError):
                    worker.phase_rows(cooked, v)

    def test_native_trace_refuses_missing_spliced_or_untyped_fields(self):
        pin = "d"*64
        answer = {"type": "choice", "choice": "option_76", "probabilities": dict.fromkeys(worker.NATIVE_OPTIONS,.013)}
        trace = {"id": "toy", "tokens": 100, "layer_calls": 24, "readouts": [], "wall_ms": 1, "cuda_event_ms": .2,
                 "packet_sha256": pin, "native_answer": answer,
                 "callback_counts": {"stock_calls":1, "prelude_calls":0, "block_calls":0, "coda_calls":0}}
        tr._native_trace(trace, pin)  # rounded native mass 1.001 is valid even though >1e-6 from one.
        for bad in (dict(trace, packet_sha256="e"*64), dict(trace, payload={}), dict(trace, readouts=[{}]),
                    {k:v for k,v in trace.items() if k != "native_answer"},
                    *(dict(trace, native_answer=dict(answer, **update)) for update in (
                        {"type":"score"}, {"choice":"option_77"}, {"extra":"debug"}, {"probabilities":{"option_0":1}},
                        *({"probabilities":dict(answer["probabilities"], option_0=x)} for x in (
                            True, -0.1, float("nan"), float("inf"), 0.9)),
                        {"probabilities":dict(answer["probabilities"], extra=0)}))):
            with self.assertRaises(ValueError):
                tr._native_trace(bad, pin)

    def test_cpu_phase_export_and_verified_recovery_after_lost_ack(self):
        from joey_research.native import ParityReport
        observations = []
        events = []
        def branches(state, qs, *, window):
            qid, = qs
            labels = ("yes", "no") if qid == "audit" else tuple(qs[qid]["criteria"])
            observations.append(len(labels))
            if qid == "q1":
                self.assertEqual(state, {})
                self.assertEqual(qs[qid]["instructions"], "Invented café case?")
                self.assertEqual(list(qs[qid]["criteria"].items()), [(f"option_{i}", f"synthetic option {i}") for i in range(77)])
            logits = object()
            def probs(arg):
                self.assertIs(arg, logits)
                events.append("vector")
                return [1/len(labels)] * len(labels)
            def decode(arg):
                self.assertIs(arg, logits)
                events.append("decode")
                if qid == "audit":
                    return {"type": "noul", "noul": 0.5}
                # Deliberately not the first argmax; native output need not be the research vector.
                return {"type": "choice", "choice": labels[-1], "probabilities": dict.fromkeys(labels, 0.013)}
            return {qid: SimpleNamespace(labels=labels, layout=SimpleNamespace(input_ids=(1,)),
                    inference_context=contextlib.nullcontext, stock=lambda: events.append("forward") or logits,
                    probabilities=probs, decode=decode)}
        native = SimpleNamespace(branches=branches, verify_parity=lambda **kw: ParityReport(
            (8,12), 1, ("sequential",), 0., 0., 0., 0., True, .05, .005, 1e-5))
        cuda = SimpleNamespace(device_count=lambda: 1, is_bf16_supported=lambda: True, get_device_properties=lambda _: (
            SimpleNamespace(name="fixture A100", major=8, minor=0, total_memory=42)), synchronize=lambda: None,
            reset_peak_memory_stats=lambda: None, Event=lambda **kw: SimpleNamespace(record=lambda: None, elapsed_time=lambda _: 1),
            **{k: lambda: 1 for k in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved")})
        def phase(a, argv, *, cwd, installed):
            if installed:
                worker.main(argv[argv.index("joey_research.worker")+1:])
            else:
                self.assertIn("pip", argv)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, torch=SimpleNamespace(cuda=cuda)))
            stack.enter_context(patch("joey_research.native.NativeJoey.load", return_value=native))
            stack.enter_context(patch.dict(os.environ, {k: "1" for k in (
                "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")}, clear=True))
            for module, name in ((signal, "signal"), (signal, "setitimer"), (os, "umask")):
                stack.enter_context(patch.object(module, name))
            stack.enter_context(patch.object(tr, "_phase", side_effect=phase))
            for mode in worker.PHASES:
                with self.subTest(mode=mode):
                    a = self.phase_attempt(mode)
                    status, files = tr._result(a)
                    self.assertEqual(status, "complete")
                    self.assertEqual(observations.pop(), 77 if mode in ("public-development", "public-native") else 4 if mode in ("one-probe","routing-horizon","routing-edits") else 2)
                    self.assertFalse(observations)  # No extra real-row warmup.
                    if mode == "public-native":
                        trace = json.loads(files["trace.jsonl"])
                        self.assertEqual(trace["native_answer"]["choice"], "option_76")
                        self.assertEqual(trace["native_answer"]["probabilities"]["option_0"], .013)
                        self.assertEqual(trace["layer_calls"], 24)
                        self.assertEqual(trace["callback_counts"]["stock_calls"], 1)
                        self.assertEqual(json.loads(files["predictions.jsonl"])["probs"], [1/77]*77)
                        self.assertEqual(events, ["forward", "vector", "decode"])
                        packet, _, _, inputs = tr._phase_inputs(a.phase_gate())
                        bound = {r["id"]: tr._Decision(input_sha256(r), 77, r["task"], r["group_id"]) for r in inputs}
                        out = tr.WORK / "run/out"
                        with tr._directory(out.parent) as run:
                            for bad in (dict(trace, request_body={}), dict(trace, layer_calls=32),
                                        dict(trace, native_answer=dict(trace["native_answer"], choice="foreign"))):
                                # Even resealing these bytes under the right packet doesn't make them exportable.
                                changed = tr._encoded(bad) + b"\n"
                                m = json.loads(files["complete.json"])
                                m["result_sha256"]["trace.jsonl"] = tr._sha(changed)
                                (out / "trace.jsonl").write_bytes(changed)
                                (out / "complete.json").write_bytes(tr._encoded(m))
                                with self.assertRaises(ValueError):
                                    tr._outputs(run, packet, bound, a)
                        (out / "trace.jsonl").write_bytes(files["trace.jsonl"])
                        (out / "complete.json").write_bytes(files["complete.json"])
                    elif mode == "typed-transfer":
                        self.assertEqual(json.loads(files["trace.jsonl"])["transfer_answer"], {"type": "noul", "noul": 0.5})
                        self.assertEqual(events, ["forward", "vector", "decode"])
                    else:
                        self.assertEqual(events, ["forward", "vector"])
                        self.assertNotIn("native_answer", json.loads(files["trace.jsonl"]))
                    events.clear()
                    routes = tr._publication(a, status, files)
                    args = dict(mode=a.mode, nonce=a.nonce, admit_before=a.admit_before, hard_stop=a.hard_stop,
                                packet_sha256=a.packet_sha256, requirements_sha256=a.requirements_sha256,
                                permit_sha256=a.permit_sha256, **a.phase_gate().evidence())
                    saved = self.root / ("durable-" + mode)
                    exits = []
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                        listener.bind(("127.0.0.1", 0))
                        listener.listen()
                        port = listener.getsockname()[1]
                        server = threading.Thread(target=lambda: exits.append(tr._serve(listener, a, routes, status)),
                                                  daemon=True)
                        server.start()
                        journaled_digest = tr.pin_receipt(port, a)
                        exchange = tr._exchange
                        def lose_ack(p, att, target, lim, **kw):
                            if target == "/ack":
                                raise ConnectionError("fixture")
                            return exchange(p, att, target, lim, **kw)
                        with patch.object(tr, "_exchange", side_effect=lose_ack), self.assertRaises(tr.TransportError):
                            tr.collect(port, saved, **args, receipt_sha256=journaled_digest)
                        with self.assertRaises(tr.TransportError):
                            tr.recover(saved, attempt=a, receipt_sha256="0"*64)
                        for wrong in (replace(a, nonce="0"*64), replace(a, run_sha256="0"*64),
                                      replace(a, view_sha256="0"*64), replace(a, permit_sha256="0"*64)):
                            with self.assertRaises(tr.TransportError):
                                tr.recover(saved, attempt=wrong, receipt_sha256=journaled_digest)
                        pred = saved / "predictions.jsonl"
                        original = pred.read_bytes()
                        pred.write_bytes(original + b"\n")
                        with self.assertRaises(tr.TransportError):
                            tr.recover(saved, attempt=a, receipt_sha256=journaled_digest, port=port)
                        pred.write_bytes(original)
                        # The cross-mode source result must not be accepted under a different authorized attempt.
                        other = self.phase_attempt("calibration" if mode != "calibration" else "public-development")
                        with self.assertRaises(ValueError):
                            tr._completion_binding(json.loads(files["complete.json"]), other,
                                {n: tr._sha(files[n]) for n in ("predictions.jsonl", "trace.jsonl")})
                        doc = tr.recover(saved, attempt=a, receipt_sha256=journaled_digest, port=port)
                        self.assertEqual(doc["mode"], mode)
                        server.join(2)
                    self.assertEqual(exits, [0])
                    with patch.object(time, "time", return_value=a.hard_stop+1):
                        self.assertEqual(tr.recover(saved, attempt=a, receipt_sha256=journaled_digest), doc)
                        with self.assertRaises(tr.TransportError):
                            tr.recover(saved, attempt=a, receipt_sha256=journaled_digest, port=port)
                    for dirname in ("site", "run", "tmp"):
                        shutil.rmtree(tr.WORK / dirname)

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
