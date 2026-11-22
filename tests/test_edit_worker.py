"""Invented generic rows, callbacks and receipts; no study, model, or listener."""
import contextlib
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from joey_research import transport as tr, worker
from joey_research.evaluation import input_sha256
from joey_research.loop import LoopPlan
from joey_research.native import ParityReport
from joey_research.native_compact import ReloadReport


EDIT = worker.EDIT_MODE
BASE = {"window": [8, 12], "reverse": False, "loop": {"k": 2, "method": "zero_extra"}}
COMPACT = {"compact_recipe": "select-20", "compact_manifest_sha256": "3"*64}
MENU = ["Enter at YEW", "Enter at ASH", "Enter at GEM", "Enter at MAP"]
P = [0.1234567890123456, 0.21, 0.3, 0.3665432109876544]
OPS = ("stock", "prelude", "block", "coda")


def lines(records):
    return b"".join(tr._encoded(row) + b"\n" for row in records)


class EditWorkerTest(unittest.TestCase):
    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(TemporaryDirectory(dir="tests", prefix="edit-fixture-"))).resolve()
        for name in ("ADMISSION", "STAGED", "WORK"):
            path = self.root / name.lower()
            path.mkdir()
            stack.enter_context(patch.object(tr, name, path))
        (tr.STAGED / "inputs").mkdir()
        (tr.WORK / "out").mkdir()
        self.saved = self.root / "saved"
        self.saved.mkdir(mode=0o700)
        self.clock = stack.enter_context(patch.object(tr.time, "time", return_value=1000))
        self.rows = [dict(id=key, group_id="toy", task="parcel_entry_route", prompt=f"Parcel {key}?",
                          choices=menu) for key, menu in (("z", MENU), ("a", MENU[::-1]))]

    def release(self, settings=BASE, *, mode=EDIT, rows=None, unused=(), view_changes=None,
                permit_changes=None, packet_changes=None):
        rows = self.rows if rows is None else rows
        shard = lines(rows)
        (tr.STAGED / "inputs" / (mode + ".jsonl")).write_bytes(shard)
        view = dict(schema="joey-label-free-view/v1", split=mode, projection=worker.PHASES[mode],
                    inputs_sha256=tr._sha(shard), split_manifest_sha256="4"*64, rows=len(rows), bytes=len(shard))
        view.update(view_changes or {})
        vp = worker.view_sha256(view)
        methods = {tr._sha(tr._encoded(s)): s for s in (settings, *unused)}
        mid = tr._sha(tr._encoded(settings))
        permit = dict(schema=f"joey-{mode}-permit/v1", split=mode, not_before=990, admit_before=1095, hard_stop=1100,
                      run_sha256="5"*64, approval_sha256="6"*64, view=view, methods=methods)
        permit_bytes = tr._encoded(permit | (permit_changes or {}))
        (tr.ADMISSION / (mode + "-permit.json")).write_bytes(permit_bytes)
        packet = dict(settings, schema=1, split=mode, deadline_utc=1100, inputs_sha256=tr._sha(shard),
                      permit_sha256=tr._sha(permit_bytes), view_sha256=vp, method_sha256=mid)
        raw = tr._encoded(packet | (packet_changes or {}))
        (tr.ADMISSION / "packet.json").write_bytes(raw)
        return tr.Attempt(mode, "7"*64, 1095, 1100, tr._sha(raw), "8"*64, tr._sha(permit_bytes), "5"*64, vp, "6"*64)

    def test_entire_permit_requires_canonical_fixed_arms_and_one_compact_pin(self):
        for changes in (
            {"loop": {}}, {"loop": {"k": 2}}, {"loop": {"k": 1, "method": "zero_extra"}},
            *({"loop": {"k": k, "method": "naive"}} for k in (3, 6, 8)),
            *({"loop": dict(BASE["loop"], readout=v)} for v in ("diagnostic", "final")),
            {"loop": {"k": 2, "method": "damped_extra", "alpha": .5}}, {"loop": dict(BASE["loop"], beta=None)},
            {"reverse": True}, {"window": [4, 8]}, {"policy": {}}, {"probe_margin": .5}, {"norm_overlay": True},
            {"order_ensemble": {"numerator": 1, "denominator": 2, "seed": 1}},
            {"compact_manifest_sha256": "3"*64}, {"compact_recipe": "select-20"},
            dict(COMPACT, compact_recipe="select-20-late"), dict(COMPACT, compact_recipe="merge-16"),
        ):
            a = self.release(unused=[BASE | changes])
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                tr._phase_inputs(a.phase_gate())
        other = dict(BASE | COMPACT, loop={"k": 4, "method": "naive"}, compact_manifest_sha256="9"*64)
        a = self.release(BASE | COMPACT, unused=[other])
        with self.assertRaises(ValueError):
            tr._phase_inputs(a.phase_gate())

    def test_same_method_hash_does_not_substitute_for_a_role_release(self):
        modes = (EDIT, worker.HORIZON_MODE)
        for mode, other in (modes, modes[::-1]):
            rows = self.rows if mode == EDIT else [dict(r, task="fixed_evidence_route") for r in self.rows]
            a = self.release(mode=mode, rows=rows)
            packet, _, _, admitted = tr._phase_inputs(a.phase_gate())
            self.assertEqual((packet["method_sha256"], admitted), (worker.method_sha256(BASE), rows))
            raw, permit = tr._packet_bytes(), tr._admission_bytes(mode + "-permit.json")
            with self.assertRaises(ValueError):
                worker.phase_packet(raw, permit, replace(a.phase_gate(), mode=other))
            for changes in ({"view_changes": {"projection": worker.PHASES[other]}},
                            {"permit_changes": {"schema": f"joey-{other}-permit/v1"}},
                            {"packet_changes": {"split": other}}, {"packet_changes": {"deadline_utc": 1101}}):
                a = self.release(mode=mode, rows=rows, **changes)
                with self.subTest(mode=mode, changes=changes), self.assertRaises(ValueError):
                    tr._phase_inputs(a.phase_gate())

    def test_crossed_period_reuses_rows_not_old_authority_or_exploratory_depth(self):
        original = self.release()
        crossed = self.release(mode=worker.CROSS_MODE)
        _, _, _, rows = tr._phase_inputs(crossed.phase_gate())
        self.assertEqual(rows, self.rows)
        self.assertNotEqual(crossed.permit_sha256, original.permit_sha256)
        with self.assertRaises(ValueError):
            tr._phase_inputs(original.phase_gate())
        four = dict(BASE, loop={"k": 4, "method": "naive"})
        with self.assertRaises(ValueError):
            tr._phase_inputs(self.release(mode=worker.CROSS_MODE, settings=four).phase_gate())
        tr._phase_inputs(self.release(settings=four).phase_gate())

    def test_mounted_paths_and_cutoff_are_checked_before_loading(self):
        a = self.release()
        args = dict(phase=EDIT, selection=False, packet=tr.ADMISSION / "packet.json", packet_sha256=a.packet_sha256,
                    inputs=tr.STAGED / "inputs" / "routing-edits.jsonl", output=tr.WORK / "run/out",
                    model_dir=tr.STAGED / "model", compact_dir=None, phase_permit_sha256=a.permit_sha256,
                    admit_before=a.admit_before, hard_stop=a.hard_stop, final_freeze_sha256=None, **a.phase_gate().evidence())
        flags = dict.fromkeys(("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"), "1")
        with patch.object(worker.os, "environ", flags), patch.object(worker, "_load", side_effect=AssertionError):
            for changes in ({"phase": worker.HORIZON_MODE}, {"inputs": tr.STAGED / "inputs/routing-horizon.jsonl"},
                            {"packet": tr.ADMISSION / "other.json"}, {"compact_dir": tr.STAGED / "compact"},
                            {"output": tr.WORK / "out"}, {"selection": True}, {"approval_sha256": "9"*64}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    worker.run(SimpleNamespace(**(args | changes)))
            # No phase fields at all: refuse the staged shard before even opening the discovery packet.
            alias = self.root / "alias"
            alias.hardlink_to(args["inputs"])
            for path in (args["inputs"], tr.STAGED / "inputs/../inputs/routing-edits.jsonl", alias):
                with self.assertRaisesRegex(ValueError, "phase gate"):
                    worker.run(SimpleNamespace(packet=args["packet"], packet_sha256=a.packet_sha256, inputs=path))
            alias.unlink()
            with patch.object(tr, "_shard_bytes", side_effect=AssertionError), self.assertRaises(ValueError):
                tr._bind_inputs(dict(split=EDIT, deadline_utc=a.hard_stop), a)
            tr._phase_inputs(a.phase_gate())
            self.clock.return_value = a.hard_stop - tr.RESERVE
            with self.assertRaises(TimeoutError):
                worker.run(SimpleNamespace(**args))
            self.clock.return_value = 1000
            args["inputs"].write_bytes(lines(self.rows[::-1]))
            with self.assertRaisesRegex(ValueError, "shard"):
                worker.run(SimpleNamespace(**args))

    def test_exact_blind_task_and_menu_in_both_directions(self):
        for change in ({"task": "fixed_evidence_route"}, {"task": "routing_2hop"}, {"choices": MENU[:3]},
                       {"choices": MENU + ["Enter at INK"]}, {"choices": [*MENU[:3], "Enter at map"]},
                       {"choices": [*MENU[:3], MENU[0]]}, *({key: 1} for key in (
                           "gold_index", "case", "condition", "hop", "seed"))):
            a = self.release(rows=[self.rows[0], self.rows[1] | change])
            with self.subTest(change=change), self.assertRaises(ValueError):
                tr._phase_inputs(a.phase_gate())
        for mode in (worker.HORIZON_MODE, worker.PROBE_MODE, worker.TRANSFER_MODE, "refinement", "calibration"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                tr._phase_inputs(self.release(mode=mode).phase_gate())
        for mode in ("discovery", "selection"):
            shard = lines(self.rows)
            (tr.STAGED / "inputs" / (mode + ".jsonl")).write_bytes(shard)
            attempt = tr.Attempt(mode, "7"*64, 1095, 1100, "8"*64, "9"*64, "a"*64 if mode == "selection" else None)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                tr._bind_inputs(dict(split=mode, deadline_utc=1100, inputs_sha256=tr._sha(shard)), attempt)
            with self.assertRaises(ValueError):
                worker.measure_choice(None, self.rows[0], (8, 12), LoopPlan(), split=mode)
        pkt = tr._encoded(dict(BASE, schema=1, split="discovery", deadline_utc=1100, inputs_sha256=tr._sha(shard)))
        packet_path = tr.ADMISSION / "packet.json"
        packet_path.write_bytes(pkt)
        args = SimpleNamespace(packet=packet_path, packet_sha256=tr._sha(pkt), inputs=tr.STAGED / "inputs/discovery.jsonl")
        flags = dict.fromkeys(("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"), "1")
        with patch.object(worker.os, "environ", flags), patch.object(worker.signal, "signal"), \
                patch.object(worker.signal, "setitimer"), self.assertRaisesRegex(ValueError, "own phase"):
            worker.run(args)

    def fixture(self, settings):
        a = self.release(settings)
        packet, window, plan, rows = tr._phase_inputs(a.phase_gate())
        small = "compact_recipe" in settings
        preds, traces, observed = [], [], []
        for i, row in enumerate(rows):
            probs = P if i == 0 else P[::-1]
            calls = []

            def branches(state, qs, *, window):
                q = qs["decision"]
                self.assertEqual((window, state, list(q["criteria"].items())),
                                 ((8, 12), row["prompt"], [(f"opt_{j}", c) for j, c in enumerate(row["choices"])]))
                return {"decision": SimpleNamespace(labels=tuple(q["criteria"]), layout=SimpleNamespace(input_ids=(1,)),
                        stock=lambda: calls.append("stock"), prelude=lambda: calls.append("prelude"),
                        block=lambda h, s: calls.append("block"), coda=lambda h, s: calls.append("coda"),
                        probabilities=lambda z: probs, inference_context=contextlib.nullcontext, fresh_state=object)}

            native = SimpleNamespace(branches=branches)
            if small:
                native.physical_to_source = (*range(12), *range(16, 24))
            p, t = worker.measure_choice(native, row, window, plan, split=EDIT)
            self.assertEqual(p, dict(id=row["id"], input_sha256=input_sha256(row), probs=probs))
            preds.append(p)
            traces.append(dict(t, wall_ms=2, cuda_event_ms=1, packet_sha256=a.packet_sha256))
            observed.append(calls)
        bound = {r["id"]: tr._Decision(input_sha256(r), 4, r["task"], r["group_id"]) for r in rows}
        metrics = dict(schema=1, batch_size=1, decisions=len(rows), hardware="A100", total_device_bytes=100,
                       peak_allocated_bytes=1, steady_allocated_bytes=1, peak_reserved_bytes=2, steady_reserved_bytes=2,
                       warmup_load_and_parity_s=1, wall_including_persistence_s=1, decisions_per_wall_s=2,
                       warm_latency_ms={"50": 2, "95": 2, "99": 2}, split=EDIT, inputs_sha256=packet["inputs_sha256"],
                       method_sha256=packet["method_sha256"], permit_sha256=a.permit_sha256, packet_sha256=a.packet_sha256,
                       parity=asdict(ParityReport(window, 1, ("sequential",), 0, 0, 0, 0, True, .05, .005, 1e-5)),
                       **a.phase_gate().evidence())
        if small:
            pin = settings["compact_manifest_sha256"]
            metrics.update(physical_layers=20, compact_manifest_sha256=pin,
                           reload_parity=asdict(ReloadReport(pin, native.physical_to_source, 1, 0, 0, True)))
        return a, packet, bound, preds, traces, metrics, observed

    def seal(self, a, preds, traces, metrics):
        files = {"predictions.jsonl": lines(preds), "trace.jsonl": lines(traces)}
        files["complete.json"] = tr._encoded(dict(metrics, result_sha256={n: tr._sha(b) for n, b in files.items()}))
        manifest, receipt = tr._publication(a, "complete", files)[b"/manifest"]
        for name, data in files.items():
            (tr.WORK / "out" / name).write_bytes(data)
        for name, data in (files | {"manifest.json": manifest}).items():
            path = self.saved / name
            path.write_bytes(data)
            path.chmod(0o600)
        return files, receipt

    def test_all_arms_export_and_recover_lossless_four_slot_results(self):
        for k, method, ops in ((1, "naive", ["stock"]), (2, "zero_extra", ["prelude", "block", "coda"]),
                               (2, "naive", ["prelude", "block", "block", "coda"]),
                               (4, "naive", ["prelude", *(["block"] * 4), "coda"])):
            for depth, extra in ((24, {}), (20, COMPACT)):
                with self.subTest(k=k, method=method, depth=depth):
                    a, packet, bound, preds, traces, metrics, observed = self.fixture(
                        dict(BASE, loop={"k": k, "method": method}, **extra))
                    self.assertEqual(observed, [ops, ops])
                    for t in traces:
                        self.assertEqual(t["callback_counts"], {op + "_calls": ops.count(op) for op in OPS})
                        self.assertEqual((t["readouts"], t["layer_calls"]), ([], depth + 4 * max(0, ops.count("block")-1)))
                    files, receipt = self.seal(a, preds, traces, metrics)
                    with tr._directory(tr.WORK) as run:
                        self.assertEqual(tr._outputs(run, packet, bound, a), files)
                    with patch.object(tr.time, "time", return_value=a.hard_stop + 1), \
                            patch.object(tr, "_shard_bytes", side_effect=AssertionError):
                        doc = tr.recover(self.saved, attempt=a, receipt_sha256=receipt)
                        self.assertEqual((doc["mode"], doc["status"]), (EDIT, "complete"))
                        self.assertEqual((self.saved / "predictions.jsonl").read_bytes(), lines(preds))
                        with self.assertRaises(tr.TransportError):
                            tr.recover(self.saved, attempt=replace(a, mode=worker.HORIZON_MODE), receipt_sha256=receipt)

    def test_rehashed_files_still_require_the_admitted_arm_slots_and_pins(self):
        a, packet, bound, preds, traces, metrics, _ = self.fixture(BASE)
        stock = dict(zip((op + "_calls" for op in OPS), (1, 0, 0, 0)))
        deep = worker.method_sha256(dict(BASE, loop={"k": 8, "method": "naive"}))
        with tr._directory(tr.WORK) as run:
            for p, t, m in (
                ({"probs": [0, 0, 1]}, {}, {}), ({"probs": [True, 0, 0, 0]}, {}, {}),
                ({"probs": [1, 0, 0, 1]}, {}, {}), ({"case": 1}, {}, {}), ({}, {"condition": "x"}, {}),
                ({}, {"readouts": [1]}, {}), ({}, {"id": "a"}, {}), ({}, {"packet_sha256": "0"*64}, {}),
                ({}, {"callback_counts": stock}, {}),  # Same 24 layers, but not a split zero-extra call.
                ({}, {"layer_calls": 28}, {}), ({}, {}, {"method_sha256": deep}),
                ({}, {}, {"method_sha256": worker.method_sha256(dict(BASE, loop={"k": 1, "method": "naive"}))}),
                ({}, {}, {"view_sha256": "0"*64}), ({}, {}, {"unsealed": 1}),
            ):
                _, receipt = self.seal(a, [preds[0] | p, preds[1]], [traces[0] | t, traces[1]], metrics | m)
                with self.subTest(p=p, t=t, m=m):
                    with self.assertRaises(ValueError):
                        tr._outputs(run, packet, bound, a)
                    with self.assertRaises(tr.TransportError):
                        tr.recover(self.saved, attempt=a, receipt_sha256=receipt)
            self.seal(a, preds, traces, metrics)
            key = preds[0]["id"]
            for wrong in (dict(reversed(bound.items())), bound | {key: replace(bound[key], task="fixed_evidence_route")},
                          bound | {key: replace(bound[key], digest="0"*64)}):
                with self.assertRaises(ValueError):
                    tr._outputs(run, packet, wrong, a)
            for wrong in ({"view_sha256": "0"*64}, {"reverse": True}, {"loop": {"k": 4, "method": "naive"}}):
                with self.assertRaises(ValueError):
                    tr._outputs(run, packet | wrong, bound, a)

    def test_execution_rejects_wrong_physical_map_or_rendered_slots(self):
        row = self.rows[0]
        branch = SimpleNamespace(labels=("opt_1", "opt_0", "opt_2", "opt_3"))
        native = SimpleNamespace(branches=lambda *a, **kw: {"decision": branch})
        with patch.object(worker, "run_loop", side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError, "option order"):
                worker.measure_choice(native, row, (8, 12), LoopPlan(), split=EDIT)
            branch.labels = ("opt_0", "opt_1", "opt_2", "opt_3")
            for changes in ({"reverse": True}, {"policy": object()}, {"probe_margin": .5}):
                with self.assertRaises(ValueError):
                    worker.measure_choice(native, row, (8, 12), LoopPlan(), split=EDIT, **changes)
            with self.assertRaises(ValueError):
                worker._measure_ensemble(native, row, (8, 12), LoopPlan(), EDIT, second=False)
            native.physical_to_source = tuple(range(20))
            with self.assertRaisesRegex(ValueError, "payload"):
                worker.measure_choice(native, row, (8, 12), LoopPlan(), split=EDIT)
