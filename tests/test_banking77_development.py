"""Train-derived adapter tests. Every source record and category below is invented."""

from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import banking77_development as d
from joey_research import evaluation as ev


NAMES = [f"fixture_label_{i:02d}" for i in range(77)]


def source(shift=0):
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(["text", "category"])
    for i in range(1500):
        writer.writerow([f"Invented, ticket\n{i}", NAMES[(i + shift) % 77]])
        writer.writerow([f"invented,\u00a0TICKET  {i}", NAMES[(i + shift + 1) % 77]])
    return {"train.csv": out.getvalue().encode(), "categories.json": json.dumps(NAMES).encode()}


def fixture_pins(raw):
    return patch.multiple(d, ROWS=3000, SOURCES={n: (len(v), hashlib.sha256(v).hexdigest())
                                                for n, v in raw.items()})


def lines(raw):
    return [json.loads(s) for s in raw.decode().split("\n") if s]


def small_cases():
    return [{"id": f"fx{i}", "group_id": f"fxg{i}", "task": d.TASK, "prompt": f"invented {i}",
             "choices": list(NAMES)} for i in range(3)]


def native_result(row, probs, declared):
    return {"id": row["id"], "input_sha256": ev.input_sha256(row), "status": "ok",
            "response": {"answers": {"q1": {
                "type": "choice", "choice": f"option_{declared}",
                "probabilities": dict(reversed(list(zip(d.native.OPTIONS, probs))))}}}}


class DevelopmentTests(unittest.TestCase):
    def test_512_hash_stratified_groups_and_unsupervised_membership(self):
        raw, shifted = source(), source(11)
        with fixture_pins(raw):
            a = d._exports(raw)
        with fixture_pins(shifted):
            b = d._exports(shifted)
        self.assertNotEqual(a[d.LABELS], b[d.LABELS])
        for field in (d.INPUTS, d.REQUESTS, d.SPLITS, "coverage.json"):
            self.assertEqual(a[field], b[field])
        coverage = json.loads(a["coverage.json"])
        self.assertEqual(coverage["development_groups_per_stratum"], [32] * 16)
        self.assertEqual((coverage["source_groups"], coverage["development_groups"],
                          coverage["development_rows"]), (1500, 512, 1024))
        membership = lines(a[d.SPLITS])
        all_ids = [k for g in membership for k in g["ids"]]
        selected_ids = {k for g in membership if g["partition"] == "development" for k in g["ids"]}
        self.assertEqual(len(set(all_ids)), 3000)
        self.assertTrue(all(len(g["ids"]) == 2 for g in membership))
        labels = {r["id"]: r for r in lines(a[d.LABELS])}
        self.assertEqual(set(labels), selected_ids)
        inputs, requests = lines(a[d.INPUTS]), lines(a[d.REQUESTS])
        self.assertEqual([r["id"] for r in inputs], sorted(selected_ids))
        for r, req in zip(inputs, requests, strict=True):
            self.assertEqual(set(r), ev.INPUT_FIELDS)
            self.assertEqual(set(req), {"id", "input_sha256", "payload"})
            self.assertEqual(req["id"], r["id"])
            self.assertEqual(req["input_sha256"], labels[r["id"]]["input_sha256"])
            self.assertEqual(req["input_sha256"], ev.input_sha256(r))
            self.assertEqual(req["payload"]["state"], {})
            self.assertEqual(req["payload"]["questions"], {"q1": {
                "type": "choice", "instructions": r["prompt"], "criteria": dict(zip(d.native.OPTIONS, NAMES))
            }})
            self.assertTrue(r["prompt"].startswith("Classify the banking intent of this user request:\n"))
            self.assertEqual(list(req["payload"]["questions"]["q1"]["criteria"]), list(d.native.OPTIONS))
            self.assertEqual(r["choices"], NAMES)
        with self.assertRaisesRegex(ev.EvaluationError, "insufficient independent"):
            d.split_groups([d.BlindRow(i, "same normalized complaint") for i in range(520)])

    def test_different_decision_rules_share_correct_probability_scores(self):
        rows = small_cases()
        labels = [ev.label_row(r, i) for i, r in enumerate(rows)]
        ps = [[.1, .9, 0.], [.8, .2, 0.], [.2, .1, .7]]
        ps = [p + [0.] * 74 for p in ps]
        n = [native_result(r, p, c) for r, p, c in zip(rows, ps, [0, 0, 2])]
        g = [dict(id=r["id"], input_sha256=ev.input_sha256(r), probs=p) for r, p in zip(rows, ps)]
        a = d.score_development(rows, labels, reversed(n), projection="native")
        b = d.score_development(rows, labels, g, projection="research")
        self.assertAlmostEqual(a["observed_class_macro_f1"], 5 / 9)
        self.assertAlmostEqual(b["observed_class_macro_f1"], 1 / 3)
        self.assertEqual(a["declared_choice_not_first_argmax"], 1)
        self.assertEqual(a["choice_accuracy_completed"], 2 / 3)
        self.assertEqual(a["probability_metrics"], b["probability_metrics"])
        self.assertAlmostEqual(a["probability_metrics"]["nll"], -math.log(.1 * .2 * .7) / 3)
        self.assertAlmostEqual(a["probability_metrics"]["brier"], (1.62 + 1.28 + .14) / 3)

    def test_incomplete_or_invalid_predictions_preserve_denominator(self):
        rows = small_cases()
        labels = [ev.label_row(r, i) for i, r in enumerate(rows)]
        # Within native 0.01 tolerance; deliberately outside the stricter research tolerance.
        accepted = native_result(rows[0], [.995] + [0.] * 76, 0)
        unsupported = {"id": "fx1", "input_sha256": ev.input_sha256(rows[1]), "status": "unsupported"}
        s = d.score_development(rows, labels, [accepted, unsupported], projection="native")
        self.assertEqual(s["status_counts"], {"ok": 1, "unsupported": 1, "pending": 1})
        self.assertEqual((s["choice_coverage"], s["groups_with_any_choice"]), (1 / 3, 1))
        self.assertEqual(s["observed_class_macro_f1"], 1.)
        self.assertEqual(s["probability_metrics"]["valid"], 0)
        self.assertIsNone(s["probability_metrics"]["brier"])
        for values in ([accepted, accepted], [dict(accepted, id="BANKING77:test:0")],
                       [dict(accepted, input_sha256="stale")]):
            with self.assertRaises(ev.EvaluationError):
                d.score_development(rows, labels, values, projection="native")

    def test_host_allowlist_no_redirect_and_no_reads_before_pod_check(self):
        with patch.object(d.http.client, "HTTPSConnection") as connect:
            for name in ["test.csv", "../train.csv", "https://example.org/categories.json"]:
                with self.assertRaisesRegex(ev.EvaluationError, "allowlist"):
                    d._fetch(name)
            connect.assert_not_called()
            conn = connect.return_value
            response = conn.getresponse.return_value
            response.status = 302
            with self.assertRaisesRegex(ev.EvaluationError, "302"):
                d._fetch("train.csv")
            response.read.assert_not_called()
            conn.request.assert_called_once_with(
                "GET", d.BASE_PATH + "train.csv", headers={"Accept-Encoding": "identity"})
            conn.close.assert_called_once()
        with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "fixture"}), patch.object(d, "_fetch") as fetch:
            with self.assertRaisesRegex(ev.EvaluationError, "never in a pod"):
                d.stage(Path("must-not-exist"), fetch=True)
            fetch.assert_not_called()

    def test_local_fixture_stage_and_pinned_cli_score(self):
        raw = source()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tree, fixture_pins(raw):
            work = Path(tree)
            local, out = work / "source", work / "output"
            local.mkdir()
            for name, data in raw.items():
                (local / name).write_bytes(data)
            # The test stays in its private scratch directory rather than staging real data.
            with patch.object(d, "_host_only"), patch.object(d, "_new_private_dir", lambda p: p.mkdir(mode=0o700)):
                pin = d.stage(out, source_dir=local)
                with self.assertRaises(FileExistsError):
                    d.stage(out, source_dir=local)
            self.assertEqual(set(p.name for p in (out / "inference").iterdir()), {"inputs.jsonl", "requests.jsonl"})
            self.assertEqual((out / d.LABELS).stat().st_mode & 0o777, 0o600)
            self.assertEqual((out / "custody").stat().st_mode & 0o777, 0o700)
            manifest = json.loads((out / "manifest.json").read_bytes())
            self.assertEqual(set(manifest["source_files"]), {"train.csv", "categories.json"})
            empty = work / "preds"
            empty.write_bytes(b"")
            with patch.object(sys, "argv", ["prog", "score", "--bundle", str(out), "--manifest-sha256", pin,
                                           "--projection", "research", "--predictions", str(empty)]), \
                    redirect_stdout(io.StringIO()) as output:
                d.main()
            report = json.loads(output.getvalue())
            self.assertEqual(report["sampling_coverage"]["development_groups"], 512)
            self.assertEqual(report["status_counts"], {"pending": 1024})
            self.assertIsNone(report["observed_class_macro_f1"])
            (out / d.INPUTS).write_bytes(b"")
            with self.assertRaisesRegex(ev.EvaluationError, "file differs"):
                d._read_bundle(out, pin)
            with self.assertRaisesRegex(ev.EvaluationError, "manifest differs"):
                d._read_bundle(out, "0" * 64)
            with self.assertRaisesRegex(ev.EvaluationError, "SHA-256 differs"):
                d._verify_sources({**raw, "train.csv": raw["train.csv"][::-1]})
            with self.assertRaisesRegex(ev.EvaluationError, "exactly"):
                d._verify_sources({**raw, "test.csv": b"not a dataset"})


if __name__ == "__main__":
    unittest.main()
