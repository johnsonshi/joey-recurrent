"""Invented schema fixtures only; never opens or downloads an upstream split."""

import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import decision_index_banking77 as b
from joey_research import evaluation as ev


NAMES = tuple(f"fabricated_type_{i:02d}" for i in range(77))


def case(i, text=None):
    return b.project_blind({"source_index": i, "text": text or f"fabricated utterance {i}"}, NAMES)


def label(c, y):
    return ev.label_row(c.research_input(), y)


def result(c, selected, *, peak=None, status="ok"):
    record = {"run_id": c.run_id, "group_id": c.source_id, "catalog_id": 4,
              "status": status, "payload_sha256": c.native_digest()}
    if status == "ok":
        peak = selected if peak is None else peak
        record["response"] = {"answers": {"q1": {"type": "choice", "choice": f"option_{selected}",
                                                 "probabilities": {f"option_{i}": float(i == peak)
                                                                   for i in range(76, -1, -1)}}}}
    return record


class BankingTests(unittest.TestCase):
    def test_exact_native_request_shape_order_and_blindness(self):
        c = case(13, "Fixture line one.\nLine two.")
        native = c.native_payload()
        self.assertEqual(native["state"], {})
        self.assertEqual(list(native["questions"]), ["q1"])
        q = native["questions"]["q1"]
        self.assertEqual(q["instructions"], "Classify the banking intent of this user request:\nFixture line one.\nLine two.")
        self.assertEqual(list(q["criteria"].items()), [(f"option_{i}", x) for i, x in enumerate(NAMES)])
        self.assertEqual((c.source_id, c.run_id), ("BANKING77:test:13", "4:BANKING77:BANKING77:test:13"))
        payload = {"model": "jev-1.13.0", "state": {}, "questions": {"q1": {
            "type": "choice", "instructions": q["instructions"], "criteria": dict(q["criteria"])}}}
        expected_hash = hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                                   separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(c.native_digest(), expected_hash)
        duplicate = case(12, "Fixture line one.  Line two.")
        self.assertNotEqual(c.source_id, duplicate.source_id)
        self.assertEqual(c.research_input()["group_id"], duplicate.research_input()["group_id"])
        for record, names in [
            ({"source_index": 0, "text": "x", "category": "forbidden"}, NAMES),
            ({"source_index": 0, "text": "x"}, NAMES[:4]),
            ({"source_index": -1, "text": "x"}, NAMES),
        ]:
            with self.assertRaises(ev.EvaluationError):
                b.project_blind(record, names)

    def test_native_semantic_macro_f1_declared_choice_and_coverage(self):
        cs = [case(i) for i in range(7)]
        labels = [label(c, y) for c, y in zip(cs, [0, 0, 1, 0, 1, 2, 3])]
        results = [result(cs[0], 0), result(cs[1], 2, peak=0), result(cs[2], 0),
                   result(cs[3], 0, status="unsupported"), result(cs[4], 0, status="abstained"),
                   result(cs[5], 0, status="error")]
        s = b.score_native(reversed(cs), reversed(labels), reversed(results))
        # Three scored pairs (0,0), (0,2), (1,0). F1 per observed class: .5, 0, 0.
        self.assertAlmostEqual(s["score"], 1/6)
        self.assertEqual(s["field_accuracy_scored"], 1/3)
        self.assertEqual([s[k] for k in ("answered", "unsupported", "abstained", "errors", "pending")],
                         [3, 1, 1, 1, 1])
        self.assertEqual(s["observed_gold_or_predicted_classes"], 3)
        self.assertEqual(s["answered_fraction"], 3/7)
        self.assertEqual(s["declared_choice_not_first_argmax"], 1)
        self.assertIsNone(b.score_native(cs, labels, [])["score"])

    def test_refuses_stale_missing_duplicate_and_invalid_ok_results(self):
        c = case(0)
        good = result(c, 0)
        bad_prob = result(c, 0)
        bad_prob["response"]["answers"]["q1"]["probabilities"]["option_2"] = float("nan")
        for results, labels in [
            ([bad_prob], [label(c, 0)]), ([good, good], [label(c, 0)]),
            ([{**good, "run_id": "other"}], [label(c, 0)]),
            ([{**good, "payload_sha256": "different"}], [label(c, 0)]),
            ([{**good, "group_id": "other"}], [label(c, 0)]),
            ([good], []), ([good], [label(case(0, "different"), 0)]),
        ]:
            with self.assertRaises(ev.EvaluationError):
                b.score_native([c], labels, results)

    def test_fixture_csv_parsing_and_source_receipt_refusal(self):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerows([["text", "category"], ['Fixture with, a comma\nand a newline', NAMES[32]]])
        raw = stream.getvalue().encode()
        cases, labels = b._decode_release(raw, NAMES)
        self.assertEqual(cases[0].text, 'Fixture with, a comma\nand a newline')
        self.assertEqual(labels, [label(cases[0], 32)])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as t:
            p = Path(t)
            pin = "1" * 64
            args = dict(finalists="", finalists_sha256="", manifest={}, manifest_sha256=pin,
                        arms={"a"}, baseline="a")
            with self.assertRaisesRegex(ev.EvaluationError, "finalists receipt"):
                b.load_released_test(p/"not-opened", p/"also-not-opened", **args)
            manifest = ev.freeze_split([cases[0].research_input()], seed=1, weights={"final": 1})
            receipt = {"schema": "joey-finalists-v1", "manifest_sha256": manifest["sha256"], "baseline": "a",
                       "arms": {"a": "2"*64}, "passes": None, "primary": "macro.accuracy"}
            path = p/"receipt"
            path.write_text(json.dumps(receipt))
            args.update(finalists=str(path), finalists_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        manifest=manifest, manifest_sha256=manifest["sha256"])
            vocab = p/"fake_categories"
            vocab.write_text(json.dumps(NAMES))
            with self.assertRaisesRegex(ev.EvaluationError, "full final-only"):
                b.load_released_test(p/"not-opened", vocab, **args)
            with patch.object(b, "CASES", 1), self.assertRaisesRegex(ev.EvaluationError, "pinned source"):
                b.load_released_test(p/"not-opened", vocab, **args)
            csv_path = p/"fabricated_csv"
            csv_path.write_bytes(raw)
            # Substitute *only in this test* the universe and hashes of the hand-built fixture.
            with patch.multiple(b, CASES=1, CSV_SHA256=hashlib.sha256(raw).hexdigest(),
                                CATEGORIES_SHA256=hashlib.sha256(vocab.read_bytes()).hexdigest()):
                self.assertEqual(b.load_released_test(csv_path, vocab, **args), (cases, labels))


if __name__ == "__main__":
    unittest.main()
