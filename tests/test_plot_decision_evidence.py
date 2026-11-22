"""Invented aggregate-only fixtures. Run with python3 -B -m unittest discover -s tests."""

import copy
import importlib.util
import io
import json
import re
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_decision_evidence.py"
spec = importlib.util.spec_from_file_location("plot_decision_evidence", SCRIPT)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def thresholds(*pairs):
    return {t: {"accepted": a, "wrong": e}
            for t, (a, e) in zip(("0.80", "0.90", "0.95", "0.99"), pairs)}


def fixture():
    """A complete v1 group, entirely invented; no identifiers or per-example rows."""
    return {
        "version": 1, "split": "Toy A & B", "projection": "Binary <choice>",
        "n": 8, "depth_unit": "layers", "interval_level": 0.95,
        "timing": {"statistic": "median", "per": "batch", "batch_size": 4},
        "series": [
            {"model": "Cedar", "family": "Grove", "path": "ordinary", "passes": [
                {"pass": 1, "depth": 4, "correct": 5, "brier": 0.5888, "cuda_ms": 3,
                 "stored_bytes": 2400,
                 "bins": [{"index": 3, "n": 0},
                          {"index": 9, "n": 2, "correct": 1, "mean_confidence": 0.60},
                          {"index": 12, "n": 1, "correct": 1, "mean_confidence": 0.80},
                          {"index": 13, "n": 1, "correct": 0, "mean_confidence": 0.90},
                          {"index": 14, "n": 4, "correct": 3, "mean_confidence": 0.97,
                           "accuracy_interval": [0.40, 0.89]}],
                 "thresholds": thresholds((6, 2), (5, 2), (4, 1), (2, 1)),
                 "ties": [{"confidence": score, "n": n, "wrong": wrong}
                          for score, n, wrong in [(0.99, 2, 1), (0.95, 2, 0), (0.90, 1, 1),
                                                  (0.80, 1, 0), (0.60, 2, 1)]]},
                {"pass": 2, "depth": 8, "correct": 6, "cuda_ms": 6,
                 "from_previous": {"matched": 8, "corrections": 2, "corruptions": 1, "overthought": 1},
                 "thresholds": thresholds((3, 1), (1, 0), (0, 0), (0, 0))},
                {"pass": 3, "depth": 12, "correct": 5, "brier": 0.69,
                 "from_previous": {"matched": 8, "corrections": 1, "corruptions": 2, "overthought": 3},
                 "thresholds": thresholds((8, 3), (4, 2), (2, 1), (1, 1))}],
             "adaptive": {"accepted": 5, "wrong": 1}},
            {"model": "Cedar", "family": "Grove", "path": "stock_split", "passes": [
                {"pass": 1, "depth": 4, "correct": 6, "brier": 0.47, "cuda_ms": 2.75,
                 "stored_bytes": 2600}],
             "adaptive": {"accepted": 0, "wrong": 0}},
            {"model": "Dune", "family": "Ridge", "passes": [
                {"pass": 1, "depth": 6, "correct": 7, "cuda_ms": 4.25, "stored_bytes": 4000},
                {"pass": 2, "depth": 12, "correct": 6, "cuda_ms": 8.5}]},
        ]}


NS = "{http://www.w3.org/2000/svg}"


def nodes(root, kind):
    return root.findall(f".//*[@class='{kind}']")


def rows(root):
    return [[cell.text for cell in row.findall(NS + "text")] for row in nodes(root, "table-row")]


def to_fractions(axes, x, y):
    """Recover fractions from the SVG grid itself, independent of its pixel dimensions."""
    grid = [edge for edge in axes.findall(NS + "line")
            if edge.get("y1") == edge.get("y2") and edge.get("x1") != edge.get("x2")]
    left, right = (float(grid[0].get(k)) for k in ("x1", "x2"))
    ys = [float(edge.get("y1")) for edge in grid]
    return (float(x)-left)/(right-left), (max(ys)-float(y))/(max(ys)-min(ys))


def plotted_rate(axes, x, y):
    ticks = [float(t.text) for t in axes.findall(NS+"text") if t.get("text-anchor") == "end"]
    return to_fractions(axes, x, y)[1] * max(ticks)/100


class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blobs = plot.render_evidence(fixture())
        cls.views = {k: ET.fromstring(v) for k, v in cls.blobs.items()}

    def test_seven_standalone_views_share_only_the_declared_group(self):
        self.assertEqual(set(self.views), {s + ".svg" for s in (
            "reliability", "depth", "confident_wrong", "transitions", "cuda_time", "storage", "selective_risk")})
        data = plot.load_summary(io.StringIO(json.dumps(fixture())))
        original = copy.deepcopy(data)
        self.assertEqual(plot.render_evidence(data), self.blobs)
        self.assertEqual(data, original)
        for filename, root in self.views.items():
            with self.subTest(filename=filename):
                self.assertIn("Split: Toy A & B", "".join(root.itertext()))
                self.assertIn("Projection: Binary <choice>", "".join(root.itertext()))
                self.assertIn("Matched N = 8", "".join(root.itertext()))
                self.assertIn(b"&amp; B", self.blobs[filename])
                self.assertIn(b"&lt;choice&gt;", self.blobs[filename])
                self.assertFalse(any(x.tag == NS+"image" or "href" in x.attrib for x in root.iter()))

    def test_calibration_means_support_absence_and_asymmetric_interval(self):
        root = self.views["reliability.svg"]
        self.assertEqual(len(nodes(root, "identity")), 1)
        self.assertEqual(len(nodes(root, "bin-accuracy")), 4)  # Includes a measured zero accuracy.
        table = rows(root)
        self.assertEqual(len(table), 15)
        self.assertEqual(table[3], ["4", "0", "—", "—"])
        self.assertEqual(table[4], ["5", "—", "—", "—"])
        self.assertEqual(table[14], ["15", "4", "97%", "75%"])
        axes, = nodes(root, "axes")
        point = nodes(axes, "bin-accuracy")[-1]
        x, y = to_fractions(axes, point.get("cx"), point.get("cy"))
        self.assertAlmostEqual(x, .97, places=5)  # Not a bin midpoint.
        self.assertAlmostEqual(y, .75, places=5)
        ci, = nodes(axes, "accuracy-interval")
        for end, expected in (("1", .40), ("2", .89)):
            self.assertAlmostEqual(to_fractions(axes, ci.get("x"+end), ci.get("y"+end))[1], expected, places=5)

    def test_depth_panels_use_distinct_scales_models_and_missing_score_gaps(self):
        a, b = nodes(self.views["depth.svg"], "axes")
        self.assertIn("accuracy (%)", a.get("aria-label"))
        self.assertIn("Brier score by Completed depth (layers)", b.get("aria-label"))
        lines = nodes(a, "depth-accuracy-line")
        self.assertEqual(len({line.get("stroke") for line in lines}), 3)
        depth = nodes(a, "depth-accuracy")[1]
        self.assertEqual(to_fractions(a, depth.get("cx"), depth.get("cy")), (.5, .75))
        brier = nodes(b, "depth-brier")[0]
        self.assertAlmostEqual(to_fractions(b, brier.get("cx"), brier.get("cy"))[1]*2, .5888, places=5)
        self.assertNotIn("L", nodes(b, "depth-brier-line")[0].get("d"))

    def test_confident_errors_use_both_denominators_and_break_at_zero_support(self):
        root = self.views["confident_wrong.svg"]
        joint, cond = nodes(root, "axes")
        self.assertEqual(len(nodes(joint, "joint-error")), 12)
        self.assertEqual(len(nodes(cond, "conditional-error")), 10)
        self.assertIn("2 / 6 (75%)", rows(root)[0])
        self.assertIn("0 / 0 (0%); undefined", rows(root)[1])
        jp = nodes(joint, "joint-error")[0]
        cp = nodes(cond, "conditional-error")[0]
        self.assertAlmostEqual(plotted_rate(joint, jp.get("cx"), jp.get("cy")), 2/8, places=5)
        self.assertAlmostEqual(plotted_rate(cond, cp.get("cx"), cp.get("cy")), 2/6, places=5)
        self.assertNotIn("L", nodes(cond, "conditional-error-line")[2].get("d"))

    def test_matched_flips_and_history_are_distinct(self):
        root = self.views["transitions.svg"]
        widths = lambda key: [float(b.get("width")) for b in nodes(root, key)]
        unit = widths("corruptions")[0]
        self.assertGreater(unit, 0)
        self.assertEqual(widths("corrections"), [2*unit, unit])
        self.assertEqual(widths("corruptions"), [unit, 2*unit])
        self.assertEqual(widths("overthought"), [unit, 3*unit])

    def test_costs_keep_family_encoding_and_execution_path(self):
        cuda, storage = self.views["cuda_time.svg"], self.views["storage.svg"]
        self.assertIn("Median measured CUDA time (ms/batch)", "".join(cuda.itertext()))
        self.assertIn("Batch size: 4", "".join(cuda.itertext()))
        self.assertIn(["2", "1", "4", "2.75", "6", "75%"], rows(cuda))
        self.assertIn(["2", "1", "4", "2600", "6", "75%"], rows(storage))
        self.assertEqual(len(nodes(cuda, "cost-point")), 5)
        marks = nodes(storage, "cost-point")
        self.assertEqual([m.tag for m in marks], [NS+"circle", NS+"path", NS+"rect"])
        self.assertEqual(marks[0].get("fill"), marks[1].get("fill"))
        self.assertNotEqual(marks[0].get("fill"), marks[2].get("fill"))
        self.assertEqual(marks[0].get("fill"), nodes(cuda, "cost-point")[0].get("fill"))

    def test_empirical_risk_admits_whole_ties_and_separates_observed_policy(self):
        root = self.views["selective_risk.svg"]
        fixed, adaptive = nodes(root, "axes")
        dots, = nodes(fixed, "empirical-ties")
        points = re.findall(r"M([\d.]+),([\d.]+)h0", dots.get("d"))
        self.assertEqual(len(points), 5)
        for (x, y), (a, e) in zip(points, [(2, 1), (4, 1), (5, 2), (6, 2), (8, 3)]):
            cov, _ = to_fractions(fixed, x, y)
            self.assertAlmostEqual(cov, a/8, places=5)
            self.assertAlmostEqual(plotted_rate(fixed, x, y), e/a, places=5)
        self.assertFalse(nodes(fixed, "adaptive-point"))
        self.assertFalse(nodes(adaptive, "empirical-ties"))
        self.assertEqual(len(nodes(adaptive, "adaptive-point")), 1)
        self.assertEqual(rows(root)[0][1:], ["5", "1", "62.5%", "20%"])
        self.assertEqual(rows(root)[1][1:], ["0", "0", "0%", "— (undefined)"])

    def test_partial_releases_do_not_synthesize_other_views(self):
        d = fixture()
        d["series"] = [d["series"][0]]
        s = d["series"][0]
        s["passes"] = []
        s["adaptive"] = {"accepted": 0, "wrong": 0}
        out = plot.render_evidence(d)
        self.assertEqual(list(out), ["selective_risk.svg"])
        self.assertFalse(nodes(ET.fromstring(out["selective_risk.svg"]), "adaptive-point"))
        del s["adaptive"]
        s["passes"] = [{"pass": 7, "depth": 14, "correct": 0, "bins": [{"index": 1, "n": 0}]}]
        self.assertEqual(list(plot.render_evidence(d)), ["depth.svg"])

    def test_rare_measured_errors_remain_visible_on_labeled_axes(self):
        n = 1_000_000
        r = {"pass": 1, "depth": 1, "correct": n-1,
             "thresholds": thresholds((100, 1), (90, 1), (50, 0), (0, 0))}
        d = {"version": 1, "split": "Toy", "projection": "Top", "n": n, "depth_unit": "passes",
             "series": [{"model": "M", "family": "F", "passes": [r]}]}
        axis = nodes(ET.fromstring(plot.render_evidence(d)["confident_wrong.svg"]), "axes")[0]
        dot = nodes(axis, "joint-error")[0]
        x, y = dot.get("cx"), dot.get("cy")
        self.assertGreater(to_fractions(axis, x, y)[1], .4)
        self.assertAlmostEqual(plotted_rate(axis, x, y), 1/n, places=12)

    def test_public_label_text_is_not_markup_or_title_content(self):
        d = fixture()
        d["series"][2]["model"] = 'A & <script> "B"'
        for blob in plot.render_evidence(d).values():
            root = ET.fromstring(blob)
            self.assertFalse(root.findall(".//" + NS + "script"))
            for title in root.iter(NS + "title"):
                self.assertNotIn("<script>", title.text)
        self.assertIn(b'&amp; &lt;script&gt;', plot.render_evidence(d)["cuda_time.svg"])


class ValidationTests(unittest.TestCase):
    def test_types_bounds_and_malformed_counts_fail_before_render(self):
        p = ("series", 0, "passes", 0)
        bad = [
            (("version",), True), (("n",), 0), (("n",), 2**53), (("split",), ["Toy"]),
            (("projection",), "bad\x01label"), (("split",), "a"*25), (("split",), "a/b"),
            (("interval_level",), 1), (("series", 0, "path"), []),
            (("timing", "batch_size"), 1.0), (("timing", "per"), "token"),
            (p+("pass",), False), (p+("depth",), 0), (p+("correct",), 9),
            (p+("brier",), float("nan")), (p+("brier",), 2.01),
            (p+("cuda_ms",), float("inf")), (p+("cuda_ms",), 0), (p+("stored_bytes",), 12.5),
            (p+("bins", 0, "index"), 15), (p+("bins", 1, "index"), 3),
            (p+("bins", 1, "correct"), 3), (p+("bins", 1, "mean_confidence"), .58),
            (p+("bins", 4, "accuracy_interval"), [.8, .7]),
            (p+("bins", 4, "accuracy_interval"), [0, float("nan")]),
            (p+("thresholds", "0.80", "accepted"), 3), (p+("thresholds", "0.80", "wrong"), 4),
            (p+("ties", 1, "confidence"), .99), (p+("ties", 2, "n"), 0),
            (p+("ties", 1, "wrong"), 1), (("series", 1, "adaptive", "wrong"), 1),
            (("series", 0, "passes", 1, "from_previous", "matched"), 7),
            (("series", 0, "passes", 1, "from_previous", "corrections"), 1),
            (("series", 0, "passes", 1, "from_previous", "overthought"), 2),
            (("series", 0, "passes", 2, "pass"), 4),
        ]
        for path, value in bad:
            with self.subTest(path=path, value=value):
                data = fixture()
                parent = data
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = value
                with self.assertRaises(ValueError):
                    plot.render_evidence(data)

    def test_redundant_summaries_must_agree_and_provenance_is_required(self):
        for mutation in ("timing", "interval_level", "bin", "inclusive threshold", "extra", "empty"):
            with self.subTest(mutation=mutation):
                d = fixture()
                r = d["series"][0]["passes"][0]
                if mutation in ("timing", "interval_level"):
                    del d[mutation]
                elif mutation == "bin":
                    r["bins"][-1]["mean_confidence"] = .98
                elif mutation == "inclusive threshold":
                    r["thresholds"]["0.99"] = {"accepted": 0, "wrong": 0}
                elif mutation == "extra":
                    r["vram_bytes"] = r.pop("stored_bytes")
                else:
                    r["bins"][0]["mean_confidence"] = .22
                with self.assertRaises(ValueError):
                    plot.render_evidence(d)

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_literals(self):
        for source in ('{"series":[{"n":1,"n":2}]}', '{"brier":NaN}', '{"confidence":Infinity}'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                plot.load_summary(io.StringIO(source))


if __name__ == "__main__":
    unittest.main()
