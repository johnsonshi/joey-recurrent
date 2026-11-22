"""Isolated synthetic checks. Discovery needs only the standard library."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import unittest

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    np = None

if np is not None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "mlp_channel_alignment.py"
    spec = importlib.util.spec_from_file_location("mlp_channel_alignment", path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)


def layer(m, h, seed):
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal(shape).astype("f4") for k, shape in
            (("gate", (m, h)), ("up", (m, h)), ("down", (h, m)))}


def permute(a, p):
    b = {k: np.empty_like(v) for k, v in a.items()}
    b["gate"][p] = a["gate"]
    b["up"][p] = a["up"]
    b["down"][:, p] = a["down"]
    return b


def entries(layer_, name, coords):
    a = layer_[name]
    return (a[coords, :].T if name == "down" else a[:, coords]).astype("f8")


def probe_sse(a, b, coords, p):
    return sum(np.sum((entries(a, k, coords) - entries(b, k, coords)[p]) ** 2)
               for k in a)


@unittest.skipIf(np is None, "optional numerical tests require NumPy")
class ChannelAlignmentTests(unittest.TestCase):
    def test_recovers_noninvolutive_permutation_without_writing_weights(self):
        m, h = 137, 20  # Spans multiple match, metric and action tiles.
        p = (np.arange(m) + 9) % m
        a = layer(m, h, 101)
        b = permute(a, p)
        for v in (*a.values(), *b.values()):
            v.flags.writeable = False
        got, d = core.analyze_pair(a, b, seed=221, sketch=5)
        np.testing.assert_array_equal(got, p)
        digest = hashlib.sha256(struct.pack("<%dq" % m, *p)).hexdigest()
        self.assertEqual(d["permutation_sha256"], digest)
        for split in ("fit", "audit"):
            pooled = d[split]["pooled"]
            self.assertGreater(pooled["identity_sse"], 0)
            self.assertEqual(pooled["aligned_sse"], 0)
            self.assertEqual(pooled["fraction_sse_removed"], 1)
            self.assertEqual(pooled["count"], 3 * m * (h // 2))
            for t in d[split]["projections"].values():
                self.assertAlmostEqual(t["aligned_cosine"], 1.0)
        self.assertTrue(d["action_check"]["passed"])

    def test_only_sketch_selects_mapping_even_when_audit_prefers_identity(self):
        m, h, seed = 13, 30, 23
        p = (np.arange(m) + 1) % m
        a = layer(m, h, 504)
        fit, audit = core.coordinate_split(h, seed=seed)
        np.testing.assert_array_equal(np.r_[fit, audit], np.random.default_rng(seed).permutation(h))
        for overridden, split in ((audit, "audit"), (fit[3:], "fit")):
            with self.subTest(split=split):
                b = permute(a, p)
                for k in ("gate", "up"):
                    b[k][:, overridden] = a[k][:, overridden]
                b["down"][overridden, :] = a["down"][overridden, :]
                got, d = core.analyze_pair(a, b, seed=seed, sketch=3)
                np.testing.assert_array_equal(got, p)
                self.assertEqual(d["coordinates"]["fit"], fit.tolist())
                self.assertEqual(d["coordinates"]["audit"], audit.tolist())
                self.assertEqual(d["coordinates"]["sketch"], sorted(fit[:3]))
                other = "audit" if split == "fit" else "fit"
                self.assertEqual(d[other]["pooled"]["aligned_sse"], 0)
                self.assertGreater(d[split]["pooled"]["aligned_sse"], 0)
                if split == "audit":
                    self.assertEqual(d["audit"]["pooled"]["identity_sse"], 0)
                    self.assertIsNone(d["audit"]["pooled"]["fraction_sse_removed"])

    def test_down_columns_alone_identify_channels(self):
        a = layer(7, 12, 3)
        a["gate"].fill(0)
        a["up"].fill(0)
        p = np.array([2, 5, 1, 0, 6, 4, 3])
        got, d = core.analyze_pair(a, permute(a, p), seed=41)
        np.testing.assert_array_equal(got, p)
        down = d["audit"]["projections"]["down"]
        self.assertEqual(down["aligned_sse"], 0)
        self.assertEqual(down["identity_sse"], d["audit"]["pooled"]["identity_sse"])
        self.assertGreater(down["identity_sse"], 0)

    def test_signed_cosines_have_equal_projection_weight(self):
        a = layer(2, 6, 0)
        for k in a:
            v = a[k].T if k == "down" else a[k]
            v[0], v[1] = 1, -1
        a["gate"] *= 1e18
        b = permute(a, [1, 0])
        b["gate"][:] = a["gate"]  # Raw magnitudes (or unsigned scores) favor identity.
        got, d = core.analyze_pair(a, b, seed=2, sketch=1)
        np.testing.assert_array_equal(got, [1, 0])
        self.assertLess(d["audit"]["pooled"]["fraction_sse_removed"], 0)

    def test_global_priority_cutoff_ties_and_best_live_fallback(self):
        a, b = layer(4, 2, 0), layer(4, 2, 1)
        for k in ("gate", "up", "down"):
            a[k].fill(1)
            v = b[k].T if k == "down" else b[k]
            v[:] = np.array([1, -1, 1, 1])[:, None]
        a["down"][:, 0] = 0
        b["down"][:, 2:] = -1
        # Source 1 beats 0 for dest 0; in fallback, 0 prefers tied 2/3 over 1.
        for k, count in ((1, 1), (2, 2), (4, 4)):
            with self.subTest(k=k):
                for _ in range(2):
                    got, d = core.analyze_pair(a, b, seed=3, shortlist=k)
                    np.testing.assert_array_equal(got, [2, 0, 3, 1])
                    self.assertEqual(d["matching"]["shortlist_matches"], count)
                    self.assertEqual(d["matching"]["fallback"], 4 - count)

    def test_exact_sums_and_null_probe_against_direct_indexing(self):
        m, h, seed = 17, 26, 17
        # Noncontiguous views, including down, as can occur with mapped weights.
        def strided(a):
            return {k: v[::2, :] if k == "down" else v[:, ::2] for k, v in a.items()}

        a, b = (strided(layer(m, 2 * h, i)) for i in (203, 204))
        p, d = core.analyze_pair(a, b, seed=seed, sketch=4)
        for split in ("fit", "audit"):
            coords = d["coordinates"][split]
            total = d[split]["pooled"]
            for k, t in d[split]["projections"].items():
                x, y = entries(a, k, coords), entries(b, k, coords)
                self.assertEqual(t["count"], x.size)
                self.assertAlmostEqual(t["earlier_ss"], np.sum(x**2))
                self.assertAlmostEqual(t["later_ss"], np.sum(y**2))
                for label, z in (("identity", y), ("aligned", y[p])):
                    dot = np.sum(x * z)
                    cosine = dot / (np.linalg.norm(x) * np.linalg.norm(z))
                    self.assertAlmostEqual(t[label + "_sse"], np.sum((x - z)**2))
                    self.assertAlmostEqual(t[label + "_dot"], dot)
                    self.assertAlmostEqual(t[label + "_cosine"], cosine)
            self.assertAlmostEqual(total["identity_sse"], probe_sse(a, b, coords, np.arange(m)))
            self.assertAlmostEqual(total["aligned_sse"], probe_sse(a, b, coords, p))

        probe = d["null_probe"]
        coords = probe["coordinates"]
        self.assertEqual(coords, sorted(d["coordinates"]["audit"][:8]))
        self.assertLess(len(coords), h // 2)
        self.assertEqual(probe["count"], 3 * m * len(coords))
        self.assertAlmostEqual(probe["aligned_sse"], probe_sse(a, b, coords, p))
        self.assertAlmostEqual(probe["identity_sse"], probe_sse(a, b, coords, np.arange(m)))
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
        expected = [probe_sse(a, b, coords, rng.permutation(m)) for _ in range(32)]
        np.testing.assert_allclose(probe["shuffled_sse"], expected, rtol=1e-14)
        no_null, disabled = core.analyze_pair(a, b, seed=seed, sketch=4, null_reps=0)
        np.testing.assert_array_equal(no_null, p)
        self.assertEqual(disabled["audit"], d["audit"])
        self.assertEqual(disabled["null_probe"]["shuffled_sse"], [])

    def test_small_residual_is_not_lost_to_large_weight_norms(self):
        a, b = layer(3, 8, 0), layer(3, 8, 1)
        for v in (*a.values(), *b.values()):
            v.fill(1e18)
        _, audit = core.coordinate_split(8, seed=77)
        a["up"][1, audit[0]], b["up"][1, audit[0]] = 0, 0.5
        _, d = core.analyze_pair(a, b, seed=77)
        self.assertEqual(d["audit"]["pooled"]["identity_sse"], 0.25)
        self.assertEqual(d["audit"]["pooled"]["aligned_sse"], 0.25)

    def test_action_is_full_swiglu_in_both_channel_orders(self):
        m, h = 9, 6
        a = layer(m, h, 13)
        a["gate"] *= 4000  # Exercises large negative and positive gate values.
        x = core._activations(h)
        g = a["gate"].astype("f8") @ x
        sigmoid = np.exp(-np.logaddexp(0., -g))
        expected = a["down"].astype("f8") @ ((g * sigmoid) * (a["up"].astype("f8") @ x))
        p = (np.arange(m) + 4) % m
        with np.errstate(over="raise", invalid="raise"):
            for order in (np.arange(m), p):
                actual = core._action(core._channel_views(a), order, x, rows=2)
                np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-24)
        check = core._action_check(core._channel_views(a), p, rows=2)
        self.assertTrue(check["passed"])
        for i, name in enumerate(("unit_0", "unit_1", "near_zero", "zero")):
            ref = check["vectors"][name]["reference_l2"]
            self.assertTrue(math.isclose(ref, np.linalg.norm(expected[:, i]), rel_tol=1e-13))
        tiny = check["vectors"]["near_zero"]
        self.assertTrue(0 < tiny["reference_l2"] < 1e-8)
        self.assertIsNotNone(tiny["relative_l2"])
        self.assertIsNone(check["vectors"]["zero"]["relative_l2"])

    def test_zero_denominators_and_degenerate_ties(self):
        a = layer(9, 6, 100)
        for v in a.values():
            v.fill(0)
        for k in (1, 3, 8, 20):
            p, d = core.analyze_pair(a, a, seed=12, shortlist=k, null_reps=5)
            np.testing.assert_array_equal(p, np.arange(9))
            self.assertEqual(d["matching"]["fallback"], max(0, 9 - k))
            self.assertIsNone(d["fit"]["pooled"]["fraction_sse_removed"])
            for t in d["audit"]["projections"].values():
                self.assertIsNone(t["identity_cosine"])
                self.assertIsNone(t["aligned_cosine"])
            self.assertEqual(d["null_probe"]["shuffled_sse"], [0.] * 5)
            self.assertTrue(d["action_check"]["passed"])
            json.dumps(d, allow_nan=False)

    def test_rejects_invalid_layout_and_excess_workspace_before_allocating(self):
        a = layer(3, 8, 42)
        for b in (dict(a, down=a["down"].T), dict(a, up=a["up"].astype("f8"))):
            with self.assertRaises(ValueError):
                core.analyze_pair(a, b, seed=0)
        with self.assertRaises(ValueError):
            core.coordinate_split(7, seed=0)
        for opts in ({"sketch": 0}, {"shortlist": -1}, {"null_reps": -1}):
            with self.assertRaises(ValueError):
                core.analyze_pair(a, a, seed=0, **opts)
        _, audit = core.coordinate_split(8, seed=0)
        a["up"][1, audit[0]] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            core.analyze_pair(a, a, seed=0)

        # Zero-stride shape fixture: no model-sized array is read or allocated.
        v = np.broadcast_to(np.float32(0), (6144, 2048))
        big = dict(gate=v, up=v, down=v.T)
        with self.assertRaisesRegex(ValueError, "byte limit"):
            core.analyze_pair(big, big, seed=0, sketch=1024)
        w = core._workspace(6144, 2048, 64, 16, 8)
        self.assertLessEqual(w["pair_tile_bytes"], w["tile_limit_bytes"])
        self.assertLessEqual(w["weight_tile_bytes"], w["tile_limit_bytes"])
        self.assertLess(w["pair_tile_rows"], 6144)


if __name__ == "__main__":
    unittest.main()
