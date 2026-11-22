"""Invented matrices only; NumPy is an optional offline diagnostic dependency."""
import hashlib
import io
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    raise unittest.SkipTest("weight geometry requires optional NumPy") from None
from scripts import compact_spectrum as cs


def measure(a, b, ranks=(1,)):
    return cs.measure(a.shape, lambda s: (a[s], b[s]), ranks, seed=31, block=5)


class SpectrumTests(unittest.TestCase):
    def test_result_is_private_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            out = root / "geometry.json"
            cs.write_new(out, b'{"run":1}\n')
            with self.assertRaises(FileExistsError):
                cs.write_new(out, b'{"run":2}\n')
            self.assertEqual(out.read_bytes(), b'{"run":1}\n')
            self.assertFalse(out.stat().st_mode & 0o077)
            self.assertEqual(list(root.iterdir()), [out])

    def test_known_spectrum_and_byte_cost(self):
        d = np.zeros((7, 5))
        d[[0, 1, 2], [0, 1, 2]] = (4, -2, 1)
        got = cs.summarize([measure(-d, d, (1, 2, 4))])
        self.assertAlmostEqual(got["cosine"], -1)
        self.assertAlmostEqual(got["relative_frob_difference_over_A"], 2)
        self.assertEqual(got["bf16_two_originals_bytes"], 140)
        for r, energy in [(1, 16/21), (2, 20/21), (4, 1)]:
            row = got["by_rank"][str(r)]
            self.assertAlmostEqual(row["delta_energy_achieved"], energy)
            self.assertEqual(row["bf16_base_plus_factor_bytes"], 70 + 24*r)

    def test_seeded_approximation_against_dense_reconstruction(self):
        rng = np.random.default_rng(17)
        a, b = (cs.bf16(rng.normal(size=(37, 41))) for _ in range(2))
        score = measure(a, b, (1, 4))
        self.assertEqual(score, measure(a, b, (1, 4)))
        d = (b-a)/2
        mid = cs.bf16((a+b)/2)
        l, t = cs.factors(d, 4, 31, 5)
        got = cs.summarize([score])
        optimum = np.linalg.svd(d, compute_uv=False)**2
        for r in (1, 4):
            row = got["by_rank"][str(r)]
            err = np.linalg.norm(d - l[:, :r] @ t[:r]) / np.linalg.norm(d)
            self.assertAlmostEqual(row["delta_relative_frob_error"], err)
            self.assertLessEqual(row["delta_energy_achieved"], optimum[:r].sum()/optimum.sum() + 1e-12)
            delta_q = cs.bf16(l[:, :r]) @ cs.bf16(t[:r])
            energy_q = 1 - (np.linalg.norm(d-delta_q)/np.linalg.norm(d))**2
            self.assertAlmostEqual(row["bf16_delta_energy_achieved"], energy_q)
            num = np.linalg.norm(a - (mid-delta_q))**2 + np.linalg.norm(b - (mid+delta_q))**2
            den = np.linalg.norm(a)**2 + np.linalg.norm(b)**2
            self.assertAlmostEqual(row["bf16_pair_relative_frob_error"], np.sqrt(num/den))

    def test_pooling_and_undefined_metrics(self):
        x = np.ones((3, 2))
        equal = measure(3*x, 3*x)
        self.assertIsNone(cs.summarize([equal])["by_rank"]["1"]["delta_energy_achieved"])
        pooled = cs.summarize([equal, measure(x, -x)])
        self.assertAlmostEqual(pooled["cosine"], 0.8)
        self.assertAlmostEqual(pooled["relative_frob_difference_over_A"], np.sqrt(24/60))
        zero = cs.summarize([measure(0*x, 0*x)])
        self.assertIsNone(zero["cosine"])
        self.assertIsNone(zero["relative_frob_difference_over_A"])

    def test_reject_bad_matrix_before_fit(self):
        x = np.ones((3, 4))
        for invalid in (x[:, :2], x*np.nan, x*np.inf):
            with self.assertRaises(ValueError):
                measure(x, invalid)
        with self.assertRaisesRegex(ValueError, "memory budget"):
            cs.measure((65536, 65536), lambda _: self.fail("should not read any block"))

    def test_bf16_offset_and_ties(self):
        data = io.BytesIO(b"abc" + struct.pack("<6H", 0, 0, 0xbf80, 0x3e80, 0x4000, 0x3f00))
        tensor = SimpleNamespace(offset=3, shape=(3, 2))
        np.testing.assert_array_equal(cs.read_rows(data, tensor, slice(1, 3)), [[-1, .25], [2, .5]])
        values = np.array([1 + 1/256, 1 + 3/256, -1 - 1/256])
        np.testing.assert_array_equal(cs.bf16(values), [1, 1 + 1/64, -1])
        with self.assertRaisesRegex(ValueError, "short tensor"):
            cs.read_rows(data, tensor, slice(1, 4))

    def test_full_pin_not_prefix(self):
        pin = hashlib.sha256(b"abcdefgh").hexdigest()
        self.assertEqual(cs.verify_pin(io.BytesIO(b"abcdefgh"), pin), pin)
        for raw, expected in ((b"abcdefgx", pin), (b"abcdefgh", pin[:16])):
            with self.assertRaises(ValueError):
                cs.verify_pin(io.BytesIO(raw), expected)

    def test_scope_rejects_wrong_family_or_orientation(self):
        cycle = ("linear_attention",)*3 + ("full_attention",)
        suffixes = {f"mlp.{p}_proj.weight" for p in ("gate", "up", "down")}
        hdr = {}
        for i in range(8, 16):
            pre = f"text.{i}."
            hdr[pre + ("self_attn.q" if i % 4 == 3 else "linear_attn.a")] = None
            hdr[pre + "input_layernorm.weight"] = SimpleNamespace(shape=(5,))
            for name in suffixes:
                hdr[pre + name] = SimpleNamespace(shape=(5, 7) if "down" in name else (7, 5),
                                                  dtype="BF16")

        def pairs(h):
            return list(cs.matched_pairs(h, "text.", cycle, suffixes))

        self.assertEqual([(ids, p) for ids, _, p, _, _ in pairs(hdr)],
                         [((i, i+4), p) for i in range(8, 12) for p in ("gate", "up", "down")])
        wrong = dict(hdr)
        wrong["text.14.self_attn.q"] = wrong.pop("text.14.linear_attn.a")
        with self.assertRaisesRegex(ValueError, "layer type"):
            pairs(wrong)
        hdr["text.15.mlp.down_proj.weight"].shape = (7, 5)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            pairs(hdr)


if __name__ == "__main__":
    unittest.main()
