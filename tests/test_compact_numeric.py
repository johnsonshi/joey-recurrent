"""Bit-level checks of the scalar merge with an independent exact-rational rounding oracle."""

from fractions import Fraction
from pathlib import Path
import random
import tempfile
import unittest

from joey_research.compact import CompactError, OutputTensor, Tensor, _cast, _plan


def power2(exponent):
    return Fraction(2) ** exponent


def exact(word, fraction_bits):
    exponent = (word >> fraction_bits) & 255
    if exponent == 255:
        raise ValueError("non-finite input")
    fraction = word & ((1 << fraction_bits) - 1)
    if exponent:
        fraction += 1 << fraction_bits
    value = fraction * power2(max(1, exponent) - 127 - fraction_bits)
    return -value if word >> (8 + fraction_bits) else value


def quantize(value, fraction_bits, negative_zero=False):
    sign = int(value < 0 or (value == 0 and negative_zero)) << (8 + fraction_bits)
    mag = abs(value)
    if not mag:
        return sign
    exponent = mag.numerator.bit_length() - mag.denominator.bit_length()
    exponent -= mag < power2(exponent)
    exponent = max(-126, exponent)
    unit = 1 << fraction_bits
    significand = round(mag / power2(exponent - fraction_bits))  # Fraction rounds exact halves to even
    if significand == 2 * unit:
        exponent += 1
        significand = unit
    if exponent > 127:
        raise OverflowError("rounds to infinity")
    if significand < unit:
        return sign | significand
    return sign | ((exponent + 127) << fraction_bits) | (significand - unit)


def reference(words, weights, fraction_bits=7):
    acc = 0
    for word, weight in zip(words, weights, strict=True):
        w = exact(quantize(Fraction(weight), 23), 23)
        product = quantize(exact(word, fraction_bits) * w, 23, bool(word >> (8 + fraction_bits)))
        acc = quantize(exact(acc, 23) + exact(product, 23), 23, acc == product == 0x80000000)
    return quantize(exact(acc, 23), fraction_bits, acc == 0x80000000)


class MergeNumericsTest(unittest.TestCase):
    def run_kernel(self, rows, weights, fraction_bits=7):
        width = 2 if fraction_bits == 7 else 4
        dtype = "BF16" if width == 2 else "F32"
        group = {"cycles": list(range(len(weights))), "weights": weights}
        groups, _ = _plan({"format": "joey-compact/v1", "groups": [group], "repeat_cycles": [0, 1]}, 6)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1], prefix=".test-numeric-") as tmp:
            tensors = []
            for i, data in enumerate(zip(*rows, strict=True)):
                path = Path(tmp) / str(i)
                size = path.write_bytes(b"".join(word.to_bytes(width, "little") for word in data))
                tensors.append(Tensor(path, 0, dtype, (1, len(rows)), size))
            result = b"".join(OutputTensor(tuple(tensors), groups[0].coefficients).data())
        return [int.from_bytes(result[i:i+width], "little") for i in range(0, len(result), width)]

    def check(self, rows, weights, expected=None):
        oracle = [reference(row, weights) for row in rows]
        if expected is not None:
            self.assertEqual(oracle, expected)
        self.assertEqual(self.run_kernel(rows, weights), oracle)

    def test_signed_zero_subnormal_and_midpoints(self):
        rows = [(0x8000, 0x8000), (0x8001, 0x8000), (0x0001, 0), (0x8001, 0x8002),
                (0x007f, 0x0080), (0x807f, 0x8080), (0x3f80, 0x3f81), (0x3f81, 0x3f82),
                (0xbf80, 0xbf81), (0xbf81, 0xbf82), (0x7f7f, 0xff7f), (0x7f7f, 0x7f7f)]
        self.check(rows, [0.5, 0.5], [0, 0x8000, 0, 0x8002, 0x0080, 0x8080, 0x3f80, 0x3f82,
                                      0xbf80, 0xbf82, 0, 0x7f7f])
        for delta, want in [(-2**-12, 0x3f80), (2**-12, 0x3f81)]:
            self.check([(0x3f80, 0x3f81)], [0.5-delta, 0.5+delta], [want])

    def test_product_rounding_and_gradual_underflow(self):
        rows = [(1, 1, 0), (0x8001, 0x8001, 0)]
        # A one-FP32-subnormal increment moves a BF16 midpoint; half that increment rounds to zero.
        self.check(rows, [0.5, 2**-16, 0.5-2**-16], [1, 0x8001])
        self.check(rows, [0.5, 2**-17, 0.5-2**-17], [0, 0x8000])
        # The rounded first product cancels the independently rounded second product, not an FMA.
        self.check([(0xbf81, 0x3f81, 0)], [1/3, 1/3, 1/3], [0])
        # The middle add must round before the final cancellation.
        self.check([(0x4c80, 0x3f80, 0xcc80)], [0.25, 0.5, 0.25], [0])

    def test_wide_finite_range_across_a_chunk_boundary(self):
        rng = random.Random(421)

        def finite():
            return rng.randrange(0x7f80) | (rng.randrange(2) << 15)

        rows = [(finite(), finite()) for _ in range(4097)]
        self.check(rows, [0.3, 0.7])

    def test_overflow_and_nonfinite_are_rejected(self):
        for sign in (0, 0x80000000):
            # The valid plan tolerance admits this one-ulp change in a coefficient. The FP32 sum overflows.
            row = (sign | 0x7f7fffff,) * 2
            weights = [0.5, 0.5 + 2**-24]
            with self.assertRaises(OverflowError):
                reference(row, weights, 23)
            with self.assertRaises(CompactError):
                self.run_kernel([row], weights, 23)
            # Final-cast threshold: the preceding finite FP32 word still rounds to maximum finite BF16.
            below = exact(sign | 0x7f7f7fff, 23)
            self.assertEqual(_cast([float(below)], "BF16"), quantize(below, 7).to_bytes(2, "little"))
            threshold = exact(sign | 0x7f7f8000, 23)
            with self.assertRaises(OverflowError):
                quantize(threshold, 7)
            with self.assertRaises(CompactError):
                _cast([float(threshold)], "BF16")
        for special in (0x7f80, 0xff80, 0x7f81, 0xffc0):
            with self.assertRaises(ValueError):
                reference((special, 0), [0.25, 0.75])
            with self.assertRaises(CompactError):
                self.run_kernel([(special, 0)], [0.25, 0.75])
