import unittest
from fractions import Fraction

from benchmarks.paired_precision import _binomial, coverage


class PrecisionTest(unittest.TestCase):
    def test_null_distribution_and_single_row_equivalence(self):
        self.assertEqual(_binomial(2,Fraction(1,2)),(Fraction(1,4),Fraction(1,2),Fraction(1,4)))
        r=coverage(12,1)["resampling"]
        self.assertEqual(r["group"],r["row"])

    def test_identical_renderings_do_not_increase_independent_source_count(self):
        grouped=coverage(12,8)["resampling"]
        baseline=coverage(12,1)["resampling"]
        self.assertEqual(grouped["group"],baseline["group"])
        self.assertLess(grouped["row"]["coverage"],grouped["group"]["coverage"])
        self.assertLess(grouped["row"]["mean_width"],grouped["group"]["mean_width"])
