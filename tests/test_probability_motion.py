import unittest

from joey_research.evaluation import EvaluationError,input_sha256,label_row
from joey_research.probability_motion import compare


class ProbabilityMotion(unittest.TestCase):
    def test_mass_and_concentration_identity_and_distinct_error_denominators(self):
        rows=[{"id":str(i),"group_id":str(i),"task":"toy","prompt":f"case {i}","choices":["a","b","c"]} for i in range(3)]
        ys=[label_row(r,0) for r in rows]
        def stream(p):return [{"id":r["id"],"input_sha256":input_sha256(r),"probs":x} for r,x in zip(rows,p,strict=True)]
        old=stream([[.4,.5,.1],[.1,.6,.3],[.6,.2,.2]])
        new=stream([[.6,.3,.1],[.02,.03,.95],[.7,.2,.1]])
        r=compare(rows,ys,old,new)["micro"]
        self.assertAlmostEqual(r["mean_brier_change"],r["mean_truth_term"]+r["mean_concentration_term"])
        self.assertAlmostEqual(sum(s["brier_contribution_to_panel_mean"] for s in r["transitions"].values()),r["mean_brier_change"])
        self.assertEqual(r["baseline_top_by_own_outcome"]["incorrect"]["n"],2)
        self.assertEqual(r["candidate_top_by_own_outcome"]["incorrect"]["mean"],.95)
        self.assertEqual(r["persistently_wrong"]["new_threshold_crossings"]["0.9"],1)
        self.assertIsNone(r["transitions"]["corrupted"]["mean_brier_change"])
        with self.assertRaises(EvaluationError):compare(rows,ys,old,new[:-1])
