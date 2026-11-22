import unittest

from benchmarks.routing import iter_inputs,inverse_label
from benchmarks.routing_consistency import group_outcomes,contrast
from joey_research.evaluation import EvaluationError


class RouteConsistency(unittest.TestCase):
    def test_semantic_alignment_and_graph_resampling(self):
        rows=list(iter_inputs(namespace="invented-pairs",seed=7,groups_per_depth=3,depths=(2,)))
        gold=[inverse_label(r) for r in rows]
        def make(row,y,perfect):
            opts=sorted(row["choices"])
            i=opts.index(row["choices"][y["gold_index"]])
            bad=row["prompt"].startswith("Required destination:") and not perfect
            pick=opts[(i+bad)%len(opts)]
            return {"id":y["id"],"input_sha256":y["input_sha256"],"probs":[float(x==pick) for x in row["choices"]]}
        raw=[make(r,y,False) for r,y in zip(rows,gold,strict=True)]
        better=[make(r,y,True) for r,y in zip(rows,gold,strict=True)]
        a,b=group_outcomes(rows,gold,raw),group_outcomes(rows,gold,better)
        d=contrast(a,b,reps=20)
        self.assertEqual(d["equal_task"]["semantic_agreement"],{"delta":1.,"ci95":[1.,1.]})
        self.assertEqual(d["equal_task"]["total_variation"]["delta"],-1.)
        self.assertEqual(d["equal_task"]["arrow_accuracy"]["delta"],0.)
        self.assertEqual(d["graphs_per_task"],{"routing_2hop":3})
        with self.assertRaises(EvaluationError):group_outcomes(rows[:-1],gold[:-1],raw[:-1])
