import unittest

from joey_research.evaluation import EvaluationError,input_sha256,label_row
from joey_research.probe_replay import audit,fit,load,partition,simulate


class Replay(unittest.TestCase):
    def setUp(self):
        self.rows=[{"id":str(i),"group_id":str(i//2),"task":"route","prompt":f"toy{i}","choices":["east","west"]} for i in range(16)]
        p,t=[],[]
        for i,r in enumerate(self.rows):
            first=[.75,.25] if i%2==0 else [.375,.625]
            seq=[first]+[[.9,.1]]*5
            p.append({"id":r["id"],"input_sha256":input_sha256(r),"probs":seq[-1]})
            t.append({"id":r["id"],"packet_sha256":"a"*64,"callback_counts":{"stock_calls":0,"prelude_calls":1,"block_calls":6,"coda_calls":6},
                      "layer_calls":104,"readouts":[{"block_evaluations":j+1,"kind":"iterate" if j else "first_pass","probs":z} for j,z in enumerate(seq)]})
        self.raw=load(self.rows,p,t,packet_sha256="a"*64,physical_depth=24);self.p=p;self.t=t
        self.sets=partition(self.rows,fit_per_task=5,audit_per_task=3)
        self.halves={k:[r for r in self.raw if self.sets[r.row["group_id"]]==k] for k in ("fit","audit")}
        self.gold={k:[label_row(x.row,0) for x in xs] for k,xs in self.halves.items()}

    def test_stop_at_margin_boundary_and_account_two_codas_not_six(self):
        pred,cost=simulate(self.raw,cap=4,threshold=.5,physical_depth=24)
        self.assertEqual(pred[0]["probs"],[.75,.25])
        self.assertEqual(pred[1]["probs"],[.9,.1])
        self.assertEqual(cost["continued"],8)
        self.assertEqual(cost["coda_calls"],24)
        self.assertEqual(cost["mean_logical_layers"],36)
        self.assertFalse(cost["independently_timed"])
        self.t[0]["readouts"][2]["kind"]="substep"
        with self.assertRaises(EvaluationError):load(self.rows,self.p,self.t,packet_sha256="a"*64,physical_depth=24)

    def test_frozen_fit_has_a_disjoint_audit_and_cannot_peek(self):
        rec=fit(self.halves["fit"],self.gold["fit"],sets=self.sets,physical_depth=24)
        self.assertEqual(rec["selected"],{"cap":2,"threshold":.5})
        result=audit(self.halves["audit"],self.gold["audit"],sets=self.sets,fit_record=rec,physical_depth=24,reps=10)
        self.assertTrue(result["qualified"])
        self.assertEqual(result["comparisons"]["1"]["transitions"]["corrections"],3)
        with self.assertRaises(EvaluationError):fit(self.raw,self.gold["fit"],sets=self.sets,physical_depth=24)
        with self.assertRaises(EvaluationError):fit(self.halves["fit"],self.gold["audit"],sets=self.sets,physical_depth=24)
