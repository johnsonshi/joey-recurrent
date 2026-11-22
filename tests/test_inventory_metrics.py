import copy
import unittest

from joey_research import evaluation as ev
from joey_research.inventory_metrics import Menu, decisions, paired, summarize


def fixture(kind, *, improved=False):
    rows, gold, pred, typed, menus = [], [], [], [], {}
    for g in range(2):
        for v in range(2):
            k = f"item-{g}-{v}"
            if kind == "score":
                keys = meanings = ("0", "1", "2", "3")
                p = [.5,.1,.1,.3] if not improved else [0.,0.,1.,0.]
                y = 2
                answer = {"type":kind, "score":round(sum(i*z for i,z in enumerate(p)),6), "probabilities":dict(zip(keys,p))}
            elif kind == "choice":
                meanings = ("a","b","c","d","tie") if v == 0 else ("tie","d","c","b","a")
                keys = tuple("key_"+m for m in meanings)
                probs = {"a":.0, "b":.1, "c":.1, "d":.4, "tie":.4}
                p = [probs[m] for m in meanings]
                y = meanings.index("tie")
                answer = {"type":kind, "choice":"key_tie", "probabilities":dict(zip(keys,p))}
            else:
                keys = meanings = ("yes","no")
                p = [.4999998,.5000002] if not improved else [0.,1.]
                y = 0
                answer = {"type":"noul","noul":round(p[0],6)}
            r = {"id":k,"group_id":str(g),"task":kind,"prompt":"invented "+k,"choices":list(keys)}
            base = {"id":k,"input_sha256":ev.input_sha256(r)}
            rows.append(r); gold.append(ev.label_row(r,y))
            pred.append({**base,"probs":p}); typed.append({**base,"answer":answer})
            menus[k] = Menu(1,2,kind,keys,meanings)
    return rows, gold, pred, typed, menus


def records(f):
    return decisions(*f[:4], menus=f[4])


class InventoryMeasures(unittest.TestCase):
    def test_native_ordinal_expectation_is_not_its_modal_class(self):
        raw = records(fixture("score"))
        r = summarize(raw)["score"]
        self.assertEqual(r["means"]["argmax_accuracy"], 0)
        self.assertAlmostEqual(r["means"]["ordinal_mae"], .8)
        self.assertAlmostEqual(r["means"]["ordinal_rps"], (.5**2+.6**2+.3**2)/3)
        self.assertEqual(r["support"]["mean_abs_within_group_expected_score_difference"], 0)
        p = paired(raw, records(fixture("score",improved=True)), reps=32)["score"]
        self.assertEqual(p["argmax_corrections"],4)
        self.assertAlmostEqual(p["deltas"]["ordinal_rmse"]["delta"],-.8)
        self.assertEqual(p["deltas"]["ordinal_rmse"]["ci95"],[-.8,-.8])

    def test_rounded_binary_threshold_and_exact_zero_native_loss(self):
        raw = records(fixture("noul"))
        r = summarize(raw)["noul"]
        self.assertEqual(r["means"]["binary_accuracy"],1)
        self.assertEqual(r["support"]["native_not_first_argmax"],4)
        self.assertEqual(r["means"]["binary_brier"],.25)
        self.assertIsNone(r["confidence"]["0.99"]["conditional_error"])
        perfect_no = records(fixture("noul",improved=True))
        self.assertIsNone(summarize(perfect_no)["noul"]["means"]["binary_log_loss"])
        self.assertEqual(paired(raw,perfect_no,reps=8)["noul"]["deltas"]["binary_log_loss"]["status"],"nonfinite_loss")

    def test_declared_choice_uses_its_actual_key_across_reordered_menus(self):
        f=fixture("choice");r=summarize(records(f))["choice"]
        self.assertEqual(r["support"]["correct_declared_tie"],4)
        self.assertEqual(r["support"]["top_semantic_agreement_groups"],0)
        self.assertEqual(r["support"]["native_declared_agreement_groups"],2)
        self.assertEqual(r["means"]["argmax_accuracy"],.5)
        for bad in ("pair","hash","answer","sequence"):
            z=copy.deepcopy(f)
            if bad=="pair":z[1][0]=ev.label_row(z[0][0],0)
            if bad=="hash":z[2][0]["input_sha256"]="0"*64
            if bad=="answer":z[3][0]["answer"]["probabilities"]["key_b"]=.2
            if bad=="sequence":z[3].reverse()
            with self.subTest(bad=bad), self.assertRaises(ValueError):records(z)
