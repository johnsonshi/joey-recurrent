import unittest

from benchmarks.fixed_horizon import HOPS,adjudicate,check_blind_panel,group_seeds,iter_inputs
from joey_research.evaluation import EvaluationError


class FixedHorizon(unittest.TestCase):
    def test_graph_and_menu_are_identical_for_all_horizons_and_answers_balance(self):
        rows=list(iter_inputs(27,3));m=group_seeds(27,3)
        self.assertEqual(check_blind_panel(rows,m)["rows"],96)
        self.assertTrue(set(m).isdisjoint(group_seeds(29,3)))
        cells=set();counts={}
        for r in rows:
            y,design=adjudicate(r);key=(r["group_id"],design["entry"],design["rendering"],design["hop"])
            self.assertNotIn(key,cells);cells.add(key)
            self.assertEqual(r["choices"][y["gold_index"]],"Enter at "+design["entry"])
            counts[design["hop"]]=counts.get(design["hop"],0)+1
        self.assertEqual(counts,{h:24 for h in HOPS})
        bad=[dict(x) for x in rows];bad[0]["choices"]=bad[0]["choices"][1:]+bad[0]["choices"][:1]
        with self.assertRaises(EvaluationError):check_blind_panel(bad,m)
        with self.assertRaises(EvaluationError):check_blind_panel(rows[1:],m)

    def test_checker_solves_goal_from_display_not_id_or_generator_metadata(self):
        r=next(iter(iter_inputs(44,1)));y,d=adjudicate(r)
        lines=r["prompt"].splitlines()
        a,b=lines[2].split(" -> ");bad={**r,"prompt":"\n".join([*lines[:2],f"{a} -> {a}",*lines[3:]])}
        with self.assertRaises(EvaluationError):adjudicate(bad)
        altered={**r,"id":"unrelated-id","group_id":"unrelated-group","choices":list(reversed(r["choices"]))}
        q,other=adjudicate(altered)
        self.assertEqual(other,d);self.assertEqual(q["gold_index"],3-y["gold_index"]);self.assertNotEqual(q["input_sha256"],y["input_sha256"])
