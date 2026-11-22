import contextlib
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from joey_research import worker
from joey_research.loop import LoopPlan
from joey_research.loop_analysis import Policy, Stop
from joey_research.worker import _packet, choice_request, compact_recipe, measure_choice


class WorkerTest(unittest.TestCase):
    row = {"id":"one", "group_id":"graph", "task":"choice", "prompt":"Which fits?", "choices":["Red","Green","Blue"]}

    def test_constructed_compact_plan_matches_its_independent_export_recipe(self):
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        for name in ("select-20","select-16","merge-16"):
            self.assertEqual(compact_recipe(name),json.loads((root/"configs/compact"/(name+".json")).read_bytes()))
        packet={"schema":1,"split":"discovery","inputs_sha256":"0"*64,"loop":{"k":1},"window":[8,12],
                "reverse":False,"deadline_utc":100,"compact_recipe":"select-20"}
        with self.assertRaisesRegex(ValueError,"independent"):_packet(json.dumps(packet).encode())
        packet["compact_manifest_sha256"]="1"*64
        self.assertEqual(_packet(json.dumps(packet).encode())[0]["compact_recipe"],"select-20")

    def test_reverse_is_mapped_back_and_does_not_send_bookkeeping(self):
        class Native:
            def branches(inner, state, questions, *, window):
                self.assertEqual(state, "Which fits?")
                self.assertEqual(list(questions["decision"]["criteria"].values()), ["Blue","Green","Red"])
                self.assertEqual(set(questions["decision"]), {"type","instructions","criteria"})
                branch = SimpleNamespace(labels=("opt_2","opt_1","opt_0"), layout=SimpleNamespace(input_ids=(1,2)),
                                         stock=lambda: [3,2,1], probabilities=lambda _: (.7,.2,.1),
                                         inference_context=contextlib.nullcontext)
                return {"decision":branch}
        pred, trace = measure_choice(Native(), self.row, (8,12), LoopPlan(), reverse=True)
        self.assertEqual(pred["probs"], [.1,.2,.7])
        self.assertEqual(trace["layer_calls"], 24)

    def test_reject_label_leak_and_final_packet(self):
        with self.assertRaises(ValueError):
            choice_request({**self.row, "gold_index":0}, (0,1,2))
        packet = {"schema":1,"split":"final","inputs_sha256":"0"*64,"window":[4,8],"loop":{"k":2},
                  "reverse":False,"deadline_utc":1}
        with self.assertRaisesRegex(ValueError, "discovery"):
            _packet(json.dumps(packet).encode())
        for k in (0,9):
            with self.assertRaises(ValueError):
                _packet(json.dumps({**packet,"split":"discovery","loop":{"k":k}}).encode())

    def test_same_compact_counterpart_charges_physical_coda(self):
        class Compact:
            physical_to_source = tuple(range(20))
            def branches(self, state, questions, *, window):
                return {"decision":SimpleNamespace(labels=tuple(questions["decision"]["criteria"]),
                       layout=SimpleNamespace(input_ids=(1,)), stock=lambda: 0, prelude=lambda: 0,
                       block=lambda h,s:h+1, coda=lambda h,s:h, fresh_state=lambda: object(),
                       probabilities=lambda h: (.5,.25,.25), inference_context=contextlib.nullcontext)}
        for k, expected in ((1,20),(2,24)):
            _,trace = measure_choice(Compact(), self.row, (8,12), LoopPlan(k=k))
            self.assertEqual(trace["layer_calls"],expected)

    def test_live_policy_pays_for_stop_readout_and_restores_original_order(self):
        class Native:
            def branches(self,state,questions,*,window):
                return {"decision":SimpleNamespace(labels=tuple(questions["decision"]["criteria"]),
                        layout=SimpleNamespace(input_ids=(1,2)), temperature=1., prelude=lambda:0,
                        block=lambda h,s:h+1,coda=lambda h,s:[4.,0.,-2.] if h==2 else [0.,0.,0.],
                        fresh_state=lambda:object(),inference_context=contextlib.nullcontext)}
        with patch.object(worker, "SlotOrder", wraps=worker.SlotOrder) as order:
            pred,trace=measure_choice(Native(),self.row,(8,12),LoopPlan(k=5),reverse=True,
                                      policy=Policy(stop=Stop("margin",0.5)),split="selection")
        self.assertEqual({c.args[0] for c in order.call_args_list}, {"selection"})
        self.assertGreater(pred["probs"][2],.9)
        self.assertEqual((trace["policy_decision_pass"],trace["layer_calls"],trace["callback_counts"]["coda_calls"]),(2,40,2))

    def test_selection_permit_verifies_complete_declared_settings_and_both_byte_pins(self):
        sha = lambda value: hashlib.sha256(value).hexdigest()
        settings = {"reverse": False, "window": [4, 8], "loop": {"k": 2}}
        mid = sha(b'{"loop":{"k":2},"reverse":false,"window":[4,8]}')
        self.assertEqual(worker.method_sha256(settings), mid)
        permit = {"schema": "joey-promotion-permit/v1", "split": "selection", "split_manifest_sha256": "a"*64,
                  "inputs_sha256": "b"*64, "methods": {mid: settings}}

        def admission(*, packet_updates=None, permit_updates=None, extra_methods=None,
                      expected_nom=None, expected_packet=None):
            doc = dict(permit, methods={**permit["methods"], **(extra_methods or {})})
            nom = json.dumps(dict(doc, **(permit_updates or {})), indent=2).encode()
            pkt = {"schema": 1, "split": "selection", "deadline_utc": 123, "inputs_sha256": permit["inputs_sha256"],
                   "permit_sha256": sha(nom), "method_sha256": mid, **settings, **(packet_updates or {})}
            raw = json.dumps(pkt).encode()
            return worker.selection_packet(raw, nom, packet_sha256=expected_packet or sha(raw),
                                           nomination_sha256=expected_nom or sha(nom))

        valid, _, _ = admission()
        with self.assertRaises(ValueError):
            _packet(json.dumps(valid).encode())
        for wrong in ({"window": [8, 12]}, {"loop": {"k": 2, "readout": "final"}}, {"reverse": True},
                      {"policy": {}}, {"compact_recipe": "select-20", "compact_manifest_sha256": "c"*64},
                      {"order_ensemble": {"numerator": 1, "denominator": 2, "seed": 3}},
                      {"method_sha256": "d"*64}, {"permit_sha256": "c"*64}, {"inputs_sha256": "f"*64},
                      {"family": "not-settings"}, {"split": "final"}, {"split": "discovery"}, {"split": "calibration"}):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                admission(packet_updates=wrong)
        for wrong in ({"permit_updates": {"split": "discovery"}}, {"permit_updates": {"split_manifest_sha256": ""}},
                      {"permit_updates": {"methods": {}}}, {"permit_updates": {"notes": "not allowed"}},
                      {"expected_nom": "0"*64}, {"expected_packet": "0"*64}, {"extra_methods": {"0"*64: settings}}):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                admission(**wrong)

        # Use independently pinned (even valid) optional settings, not just core fields.
        for method in (dict(settings, compact_recipe="select-20", compact_manifest_sha256="e"*64),
                       dict(settings, policy={"stop": {"kind": "margin", "threshold": 1.0}}),
                       dict(settings, loop={"k": 2, "method": "zero_extra"},
                            order_ensemble={"numerator": 1, "denominator": 2, "seed": 7})):
            key = worker.method_sha256(method)
            admission(extra_methods={key: method}, packet_updates={**method, "method_sha256": key})
            for optional in ("policy", "compact_recipe", "compact_manifest_sha256", "order_ensemble"):
                if optional in method:
                    altered = {k: v for k, v in method.items() if k != optional}
                    with self.assertRaises(ValueError):
                        admission(extra_methods={key: method}, packet_updates={**altered, "method_sha256": key})

    def test_spend_is_by_whole_groups_stratified_without_using_rows_or_scores(self):
        # At 2/3, A:1+1/3, B:1+1/3, C:0+2/3; three of five, not four.
        groups = [("A", "a1"), ("A", "a2"), ("B", "b1"), ("B", "b2"), ("C", "c1")]
        cfg = {"numerator": 2, "denominator": 3, "seed": 42}
        choose = lambda pairs, c=cfg: worker.ensemble_groups(pairs, c, split="selection")
        def ranked(task, group):
            raw = json.dumps(["joey-order-ensemble/v1", "selection", 42, task, group], separators=(",", ":")).encode()
            return hashlib.sha256(raw).hexdigest()
        expected = {"c1", min(("a1", "a2"), key=lambda g: ranked("A", g)),
                    min(("b1", "b2"), key=lambda g: ranked("B", g))}
        self.assertEqual(choose(groups), expected)
        self.assertEqual(choose(groups[::-1] + groups[:2] * 10), expected)
        self.assertEqual(choose(groups, dict(cfg, numerator=0)), set())
        self.assertEqual(choose(groups, dict(cfg, numerator=3)), {g for _, g in groups})
        with self.assertRaises(ValueError):
            choose(groups + [("C", "a1")])

    def test_ensemble_is_a_bounded_split_only_control(self):
        cfg = {"numerator": 1, "denominator": 4, "seed": 0}
        base = {"schema": 1, "split": "discovery", "inputs_sha256": "d"*64, "window": [8, 12],
                "loop": {"k": 2, "method": "zero_extra"}, "reverse": False, "deadline_utc": 123, "order_ensemble": cfg}
        self.assertEqual(_packet(json.dumps(base).encode())[2], LoopPlan(k=2, method="zero_extra"))
        for wrong in ({"reverse": True}, {"policy": {}}, {"compact_manifest_sha256": "e"*64},
                      {"loop": {"k": 1, "method": "zero_extra"}}, {"loop": {"k": 2}},
                      {"loop": {"k": 2, "method": "zero_extra", "readout": "diagnostic"}},
                      *({"order_ensemble": dict(cfg, **edit)} for edit in (
                          {"seed": True}, {"seed": 2**32}, {"numerator": 1.0}, {"numerator": -1}, {"numerator": 5},
                          {"denominator": 0}, {"denominator": 10001}, {"rank_by": "confidence"}))):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                _packet(json.dumps(dict(base, **wrong)).encode())
