import contextlib
import json
import unittest
from types import SimpleNamespace

from joey_research.loop import LoopPlan
from joey_research.worker import _packet, choice_request, measure_choice


class WorkerTest(unittest.TestCase):
    row = {"id":"one", "group_id":"graph", "task":"choice", "prompt":"Which fits?", "choices":["Red","Green","Blue"]}

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
