from contextlib import nullcontext
from types import SimpleNamespace
import unittest

from joey_research.loop import LoopPlan
from joey_research.probe_runtime import validate_trace
from joey_research.worker import measure_choice,method_sha256


class Hidden:
    def __init__(self,depth):
        self.depth=depth

    def clone(self):
        return Hidden(self.depth)


class ProbeIntegration(unittest.TestCase):
    def test_actual_single_probe_and_forensic_trace_agree_at_both_exits(self):
        row={"task":"routing_2hop","id":"toy","group_id":"toy-group","prompt":"Invented case","choices":["a","b","c","d"]}
        for threshold,stop in ((.1,True),(.5,False)):
            events=[]
            def coda(h,state):
                events.append(h.depth)
                return [.4,.3,.2,.1] if h.depth==1 else [.7,.1,.1,.1]
            b=SimpleNamespace(labels=tuple("opt_"+str(i) for i in range(4)),layout=SimpleNamespace(input_ids=[1,2,3]),
                              temperature=1,inference_context=nullcontext,prelude=lambda:Hidden(0),fresh_state=object,
                              block=lambda h,s:Hidden(h.depth+1),coda=coda,probabilities=lambda z:z)
            n=SimpleNamespace(branches=lambda *a,**k:{"decision":b})
            p,t=measure_choice(n,row,(8,12),LoopPlan(k=4),probe_margin=threshold,split="one-probe")
            s={"loop":{"k":4,"method":"naive"},"window":[8,12],"probe_margin":threshold,"reverse":False}
            t.update(packet_sha256="a"*64,wall_ms=1,cuda_event_ms=1)
            validate_trace(t,p["probs"],settings=s,packet_sha256="a"*64)
            self.assertEqual(events,[1] if stop else [1,4])
            for broken in (dict(t,layer_calls=t["layer_calls"]+1),
                           dict(t,probe={**t["probe"],"first_margin":.25}),
                           dict(t,probe={**t["probe"],"exit_pass":4 if stop else 1})):
                with self.assertRaises(ValueError):validate_trace(broken,p["probs"],settings=s,packet_sha256="a"*64)
        for extra in ({"probe_margin":True},{"order_ensemble":{}},{"reverse":True},{"policy":{}}):
            with self.assertRaises(ValueError):method_sha256({**s,**extra})
