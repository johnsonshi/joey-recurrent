import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from joey_research.scheduler import Gate,Manifest,PolicyError,Queue,Split,TrialKey
from joey_research.worker import method_sha256
from test_scheduler import Clock,RUN,START,digest


class RefinementParity(unittest.TestCase):
    def test_native_measurement_cannot_admit_changed_payload_recurrence_or_test(self):
        with tempfile.TemporaryDirectory() as root:
            q=Queue.create(Path(root)/"toy.sqlite",run_id=RUN,start_utc=START,
                           gpu_classes=("A100-loop","A100-compact"),clock=Clock())
            self.addCleanup(q.close)
            payload=digest("physical")
            def register(k,split=Split.DEV):
                m={"window":[8,12],"reverse":False,"loop":{"method":"naive","k":k},
                   "compact_recipe":"select-20-late","compact_manifest_sha256":payload}
                key=TrialKey(payload,digest(split.value),digest("code"),method_sha256(m),1,digest("scorer"),"bf16")
                doc={"schema":"joey-refinement-trial/v1","run_id":RUN,"mode":"refinement","slot":1,"branch":"compact",
                     "trial_id":key.trial_id,"inputs_sha256":key.dataset,"key":asdict(key),"settings":m,"method_sha256":key.config}
                raw=json.dumps(doc,sort_keys=True,separators=(",",":")).encode()
                q.register(Manifest(hashlib.sha256(raw).hexdigest(),key,split,"A100-compact",1))
                q.attest(key.trial_id,Gate.OFFLINE_STAGING,digest("staging"+key.trial_id))
                return key.trial_id,raw
            tid,raw=register(1)
            self.assertIsNone(q.reserve(trial_id=tid))
            for wrong,data in (register(2),register(1,Split.TEST),(tid,raw.replace(payload.encode(),b"0"*64))):
                with self.assertRaises(PolicyError):q.reserve_refinement_control(wrong,data)
            self.assertFalse(q.outstanding())
            ticket=q.reserve_refinement_control(tid,raw)
            self.assertEqual((ticket.trial_id,ticket.split,ticket.slot),(tid,Split.DEV,1))
            self.assertIsNone(q.reserve_refinement_control(tid,raw))
