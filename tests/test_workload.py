import hashlib
import json
from dataclasses import replace
import unittest

from joey_research.scheduler import Admission, Split
from joey_research.workload import AdmissionError, BoundTarget, compile_discovery


class WorkloadTest(unittest.TestCase):
    def setUp(self):
        self.target=BoundTarget("study-fixture","namespace-uid","fixture-run","mirror.example.invalid","sha256:"+"6"*64,
                                "output-claim","volume-uid","accelpool",("fixture-a","fixture-b"),
                                (None,("compute-class","exclusive","NoSchedule")))
        self.ticket=Admission("a"*64,"b"*64,"c"*64,"d"*64,Split.DEV,1,0,1,"A100-loop",10,30,100,20000)
        self.fields={"schema":1,"split":"discovery","inputs_sha256":"e"*64,"loop":{"method":"naive","k":2},
                     "window":[8,12],"reverse":False,"deadline_utc":100}

    def compile(self,ticket=None,target=None,**updates):
        raw=json.dumps(self.fields,separators=(",",":")).encode()
        params=dict(now=11,expected_run="a"*64,namespace_uid="namespace-uid",claim_uid="volume-uid",
                    manifest_digest="d"*64,packet_digest=hashlib.sha256(raw).hexdigest(),split_digest="e"*64)
        return compile_discovery(target or self.target,ticket or self.ticket,raw,**{**params,**updates})

    def test_each_branch_has_one_scheduler_placed_nonprivileged_job(self):
        for slot in (0,1):
            t=replace(self.ticket,slot=slot,gpu_class=("A100-loop","A100-compact")[slot])
            if slot:self.fields["compact_manifest_sha256"]="f"*64
            cm,job=self.compile(t)
            self.assertEqual(job["metadata"]["namespace"],self.target.namespace)
            self.assertTrue(cm["immutable"])
            spec=job["spec"]
            self.assertEqual((spec["parallelism"],spec["completions"],spec["backoffLimit"],spec["activeDeadlineSeconds"]),(1,1,0,89))
            pod=spec["template"]["spec"]; container,=pod["containers"]
            self.assertNotIn("nodeName",pod)
            self.assertEqual(pod["nodeSelector"]["kubernetes.io/hostname"],self.target.hosts[slot])
            self.assertFalse(pod["automountServiceAccountToken"])
            self.assertEqual(container["resources"]["limits"]["nvidia.com/gpu"],"1")
            self.assertEqual(container["command"],["python","-m","joey_research.worker"])
            self.assertTrue(container["securityContext"]["readOnlyRootFilesystem"])
            self.assertTrue(container["image"].endswith("@"+self.target.image_digest))
            self.assertEqual(pod.get("tolerations",[]),[] if slot==0 else
                             [{"key":"compute-class","value":"exclusive","operator":"Equal","effect":"NoSchedule"}])
            self.assertNotIn("hostPath",json.dumps(pod))
            self.assertEqual(next(v["persistentVolumeClaim"] for v in pod["volumes"] if v["name"]=="results"),
                             {"claimName":"output-claim"})

    def test_refuse_stale_identity_time_or_different_bytes(self):
        for change in ({"now":30},{"manifest_digest":"0"*64},{"expected_run":"0"*64},{"namespace_uid":"other"},
                       {"claim_uid":"other"},{"packet_digest":"0"*64},{"split_digest":"0"*64}):
            with self.subTest(change=change),self.assertRaises(AdmissionError):self.compile(**change)
        for change in ({"split":Split.TEST},{"gpu_class":"A100-compact"},{"run_deadline":101}):
            with self.assertRaises(AdmissionError):self.compile(replace(self.ticket,**change))
        for ns in ("default","kube-system"):
            with self.assertRaises(AdmissionError):replace(self.target,namespace=ns)
        shared=replace(self.target,namespace="default",shared_approval_sha256="5"*64)
        with self.assertRaises(AdmissionError):self.compile(target=shared)
        cm,job=self.compile(target=shared,namespace_approval_digest="5"*64)
        for obj in (cm,job):
            self.assertEqual(obj["metadata"]["namespace"],"default")
            self.assertEqual(obj["metadata"]["labels"]["research.run/id"],shared.run_label)
        self.fields["deadline_utc"]=101
        with self.assertRaises(AdmissionError):self.compile()

    def test_pull_secret_and_ordinary_priority_are_bound_not_mutations(self):
        target=replace(self.target,image_pull_secret=("study-pull","uid-one"),ordinary_priority_proof="6"*64)
        for kwargs in ({"pull_secret_uid":"wrong","scheduling_proof":"6"*64},
                       {"pull_secret_uid":"uid-one","scheduling_proof":"7"*64}):
            with self.assertRaises(AdmissionError):self.compile(target=target,**kwargs)
        _,job=self.compile(target=target,pull_secret_uid="uid-one",scheduling_proof="6"*64)
        pod=job["spec"]["template"]["spec"]
        self.assertEqual(pod["imagePullSecrets"],[{"name":"study-pull"}])
        self.assertNotIn("serviceAccountName",pod)
        self.assertNotIn("preemptionPolicy",pod)
        self.assertNotIn("priorityClassName",pod)
