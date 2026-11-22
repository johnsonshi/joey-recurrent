"""Compile one verified discovery admission into bounded, namespaced Kubernetes objects. No API client."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from .scheduler import Admission, Split
from .worker import _packet


class AdmissionError(ValueError):
    pass


def _require(condition, category):
    if not condition:
        raise AdmissionError(category)


def _dns(value):
    return isinstance(value, str) and re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", value)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)


@dataclass(frozen=True)
class BoundTarget:
    """Values come from a private, independently verified control plane, never from a trial proposal."""
    namespace: str
    namespace_uid: str
    run_label: str
    registry: str
    image_digest: str
    # Existing, run-exclusive durable claim; this compiler cannot create storage or claim credentials.
    result_claim: str
    result_claim_uid: str
    pool: str
    hosts: tuple[str, str]
    # Each item is the exact optional accelerator taint (key, value, effect) on that slot.
    taints: tuple[tuple[str, str, str] | None, tuple[str, str, str] | None]
    shared_approval_sha256: str | None = None

    def __post_init__(self):
        _require(_dns(self.namespace) and not self.namespace.startswith("kube-") and
                 (_sha(self.shared_approval_sha256) if self.namespace=="default" else self.shared_approval_sha256 is None),
                 "verified namespace selection required")
        _require(self.namespace_uid and _dns(self.run_label) and _dns(self.pool), "invalid local scope")
        _require(_dns(self.result_claim) and self.result_claim_uid, "preverified existing private output volume required")
        _require(isinstance(self.hosts, tuple) and len(self.hosts) == 2 and len(set(self.hosts)) == 2 and
                 all(_dns(h) for h in self.hosts), "two distinct verified pool hosts required")
        _require(isinstance(self.registry, str) and "." in self.registry and len(self.registry) < 254 and
                 all(_dns(part) for part in self.registry.split(".")), "verified registry hostname required")
        _require(isinstance(self.image_digest, str) and self.image_digest.startswith("sha256:") and
                 _sha(self.image_digest[7:]), "immutable image digest required")
        _require(isinstance(self.taints, tuple) and len(self.taints) == 2, "one taint record per host required")
        for taint in self.taints:
            if taint is not None:
                _require(isinstance(taint, tuple) and len(taint) == 3, "invalid slot taint")
                key,value,effect = taint
                _require(isinstance(key, str) and re.fullmatch(r"(?:[a-z0-9.-]+/)?[a-z][a-z0-9.-]{0,62}", key)
                         and not key.startswith(("node.kubernetes.io/", "node-role.kubernetes.io/", "CriticalAddonsOnly"))
                         and _dns(value) and effect == "NoSchedule", "not a scoped scheduling taint")


def compile_discovery(target: BoundTarget, ticket: Admission, packet: bytes, *, now: int,
                      expected_run: str, namespace_uid: str, claim_uid: str, manifest_digest: str,
                      packet_digest: str, split_digest: str, namespace_approval_digest: str | None = None) -> tuple[dict, dict]:
    """Build a single packet ConfigMap + Job. Verification of mirror, RBAC, custody and exclusivity precedes this."""
    _require(type(ticket) is Admission and type(now) is int and ticket.split is Split.DEV, "discovery admission required")
    _require(target.namespace_uid == namespace_uid and ticket.run_id == expected_run and _sha(expected_run),
             "project target or run changed")
    _require(target.shared_approval_sha256 == namespace_approval_digest, "namespace approval differs from bound target")
    _require(claim_uid == target.result_claim_uid, "durable output identity changed")
    _require(_sha(manifest_digest) and ticket.manifest_sha == manifest_digest and _sha(ticket.trial_id) and _sha(ticket.token),
             "manifest or ticket hash mismatch")
    _require(type(ticket.slot) is int and ticket.slot in (0,1) and ticket.generation >= 1 and ticket.attempt in (1,2),
             "invalid slot reservation")
    _require(ticket.gpu_class == ("A100-loop","A100-compact")[ticket.slot], "slot allocation mismatch")
    _require(ticket.issued_at <= now < ticket.admit_before <= ticket.hard_stop <= ticket.run_deadline and
             ticket.hard_stop <= ticket.run_deadline - 3*3600 and 0 < ticket.hard_stop - now <= 60*60,
             "outside development admission budget")
    _require(isinstance(packet, bytes) and len(packet) <= 8192 and hashlib.sha256(packet).hexdigest() == packet_digest,
             "packet differs from verified input")
    obj, _, _ = _packet(packet)
    _require(_sha(split_digest) and obj["inputs_sha256"] == split_digest and
             type(obj["deadline_utc"]) is int and obj["deadline_utc"] == ticket.hard_stop,
             "discovery split or absolute cutoff differs")
    compact = "compact_manifest_sha256" in obj
    _require(compact == (ticket.slot == 1), "packet belongs to the other branch")
    name = "jr-" + ticket.token[:40]
    labels = {"app.kubernetes.io/part-of":"joey-recurrent", "research.run/id":target.run_label,
              "research.run/slot":str(ticket.slot)}
    annotations = {"research.run/token":ticket.token, "research.run/manifest":ticket.manifest_sha,
                   "research.run/packet":packet_digest, "research.run/generation":str(ticket.generation)}
    def metadata(n):
        return {"name":n,"namespace":target.namespace,"labels":dict(labels),"annotations":dict(annotations)}
    cm = {"apiVersion":"v1","kind":"ConfigMap","metadata":metadata(name+"-input"),"immutable":True,
          "data":{"packet.json":packet.decode("utf-8")}}
    args = ["--model-dir","/staged/model","--inputs","/staged/inputs/discovery.jsonl",
            "--packet","/admission/packet.json","--packet-sha256",packet_digest,"--output","/results/"+ticket.token]
    if compact:
        args += ["--compact-dir","/staged/compact"]
    resource = {"cpu":"2","memory":"24Gi","ephemeral-storage":"6Gi","nvidia.com/gpu":"1"}
    container = {"name":"research","image":target.registry+"/study/decision-worker@"+target.image_digest,
                 "imagePullPolicy":"IfNotPresent", "command":["python","-m","joey_research.worker"],
                 "args":args,
                 "resources":{"requests":dict(resource),"limits":dict(resource)},
                 "env":[{"name":n,"value":v} for n,v in (
                     ("HF_HUB_OFFLINE","1"),("HF_DATASETS_OFFLINE","1"),("TRANSFORMERS_OFFLINE","1"),
                     ("HF_HUB_DISABLE_TELEMETRY","1"),("DO_NOT_TRACK","1"),("PYTHONDONTWRITEBYTECODE","1"),
                     ("HOME","/work"),("XDG_CACHE_HOME","/work/cache"),("TMPDIR","/work/tmp"))],
                 "securityContext":{"allowPrivilegeEscalation":False,"readOnlyRootFilesystem":True,
                                    "capabilities":{"drop":["ALL"]}},
                 "volumeMounts":[{"name":"work","mountPath":"/work"},
                                 {"name":"results","mountPath":"/results"},
                                 {"name":"packet","mountPath":"/admission","readOnly":True}]}
    pod = {"automountServiceAccountToken":False, "enableServiceLinks":False, "restartPolicy":"Never",
           "preemptionPolicy":"Never", "terminationGracePeriodSeconds":0,
           "securityContext":{"runAsNonRoot":True,"runAsUser":65532,"runAsGroup":65532,
                              "seccompProfile":{"type":"RuntimeDefault"}},
           "nodeSelector":{"kubernetes.azure.com/agentpool":target.pool,"kubernetes.io/hostname":target.hosts[ticket.slot]},
           "containers":[container],"volumes":[{"name":"work","emptyDir":{"sizeLimit":"6Gi"}},
                         {"name":"results","persistentVolumeClaim":{"claimName":target.result_claim}},
                         {"name":"packet","configMap":{"name":name+"-input","items":[{"key":"packet.json","path":"packet.json"}]}}]}
    if target.taints[ticket.slot]:
        key,value,effect=target.taints[ticket.slot]
        pod["tolerations"]=[{"key":key,"operator":"Equal","value":value,"effect":effect}]
    job = {"apiVersion":"batch/v1","kind":"Job","metadata":metadata(name),
           "spec":{"backoffLimit":0,"parallelism":1,"completions":1,"activeDeadlineSeconds":ticket.hard_stop-now,
                   "template":{"metadata":{"labels":dict(labels),"annotations":dict(annotations)},"spec":pod}}}
    return cm, job
