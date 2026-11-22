"""Development-only selection and audit of a one-intermediate-coda policy from completed readouts."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
import re

from . import evaluation as ev


@dataclass(frozen=True)
class Recorded:
    row: dict
    digest: str
    passes: tuple[tuple[float, ...], ...]


def load(inputs, predictions, traces, *, packet_sha256, physical_depth, window=(8,12)):
    if (not isinstance(packet_sha256,str) or not re.fullmatch("[0-9a-f]{64}",packet_sha256)
            or physical_depth not in (20,24) or tuple(window)!=(8,12)):
        raise ev.EvaluationError("expected a pinned six-pass diagnostic in the original physical window")
    rows=list(inputs)
    ix=ev._index_inputs(rows)
    out=[]
    for row,p,t in zip(rows,predictions,traces,strict=True):
        k=row["id"]
        if (p.keys()!={"id","input_sha256","probs"} or p["id"]!=k or p["input_sha256"]!=ix[k][1] or t["id"]!=k
                or t["packet_sha256"]!=packet_sha256 or t["callback_counts"]!={"stock_calls":0,"prelude_calls":1,"block_calls":6,"coda_calls":6}
                or any(type(v) is not int for v in t["callback_counts"].values())
                or type(t["layer_calls"]) is not int or t["layer_calls"]!=8+6*4+6*(physical_depth-12)):
            raise ev.EvaluationError("trajectory not bound to this row, method, and physical execution")
        vs=[]
        for i,d in enumerate(t["readouts"],1):
            if (d.keys()!={"block_evaluations","kind","probs"} or type(d["block_evaluations"]) is not int or d["block_evaluations"]!=i
                    or d["kind"]!=("first_pass" if i==1 else "iterate") or not isinstance(d["probs"],list)
                    or len(d["probs"])!=len(row["choices"]) or any(type(x) not in (int,float) or not 0<=x<=1 for x in d["probs"])
                    or abs(math.fsum(d["probs"])-1)>1e-6):
                raise ev.EvaluationError("invalid completed-pass vector")
            vs.append(tuple(d["probs"]))
        if len(vs)!=6 or list(vs[-1])!=p["probs"]:
            raise ev.EvaluationError("not an exactly recorded six-pass endpoint")
        out.append(Recorded(row,ix[k][1],tuple(vs)))
    return out


def partition(inputs, *, fit_per_task, audit_per_task):
    ix=ev._index_inputs(inputs)
    if any(type(n) is not int or n<1 for n in (fit_per_task,audit_per_task)):
        raise ev.EvaluationError("partition sizes must be positive integer graph counts")
    by_task=defaultdict(set)
    for r,_ in ix.values():
        by_task[r["task"]].add(r["group_id"])
    sets={}
    for groups in by_task.values():
        if len(groups)!=fit_per_task+audit_per_task:
            raise ev.EvaluationError("wrong source group count")
        ordered=sorted(groups,key=lambda g:(hashlib.sha256(("joey-first-margin-v1:"+g).encode()).digest(),g))
        sets.update({g:"fit" if i<fit_per_task else "audit" for i,g in enumerate(ordered)})
    return sets


def _projection(trajectory, depth):
    return {"id":trajectory.row["id"],"input_sha256":trajectory.digest,"probs":list(trajectory.passes[depth-1])}


def simulate(recorded, *, cap, threshold, physical_depth):
    if (type(cap) is not int or cap not in (2,3,4,6) or type(threshold) not in (int,float)
            or threshold not in (0,.1,.25,.5,.75,1.) or type(physical_depth) is not int or physical_depth not in (20,24)):
        raise ev.EvaluationError("one-probe setting outside the frozen grid")
    preds=[];continued=0
    for x in recorded:
        a,b=sorted(x.passes[0],reverse=True)[:2]
        proceed=a-b<threshold
        continued+=proceed
        preds.append(_projection(x,cap if proceed else 1))
    if not preds:
        raise ev.EvaluationError("empty replay sample")
    n=len(preds)
    # A continuation pays one additional coda and cap-1 block traversals.
    layers=n*physical_depth+continued*((cap-1)*4+physical_depth-12)
    return preds,{"n":n,"continued":continued,"continue_fraction":continued/n,"logical_layers":layers,
                  "mean_logical_layers":layers/n,"block_calls":n+continued*(cap-1),"coda_calls":n+continued,
                  "independently_timed":False}


def _eligible(continued,n,visits,depth):
    return n<=10*continued<=9*n and visits<=n*(depth+12)


def _membership(recorded, sets, which):
    ids={x.row["id"] for x in recorded}
    seen={x.row["group_id"] for x in recorded}
    if (not ids or len(ids)!=len(recorded) or seen!={g for g,part in sets.items() if part==which}
            or set(sets.values())!={"fit","audit"}):
        raise ev.EvaluationError("wrong group partition")


def fit(recorded, labels, *, sets, physical_depth):
    """This call receives only fit graphs and their labels. It never takes audit data."""
    _membership(recorded,sets,"fit")
    rows=[x.row for x in recorded];labels=list(labels);table=[]
    for cap in (2,3,4,6):
        for threshold in (0.,.1,.25,.5,.75,1.):
            preds,cost=simulate(recorded,cap=cap,threshold=threshold,physical_depth=physical_depth)
            score=ev.evaluate(rows,labels,{"policy":preds},baseline="policy",bootstrap_reps=0)
            if not score["all_distributions_valid"]:
                raise ev.EvaluationError("invalid replay distribution")
            mi=score["arms"]["policy"]["macro"]
            table.append({"cap":cap,"threshold":threshold,"accuracy":mi["accuracy"],"brier":mi["brier"],
                          "eligible":_eligible(cost["continued"],cost["n"],cost["logical_layers"],physical_depth),**cost})
    ranked=sorted((x for x in table if x["eligible"]),
                  key=lambda x:(-x["accuracy"],x["brier"],x["logical_layers"],x["cap"],x["threshold"]))
    return {"split":"fit","fit_ids":sorted(r["id"] for r in rows),"grid":table,
            "selected":{k:ranked[0][k] for k in ("cap","threshold")} if ranked else None}


def audit(recorded, labels, *, sets, fit_record, physical_depth, reps=2000, seed=4117):
    _membership(recorded,sets,"audit")
    rows=[x.row for x in recorded];labels=list(labels)
    if (fit_record["split"]!="fit" or set(fit_record["fit_ids"]) & {r["id"] for r in rows}):
        raise ev.EvaluationError("audit leaks fit membership")
    if fit_record["selected"] is None:
        return {"qualified":False,"reason":"no_fit_policy","n":len(rows),"measured_latency":False}
    chosen=fit_record["selected"]
    ps,cost=simulate(recorded,physical_depth=physical_depth,**chosen)
    scores={};pairs={}
    for k in (1,2,4):
        ctrl=[_projection(x,k) for x in recorded]
        s=ev.evaluate(rows,labels,{"fixed":ctrl,"policy":ps},baseline="fixed",threshold=.9,
                      bootstrap_reps=reps,bootstrap_seed=seed,pass_series=["fixed","policy"])
        if not s["all_distributions_valid"]:raise ev.EvaluationError("invalid audited stream")
        scores[str(k)]=s["arms"]["fixed"];scores["policy"]=s["arms"]["policy"]
        pairs[str(k)]={"candidate_minus_fixed":s["paired_deltas"]["policy"],"transitions":s["passes"]["first_to_last"],
                       "mean_fixed_logical_layers":physical_depth+(k-1)*4}
    m=scores["policy"]["macro"];first=scores["1"]["macro"]
    # The audit requires a nondegenerate policy, not a second optimization using the fit fraction bounds.
    qualified=(0<cost["continued"]<cost["n"] and cost["logical_layers"]<=cost["n"]*(physical_depth+12)
               and m["accuracy"]>first["accuracy"] and m["brier"]<=first["brier"])
    return {"qualified":qualified,"policy":chosen,"n":len(rows),"cost":cost,"comparisons":pairs,"scores":scores,
            "measured_latency":False,"scope":"counterfactual from separately recorded six-pass probabilities; not a live model run"}
