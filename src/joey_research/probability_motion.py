"""Describe how two complete probability vectors changed; do not equate Brier with calibration."""
from __future__ import annotations

from collections import defaultdict
import math

from . import evaluation as ev


def _mean(values):
    return math.fsum(values)/len(values) if values else None


def compare(inputs, labels, baseline, candidate):
    """Micro, task and correctness-transition summaries on the same ID- and menu-bound examples."""
    ix=ev._index_inputs(inputs)
    ys=ev._bind(labels,ix,({"id","input_sha256","gold_index"},),"labels")
    ps=[ev._bind(arm,ix,({"id","input_sha256","probs"},),"predictions") for arm in (baseline,candidate)]
    groups=defaultdict(list);all_rows=[]
    for i,(r,_) in ix.items():
        y=ys[i]["gold_index"]
        if type(y) is not int or not 0<=y<len(r["choices"]):raise ev.EvaluationError("invalid gold index")
        a,b=(ev._observe(r,y,p[i]) for p in ps)
        if a.error or b.error:raise ev.EvaluationError("complete valid paired distributions required")
        def normalized(raw):
            total=math.fsum(raw)
            return [z/total for z in raw]
        p,q=(normalized(v[i]["probs"]) for v in ps)
        concentration=math.fsum((z-x)*(z+x) for x,z in zip(p,q,strict=True))
        truth_term=-2*(q[y]-p[y]);delta=b.brier-a.brier
        if abs(delta-math.fsum((truth_term,concentration)))>2e-12:raise ev.EvaluationError("Brier identity failed")
        bucket=("still_correct" if a.correct and b.correct else "corrected" if b.correct else "corrupted" if a.correct
                else "still_wrong_same_choice" if a.choice==b.choice else "still_wrong_changed_choice")
        one=(a,b,bucket,p[y],q[y],truth_term,concentration,delta)
        groups[r["task"]].append(one);all_rows.append(one)
    return {"n":len(all_rows),"micro":_summary(all_rows),"per_task":{t:_summary(v) for t,v in sorted(groups.items())},
            "direction":"candidate minus baseline","identity":"delta summed multiclass Brier = -2 delta p(truth) + delta squared L2 concentration",
            "note":"These are algebraic terms, not separate reliability or resolution estimators. Transition summaries condition on observed outcomes."}


def _summary(items):
    def terms(xs):
        return {"n":len(xs),"baseline_mean_p_truth":_mean([x[3] for x in xs]),"candidate_mean_p_truth":_mean([x[4] for x in xs]),
                "baseline_mean_top":_mean([x[0].confidence for x in xs]),"candidate_mean_top":_mean([x[1].confidence for x in xs]),
                "mean_truth_term":_mean([x[5] for x in xs]),"mean_concentration_term":_mean([x[6] for x in xs]),
                "mean_brier_change":_mean([x[7] for x in xs])}
    summary=terms(items)
    for label,k in (("baseline",0),("candidate",1)):
        summary[label+"_top_by_own_outcome"]={}
        for correct,name in ((True,"correct"),(False,"incorrect")):
            v=[x[k].confidence for x in items if x[k].correct==correct]
            summary[label+"_top_by_own_outcome"][name]={"n":len(v),"mean":_mean(v)}
    summary["transitions"]={}
    for b in ("still_correct","corrected","corrupted","still_wrong_same_choice","still_wrong_changed_choice"):
        xs=[x for x in items if x[2]==b]
        summary["transitions"][b]={**terms(xs),"brier_contribution_to_panel_mean":math.fsum(x[7] for x in xs)/len(items)}
    persist=[x for x in items if not x[0].correct and not x[1].correct]
    summary["persistently_wrong"]={"n":len(persist),"top_confidence_rose":sum(x[1].confidence>x[0].confidence for x in persist),
                 "new_threshold_crossings":{str(t):sum(x[0].confidence<t<=x[1].confidence for x in persist) for t in (.8,.9,.95,.99)}}
    return summary
