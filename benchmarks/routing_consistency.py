"""Compare predictions for the same route goal under the two existing surface renderings."""
from __future__ import annotations

from collections import defaultdict
import math
import random
import re

from benchmarks import routing
from joey_research import evaluation as ev


def _parsed(row):
    lines = row["prompt"].splitlines()
    for template, header in enumerate(routing._HEADERS):
        m = re.fullmatch(re.escape(header).replace(re.escape("{goal}"), "([A-Z]{3})"), "\n".join(lines[:2]))
        if m:
            break
    else:
        raise ev.EvaluationError("unrecognized route rendering")
    graph = []
    for line in lines[2:]:
        edge = (routing._ARROW if template == 0 else routing._RECEIVES).fullmatch(line)
        if not edge:
            raise ev.EvaluationError("unrecognized directed edge")
        a,b = edge.groups()
        graph.append((a,b) if template == 0 else (b,a))
    return m.group(1), template, tuple(sorted(graph))


def group_outcomes(inputs, labels, predictions):
    """Scores each independent graph once; choices are translated back to the displayed station."""
    ix = ev._index_inputs(inputs)
    gold = ev._bind(labels, ix, ({"id","input_sha256","gold_index"},), "labels")
    pred = ev._bind(predictions, ix, ({"id","input_sha256","probs"},), "predictions")
    grouped = defaultdict(dict)
    facts = {}
    for k,(r,_) in ix.items():
        g = r["group_id"]
        y = gold[k]["gold_index"]
        if type(y) is not int or gold[k] != routing.inverse_label(r):
            raise ev.EvaluationError("released gold does not match this displayed route")
        end,t,edges = _parsed(r)
        menu = r["choices"]
        signature = r["task"], edges, frozenset(menu)
        if facts.setdefault(g,signature) != signature or (end,t) in grouped[g]:
            raise ev.EvaluationError("different graphs, menus or duplicate goal/rendering in a group")
        o = ev._observe(r,y,pred[k])
        if o.error:
            raise ev.EvaluationError("requires all unrounded probability vectors")
        p = pred[k]["probs"]
        z = math.fsum(p)
        grouped[g][end,t] = menu[y], menu[o.choice], {a:v/z for a,v in zip(menu,p,strict=True)}
    out = {}
    for g,items in sorted(grouped.items()):
        task, edges, menu = facts[g]
        goals = {b for a,b in edges} - {a for a,b in edges}
        if set(items) != {(goal,t) for goal in goals for t in (0,1)} or len(goals) != len(menu):
            raise ev.EvaluationError("incomplete counterfactual goal pair")
        totals = defaultdict(float)
        for goal in sorted(goals):
            y,a,p = items[goal,0]
            other,b,q = items[goal,1]
            if y != other:
                raise ev.EvaluationError("paired displayed route has inconsistent answers")
            totals["semantic_agreement"] += a == b
            totals["both_correct"] += a == y == b
            totals["consistent_wrong"] += a == b and a != y
            totals["arrow_accuracy"] += a == y
            totals["receiver_first_accuracy"] += b == y
            totals["total_variation"] += .5*math.fsum(abs(p[c]-q[c]) for c in sorted(menu))
        out[g] = {"task":task,"goals":len(goals),"sums":dict(totals)}
    return out


def contrast(base, candidate, *, reps=2000, seed=4117):
    """Report pointwise per-hop and equal-task paired intervals; resample graphs, not goals."""
    if not base or set(base) != set(candidate) or type(reps) is not int or not 1 <= reps <= 10000 or type(seed) is not int:
        raise ev.EvaluationError("invalid paired route consistency panel")
    pools = defaultdict(list)
    metrics = tuple(sorted(next(iter(base.values()))["sums"]))
    for g in sorted(base):
        a,b = base[g],candidate[g]
        if ((a["goals"],a["task"]) != (b["goals"],b["task"]) or type(a["goals"]) is not int or a["goals"] <= 0
                or set(a["sums"]) != set(metrics) or set(b["sums"]) != set(metrics)):
            raise ev.EvaluationError("unpaired route-group endpoint")
        pools[a["task"]].append((a["goals"], *(b["sums"][m]-a["sums"][m] for m in metrics)))
    def estimate(sample):
        n,*s = (math.fsum(v) for v in zip(*sample))
        return [v/n for v in s]
    rng = random.Random(seed)
    point = {t:estimate(group) for t,group in pools.items()}
    ds = {t:[[] for _ in metrics] for t in pools}
    macro = [[] for _ in metrics]
    for _ in range(reps):
        draws = []
        for t,p in sorted(pools.items()):
            v = estimate([p[rng.randrange(len(p))] for _ in p])
            for l,z in zip(ds[t],v,strict=True):l.append(z)
            draws.append(v)
        for l,col in zip(macro,zip(*draws),strict=True):l.append(math.fsum(col)/len(col))
    def pack(values, samples, eligible):
        return {m:{"delta":v,"ci95":[ev._percentile(sorted(l),p) for p in (.025,.975)] if eligible else None}
                for m,v,l in zip(metrics,values,samples,strict=True)}
    equal = [math.fsum(v)/len(pools) for v in zip(*point.values())]
    return {"per_task":{t:pack(point[t],ds[t],len(vals)>=2) for t,vals in sorted(pools.items())},
            "equal_task":pack(equal,macro,all(len(p)>=2 for p in pools.values())),
            "graphs_per_task":{t:len(p) for t,p in sorted(pools.items())},"reps":reps,"seed":seed}
