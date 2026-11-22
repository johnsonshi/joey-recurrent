"""Fresh routing questions with 24 displayed edges regardless of the required reasoning distance."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import random
import re

from joey_research.evaluation import EvaluationError, input_sha256, label_row


HOPS = (1, 2, 4, 6)
TASK = "fixed_evidence_route"
_NAMESPACE = "fixed-evidence-horizon-v1"
_HEADERS = (
    "Target station: {}. Which entry station's route includes this target?\nRead edges from sender to receiver. Records are unordered.",
    "Find the entry station that eventually sends a parcel through {}.\nEach record gives the receiver first, then the sender. Records are unordered.",
)
_ARROW = re.compile(r"([A-Z]{3}) -> ([A-Z]{3})")
_RECEIVER = re.compile(r"([A-Z]{3}) gets parcels from ([A-Z]{3})")
_OPTION = re.compile(r"Enter at ([A-Z]{3})")


def _seed(seed, index, field):
    return hashlib.sha256(f"{_NAMESPACE}:{seed}:{index}:{field}".encode()).hexdigest()


def _ids(seed, count):
    if type(seed) is not int or type(count) is not int or not 0 <= seed < 2**32 or not 1 <= count <= 1000:
        raise EvaluationError("invalid fixed-horizon sampling bounds")
    for i in range(count):
        yield i, "fh-g-" + _seed(seed, i, "group")


def group_seeds(seed, graphs):
    """Blind assignment: no target-to-root mapping."""
    return {g: seed for _,g in _ids(seed, graphs)}


def iter_inputs(seed, graphs):
    for i,g in _ids(seed, graphs):
        def rng(tag):
            return random.Random(int(_seed(seed,i,tag),16))
        nums = rng("stations").sample(range(26**3), 28)
        names = ["".join(chr(65+n//26**k%26) for k in (2,1,0)) for n in nums]
        levels = [names[j:j+4] for j in range(0,28,4)]
        edges = []
        for h in range(6):
            destinations = levels[h+1].copy()
            rng("links:"+str(h)).shuffle(destinations)
            edges.extend(zip(levels[h],destinations,strict=True))
        previous_menu = previous_edges = None
        for wording in range(2):
            ordered = edges.copy(); choices = [f"Enter at {r}" for r in levels[0]]
            rng("records:"+str(wording)).shuffle(ordered)
            rng("menu:"+str(wording)).shuffle(choices)
            if choices == previous_menu: choices = choices[1:] + choices[:1]
            if ordered == previous_edges: ordered = ordered[1:] + ordered[:1]
            previous_menu, previous_edges = choices, ordered
            lines = [f"{s} -> {t}" if not wording else f"{t} gets parcels from {s}" for s,t in ordered]
            body = "\n".join(lines)
            for hop in HOPS:
                for target in levels[hop]:
                    yield {"id":"fh-i-"+_seed(seed,i,f"{wording}:{target}"), "group_id":g, "task":TASK,
                           "prompt":_HEADERS[wording].format(target)+"\n"+body, "choices":choices.copy()}


def shown(row):
    """Decode only public display syntax. No path is traversed and no gold choice is computed."""
    input_sha256(row)
    if row["task"] != TASK or len(row["choices"]) != 4:
        raise EvaluationError("expected a four-entry fixed-evidence question")
    lines = row["prompt"].splitlines()
    if len(lines) != 26:
        raise EvaluationError("fixed-evidence graph must display exactly 24 edge records")
    for wording,template in enumerate(_HEADERS):
        m = re.fullmatch(re.escape(template).replace(re.escape("{}"),r"([A-Z]{3})"), "\n".join(lines[:2]))
        if m:
            goal=m.group(1);break
    else: raise EvaluationError("unknown fixed-evidence header")
    edges=[]
    for line in lines[2:]:
        m=(_ARROW if not wording else _RECEIVER).fullmatch(line)
        if not m:raise EvaluationError("malformed displayed edge")
        a,b=m.groups();edges.append((a,b) if not wording else (b,a))
    opts=[]
    for choice in row["choices"]:
        m=_OPTION.fullmatch(choice)
        if not m:raise EvaluationError("malformed entry option")
        opts.append(m.group(1))
    parents,children=(set(col) for col in zip(*edges))
    if (len(parents)!=24 or len(children)!=24 or len(parents|children)!=28 or set(opts)!=parents-children or goal not in children):
        raise EvaluationError("displayed sources, targets, or entry menu are invalid")
    return goal,wording,tuple(edges),tuple(opts)


def check_blind_panel(rows, seed_by_group):
    """Check shared evidence, ordered menus and goal coverage without deriving any mapping to gold."""
    cells=defaultdict(lambda:defaultdict(list))
    ids=set();graphs=set()
    for r in rows:
        goal,w,edges,opts=shown(r)
        if r["id"] in ids or r["group_id"] not in seed_by_group:raise EvaluationError("unexpected blind item or group")
        ids.add(r["id"]);cells[r["group_id"]][w].append((goal,edges,opts))
    if set(cells)!=set(seed_by_group) or not cells:raise EvaluationError("incomplete group assignment")
    for group in cells.values():
        if set(group)!={0,1}:raise EvaluationError("missing alternate rendering")
        targets=[]
        for w in (0,1):
            variant=group[w];goals={x[0] for x in variant}
            if len(goals)!=16 or len(variant)!=16 or any(x[1:]!=variant[0][1:] for x in variant):
                raise EvaluationError("edge order or menu changes across goals within a rendering")
            targets.append(goals)
        _,a,b=group[0][0];_,c,d=group[1][0]
        key=tuple(sorted(a))
        if (targets[0]!=targets[1] or key!=tuple(sorted(c)) or a==c or set(b)!=set(d) or b==d or key in graphs):
            raise EvaluationError("renderings do not preserve one distinct graph")
        graphs.add(key)
    return {"groups":len(cells),"rows":len(ids),"groups_by_seed":dict(sorted(Counter(seed_by_group.values()).items()))}


def adjudicate(row):
    """Independently solve all four displayed chains. Call only after score release."""
    goal,w,edges,entries=shown(row)
    outgoing=dict(edges);visited=set();answers={}
    for root in entries:
        node=root;depth=0
        while node in outgoing:
            if node in visited:raise EvaluationError("cycle or non-disjoint chains")
            visited.add(node);node=outgoing[node];depth+=1;answers[node]=(root,depth)
        if depth!=6:raise EvaluationError("not four complete six-edge chains")
    if len(visited)!=24 or goal not in answers:raise EvaluationError("unreachable target or unused edge")
    root,hop=answers[goal]
    if hop not in HOPS:raise EvaluationError("target outside predeclared distances")
    return label_row(row,entries.index(root)),{"hop":hop,"entry":root,"rendering":w}
