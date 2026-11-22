"""Paired comparison of two payloads' distance effects on the same released routing graphs."""
from collections import Counter, defaultdict
from collections.abc import Mapping

from . import horizon_metrics as hm
from joey_research.evaluation import EvaluationError

ROLES = ("full_split", "full_four", "compact_split", "compact_four")


def payload_interaction(rows, labels, arms, *, design_by_id, tokens, reps=2000, seed=4117):
    """Full minus compact, each comparing four visits to its own split-only path. No timing comparison."""
    if (type(reps) is not int or not 0 <= reps <= 10000 or type(seed) is not int
            or not all(isinstance(m, Mapping) and set(m) == set(ROLES) for m in (arms, tokens))):
        raise EvaluationError("expected four named streams, four token maps and a bounded bootstrap")
    rows, labels = list(rows), list(labels)
    panels = [hm._panel(rows, labels, arms[b], arms[c], design_by_id, tokens[b], tokens[c])
              for b, c in (ROLES[:2], ROLES[2:])]
    if tokens[ROLES[0]] != tokens[ROLES[2]]:
        raise EvaluationError("cross-payload token lengths differ on the same decision")
    by_seed = defaultdict(list)
    for group in sorted(panels[0]):
        a, b = (panel[group] for panel in panels)
        # Form graph-level paired differences BEFORE drawing graph multiplicities.
        full, compact = (hm._graph_sums(g, {h: [] for h in hm._HOPS}, Counter()) for g in (a, b))
        diff = {}
        for key, f in full.items():
            c = compact[key]
            if f.n != c.n:
                raise EvaluationError("different graph support for the two payloads")
            diff[key] = hm._Sums(f.n, f.accuracy - c.accuracy, f.brier - c.brier)
        by_seed[a.seed].append(diff)
    strata = [by_seed[k] for k in sorted(by_seed)]
    effects = hm._bootstrap(strata, reps, seed)
    return {"direction": "full minus compact; four minus split; six hops minus two",
            "bootstrap": {"seed": seed, "reps": reps}, "primary": effects["primary"],
            "exact_token": effects["exact_token"],
            "support": hm._support([g for s in strata for g in s]),
            "by_seed": {str(k): hm._support(v) for k, v in sorted(by_seed.items())},
            "by_hop": {str(h): effects[h] for h in hm._HOPS}}
