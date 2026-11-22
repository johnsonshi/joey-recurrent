"""Exact null coverage in a paired, clustered stress case; no checkpoint or task labels required."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from fractions import Fraction


@lru_cache(maxsize=None)
def _binomial(n: int, p: Fraction) -> tuple[Fraction, ...]:
    if p == 0:
        return (Fraction(1),) + (Fraction(0),) * n
    if p == 1:
        return (Fraction(0),) * n + (Fraction(1),)
    prob = (1 - p) ** n
    out = [prob]
    for k in range(1, n + 1):
        prob = prob * (n - k + 1) * p / (k * (1 - p))
        out.append(prob)
    return tuple(out)


def _quantile(probabilities, cutoff):
    cumulative = Fraction(0)
    for k, p in enumerate(probabilities):
        cumulative += p
        if cumulative >= cutoff:
            return k
    raise ValueError("invalid probability distribution")


def coverage(groups: int, rows_per_group: int) -> dict:
    """Percentile CI under perfectly correlated decisions within each independent source group."""
    if type(groups) is not int or not 2 <= groups <= 64 or type(rows_per_group) is not int or not 1 <= rows_per_group <= 8:
        raise ValueError("bounded design: 2..64 groups, 1..8 decisions per group")
    # Each group's paired change is either +1 or -1 with equal probability.
    # All rows of a group have the same change, so the underlying true effect is zero.
    outer = _binomial(groups, Fraction(1,2))
    result = {}
    for label, n in (("group",groups),("row",groups*rows_per_group)):
        cover = width = Fraction(0)
        for plus, mass in enumerate(outer):
            conditional = _binomial(n,Fraction(plus,groups))
            lo = Fraction(2*_quantile(conditional,Fraction(1,40)),n) - 1
            hi = Fraction(2*_quantile(conditional,Fraction(39,40)),n) - 1
            if lo <= 0 <= hi:
                cover += mass
            width += mass * (hi-lo)
        result[label] = {"coverage":float(cover),"mean_width":float(width),"coverage_fraction":str(cover)}
    return {"groups":groups,"rows_per_group":rows_per_group,"true_paired_effect":0,"nominal":0.95,
            "resampling":result,"iid_row_design_effect":rows_per_group}


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--groups",type=int,nargs="+",default=[12,20,36])
    cli.add_argument("--rows-per-group",type=int,default=8)
    args=cli.parse_args()
    print(json.dumps([coverage(g,args.rows_per_group) for g in args.groups],allow_nan=False,indent=2))
