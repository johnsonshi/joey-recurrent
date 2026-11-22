"""Method and observation contract for an independently released live one-probe comparison."""
from __future__ import annotations

import math
import re

MODE = "one-probe"
MARGINS = (0., .1, .25, .5, .75, 1.)
CAPS = (2, 3, 4, 6)


def method(settings):
    if type(settings) is not dict:raise ValueError("expected a probe method mapping")
    small = "compact_recipe" in settings
    fields = {"window","loop","reverse"} | ({"compact_recipe","compact_manifest_sha256"} if small else set())
    probed = "probe_margin" in settings
    if probed:
        fields.add("probe_margin")
    if (type(settings) is not dict or set(settings)!=fields or settings["reverse"] is not False
            or settings["window"]!=[8,12] or any(type(i) is not int for i in settings["window"])
            or type(settings["loop"]) is not dict or set(settings["loop"])!={"k","method"}
            or type(settings["loop"]["k"]) is not int):
        raise ValueError("invalid one-probe method")
    k, name = settings["loop"]["k"], settings["loop"]["method"]
    if (small and (settings["compact_recipe"]!="select-20" or type(settings["compact_manifest_sha256"]) is not str
                   or not re.fullmatch("[0-9a-f]{64}",settings["compact_manifest_sha256"]))):
        raise ValueError("unbound one-probe compact payload")
    if probed:
        t = settings["probe_margin"]
        if type(t) not in (int,float) or t not in MARGINS or name!="naive" or k not in CAPS:
            raise ValueError("invalid first-margin policy")
    elif not (name=="naive" and k in (1,*CAPS) or name=="zero_extra" and k==2):
        raise ValueError("unsupported fixed probe control")
    return small, k, name, settings.get("probe_margin")


def vector(p):
    if (type(p) is not list or len(p)!=4 or not all(type(x) in (float,int) and 0<=x<=1 for x in p)
            or abs(math.fsum(p)-1.)>1e-6):
        raise ValueError("invalid four-choice probe distribution")


def validate_trace(trace, probs, *, settings, packet_sha256):
    small,k,name,threshold=method(settings)
    fields={"id","tokens","layer_calls","callback_counts","readouts","wall_ms","cuda_event_ms","packet_sha256"}
    if threshold is not None:fields.add("probe")
    if (type(trace) is not dict or set(trace)!=fields or type(trace["id"]) is not str or not trace["id"].strip()
            or trace["packet_sha256"]!=packet_sha256 or trace["readouts"]!=[] or type(trace["tokens"]) is not int
            or not 1<=trace["tokens"]<=16384 or any(type(trace[x]) not in (float,int) or not 0<=trace[x]<math.inf
                                                   for x in ("wall_ms","cuda_event_ms"))):
        raise ValueError("invalid or unbound probe trace")
    vector(probs)
    if threshold is None:
        effective=1 if name=="zero_extra" else k
        expected=(1,0,0,0) if k==1 else (0,1,effective,1)
    else:
        d=trace["probe"]
        if type(d) is not dict or set(d)!={"first_probs","first_margin","exit_pass","reason"}:
            raise ValueError("invalid intermediate observation")
        first=d["first_probs"]
        vector(first)
        a,b=sorted(first,reverse=True)[:2]
        margin=a-b
        stop=margin>=threshold
        effective=1 if stop else k
        if (type(d["first_margin"]) not in (float,int) or d["first_margin"]!=margin or type(d["exit_pass"]) is not int
                or d["exit_pass"]!=effective or d["reason"]!=("margin_met" if stop else "continue_to_cap")
                or stop and first!=probs):
            raise ValueError("reported probe exit disagrees with its frozen margin")
        expected=(0,1,effective,1 if stop else 2)
    counts=trace["callback_counts"];keys=("stock_calls","prelude_calls","block_calls","coda_calls")
    if (type(counts) is not dict or set(counts)!=set(keys) or not all(type(x) is int for x in counts.values())
            or tuple(counts[x] for x in keys)!=expected):
        raise ValueError("probe callback counts do not match execution")
    s,p,b,c=expected;depth=20 if small else 24
    if type(trace["layer_calls"]) is not int or trace["layer_calls"]!=s*depth+8*p+4*b+(depth-12)*c:
        raise ValueError("probe layer accounting mismatch")
