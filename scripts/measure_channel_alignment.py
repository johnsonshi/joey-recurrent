"""Measure one predeclared channel assignment per middle-layer pair on a local read-only checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time


def independently_check(a,b,p,z,seed):
    import numpy as np
    from mlp_channel_alignment import coordinate_split
    m,h=a["up"].shape
    f,c=coordinate_split(h,seed=seed)
    if (z["coordinates"]["fit"]!=f.tolist() or z["coordinates"]["audit"]!=c.tolist() or set(f)&set(c)
            or sorted(np.concatenate((f,c)).tolist())!=list(range(h)) or sorted(p.tolist())!=list(range(m))
            or hashlib.sha256(np.asarray(p,dtype="<i8").tobytes()).hexdigest()!=z["permutation_sha256"]):
        raise ValueError("unbound held-out coordinates or channel bijection")
    for part in ("gate","up","down"):
        x=a[part] if part!="down" else a[part].T
        y=b[part] if part!="down" else b[part].T
        identity=[];aligned=[]
        # Aggregate by hidden-coordinate stripes, instead of the core's source-row reduction.
        for i in range(0,len(c),16):
            ix=c[i:i+16];aa=x[:,ix].astype(np.float64);bb=y[:,ix].astype(np.float64)
            for accum,value in ((identity,bb),(aligned,bb[p])):
                err=aa-value;accum.append(float(np.sum(err*err,dtype=np.float64)))
        stats=z["audit"]["projections"][part]
        if (not math.isclose(math.fsum(identity),stats["identity_sse"],rel_tol=2e-12,abs_tol=1e-12) or
                not math.isclose(math.fsum(aligned),stats["aligned_sse"],rel_tol=2e-12,abs_tol=1e-12)):
            raise ValueError("independent held-out weight-error sums disagree")


def aggregate(results,gate):
    pools={}
    for split in ("fit","audit"):
        vals=[r["measurements"][split]["pooled"] for r in results]
        a,b=(math.fsum(x[key] for x in vals) for key in ("identity_sse","aligned_sse"))
        pools[split]={"identity_sse":a,"aligned_sse":b,"count":sum(x["count"] for x in vals),"ratio":b/a if a else None}
    ratios=[]
    for r in results:
        x=r["measurements"]["audit"]["pooled"];ratios.append(x["aligned_sse"]/x["identity_sse"] if x["identity_sse"] else None)
    actions=all(r["measurements"]["action_check"]["passed"] for r in results)
    qualifying=sum(x is not None and x<=gate["maximum_per_pair_ratio"] for x in ratios)
    passed=bool(len(results)==4 and actions and pools["audit"]["ratio"] is not None
                and pools["audit"]["ratio"]<=gate["maximum_pooled_audit_sse_ratio"] and qualifying>=gate["minimum_pairs_passing"])
    return {"pooled":pools,"pair_audit_ratios":ratios,"qualifying_pairs":qualifying,"action_equivalence":actions,"export_gate_passed":passed}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--protocol-sha256",required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1];spec=(root/"configs/evaluation/channel-alignment.json").read_bytes()
    if hashlib.sha256(spec).hexdigest()!=args.protocol_sha256:raise ValueError("wrong protocol hash")
    rule=json.loads(spec)
    if (rule["schema"]!="joey-common-mlp-channel-basis/v1" or rule["affects_confirmation"] is not False
            or rule["budget"]!={"cpu_threads":1,"wall_seconds":2400} or rule["layer_pairs"]!=[[8,12],[9,13],[10,14],[11,15]]
            or rule["projections"]!=["gate","up","down"]):raise ValueError("unexpected weight-only protocol")
    signal.alarm(rule["budget"]["wall_seconds"]);resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.umask(0o077)
    for var in ("VECLIB_MAXIMUM_THREADS","OPENBLAS_NUM_THREADS","GOTO_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS","OMP_NUM_THREADS","OMP_THREAD_LIMIT"):
        os.environ[var]="1"
    import numpy as np
    from compact_spectrum import check_output,model_pin,verify_pin,matched_pairs,write_new
    from mlp_channel_alignment import analyze_pair
    sys.path.insert(0,str(root/"src"));from joey_research import compact
    path=args.weights;check_output(args.output)
    if (not path.is_absolute() or path.resolve(strict=True)!=path or not path.is_file() or path.stat().st_mode&0o222
            or compact._PIN_WEIGHTS!=model_pin()):raise ValueError("not a direct read-only pinned source")
    start=time.monotonic();results=[]
    with path.open("rb") as handle:
        orig=os.fstat(handle.fileno());pin=verify_pin(handle,model_pin())
        header,_=compact._shard(path);seen=list(matched_pairs(header,compact._LAYERS,compact._CYCLE,compact._MERGE_SUFFIXES))
        if len(seen)!=12:raise ValueError("wrong matched MLP scope")
        def load(t):
            if t.dtype!="BF16" or t.numel!=6144*2048:raise ValueError("unexpected physical projection")
            handle.seek(t.offset);raw=handle.read(t.nbytes)
            if len(raw)!=t.nbytes:raise ValueError("short read of BF16 weights")
            u=np.frombuffer(raw,dtype="<u2").astype(np.uint32);u<<=16
            return u.view(np.float32).reshape(t.shape)
        for ids in rule["layer_pairs"]:
            pair=[x for x in seen if list(x[0])==ids]
            if set(x[2] for x in pair)!=set(rule["projections"]) or len({x[1] for x in pair})!=1:raise ValueError("incompatible layer pair")
            a={x[2]:load(x[3]) for x in pair};b={x[2]:load(x[4]) for x in pair}
            seed=rule["seed_by_earlier_layer"][str(ids[0])]
            perm,z=analyze_pair(a,b,seed=seed,sketch=rule["sketch_coordinates_per_projection"],shortlist=rule["shortlist"],null_reps=rule["null_reps"])
            independently_check(a,b,perm,z,seed)
            results.append({"layers":ids,"type":pair[0][1],"permutation":perm.tolist(),"measurements":z})
            del a,b,perm
        end=path.stat()
        if not os.path.samestat(orig,end) or (orig.st_size,orig.st_mtime_ns,orig.st_ctime_ns)!=(end.st_size,end.st_mtime_ns,end.st_ctime_ns):
            raise ValueError("source file changed")
    report={"schema":"joey-measured-channel-alignment/v1","source_sha256":pin,"protocol_sha256":args.protocol_sha256,
            "numpy_version":np.__version__,"pairs":results,"aggregate":aggregate(results,rule["action_gate"]),
            "elapsed_seconds":time.monotonic()-start,"independent_audit_checked":True,"inference_performed":False,"checkpoint_exported":False,
            "note":"One common bijection of a SwiGLU layer's intermediate channels. Weight similarity is not a decision metric."}
    raw=json.dumps(report,sort_keys=True,allow_nan=False,separators=(",",":")).encode()+b"\n"
    write_new(args.output,raw)
    print("Measured four channel bases; independent held-out sums verified. Report SHA-256:",hashlib.sha256(raw).hexdigest())


if __name__=="__main__":
    try:main()
    except (OSError,ValueError,RuntimeError,ImportError) as e:
        print("Numerical study stopped:",type(e).__name__,file=sys.stderr)
        raise SystemExit(1)
