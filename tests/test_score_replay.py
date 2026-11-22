import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from joey_research import evaluation as ev
from joey_research.score_replay import replay


def put(root, data):
    raw = b"".join(json.dumps(r).encode()+b"\n" for r in data)
    pin = hashlib.sha256(raw).hexdigest()
    (root / "objects" / pin).write_bytes(raw)
    return pin


def seal(root, manifest):
    raw = json.dumps(manifest, sort_keys=True).encode()
    (root / "manifest.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class ScoreReplayTests(unittest.TestCase):
    def test_complete_metrics_intervals_and_reordered_stream_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/"objects").mkdir()
            rows=[{"id":str(i),"task":"toy","group_id":str(i//2),"prompt":f"toy case {i}","choices":["left","right"]}
                  for i in range(8)]
            labels=[ev.label_row(r,i%2) for i,r in enumerate(rows)]
            def arm(v):return [{"id":r["id"],"input_sha256":ev.input_sha256(r),"probs":v[i%2]} for i,r in enumerate(rows)]
            data={"raw:a":arm([[.8,.2],[.6,.4]]),"raw:b":arm([[.9,.1],[.2,.8]])}
            x=ev.evaluate(rows,labels,data,baseline="raw:a",threshold=.8,pass_series=list(data),bootstrap_reps=2000,bootstrap_seed=4117)
            m={"schema":"joey-released-score-replay/v1","split":"selection","source_report_sha256":"c"*64,
               "answer_release_sha256":"d"*64,"rows":8,"groups":4,"bootstrap_reps":2000,"bootstrap_seed":4117,
               "inputs_sha256":put(root,rows),"labels_sha256":put(root,labels),"omitted_posthoc":[],
               "arms":{n:{"method_sha256":n[-1]*64,"predictions_sha256":put(root,z),"primary_sha256":ev._hash(x["arms"][n])}
                       for n,z in data.items()},"pairs":[{"baseline":"raw:a","candidate":"raw:b","delta_sha256":ev._hash(x["paired_deltas"]["raw:b"]),
                                                           "transitions_sha256":ev._hash(x["passes"]["first_to_last"])}]}
            good=seal(root,m)
            self.assertEqual(replay(root,manifest_sha256=good)["pairs"],1)
            with self.assertRaises(ValueError):replay(root,manifest_sha256="0"*64)
            y=ev.evaluate(rows,labels,data,baseline="raw:a",threshold=.9,pass_series=list(data),bootstrap_reps=2000,bootstrap_seed=4117)
            final={**m,"split":"final","arms":{**m["arms"],"posthoc:a":m["arms"]["raw:a"]},"omitted_posthoc":["b"],
                   "pairs":[{**m["pairs"][0],"delta_sha256":ev._hash(y["paired_deltas"]["raw:b"])}]}
            result=replay(root,manifest_sha256=seal(root,final))
            self.assertEqual((result["arms"],result["omitted_posthoc"]),(3,1))
            extra=root/"objects"/("f"*64);extra.write_bytes(b"not part of this release")
            with self.assertRaisesRegex(ValueError,"inventory"):replay(root,manifest_sha256=seal(root,m))
            extra.unlink()
            m["pairs"][0]["transitions_sha256"]="0"*64
            with self.assertRaisesRegex(ValueError,"paired result"):replay(root,manifest_sha256=seal(root,m))
            (root/"objects"/m["arms"]["raw:b"]["predictions_sha256"]).unlink()
            m["arms"]["raw:b"]["predictions_sha256"]=put(root,list(reversed(data["raw:b"])))
            with self.assertRaisesRegex(ValueError,"reordered"):replay(root,manifest_sha256=seal(root,m))

    def test_path_in_place_of_digest_and_unaccounted_posthoc_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            m={"schema":"joey-released-score-replay/v1","split":"final","source_report_sha256":"c"*64,
               "answer_release_sha256":"d"*64,"rows":8,"groups":4,"bootstrap_reps":2000,"bootstrap_seed":4117,
               "inputs_sha256":"0"*64,"labels_sha256":"1"*64,"omitted_posthoc":[],"pairs":[],
               "arms":{"raw:a":{"method_sha256":"a"*64,"predictions_sha256":"a"*64,"primary_sha256":"a"*64}}}
            with self.assertRaisesRegex(ValueError,"projection coverage"):replay(root,manifest_sha256=seal(root,m))
            m["omitted_posthoc"]=["a"];m["arms"]["raw:a"]["predictions_sha256"]="../elsewhere"
            with self.assertRaisesRegex(ValueError,"replay arm"):replay(root,manifest_sha256=seal(root,m))
