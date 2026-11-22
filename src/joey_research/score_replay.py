"""Read-only replay of quality metrics from a released, independently pinned score bundle."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import stat

from . import evaluation as ev


def _require(condition, reason):
    if not condition:
        raise ev.EvaluationError(reason)


def _read(path, pin, limit):
    _require(ev._sha256_text(pin) and path.resolve(strict=True) == path, "invalid replay file or pin")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as f:
        s = os.fstat(f.fileno())
        _require(stat.S_ISREG(s.st_mode) and 0 < s.st_size <= limit, "invalid replay object")
        raw = f.read(s.st_size + 1)
        _require(len(raw) == s.st_size and hashlib.sha256(raw).hexdigest() == pin, "replay object digest differs")
    return raw


def replay(directory, *, manifest_sha256):
    """Recalculate all declared arm summaries, whole-group paired intervals and outcome transitions."""
    base = Path(directory).absolute()
    d = ev._strict_json(_read(base / "manifest.json", manifest_sha256, 1 << 20).decode("utf-8"))
    _require(isinstance(d, dict) and set(d) == {"schema", "split", "source_report_sha256", "answer_release_sha256",
             "rows", "groups", "inputs_sha256", "labels_sha256", "arms", "pairs", "omitted_posthoc",
             "bootstrap_reps", "bootstrap_seed"}, "unexpected replay schema")
    _require(d["schema"] == "joey-released-score-replay/v1" and d["split"] in ("selection", "final")
             and all(ev._sha256_text(d[k]) for k in ("source_report_sha256","answer_release_sha256","inputs_sha256","labels_sha256"))
             and type(d["rows"]) is int and type(d["groups"]) is int and 1 <= d["groups"] <= d["rows"] <= 20000
             and type(d["bootstrap_reps"]) is int and d["bootstrap_reps"] == 2000
             and type(d["bootstrap_seed"]) is int and d["bootstrap_seed"] == 4117,
             "unsupported replay settings")
    arms, pairs = d["arms"], d["pairs"]
    _require(isinstance(arms, dict) and 0 < len(arms) <= 64 and isinstance(pairs, list) and len(pairs) <= 128,
             "invalid replay panel")
    _require(isinstance(d["omitted_posthoc"], list) and all(type(n) is str for n in d["omitted_posthoc"])
             and len(set(d["omitted_posthoc"])) == len(d["omitted_posthoc"]), "invalid posthoc coverage")
    for name, a in arms.items():
        _require(re.fullmatch(r"(?:raw|posthoc):[A-Za-z0-9_.-]{1,96}", name) and isinstance(a, dict)
                 and a.keys() == {"method_sha256","predictions_sha256","primary_sha256"} and all(map(ev._sha256_text,a.values())),
                 "invalid replay arm")
    raw_names = {n[4:] for n in arms if n.startswith("raw:")}
    post_names = {n[8:] for n in arms if n.startswith("posthoc:")}
    omitted = set(d["omitted_posthoc"])
    _require(raw_names and not (omitted & post_names) and
             (not post_names and not omitted if d["split"] == "selection" else raw_names == (post_names | omitted)),
             "incomplete or overlapping projection coverage")
    _require(len({arms["raw:"+n]["method_sha256"] for n in raw_names}) == len(raw_names),
             "duplicate raw method identity")
    for n in post_names:
        _require(arms["posthoc:"+n]["method_sha256"] == arms["raw:"+n]["method_sha256"], "posthoc method mismatch")
    for p in pairs:
        _require(isinstance(p, dict) and p.keys() == {"baseline","candidate","delta_sha256","transitions_sha256"}
                 and p["baseline"] in arms and p["candidate"] in arms and p["baseline"] != p["candidate"]
                 and p["baseline"].partition(":")[0] == p["candidate"].partition(":")[0]
                 and all(ev._sha256_text(p[k]) for k in ("delta_sha256","transitions_sha256")), "invalid replay contrast")

    objects = base / "objects"
    pins = {d["inputs_sha256"], d["labels_sha256"], *(a["predictions_sha256"] for a in arms.values())}
    _require(objects.resolve(strict=True) == objects and objects.is_dir()
             and {p.name for p in base.iterdir()} == {"manifest.json", "objects"}
             and {p.name for p in objects.iterdir()} == pins, "replay inventory differs")

    def records(pin):
        raw = _read(objects / pin, pin, 64 << 20)
        lines = raw.splitlines()
        _require(len(lines) == d["rows"], "incorrect replay row count")
        return [ev._strict_json(line.decode("utf-8")) for line in lines]

    rows = records(d["inputs_sha256"])
    ix = ev._index_inputs(rows)
    _require(len(ix) == len(rows) and len({r["group_id"] for r in rows}) == d["groups"], "wrong replay cohort")
    ids = [r["id"] for r in rows]

    def bound(pin, key):
        data = records(pin)
        for r,k in zip(data, ids, strict=True):
            _require(isinstance(r, dict) and r.keys() == {"id","input_sha256",key}
                     and r["id"] == k and r["input_sha256"] == ix[k][1], "unpaired or reordered replay stream")
        return data

    labels = bound(d["labels_sha256"], "gold_index")
    predictions = {}
    for n,a in arms.items():
        pred = bound(a["predictions_sha256"], "probs")
        v = ev.evaluate(rows, labels, {n:pred}, baseline=n, threshold=.8, bootstrap_reps=0)
        _require(v["all_distributions_valid"] and ev._hash(v["arms"][n]) == a["primary_sha256"],
                 "computed quality differs from the released report")
        predictions[n] = pred
    threshold = .8 if d["split"] == "selection" else .9
    for p in pairs:
        a,b = p["baseline"],p["candidate"]
        v = ev.evaluate(rows, labels, {n:predictions[n] for n in (a,b)}, baseline=a, threshold=threshold,
                        pass_series=[a,b], bootstrap_reps=d["bootstrap_reps"], bootstrap_seed=d["bootstrap_seed"])
        _require(ev._hash(v["paired_deltas"][b]) == p["delta_sha256"]
                 and ev._hash(v["passes"]["first_to_last"]) == p["transitions_sha256"],
                 "computed paired result differs from the released report")
    return {"split":d["split"], "rows":len(rows), "groups":d["groups"], "arms":len(arms), "pairs":len(pairs),
            "omitted_posthoc":len(omitted), "source_report_sha256":d["source_report_sha256"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    a = p.parse_args()
    try:
        r = replay(a.bundle, manifest_sha256=a.manifest_sha256)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        p.exit(2, "Replay refused: invalid bundle, missing evidence or score disagreement.\n")
    print(f"Reproduced {r['arms']} arm summaries and {r['pairs']} paired contrasts on {r['rows']} "
          f"{r['split']} decisions in {r['groups']} groups. No inference was performed.")


if __name__ == "__main__":
    main()
