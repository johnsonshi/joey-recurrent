"""Host-side BANKING77 train-derived development diagnostic; not an official evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable
import csv
from dataclasses import dataclass
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import stat
import unicodedata

from benchmarks import decision_index_banking77 as native
from joey_research import evaluation as ev


REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
HOST = "raw.githubusercontent.com"
BASE_PATH = f"/PolyAI-LDN/task-specific-datasets/{REVISION}/banking_data/"
SOURCES = {
    "train.csv": (839073, "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"),
    "categories.json": (2036, native.CATEGORIES_SHA256),
}
ROWS = 10003
SALT = "joey:banking77:train-development:v1"
TASK = "banking77_train_development"
PER_STRATUM = 32  # 16 disjoint hash strata; 512 groups, not 512 rows.
INPUTS = "inference/inputs.jsonl"
REQUESTS = "inference/requests.jsonl"
LABELS = "custody/labels.jsonl"
SPLITS = "custody/groups.jsonl"
FILES = (INPUTS, REQUESTS, LABELS, SPLITS, "coverage.json")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(data) -> bytes:
    # Retain insertion order, particularly option_2 before option_10.
    return (json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def _jsonl(rows: Iterable[ev.Row]) -> bytes:
    return b"".join(map(_json, rows))


def _host_only() -> None:
    if ("KUBERNETES_SERVICE_HOST" in os.environ or "KUBERNETES_PORT" in os.environ
            or Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()):
        raise ev.EvaluationError("stage only on the coordinator host, never in a pod/container")


def _size(name: str) -> int:
    if name not in SOURCES:
        raise ev.EvaluationError("file outside the two-file acquisition allowlist")
    return SOURCES[name][0]


def _fetch(name: str) -> bytes:
    limit = _size(name)
    conn = http.client.HTTPSConnection(HOST, timeout=30)
    try:
        conn.request("GET", BASE_PATH + name, headers={"Accept-Encoding": "identity"})
        response = conn.getresponse()
        if response.status != 200 or response.getheader("Content-Encoding", "identity") != "identity":
            raise ev.EvaluationError(f"refused source response for {name} (HTTP {response.status})")
        return response.read(limit + 1)
    finally:
        conn.close()


def _local(directory: Path, name: str) -> bytes:
    size = _size(name)
    with os.fdopen(os.open(directory / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as f:
        info = os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise ev.EvaluationError(f"not the pinned regular file: {name}")
        return f.read(size + 1)


def _verify_sources(data: dict[str, bytes]) -> None:
    if data.keys() != SOURCES.keys():
        raise ev.EvaluationError("expected exactly the approved train and vocabulary files")
    for name, value in data.items():
        if (len(value), _sha(value)) != SOURCES[name]:
            raise ev.EvaluationError(f"source length or SHA-256 differs for {name}")


def _hash(kind: str, value: object) -> str:
    return ev._hash([SALT, kind, value])


@dataclass(frozen=True, slots=True)
class BlindRow:
    source_index: int
    text: str

    @property
    def id(self) -> str:
        return "b77-tr-" + _hash("id", self.source_index)

    @property
    def group(self) -> str:
        text = " ".join(unicodedata.normalize("NFKC", self.text).casefold().split())
        if not text:
            raise ev.EvaluationError("missing or blank complaint")
        return _hash("normalized-text-group", text)


def split_groups(rows: Iterable[BlindRow]) -> tuple[list[ev.Row], ev.Row]:
    groups = defaultdict(list)
    seen = set()
    for r in rows:
        if (type(r.source_index) is not int or r.source_index < 0
                or not isinstance(r.text, str) or r.id in seen):
            raise ev.EvaluationError("invalid or duplicate blind source record")
        seen.add(r.id)
        groups[r.group].append(r.id)
    strata = [[] for _ in range(16)]
    for group in groups:
        strata[int(group[0], 16)].append(group)
    selected = set()
    for stratum in strata:
        if len(stratum) < PER_STRATUM:
            raise ev.EvaluationError("insufficient independent groups in a required hash stratum")
        selected.update(sorted(stratum, key=lambda g: (_hash("within-stratum", g), g))[:PER_STRATUM])
    plan = [
        {"group_id": g, "stratum": int(g[0], 16),
         "partition": "development" if g in selected else "unused", "ids": sorted(ids)}
        for g, ids in sorted(groups.items())
    ]
    counts = {
        "source_rows": len(seen), "source_groups": len(groups),
        "source_duplicate_rows": len(seen) - len(groups), "source_groups_per_stratum": list(map(len, strata)),
        "development_groups_per_stratum": [PER_STRATUM] * 16, "development_groups": len(selected),
        "development_rows": sum(len(groups[g]) for g in selected),
        "unused_groups": len(groups) - len(selected),
    }
    return plan, counts


def _exports(data: dict[str, bytes]) -> dict[str, bytes]:
    _verify_sources(data)
    cats = native._vocabulary(json.loads(data["categories.json"]))
    reader = csv.DictReader(io.StringIO(data["train.csv"].decode("utf-8"), newline=""), strict=True)
    if sorted(reader.fieldnames or []) != ["category", "text"]:
        raise ev.EvaluationError("unexpected training CSV header")
    records = list(reader)
    if len(records) != ROWS or any(r.keys() != {"text", "category"} for r in records):
        raise ev.EvaluationError("unexpected training CSV width or record count")
    blind = [BlindRow(i, r["text"]) for i, r in enumerate(records)]
    plan, coverage = split_groups(blind)
    wanted = {g["group_id"] for g in plan if g["partition"] == "development"}
    inputs, requests, labels = [], [], []
    criteria = dict(zip(native.OPTIONS, cats))
    for r in sorted((r for r in blind if r.group in wanted), key=lambda r: r.id):
        instructions = native._PREFIX + r.text
        row = {"id": r.id, "group_id": r.group, "task": TASK, "prompt": instructions,
               "choices": list(cats)}
        inputs.append(row)
        requests.append({"id": r.id, "input_sha256": ev.input_sha256(row),
                         "payload": {"state": {}, "questions": {"q1": {
                             "type": "choice", "instructions": instructions, "criteria": criteria}}}})
        # Only after group assignment; no unused-partition outcomes are looked up or emitted.
        category = records[r.source_index]["category"]
        if category not in cats:
            raise ev.EvaluationError("selected training record has no valid original category")
        labels.append(ev.label_row(row, cats.index(category)))
    return {INPUTS: _jsonl(inputs), REQUESTS: _jsonl(requests), LABELS: _jsonl(labels),
            SPLITS: _jsonl(plan), "coverage.json": _json(coverage)}


def _new_private_dir(out: Path) -> None:
    if (not out.is_absolute() or not out.parent.is_dir() or out.is_symlink()
            or any((p / ".git").exists() for p in (out.resolve(), *out.resolve().parents))):
        raise ev.EvaluationError("use a fresh absolute output path outside any repository")
    out.mkdir(mode=0o700)  # Exclusive: never overwrite an earlier selection.


def _write(path: Path, value: bytes) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as writer:
        writer.write(value)


def stage(out: Path, *, fetch: bool = False, source_dir: Path | None = None) -> str:
    """Coordinator host only. Return the SHA-256 of the complete custody manifest."""
    _host_only()
    if fetch == (source_dir is not None):
        raise ev.EvaluationError("choose exactly one of fetch and a local source directory")
    _new_private_dir(out)
    source = {n: _local(source_dir, n) if source_dir is not None else _fetch(n) for n in SOURCES}
    files = _exports(source)
    for directory in ("inference", "custody"):
        (out / directory).mkdir(mode=0o700)
    for name, data in files.items():
        _write(out / name, data)
    manifest = _json({
        "schema": "banking77-train-development-v1", "salt": SALT, "task": TASK,
        "unicode_version": unicodedata.unidata_version, "dataset_revision": REVISION,
        "license": "CC-BY-4.0", "native_format_revision": native.KIT_REVISION,
        "source_files": {name: {"url": f"https://{HOST}{BASE_PATH}{name}", "size": size,
                                "sha256": digest} for name, (size, digest) in SOURCES.items()},
        "file_sha256": {name: _sha(data) for name, data in files.items()},
    })
    _write(out / "manifest.json", manifest)  # Last; a partial stage is not a ready bundle.
    return _sha(manifest)


def _read_bundle(bundle: Path, pin: str) -> tuple[list[ev.Row], list[ev.Row], ev.Row]:
    raw = (bundle / "manifest.json").read_bytes()
    if _sha(raw) != pin:
        raise ev.EvaluationError("staged manifest differs from independent pin")
    m = ev._strict_json(raw.decode("utf-8"))
    if (m.get("schema") != "banking77-train-development-v1" or m.get("salt") != SALT
            or m.get("task") != TASK or m.get("dataset_revision") != REVISION
            or set(m.get("file_sha256", {})) != set(FILES)):
        raise ev.EvaluationError("not this train-development protocol")
    parsed = {}
    for name in FILES:
        data = (bundle / name).read_bytes()
        if _sha(data) != m["file_sha256"][name]:
            raise ev.EvaluationError(f"staged file differs from pin: {name}")
        if name in (INPUTS, LABELS):
            parsed[name] = [ev._strict_json(s) for s in data.decode("utf-8").split("\n") if s]
        elif name == "coverage.json":
            parsed[name] = ev._strict_json(data.decode("utf-8"))
    return parsed[INPUTS], parsed[LABELS], parsed["coverage.json"]


def score_development(inputs: Iterable[ev.Row], labels: Iterable[ev.Row],
                      predictions: Iterable[ev.Row], *, projection: str) -> ev.Row:
    """Score one arm on fixed IDs. Macro-F1 uses completed choices, proper scores require coverage."""
    rows = list(inputs)
    ix = ev._index_inputs(rows)
    if (projection not in ("native", "research")
            or any(r["task"] != TASK or len(r["choices"]) != 77 or r["choices"] != rows[0]["choices"]
                   for r in rows)):
        raise ev.EvaluationError("expected the train-development task, its full fixed menu and a known projection")
    labels = list(labels)
    gold = ev._bind(labels, ix, ({"id", "input_sha256", "gold_index"},), "labels")
    if any(type(r["gold_index"]) is not int or not 0 <= r["gold_index"] < 77 for r in labels):
        raise ev.EvaluationError("gold label outside the original 77-category space")
    supplied = {}
    for r in predictions:
        if not isinstance(r, dict) or not isinstance(r.get("id"), str) or r["id"] not in ix:
            raise ev.EvaluationError("unexpected prediction ID")
        key = r["id"]
        if key in supplied or r.get("input_sha256") != ix[key][1]:
            raise ev.EvaluationError("repeated or stale prediction")
        supplied[key] = r
    statuses = Counter()
    actual, predicted, tp = Counter(), Counter(), Counter()
    groups = set()
    disagreements = 0
    vectors = []
    binding = {"id", "input_sha256"}
    for r in rows:
        key = r["id"]
        result = supplied.get(key)
        item = {"id": key, "input_sha256": ix[key][1]}
        y = gold[key]["gold_index"]
        choice = None
        if result is None:
            status = "pending"
        elif projection == "native":
            status = result.get("status")
            if status not in ("ok", "abstained", "unsupported", "error", "pending"):
                raise ev.EvaluationError("unknown native status")
            if set(result) != binding | ({"status", "response"} if status == "ok" else {"status"}):
                raise ev.EvaluationError("unexpected native prediction fields")
            if status == "ok":
                choice, probs = native._selected_index(result)
                item["probs"] = probs
                disagreements += choice != probs.index(max(probs))
        else:
            if set(result) == binding | {"error"} and ev._text(result["error"]):
                status = "error"
            elif set(result) == binding | {"probs"}:
                item["probs"] = result["probs"]
                obs = ev._observe(r, y, item)
                status = "ok" if obs.error is None else "invalid_distribution"
                choice = obs.choice
            else:
                raise ev.EvaluationError("unexpected research prediction fields")
        statuses[status] += 1
        if choice is not None:
            actual[y] += 1
            predicted[choice] += 1
            tp[y] += y == choice
            groups.add(r["group_id"])
        if "probs" not in item:
            item["error"] = status
        vectors.append(item)
    obs_classes = actual.keys() | predicted.keys()
    f1 = sum(2 * tp[k] / (actual[k] + predicted[k]) for k in sorted(obs_classes))
    research = ev.evaluate(rows, labels, {projection: vectors}, baseline=projection, bootstrap_reps=0)
    ok = statuses["ok"]
    return {"task": TASK, "scope": "train-derived development; not an official benchmark score",
            "projection": projection, "n": len(rows), "groups": len({r["group_id"] for r in rows}),
            "states_with_choice": ok, "groups_with_any_choice": len(groups),
            "choice_coverage": ok / len(rows), "status_counts": dict(statuses),
            "scored_gold_or_predicted_classes": len(obs_classes),
            "observed_class_macro_f1": f1 / len(obs_classes) if obs_classes else None,
            "choice_accuracy_completed": sum(tp.values()) / ok if ok else None,
            "declared_choice_not_first_argmax": disagreements if projection == "native" else None,
            "probability_metrics": research["arms"][projection]["micro"]}


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stage", help="on the coordinator host; never run in a pod")
    st.add_argument("--out", type=Path, required=True)
    source = st.add_mutually_exclusive_group(required=True)
    source.add_argument("--fetch", action="store_true", help="GET only the two pinned public files")
    source.add_argument("--source-dir", type=Path, help="offline directory containing those files")
    sc = sub.add_parser("score")
    sc.add_argument("--bundle", type=Path, required=True)
    sc.add_argument("--manifest-sha256", required=True)
    sc.add_argument("--projection", choices=("research", "native"), required=True)
    sc.add_argument("--predictions", type=Path, required=True)
    args = cli.parse_args()
    try:
        if args.cmd == "stage":
            print(stage(args.out, fetch=args.fetch, source_dir=args.source_dir))
        else:
            rows, labels, coverage = _read_bundle(args.bundle, args.manifest_sha256)
            report = score_development(rows, labels, ev.read_jsonl(args.predictions), projection=args.projection)
            print(json.dumps({**report, "manifest_sha256": args.manifest_sha256,
                              "sampling_coverage": coverage}, allow_nan=False))
    except (ValueError, OSError, csv.Error, http.client.HTTPException) as exc:
        cli.exit(2, f"banking77-development: {exc}\n")


if __name__ == "__main__":
    main()
