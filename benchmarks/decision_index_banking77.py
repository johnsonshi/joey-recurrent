"""Local BANKING77 Choice adapter for Decision Index edition 0.3, catalog 4.

No downloads, bundled questions, labels, source execution, or index computation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import unicodedata

from joey_research import evaluation as ev


KIT_REVISION = "f5a6e4fd85fed05e77878dfc4966d0568bfb45ec"
EDITION = "0.3"
REFERENCE_MODEL = "jev-1.13.0"
CSV_SHA256 = "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"
CATEGORIES_SHA256 = "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63"
CASES = 3080
OPTIONS = tuple(f"option_{i}" for i in range(77))
_PREFIX = "Classify the banking intent of this user request:\n"


def _pinned_bytes(path: str | Path, digest: str) -> bytes:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ev.EvaluationError("local BANKING77 file does not match the pinned source")
    return raw


def _vocabulary(names: Sequence[str]) -> tuple[str, ...]:
    if (not isinstance(names, (tuple, list)) or len(names) != len(OPTIONS)
            or not all(isinstance(s, str) and s.strip() for s in names)
            or len(set(names)) != len(OPTIONS)):
        raise ev.EvaluationError("BANKING77 requires 77 unique category descriptions in source order")
    return tuple(names)


def load_categories(path: str | Path) -> tuple[str, ...]:
    """Read the pinned local menu only. It has no state-specific outcomes."""
    return _vocabulary(json.loads(_pinned_bytes(path, CATEGORIES_SHA256)))


@dataclass(frozen=True, slots=True)
class Case:
    index: int
    text: str
    categories: tuple[str, ...]

    def __post_init__(self):
        if type(self.index) is not int or not 0 <= self.index < CASES:
            raise ev.EvaluationError("expected the original zero-based BANKING77 test row index")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ev.EvaluationError("missing BANKING77 utterance")
        object.__setattr__(self, "categories", _vocabulary(self.categories))

    @property
    def source_id(self) -> str:
        return f"BANKING77:test:{self.index}"

    @property
    def run_id(self) -> str:
        return f"4:BANKING77:{self.source_id}"

    def native_payload(self) -> ev.Row:
        return {"state": {}, "questions": {"q1": {
            "type": "choice", "instructions": _PREFIX + self.text,
            "criteria": dict(zip(OPTIONS, self.categories)),
        }}}

    def native_digest(self) -> str:
        # Unlike Joey's row hash, the kit hashes insertion order and includes the reference model.
        payload = {"model": REFERENCE_MODEL, **self.native_payload()}
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def research_input(self) -> ev.Row:
        state = " ".join(unicodedata.normalize("NFKC", self.text).casefold().split())
        group = "banking77-text-" + hashlib.sha256(state.encode("utf-8")).hexdigest()
        return {"id": self.source_id, "group_id": group, "task": "banking77_intent",
                "prompt": _PREFIX + self.text, "choices": list(self.categories)}


def project_blind(record: ev.Row, categories: Sequence[str]) -> Case:
    """Project a label-free local export. Reject a source row carrying its gold."""
    if not isinstance(record, dict) or set(record) != {"source_index", "text"}:
        raise ev.EvaluationError("blinded record must contain exactly source_index and text")
    return Case(record["source_index"], record["text"], _vocabulary(categories))


def _decode_release(raw_csv: bytes, categories: tuple[str, ...]) -> tuple[list[Case], list[ev.Row]]:
    reader = csv.DictReader(io.StringIO(raw_csv.decode("utf-8"), newline=""), strict=True)
    if reader.fieldnames is None or sorted(reader.fieldnames) != ["category", "text"]:
        raise ev.EvaluationError("unexpected BANKING77 CSV fields")
    cases, labels = [], []
    for i, record in enumerate(reader):
        if set(record) != {"text", "category"} or record["category"] not in categories:
            raise ev.EvaluationError("malformed row or unknown source category")
        case = project_blind({"source_index": i, "text": record["text"]}, categories)
        cases.append(case)
        labels.append(ev.label_row(case.research_input(), categories.index(record["category"])))
    return cases, labels


def load_released_test(test_csv: str | Path, categories_json: str | Path, *,
                       finalists: str, finalists_sha256: str, manifest: ev.Row, manifest_sha256: str,
                       arms: set[str], baseline: str, passes: list[str] | None = None
                       ) -> tuple[list[Case], list[ev.Row]]:
    """Custodian only. Check the pinned freeze before opening either source file."""
    ev._check_finalists(finalists, finalists_sha256, {"sha256": manifest_sha256}, arms, baseline, passes)
    if (not isinstance(manifest, dict) or manifest.get("schema") != ev.SCHEMA
            or manifest.get("sha256") != manifest_sha256
            or ev._hash({k: v for k, v in manifest.items() if k != "sha256"}) != manifest_sha256
            or not isinstance(manifest.get("entries"), list)):
        raise ev.EvaluationError("missing or unpinned input manifest")
    entries = {}
    for entry in manifest["entries"]:
        if not isinstance(entry, dict):
            raise ev.EvaluationError("malformed input manifest")
        if entry.get("task") == "banking77_intent":
            kid = entry.get("id")
            if (not isinstance(kid, str) or kid in entries or entry.get("split") != "final"
                    or not ev._sha256_text(entry.get("input_sha256"))):
                raise ev.EvaluationError("combined BANKING77 test requires full final-only release")
            entries[kid] = entry["input_sha256"]
    if entries.keys() != {f"BANKING77:test:{i}" for i in range(CASES)}:
        raise ev.EvaluationError("combined BANKING77 test requires full final-only release")
    categories = load_categories(categories_json)
    cases, labels = _decode_release(_pinned_bytes(test_csv, CSV_SHA256), categories)
    if {l["id"]: l["input_sha256"] for l in labels} != entries:
        raise ev.EvaluationError("released source does not match the frozen blinded inputs")
    return cases, labels


def _selected_index(result: ev.Row) -> tuple[int, list[float]]:
    response = result.get("response")
    answers = response.get("answers") if isinstance(response, dict) else None
    if not isinstance(answers, dict) or set(answers) != {"q1"}:
        raise ev.EvaluationError("an ok BANKING77 result must answer exactly q1")
    a = answers["q1"]
    if not isinstance(a, dict) or a.get("type") != "choice" or a.get("choice") not in OPTIONS:
        raise ev.EvaluationError("invalid native choice")
    p = a.get("probabilities")
    if (not isinstance(p, dict) or set(p) != set(OPTIONS)
            or any(type(v) not in (float, int) or not 0 <= v <= 1 for v in p.values())
            or abs(sum(p.values()) - 1) > .01):
        raise ev.EvaluationError("native choice requires a valid probability for all 77 options")
    return OPTIONS.index(a["choice"]), [p[key] for key in OPTIONS]


def score_native(cases: Iterable[Case], labels: Iterable[ev.Row], results: Iterable[ev.Row]) -> ev.Row:
    """Native single-task statistic on complete groups; never a public/Full index."""
    cases = list(cases)
    if not cases or len({c.index for c in cases}) != len(cases):
        raise ev.EvaluationError("empty selection or duplicate source index")
    if any(c.categories != cases[0].categories for c in cases):
        raise ev.EvaluationError("selection mixes BANKING77 menus or option orders")
    by_run = {c.run_id: c for c in cases}
    by_id = {c.source_id: c for c in cases}
    gold = {}
    for label in labels:
        if not isinstance(label, dict) or set(label) != {"id", "input_sha256", "gold_index"}:
            raise ev.EvaluationError("invalid label schema")
        key, g = label["id"], label["gold_index"]
        if not isinstance(key, str) or key not in by_id or key in gold:
            raise ev.EvaluationError("unknown or duplicate label")
        if (type(g) is not int or not 0 <= g < len(OPTIONS)
                or label["input_sha256"] != ev.input_sha256(by_id[key].research_input())):
            raise ev.EvaluationError("invalid or stale label")
        gold[key] = g
    if set(gold) != set(by_id):
        raise ev.EvaluationError("missing label for a selected state")
    seen = set()
    statuses: Counter[str] = Counter()
    true_count, pred_count, true_positive = Counter(), Counter(), Counter()
    disagree = 0
    for result in results:
        if not isinstance(result, dict):
            raise ev.EvaluationError("invalid result record")
        rid = result.get("run_id")
        if not isinstance(rid, str) or rid not in by_run or rid in seen:
            raise ev.EvaluationError("unknown or duplicate native run ID")
        case = by_run[rid]
        if result.get("payload_sha256") != case.native_digest():
            raise ev.EvaluationError("stale native payload digest")
        if result.get("group_id", case.source_id) != case.source_id or result.get("catalog_id", 4) != 4:
            raise ev.EvaluationError("wrong native group or catalog")
        status = result.get("status")
        if status not in ("ok", "error", "unsupported", "abstained", "pending"):
            raise ev.EvaluationError("unknown native result status")
        seen.add(rid)
        statuses[status] += 1
        if status == "ok":
            selected, probs = _selected_index(result)
            target = gold[case.source_id]
            # Option IDs are positions. Native macro-F1 operates on semantic categories.
            y, p = case.categories[target], case.categories[selected]
            true_count[y] += 1
            pred_count[p] += 1
            true_positive[y] += y == p
            disagree += selected != probs.index(max(probs))
    statuses["pending"] += len(cases) - len(seen)
    observed = true_count.keys() | pred_count.keys()
    macro_f1 = (sum(2 * true_positive[k] / (true_count[k] + pred_count[k]) for k in sorted(observed))
                / len(observed)) if observed else None
    ok = statuses["ok"]
    return {
        "edition": EDITION, "catalog_id": 4, "scope": "local_selected_cases_not_an_index",
        "requests": len(cases), "official_test_requests": CASES, "answered": ok,
        "scored_requests": ok, "unsupported": statuses["unsupported"], "errors": statuses["error"],
        "abstained": statuses["abstained"], "pending": statuses["pending"],
        "answered_fraction": ok / len(cases), "selected_fraction_of_test": len(cases) / CASES,
        "observed_gold_or_predicted_classes": len(observed),
        "metric": "macro-F1", "score": macro_f1,
        "field_accuracy_scored": sum(true_positive.values()) / ok if ok else None,
        "declared_choice_not_first_argmax": disagree,
    }
