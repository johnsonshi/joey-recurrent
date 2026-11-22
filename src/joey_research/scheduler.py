"""A two-slot, restartable queue. It does not execute manifests or access a provider."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol


HOUR = 3600
FREEZE_AFTER_S = 23 * HOUR
FINALIZE_AFTER_S = 25 * HOUR
WALL_TARGET_S = 26 * HOUR
MAX_EXTENSION_HOURS = 4
MAX_WALL_S = WALL_TARGET_S + MAX_EXTENSION_HOURS * HOUR
GPU_CAP_S = 48 * HOUR  # Across both one-A100 slots, not two times the wall target.
MAX_ATTEMPTS = 3  # Includes the original admission.
_MAX_UTC = (1 << 62) - 1


class PolicyError(ValueError):
    """Input or action violates the persisted experiment policy."""


class ClockRegression(PolicyError):
    """UTC is behind a time already witnessed by this database. Fail closed."""


class BackendUnavailable(Exception):
    """An ambiguous transport/service failure, never proof of non-execution."""


class Split(str, Enum):
    DEV = "dev"
    TEST = "test"


class Gate(str, Enum):
    OFFLINE_STAGING = "offline_staging"
    STOCK_PARITY = "stock_parity"


class Role(str, Enum):
    STOCK = "stock"
    SELECTED = "selected"
    ZERO_EXTRA = "zero_extra"
    ORDER_ENSEMBLE = "order_ensemble"


class Outcome(str, Enum):
    UNKNOWN = "unknown"
    LIVE = "live"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYABLE = "retryable_failure"
    FENCED_ABSENT = "fenced_absent"


def _utc() -> int:
    return time.time_ns() // 1_000_000_000


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode("ascii")).hexdigest()


def _digest(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PolicyError("expected a lowercase SHA-256 digest")
    return value


def _tag(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,47}", value):
        raise PolicyError("expected a short class or precision identifier")
    return value


def _integer(value: object, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise PolicyError(f"expected an integer in [{low}, {high}]")
    return value


@dataclass(frozen=True)
class TrialKey:
    model: str
    dataset: str  # Split-specific: the development and held-out digests must differ.
    code: str
    config: str
    seed: int
    scorer: str
    precision: str

    def __post_init__(self) -> None:
        for digest in (self.model, self.dataset, self.code, self.config, self.scorer):
            _digest(digest)
        _integer(self.seed, 0, (1 << 63) - 1)
        _tag(self.precision)

    @property
    def trial_id(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class Manifest:
    """Descriptor of an immutable artifact already validated outside the queue."""

    sha256: str
    key: TrialKey
    split: Split
    gpu_class: str
    budget_s: int  # Total attempt envelope, including admission latency.

    def __post_init__(self) -> None:
        _digest(self.sha256)
        if not isinstance(self.key, TrialKey) or not isinstance(self.split, Split):
            raise PolicyError("expected a typed key and split")
        _tag(self.gpu_class)
        _integer(self.budget_s, 1, MAX_WALL_S)


@dataclass(frozen=True)
class Pick:
    dev_id: str
    test_id: str
    role: Role = Role.SELECTED

    def __post_init__(self) -> None:
        _digest(self.dev_id)
        _digest(self.test_id)
        if not isinstance(self.role, Role):
            raise PolicyError("expected a typed frozen role")


@dataclass(frozen=True)
class MethodWitness:
    """Exact canonical worker settings; a new control must name its selected Pick."""

    settings_json: bytes
    parent: Pick | None = None


def _verified_settings(raw: bytes, config: str) -> dict:
    # Legacy nominations need neither worker settings nor a worker installation.
    from .worker import method_sha256

    if type(raw) is not bytes or not 0 < len(raw) <= 8192:
        raise PolicyError("expected bounded canonical method bytes")
    try:
        settings = json.loads(raw)
        canonical = _json(settings).encode("ascii")
        digest = method_sha256(settings)  # Includes the worker's exact settings validation.
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise PolicyError("invalid worker method settings") from exc
    if raw != canonical or digest != config:
        raise PolicyError("method witness is not the exact registered canonical configuration")
    return settings


def _check_method_role(role: Role, settings: dict) -> None:
    loop = settings["loop"]
    k, method = loop.get("k", 1), loop.get("method", "naive")
    if "policy" in settings or loop.get("readout", "final") != "final":
        raise PolicyError("confirmatory picks require fixed passes and a final readout")
    if ("order_ensemble" in settings) != (role == Role.ORDER_ENSEMBLE):
        raise PolicyError("order-ensemble settings and role must agree")
    if role == Role.SELECTED:
        if k == 1 or method == "zero_extra":
            raise PolicyError("a witnessed selected pick must be a recurrent candidate")
    elif role == Role.STOCK:
        if k != 1 or method != "naive" or settings["reverse"]:
            raise PolicyError("ordinary stock must be a forward one-pass call")
    elif role == Role.ZERO_EXTRA:
        if k == 1 or method != "zero_extra":
            raise PolicyError("zero-extra stock must actually take the one-traversal split path")
    elif role == Role.ORDER_ENSEMBLE:
        # method_sha256 already requires forward full stock, k=2 zero_extra and
        # a valid whole-group budget. The allocation seed is not the trial seed.
        if settings["order_ensemble"]["seed"] != 91473:  # configs/evaluation/promotion.json
            raise PolicyError("order ensemble differs from the preregistered group allocation seed")


def _bind_methods(keys: dict[Pick, tuple[TrialKey, TrialKey]],
                  witnesses: Mapping[Pick, MethodWitness] | None,
                  payloads: Mapping[str, str | None] | None) -> tuple[dict[Pick, dict], dict[str, str | None]]:
    if not isinstance(witnesses, Mapping) or not isinstance(payloads, Mapping):
        raise PolicyError("a control proposal needs method witnesses and independent model payload pins")
    witnesses, payloads = dict(witnesses), dict(payloads)
    if witnesses.keys() != keys.keys() or payloads.keys() != {dev.model for dev, _ in keys.values()}:
        raise PolicyError("witnesses and payload pins must cover exactly the nominated pairs and models")
    compact_pins = [_digest(pin) for pin in payloads.values() if pin is not None]
    if len(compact_pins) != len(set(compact_pins)):
        raise PolicyError("one compact payload cannot be nominated under multiple model identities")

    evidence = {}
    claimed_roles = set()
    recipes = {}
    for p, (dev, test) in keys.items():
        w = witnesses[p]
        if not isinstance(w, MethodWitness) or (dev.trial_id, test.trial_id) != (p.dev_id, p.test_id):
            raise PolicyError("expected a method witness for this exact registered trial pair")
        # The pair check in nominate already proved dev.config == test.config.
        settings = _verified_settings(w.settings_json, dev.config)
        _check_method_role(p.role, settings)
        if settings.get("compact_manifest_sha256") != payloads[dev.model]:
            raise PolicyError("method differs from the independently pinned full or compact payload")
        recipe = settings.get("compact_recipe")
        if recipes.setdefault(dev.model, recipe) != recipe:
            raise PolicyError("compact recipe differs for the same payload")
        slot = (dev.model, p.role)
        if slot in claimed_roles:
            raise PolicyError("only one witnessed pick per model payload and role")
        claimed_roles.add(slot)
        evidence[p] = {"settings": settings, "method_sha256": dev.config}

    for p, (dev, test) in keys.items():
        parent = witnesses[p].parent
        if p.role in (Role.SELECTED, Role.STOCK):
            if parent is not None:
                raise PolicyError("only the additional control roles name a selected parent")
            continue
        if not isinstance(parent, Pick) or parent.role != Role.SELECTED or parent not in keys:
            raise PolicyError("control parent must be an exact selected pair in this proposal")
        parent_dev, parent_test = keys[parent]
        # Across the contrast only the method may differ, on BOTH splits.
        if replace(dev, config=parent_dev.config) != parent_dev or test.dataset != parent_test.dataset:
            raise PolicyError("control and selected parent differ in trial identity")
        settings, selected = evidence[p]["settings"], evidence[parent]["settings"]
        if settings["window"] != selected["window"] or settings["reverse"] != selected["reverse"]:
            raise PolicyError("control must match its selected window and option order")
        if p.role == Role.ORDER_ENSEMBLE:
            lo, hi = selected["window"]
            extra_visits = (hi - lo) * (selected["loop"]["k"] - 1)
            spend = settings["order_ensemble"]
            if 24 * spend["numerator"] != extra_visits * spend["denominator"]:
                raise PolicyError("whole-group spend does not match the selected extra layer visits")
        evidence[p]["selected_parent"] = {**asdict(parent), "method_sha256": parent_dev.config}
    return evidence, payloads


@dataclass(frozen=True)
class Admission:
    run_id: str
    token: str
    trial_id: str
    manifest_sha: str
    split: Split
    attempt: int
    slot: int
    generation: int
    gpu_class: str
    issued_at: int
    admit_before: int  # Exclusive; heartbeats never extend permission to start.
    hard_stop: int  # Stop all physical work by this UTC instant.
    run_deadline: int


@dataclass(frozen=True)
class Observation:
    token: str
    outcome: Outcome
    at: int | None = None  # Actual worker pulse or finished/fenced UTC, not poll time.
    fence_sha: str | None = None
    result_sha: str | None = None

    def __post_init__(self) -> None:
        _digest(self.token)
        if not isinstance(self.outcome, Outcome):
            raise PolicyError("expected a typed observation")
        if self.outcome == Outcome.UNKNOWN:
            if self.at is not None:
                raise PolicyError("unknown observation cannot attest a time")
        else:
            _integer(self.at, 0, _MAX_UTC)
        if self.outcome in (Outcome.UNKNOWN, Outcome.LIVE):
            if self.fence_sha is not None:
                raise PolicyError("a nonterminal observation cannot attest a fence")
        else:
            _digest(self.fence_sha)
        if self.outcome == Outcome.SUCCEEDED:
            _digest(self.result_sha)
        elif self.result_sha is not None:
            raise PolicyError("results require successful terminal evidence")


class Backend(Protocol):
    """Both calls must be bounded; see docs/scheduler.md for physical fencing."""

    def admit(self, ticket: Admission) -> Observation: ...

    def observe(self, ticket: Admission) -> Observation: ...


_SCHEMA = """
CREATE TABLE run (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1), run_id TEXT NOT NULL,
    start_utc INTEGER NOT NULL, last_seen INTEGER NOT NULL, lease_s INTEGER NOT NULL,
    watchdog_at INTEGER, proposal TEXT, closed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE slots (
    slot INTEGER PRIMARY KEY CHECK(slot IN (0,1)), gpu_class TEXT NOT NULL
);
CREATE TABLE datasets (
    digest TEXT PRIMARY KEY, split TEXT NOT NULL CHECK(split IN ('dev','test'))
);
CREATE TABLE trials (
    id TEXT PRIMARY KEY, manifest_sha TEXT UNIQUE NOT NULL, key_json TEXT NOT NULL,
    split TEXT NOT NULL CHECK(split IN ('dev','test')), gpu_class TEXT NOT NULL,
    budget_s INTEGER NOT NULL, priority INTEGER NOT NULL, max_attempts INTEGER NOT NULL
        CHECK(max_attempts BETWEEN 1 AND 3 AND (split='dev' OR max_attempts=1)),
    state TEXT NOT NULL CHECK(state IN
        ('candidate','queued','active','succeeded','failed','expired','not_selected')),
    result_sha TEXT, UNIQUE(id,split)
);
CREATE TABLE gates (
    trial_id TEXT NOT NULL REFERENCES trials, kind TEXT NOT NULL
        CHECK(kind IN ('offline_staging','stock_parity')), proof TEXT NOT NULL,
    PRIMARY KEY(trial_id,kind)
);
CREATE TABLE dependencies (
    trial_id TEXT NOT NULL REFERENCES trials, parent TEXT NOT NULL REFERENCES trials,
    PRIMARY KEY(trial_id,parent)
);
CREATE TABLE attempts (
    token TEXT PRIMARY KEY, trial_id TEXT NOT NULL, split TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal>=1 AND (split='dev' OR ordinal=1)),
    slot INTEGER NOT NULL REFERENCES slots, generation INTEGER NOT NULL,
    issued_at INTEGER NOT NULL, admit_before INTEGER NOT NULL, hard_stop INTEGER NOT NULL,
    lease_until INTEGER NOT NULL, pulse INTEGER, state TEXT NOT NULL
        CHECK(state IN ('reserved','live','reconciling','closed')),
    FOREIGN KEY(trial_id,split) REFERENCES trials(id,split),
    UNIQUE(trial_id,ordinal), UNIQUE(slot,generation)
);
CREATE UNIQUE INDEX occupied_slot ON attempts(slot) WHERE state!='closed';
CREATE UNIQUE INDEX active_trial ON attempts(trial_id) WHERE state!='closed';
CREATE UNIQUE INDEX once_only_test ON attempts(trial_id) WHERE split='test';
CREATE TABLE receipts (
    token TEXT PRIMARY KEY REFERENCES attempts, observation TEXT NOT NULL,
    recorded_at INTEGER NOT NULL, accepted INTEGER NOT NULL CHECK(accepted IN (0,1))
);
CREATE TABLE seals (
    name TEXT PRIMARY KEY CHECK(name IN ('freeze','finalize')),
    effective_at INTEGER NOT NULL, recorded_at INTEGER NOT NULL,
    sha256 TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE events (
    seq INTEGER PRIMARY KEY, utc INTEGER NOT NULL, kind TEXT NOT NULL,
    trial_id TEXT REFERENCES trials, token TEXT REFERENCES attempts, detail TEXT NOT NULL
);
CREATE VIEW trial_registry AS
    SELECT t.*, (SELECT COUNT(*) FROM attempts a WHERE a.trial_id=t.id) AS attempts,
        (SELECT a.token FROM attempts a WHERE a.trial_id=t.id
            ORDER BY a.ordinal DESC LIMIT 1) AS latest_token,
        (SELECT proof FROM gates WHERE trial_id=t.id AND kind='offline_staging') AS offline_sha,
        (SELECT proof FROM gates WHERE trial_id=t.id AND kind='stock_parity') AS parity_sha
    FROM trials t;
CREATE VIEW status_counts AS
    SELECT split,state,COUNT(*) AS count FROM trials GROUP BY split,state;
"""


def _terminal_gpu_s(issued_at: int, hard_stop: int, outcome: str, at: int) -> int:
    if outcome == Outcome.FENCED_ABSENT:
        return 0  # Irreversible proof that this token never started.
    # Timestamps have only second precision. Include the partial terminal second,
    # but not past the enforced stop unless the evidence itself is later.
    end = hard_stop if at == hard_stop else at + 1
    return end - issued_at


def _install_budget(db: sqlite3.Connection, start: int, offsets: tuple[int, int, int],
                    prior_gpu_s: int = 0) -> None:
    """Called inside the creation/upgrade transaction; v1 evidence is never rewritten."""
    db.execute("CREATE TABLE run_budget (singleton INTEGER PRIMARY KEY CHECK(singleton=1) REFERENCES run, "
               "freeze_at INTEGER NOT NULL, finalize_at INTEGER NOT NULL, deadline INTEGER NOT NULL, "
               "cap_s INTEGER NOT NULL, charged_s INTEGER NOT NULL CHECK(charged_s>=0))")
    db.execute("CREATE TRIGGER pinned_run_limits BEFORE UPDATE OF singleton,freeze_at,finalize_at,deadline,cap_s "
               "ON run_budget BEGIN SELECT RAISE(ABORT,'immutable run limits'); END")
    db.execute("CREATE TRIGGER monotone_gpu_charge BEFORE UPDATE OF charged_s ON run_budget "
               "WHEN NEW.charged_s<OLD.charged_s BEGIN SELECT RAISE(ABORT,'cannot reduce GPU charges'); END")
    charged = prior_gpu_s
    for r in db.execute("SELECT issued_at,hard_stop,observation FROM attempts LEFT JOIN receipts USING(token) "
                        "WHERE state='closed'"):
        if r["observation"] is None:
            raise PolicyError("cannot resume a closed attempt without terminal evidence")
        observation = json.loads(r["observation"])
        charged += _terminal_gpu_s(r["issued_at"], r["hard_stop"], observation["outcome"], observation["at"])
    db.execute("INSERT INTO run_budget VALUES(1,?,?,?,?,?)",
               (*(start + offset for offset in offsets), GPU_CAP_S, charged))
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS once_extension ON events(kind) WHERE kind='timing_extended'")
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS reservation_event ON events(token) WHERE kind='reserved'")
    db.execute("PRAGMA user_version = 2")


class Queue:
    """Use one connection per thread/process and a single durable DB per experiment."""

    def __init__(self, db: sqlite3.Connection, clock: Callable[[], int]):
        self._db = db
        self._clock = clock
        row = self._db.execute("SELECT * FROM run JOIN run_budget USING(singleton) WHERE singleton=1").fetchone()
        if row is None:
            raise PolicyError("missing experiment origin; do not recreate an active queue")
        self._run_id = row["run_id"]
        self._start_utc = row["start_utc"]
        self._lease_s = row["lease_s"]
        self._scheduled_freeze = row["freeze_at"]
        self._test_window_s = row["finalize_at"] - row["freeze_at"]
        self._nominal_deadline = row["deadline"]

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def start_utc(self) -> int:
        return self._start_utc

    @property
    def lease_s(self) -> int:
        return self._lease_s

    @property
    def freeze_at(self) -> int:
        seal = self._db.execute("SELECT effective_at FROM seals WHERE name='freeze'").fetchone()
        if seal:
            return seal[0]
        extension = self._extension()
        return extension["freeze_at"] if extension else self._scheduled_freeze

    @property
    def finalize_at(self) -> int:
        seal = self._db.execute("SELECT effective_at FROM seals WHERE name='finalize'").fetchone()
        return seal[0] if seal else self.freeze_at + self._test_window_s

    @property
    def nominal_deadline(self) -> int:
        return self._nominal_deadline

    def _extension(self) -> dict | None:
        r = self._db.execute("SELECT utc,detail FROM events WHERE kind='timing_extended'").fetchone()
        return {**json.loads(r["detail"]), "recorded_at": r["utc"]} if r else None

    @property
    def deadline(self) -> int:
        extension = self._extension()
        return extension["deadline"] if extension else self.nominal_deadline

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        db = sqlite3.connect(path.absolute().as_uri() + "?mode=rw", uri=True,
                             timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        return db

    @classmethod
    def create(cls, path: str | Path, *, run_id: str, start_utc: int,
               gpu_classes: tuple[str, str], lease_s: int = 90, prior_gpu_s: int = 0,
               clock: Callable[[], int] = _utc) -> Queue:
        """start_utc is the first verified post-setup UTC, not the setup launch."""
        _digest(run_id)
        now = _integer(clock(), 0, _MAX_UTC)
        _integer(start_utc, 0, min(now, _MAX_UTC - MAX_WALL_S))
        _integer(lease_s, 1, HOUR)
        _integer(prior_gpu_s, 0, GPU_CAP_S)
        if not isinstance(gpu_classes, tuple) or len(gpu_classes) != 2:
            raise PolicyError("bind exactly two physical slots")
        for gpu_class in gpu_classes:
            _tag(gpu_class)
        path = Path(path)
        with path.open("xb"):
            pass  # Never reset or replace an existing run.
        db = cls._connect(path)
        try:
            db.executescript("BEGIN IMMEDIATE;\n" + _SCHEMA)
            for table in ("receipts", "seals", "events"):
                for op in ("UPDATE", "DELETE"):
                    db.execute(f"CREATE TRIGGER immutable_{table}_{op} BEFORE {op} ON {table} "
                               "BEGIN SELECT RAISE(ABORT,'immutable evidence'); END")
            db.execute("INSERT INTO run(singleton,run_id,start_utc,last_seen,lease_s) "
                       "VALUES(1,?,?,?,?)", (run_id, start_utc, now, lease_s))
            _install_budget(db, start_utc, (FREEZE_AFTER_S, FINALIZE_AFTER_S, WALL_TARGET_S), prior_gpu_s)
            db.executemany("INSERT INTO slots VALUES(?,?)", enumerate(gpu_classes))
            queue = cls(db, clock)
            queue._event(now, "created", detail={"start_utc": start_utc, "prior_gpu_s": prior_gpu_s})
            queue._advance(now)
            db.commit()
            return queue
        except BaseException:
            db.close()
            raise

    @classmethod
    def resume(cls, path: str | Path, *, run_id: str, clock: Callable[[], int] = _utc) -> Queue:
        _digest(run_id)
        db = cls._connect(Path(path))
        try:
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version == 1:
                row = db.execute("SELECT run_id,start_utc FROM run WHERE singleton=1").fetchone()
                if row is None or row["run_id"] != run_id:
                    raise PolicyError("run identity mismatch")
                _install_budget(db, row["start_utc"], (21 * HOUR, 23 * HOUR, 24 * HOUR))
            elif version != 2:
                raise PolicyError("unsupported or incomplete queue database")
            queue = cls(db, clock)
            if queue.run_id != run_id:
                raise PolicyError("run identity mismatch")
            queue._witness()
            db.commit()
            return queue
        except BaseException:
            db.close()
            raise

    def close(self) -> None:
        self._db.close()

    def _witness(self) -> int:
        now = _integer(self._clock(), 0, _MAX_UTC)
        previous = self._db.execute("SELECT last_seen FROM run").fetchone()[0]
        if now < previous:
            raise ClockRegression("UTC regressed; no admission until the clock is trustworthy")
        self._db.execute("UPDATE run SET last_seen=?", (now,))
        self._advance(now)
        return now

    @contextmanager
    def _transaction(self) -> Iterator[int]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            now = self._witness()
            self._db.execute("SAVEPOINT operation")
            try:
                yield now
            except PolicyError:
                # Reject the request, but do not un-witness a deadline or undo its seal.
                self._db.execute("ROLLBACK TO operation")
                self._db.commit()
                raise
            self._db.commit()
        finally:
            if self._db.in_transaction:
                self._db.rollback()

    def _event(self, now: int, kind: str, trial: str | None = None, token: str | None = None,
               detail: object = None) -> None:
        self._db.execute("INSERT INTO events(utc,kind,trial_id,token,detail) VALUES(?,?,?,?,?)",
                         (now, kind, trial, token, _json(detail)))

    def _phase(self, now: int) -> str:
        if now < self.freeze_at:
            return "search"
        if now < self.finalize_at:
            return "test"
        return "closed" if self._db.execute("SELECT closed FROM run").fetchone()[0] else "finalize"

    def _seal(self, name: str, effective: int, now: int, payload: dict, reason_sha: str | None) -> None:
        value = {"run_id": self.run_id, "name": name, "effective_at": effective, **payload}
        if reason_sha is not None:
            value["reason_sha"] = reason_sha
        digest = _hash(value)
        self._db.execute("INSERT INTO seals VALUES(?,?,?,?,?)",
                         (name, effective, now, digest, _json(value)))
        self._event(now, name, detail={"effective_at": effective, "sha256": digest})

    def _freeze(self, effective: int, now: int, reason_sha: str | None = None) -> None:
        db = self._db
        proposal = db.execute("SELECT proposal FROM run").fetchone()[0]
        plan = json.loads(proposal) if proposal else {"decision_sha": None, "picks": []}
        self._seal("freeze", effective, now, plan, reason_sha)
        db.executemany("UPDATE trials SET state='queued' WHERE id=? AND state='candidate'",
                       ((p["test_id"],) for p in plan["picks"]))
        db.execute("UPDATE trials SET state='not_selected' WHERE state='candidate'")
        db.execute("UPDATE trials SET state='expired' WHERE state='queued' AND split='dev'")

    def _finalize(self, effective: int, now: int, reason_sha: str | None = None) -> None:
        db = self._db
        db.execute("UPDATE trials SET state='expired' WHERE state='queued'")
        freeze = db.execute("SELECT sha256,payload FROM seals WHERE name='freeze'").fetchone()
        ids = {p["test_id"] for p in json.loads(freeze["payload"])["picks"]}
        results = []
        for trial in sorted(ids):
            row = db.execute(
                "SELECT t.id,t.state,t.result_sha,a.token,r.observation,r.accepted FROM trials t "
                "LEFT JOIN attempts a ON a.trial_id=t.id LEFT JOIN receipts r ON r.token=a.token "
                "WHERE t.id=?", (trial,)).fetchone()
            results.append({"test_id": trial, "state": row["state"], "token": row["token"],
                            "result_sha": row["result_sha"],
                            "accepted": bool(row["accepted"]) if row["accepted"] is not None else None,
                            "receipt": json.loads(row["observation"]) if row["observation"] else None})
        self._seal("finalize", effective, now, {"freeze_sha": freeze["sha256"], "results": results}, reason_sha)

    def _mark_closed(self, effective: int, now: int, reason_sha: str | None = None) -> None:
        if self._db.execute("UPDATE run SET closed=1 WHERE closed=0").rowcount:
            detail: dict[str, int | str] = {"effective_at": effective}
            if reason_sha is not None:
                detail["reason_sha"] = reason_sha
            self._event(now, "closed", detail=detail)

    def _advance(self, now: int) -> None:
        db = self._db
        freeze = self.freeze_at
        if now >= freeze and not db.execute("SELECT 1 FROM seals WHERE name='freeze'").fetchone():
            self._freeze(freeze, now)
        cutoff = freeze + self._test_window_s
        if now >= cutoff and not db.execute("SELECT 1 FROM seals WHERE name='finalize'").fetchone():
            self._finalize(cutoff, now)
        if now >= self.deadline:
            self._mark_closed(self.deadline, now)
        stale = db.execute("SELECT token,trial_id FROM attempts WHERE state IN ('reserved','live') "
                           "AND lease_until<=? ORDER BY slot", (now,)).fetchall()
        for row in stale:
            db.execute("UPDATE attempts SET state='reconciling' WHERE token=?", (row["token"],))
            self._event(now, "lease_expired", row["trial_id"], row["token"])

    def _trial(self, trial_id: str) -> sqlite3.Row:
        row = self._db.execute("SELECT * FROM trials WHERE id=?", (_digest(trial_id),)).fetchone()
        if row is None:
            raise PolicyError("unknown trial")
        return row

    def register(self, manifest: Manifest, *, priority: int = 100, max_attempts: int = 1,
                 requires: Iterable[str] = ()) -> str:
        """Register a dev trial or a dormant held-out template before the freeze."""
        if not isinstance(manifest, Manifest):
            raise PolicyError("expected a prevalidated manifest descriptor")
        _integer(priority, 0, 1_000_000)
        _integer(max_attempts, 1, 1 if manifest.split == Split.TEST else MAX_ATTEMPTS)
        parents = sorted({_digest(p) for p in requires})
        tid = manifest.key.trial_id
        fields = (manifest.sha256, _json(asdict(manifest.key)), manifest.split.value,
                  manifest.gpu_class, manifest.budget_s, priority, max_attempts)
        with self._transaction() as now:
            if now >= self.freeze_at:
                raise PolicyError("registration closed at the search cutoff")
            old = self._db.execute(
                "SELECT manifest_sha,key_json,split,gpu_class,budget_s,priority,max_attempts "
                "FROM trials WHERE id=?", (tid,)).fetchone()
            if old:
                deps = self._db.execute(
                    "SELECT parent FROM dependencies WHERE trial_id=? ORDER BY parent", (tid,))
                if tuple(old) != fields or parents != [p[0] for p in deps]:
                    raise PolicyError("conflicting descriptor or policy for the same seven-field key")
                return tid
            if self._db.execute("SELECT 1 FROM trials WHERE manifest_sha=?", (manifest.sha256,)).fetchone():
                raise PolicyError("manifest digest already bound to a different identity")
            dataset = manifest.key.dataset
            bound = self._db.execute("SELECT split FROM datasets WHERE digest=?", (dataset,)).fetchone()
            if bound and bound[0] != manifest.split.value:
                raise PolicyError("dataset digest cannot be used on both sides of the holdout")
            for parent in parents:
                if self._trial(parent)["split"] != Split.DEV:
                    raise PolicyError("held-out evidence cannot be a prerequisite")
            self._db.execute("INSERT OR IGNORE INTO datasets VALUES(?,?)", (dataset, manifest.split.value))
            self._db.execute("INSERT INTO trials(id,manifest_sha,key_json,split,gpu_class,budget_s,"
                             "priority,max_attempts,state) VALUES(?,?,?,?,?,?,?,?,?)",
                             (tid, *fields, "queued" if manifest.split == Split.DEV else "candidate"))
            self._db.executemany("INSERT INTO dependencies VALUES(?,?)", ((tid, p) for p in parents))
            self._event(now, "registered", tid, detail={"manifest_sha": manifest.sha256})
        return tid

    def attest(self, trial_id: str, gate: Gate, proof_sha: str) -> None:
        """Attach successful, immutable evidence for this exact manifest and GPU class."""
        _digest(proof_sha)
        if not isinstance(gate, Gate):
            raise PolicyError("expected a typed preflight gate")
        with self._transaction() as now:
            trial = self._trial(trial_id)
            cutoff = self.freeze_at if trial["split"] == Split.DEV else self.finalize_at
            if now >= cutoff:
                raise PolicyError("preflight window closed")
            previous = self._db.execute("SELECT proof FROM gates WHERE trial_id=? AND kind=?",
                                        (trial_id, gate.value)).fetchone()
            if previous:
                if previous[0] != proof_sha:
                    raise PolicyError("preflight proof is immutable")
                return
            self._db.execute("INSERT INTO gates VALUES(?,?,?)", (trial_id, gate.value, proof_sha))
            self._event(now, "gate_passed", trial_id, detail={"gate": gate.value, "proof": proof_sha})

    def nominate(self, picks: Iterable[Pick], *, decision_sha: str,
                 witnesses: Mapping[Pick, MethodWitness] | None = None,
                 payloads: Mapping[str, str | None] | None = None) -> None:
        """Replace the dev-only proposal; new controls require exact pair and payload witnesses."""
        _digest(decision_sha)
        entries = list(picks)
        if not all(isinstance(p, Pick) for p in entries):
            raise PolicyError("expected typed picks")
        with self._transaction() as now:
            if now >= self.freeze_at:
                raise PolicyError("selection is frozen")
            witnessed = witnesses is not None or payloads is not None or any(
                p.role in (Role.ZERO_EXTRA, Role.ORDER_ENSEMBLE) for p in entries)
            cohorts = set()
            plan = []
            keys = {}
            entries.sort(key=lambda x: (x.test_id, x.role.value, x.dev_id))
            for p in entries:
                dev, test = self._trial(p.dev_id), self._trial(p.test_id)
                if dev["split"] != Split.DEV or dev["state"] != "succeeded" or test["split"] != Split.TEST:
                    raise PolicyError("selection requires completed dev evidence and a test template")
                dk, tk = json.loads(dev["key_json"]), json.loads(test["key_json"])
                dev_split, test_split = dk.pop("dataset"), tk.pop("dataset")
                if dk != tk or dev_split == test_split:
                    raise PolicyError("only the split-specific dataset may change for frozen evaluation")
                cohort = (tk["model"], test_split, p.role)
                if cohort in cohorts:
                    raise PolicyError("more than one pick for the same model, held-out dataset and role")
                cohorts.add(cohort)
                if witnessed:
                    keys[p] = TrialKey(dataset=dev_split, **dk), TrialKey(dataset=test_split, **tk)
                plan.append({**asdict(p), "dev_result_sha": dev["result_sha"],
                             "dev_manifest_sha": dev["manifest_sha"], "test_manifest_sha": test["manifest_sha"]})
            proposal = {"decision_sha": decision_sha, "picks": plan}
            if witnessed:
                evidence, proposal["payloads"] = _bind_methods(keys, witnesses, payloads)
                for p, item in zip(entries, plan, strict=True):
                    item.update(evidence[p])
            encoded = _json(proposal)
            if self._db.execute("SELECT proposal FROM run").fetchone()[0] != encoded:
                self._db.execute("UPDATE run SET proposal=?", (encoded,))
                self._event(now, "nominated", detail=proposal)

    def _require_idle(self) -> None:
        if self._db.execute("SELECT 1 FROM attempts WHERE state!='closed' LIMIT 1").fetchone():
            raise PolicyError("a physical reservation is outstanding")

    def extend_once(self, *, hours: int, evidence_sha: str) -> dict:
        """Authorize one bounded pre-freeze shift. A retry of the same decision adds no time."""
        _integer(hours, 1, MAX_EXTENSION_HOURS)
        _digest(evidence_sha)
        with self._transaction() as now:
            if now >= self.freeze_at:
                raise PolicyError("cannot extend a frozen run")
            prior = self._extension()
            if prior is not None:
                if (prior["hours"], prior["evidence_sha"]) != (hours, evidence_sha):
                    raise PolicyError("the one timing extension has already been used")
            else:
                shift = hours * HOUR
                freeze = self._scheduled_freeze + shift
                deadline = _integer(self.nominal_deadline + shift, 0, _MAX_UTC)
                self._event(now, "timing_extended", detail={
                    "hours": hours, "evidence_sha": evidence_sha, "freeze_at": freeze,
                    "finalize_at": freeze + self._test_window_s, "deadline": deadline})
            return self._status(now)

    def freeze_early(self, *, reason_sha: str) -> None:
        """Seal the current dev-only plan now; move the two-hour held-out window earlier."""
        _digest(reason_sha)
        with self._transaction() as now:
            if now >= self.freeze_at:
                raise PolicyError("selection is frozen")
            self._require_idle()
            self._freeze(now, now, reason_sha)

    def finish_early(self, *, reason_sha: str) -> dict:
        """Close an idle queue at the witnessed UTC, never skipping a feasible frozen pick."""
        _digest(reason_sha)
        with self._transaction() as now:
            if self._phase(now) == "closed":
                return self._status(now)
            self._require_idle()
            if now < self.freeze_at:
                self._freeze(now, now, reason_sha)
            if now < self.finalize_at:
                available = min(self.finalize_at - now, self._gpu_budget()["remaining_s"])
                if self._db.execute("SELECT 1 FROM trials WHERE split='test' AND state='queued' "
                                    "AND budget_s<=? LIMIT 1", (available,)).fetchone():
                    raise PolicyError("frozen evaluations still fit; finish them before closing early")
                self._finalize(now, now, reason_sha)
            self._mark_closed(now, now, reason_sha)
            return self._status(now)

    def _gpu_budget(self) -> dict:
        # The two occupied slots hold their *entire* envelopes until an irreversible
        # receipt settles them. Never discount unknowns, expired leases or the send gap.
        cap, charged = self._db.execute("SELECT cap_s,charged_s FROM run_budget").fetchone()
        reserved = 0
        overrun = False
        for issued, stop, pulse in self._db.execute(
                "SELECT issued_at,hard_stop,pulse FROM attempts WHERE state!='closed'"):
            if pulse is not None and pulse > stop:
                overrun = True  # No trustworthy upper bound until fenced; withhold all headroom.
                stop = pulse + 1
            reserved += stop - issued
        if overrun:
            reserved = max(reserved, cap - charged)
        return {"cap_s": cap, "charged_s": charged, "reserved_s": reserved,
                "remaining_s": max(0, cap - charged - reserved)}

    def _as_ticket(self, row: sqlite3.Row) -> Admission:
        # A reservation without an override was issued against the pinned nominal target.
        deadline = json.loads(row["reservation"]).get("run_deadline", self.nominal_deadline)
        return Admission(self.run_id, row["token"], row["trial_id"], row["manifest_sha"],
                         Split(row["split"]), row["ordinal"], row["slot"], row["generation"],
                         row["gpu_class"], row["issued_at"], row["admit_before"], row["hard_stop"],
                         deadline)

    def reserve(self, *, trial_id: str | None = None) -> Admission | None:
        """Durably consume one admission before I/O; only its reserving caller may send it."""
        return self._reserve(target=_digest(trial_id) if trial_id is not None else None)

    def reserve_native_control(self, trial_id: str, proposal: bytes) -> Admission | None:
        """Acquire a slot to *measure* missing one-pass parity, never attest it in advance.

        This route is dev-only and checks the actual registered proposal bytes;
        a recurrent candidate cannot opt into the bootstrap gate by changing a label.
        All ordinary slot, budget, once-only and cutoff checks still apply.
        """
        _digest(trial_id)
        if not isinstance(proposal, bytes) or len(proposal) > 16384:
            raise PolicyError("invalid native control proposal")
        try:
            d = json.loads(proposal)
            settings = d["settings"]
            if d["branch"] == "loop":
                valid = (set(settings) == {"family", "loop", "window"} and settings["family"] == "additive"
                         and settings["loop"] == {"k": 1, "method": "naive"} and type(settings["loop"]["k"]) is int
                         and len(settings["window"]) == 2 and all(type(i) is int for i in settings["window"])
                         and 0 <= settings["window"][0] < settings["window"][1] < 24)
            else:
                valid = (d["branch"] == "compact" and set(settings) == {"family", "plan", "plan_sha256", "k"}
                         and settings["family"] == "compact" and settings["plan"] in ("select-20", "select-16", "merge-16")
                         and _digest(settings["plan_sha256"]) and type(settings["k"]) is int and settings["k"] == 1)
            valid = (valid and d["mode"] == "screen" and d["split"] == "discovery"
                     and proposal == (_json(d)+"\n").encode("ascii"))
        except (ValueError, TypeError, KeyError, IndexError):
            raise PolicyError("unrecognized native control") from None
        if not valid:
            raise PolicyError("not an exact one-pass discovery control")
        return self._reserve((trial_id, hashlib.sha256(proposal).hexdigest(), _hash(settings), "A100-"+d["branch"]))

    def reserve_refinement_control(self, trial_id: str, proposal: bytes) -> Admission | None:
        """Measure one-pass parity for a new physical development payload, not a finalist."""
        from .worker import REFINEMENT_RECIPES, method_sha256

        _digest(trial_id)
        if type(proposal) is not bytes or not 0 < len(proposal) <= 16384:
            raise PolicyError("invalid refinement control proposal")
        try:
            d = json.loads(proposal)
            k, m = TrialKey(**d["key"]), d["settings"]
            valid = (proposal == _json(d).encode("ascii") and d["schema"] == "joey-refinement-trial/v1"
                     and d["mode"] == "refinement" and d["run_id"] == self.run_id and d["branch"] == "compact"
                     and type(d["slot"]) is int and d["slot"] == 1 and k.trial_id == trial_id == d["trial_id"]
                     and k.dataset == d["inputs_sha256"] and k.config == d["method_sha256"] == method_sha256(m)
                     and k.model == m["compact_manifest_sha256"] and m["compact_recipe"] in REFINEMENT_RECIPES
                     and set(m) == {"window", "loop", "reverse", "compact_recipe", "compact_manifest_sha256"}
                     and m["window"] == [8, 12] and not m["reverse"]
                     and m["loop"] == {"method":"naive", "k":1} and type(m["loop"]["k"]) is int)
        except (TypeError, KeyError, ValueError, IndexError, RecursionError):
            raise PolicyError("invalid refinement control witness") from None
        if not valid:
            raise PolicyError("not an exact new-payload development control")
        return self._reserve((trial_id, hashlib.sha256(proposal).hexdigest(), k.config, "A100-compact"))

    def reserve_transfer_anchor(self, trial_id: str, proposal: bytes) -> Admission | None:
        """Measure the same payload's actual one-pass parity on its newly assigned A100."""
        from .native import MODEL_SHA256
        from .worker import method_sha256

        _digest(trial_id)
        if type(proposal) is not bytes or not 0 < len(proposal) <= 16384:
            raise PolicyError("invalid transfer control proposal")
        try:
            d = json.loads(proposal)
            k, m = TrialKey(**d["key"]), d["settings"]
            small = "compact_recipe" in m
            slot = 0 if small else 1
            fields = {"window", "loop", "reverse"} | ({"compact_recipe", "compact_manifest_sha256"} if small else set())
            valid = (d["schema"] == "joey-typed-transfer-trial/v1" and d["mode"] == "typed-transfer"
                     and d["run_id"] == self.run_id and d["payload"] == ("select-20" if small else "full")
                     and type(d["slot"]) is int and d["slot"] == slot and d["anchor_trial"] is None
                     and proposal == _json(d).encode("ascii") and set(m) == fields
                     and k.trial_id == trial_id == d["trial_id"] and k.config == d["method_sha256"] == method_sha256(m)
                     and k.dataset == d["inputs_sha256"] and m["window"] == [8, 12] and m["reverse"] is False
                     and m["loop"] == {"method": "naive", "k": 1} and type(m["loop"]["k"]) is int
                     and (k.model == m["compact_manifest_sha256"] and m["compact_recipe"] == "select-20"
                          if small else k.model == MODEL_SHA256["model.safetensors"]))
        except (TypeError, KeyError, ValueError, IndexError, AttributeError, RecursionError):
            raise PolicyError("invalid transfer control witness") from None
        if not valid:
            raise PolicyError("not a crossed-slot native development control")
        return self._reserve((trial_id, hashlib.sha256(proposal).hexdigest(), k.config, ("A100-loop", "A100-compact")[slot]))

    def _reserve(self, control: tuple[str, str, str, str] | None = None, target: str | None = None) -> Admission | None:
        with self._transaction() as now:
            if now >= self.finalize_at:
                return None
            split, cutoff = (Split.DEV, self.freeze_at) if now < self.freeze_at else (Split.TEST, self.finalize_at)
            if control and split != Split.DEV:
                raise PolicyError("native parity acquisition ends at the development freeze")
            gate = "AND (SELECT COUNT(*) FROM gates g WHERE g.trial_id=t.id)=2 "
            extra = ()
            if target is not None:
                gate += "AND t.id=? "
                extra = (target,)
            if control:
                member = self._trial(control[0])
                if (member["split"] != Split.DEV or member["manifest_sha"] != control[1] or
                        json.loads(member["key_json"])["config"] != control[2] or member["gpu_class"] != control[3]):
                    raise PolicyError("native control registration does not match")
                gate = ("AND t.id=? AND EXISTS(SELECT 1 FROM gates g WHERE g.trial_id=t.id AND g.kind='offline_staging') "
                        "AND NOT EXISTS(SELECT 1 FROM gates g WHERE g.trial_id=t.id AND g.kind='stock_parity') ")
                extra = (control[0],)
            available = min(cutoff - now, self._gpu_budget()["remaining_s"])
            row = self._db.execute(
                "SELECT t.*,s.slot FROM trials t JOIN slots s ON s.gpu_class=t.gpu_class "
                "WHERE t.split=? AND t.state='queued' AND t.budget_s<=? "
                "AND NOT EXISTS(SELECT 1 FROM attempts a WHERE a.slot=s.slot AND a.state!='closed') "
                + gate +
                "AND NOT EXISTS(SELECT 1 FROM dependencies d JOIN trials p ON p.id=d.parent "
                "               WHERE d.trial_id=t.id AND p.state!='succeeded') "
                "ORDER BY t.priority,t.id,s.slot LIMIT 1", (split.value, available, *extra)).fetchone()
            if row is None:
                return None
            slot, tid = row["slot"], row["id"]
            ordinal = self._db.execute("SELECT COUNT(*)+1 FROM attempts WHERE trial_id=?", (tid,)).fetchone()[0]
            if ordinal > row["max_attempts"]:
                raise PolicyError("attempt budget exhausted")
            generation = self._db.execute("SELECT COALESCE(MAX(generation),0)+1 FROM attempts "
                                          "WHERE slot=?", (slot,)).fetchone()[0]
            token = _hash([self.run_id, tid, ordinal, slot, generation])
            stop = now + row["budget_s"]
            admit_before = min(stop, now + self.lease_s)
            deadline = self.deadline
            self._db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,NULL,'reserved')",
                             (token, tid, split.value, ordinal, slot, generation, now,
                              admit_before, stop, admit_before))
            self._db.execute("UPDATE trials SET state='active' WHERE id=?", (tid,))
            detail = {"slot": slot, "generation": generation, "admit_before": admit_before, "hard_stop": stop}
            if control:
                detail["purpose"] = "measure-native-one-pass-parity"
            if deadline != self.nominal_deadline:
                detail["run_deadline"] = deadline
            self._event(now, "reserved", tid, token, detail)
            return Admission(self.run_id, token, tid, row["manifest_sha"], split, ordinal, slot,
                             generation, row["gpu_class"], now, admit_before, stop, deadline)

    def outstanding(self) -> list[Admission]:
        """Observation-only tickets: this API never authorizes sending/replaying admit."""
        with self._transaction():
            rows = self._db.execute("SELECT a.*,t.manifest_sha,t.gpu_class,e.detail AS reservation FROM attempts a "
                                    "JOIN trials t ON t.id=a.trial_id JOIN events e ON e.token=a.token "
                                    "AND e.kind='reserved' WHERE a.state!='closed' ORDER BY a.slot")
            return [self._as_ticket(r) for r in rows]

    def record(self, observation: Observation) -> None:
        """Record a worker observation. Only a terminal fence releases the physical slot."""
        if not isinstance(observation, Observation):
            raise PolicyError("expected a typed observation")
        o = observation
        with self._transaction() as now:
            row = self._db.execute(
                "SELECT a.*,t.max_attempts FROM attempts a JOIN trials t ON t.id=a.trial_id "
                "WHERE token=?", (o.token,)).fetchone()
            if row is None:
                raise PolicyError("observation for unknown token")
            if o.at is not None and not row["issued_at"] <= o.at <= now:
                raise PolicyError("observation timestamp is outside the observed attempt history")
            tid = row["trial_id"]
            terminal = o.outcome not in (Outcome.UNKNOWN, Outcome.LIVE)
            if row["state"] == "closed":
                if terminal:
                    prior = self._db.execute("SELECT observation FROM receipts WHERE token=?",
                                             (o.token,)).fetchone()[0]
                    if prior != _json(asdict(o)):
                        raise PolicyError("conflicting immutable terminal receipt")
                return  # A delayed pulse for a fenced token cannot touch a reused slot.
            if not terminal:
                if o.outcome == Outcome.UNKNOWN:
                    if row["state"] != "reconciling":
                        self._db.execute("UPDATE attempts SET state='reconciling' WHERE token=?", (o.token,))
                        self._event(now, "unknown", tid, o.token)
                    return
                if o.at is None:
                    raise PolicyError("missing worker pulse")
                if row["pulse"] is not None and o.at <= row["pulse"]:
                    return  # Polling the same heartbeat cannot renew a stale lease.
                lease = min(row["hard_stop"], o.at + self.lease_s)
                state = "live" if now < lease else "reconciling"
                self._db.execute("UPDATE attempts SET state=?,pulse=?,lease_until=? WHERE token=?",
                                 (state, o.at, lease, o.token))
                self._event(now, "heartbeat", tid, o.token, {"pulse": o.at, "lease_until": lease})
                return
            if o.at is None:
                raise PolicyError("missing terminal timestamp")
            if row["pulse"] is not None and (o.at < row["pulse"] or o.outcome == Outcome.FENCED_ABSENT):
                raise PolicyError("terminal receipt contradicts a recorded worker pulse")
            cutoff = self.freeze_at if row["split"] == Split.DEV else self.finalize_at
            accepted = o.outcome == Outcome.SUCCEEDED and o.at <= row["hard_stop"] and now < cutoff
            if accepted:
                state = "succeeded"
            elif now >= cutoff or o.outcome == Outcome.SUCCEEDED:
                state = "expired"
            elif (o.outcome in (Outcome.RETRYABLE, Outcome.FENCED_ABSENT) and
                  row["split"] == Split.DEV and row["ordinal"] < row["max_attempts"]):
                state = "queued"
            else:
                state = "failed"
            charged = _terminal_gpu_s(row["issued_at"], row["hard_stop"], o.outcome, o.at)
            self._db.execute("INSERT INTO receipts VALUES(?,?,?,?)", (o.token, _json(asdict(o)), now, accepted))
            self._db.execute("UPDATE run_budget SET charged_s=charged_s+?", (charged,))
            self._db.execute("UPDATE attempts SET state='closed' WHERE token=?", (o.token,))
            self._db.execute("UPDATE trials SET state=?,result_sha=? WHERE id=?",
                             (state, o.result_sha if accepted else None, tid))
            self._event(now, "terminal", tid, o.token, {"outcome": o.outcome.value, "fence_sha": o.fence_sha,
                                                       "result_sha": o.result_sha, "accepted": accepted,
                                                       "state": state, "charged_s": charged})

    def _poll(self, ticket: Admission, method: Callable[[Admission], Observation], *, admit: bool) -> None:
        with self._transaction() as now:
            if now >= (ticket.admit_before if admit else ticket.run_deadline):
                return
            if self._db.execute("SELECT state FROM attempts WHERE token=?", (ticket.token,)).fetchone()[0] == "closed":
                return
        try:
            observed = method(ticket)
        except (BackendUnavailable, TimeoutError, ConnectionError):
            observed = Observation(ticket.token, Outcome.UNKNOWN)
        if not isinstance(observed, Observation) or observed.token != ticket.token:
            raise PolicyError("backend response does not match request token")
        self.record(observed)

    def tick(self, backend: Backend) -> dict:
        """One watchdog cycle: observe inflight tokens first, then admit at most two new ones."""
        with self._transaction() as now:
            if self._db.execute("SELECT closed FROM run").fetchone()[0]:
                return self._status(now)
            self._db.execute("UPDATE run SET watchdog_at=?", (now,))
        for ticket in self.outstanding():
            self._poll(ticket, backend.observe, admit=False)
        for _ in range(2):
            ticket = self.reserve()
            if ticket is None:
                break
            self._poll(ticket, backend.admit, admit=True)
        return self.status()

    def run(self, backend: Backend, *, poll_s: int | None = None,
            sleep: Callable[[float], None] = time.sleep) -> dict:
        """Foreground driver; no subprocesses, daemon, network client or implicit launch."""
        poll_s = min(10, self.lease_s) if poll_s is None else _integer(poll_s, 1, self.lease_s)
        while True:
            status = self.tick(backend)
            now = status["as_of"]
            if status["phase"] == "closed":
                return status
            boundary = min(status[k] for k in ("freeze_at", "finalize_at", "deadline") if status[k] > now)
            sleep(min(poll_s, boundary - now))

    def _status(self, now: int) -> dict:
        slots = self._db.execute(
            "SELECT s.slot,s.gpu_class,a.token,a.trial_id,a.generation,a.state,a.lease_until,a.hard_stop "
            "FROM slots s LEFT JOIN attempts a ON a.slot=s.slot AND a.state!='closed' ORDER BY s.slot")
        return {"run_id": self.run_id, "start_utc": self.start_utc, "as_of": now, "phase": self._phase(now),
                "freeze_at": self.freeze_at, "finalize_at": self.finalize_at, "deadline": self.deadline,
                "nominal_deadline": self.nominal_deadline, "extension": self._extension(),
                "gpu_budget": self._gpu_budget(),
                "watchdog_at": self._db.execute("SELECT watchdog_at FROM run").fetchone()[0],
                "slots": [dict(r) for r in slots],
                "counts": [dict(r) for r in self._db.execute("SELECT * FROM status_counts ORDER BY split,state")],
                "seals": {r[0]: r[1] for r in self._db.execute("SELECT name,sha256 FROM seals")},
                "event_seq": self._db.execute("SELECT MAX(seq) FROM events").fetchone()[0]}

    def status(self) -> dict:
        with self._transaction() as now:
            return self._status(now)

    def _registry(self, status: dict) -> list[dict]:
        deps = {}
        for d in self._db.execute("SELECT d.trial_id,d.parent,p.state FROM dependencies d "
                                  "JOIN trials p ON p.id=d.parent ORDER BY d.trial_id,d.parent"):
            deps.setdefault(d["trial_id"], []).append((d["parent"], d["state"]))
        rows = []
        for row in self._db.execute("SELECT * FROM trial_registry ORDER BY priority,id"):
            item = dict(row)
            item["key"] = json.loads(item.pop("key_json"))
            item["requires"] = [tid for tid, _ in deps.get(item["id"], ())]
            item["unmet_requires"] = [tid for tid, state in deps.get(item["id"], ()) if state != "succeeded"]
            blocked = []
            if item["state"] == "candidate":
                blocked.append("freeze")
            if item["state"] == "queued":
                if not item["offline_sha"]:
                    blocked.append(Gate.OFFLINE_STAGING.value)
                if not item["parity_sha"]:
                    blocked.append(Gate.STOCK_PARITY.value)
                if item["unmet_requires"]:
                    blocked.append("prerequisite")
                slots = [s for s in status["slots"] if s["gpu_class"] == item["gpu_class"]]
                if not slots:
                    blocked.append("gpu_class")
                elif all(s["token"] is not None for s in slots):
                    blocked.append("slot_busy")
                cutoff = status["freeze_at"] if item["split"] == Split.DEV else status["finalize_at"]
                if item["budget_s"] > cutoff - status["as_of"]:
                    blocked.append("time_budget")
                if item["budget_s"] > status["gpu_budget"]["remaining_s"]:
                    blocked.append("gpu_budget")
            item["waiting_for"] = blocked
            rows.append(item)
        return rows

    def registry(self) -> list[dict]:
        with self._transaction() as now:
            return self._registry(self._status(now))

    def snapshot(self) -> dict:
        """Consistent, JSON-serializable registry/status projection at one event cursor."""
        with self._transaction() as now:
            status = self._status(now)
            return {"status": status, "registry": self._registry(status)}

    def history(self, trial_id: str) -> list[dict]:
        """All slot generations and immutable physical receipts for one seven-field key."""
        with self._transaction():
            self._trial(trial_id)
            rows = self._db.execute(
                "SELECT a.*,r.recorded_at,r.accepted,r.observation FROM attempts a "
                "LEFT JOIN receipts r ON r.token=a.token WHERE a.trial_id=? ORDER BY ordinal", (trial_id,))
            return [{**dict(r), "observation": json.loads(r["observation"]) if r["observation"] else None}
                    for r in rows]

    def events(self, after: int = 0, limit: int = 1000) -> list[dict]:
        _integer(after, 0, (1 << 63) - 1)
        _integer(limit, 1, 10_000)
        with self._transaction():
            rows = self._db.execute("SELECT * FROM events WHERE seq>? ORDER BY seq LIMIT ?", (after, limit))
            return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]

    def seal(self, name: str) -> dict | None:
        if name not in ("freeze", "finalize"):
            raise PolicyError("unknown seal")
        with self._transaction():
            r = self._db.execute("SELECT * FROM seals WHERE name=?", (name,)).fetchone()
            return {**dict(r), "payload": json.loads(r["payload"])} if r is not None else None
