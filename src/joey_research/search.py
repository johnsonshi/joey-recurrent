"""Deterministic paired-dev screening. No inference, label reader, I/O or admissions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Iterable

from .scheduler import PolicyError, TrialKey, _digest, _hash, _integer


PPM = 1_000_000
ACCURACY_MARGIN = 20_000  # 0.02 absolute accuracy
BRIER_MARGIN = 10_000  # 0.01 mean summed multiclass Brier score
BRANCHES = ("loop", "compact")
COST_AXES = ("latency_us", "layer_millicalls")


def _dev(split: str) -> None:
    if split != "dev":
        raise PolicyError("selection accepts validated development evidence only")


@dataclass(frozen=True)
class Arm:
    id: str  # Digest of the immutable configuration, independent of rung membership.
    kind: str
    payload_sha: str
    k: int

    def __post_init__(self) -> None:
        _digest(self.id)
        _digest(self.payload_sha)
        _integer(self.k, 1, 1 << 31)
        if self.kind not in (*BRANCHES, "stock") or (self.kind == "stock" and self.k != 1):
            raise PolicyError("invalid arm kind or stock recurrence")


@dataclass(frozen=True)
class Candidate:
    arm: Arm
    stock: Arm
    recovery: Arm | None = None
    priority: int = 100  # Used for initial work ordering, never as a quality estimate.

    def __post_init__(self) -> None:
        _integer(self.priority, 0, 1_000_000)
        if not isinstance(self.arm, Arm) or not isinstance(self.stock, Arm):
            raise PolicyError("expected typed candidate and reference arms")
        if self.arm.kind not in BRANCHES or self.stock.kind != "stock" or self.arm.id == self.stock.id:
            raise PolicyError("a competing arm requires a distinct predeclared stock reference")
        if self.arm.kind == "compact":
            r = self.recovery
            if (self.arm.k <= 1 or not isinstance(r, Arm) or r.kind != "compact" or r.k != 1
                    or r.payload_sha != self.arm.payload_sha or r.id in (self.arm.id, self.stock.id)):
                raise PolicyError("compact recovery requires a separate k=1 control on the same payload")
        elif self.recovery is not None:
            raise PolicyError("only a compact candidate has a compact recovery control")

    @property
    def controls(self) -> tuple[Arm, ...]:
        return (self.stock, self.recovery) if self.recovery is not None else (self.stock,)


@dataclass(frozen=True)
class Rung:
    ids: tuple[str, ...]  # Predeclared paired dev members, no labels.
    keep_per_branch: int

    def __post_init__(self) -> None:
        _integer(self.keep_per_branch, 1, 1_000_000)
        ids = tuple(sorted(_digest(i) for i in self.ids))
        if not ids or len(set(ids)) != len(ids):
            raise PolicyError("rung membership must be nonempty and unique")
        object.__setattr__(self, "ids", ids)

    @property
    def membership_sha(self) -> str:
        return _hash(self.ids)


@dataclass(frozen=True)
class Plan:
    study_sha: str  # Binds model, scorer, device/protocol, seed and allowed dev partition.
    method_sha: str  # Preregistered paired interval method, level and sequential-look policy.
    candidates: tuple[Candidate, ...]
    rungs: tuple[Rung, ...]
    confidence_ppm: int
    match_cost_axis: str
    max_cost_gap_ppm: int  # Relative to positive measured stock cost.

    def __post_init__(self) -> None:
        _digest(self.study_sha)
        _digest(self.method_sha)
        _integer(self.confidence_ppm, 1, PPM)
        _integer(self.max_cost_gap_ppm, 0, PPM)
        if self.match_cost_axis not in COST_AXES:
            raise PolicyError("declare a comparable measured cost axis")
        if not all(isinstance(c, Candidate) for c in self.candidates) or not all(
                isinstance(r, Rung) for r in self.rungs):
            raise PolicyError("expected preregistered candidates and rungs")
        candidates = tuple(sorted(self.candidates, key=lambda c: c.arm.id))
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "rungs", tuple(self.rungs))
        if {c.arm.kind for c in candidates} != set(BRANCHES):
            raise PolicyError("reserve one initial slot for loop and one for compact")
        arms = [a for c in candidates for a in (c.arm, *c.controls)]
        if len(set(arms)) != len({a.id for a in arms}):
            raise PolicyError("conflicting arm identity")
        ids = {c.arm.id for c in candidates}
        if len(ids) != len(candidates) or ids & {a.id for c in candidates for a in c.controls}:
            raise PolicyError("candidates and paired controls must be distinct")
        if not self.rungs or self.rungs[-1].keep_per_branch > 2:
            raise PolicyError("predeclare at most two finalists per branch (controls excluded)")
        for previous, current in zip(self.rungs, self.rungs[1:]):
            if (not set(previous.ids) < set(current.ids)
                    or current.keep_per_branch > previous.keep_per_branch):
                raise PolicyError("successive rungs must expand paired members and never expand survivor caps")

    @property
    def sha256(self) -> str:
        return _hash({"version": 1, "plan": asdict(self), "accuracy_margin": ACCURACY_MARGIN,
                      "brier_margin": BRIER_MARGIN})


@dataclass(frozen=True)
class MeasuredCost:
    latency_us: int  # Same device, batch, workload and summary convention for the whole study.
    layer_millicalls: int
    peak_bytes: int
    payload_bytes: int
    proof_sha: str

    def __post_init__(self) -> None:
        _digest(self.proof_sha)
        for n in (self.latency_us, self.layer_millicalls, self.peak_bytes, self.payload_bytes):
            _integer(n, 1, (1 << 63) - 1)


@dataclass(frozen=True)
class Item:
    id: str
    correct: bool
    confidence_ppm: int
    brier_ppm: int  # Per-item SUM over classes, not class-averaged.

    def __post_init__(self) -> None:
        _digest(self.id)
        if type(self.correct) is not bool:
            raise PolicyError("correctness must be a validated boolean")
        _integer(self.confidence_ppm, 0, PPM)
        _integer(self.brier_ppm, 0, 2 * PPM)


@dataclass(frozen=True)
class DevRun:
    split: str
    study_sha: str
    arm: Arm
    dev_id: str  # Scheduler's dev trial ID on this rung.
    result_sha: str
    validation_sha: str
    cost: MeasuredCost
    items: tuple[Item, ...]

    def __post_init__(self) -> None:
        _dev(self.split)
        for d in (self.study_sha, self.dev_id, self.result_sha, self.validation_sha):
            _digest(d)
        if not isinstance(self.cost, MeasuredCost):
            raise PolicyError("missing explicit cost evidence")
        if not isinstance(self.arm, Arm) or not all(isinstance(i, Item) for i in self.items):
            raise PolicyError("expected a typed arm and scored members")
        items = tuple(sorted(self.items, key=lambda x: x.id))
        if not items or len({i.id for i in items}) != len(items):
            raise PolicyError("empty or duplicate dev items")
        object.__setattr__(self, "items", items)

    @classmethod
    def from_registry(cls, row: dict, *, study_sha: str, arm: Arm, validation_sha: str,
                      cost: MeasuredCost, items: tuple[Item, ...]) -> DevRun:
        """Import a successful dev row only; a physical receipt alone is not success."""
        _dev(row.get("split"))
        if row.get("state") != "succeeded":
            raise PolicyError("queue has no accepted result for this dev trial")
        try:
            key = TrialKey(**row["key"])
        except (KeyError, TypeError) as exc:
            raise PolicyError("missing valid trial identity in queue registry") from exc
        if key.config != arm.id or key.trial_id != row.get("id"):
            raise PolicyError("queue identity does not match this arm")
        return cls("dev", study_sha, arm, row["id"], row["result_sha"], validation_sha, cost, items)


@dataclass(frozen=True)
class Band:
    """Paired interval for candidate minus reference, in millionths of score units."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        _integer(self.lo, -2 * PPM, 2 * PPM)
        _integer(self.hi, self.lo, 2 * PPM)


@dataclass(frozen=True)
class Paired:
    split: str
    study_sha: str
    method_sha: str
    membership_sha: str
    candidate_result: str
    reference_result: str
    proof_sha: str
    accuracy: Band
    brier: Band
    joint_error: Band
    conditional_error: Band | None
    coverage: Band

    def __post_init__(self) -> None:
        _dev(self.split)
        for d in (self.study_sha, self.method_sha, self.membership_sha, self.candidate_result,
                  self.reference_result, self.proof_sha):
            _digest(d)
        if self.candidate_result == self.reference_result:
            raise PolicyError("self-comparison is not paired evidence")
        for name in ("accuracy", "brier", "joint_error", "conditional_error", "coverage"):
            band = getattr(self, name)
            if name == "conditional_error" and band is None:
                continue  # Undefined or insufficient high-confidence support is not zero risk.
            if not isinstance(band, Band):
                raise PolicyError("missing paired interval")
            if name != "brier" and (band.lo < -PPM or band.hi > PPM):
                raise PolicyError("rate-difference interval outside its support")


@dataclass(frozen=True)
class Rates:
    n: int
    correct: int
    brier_sum_ppm: int
    high: int
    high_wrong: int

    @classmethod
    def from_run(cls, run: DevRun, cut: int) -> Rates:
        high = [i for i in run.items if i.confidence_ppm >= cut]
        return cls(len(run.items), sum(i.correct for i in run.items), sum(i.brier_ppm for i in run.items),
                   len(high), sum(not i.correct for i in high))

    @property
    def accuracy(self) -> Fraction:
        return Fraction(self.correct, self.n)

    @property
    def brier(self) -> Fraction:
        return Fraction(self.brier_sum_ppm, self.n * PPM)

    @property
    def joint_error(self) -> Fraction:
        return Fraction(self.high_wrong, self.n)

    @property
    def conditional_error(self) -> Fraction | None:
        return Fraction(self.high_wrong, self.high) if self.high else None

    @property
    def coverage(self) -> Fraction:
        return Fraction(self.high, self.n)

    def report(self) -> dict:
        def ratio(value: Fraction | None):
            return None if value is None else {"numerator": value.numerator, "denominator": value.denominator}
        return {**asdict(self), "accuracy": ratio(self.accuracy), "summed_brier": ratio(self.brier),
                "joint_high_confidence_error": ratio(self.joint_error),
                "conditional_high_confidence_error": ratio(self.conditional_error),
                "high_confidence_coverage": ratio(self.coverage)}


def _check_bands(pair: Paired, a: Rates, b: Rates) -> None:
    for name in ("accuracy", "brier", "joint_error", "conditional_error", "coverage"):
        v, w, band = getattr(a, name), getattr(b, name), getattr(pair, name)
        if v is None or w is None:
            if band is not None:
                raise PolicyError("numeric conditional risk for a zero-coverage run")
        elif band is not None and not band.lo <= (v - w) * PPM <= band.hi:
            raise PolicyError(f"paired {name} interval does not contain the supplied-data estimate")


@dataclass(frozen=True)
class Assessment:
    arm: Arm
    dev_id: str
    rates: Rates
    cost: MeasuredCost
    stock_pair: Paired
    recovery_pair: Paired | None
    compact_noninferiority: str | None
    recovery_gain: bool | None
    eliminate: tuple[str, ...]
    review: tuple[str, ...]

    @property
    def clear(self) -> bool:
        return not self.eliminate and not self.review


def _safety(role: str, pair: Paired, a: Rates, b: Rates, reject: list, review: list) -> None:
    for name in ("joint_error", "conditional_error"):
        band = getattr(pair, name)
        v, w = getattr(a, name), getattr(b, name)
        if band is None:
            review.append(f"{role}.{name}.undefined_interval")
        elif band.lo > 0:
            reject.append(f"{role}.{name}.regression")
        elif v > w:
            review.append(f"{role}.{name}.observed_increase")
        elif band.hi > 0:
            review.append(f"{role}.{name}.cannot_exclude_increase")


def _assess(plan: Plan, candidate: Candidate, runs: dict, rates: dict, pairs: dict) -> Assessment:
    a, s = runs[candidate.arm.id], runs[candidate.stock.id]
    ar, sr = rates[a.arm.id], rates[s.arm.id]
    stock = pairs[(a.result_sha, s.result_sha)]
    reject, review = [], []
    _check_bands(stock, ar, sr)
    _safety("stock", stock, ar, sr, reject, review)
    cost, stock_cost = getattr(a.cost, plan.match_cost_axis), getattr(s.cost, plan.match_cost_axis)
    if abs(cost - stock_cost) * PPM > stock_cost * plan.max_cost_gap_ppm:
        reject.append("stock.measured_cost_unmatched")
    ni, recovery, gain = None, None, None
    if candidate.recovery is not None:
        r = runs[candidate.recovery.id]
        if a.cost.payload_bytes != r.cost.payload_bytes:
            raise PolicyError("same compact payload has inconsistent measured bytes")
        if a.cost.payload_bytes >= s.cost.payload_bytes:
            reject.append("compact.no_measured_payload_reduction")
        if stock.accuracy.lo > -ACCURACY_MARGIN and stock.brier.hi < BRIER_MARGIN:
            ni = "within_both_margins"
        elif stock.accuracy.hi < -ACCURACY_MARGIN or stock.brier.lo > BRIER_MARGIN:
            ni = "outside_margin"
            reject.append("compact.loss_exceeds_margin")
        else:
            ni = "inconclusive"
            review.append("compact.noninferiority_not_established")
        rr = rates[r.arm.id]
        recovery = pairs[(a.result_sha, r.result_sha)]
        _check_bands(recovery, ar, rr)
        _safety("k1", recovery, ar, rr, reject, review)
        if recovery.accuracy.hi < 0 or recovery.brier.lo > 0:
            reject.append("k1.quality_regression")
        # A joint, conservative recovery signal; compression noninferiority alone is not recovery.
        gain = (recovery.accuracy.lo >= 0 and recovery.brier.hi <= 0
                and (recovery.accuracy.lo > 0 or recovery.brier.hi < 0))
    elif stock.accuracy.hi < 0 or stock.brier.lo > 0:
        reject.append("stock.quality_regression")
    return Assessment(a.arm, a.dev_id, ar, a.cost, stock, recovery, ni, gain,
                      tuple(reject), tuple(review))


def _vector(a: Assessment) -> tuple:
    # Descriptive Pareto order, not a statistical claim of pairwise superiority.
    r, c = a.rates, a.cost
    return (-r.accuracy, r.brier, r.joint_error, r.conditional_error,
            c.latency_us, c.layer_millicalls, c.peak_bytes, c.payload_bytes)


def _dominates(a: Assessment, b: Assessment) -> bool:
    v, w = _vector(a), _vector(b)
    if None in v or None in w:
        return False  # Do not impute an error rate for zero coverage.
    return all(x <= y for x, y in zip(v, w)) and any(x < y for x, y in zip(v, w))


@dataclass(frozen=True)
class Control:
    arm: Arm
    dev_id: str
    result_sha: str
    rates: Rates
    cost: MeasuredCost


@dataclass(frozen=True)
class Decision:
    rung: int
    final: bool
    decision_sha: str
    assessments: tuple[Assessment, ...]
    controls: tuple[Control, ...]
    frontier: tuple[str, ...]
    kept: tuple[str, ...]  # Arm IDs; only final kept arms are nominated.
    control_dev_ids: tuple[str, ...]
    nominee_dev_ids: tuple[str, ...]

    def report(self) -> dict:
        """JSON-ready evidence, including explicit error denominators and coverage."""
        data = asdict(self)
        for key in ("assessments", "controls"):
            for row, obj in zip(data[key], getattr(self, key)):
                row["rates"] = obj.rates.report()
        return data


class Search:
    """Replay the same preregistered plan and rungs for the same decisions."""

    def __init__(self, plan: Plan):
        self._plan = plan
        self._catalog = {c.arm.id: c for c in plan.candidates}
        self._active = tuple(sorted(self._catalog))
        self._round = 0
        self._previous_items: dict[str, dict[str, Item]] = {}
        self._proof = plan.sha256

    @property
    def plan(self) -> Plan:
        return self._plan

    def branch_wave(self, completed: Iterable[str] = ()) -> tuple[str | None, str | None]:
        """Propose at most one remaining candidate on each logical lane, loop then compact."""
        done = set(completed)
        if not done <= set(self._active):
            raise PolicyError("unexpected completed arm on this rung")
        if self._round >= len(self.plan.rungs):
            return None, None
        def head(branch: str) -> str | None:
            pending = [self._catalog[i] for i in self._active if i not in done
                       and self._catalog[i].arm.kind == branch]
            return min(pending, key=lambda c: (c.priority, c.arm.id)).arm.id if pending else None
        return head("loop"), head("compact")

    def initial_slots(self) -> tuple[str, str]:
        if self._round != 0:
            raise PolicyError("initial slots apply only to the opening rung")
        loop, compact = self.branch_wave()
        if loop is None or compact is None:
            raise PolicyError("both initial branches are required")
        return loop, compact

    @property
    def active(self) -> tuple[str, ...]:
        return self._active

    def required_controls(self) -> tuple[str, ...]:
        return tuple(sorted({a.id for i in self._active for a in self._catalog[i].controls}))

    def advance(self, runs: Iterable[DevRun], paired: Iterable[Paired]) -> Decision:
        if self._round >= len(self.plan.rungs) or not self._active:
            raise PolicyError("screening has ended; start a different study rather than adding candidates")
        rung = self.plan.rungs[self._round]
        final = self._round == len(self.plan.rungs) - 1
        expected = {a.id: a for i in self.active for a in
                    (self._catalog[i].arm, *self._catalog[i].controls)}
        cache = {}
        for run in runs:
            if not isinstance(run, DevRun):
                raise PolicyError("unvalidated run input")
            _dev(run.split)
            if (run.study_sha != self.plan.study_sha or expected.get(run.arm.id) != run.arm
                    or run.arm.id in cache or tuple(i.id for i in run.items) != rung.ids):
                raise PolicyError("misbound, duplicate, or unpaired run")
            old = self._previous_items.get(run.arm.id, {})
            if any(i.id in old and old[i.id] != i for i in run.items):
                raise PolicyError("overlapping validated results changed between nested rungs")
            cache[run.arm.id] = run
        if (cache.keys() != expected.keys() or len({r.dev_id for r in cache.values()}) != len(cache)
                or len({r.result_sha for r in cache.values()}) != len(cache)):
            raise PolicyError("missing required runs or aliased trial/result identities")
        required = {(cache[i].result_sha, cache[a.id].result_sha)
                    for i in self.active for a in self._catalog[i].controls}
        bounds = {}
        membership = rung.membership_sha
        for p in paired:
            if not isinstance(p, Paired):
                raise PolicyError("unvalidated paired input")
            _dev(p.split)
            key = (p.candidate_result, p.reference_result)
            if (p.study_sha != self.plan.study_sha or p.method_sha != self.plan.method_sha
                    or p.membership_sha != membership or key not in required or key in bounds):
                raise PolicyError("wrong method, members, orientation, or duplicate paired evidence")
            bounds[key] = p
        if bounds.keys() != required:
            raise PolicyError("missing paired uncertainty for a required control")
        rates = {i: Rates.from_run(run, self.plan.confidence_ppm) for i, run in cache.items()}
        records = tuple(_assess(self.plan, self._catalog[i], cache, rates, bounds) for i in self.active)
        kept, frontier = [], []
        for branch in BRANCHES:
            promotable = [r for r in records if r.arm.kind == branch and not r.eliminate
                          and (not final or r.clear)]
            # Never let an uncertainty-flagged arm displace a clear arm by Pareto dominance.
            for clear in (True, False):
                tier = [r for r in promotable if r.clear == clear]
                front = [r for r in tier if not any(_dominates(o, r) for o in tier)]
                front.sort(key=lambda r: (-r.stock_pair.accuracy.lo, r.stock_pair.brier.hi,
                                          r.stock_pair.joint_error.hi, r.cost.latency_us, r.arm.id))
                frontier.extend(r.arm.id for r in front)
                room = rung.keep_per_branch - sum(self._catalog[i].arm.kind == branch for i in kept)
                kept.extend(r.arm.id for r in front[:room])
        control_ids = tuple(sorted({cache[a.id].dev_id for i in kept for a in self._catalog[i].controls}))
        nominees = tuple(cache[i].dev_id for i in kept) if final else ()
        controls = tuple(Control(cache[i].arm, cache[i].dev_id, cache[i].result_sha, rates[i], cache[i].cost)
                         for i in self.required_controls())
        proof = _hash({"previous": self._proof, "rung": self._round, "members": membership,
                       "runs": [(i, cache[i].dev_id, cache[i].result_sha, cache[i].validation_sha,
                                 asdict(cache[i].cost), _hash([asdict(x) for x in cache[i].items]),
                                 asdict(rates[i])) for i in sorted(cache)],
                       "pairs": [asdict(bounds[k]) for k in sorted(bounds)],
                       "frontier": frontier, "kept": kept, "controls": control_ids, "nominees": nominees})
        decision = Decision(self._round, final or not kept, proof, records, controls, tuple(frontier),
                            tuple(kept), control_ids, nominees)
        self._round += 1
        self._active = tuple(sorted(kept))
        self._previous_items = {i: {x.id: x for x in cache[i].items}
                                for i in (*self._active, *self.required_controls())} if not final else {}
        self._proof = proof
        return decision
