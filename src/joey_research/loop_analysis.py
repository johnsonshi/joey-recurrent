"""Label-free decisions from completed repeat passes over the same native slots."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Generic, Literal

from .loop import H, O, S, LoopCallbacks, LoopCounts, LoopPlan, LoopResult, Readout, run_loop

Slot = int | str
Vector = tuple[float, ...]
Pool = Literal["first", "final", "probability_mean", "normalized_logit_mean", "first_anchor"]
Fallback = Literal["first", "aggregate"]


def _unit(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


@dataclass(frozen=True)
class SlotOrder:
    split: str
    input_key: str
    slots: tuple[Slot, ...]

    def __post_init__(self) -> None:
        if not all(isinstance(s, str) and s.strip() for s in (self.split, self.input_key)):
            raise ValueError("split and input_key must be nonempty")
        if not isinstance(self.slots, (list, tuple)):
            raise ValueError("supply the ordered native slot keys, not a mapping or set")
        slots = tuple(self.slots)
        if not all(type(s) in (int, str) and s != "" for s in slots):
            raise ValueError("slot keys must be integers or nonempty strings")
        if len(slots) < 2 or len(set(slots)) != len(slots):
            raise ValueError("need at least two distinct, unambiguous slots")
        object.__setattr__(self, "slots", slots)


@dataclass(frozen=True)
class _Scores:
    order: SlotOrder
    logits: Vector


class _Bound(Generic[H, S, O]):
    def __init__(self, question: LoopCallbacks[H, S, O], native_order: Callable[[], SlotOrder]):
        self.question, self.native_order = question, native_order
        self.order = native_order()
        if not isinstance(self.order, SlotOrder):
            raise ValueError("native_order must provide SlotOrder metadata")
        self.prelude, self.block = question.prelude, question.block
        self.inference_context, self.fresh_state = question.inference_context, question.fresh_state

    def _capture(self, raw: O) -> _Scores:
        if self.native_order() != self.order:
            raise ValueError("input identity or native slot order changed during the run")
        values = raw.tolist() if hasattr(raw, "tolist") else raw
        if (
            not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray))
            or len(values) != len(self.order.slots)
            or not all(type(x) in (float, int) and math.isfinite(x) for x in values)
        ):
            raise ValueError("expected one finite logit per ordered native slot")
        return _Scores(self.order, tuple(map(float, values)))

    def stock(self) -> _Scores:
        return self._capture(self.question.stock())

    def coda(self, hidden: H, state: S) -> _Scores:
        return self._capture(self.question.coda(hidden, state))


@dataclass(frozen=True)
class Stop:
    kind: Literal["margin", "entropy", "stability"]
    threshold: float | int

    def __post_init__(self) -> None:
        if self.kind == "stability":
            if type(self.threshold) is not int or self.threshold < 2:
                raise ValueError("stability requires at least two consecutive unique winners")
        elif self.kind not in ("margin", "entropy") or not _unit(self.threshold):
            raise ValueError("margin and normalized entropy thresholds must lie in [0, 1]")


@dataclass(frozen=True)
class Policy:
    pool: Pool = "final"
    stop: Stop | None = None
    fallback: Fallback = "aggregate"
    first_weight: float | None = None

    def __post_init__(self) -> None:
        if self.pool not in (
            "first", "final", "probability_mean", "normalized_logit_mean", "first_anchor"
        ):
            raise ValueError("unknown pooling rule")
        if self.pool == "first_anchor" and not _unit(self.first_weight):
            raise ValueError("first_anchor requires first_weight in [0, 1]")
        if self.pool != "first_anchor" and self.first_weight is not None:
            raise ValueError("first_weight is only used by first_anchor")
        if self.stop is not None and not isinstance(self.stop, Stop):
            raise ValueError("stop must be a Stop rule")
        if self.fallback not in ("aggregate", "first") or (
            self.stop is None and self.fallback != "aggregate"
        ):
            raise ValueError("fallback must be a supported action for an active stop rule")


def _repeat_only(plan: LoopPlan) -> None:
    if plan.method not in ("naive", "damped_extra"):
        raise ValueError("analysis requires completed naive or damped_extra passes, not RK stages")


def _counts(k: int, blocks: int, codas: int) -> LoopCounts:
    return LoopCounts(1, 0, 0, 0) if k == 1 else LoopCounts(0, 1, blocks, codas)


@dataclass(frozen=True)
class PassTrace:
    plan: LoopPlan
    result: LoopResult[_Scores]

    def __post_init__(self) -> None:
        _repeat_only(self.plan)
        k, r = self.plan.k, self.result
        if (
            self.plan.readout != "diagnostic" or r.stopped_early or len(r.readouts) != k
            or r.counts != _counts(k, k, k)
        ):
            raise ValueError("replay requires a full trace with one observed coda readout per pass")
        for i, probe in enumerate(r.readouts, 1):
            expected = "stock" if k == 1 else ("first_pass" if i == 1 else "iterate")
            if probe.block_evaluations != i or probe.kind != expected:
                raise ValueError("readouts must be consecutive, completed repeat passes")
            if not isinstance(probe.logits, _Scores) or probe.logits.order != self.order:
                raise ValueError("readout has no matching input and native slot binding")
        if r.logits != r.readouts[-1].logits:
            raise ValueError("returned logits differ from the final pass readout")

    @property
    def order(self) -> SlotOrder:
        return self.result.readouts[0].logits.order


def record(
    plan: LoopPlan, question: LoopCallbacks[H, S, O], native_order: Callable[[], SlotOrder],
) -> PassTrace:
    """Run all requested passes and coda probes; snapshot metadata as each score is emitted."""
    _repeat_only(plan)
    diagnostic = replace(plan, readout="diagnostic")
    return PassTrace(diagnostic, run_loop(diagnostic, _Bound(question, native_order)))


def _softmax(values: Vector) -> Vector:
    peak = max(values)
    weights = tuple(math.exp(v - peak) for v in values)
    total = math.fsum(weights)
    return tuple(v / total for v in weights)


def _standardize(values: Vector) -> Vector:
    scale = max(map(abs, values))
    scaled = tuple(v / scale for v in values) if scale else values
    center = math.fsum(scaled) / len(scaled)
    delta = tuple(v - center for v in scaled)
    rms = math.sqrt(math.fsum(v * v for v in delta) / len(delta))
    return tuple(v / rms for v in delta) if rms else (0.0,) * len(values)


def _top(values: Vector) -> int:
    return values.index(max(values))


def _pool(rows: Sequence[_Scores], rule: Pool, weight: float | None) -> tuple[Vector, int]:
    if rule in ("first", "final") or (rule == "first_anchor" and weight in (0, 1)):
        logits = rows[0 if rule == "first" or weight == 1 else -1].logits
        return _softmax(logits), _top(logits)
    if rule == "first_anchor":
        assert weight is not None
        first, last = _softmax(rows[0].logits), _softmax(rows[-1].logits)
        p = tuple(weight * a + (1 - weight) * b for a, b in zip(first, last, strict=True))
    else:
        transform = _softmax if rule == "probability_mean" else _standardize
        columns = zip(*(transform(r.logits) for r in rows), strict=True)
        mean = tuple(math.fsum(column) / len(rows) for column in columns)
        p = mean if rule == "probability_mean" else _softmax(mean)
    return p, _top(p)


def _trigger(rows: Sequence[_Scores], stop: Stop) -> bool:
    logits = rows[-1].logits
    top = _top(logits)
    if logits.count(logits[top]) != 1:
        return False
    if stop.kind == "stability":
        n = int(stop.threshold)
        return len(rows) >= n and all(
            _top(r.logits) == top and r.logits.count(r.logits[top]) == 1 for r in rows[-n:]
        )
    p = _softmax(logits)
    if stop.kind == "margin":
        return p[top] - max(v for i, v in enumerate(p) if i != top) >= stop.threshold
    entropy = -math.fsum(v * math.log(v) for v in p if v) / math.log(len(p))
    return entropy <= stop.threshold


def _reason(rows: Sequence[_Scores], policy: Policy, k: int) -> str | None:
    if policy.stop is not None:
        if _trigger(rows, policy.stop):
            return policy.stop.kind
        return "no_trigger" if len(rows) == k else None
    if policy.pool == "first":
        return "first"
    return "budget" if len(rows) == k else None


@dataclass(frozen=True)
class Decision:
    order: SlotOrder
    probabilities: Vector
    slot_index: int
    decision_pass: int
    reason: str
    fallback_used: Fallback | None
    executed: LoopCounts
    recorded: LoopCounts | None
    simulated: LoopCounts | None

    @property
    def slot(self) -> Slot:
        return self.order.slots[self.slot_index]


def _decision(
    rows: Sequence[_Scores], policy: Policy, at: int, reason: str, executed: LoopCounts,
    *, recorded: LoopCounts | None = None, simulated: LoopCounts | None = None,
) -> Decision:
    fallback = policy.fallback if reason == "no_trigger" else None
    pool = "first" if fallback == "first" else policy.pool
    p, index = _pool(rows, pool, policy.first_weight)
    return Decision(rows[0].order, p, index, at, reason, fallback, executed, recorded, simulated)


def actuate(
    plan: LoopPlan, question: LoopCallbacks[H, S, O], native_order: Callable[[], SlotOrder],
    policy: Policy,
) -> Decision:
    """Run the live policy and report actual calls, including the readout used to stop."""
    _repeat_only(plan)
    bound = _Bound(question, native_order)
    if policy.pool == "final" and policy.stop is None:
        r = run_loop(replace(plan, readout="final"), bound)
        return _decision([r.logits], policy, plan.k, "budget", r.counts)
    seen: list[_Scores] = []

    def stop(probe: Readout[_Scores]) -> bool:
        seen.append(probe.logits)
        return _reason(seen, policy, plan.k) is not None

    r = run_loop(replace(plan, readout="diagnostic"), bound, stop_when=stop)
    rows = [probe.logits for probe in r.readouts]
    reason = _reason(rows, policy, plan.k)
    assert reason is not None
    return _decision(rows, policy, len(rows), reason, r.counts)


def replay(trace: PassTrace, policy: Policy) -> Decision:
    """No model execution. Separate recording costs from counterfactual live-policy costs."""
    rows: list[_Scores] = []
    for probe in trace.result.readouts:
        rows.append(probe.logits)
        reason = _reason(rows, policy, trace.plan.k)
        if reason is not None:
            at = len(rows)
            codas = 1 if policy.pool == "final" and policy.stop is None else at
            return _decision(
                rows, policy, at, reason, LoopCounts(0, 0, 0, 0), recorded=trace.result.counts,
                simulated=_counts(trace.plan.k, at, codas),
            )
    raise ValueError("trace ended without a policy decision")


@dataclass(frozen=True)
class ThresholdFit:
    policy: Policy
    inputs: int
    agreements_with_final: int
    simulated_blocks: int
    simulated_codas: int
    recorded_codas: int


def fit_threshold(
    expected_inputs: Iterable[SlotOrder], traces: Iterable[PassTrace], policy: Policy,
    thresholds: Iterable[float | int], *, min_agreement: float = 1.0,
) -> ThresholdFit:
    """Pick a threshold by paired agreement with full-budget predictions, not answer labels."""
    if policy.stop is None or not _unit(min_agreement):
        raise ValueError("fit requires a stop rule and minimum agreement in [0, 1]")
    expected = tuple(expected_inputs)
    if (
        not expected or len({o.split for o in expected}) != 1
        or expected[0].split not in ("discovery", "selection")
    ):
        raise ValueError("fit on a single nonempty discovery or selection split only")
    if len({o.input_key for o in expected}) != len(expected):
        raise ValueError("duplicate manifest input")
    data = tuple(traces)
    if len(data) != len(expected) or {t.order for t in data} != set(expected):
        raise ValueError("traces must cover the manifest exactly, including native slot order")
    if any(t.plan != data[0].plan for t in data):
        raise ValueError("fit requires a common repeat plan")
    recorded = sum(t.result.counts.coda_readouts for t in data)
    best = None
    for value in thresholds:
        candidate = replace(policy, stop=replace(policy.stop, threshold=value))
        matches = blocks = codas = 0
        for trace in data:
            d = replay(trace, candidate)
            assert d.simulated is not None
            matches += d.slot_index == _top(trace.result.logits.logits)
            blocks += d.simulated.block_evaluations
            codas += d.simulated.coda_readouts
        if matches / len(data) < min_agreement:
            continue
        rank = (blocks, codas, -matches)
        if best is None or rank < best[0]:
            best = (rank, ThresholdFit(candidate, len(data), matches, blocks, codas, recorded))
    if best is None:
        raise ValueError("no candidate threshold met the paired agreement requirement")
    return best[1]
