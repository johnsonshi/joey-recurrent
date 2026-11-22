"""Inference-only, full-sequence recurrence over a caller-supplied frozen block."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar


class _Vector(Protocol):
    def __add__(self: H, other: H, /) -> H: ...
    def __sub__(self: H, other: H, /) -> H: ...
    def __mul__(self: H, scalar: float, /) -> H: ...


H = TypeVar("H", bound=_Vector)
S = TypeVar("S")
O = TypeVar("O")
O_co = TypeVar("O_co", covariant=True)
Method = Literal["zero_extra", "naive", "damped_extra", "euler", "rk", "prelude_anchor"]
ReadoutMode = Literal["final", "diagnostic"]
ProbeKind = Literal["stock", "first_pass", "iterate", "substep", "stage_output", "endpoint"]


def _integer(name: str, value: int, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _unit(name: str, value: float | None) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    value = _finite(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class RKTableau:
    """Explicit unit-step tableau; row i has exactly i earlier-stage coefficients."""

    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]

    def __post_init__(self) -> None:
        rows = tuple(tuple(_finite("a[i,j]", v) for v in row) for row in self.a)
        weights = tuple(_finite("b[i]", v) for v in self.b)
        if (
            not weights
            or len(rows) != len(weights)
            or any(len(row) != i for i, row in enumerate(rows))
        ):
            raise ValueError("an s-stage explicit tableau needs s rows of lengths 0, ..., s-1")
        if not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-12, rel_tol=0.0):
            raise ValueError("unit-step output weights must sum to one")
        object.__setattr__(self, "a", rows)
        object.__setattr__(self, "b", weights)


@dataclass(frozen=True)
class LoopPlan:
    k: int = 1
    method: Method = "naive"
    readout: ReadoutMode = "final"
    alpha: float | None = None
    beta: float | None = None
    tableau: RKTableau | None = None

    def __post_init__(self) -> None:
        _integer("k", self.k, 1)
        if self.method not in (
            "zero_extra", "naive", "damped_extra", "euler", "rk", "prelude_anchor"
        ):
            raise ValueError(f"unknown method: {self.method!r}")
        if self.readout not in ("final", "diagnostic"):
            raise ValueError(f"unknown readout mode: {self.readout!r}")
        if self.method == "damped_extra":
            _unit("alpha", self.alpha)
        elif self.alpha is not None:
            raise ValueError("alpha applies only to damped_extra")
        if self.tableau is not None:
            if self.method != "rk" or not isinstance(self.tableau, RKTableau):
                raise ValueError("tableau must be an RKTableau for method='rk'")
            if len(self.tableau.b) != self.k:
                raise ValueError("k must equal the number of tableau stages")
        if self.method in ("rk", "prelude_anchor") and self.tableau is None:
            _unit("beta", self.beta)
        elif self.beta is not None:
            raise ValueError("beta applies only to the built-in rk and prelude_anchor rules")


class LoopCallbacks(Protocol[H, S, O_co]):
    """Bind one immutable prompt. block and coda receive new call-local state."""

    def stock(self) -> O_co: ...
    def prelude(self) -> H: ...
    def block(self, hidden: H, state: S, /) -> H: ...
    def coda(self, hidden: H, state: S, /) -> O_co: ...
    def fresh_state(self) -> S: ...
    def inference_context(self) -> AbstractContextManager[object]: ...


@dataclass(frozen=True)
class LoopCounts:
    """Observed split calls and opaque stock calls; each stock call includes one of each part."""

    stock_calls: int
    prelude_calls: int
    block_calls: int
    coda_calls: int

    @property
    def block_evaluations(self) -> int:
        return self.stock_calls + self.block_calls

    @property
    def coda_readouts(self) -> int:
        return self.stock_calls + self.coda_calls

    def layer_evaluations(self, prelude_layers: int, block_layers: int, coda_layers: int) -> int:
        """Count layer invocations, not FLOPs, latency, tokens, or head projections."""
        _integer("prelude_layers", prelude_layers, 0)
        _integer("block_layers", block_layers, 1)
        _integer("coda_layers", coda_layers, 0)
        return (
            prelude_layers * (self.stock_calls + self.prelude_calls)
            + block_layers * self.block_evaluations
            + coda_layers * self.coda_readouts
        )


@dataclass(frozen=True)
class Readout(Generic[O]):
    block_evaluations: int
    kind: ProbeKind
    logits: O


@dataclass(frozen=True)
class LoopResult(Generic[O]):
    logits: O
    counts: LoopCounts
    readouts: tuple[Readout[O], ...]
    stopped_early: bool = False


def _advance(y: H, gy: H, scale: float) -> H:
    if scale == 0:
        return y
    if scale == 1:
        return gy
    return y + (gy - y) * scale


def _anchored(endpoint: H, anchor: H, beta: float) -> H:
    if beta == 0:
        return endpoint
    if beta == 1:
        return anchor
    return endpoint * (1 - beta) + anchor * beta


def _combine(base: H, slopes: list[H], weights: tuple[float, ...]) -> H:
    y = base
    for slope, weight in zip(slopes, weights, strict=True):
        if weight:
            y = y + slope * weight
    return y


def _window(plan: LoopPlan, y0: H, g: Callable[[H], H]) -> Iterator[tuple[ProbeKind, H]]:
    if plan.method in ("zero_extra", "naive", "damped_extra"):
        y = g(y0)
        yield "first_pass", y
        if plan.method != "zero_extra":
            for _ in range(1, plan.k):
                gy = g(y)
                y = gy if plan.alpha is None else _advance(y, gy, plan.alpha)
                yield "iterate", y
        return

    if plan.tableau is not None:
        slopes: list[H] = []
        for i, row in enumerate(plan.tableau.a):
            stage = _combine(y0, slopes, row)
            gy = g(stage)
            slopes.append(gy - stage)
            if i + 1 < plan.k:
                yield "stage_output", gy
        yield "endpoint", _combine(y0, slopes, plan.tableau.b)
        return

    y, anchor = y0, y0
    for i in range(plan.k):
        gy = g(y)
        if i == 0 and plan.method == "rk":
            anchor = gy
        y = _advance(y, gy, 1 / plan.k)
        if i + 1 < plan.k:
            if plan.method == "euler":
                yield "substep", y
            else:
                yield "stage_output", gy
    # For Chen's tableau, the weighted slope sum is exactly this terminal blend
    # in real arithmetic. The anchor was obtained in stage one, not in a pilot.
    yield "endpoint", y if plan.beta is None else _anchored(y, anchor, plan.beta)


def run_loop(
    plan: LoopPlan,
    callbacks: LoopCallbacks[H, S, O],
    *,
    stop_when: Callable[[Readout[O]], bool] | None = None,
) -> LoopResult[O]:
    """Execute one bound prompt. Only completed repeat/damped iterates can stop early."""
    if stop_when is not None and (
        plan.readout != "diagnostic" or plan.method not in ("naive", "damped_extra")
    ):
        raise ValueError("stop_when requires diagnostic readouts of naive or damped_extra iterates")

    with callbacks.inference_context():
        if plan.k == 1:
            logits = callbacks.stock()
            trace = (Readout(1, "stock", logits),) if plan.readout == "diagnostic" else ()
            return LoopResult(logits, LoopCounts(1, 0, 0, 0), trace)

        block_calls = 0

        def g(hidden: H) -> H:
            nonlocal block_calls
            block_calls += 1
            return callbacks.block(hidden, callbacks.fresh_state())

        y0 = callbacks.prelude()
        trace_items: list[Readout[O]] = []
        coda_calls = 0
        budget = 1 if plan.method == "zero_extra" else plan.k
        for kind, hidden in _window(plan, y0, g):
            final = block_calls == budget
            if not final and plan.readout == "final":
                continue
            logits = callbacks.coda(hidden, callbacks.fresh_state())
            coda_calls += 1
            probe = Readout(block_calls, kind, logits)
            if plan.readout == "diagnostic":
                trace_items.append(probe)
            stop = False if final or stop_when is None else stop_when(probe)
            if type(stop) is not bool:
                raise TypeError("stop_when must return a boolean")
            if final or stop:
                counts = LoopCounts(0, 1, block_calls, coda_calls)
                return LoopResult(logits, counts, tuple(trace_items), stopped_early=stop)
    raise RuntimeError("loop did not produce an endpoint")
