"""One live margin check on a single bound native branch."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

from .loop import LoopCounts


class _Hidden(Protocol):
    def clone(self: H) -> H: ...


H = TypeVar("H", bound=_Hidden)
S = TypeVar("S")
L = TypeVar("L")
T = TypeVar("T")


class OneProbeBranch(Protocol[H, S, L]):
    """One input, physical window and native slot order for the entire call."""

    @property
    def labels(self) -> Sequence[str | int]: ...
    def inference_context(self) -> AbstractContextManager[object]: ...
    def prelude(self) -> H: ...
    def fresh_state(self) -> S: ...
    def block(self, hidden: H, state: S, /) -> H: ...
    def coda(self, hidden: H, state: S, /) -> L: ...
    def probabilities(self, logits: L, /) -> Sequence[float]: ...


@dataclass(frozen=True)
class OneProbeResult(Generic[L]):
    logits: L
    labels: tuple[str | int, ...]
    first_probabilities: tuple[float, ...]
    first_margin: float
    exit_depth: int
    reason: Literal["margin", "cap"]
    counts: LoopCounts


def _slot_order(labels: Sequence[str | int]) -> tuple[str | int, ...]:
    if not isinstance(labels, (tuple, list)):
        raise ValueError("supply the ordered native slot keys as a tuple or list")
    slots = tuple(labels)
    if (len(slots) < 2 or any(type(s) not in (str, int) or s == "" for s in slots)
            or len(set(slots)) != len(slots)):
        raise ValueError("need at least two distinct native slot keys (nonempty strings or integers)")
    return slots


def _finite_vector(raw: object, width: int, name: str) -> tuple[float, ...]:
    values = raw.tolist() if hasattr(raw, "tolist") else raw
    error = f"{name} must contain one finite number per ordered native slot"
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(error)
    values = tuple(values)
    if len(values) != width or any(type(x) not in (float, int) for x in values):
        raise ValueError(error)
    try:
        vector = tuple(map(float, values))
    except OverflowError as exc:
        raise ValueError(error) from exc
    if not all(map(math.isfinite, vector)):
        raise ValueError(error)
    return vector


def _probe_copy(hidden: H) -> H:
    """Require native clone semantics; a detach or shallow view is not sufficient."""
    clone = getattr(hidden, "clone", None)
    if not callable(clone):
        raise TypeError("probe hidden must support an independent native clone()")
    copied = clone()
    if copied is hidden or type(copied) is not type(hidden):
        raise ValueError("probe clone must be a separate hidden in the native representation")
    return copied


def run_one_probe(
    branch: OneProbeBranch[H, S, L], *, cap: int, threshold: float
) -> OneProbeResult[L]:
    """Return the first split coda at margin >= threshold, otherwise the coda at cap."""
    if type(cap) is not int or cap not in (2, 3, 4, 6):
        raise ValueError("cap must be an integer in {2, 3, 4, 6}")
    if type(threshold) not in (float, int) or threshold not in (0, 0.10, 0.25, 0.50, 0.75, 1):
        raise ValueError("threshold must be one of 0, 0.10, 0.25, 0.50, 0.75, 1.0")

    labels = _slot_order(branch.labels)
    width = len(labels)
    coda_calls = 0

    def check_order() -> None:
        current = branch.labels
        # The native tuple contains only immutable keys. A list must be checked by value.
        if current is not labels and _slot_order(current) != labels:
            raise ValueError("native slot order changed during the run")

    def invoke(stage: Callable[[H, S], T], hidden: H) -> T:
        check_order()
        state = branch.fresh_state()
        check_order()
        out = stage(hidden, state)
        check_order()
        return out

    def readout(hidden: H) -> L:
        nonlocal coda_calls
        logits = invoke(branch.coda, hidden)
        coda_calls += 1
        _finite_vector(logits, width, "logits")
        check_order()
        return logits  # Preserve the native object, not the validation copy.

    with branch.inference_context():
        check_order()
        hidden = invoke(branch.block, branch.prelude())
        block_calls = 1
        logits = readout(_probe_copy(hidden))
        first = _finite_vector(branch.probabilities(logits), width, "first probabilities")
        check_order()
        if any(p < 0 or p > 1 for p in first) or abs(math.fsum(first) - 1) > 1e-6:
            raise ValueError("first probabilities must be in [0, 1] and sum to one")
        top = first.index(max(first))
        margin = first[top] - max(p for i, p in enumerate(first) if i != top)

        reason: Literal["margin", "cap"]
        if margin >= threshold:
            reason = "margin"
        else:
            reason = "cap"
            del logits
            for _ in range(1, cap):
                hidden = invoke(branch.block, hidden)
                block_calls += 1
            logits = readout(hidden)

    check_order()
    counts = LoopCounts(stock_calls=0, prelude_calls=1, block_calls=block_calls, coda_calls=coda_calls)
    return OneProbeResult(logits, labels, first, margin, block_calls, reason, counts)
