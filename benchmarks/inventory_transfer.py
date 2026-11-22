"""Synthetic inventory reconciliation for three native decision types. No I/O."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
import hashlib
import json
import random
import re
from typing import NamedTuple


_VERIFY, _LEVEL, _CHOICE = "stock-eligibility", "stock-level", "stock-choice"
NATIVE_TYPES = {_VERIFY: "noul", _LEVEL: "score", _CHOICE: "choice"}
NATIVE_INSTRUCTIONS = {
    _VERIFY: "After posted entries only, does the focus depot have at least five units?",
    _LEVEL: "After posted entries only, rate the final stock of the focus depot using the ordered levels.",
    _CHOICE: "After posted entries only, which depot has the most units? If there is no unique maximum, choose No unique depot.",
}
_DEPOTS = ("Aster", "Birch", "Kelm", "Rill")
_TIE = "No unique depot"
_BINARY = ("Yes", "No")
_LEVELS = ("Empty (0 units)", "Low (1 to 3 units)", "Medium (4 to 6 units)", "High (7 or more units)")
_OPTIONS = {_VERIFY: _BINARY, _LEVEL: _LEVELS, _CHOICE: (*_DEPOTS, _TIE)}
_LENGTHS = (2, 6, 12)
_HEADER = "Warehouse inventory ledger (units; records unordered)"
_FIELDS = {"id", "group_id", "task", "prompt", "choices"}


def _rng(domain: str, *key) -> random.Random:
    return random.Random(json.dumps(["inventory-transfer-v1", domain, *key], separators=(",", ":")))


class _Cell(NamedTuple):
    seed: int
    size: int
    task: str
    length: int


def _cells(seed: int, size: int) -> Iterator[_Cell]:
    if type(seed) is not int or type(size) is not int or size < 1:
        raise ValueError("expected an integer seed and a positive integer groups_per_cell")
    for task in _OPTIONS:
        for length in _LENGTHS:
            yield _Cell(seed, size, task, length)


def _identities(cell: _Cell):
    # Independent of balancing, sampling attempts, rendering and gold.
    rng = _rng("identities", *cell)
    for _ in range(cell.size):
        yield tuple(f"{rng.getrandbits(128):032x}" for _ in range(3))


def group_manifest(seed: int, groups_per_cell: int) -> dict:
    """Label-free group_id -> {seed, task, difficulty (entry count), renderings: {id: 0|1}}."""
    manifest = {}
    for cell in _cells(seed, groups_per_cell):
        for group, first, second in _identities(cell):
            manifest[group] = {"seed": seed, "task": cell.task, "difficulty": cell.length,
                               "renderings": {first: 0, second: 1}}
    return manifest


def _schedule(cell: _Cell):
    rng = _rng("allocation", *cell)
    options = _OPTIONS[cell.task]
    copies, extra = divmod(cell.size, len(options))
    schedule = list(options) * copies + rng.sample(options, extra)
    rng.shuffle(schedule)
    return schedule


class _Entry(NamedTuple):
    posted: bool
    units: int
    source: str | None
    dest: str | None

    def text(self) -> str:
        if self.source is None:
            action = f"add {self.units} to {self.dest}"
        elif self.dest is None:
            action = f"remove {self.units} from {self.source}"
        else:
            action = f"transfer {self.units} from {self.source} to {self.dest}"
        return f"{'POSTED' if self.posted else 'VOID'} | {action}"


def _closing(rng, task, target, focus):
    if task == _CHOICE:
        top = rng.randint(5, 13)
        winners = (target,) if target != _TIE else rng.sample(_DEPOTS, rng.choice((2, 2, 2, 3, 4)))
        return {d: top if d in winners else top - rng.randint(1, min(6, top)) for d in _DEPOTS}
    if task == _VERIFY:
        band = (5, 13) if target == "Yes" else (0, 4)
    else:
        band = ((0, 0), (1, 3), (4, 6), (7, 13))[_LEVELS.index(target)]
    return {d: rng.randint(*band) if d == focus else rng.randint(0, 13) for d in _DEPOTS}


def _focal_entry(rng, stock, focus, peers):
    # Undo a change to reach a freshly sampled preceding stock, not a fixed
    # opening or a special receipt with the desired closing count as its amount.
    prior = rng.choice([x for x in range(15) if x != stock[focus]])
    delta = stock[focus] - prior
    if delta > 0:
        return _Entry(True, delta, rng.choice(peers) if rng.randrange(2) else None, focus)
    funded = [d for d in peers if stock[d] >= -delta]
    dest = rng.choice(funded) if funded and rng.randrange(2) else None
    return _Entry(True, -delta, focus, dest)


def _free_entry(rng, names, stock=None):
    # To undo a posted credit, its destination must contain the credited units.
    eligible = [d for d in names if stock is None or stock[d] > 0]
    op = rng.choice(("add", "remove", "transfer")) if eligible else "remove"
    if op == "remove":
        return _Entry(stock is not None, rng.randint(1, 9), rng.choice(names), None)
    dest = rng.choice(eligible)
    units = rng.randint(1, 9 if stock is None else min(9, stock[dest]))
    source = None if op == "add" else rng.choice([d for d in names if d != dest])
    return _Entry(stock is not None, units, source, dest)


def _backdate(rng, closing, focus, length):
    if length == 2:
        focal, off = 1, 0
    elif length == 6:
        focal, off = rng.randint(2, 3), rng.randint(1, 2)
    else:
        focal, off = rng.randint(4, 7), rng.randint(2, 3)
    slots = [True] * focal + [False] * off
    rng.shuffle(slots)
    peers = [d for d in _DEPOTS if d != focus]
    stock = closing.copy()
    entries = []
    for touches_focus in slots:
        e = _focal_entry(rng, stock, focus, peers) if touches_focus else _free_entry(rng, peers, stock)
        if e.dest is not None:
            stock[e.dest] -= e.units
        if e.source is not None:
            stock[e.source] += e.units
        entries.append(e)
    entries.extend(_free_entry(rng, _DEPOTS) for _ in range(length - len(slots)))
    return stock, entries


def _book(rng, task, target, length):
    focus = rng.choice(_DEPOTS)
    # Sometimes the opening leader survives. Making it *always* wrong would
    # itself convey information. Focus is fixed before this nuisance sampling.
    keep_opening_rank = rng.randrange(5) == 0
    while True:
        closing = _closing(rng, task, target, focus)
        opening, entries = _backdate(rng, closing, focus, length)
        if task != _CHOICE:
            return focus, opening, entries
        high_before, high_after = max(opening.values()), max(closing.values())
        before = [d for d in _DEPOTS if opening[d] == high_before]
        opening_answer = before[0] if len(before) == 1 else _TIE
        contenders = set(before) | {d for d in _DEPOTS if closing[d] == high_after}
        if ((opening_answer == target) == keep_opening_rank
                and any(e.posted and (e.source in contenders or e.dest in contenders) for e in entries)):
            return focus, opening, entries


def _permutation(rng, items, previous):
    # Independent stream per component/view, conditioned only on looking different.
    while True:
        order = rng.sample(items, len(items))
        if order != previous:
            return order


def iter_inputs(seed: int, groups_per_cell: int) -> Iterator[dict]:
    """Yield exactly two label-free decisions per independent task/difficulty group."""
    for cell in _cells(seed, groups_per_cell):
        task = cell.task
        for i, (ids, target) in enumerate(zip(_identities(cell), _schedule(cell))):
            group, *row_ids = ids
            key = (*cell, i)
            focus, opening, entries = _book(_rng("book", *key), task, target, cell.length)
            opens = [f"Opening {d}: {units}" for d, units in opening.items()]
            tickets = _rng("tickets", *key).sample(range(1 << 30), cell.length)
            records = [f"R-{t:08X} | {e.text()}" for t, e in zip(tickets, entries)]
            prior_opens = prior_records = prior_menu = None
            for view, row_id in enumerate(row_ids):
                headings = _permutation(_rng("openings", *key, view), opens, prior_opens)
                lines = _permutation(_rng("receipts", *key, view), records, prior_records)
                options = _OPTIONS[task]
                menu = (_permutation(_rng("menu", *key, view), options, prior_menu)
                        if task == _CHOICE else list(options))
                prompt = "\n".join((_HEADER, f"Focus depot: {focus}", *headings, *lines))
                prior_opens, prior_records, prior_menu = headings, lines, menu
                yield {"id": row_id, "group_id": group, "task": task, "prompt": prompt, "choices": menu.copy()}


# The adjudicator reads facts, not a renderer, generator configuration or RNG.
_NAME = r"[A-Z][a-z]*"
_UINT = r"(?:0|[1-9][0-9]*)"


def _read_stock(text: str) -> tuple[str, dict[str, int]]:
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != _HEADER:
        raise ValueError("missing inventory ledger header")
    opening: dict[str, int] = {}
    focus = None
    tickets, statuses, mentioned = set(), set(), set()
    net: Counter[str] = Counter()
    for line in lines[1:]:
        if m := re.fullmatch(rf"Focus depot: ({_NAME})", line):
            if focus is not None:
                raise ValueError("duplicate focus")
            focus = m[1]
            continue
        if m := re.fullmatch(rf"Opening ({_NAME}): ({_UINT})", line):
            if m[1] in opening:
                raise ValueError("duplicate opening")
            opening[m[1]] = int(m[2])
            continue
        fields = [part.strip() for part in line.split("|")]
        if len(fields) != 3:
            raise ValueError("expected a ledger fact or a three-field receipt")
        ticket, status, body = fields
        if not re.fullmatch(r"R-[0-9A-F]{8}", ticket) or ticket in tickets or status not in {"POSTED", "VOID"}:
            raise ValueError("invalid receipt ID, repeated ID or unknown status")
        tickets.add(ticket)
        statuses.add(status)
        match body.split():
            case ["add", qty, "to", dest]:
                legs = [(dest, 1)]
            case ["remove", qty, "from", src]:
                legs = [(src, -1)]
            case ["transfer", qty, "from", src, "to", dest] if src != dest:
                legs = [(src, -1), (dest, 1)]
            case _:
                raise ValueError("invalid inventory action")
        if not re.fullmatch(r"[1-9][0-9]*", qty):
            raise ValueError("receipt quantity must be a positive integer")
        for depot, sign in legs:
            mentioned.add(depot)  # A void receipt must still be well formed.
            if status == "POSTED":
                net[depot] += sign * int(qty)
    if len(opening) != 4 or focus not in opening or not mentioned <= opening.keys():
        raise ValueError("expected four declared depots and no undeclared references")
    if len(tickets) not in _LENGTHS or statuses != {"POSTED", "VOID"}:
        raise ValueError("expected a 2-, 6- or 12-record ledger with both statuses")
    closing = {d: units + net[d] for d, units in opening.items()}
    if min(closing.values()) < 0:
        raise ValueError("negative final inventory")
    return focus, closing


def adjudicate(row: dict) -> dict:
    """Reconcile only this displayed state and bind its zero-based answer to the exact input."""
    if not isinstance(row, dict) or row.keys() != _FIELDS:
        raise ValueError("expected exactly id, group_id, task, prompt, choices")
    if not all(isinstance(row[k], str) and row[k].strip() for k in ("id", "group_id", "task", "prompt")):
        raise ValueError("expected nonempty text fields")
    task, menu = row["task"], row["choices"]
    if task not in _OPTIONS or not isinstance(menu, list) or not all(isinstance(s, str) for s in menu):
        raise ValueError("unknown task or malformed choice list")
    focus, stock = _read_stock(row["prompt"])
    if task == _CHOICE:
        if len(menu) != 5 or set(menu) != {*stock, _TIE}:
            raise ValueError("choice menu must contain the four declared depots and the tie option")
        high = max(stock.values())
        leaders = [d for d, units in stock.items() if units == high]
        index = menu.index(leaders[0] if len(leaders) == 1 else _TIE)
    else:
        if tuple(menu) != _OPTIONS[task]:
            raise ValueError("invalid native menu or order")
        units = stock[focus]
        index = int(units < 5) if task == _VERIFY else sum(units >= cut for cut in (1, 4, 7))
    data = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"id": row["id"], "input_sha256": hashlib.sha256(data.encode("utf-8")).hexdigest(), "gold_index": index}
