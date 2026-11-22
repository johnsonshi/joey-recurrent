"""Matched routing questions: one fixed target and two complementary root-edge edits."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import hashlib
from itertools import pairwise
import json
import random
import re
from typing import NamedTuple

from joey_research.evaluation import EvaluationError, input_sha256, label_row


TASK = "parcel_entry_route"
_NAMESPACE = "routing-edits-v1"
_HOPS = (2, 6)
_EDITS = ("target_swap", "other_swap")
_CONDITIONS = ("base", *_EDITS)
_CELLS = {(w, c) for w in (0, 1) for c in _CONDITIONS}
_META_FIELDS = {"case_id", "hop", "rendering", "condition", "seed"}

# Pin the two fixed_horizon display conventions, not its task, sampling or labeler.
_HEADERS = (
    "Target station: {}. Which entry station's route includes this target?\n"
    "Read edges from sender to receiver. Records are unordered.",
    "Find the entry station that eventually sends a parcel through {}.\n"
    "Each record gives the receiver first, then the sender. Records are unordered.",
)
_STATION = r"([A-Z]{3})"
_HEADER_RE = tuple(re.compile(re.escape(h).replace(re.escape("{}"), _STATION)) for h in _HEADERS)
_EDGE_RE = (re.compile(f"{_STATION} -> {_STATION}"),
            re.compile(f"{_STATION} gets parcels from {_STATION}"))
_OPTION_RE = re.compile(f"Enter at {_STATION}")


def _digest(*parts):
    data = json.dumps([_NAMESPACE, *parts], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def _case_id(group, seed, goal, hop):
    return _digest("case", group, seed, goal, hop)


def _row_id(case_id, rendering, condition, prompt, choices):
    # Bind the role to the actual display without putting the role in the request.
    return _digest("input", case_id, rendering, condition, prompt, choices)


def _valid_seed(seed):
    return type(seed) is int and 0 <= seed < 2**32


def _rng(group, field):
    return random.Random(int(_digest("draw", group, field), 16))


def _orders(group, field, items):
    first, second = list(items), list(items)
    _rng(group, field + ":0").shuffle(first)
    _rng(group, field + ":1").shuffle(second)
    if first == second:
        second = second[1:] + second[:1]
    return first, second


def _swap(base, pair):
    a, b = pair
    edited = base.copy()
    edited[a], edited[b] = base[b], base[a]
    return edited


def _body(graph, sources, rendering):
    if rendering == 0:
        return "\n".join(f"{s} -> {graph[s]}" for s in sources)
    return "\n".join(f"{graph[s]} gets parcels from {s}" for s in sources)


def iter_cases(seed, graphs):
    """Yield (evaluator input, separate blind design) in independently seeded groups of 48."""
    if not _valid_seed(seed) or type(graphs) is not int or not 1 <= graphs <= 1000:
        raise EvaluationError("invalid routing-edits sampling bounds")
    for index in range(graphs):
        group = _digest("group", seed, index)
        numbers = _rng(group, "stations").sample(range(26**3), 28)
        names = ["".join(chr(65 + n // k % 26) for k in (676, 26, 1)) for n in numbers]
        chains = [names[i:i + 7] for i in range(0, 28, 7)]
        base = {s: t for chain in chains for s, t in pairwise(chain)}
        roots = [chain[0] for chain in chains]
        paired = roots.copy()
        _rng(group, "pairs").shuffle(paired)
        first_pair, second_pair = paired[:2], paired[2:]
        variants = base, _swap(base, first_pair), _swap(base, second_pair)

        orders = zip(_orders(group, "records", base), _orders(group, "menu", roots), strict=True)
        for w, (sources, menu) in enumerate(orders):
            original, first, second = (_body(g, sources, w) for g in variants)
            choices = [f"Enter at {root}" for root in menu]
            for chain in chains:
                edits = (first, second) if chain[0] in first_pair else (second, first)
                for hop in _HOPS:
                    goal = chain[hop]
                    case = _case_id(group, seed, goal, hop)
                    header = _HEADERS[w].format(goal)
                    for condition, body in zip(_CONDITIONS, (original, *edits), strict=True):
                        prompt = header + "\n" + body
                        row = {"id": _row_id(case, w, condition, prompt, choices), "group_id": group,
                               "task": TASK, "prompt": prompt, "choices": choices.copy()}
                        meta = {"case_id": case, "hop": hop, "rendering": w,
                                "condition": condition, "seed": seed}
                        yield row, meta


class _Display(NamedTuple):
    goal: str
    rendering: int
    edges: tuple[tuple[str, str], ...]  # sender, receiver, in printed record order
    entries: tuple[str, ...]


def _decode(row):
    """Read an allowlisted row literally; no whitespace normalization or reachability queries."""
    lines = row["prompt"].split("\n")
    if row["task"] != TASK or len(lines) != 26 or len(row["choices"]) != 4:
        raise EvaluationError("expected a routing-edits question with 24 records and four entries")
    header = "\n".join(lines[:2])
    for w, pattern in enumerate(_HEADER_RE):
        match = pattern.fullmatch(header)
        if match:
            goal = match[1]
            break
    else:
        raise EvaluationError("invalid routing header")

    edges, entries = [], []
    for line in lines[2:]:
        match = _EDGE_RE[w].fullmatch(line)
        if not match:
            raise EvaluationError("invalid routing record")
        a, b = match.groups()
        edges.append((a, b) if w == 0 else (b, a))
    for choice in row["choices"]:
        match = _OPTION_RE.fullmatch(choice)
        if not match:
            raise EvaluationError("invalid entry option")
        entries.append(match[1])
    sources, destinations = (set(column) for column in zip(*edges))
    if (len(sources) != 24 or len(destinations) != 24 or len(sources | destinations) != 28
            or set(entries) != sources - destinations or goal not in destinations):
        raise EvaluationError("invalid displayed nodes, degrees, target or roots")
    return _Display(goal, w, tuple(edges), tuple(entries))


def _bind_design(row, display, meta):
    if not isinstance(meta, Mapping) or set(meta) != _META_FIELDS:
        raise EvaluationError("unexpected blind design fields")
    hop, w, condition, seed = (meta[k] for k in ("hop", "rendering", "condition", "seed"))
    if (type(hop) is not int or hop not in _HOPS or type(w) is not int or w != display.rendering
            or condition not in _CONDITIONS or not _valid_seed(seed)):
        raise EvaluationError("invalid blind design cell")
    case = _case_id(row["group_id"], seed, display.goal, hop)
    expected = _row_id(case, w, condition, row["prompt"], row["choices"])
    if meta["case_id"] != case or row["id"] != expected:
        raise EvaluationError("case or input ID does not match its design and display")


def _edited_roots(base, edited):
    if base.goal != edited.goal or base.entries != edited.entries or base.rendering != edited.rendering:
        raise EvaluationError("header or ordered menu changed within a triple")
    changes = []
    for (s, old), (source, new) in zip(base.edges, edited.edges, strict=True):
        if s != source:
            raise EvaluationError("sender or record order changed within a triple")
        if old != new:
            changes.append((s, old, new))
    if len(changes) != 2:
        raise EvaluationError("expected exactly two changed destinations")
    (a, old_a, new_a), (b, old_b, new_b) = changes
    if (new_a, new_b) != (old_b, old_a) or a not in base.entries or b not in base.entries:
        raise EvaluationError("edit is not a swap of two original root-edge destinations")
    return {a, b}


def _alternates(cells, w, reference):
    base = cells[w, "base"]
    if base.edges != reference.edges or base.entries != reference.entries:
        raise EvaluationError("base evidence or menu varies across targets")
    target, other = (cells[w, c] for c in _EDITS)
    if _edited_roots(base, target) | _edited_roots(base, other) != set(base.entries):
        raise EvaluationError("the two edited root pairs are not complementary")
    return frozenset(target.edges), frozenset(other.edges)


def _audit_group(records):
    if len(records) != 48:
        raise EvaluationError("incomplete or oversized routing group")
    cells = defaultdict(dict)
    cases, seeds = set(), set()
    for row, meta in records:
        display = _decode(row)
        _bind_design(row, display, meta)
        cases.add(meta["case_id"])
        seeds.add(meta["seed"])
        slots = cells[display.goal, meta["hop"]]
        slot = display.rendering, meta["condition"]
        if slot in slots:
            raise EvaluationError("duplicate design cell")
        slots[slot] = display
    if (len(seeds) != 1 or len(cases) != 8 or len({goal for goal, hop in cells}) != 8
            or Counter(hop for goal, hop in cells) != {2: 4, 6: 4}
            or any(slots.keys() != _CELLS for slots in cells.values())):
        raise EvaluationError("expected eight distinct targets with six matched cells each")

    example = next(iter(cells.values()))
    a, b = (example[w, "base"] for w in (0, 1))
    original = frozenset(a.edges)
    if original != frozenset(b.edges) or a.edges == b.edges or a.entries == b.entries:
        raise EvaluationError("renderings must reorder the same base records and menu")
    variants, balance = set(), Counter()
    for (_, hop), slots in cells.items():
        edits = _alternates(slots, 0, a)
        if edits != _alternates(slots, 1, b):
            raise EvaluationError("a case changes its assigned graphs across renderings")
        for condition, graph in zip(_EDITS, edits, strict=True):
            variants.add(graph)
            balance[graph, hop, condition] += 1
    if len(variants) != 2 or any(balance[g, h, c] != 2 for g in variants for h in _HOPS for c in _EDITS):
        raise EvaluationError("expected two reused alternate graphs, balanced by hop and role")
    return cases, variants | {original}


def check_blind_panel(rows, design_by_id):
    """Audit complete groups against {row ID: meta}, independent of row order.

    Return counts only. Hop and role are bound design fields, not inferred answers;
    route semantics are checked separately on fixtures or after release.
    """
    if not isinstance(design_by_id, Mapping) or not design_by_id:
        raise EvaluationError("expected a nonempty design keyed by input ID")
    groups, ids = defaultdict(list), set()
    for row in rows:
        input_sha256(row)
        key = row["id"]
        if key in ids or key not in design_by_id:
            raise EvaluationError("duplicate or unassigned input ID")
        ids.add(key)
        groups[row["group_id"]].append((row, design_by_id[key]))
    if ids != design_by_id.keys():
        raise EvaluationError("rows and design do not have identical membership")

    cases, graphs = set(), set()
    for records in groups.values():
        group_cases, group_graphs = _audit_group(records)
        if cases & group_cases or graphs & group_graphs:
            raise EvaluationError("case or actual graph reused across groups")
        cases.update(group_cases)
        graphs.update(group_graphs)
    return {"rows": len(ids), "groups": len(groups), "cases": len(cases)}


def adjudicate(row):
    """Post-release only: validate all four displayed chains and label this row's ordered menu."""
    input_sha256(row)
    display = _decode(row)
    outgoing, visited = dict(display.edges), set()
    answer = None
    for entry in display.entries:
        node = entry
        for hop in range(1, 7):
            if node not in outgoing or node in visited:
                raise EvaluationError("expected four disjoint six-edge chains")
            visited.add(node)
            node = outgoing[node]
            if node == display.goal:
                answer = entry, hop
        if node in outgoing:
            raise EvaluationError("chain continues past six edges")
    if len(visited) != 24 or answer is None or answer[1] not in _HOPS:
        raise EvaluationError("unused records or target outside the declared distances")
    entry, hop = answer
    info = {"entry": entry, "hop": hop, "rendering": display.rendering}
    return label_row(row, display.entries.index(entry)), info
