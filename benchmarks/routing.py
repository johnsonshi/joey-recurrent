"""Seeded multi-hop routing states. The generator never computes or emits labels."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
import hashlib
import json
import random
import re

from joey_research.evaluation import EvaluationError, Row, label_row


def _key(*parts: object) -> str:
    data = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _rng(*parts: object) -> random.Random:
    return random.Random(int(_key(*parts), 16))


def _name(index: int) -> str:
    return "".join(chr(65 + index // (26 ** i) % 26) for i in (2, 1, 0))


_HEADERS = (
    "A parcel must finish at {goal}. Choose the entry station that sends it there.\n"
    "Each line is one directed handoff. Line order does not set travel order.",
    "Required destination: {goal}. Which entry station routes the parcel to that destination?\n"
    "Each line names the receiver first, then the sender. The list is unordered.",
)
_ARROW = re.compile(r"([A-Z]{3}) -> ([A-Z]{3})")
_RECEIVES = re.compile(r"([A-Z]{3}) receives from ([A-Z]{3})")


def iter_inputs(*, namespace: str, seed: int, groups_per_depth: int,
                depths: Sequence[int] = (2, 3, 4), width: int = 4) -> Iterator[Row]:
    """One independent graph per group, all goals crossed with both renderings."""
    if not isinstance(namespace, str) or not namespace.strip() or type(seed) is not int:
        raise EvaluationError("supply a namespace and an integer seed")
    if type(groups_per_depth) is not int or groups_per_depth < 1:
        raise EvaluationError("groups_per_depth must be positive")
    if type(width) is not int or not 2 <= width <= 16:
        raise EvaluationError("width must be in [2,16]")
    if not depths or any(type(d) is not int or not 2 <= d <= 12 for d in depths):
        raise EvaluationError("each route must require 2 to 12 handoffs")
    if len(set(depths)) != len(depths):
        raise EvaluationError("duplicate depth")
    for depth in sorted(depths):
        for index in range(groups_per_depth):
            key = ("route-v1", namespace, seed, depth, width, index)
            group = "route-g-" + _key(*key, "group")
            names = list(map(_name, _rng(*key, "names").sample(range(26**3), width * (depth + 1))))
            layers = [names[i:i + width] for i in range(0, len(names), width)]
            edges = []
            for j, (left, right) in enumerate(zip(layers, layers[1:])):
                targets = right.copy()
                _rng(*key, "mapping", j).shuffle(targets)
                edges.extend(zip(left, targets))
            for template, header in enumerate(_HEADERS):
                order = edges.copy()
                _rng(*key, "line-order", template).shuffle(order)
                rules = [f"{src} -> {dst}" if template == 0 else f"{dst} receives from {src}"
                         for src, dst in order]
                menu = [f"Send to {name}" for name in layers[0]]
                _rng(*key, "menu", template).shuffle(menu)
                for goal in layers[-1]:
                    yield {
                        "id": "route-i-" + _key(*key, template, goal), "group_id": group,
                        "task": f"routing_{depth}hop",
                        "prompt": header.format(goal=goal) + "\n" + "\n".join(rules),
                        "choices": menu.copy(),
                    }


def inverse_label(row: Row) -> Row:
    """Solve from the displayed text only. Reject malformed or shortcut graphs."""
    prompt = row.get("prompt")
    if not isinstance(prompt, str):
        raise EvaluationError("missing routing prompt")
    lines = prompt.splitlines()
    if len(lines) < 3:
        raise EvaluationError("incomplete routing prompt")
    for t, template in enumerate(_HEADERS):
        pattern = re.escape(template).replace(re.escape("{goal}"), "([A-Z]{3})")
        header = re.fullmatch(pattern, "\n".join(lines[:2]))
        if header:
            goal = header.group(1)
            break
    else:
        raise EvaluationError("unknown routing header")
    parent = {}
    children = set()
    for line in lines[2:]:
        match = (_ARROW if t == 0 else _RECEIVES).fullmatch(line)
        if not match:
            raise EvaluationError("unknown routing rule")
        a, b = match.groups()
        src, dst = (a, b) if t == 0 else (b, a)
        if dst in parent or src in children:
            raise EvaluationError("branches or duplicate edges in route")
        parent[dst] = src
        children.add(src)
    entries = []
    if not isinstance(row.get("choices"), list):
        raise EvaluationError("missing routing choices")
    for option in row["choices"]:
        match = re.fullmatch(r"Send to ([A-Z]{3})", option) if isinstance(option, str) else None
        if not match:
            raise EvaluationError("malformed routing choice")
        entries.append(match.group(1))
    roots, leaves = children - parent.keys(), parent.keys() - children
    if set(entries) != roots or goal not in leaves:
        raise EvaluationError("menu must cover every entry and goal must be a terminal")
    used = set()
    lengths = set()
    origin = {}
    for leaf in leaves:
        cur = leaf
        length = 0
        while cur in parent:
            if cur in used:
                raise EvaluationError("reused node or cycle")
            used.add(cur)
            cur = parent[cur]
            length += 1
        lengths.add(length)
        origin[leaf] = cur
    if len(used) != len(parent) or len(lengths) != 1 or min(lengths) < 2:
        raise EvaluationError("every edge must be on a complete equal-length multi-hop path")
    return label_row(row, entries.index(origin[goal]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--namespace", required=True)
    cli.add_argument("--seed", type=int, required=True)
    cli.add_argument("--groups-per-depth", type=int, required=True)
    cli.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4])
    cli.add_argument("--width", type=int, default=4)
    for item in iter_inputs(**vars(cli.parse_args())):
        print(json.dumps(item, ensure_ascii=False))
