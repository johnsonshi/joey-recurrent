"""Offline, restricted-menu element selection. Not the native Mind2Web task/scorer."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET

from joey_research.evaluation import EvaluationError, Row, label_row


class UnscorableStep(EvaluationError):
    """Report as a coverage/validity blocker; do not silently remove the state."""


@dataclass(frozen=True)
class Projection:
    input: Row
    node_ids: tuple[str | None, ...]  # Private joining key; never feed to the model.
    unique_named_elements: int
    discarded_ambiguous_elements: int


_TAGS = frozenset({"button", "a", "input", "select", "textarea"})
_ROLES = frozenset({"button", "link", "textbox", "combobox", "checkbox", "radio", "tab"})
_HIDDEN_TAGS = frozenset({"script", "style", "head", "template", "noscript"})
_NONE = "None of the listed elements"


def _key(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"expected nonempty {name}")
    return value


def _hidden(node: ET.Element) -> bool:
    return (node.tag in _HIDDEN_TAGS or "hidden" in node.attrib
            or node.get("aria-hidden", "").lower() == "true"
            or node.get("type", "").lower() in {"hidden", "password"})


def _visible_text(node: ET.Element) -> str:
    # Do not expose hidden descendants, ID attributes, hrefs, or input values.
    parts = []
    todo = [(node, False)]
    while todo:
        cur, tail = todo.pop()
        if tail:
            parts.append(cur.tail or "")
        elif not _hidden(cur):
            parts.append(cur.text or "")
            for child in reversed(cur):
                todo.extend(((child, True), (child, False)))
    return " ".join(parts)


def _visible_nodes(root: ET.Element):
    todo = [root]
    while todo:
        node = todo.pop()
        if not _hidden(node):
            yield node
            todo.extend(reversed(node))


def _render(node: ET.Element) -> str | None:
    if node.tag not in _TAGS and node.get("role") not in _ROLES:
        return None
    role = node.get("role")
    kind = role if role in _ROLES else node.tag
    text = next((s for name in ("aria-label", "title", "placeholder")
                 if (s := node.get(name, "")).strip()), None)
    if text is None:
        text = _visible_text(node)
    text = " ".join(unicodedata.normalize("NFKC", text).split())
    if not text:
        return None
    return f"{kind}: {text[:120]}"


def project_step(trajectory: Row, index: int, *, limit: int = 5,
                 salt: str = "menu-v1") -> Projection:
    """Use pre-action fields only. `limit` counts elements, not the abstention choice."""
    if type(limit) is not int or limit < 1 or not isinstance(salt, str) or not salt:
        raise EvaluationError("expected a positive menu limit and a nonempty salt")
    tid = _require_text(trajectory.get("annotation_id"), "annotation_id")
    goal = _require_text(trajectory.get("confirmed_task"), "confirmed_task")
    steps = trajectory.get("actions")
    if not isinstance(steps, list) or type(index) is not int or not 0 <= index < len(steps):
        raise EvaluationError("invalid action index")
    step = steps[index]
    if not isinstance(step, dict):
        raise EvaluationError("expected a step object")
    uid = _require_text(step.get("action_uid"), "action_uid")
    html = _require_text(step.get("cleaned_html"), "pre-action cleaned_html")
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", html, re.I):
        raise UnscorableStep("declarations are not supported in cleaned DOM")
    try:
        root = ET.fromstring(html)
    except ET.ParseError as exc:
        raise UnscorableStep("cleaned DOM is not well-formed markup") from exc
    elements = {}
    headings = []
    for node in _visible_nodes(root):
        if node.tag in {"h1", "h2", "h3", "legend"} and len(headings) < 6:
            heading = " ".join(_visible_text(node).split())[:120]
            if heading and heading not in headings:
                headings.append(heading)
        ident = node.get("backend_node_id")
        if ident and (rendered := _render(node)):
            if ident in elements:
                raise UnscorableStep("duplicate rendered backend_node_id")
            elements[ident] = rendered
    counts = Counter(elements.values())
    unique = [k for k, text in elements.items() if counts[text] == 1]
    if not unique:
        raise UnscorableStep("no unambiguous named interactive element")
    ranked = sorted(unique, key=lambda k: (_key("subset", salt, tid, uid, k), k))
    menu: list[str | None] = [*ranked[:limit], None]
    menu.sort(key=lambda k: _key("order", salt, tid, uid, k))
    row = {
        "id": "m2w-i-" + _key(tid, uid), "group_id": "m2w-g-" + _key(tid),
        "task": "mind2web_restricted_element",
        "prompt": (f"Task: {goal}\nPage headings: {'; '.join(headings) or '(none shown)'}\n"
                   "Which listed element should be used next on this page?"),
        "choices": [_NONE if k is None else elements[k] for k in menu],
    }
    return Projection(row, tuple(menu), len(unique), len(elements) - len(unique))


def adjudicate_step(trajectory: Row, index: int, *, limit: int = 5,
                    salt: str = "menu-v1") -> Row:
    """Label-side operation only. Requires a unique target in the frozen action space."""
    projection = project_step(trajectory, index, limit=limit, salt=salt)
    positives = trajectory["actions"][index].get("pos_candidates")
    if not isinstance(positives, list) or not positives:
        raise UnscorableStep("no documented positive element in cleaned page")
    ids = set()
    for c in positives:
        if not isinstance(c, dict):
            raise UnscorableStep("malformed positive candidate")
        ids.add(_require_text(c.get("backend_node_id"), "positive backend_node_id"))
    visible = [i for i, node_id in enumerate(projection.node_ids) if node_id in ids]
    if len(visible) > 1:
        raise UnscorableStep("multiple listed positives; no unique single-label proper-score target")
    target = visible[0] if visible else projection.node_ids.index(None)
    return label_row(projection.input, target)
