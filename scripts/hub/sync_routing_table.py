"""Sync the README's five-run comparison to both release cards, or check for drift.

Run without arguments for an offline check. Use --write after editing the marked
table in the root README. Add --staging DIR to include DIR/<model>/README.md in
either operation. After publication, --live NAMESPACE checks the two public cards.
"""

import argparse
from pathlib import Path
import re
import sys
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("Joey-1-Loop", "Joey-1-Compact-Loop")
BEGIN = b"<!-- routing-results:begin -->\n"
END = b"\n<!-- routing-results:end -->"


def split_table(doc, source):
    if doc.count(BEGIN) != 1 or doc.count(END) != 1:
        raise ValueError(f"{source}: expected one marked routing table")
    before, tail = doc.split(BEGIN)
    body, marker, after = tail.partition(END)
    if not marker:
        raise ValueError(f"{source}: routing table markers are out of order")
    return before + BEGIN, body, END + after


def canonical(root):
    path = root / "README.md"
    _, table, _ = split_table(path.read_bytes(), path)
    lines = table.strip(b"\n").splitlines()
    if len(lines) != 7 or not all(line.startswith(b"|") and line.endswith(b"|")
                                  and line.count(b"|") == lines[0].count(b"|") for line in lines):
        raise ValueError(f"{path}: expected a Markdown header, separator and five comparison rows")
    return table


def card_paths(root, staging=None):
    paths = [root / "docs" / "model-cards" / (name.lower() + ".md") for name in MODELS]
    if staging is not None:
        paths.extend(staging / name / "README.md" for name in MODELS)
    return paths


def sync_local(table, paths, write=False):
    updates = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"{path}: expected a regular model card")
        before, old, after = split_table(path.read_bytes(), path)
        if old != table:
            updates[path] = before + table + after
    if updates and not write:
        raise ValueError("routing table differs: " + ", ".join(map(str, updates)))
    for path, data in updates.items():
        path.write_bytes(data)


def check_live(table, owner):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", owner):
        raise ValueError("--live requires a Hub user or organization name")
    for name in MODELS:
        repo = f"{owner}/{name}"
        with urlopen(f"https://huggingface.co/{repo}/raw/main/README.md", timeout=30) as response:
            _, actual, _ = split_table(response.read(), repo)
        if actual != table:
            raise ValueError(f"{repo}: published routing table differs from the root README")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write", action="store_true", help="copy the canonical table into local cards")
    action.add_argument("--live", metavar="NAMESPACE", help="also verify both published Hub tables")
    parser.add_argument("--staging", type=Path, help="directory containing the two complete release folders")
    args = parser.parse_args()
    table = canonical(ROOT)
    sync_local(table, card_paths(ROOT, args.staging), write=args.write)
    if args.live:
        check_live(table, args.live)
    print("All requested routing tables match the root README.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))
