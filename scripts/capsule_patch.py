"""Apply exact unified-diff hunks to in-memory, allowlisted byte strings only."""
from __future__ import annotations

import re

from capsule_io import require


def patch_path(header):
    name = header[4:].rstrip(b"\r\n").split(b"\t", 1)[0]
    if name == b"/dev/null":
        return None
    if name.startswith((b"a/", b"b/")):
        name = name[2:]
    require(re.fullmatch(rb"[A-Za-z0-9_./+-]+", name)
            and all(p and not p.startswith(b".") for p in name.split(b"/")), "unsafe patch header")
    return name.decode("ascii")


HUNK = re.compile(rb"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:[^\n]*)\n?")


def patched(originals, patch):
    files = dict(originals)
    lines = patch.splitlines(keepends=True)
    seen = set()
    i = 0
    while i < len(lines):
        if not lines[i].startswith(b"--- "):
            i += 1
            continue
        require(i + 1 < len(lines) and lines[i + 1].startswith(b"+++ "), "incomplete patch header")
        before, after = patch_path(lines[i]), patch_path(lines[i + 1])
        i += 2
        selected = before in files or after in files
        if selected:
            require(before == after and before not in seen, "unsupported closure rename, deletion or repeat")
            seen.add(before)
        # Parse counts even for excluded sections so hunk text cannot be mistaken for a filename.
        old = files[before].splitlines(keepends=True) if selected else []
        out = []
        cursor = hunks = 0
        while i < len(lines) and lines[i].startswith(b"@@"):
            m = HUNK.fullmatch(lines[i])
            require(m is not None, "unsupported patch hunk")
            a, b, c, d = m.groups()
            removed, added = int(b or 1), int(d or 1)
            start, new_start = int(a) - bool(removed), int(c) - bool(added)
            if selected:
                require(cursor <= start <= len(old), "patch position out of range")
                out.extend(old[cursor:start])
                require(len(out) == new_start, "patch output offset differs")
                cursor = start
            i += 1
            hunks += 1
            while removed or added:
                require(i < len(lines) and lines[i][:1] in (b" ", b"-", b"+"), "incomplete patch hunk")
                op, value = lines[i][:1], lines[i][1:]
                i += 1
                if i < len(lines) and lines[i].rstrip(b"\r\n") == b"\\ No newline at end of file":
                    value = value.removesuffix(b"\n")
                    i += 1
                if op != b"+":
                    require(removed > 0, "patch old count differs")
                    removed -= 1
                    if selected:
                        require(cursor < len(old) and old[cursor] == value, "patch preimage differs")
                        cursor += 1
                if op != b"-":
                    require(added > 0, "patch new count differs")
                    added -= 1
                    if selected:
                        out.append(value)
        if selected:
            require(hunks > 0, "empty closure patch")
            files[before] = b"".join(out + old[cursor:])
    require(seen, "patch did not update the approved closure")
    return files
