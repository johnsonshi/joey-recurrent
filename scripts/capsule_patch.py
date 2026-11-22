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


def new_files(patch, allowed):
    """Identify pinned new-file sections, including the publisher's headerless unified hunk."""
    lines = patch.splitlines(keepends=True)
    additions = set()
    normalized = []
    for i, line in enumerate(lines):
        normalized.append(line)
        if not line.startswith(b"new file mode "):
            continue
        m = re.fullmatch(rb"diff --git a/([A-Za-z0-9_/.-]+) b/\1\n", lines[i-1]) if i else None
        require(m and line == b"new file mode 100644\n" and i+2 < len(lines),
                "unsupported new-file patch")
        name = m[1].decode("ascii")
        if name not in allowed:
            continue
        require(name not in additions and re.fullmatch(rb"index 0+\.\.[a-f0-9]+\n", lines[i+1]),
                "invalid new-file preimage")
        additions.add(name)
        if lines[i+2].startswith(b"@@"):
            normalized.extend((b"--- /dev/null\n", b"+++ b/"+m[1]+b"\n"))
    # Keep any reconstructed header immediately before its hunk, after the index line.
    for i, line in enumerate(normalized):
        if line == b"--- /dev/null\n" and i+2 < len(normalized) and normalized[i+2].startswith(b"index "):
            normalized[i:i+3] = [normalized[i+2],normalized[i],normalized[i+1]]
    return additions,b"".join(normalized)


def patched(originals, patch, additions=()):
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
        created = before is None and after in additions
        selected = before in files or after in files or created
        key = after if created else before
        if selected:
            require((created and after not in files or before == after) and key not in seen, "unsupported closure rename, deletion or repeat")
            seen.add(key)
        # Parse counts even for excluded sections so hunk text cannot be mistaken for a filename.
        old = files[before].splitlines(keepends=True) if selected and not created else []
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
            files[key] = b"".join(out + old[cursor:])
    require(seen, "patch did not update the approved closure")
    return files
