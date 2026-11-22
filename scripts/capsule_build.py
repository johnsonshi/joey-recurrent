"""Assemble data and evidence. No installers, imports of candidate code or subprocesses."""
from __future__ import annotations

import io
import os
import re
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from capsule_io import (Refused, atomic, canonical, directory, document, download, existing, is_sha,
                        regular, require, sha, small_file, sync_dir, trusted_surface, verify_file)
from capsule_net import FLA_LICENSE, HF, ONE, fetch
from capsule_patch import patched
from capsule_policy import NATIVE, PINS, STAGING, WHEEL_LOCK, metadata_url, wheel_url


MiB = 1024**2
NOTICE_FILES = ("model-LICENSE", "model-NOTICE", "OneJev-LICENSE")
SOURCE_NAME = re.compile(r"(?:src/joey_research/(?:__init__|__main__|_?[a-z][a-z0-9_]{0,62})|scripts/[a-z][a-z0-9_]{0,62})\.py")
FORBIDDEN = {"auth", "credentials", "private", "secret", "secrets", "test", "tests", "holdout", "labels"}


def discovery(raw, fields, validate_input=None):
    """Preserve original JSONL bytes. Its identity comes from the approved study export, not a filename."""
    require(isinstance(fields, list) and 2 <= len(fields) <= 32 and
            all(isinstance(k, str) and re.fullmatch("[a-z][a-z0-9_]{0,63}", k) for k in fields)
            and len(set(fields)) == len(fields) and "id" in fields, "invalid discovery field allowlist")
    fields = set(fields)
    seen = set()

    def guard(value):
        if isinstance(value, dict):
            for k, v in value.items():
                name = re.sub("[^a-z0-9]", "", k.lower())
                require(name not in {"label", "labels", "gold", "goldlabel", "goldlabels", "goldanswer",
                                     "answer", "answers", "answerkey", "correct", "correctanswer", "target",
                                     "targets", "truth", "groundtruth", "expected", "expectedanswer", "expectedindex",
                                     "correctindex", "referenceanswer", "solution", "testlabels"},
                        "label-bearing field in discovery export")
                guard(v)
        elif isinstance(value, list):
            for v in value:
                guard(v)

    for line in io.BytesIO(raw):
        row = document(line)
        require(isinstance(row, dict) and row.keys() == fields and isinstance(row.get("id"), str) and row["id"]
                and row["id"] not in seen and len(seen) < 20000, "invalid or duplicate discovery identity")
        require(all(row.get(k, "discovery") == "discovery" for k in ("split", "partition")),
                "non-discovery split refused")
        guard(row)
        if validate_input is not None:
            try:
                valid = is_sha(validate_input(row))
            except Exception:
                valid = False
            require(valid, "study validator refused a discovery input")
        seen.add(row["id"])
    require(seen, "discovery shard is empty")
    return len(seen)


def check_spec(spec):
    fields = {"schema", "serve_patch_sha256", "notices", "discovery_sha256", "discovery_fields", "sources", "entrypoint"}
    require(isinstance(spec, dict) and set(spec) == fields and spec["schema"] == "joey-ingress-handoff/v1"
            and is_sha(spec["serve_patch_sha256"]) and is_sha(spec["discovery_sha256"])
            and isinstance(spec["notices"], dict) and set(spec["notices"]) == set(NOTICE_FILES)
            and all(is_sha(v) for v in spec["notices"].values()), "invalid coordinator handoff")
    manifest = spec["sources"]
    required = {"src/joey_research/__init__.py", "src/joey_research/evaluation.py"}
    require(isinstance(manifest, dict) and 3 <= len(manifest) <= 96 and required <= manifest.keys()
            and isinstance(spec["entrypoint"], str) and spec["entrypoint"] in manifest.keys() - required,
            "reviewed worker entry point, study package and discovery validator required")
    for path, digest in manifest.items():
        require(SOURCE_NAME.fullmatch(path) and not FORBIDDEN.intersection(PurePosixPath(path).stem.split("_"))
                and path not in PINS and is_sha(digest), "unapproved runtime source path or pin")


def export_handoff(*, discovery_bytes, discovery_sha256, validate_input, public_sources, entrypoint, references):
    """Trusted coordinator API: accept bytes already in memory, never open study files or pick a split.

    validate_input is the coordinator's trusted joey_research.evaluation.input_sha256, not imported
    from a staged archive. public_sources maps explicit package paths to (reviewed SHA-256, bytes).
    The only writes are to a new, mode-0700 inbox beside this distribution.
    """
    surface = Path(__file__).resolve().parents[1]
    try:
        return _export(surface, discovery_bytes, discovery_sha256, validate_input, public_sources, entrypoint, references)
    except Refused:
        raise
    except Exception:
        raise Refused("export stopped: invalid input or inbox unavailable") from None


def _export(surface, raw, discovery_pin, validate_input, public_sources, entrypoint, references):
    trusted_surface(surface)
    require(isinstance(raw, bytes) and 0 < len(raw) <= 64 * MiB and is_sha(discovery_pin)
            and sha(raw) == discovery_pin and callable(validate_input), "expected pinned discovery export")
    first = document(next(iter(io.BytesIO(raw))))
    require(isinstance(first, dict) and isinstance(references, dict)
            and set(references) == {"serve_patch_sha256", "notices"}, "invalid export inputs")
    fields = sorted(first)
    spec = {"schema": "joey-ingress-handoff/v1", "discovery_sha256": discovery_pin,
            "discovery_fields": fields, "entrypoint": entrypoint,
            "sources": {p: s for p, (s, _) in public_sources.items()}, **references}
    check_spec(spec)
    discovery(raw, fields, validate_input=validate_input)
    for path, (pin, code) in public_sources.items():
        require(isinstance(code, bytes) and (code or path.endswith("/__init__.py"))
                and len(code) <= MiB and sha(code) == pin, "empty or unpinned public source")
    require(sum(len(code) for _, code in public_sources.values()) <= 16 * MiB, "source package exceeds budget")
    inbox = surface / ".capsule-input"
    inbox.mkdir(mode=0o700)  # No reuse, merging, overwriting or reading a pre-existing inbox.
    for path, (_, code) in public_sources.items():
        parent = inbox
        for part in PurePosixPath(path).parts[:-1]:
            parent = directory(parent / part)
        atomic(inbox / path, code)
    atomic(inbox / "discovery.jsonl", raw)  # Never sort/re-encode question or option order.
    data = canonical(spec)
    atomic(inbox / "handoff.json", data)
    return sha(data)


def read_handoff(surface, pin):
    """Only an independent pin admits a handoff. Never walk a study checkout or dataset directory."""
    require(is_sha(pin), "independent handoff SHA-256 required")
    trusted_surface(surface)
    inbox = directory(surface / ".capsule-input", create=False)
    spec = document(small_file(inbox / "handoff.json", MiB, pin=pin, private=True))
    check_spec(spec)
    raw = small_file(inbox / "discovery.jsonl", 64 * MiB, pin=spec["discovery_sha256"], private=True)
    decisions = discovery(raw, spec["discovery_fields"])
    sources = {}
    for path, digest in spec["sources"].items():
        parent = inbox
        for part in PurePosixPath(path).parts[:-1]:
            parent = directory(parent / part, create=False)
        sources[path] = small_file(inbox / path, MiB, pin=digest, private=True, allow_empty=path.endswith("/__init__.py"))
    require(sum(map(len, sources.values())) <= 16 * MiB, "source package exceeds budget")
    return spec, raw, decisions, sources


def receipt_row(path, digest, size, origin, revision, license):
    return dict(path=path, sha256=digest, bytes=size, source=origin, revision=revision, license=license)


def native_files(net, policy, work, partials, spec):
    payload = directory(work / "payload")
    notices = directory(payload / "notices")
    qev = directory(payload / "qev")
    rows = []
    license_note = "See notices/OneJev-LICENSE and notices/model-LICENSE (checkpoint patch)"
    for name in NOTICE_FILES:
        is_model = name.startswith("model-")
        url = policy.hf(name[6:]) if is_model else policy.one("LICENSE")
        digest = spec["notices"][name]
        count = download(net, url, notices / name, partials, digest, 2 * MiB)
        rows.append(receipt_row("notices/" + name, digest, count, HF if is_model else ONE,
                                policy.model_rev if is_model else policy.one_rev,
                                policy.model_license if is_model else license_note))

    # Keep the reviewed patch only in private ingress scratch, never on the runtime import path.
    patch_file = partials / "serve.patch"
    download(net, policy.hf("serve.patch"), patch_file, partials, spec["serve_patch_sha256"], 4 * MiB)
    diff = small_file(patch_file, 4 * MiB, pin=spec["serve_patch_sha256"], private=True)
    if not all(existing(qev / n) for n in policy.qev):
        originals = {"qev/" + n: fetch(net, policy.one("qev/" + n), 2 * MiB) for n in policy.qev}
        result = patched(originals, diff)
        for name, pin in policy.qev.items():
            require(sha(result["qev/" + name]) == pin, "patched qev postimage differs from native.py")
        for name in policy.qev:
            atomic(qev / name, result["qev/" + name])
    for name, pin in policy.qev.items():
        n = verify_file(qev / name, pin, 2 * MiB)
        rows.append(receipt_row("qev/" + name, pin, n, ONE, policy.one_rev, license_note))
    return rows


def wheels(net, policy, work, partials):
    dest = directory(directory(work / "payload") / "wheelhouse")
    rows = []
    for p in policy.wheels:
        name, pin, size = p["filename"], p["sha256"], p["bytes"]
        if existing(dest / name):
            verify_file(dest / name, pin, size, size)
        else:
            url = wheel_url(p, fetch(net, metadata_url(p), 8 * MiB))
            policy.sources.add(url)  # Only after membership in the locked release is proven.
            download(net, url, dest / name, partials, pin, size, size, accept="application/octet-stream")
        rows.append(receipt_row("wheelhouse/" + name, pin, size, p["source"], p["version"], p["reported_license"]))
    return rows


def wheel_licenses(path):
    """Copy actual notice texts, not wheel code. Never extract a member to a supplied pathname."""
    with regular(path, private=True) as fd, os.fdopen(fd, "rb", closefd=False) as stream, zipfile.ZipFile(stream) as z:
        names = set()
        total = 0
        for info in z.infolist():
            member = PurePosixPath(info.filename)
            if info.is_dir() or not (any(p.lower() in ("licenses", "licences") for p in member.parts[:-1])
                                     or re.match(r"^(?:licen[cs]e|copying|notice|copyright)(?:$|[._-])",
                                                 member.name, re.I)):
                continue
            mode = info.external_attr >> 16
            require(info.filename not in names and len(names) < 1024 and not member.is_absolute()
                    and member.as_posix() == info.filename and all(p not in (".", "..") for p in member.parts)
                    and "\\" not in info.filename and not info.flag_bits & 1
                    and stat.S_IFMT(mode) in (0, stat.S_IFREG), "unsafe wheel notice member")
            names.add(info.filename)
            total += info.file_size
            require(info.file_size <= 8 * MiB and total <= 32 * MiB, "wheel notices exceed budget")
            if info.file_size:
                data = z.read(info)
                require(len(data) == info.file_size, "wheel notice size differs")
                yield info.filename, data


def source_archive(files):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, data in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o444
            member.uid = member.gid = member.mtime = 0
            member.uname = member.gname = ""
            archive.addfile(member, io.BytesIO(data))
    return out.getvalue()


def seal(work, run, files):
    """Reject extra entries without opening them, then publish the read-only tree in a single rename."""
    dirs = {PurePosixPath(".")}
    for name in files:
        dirs.update(PurePosixPath(name).parents)
    paths = dirs | {PurePosixPath(p) for p in files}
    for rel in sorted(dirs, key=lambda p: len(p.parts)):
        d = directory(work / rel, create=False)
        require({p.name for p in d.iterdir()} == {p.name for p in paths if p != rel and p.parent == rel},
                "unexpected or missing capsule entry")
    for name, (pin, size) in files.items():
        verify_file(work / name, pin, size, size)
        with regular(work / name, private=True) as fd:
            os.fchmod(fd, 0o400)
    for rel in sorted(dirs, key=lambda p: -len(p.parts)):
        (work / rel).chmod(0o500)
        sync_dir(work / rel)
    require(not existing(run / "capsule"), "destination already published")
    os.rename(work, run / "capsule")
    sync_dir(run)


def assemble(net, policy, run, handoff, handoff_pin, checkpoint, *, model_file=None):
    spec, shard, count, sources = handoff
    work, parts = directory(run / "work"), directory(run / "partials")
    atomic(run / "handoff.sha256", (handoff_pin + "\n").encode())
    payload = directory(work / "payload")
    atomic(directory(payload / "inputs") / "discovery.jsonl", shard)
    rows = [receipt_row("inputs/discovery.jsonl", spec["discovery_sha256"], len(shard),
                        "study:generated", spec["discovery_sha256"], "Coordinator-approved study discovery export")]
    if model_file is not None:
        rows.extend(checkpoint(net, policy, work, parts, model_file=model_file))
    # Resolve and verify the small source/patch closure before the large network transfers.
    rows.extend(native_files(net, policy, work, parts, spec))
    if model_file is None:
        rows.extend(checkpoint(net, policy, work, parts))
    rows.extend(wheels(net, policy, work, parts))
    require(len(rows) == len({r["path"] for r in rows}) == 59
            and sum(r["bytes"] for r in rows) <= 40 * 2**30, "incomplete or oversized payload")
    files = {"payload/" + r["path"]: (r["sha256"], r["bytes"]) for r in rows}

    def companion(path, data):
        directory((work / path).parent)
        atomic(work / path, data)
        files[path] = (sha(data), len(data))
        return sha(data)

    receipt_pin = companion("receipt.json", canonical({"schema": "joey-discovery-bundle/v1",
                                                        "assets": sorted(rows, key=lambda a: a["path"])}))
    for path in PINS:
        if path.startswith("locks/"):
            companion(path, policy.public[path])
    # The wheel set itself is preserved unchanged. Supplement its embedded notices (including FLA).
    license_index = {}
    budget = 128 * MiB
    for p in policy.wheels:
        notes = {}
        for member, data in wheel_licenses(payload / "wheelhouse" / p["filename"]):
            budget -= len(data)
            require(budget >= 0, "notice collection exceeds budget")
            name = "notices/" + sha(data) + ".txt"
            companion(name, data)
            notes[member] = name
        if "license_sha256" in p:
            pin = p["license_sha256"]
            name = "notices/" + pin + ".txt"
            dest = directory(work / "notices") / (pin + ".txt")
            n = download(net, FLA_LICENSE, dest, parts, pin, 2 * MiB)
            files[name] = (pin, n)
            notes[p["license_source"]] = name
        license_index[p["filename"]] = {"reported_license": p["reported_license"], "notices": notes}
    license_pin = companion("wheel-notices.json", canonical(license_index))
    requirements = "".join(f"./wheelhouse/{p['filename']} --hash=sha256:{p['sha256']}\n"
                           for p in sorted(policy.wheels, key=lambda p: p["filename"]))
    req_pin = companion("requirements-linux.txt", requirements.encode())

    sources = dict(sources) | {p: policy.public[p] for p in (NATIVE, STAGING)}
    for name, pin in policy.qev.items():
        sources["src/qev/" + name] = small_file(payload / "qev" / name, 2 * MiB, pin=pin, private=True)
    for name in NOTICE_FILES:
        sources["notices/" + name] = small_file(payload / "notices" / name, 2 * MiB,
                                                 pin=spec["notices"][name], private=True)
    src_manifest = companion("runtime-files.json", canonical({p: sha(b) for p, b in sources.items()}))
    runtime_pin = companion("runtime.tar", source_archive(sources))
    pins = {"schema": "joey-ingress-pins/v1", "receipt_sha256": receipt_pin, "decisions": count,
            "discovery_sha256": spec["discovery_sha256"], "wheel_lock_sha256": PINS[WHEEL_LOCK],
            "runtime_sha256": runtime_pin, "runtime_files_sha256": src_manifest, "handoff_sha256": handoff_pin,
            "serve_patch_sha256": spec["serve_patch_sha256"], "base_digest": policy.base_digest,
            "runtime_entrypoint": spec["entrypoint"],
            "wheel_notices_sha256": license_pin, "requirements_sha256": req_pin}
    pins_digest = companion("pins.json", canonical(pins))
    seal(work, run, files)
    return {k: pins[k] for k in ("receipt_sha256", "discovery_sha256", "wheel_lock_sha256", "runtime_sha256")} | {
        "pins_sha256": pins_digest}
