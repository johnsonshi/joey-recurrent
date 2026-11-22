"""Assemble an offline OCI image delta. Python >=3.10, POSIX, stdlib only.

Run with python3.14 -IBS scripts/assemble_worker_image.py MODE ... . Both modes
require --base-manifest FILE --base-config FILE --output DIR. Supply **raw**
readback bytes, not inspected/reformatted JSON. The trusted worker-base.json
beside this distribution pins the single-platform manifest, which pins the
config. Neither a base pull nor a registry client is part of this program.

setup --source-root DIR --source src/joey_research/transport.py=HEX [...]
    One small layer. Only explicitly named, independently approved files are
    opened. No capsule, pip, model, input rows, or setup/probe execution.
    Cmd: python -m joey_research.transport when reviewed; otherwise python
    --version. --entrypoint REL explicitly selects a listed module.
    A coordinator can override the command and mount its own private /work.

worker --capsule DIR --pins-sha256 HEX [--pin ARTIFACT=HEX ...]
    DIR is the sealed, materialized capsule (not the run or payload directory).
    HEX comes from the coordinator, never from the transferred pins.json.
    The receipt, discovery, runtime.tar, runtime-files.json and requirements
    must have role-specific pins. If not named in pins.json, supply independent
    --pin values (see worker --help); overrides may not contradict a pin.
    The trusted local staging verifier and Linux wheel lock are loaded ONLY
    from this distribution, never from the capsule. Source is data, not code
    on the host. The approved entrypoint is a member path; if not recorded in
    the pinned inventory, specify it with --entrypoint. A capsule predating
    transport.py can add it with the same --source-root / --source options
    as setup. An additional reviewed file cannot replace a capsule member.

    Four layers: /staged/model; /staged/wheelhouse and requirements;
    /opt/joey/{src,scripts,notices} and /staged/qev; /staged/inputs/discovery.jsonl.
    /staged/requirements-linux.txt has the exact independently pinned bytes,
    usable with cwd /staged. The base Entrypoint is preserved. Cmd is
    ["python","-m","joey_research.transport"]. The coordinator can set this
    command explicitly and pass probe/discovery arguments. Only the admitted
    transport discovery process installs the locked wheels on the worker into
    fresh /work/site; no pip before admission and no second installer. Mount
    a fresh private writable /work per launch. Offline flags do not enforce isolation:
    deny egress externally and probe the target ABI/device before real use.

Output is a *thin* OCI layout, not a standalone pullable archive. Use a
dedicated owner-only directory on a local filesystem with flock, fsync and
atomic rename. Inputs must remain stable; only one writer may use an output.
blobs/sha256/<hex> contains new uncompressed OCI tar layers, config and
manifest. Raw tar avoids both host-dependent compression and a second copy
of the tensor. index.json points at the manifest. publication-plan.json is
written last. On success, cross-mount its cross_mount descriptors from the
verified mirrored base, upload its upload_blobs, then PUT the *raw bytes* of
its manifest with that media type. Read back and verify by digest. Never
publish by scanning blobs/: old immutable objects can remain after a retry.
Re-running revalidates and reuses complete blobs (also after interruption);
only the unfinished scratch object is discarded. No host paths, timestamps,
registry names, packet arguments or discovery content enter the plan.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile


ROOT = Path(__file__).resolve().parent.parent
BASE_LOCK = ROOT / "locks/worker-base.json"
WHEEL_LOCK = ROOT / "locks/linux-inference-candidates.json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"
CONFIG = "application/vnd.oci.image.config.v1+json"
LAYER = "application/vnd.oci.image.layer.v1.tar"
INDEX = "application/vnd.oci.image.index.v1+json"
DOCKER = "application/vnd.docker."
CHUNK = 1024 * 1024
DOC_LIMIT = 8 * CHUNK
RUNTIME_LIMIT = 64 * CHUNK
HEX = re.compile(r"[0-9a-f]{64}")
SOURCE = re.compile(r"src/joey_research/[A-Za-z_][A-Za-z_0-9]*\.py")
SCRIPT = re.compile(r"scripts/[A-Za-z_][A-Za-z_0-9]*\.py")
TRANSPORT = "src/joey_research/transport.py"
WORKER = "src/joey_research/worker.py"
DIR_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
ZERO = bytes(1024)


class AssemblyError(ValueError):
    """Messages must be useful without echoing private paths or data."""


def check(condition, message):
    if not condition:
        raise AssemblyError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hex_pin(value):
    check(isinstance(value, str) and HEX.fullmatch(value), "expected a lowercase SHA-256 pin")
    return value


def encode(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("ascii")


def decode(raw):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            check(k not in out, "duplicate JSON field")
            out[k] = v
        return out

    def invalid(_):
        raise AssemblyError("non-finite JSON number")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as e:
        raise AssemblyError("invalid JSON evidence") from e


def absolute(path):
    path = Path(path)
    check(".." not in path.parts, "parent traversal in local path")
    return path.absolute()


@contextmanager
def directory(path):
    """Pin each ancestor by fd; O_NOFOLLOW on the leaf alone is insufficient."""
    path = absolute(path)
    fd = os.open(path.anchor, DIR_FLAGS)
    try:
        for part in path.parts[1:]:
            child = os.open(part, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def regular(info):
    check(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not info.st_mode & 0o7000,
          "expected a regular single-link file without special mode bits")


@contextmanager
def opened(path):
    with directory(path.parent) as parent:
        with os.fdopen(os.open(path.name, FILE_FLAGS, dir_fd=parent), "rb") as f:
            regular(os.fstat(f.fileno()))
            yield f


def small(path, pin=None, limit=DOC_LIMIT):
    with opened(path) as f:
        n = os.fstat(f.fileno()).st_size
        check(0 < n <= limit, "absent or oversized metadata")
        data = f.read(n + 1)
        check(len(data) == n, "metadata size changed")
    if pin is not None:
        check(sha(data) == hex_pin(pin), "metadata does not match its independent pin")
    return data


def copy_exact(f, size, write=None):
    """Hash exactly size bytes, never read a model, wheel or layer into RAM."""
    h = hashlib.sha256()
    while size:
        chunk = f.read(min(size, CHUNK))
        check(chunk, "truncated materialized file")
        size -= len(chunk)
        h.update(chunk)
        if write is not None:
            write(chunk)
    return h.hexdigest()


def safe_rel(name):
    check(isinstance(name, str) and 0 < len(name) <= 1024 and all(
        re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+-]{0,254}", c) for c in name.split("/")
    ), "unsafe or unsupported member path")
    return name


def descriptor(media, digest, size):
    return {"mediaType": media, "digest": "sha256:" + hex_pin(digest), "size": size}


def validate_descriptor(d, media):
    check(isinstance(d, dict) and set(d) == {"digest", "size", "mediaType"}
          and d["mediaType"] in media and isinstance(d["digest"], str)
          and re.fullmatch(r"sha256:[0-9a-f]{64}", d["digest"])
          and type(d["size"]) is int and d["size"] > 0, "invalid or external base descriptor")


def read_base(manifest_file, config_file, extra_pin):
    lock = decode(small(BASE_LOCK))
    pinned = lock["manifest_digest"]
    check(lock["platform"] == "linux/amd64" and isinstance(pinned, str)
          and re.fullmatch(r"sha256:[0-9a-f]{64}", pinned), "invalid trusted base lock")
    check(extra_pin is None or extra_pin == pinned, "base digest conflicts with trusted lock")
    raw = small(manifest_file, pinned[7:])
    manifest = decode(raw)
    check(isinstance(manifest, dict) and type(manifest.get("schemaVersion")) is int
          and manifest["schemaVersion"] == 2 and manifest.get("mediaType") in (
              MANIFEST, DOCKER + "distribution.manifest.v2+json")
          and "manifests" not in manifest and "subject" not in manifest and "artifactType" not in manifest,
          "expected a raw single-platform image manifest, not an index or artifact")
    validate_descriptor(manifest.get("config"), (CONFIG, DOCKER + "container.image.v1+json"))
    cd = manifest["config"]
    cfg_raw = small(config_file, cd["digest"][7:])
    check(len(cfg_raw) == cd["size"], "base config length mismatch")
    cfg = decode(cfg_raw)
    layers = manifest.get("layers")
    check(isinstance(layers, list) and 0 < len(layers) < 128, "invalid base layers")
    for d in layers:
        validate_descriptor(d, (LAYER, LAYER + "+gzip", LAYER + "+zstd", DOCKER + "image.rootfs.diff.tar.gzip"))
    check(isinstance(cfg, dict) and (cfg.get("os"), cfg.get("architecture")) == ("linux", "amd64")
          and isinstance(cfg.get("config"), dict), "base is not the expected Linux/amd64 image config")
    rootfs = cfg.get("rootfs")
    check(isinstance(rootfs, dict) and rootfs.get("type") == "layers"
          and isinstance(rootfs.get("diff_ids"), list) and len(rootfs["diff_ids"]) == len(layers)
          and all(isinstance(x, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", x) for x in rootfs["diff_ids"]),
          "base descriptors and uncompressed layer IDs do not correspond")
    return descriptor(manifest["mediaType"], pinned[7:], len(raw)), layers, cfg


@dataclass(frozen=True)
class Member:
    size: int
    digest: str
    path: Path | None = None
    offset: int | None = None  # Only for a slice of a verified, uncompressed runtime.tar.
    data: bytes | None = None

    @classmethod
    def inline(cls, data):
        return cls(len(data), sha(data), data=data)

    def emit(self, write):
        if self.data is not None:
            write(self.data)
            return
        with opened(self.path) as f:
            n = os.fstat(f.fileno()).st_size
            if self.offset is None:
                check(n == self.size, "materialized file length changed")
            else:
                check(self.offset + self.size <= n, "runtime member exceeds verified archive")
                f.seek(self.offset)
            check(copy_exact(f, self.size, write) == self.digest, "layer member checksum mismatch")


class TarLayer:
    """Canonical POSIX headers (PAX for long paths/large sizes), no host metadata."""

    def __init__(self, files):
        dirs = set()
        check(0 < len(files) <= 4096, "invalid layer inventory size")
        for name in files:
            for parent in PurePosixPath(safe_rel(name)).parents:
                if str(parent) not in (".", "opt"):
                    dirs.add(str(parent))
        check(dirs.isdisjoint(files), "layer file/directory collision")
        self.entries = []
        self.size = len(ZERO)
        for name in sorted(files.keys() | dirs):
            item = files.get(name)
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE if item is None else tarfile.REGTYPE
            info.mode = 0o755 if item is None else 0o444
            info.size = 0 if item is None else item.size
            header = info.tobuf(format=tarfile.PAX_FORMAT, encoding="ascii", errors="strict")
            padding = -info.size % 512
            self.entries.append((header, item, padding))
            self.size += len(header) + info.size + padding

    def emit(self, write):
        for header, member, padding in self.entries:
            write(header)
            if member is not None:
                member.emit(write)
            write(ZERO[:padding])
        write(ZERO)


def _private_dir(fd):
    s = os.fstat(fd)
    check(s.st_uid == os.getuid() and not s.st_mode & 0o077, "output directories must be owner-only")


def mkdir_at(fd, name):
    try:
        os.mkdir(name, 0o700, dir_fd=fd)
        os.fsync(fd)
    except FileExistsError:
        pass


def names_at(fd, limit):
    with os.scandir(fd) as entries:
        for i, entry in enumerate(entries):
            check(i < limit, "directory exceeds inventory budget")
            yield entry.name


class Layout:
    def __init__(self, root_fd, blobs_fd):
        self.root = root_fd
        self.blobs = blobs_fd
        self.sizes = set()
        for name in names_at(blobs_fd, 8192):
            hex_pin(name)
            s = os.stat(name, dir_fd=blobs_fd, follow_symlinks=False)
            regular(s)
            self.sizes.add(s.st_size)

    @contextmanager
    def pending(self):
        fd = os.open(".pending", os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=self.root)
        try:
            with os.fdopen(fd, "wb") as f:
                yield f
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            os.unlink(".pending", dir_fd=self.root)
            raise

    def reuse(self, digest, size):
        try:
            fd = os.open(digest, FILE_FLAGS, dir_fd=self.blobs)
        except FileNotFoundError:
            return False
        with os.fdopen(fd, "rb") as f:
            info = os.fstat(fd)
            regular(info)
            check(info.st_size == size and copy_exact(f, size) == digest,
                  "corrupt existing content-addressed blob; isolate the damaged layout")
        return True

    def blob(self, media, size, emit):
        expected = None
        # If the blob could already exist, compute its *expected* identity before
        # reserving any disk. Never trust a filename or an unauthenticated resume map.
        if size in self.sizes:
            h = hashlib.sha256()
            emit(h.update)
            expected = h.hexdigest()
            if self.reuse(expected, size):
                return descriptor(media, expected, size)
        h = hashlib.sha256()
        with self.pending() as f:
            def write(block):
                f.write(block)
                h.update(block)

            emit(write)
            check(f.tell() == size and (expected is None or expected == h.hexdigest()), "unstable layer source")
        digest = h.hexdigest()
        # Detect an existing *wrong-size* blob too; never silently replace corruption.
        if self.reuse(digest, size):
            os.unlink(".pending", dir_fd=self.root)
        else:
            os.rename(".pending", digest, src_dir_fd=self.root, dst_dir_fd=self.blobs)
            os.fsync(self.blobs)
        self.sizes.add(size)
        return descriptor(media, digest, size)

    def json_blob(self, media, obj):
        data = encode(obj)
        check(len(data) <= DOC_LIMIT, "generated OCI metadata too large")
        return self.blob(media, len(data), lambda write: write(data))

    def metadata(self, name, obj):
        with self.pending() as f:
            f.write(encode(obj))
        os.replace(".pending", name, src_dir_fd=self.root, dst_dir_fd=self.root)
        os.fsync(self.root)


@contextmanager
def layout(path):
    path = absolute(path)
    with directory(path.parent) as parent:
        mkdir_at(parent, path.name)
    with directory(path) as root:
        _private_dir(root)
        fcntl.flock(root, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in names_at(root, 5):
            if name != "blobs":
                check(name in {"index.json", "oci-layout", "publication-plan.json", ".pending"},
                      "not a dedicated OCI delta directory")
                regular(os.stat(name, dir_fd=root, follow_symlinks=False))
        mkdir_at(root, "blobs")
        with directory(path / "blobs") as blobs:
            _private_dir(blobs)
            check(set(names_at(blobs, 1)) <= {"sha256"}, "unexpected blob store")
            mkdir_at(blobs, "sha256")
            with directory(path / "blobs/sha256") as objects:
                _private_dir(objects)
                store = Layout(root, objects)
                for name in ("publication-plan.json", ".pending"):
                    try:
                        os.unlink(name, dir_fd=root)
                    except FileNotFoundError:
                        pass
                os.fsync(root)
                yield store


def assignments(values):
    out = {}
    for value in values:
        name, sep, digest = value.partition("=")
        check(sep and name not in out, "expected unique NAME=SHA256 arguments")
        out[safe_rel(name)] = hex_pin(digest)
    return out


def command(entry):
    if SOURCE.fullmatch(entry):
        return ["python", "-m", "joey_research." + PurePosixPath(entry).stem]
    check(SCRIPT.fullmatch(entry), "invalid reviewed entrypoint")
    return ["python", "/opt/joey/" + entry]


def reviewed_sources(args):
    files = {}
    sources = assignments(args.source)
    check(len(sources) <= 32 and bool(sources) == (args.source_root is not None),
          "--source-root and explicitly reviewed --source pins are required together")
    for name, digest in sources.items():
        check(SOURCE.fullmatch(name), "only named Joey package modules are admitted for setup")
        path = args.source_root / name
        with opened(path) as f:
            files["opt/joey/" + name] = Member(os.fstat(f.fileno()).st_size, digest, path)
    check(sum(x.size for x in files.values()) <= 4 * CHUNK, "setup package exceeds size budget")
    return files


def setup_files(args):
    files = reviewed_sources(args)
    entry = args.entrypoint or (TRANSPORT if "opt/joey/" + TRANSPORT in files else None)
    check(entry is None or "opt/joey/" + entry in files, "unlisted setup entrypoint")
    cmd = command(entry) if entry else ["python", "--version"]
    return [("setup", files)], cmd


# Role-specific pins, not a search for a matching hash anywhere in the document.
PIN_FIELDS = {
    "receipt.json": ("receipt_sha256",),
    "inputs/discovery.jsonl": ("discovery_sha256",),
    "runtime.tar": ("runtime_sha256", "runtime_tar_sha256", "runtime_archive_sha256"),
    "runtime-files.json": ("runtime_files_sha256", "runtime_manifest_sha256", "runtime_inventory_sha256"),
    "requirements-linux.txt": ("requirements_sha256", "requirements_linux_sha256"),
    "locks/linux-inference-candidates.json": ("wheel_lock_sha256",),
}


def artifact_pins(doc, extra):
    check(isinstance(doc, dict) and extra.keys() <= PIN_FIELDS.keys(), "invalid capsule pin roles")
    pins = {}
    for role, fields in PIN_FIELDS.items():
        found = set()
        for container in (doc, doc.get("files"), doc.get("pins")):
            if isinstance(container, dict):
                for key in (role, *fields):
                    if key in container:
                        v = container[key]
                        found.add(hex_pin(v.get("sha256") if isinstance(v, dict) else v))
        if role in extra:
            found.add(hex_pin(extra[role]))
        check(len(found) == 1, f"missing or conflicting pin; check --pin {role}=SHA256")
        pins[role] = found.pop()
    return pins


def inventory(doc):
    records = doc.get("files", doc.get("members", doc)) if isinstance(doc, dict) else doc
    if isinstance(records, dict):
        pairs = records.items()
    else:
        check(isinstance(records, list), "invalid runtime member inventory")
        pairs = ((r.get("path"), r) for r in records)
    out = {}
    for name, record in pairs:
        safe_rel(name)
        check(name not in out and len(out) < 2048, "duplicate or excessive runtime members")
        h, size = record, None
        if isinstance(record, dict):
            h = record.get("sha256")
            size = record.get("bytes", record.get("size"))
            check(size is None or type(size) is int and 0 <= size <= RUNTIME_LIMIT, "invalid runtime member size")
        out[name] = (hex_pin(h), size)
    return out


def runtime_dest(name):
    if SOURCE.fullmatch(name) or SCRIPT.fullmatch(name):
        return "opt/joey/" + name
    if name.startswith("payload/"):
        name = name[len("payload/"):]
    if name.startswith("src/qev/"):
        name = name[len("src/"):]
    if re.fullmatch(r"qev/[A-Za-z_][A-Za-z_0-9]*\.py", name):
        return "staged/" + name
    check(name.startswith("notices/"), "runtime archive contains a non-source, non-notice member")
    return "opt/joey/" + name


class TarHeaders:
    """tarfile needs only headers/short extensions here. Never honor a giant PAX read."""

    def __init__(self, f):
        self.f = f

    def read(self, n):
        check(0 <= n <= 65536, "oversized runtime tar extension")
        return self.f.read(n)

    def seek(self, offset, whence=os.SEEK_SET):
        return self.f.seek(offset, whence)

    def tell(self):
        return self.f.tell()


def runtime_members(path, pin, expected):
    files = {}
    with opened(path) as f:
        n = os.fstat(f.fileno()).st_size
        check(1024 <= n <= RUNTIME_LIMIT and n % 512 == 0 and copy_exact(f, n) == pin,
              "runtime tar size or independent checksum mismatch")
        f.seek(0)
        seen = set()
        with tarfile.open(fileobj=TarHeaders(f), mode="r:", encoding="ascii", errors="strict") as archive:
            for m in archive:
                name = safe_rel(m.name)
                check(name in expected and name not in seen, "unexpected or duplicate runtime tar member")
                seen.add(name)
                check(m.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and not m.linkname and not m.sparse
                      and not m.mode & ~0o755 and m.mode & 0o400 and m.pax_headers.keys() <= {"path", "size"}
                      and 0 <= m.size <= n - m.offset_data, "invalid tar member type, mode, extent or extension")
                digest, size = expected[name]
                dest = runtime_dest(name)
                check(dest not in files and (size is None or size == m.size), "runtime mapping collision or wrong size")
                f.seek(m.offset_data)
                check(copy_exact(f, m.size) == digest, "runtime layer member checksum mismatch")
                files[dest] = Member(m.size, digest, path, m.offset_data)
            end = archive.offset
        check(seen == expected.keys() and end + 1024 <= n, "runtime inventory or tar terminator incomplete")
        f.seek(end)
        while block := f.read(CHUNK):
            check(not any(block), "unexpected data after runtime tar members")
    return files


def requirement_lines(raw):
    text = raw.decode("ascii").replace("\r\n", "\n")
    check(all(c in "\t\n" or 32 <= ord(c) < 127 for c in text), "invalid requirement text")
    check(not re.search(r"coding[:=]", "\n".join(text.split("\n", 2)[:2])), "unsupported requirement encoding")
    pending = ""
    for line in text.splitlines():
        if "#" in line:
            check(not pending and "\\" not in line, "ambiguous requirement comment or continuation")
            line = re.sub(r"(?:^|\s+)#.*$", "", line)
        if "\\" in line:
            check(not pending and line.count("\\") == 1 and line.endswith((" \\", "\t\\")),
                  "ambiguous requirement continuation")
            pending = line[:-1]
            continue
        yield (pending + line).strip()
        pending = ""
    check(not pending, "incomplete requirement continuation")


def validate_requirements(raw, packages):
    """Bind every requirement to exactly one verified wheel, not a URL or pip option."""
    by_hash = {p["sha256"]: p for p in packages}
    seen = set()
    needs_links = has_links = False
    for line in requirement_lines(raw):
        if line in ("--find-links wheelhouse", "--find-links ./wheelhouse",
                    "--find-links=wheelhouse", "--find-links=./wheelhouse"):
            has_links = True
            continue
        if not line or line in ("--no-index", "--require-hashes", "--only-binary=:all:"):
            continue
        parts = line.split()
        check(len(parts) == 2 and parts[1].startswith("--hash=sha256:"), "expected a hashed local wheel requirement")
        digest = parts[1][len("--hash=sha256:"):]
        check(digest in by_hash and digest not in seen, "requirement is not a unique locked wheel")
        p = by_hash[digest]
        filename = safe_rel(p["filename"])
        check("/" not in filename and filename.endswith(".whl"), "invalid wheel filename")
        spec = parts[0].removeprefix("./")
        pinned_name = re.sub("[-_.]+", "-", p["package"]).lower()
        name, eq, version = spec.partition("==")
        if spec != "wheelhouse/" + filename:
            check(eq and re.sub("[-_.]+", "-", name).lower() == pinned_name and version == p["version"],
                  "requirement does not select its pinned wheel")
            needs_links = True
        seen.add(digest)
    check(seen == by_hash.keys() and seen, "requirements do not cover the entire wheel lock")
    check(not needs_links or has_links, "named requirements require an explicit local wheelhouse")


def detached_licenses(capsule, packages, runtime):
    """Wheel-embedded notices travel intact. Also retain all separately pinned licenses."""
    present = {v.digest for k, v in runtime.items() if k.startswith("opt/joey/notices/")}
    missing = {hex_pin(p["license_sha256"]) for p in packages if "license_sha256" in p} - present
    if not missing:
        return
    # Ingress also materializes detached notices under notices/. Select by
    # trusted license *content*, never copy an arbitrary tree or its pathnames.
    todo = [capsule / "notices"]
    count = total = 0
    while todo and missing:
        path = todo.pop()
        with directory(path) as fd:
            for name in sorted(names_at(fd, 2048)):
                safe_rel(name)
                count += 1
                check(count <= 4096, "detached notice inventory too large")
                s = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(s.st_mode):
                    todo.append(path / name)
                    continue
                regular(s)
                total += s.st_size
                check(s.st_size <= DOC_LIMIT and total <= RUNTIME_LIMIT, "detached notice exceeds budget")
                with opened(path / name) as f:
                    h = copy_exact(f, s.st_size)
                if h in missing:
                    dest = "opt/joey/notices/licenses/" + h
                    check(dest not in runtime, "detached notice path collision")
                    runtime[dest] = Member(s.st_size, h, path / name)
                    missing.remove(h)
    check(not missing, "missing separately pinned upstream license")


def trusted_staging():
    # -IBS keeps cwd, PYTHONPATH, .pth hooks, user site and bytecode out.
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from joey_research import staging
    finally:
        sys.path.pop(0)
    check(Path(staging.__file__).resolve() == ROOT / "src/joey_research/staging.py", "unexpected verifier origin")
    return staging


def worker_files(args, base_digest):
    capsule = absolute(args.capsule)
    out = absolute(args.output)
    check(out != capsule and capsule not in out.parents, "the sealed capsule is not an output directory")
    doc = decode(small(capsule / "pins.json", args.pins_sha256))
    lock_raw = small(WHEEL_LOCK)
    lock_role = "locks/linux-inference-candidates.json"
    extra = assignments(args.pin)
    check(extra.setdefault(lock_role, sha(lock_raw)) == sha(lock_raw), "wheel pin conflicts with trusted lock")
    pins = artifact_pins(doc, extra)
    lock = decode(lock_raw)
    verifier = trusted_staging()
    check(lock.get("base_manifest") == base_digest == verifier.BASE_DIGEST, "wheel or verifier base differs")
    receipt = small(capsule / "receipt.json", pins["receipt.json"])
    root = capsule / "payload"
    # The trusted verifier compares directory listings as sets. Bound those
    # listings even when the transferred tree is not the sealed one we expect.
    with directory(root) as fd:
        check(set(names_at(fd, 5)) == {"model", "qev", "notices", "wheelhouse", "inputs"},
              "unexpected payload root inventory")
    for group in ("model", "qev", "notices", "wheelhouse", "inputs"):
        with directory(root / group) as fd:
            for _ in names_at(fd, 512):
                pass
    try:
        verifier.verify_payload(root, receipt, receipt_sha256=pins["receipt.json"],
                                discovery_sha256=pins["inputs/discovery.jsonl"], wheel_lock_bytes=lock_raw,
                                wheel_lock_sha256=pins[lock_role])
    except (ValueError, OSError, KeyError, TypeError) as e:
        raise AssemblyError("payload refused by trusted coordinator verifier") from e

    listing = decode(small(capsule / "runtime-files.json", pins["runtime-files.json"]))
    expected = inventory(listing)
    runtime = runtime_members(capsule / "runtime.tar", pins["runtime.tar"], expected)
    check(all(f"src/joey_research/{mod}.py" in expected for mod in ("__init__", "evaluation", "native", "staging")),
          "incomplete approved worker source package")
    for dest, reviewed in reviewed_sources(args).items():
        if dest in runtime:
            check((runtime[dest].digest, runtime[dest].size) == (reviewed.digest, reviewed.size),
                  "additional source conflicts with capsule")
            reviewed.emit(lambda _: None)
        else:
            runtime[dest] = reviewed
    entries = {e for e in (doc.get("entrypoint"), listing.get("entrypoint") if isinstance(listing, dict) else None,
                           args.entrypoint) if e is not None}
    check(len(entries) == 1, "missing or conflicting reviewed worker entrypoint; check --entrypoint")
    entry = entries.pop()
    check(entry in expected, "entrypoint is not a reviewed runtime member")
    check(entry in (WORKER, TRANSPORT) and WORKER in expected and "opt/joey/" + TRANSPORT in runtime,
          "approved worker and transport dispatcher required (use --source to add reviewed transport)")
    cmd = command(TRANSPORT)

    model, wheels, discovery = {}, {}, {}
    qev = set()
    for asset in decode(receipt)["assets"]:
        name = safe_rel(asset["path"])
        m = Member(asset["bytes"], hex_pin(asset["sha256"]), root / name)
        if name.startswith("qev/"):
            dest = "staged/" + name
            qev.add(dest)
            check(dest in runtime and (runtime[dest].size, runtime[dest].digest) == (m.size, m.digest),
                  "runtime qev differs from verified payload")
        elif name.startswith("notices/"):
            dest = "opt/joey/" + name
            r = runtime.setdefault(dest, m)
            check((r.size, r.digest) == (m.size, m.digest), "runtime notice differs from verified payload")
        else:
            group = {"model": model, "wheelhouse": wheels, "inputs": discovery}[name.split("/")[0]]
            group["staged/" + name] = m
    check(qev == {k for k in runtime if k.startswith("staged/qev/")}, "unapproved extra qev module")
    detached_licenses(capsule, lock["packages"], runtime)
    requirements = small(capsule / "requirements-linux.txt", pins["requirements-linux.txt"])
    validate_requirements(requirements, lock["packages"])
    wheels["staged/requirements-linux.txt"] = Member.inline(requirements)
    return [("model", model), ("wheels", wheels), ("runtime", runtime), ("discovery", discovery)], cmd


def configure(cfg, cmd, added, mode):
    settings = cfg["config"]
    old = settings.get("Env") or []
    check(isinstance(old, list) and all(isinstance(s, str) and "=" in s for s in old), "invalid base environment")
    paths = ["/opt/joey/src"] + (["/staged", "/work/site"] if mode == "worker" else [])
    for s in old:
        if s.startswith("PYTHONPATH="):
            paths.extend(p for p in s.partition("=")[2].split(":") if p)
    new = dict.fromkeys(("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE",
                         "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK", "PIP_NO_INDEX",
                         "PIP_DISABLE_PIP_VERSION_CHECK", "PYTHONDONTWRITEBYTECODE"), "1")
    new["PYTHONPATH"] = ":".join(dict.fromkeys(paths))
    settings["Env"] = [s for s in old if s.partition("=")[0] not in new] + [f"{k}={v}" for k, v in new.items()]
    settings["Cmd"] = cmd
    cfg["rootfs"]["diff_ids"].extend(d["digest"] for _, d in added)  # Uncompressed OCI layers.
    if cfg.get("history"):
        check(isinstance(cfg["history"], list), "invalid base image history")
        cfg["history"].extend({"created_by": "offline Joey assembly: " + role} for role, _ in added)


def assemble(args):
    base, layers, cfg = read_base(args.base_manifest, args.base_config, args.base_digest)
    groups, cmd = (setup_files(args) if args.mode == "setup" else worker_files(args, base["digest"]))
    check(len(layers) + len(groups) <= 128, "extended image exceeds layer budget")
    with layout(args.output) as store:
        added = []
        for role, files in groups:
            tar = TarLayer(files)
            added.append((role, store.blob(LAYER, tar.size, tar.emit)))
        configure(cfg, cmd, added, args.mode)
        config = store.json_blob(CONFIG, cfg)
        new_layers = [d for _, d in added]
        manifest = store.json_blob(MANIFEST, {"schemaVersion": 2, "mediaType": MANIFEST, "config": config,
                                               "layers": layers + new_layers})
        plan = {"schema": "joey-oci-publication/v1", "mode": args.mode, "base_manifest": base,
                "cross_mount": layers, "upload_blobs": new_layers + [config], "manifest": manifest}
        store.metadata("oci-layout", {"imageLayoutVersion": "1.0.0"})
        store.metadata("index.json", {"schemaVersion": 2, "mediaType": INDEX, "manifests": [manifest]})
        store.metadata("publication-plan.json", plan)
        return plan


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for mode in ("setup", "worker"):
        s = sub.add_parser(mode)
        for name in ("base-manifest", "base-config", "output"):
            s.add_argument("--" + name, type=Path, required=True)
        s.add_argument("--base-digest", help="additional sha256:... pin; must agree with the trusted base lock")
        s.add_argument("--entrypoint", help="exact reviewed member path (not a command string)")
        s.add_argument("--source-root", required=mode == "setup", type=Path)
        s.add_argument("--source", action="append", required=mode == "setup", default=[], metavar="RELPATH=SHA256")
        if mode == "worker":
            s.add_argument("--capsule", required=True, type=Path)
            s.add_argument("--pins-sha256", required=True)
            s.add_argument("--pin", action="append", default=[], metavar="ARTIFACT=SHA256",
                           help="additional independent pin for " + ", ".join(PIN_FIELDS))
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        plan = assemble(args)
    except AssemblyError as e:
        p.exit(2, f"assembly refused: {e}\n")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError, tarfile.TarError, RecursionError) as e:
        p.exit(2, f"assembly refused: invalid evidence or inaccessible local artifact ({type(e).__name__})\n")
    print(plan["manifest"]["digest"])


if __name__ == "__main__":
    main()
