"""Local files and resumable byte transfer. Nothing here discovers input paths."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path


class Refused(ValueError):
    """Only use static, non-sensitive messages."""


def require(condition, message):
    if not condition:
        raise Refused(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def document(data):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, "duplicate JSON field")
            result[k] = v
        return result

    def invalid(_):
        raise Refused("non-finite JSON value")

    return json.loads(data.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid)


def real_directory(path):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(),
            "expected a real directory without links")


def trusted_surface(path):
    real_directory(path)
    info = path.stat()
    require(info.st_uid == os.getuid() and not info.st_mode & 0o022,
            "staging surface must be owned by the caller and not writable by others")


def directory(path, *, create=True):
    real_directory(path.parent)
    if create:
        path.mkdir(mode=0o700, exist_ok=True)
    real_directory(path)
    info = path.stat()
    require(info.st_uid == os.getuid() and not info.st_mode & 0o077, "directory must be owner-private")
    return path


@contextmanager
def regular(path, flags=os.O_RDONLY, *, private=False):
    real_directory(path.parent)
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "expected a single-link regular file")
        if private:
            require(info.st_uid == os.getuid() and not info.st_mode & 0o077, "file must be owner-private")
        yield fd
    finally:
        os.close(fd)


def small_file(path, limit, *, pin=None, private=False, allow_empty=False):
    with regular(path, private=private) as fd, os.fdopen(fd, "rb", closefd=False) as f:
        require(os.fstat(fd).st_size <= limit, "file exceeds size limit")
        data = f.read(limit + 1)
    require((data or allow_empty) and len(data) <= limit, "invalid input size")
    require(pin is None or sha(data) == pin, "local file differs from independent pin")
    return data


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def existing(path):
    # A dangling symlink is not a missing file.
    return os.path.lexists(path)


def atomic(path, data):
    """Idempotent immutable file. Named scratch is safe under the exclusive run lock."""
    if existing(path):
        require(small_file(path, len(data), private=True, allow_empty=True) == data, "existing output differs")
        return
    scratch = path.with_name("." + path.name + ".writing")
    with regular(scratch, os.O_CREAT | os.O_WRONLY, private=True) as fd:
        os.ftruncate(fd, 0)  # Only after the regular-file / link checks.
        with os.fdopen(fd, "wb", closefd=False) as f:
            f.write(data)
            f.flush()
        os.fsync(fd)
    os.replace(scratch, path)
    sync_dir(path.parent)


@contextmanager
def run_directory(surface, name, resume):
    require(isinstance(name, str) and re.fullmatch("[a-z0-9][a-z0-9-]{0,47}", name), "invalid run name")
    trusted_surface(surface)
    parent = directory(surface / ".capsules")
    run = parent / name
    if not resume:
        try:
            run.mkdir(mode=0o700)
        except FileExistsError:
            raise Refused("run exists; choose a fresh name or explicitly resume") from None
    directory(run, create=False)
    with regular(run / "lock", os.O_RDWR | os.O_CREAT, private=True) as fd:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refused("run already has an active writer") from None
        require(not existing(run / "capsule"), "run is already published")
        yield run


CHUNK = 1024 * 1024


def digest_open(f, limit):
    h = hashlib.sha256()
    n = 0
    for block in iter(lambda: f.read(CHUNK), b""):
        n += len(block)
        require(n <= limit, "file exceeds size limit")
        h.update(block)
    return h, n


def verify_file(path, pin, limit, size=None):
    with regular(path, private=True) as fd, os.fdopen(fd, "rb", closefd=False) as f:
        n = os.fstat(fd).st_size
        require(0 < n <= limit and size in (None, n), "existing artifact size differs")
        h, n = digest_open(f, limit)
        require(h.hexdigest() == pin, "existing artifact checksum differs")
    return n


def copy_cached_tensor(source, dest, partials, pin, limit):
    """Read one explicitly designated stock file. Never search the cache or link it into a run."""
    source = Path(source)
    require(source.name == "model.safetensors" and ".." not in source.parts, "expected an explicit stock tensor path")
    source = source.absolute()
    real_directory(source.parent)
    for parent in source.parents:
        info = parent.stat()
        require(info.st_uid == os.getuid() and not info.st_mode & 0o022, "tensor cache must be user-controlled")
        if not info.st_mode & 0o077:
            break
    else:
        raise Refused("tensor cache requires a private staging directory")

    scratch = partials / (pin + ".cache-copy")
    with regular(source) as fd, os.fdopen(fd, "rb", closefd=False) as src:
        st = os.fstat(fd)
        require(st.st_uid == os.getuid() and not st.st_mode & 0o022 and 0 < st.st_size <= limit,
                "unsafe cached tensor ownership, permissions or size")
        h, n = digest_open(src, limit)
        require(n == st.st_size and h.hexdigest() == pin, "cached tensor differs from native.py")
        if existing(dest):
            return verify_file(dest, pin, limit, n)
        # Pin the source before copying; hash the copy too, in case the producer changes the open inode.
        src.seek(0)
        copied, check = 0, hashlib.sha256()
        with regular(scratch, os.O_WRONLY | os.O_CREAT, private=True) as out:
            os.ftruncate(out, 0)
            with os.fdopen(out, "wb", closefd=False) as dst:
                for block in iter(lambda: src.read(CHUNK), b""):
                    copied += len(block)
                    require(copied <= n, "cached tensor changed during copy")
                    check.update(block)
                    dst.write(block)
                dst.flush()
            if copied != n or check.hexdigest() != pin:
                os.ftruncate(out, 0)
                raise Refused("cached tensor changed during copy")
            os.fsync(out)
    os.replace(scratch, dest)
    sync_dir(dest.parent)
    return n


def length(response):
    text = response.getheader("Content-Length")
    if text is None:
        return None
    require(re.fullmatch(r"[0-9]{1,16}", text), "invalid HTTP length")
    return int(text)


def identity(response):
    require(response.getheader("Content-Encoding", "identity").lower() == "identity",
            "encoded transfer refused")


def download(net, url, dest, partials, pin, limit, size=None, *, accept=None):
    """Only a verified, whole object can leave partials/. Re-hash the prefix on every retry."""
    require(is_sha(pin) and 0 < (size or limit) <= limit, "invalid download pin or size")
    if existing(dest):
        return verify_file(dest, pin, limit, size)
    part = partials / (pin + ".part")
    with regular(part, os.O_CREAT | os.O_RDWR, private=True) as fd, os.fdopen(fd, "r+b", closefd=False) as f:
        require(os.fstat(fd).st_size <= limit, "oversized partial")
        h, n = digest_open(f, limit)
        if h.hexdigest() != pin or n == 0 or size not in (None, n):
            if n == (size or limit):
                f.seek(0)
                f.truncate()
                h, n = hashlib.sha256(), 0
            for attempt in range(2):
                with net.get(url, offset=n, accept=accept) as r:
                    if r.status == 416 and n and attempt == 0:
                        f.seek(0)
                        f.truncate()
                        h, n = hashlib.sha256(), 0
                        continue
                    require(r.status in (200, 206), "artifact GET refused")
                    identity(r)
                    count = length(r)
                    if r.status == 206:
                        m = re.fullmatch(r"bytes ([0-9]{1,16})-([0-9]{1,16})/([0-9]{1,16})",
                                         r.getheader("Content-Range", ""))
                        require(m is not None and n > 0, "unexpected partial response")
                        start, end, total = map(int, m.groups())
                        require(start == n <= end == total - 1 and count in (None, total - start),
                                "range does not describe the requested tail")
                    else:
                        require(r.getheader("Content-Range") is None, "unexpected range on full response")
                        f.seek(0)
                        f.truncate()
                        h, n = hashlib.sha256(), 0
                        total = count
                    require(total is None or (0 < total <= limit and size in (None, total)),
                            "advertised size differs")
                    bound = size or total or limit
                    for block in iter(lambda: r.read(CHUNK), b""):
                        n += len(block)
                        require(n <= bound, "oversized artifact transfer")
                        f.write(block)
                        h.update(block)
                    require(n > 0 and total in (None, n) and size in (None, n), "incomplete transfer; resume")
                    if h.hexdigest() != pin:
                        f.seek(0)
                        f.truncate()
                        raise Refused("artifact checksum mismatch; partial discarded")
                    break
        f.flush()
        os.fsync(fd)
    os.replace(part, dest)
    sync_dir(dest.parent)
    return n
