"""One offline Joey attempt, followed by a bounded immutable loopback export."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, suppress
import ctypes
from dataclasses import dataclass, field
import errno
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import socket
import stat
import subprocess
import sys
import time
from types import MappingProxyType


HOST, PORT = "127.0.0.1", 18765
WORK, STAGED, ADMISSION = Path("/work"), Path("/staged"), Path("/admission")
SOURCE = Path(__file__).resolve().parents[1]
RESERVE = 15
_SCHEMA = "joey-loopback/v1"
_MI = 1024 * 1024
_LIMIT = {"probe.json": 8192, "challenge.bin": 65536, "predictions.jsonl": 64 * _MI,
          "trace.jsonl": 128 * _MI, "complete.json": 64 * 1024}
_FILES = {"setup_probe": ("probe.json", "challenge.bin"),
          "complete": ("predictions.jsonl", "trace.jsonl", "complete.json"),
          "failed": (), "timeout": ()}
_EXIT = {"setup_probe": 0, "complete": 0, "failed": 2, "timeout": 124}
_SHA = re.compile(r"[0-9a-f]{64}", re.ASCII)
_TORCH = re.compile(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}(?:\+(?:cu[0-9]{2,4}|cpu))?")
_CUDA = re.compile(r"[0-9]{1,3}\.[0-9]{1,3}(?:\.[0-9]{1,3})?")
_FILE_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


class TransportError(ValueError):
    """A refusal with no sensitive details."""


def _require(ok):
    if not ok:
        raise TransportError("transport refused")


def _pin(value):
    return type(value) is str and _SHA.fullmatch(value) is not None


def _integer(value, lo, hi):
    return type(value) is int and lo <= value <= hi


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _encoded(obj):
    return json.dumps(obj, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("ascii")


def _unique(pairs):
    obj = {}
    for key, value in pairs:
        _require(key not in obj)
        obj[key] = value
    return obj


def _json(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                      parse_constant=lambda _: _require(False))


def _remaining(end):
    left = end - time.time()
    if left <= 0:
        raise TimeoutError
    return left


@dataclass(frozen=True)
class Attempt:
    mode: str
    nonce: str = field(repr=False)
    admit_before: int
    hard_stop: int
    packet_sha256: str | None = None
    requirements_sha256: str | None = None

    def __post_init__(self):
        _require(self.mode in ("probe", "discovery") and _pin(self.nonce))
        _require(type(self.admit_before) is int and type(self.hard_stop) is int
                 and 0 < self.admit_before <= self.hard_stop < 2**53)
        pins = (self.packet_sha256, self.requirements_sha256)
        _require(all(map(_pin, pins)) if self.mode == "discovery" else pins == (None, None))

    @property
    def nonce_sha256(self):
        return _sha(self.nonce.encode("ascii"))

    @property
    def compute_stop(self):
        return self.hard_stop - RESERVE

    def bindings(self):
        return {"schema": _SCHEMA, "mode": self.mode, "nonce_sha256": self.nonce_sha256,
                "admit_before": self.admit_before, "hard_stop": self.hard_stop,
                "packet_sha256": self.packet_sha256, "requirements_sha256": self.requirements_sha256}


@contextmanager
def _directory(path, parent=None):
    fd = os.open(path, _FILE_FLAGS | os.O_DIRECTORY, dir_fd=parent)
    try:
        yield fd
    finally:
        os.close(fd)


def _read(directory, name, limit):
    fd = os.open(name, _FILE_FLAGS, dir_fd=directory)
    with os.fdopen(fd, "rb") as stream:
        st = os.fstat(fd)
        _require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and 0 < st.st_size <= limit)
        data = stream.read(st.st_size + 1)
        _require(len(data) == st.st_size)
        return data


def _persist(directory, name, data):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(fd)


def _inventory(directory, names):
    remaining = set(names)
    with os.scandir(directory) as entries:
        for entry in entries:
            _require(entry.name in remaining)
            remaining.remove(entry.name)
    _require(not remaining)


def _packet_bytes():
    # Kube's immutable projected ConfigMap normally uses this two-link layout.
    # Never follow a caller-supplied symlink out of the mount.
    with _directory(ADMISSION) as root:
        if stat.S_ISLNK(os.stat("packet.json", dir_fd=root, follow_symlinks=False).st_mode):
            _require(os.readlink("packet.json", dir_fd=root) == "..data/packet.json")
            target = os.readlink("..data", dir_fd=root)
            _require(re.fullmatch(r"\.\.[0-9][A-Za-z0-9_.-]{0,120}", target))
            with _directory(target, root) as projected:
                return _read(projected, "packet.json", 8192)
        return _read(root, "packet.json", 8192)


def _discovery(attempt):
    from .evaluation import input_sha256
    from .worker import _packet

    raw = _packet_bytes()
    _require(_sha(raw) == attempt.packet_sha256)
    _json(raw)  # Reject duplicate keys and non-JSON numeric constants as well.
    packet, _, _ = _packet(raw)
    _require(type(packet["deadline_utc"]) is int and packet["deadline_utc"] == attempt.hard_stop
             and _pin(packet["inputs_sha256"]))
    with _directory(STAGED) as root, _directory("inputs", root) as inputs:
        shard = _read(inputs, "discovery.jsonl", 64 * _MI)
    _require(_sha(shard) == packet["inputs_sha256"])
    rows = {}
    for line in io.BytesIO(shard):
        _remaining(attempt.compute_stop)
        row = _json(line)
        digest = input_sha256(row)  # Exact label-free schema, before starting even pip.
        _require(row["id"] not in rows and len(rows) < 20000 and len(row["choices"]) <= 255)
        rows[row["id"]] = (digest, len(row["choices"]))
    _require(rows)
    return packet, rows


_WHEEL = re.compile(rb"\./wheelhouse/([A-Za-z0-9][A-Za-z0-9._+-]{0,220}\.whl) --hash=sha256:[0-9a-f]{64}")


def _verify_lock(attempt):
    with _directory(STAGED) as root, _directory("wheelhouse", root) as wheels:
        raw = _read(root, "requirements-linux.txt", 128 * 1024)
        _require(_sha(raw) == attempt.requirements_sha256)
        # pip requirements are a language: even --no-index does not forbid a
        # direct URL or an extra -r. Admit only the capsule generator's syntax.
        names = set()
        lines = raw.split(b"\n")
        _require(lines[-1] == b"" and 1 <= len(lines) - 1 <= 256)
        for line in lines[:-1]:
            match = _WHEEL.fullmatch(line)
            _require(match)
            name = match[1].decode("ascii")
            _require(name not in names)
            names.add(name)
            st = os.stat(name, dir_fd=wheels, follow_symlinks=False)
            _require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and st.st_size > 0)
        _inventory(wheels, names)
    # The verified staged volume is read-only. pip --require-hashes verifies
    # the actual wheel bytes it consumes against these exact requirements.


def _challenge(attempt):
    return hashlib.shake_256(b"joey-loopback/challenge/v1\x00" + bytes.fromhex(attempt.nonce)).digest(65536)


def _validate_probe(doc):
    keys = {"device_count", "devices", "python", "torch_version", "cxx11_abi", "cuda"}
    _require(type(doc) is dict and set(doc) == keys)
    _require(_integer(doc["device_count"], 0, 64) and type(doc["devices"]) is list
             and len(doc["devices"]) == doc["device_count"])
    for d in doc["devices"]:
        _require(type(d) is dict and set(d) == {"a100", "vram_bytes", "capability"}
                 and type(d["a100"]) is bool and _integer(d["vram_bytes"], 1, 2**52)
                 and type(d["capability"]) is list and len(d["capability"]) == 2
                 and all(_integer(i, 0, 99) for i in d["capability"]))
    _require(type(doc["python"]) is list and len(doc["python"]) == 2
             and all(_integer(i, 0, 99) for i in doc["python"]))
    _require(type(doc["cxx11_abi"]) is bool and type(doc["torch_version"]) is str
             and _TORCH.fullmatch(doc["torch_version"]) and
             (doc["cuda"] is None or type(doc["cuda"]) is str and _CUDA.fullmatch(doc["cuda"])))


def _probe(torch):
    count = torch.cuda.device_count()
    _require(_integer(count, 0, 64))
    devices = []
    for i in range(count):
        p = torch.cuda.get_device_properties(i)
        devices.append({"a100": bool(re.search(r"\bA100\b", p.name)), "vram_bytes": p.total_memory,
                        "capability": [p.major, p.minor]})
    doc = {"device_count": count, "devices": devices, "python": list(sys.version_info[:2]),
           "torch_version": str(torch.__version__), "cxx11_abi": torch.compiled_with_cxx11_abi(),
           "cuda": torch.version.cuda}
    _validate_probe(doc)
    return doc


def _probe_child():
    import torch  # Only in the supervised, offline process, never in the exporter.
    with _directory(WORK / "run") as run:
        _persist(run, "base.json", _encoded(_probe(torch)))


def _outputs(run, packet, rows, attempt):
    with _directory("out", run) as out:
        _inventory(out, _FILES["complete"])
        files = {name: _read(out, name, _LIMIT[name]) for name in _FILES["complete"]}
    metrics = _json(files["complete.json"])
    _require(_integer(metrics["schema"], 1, 1) and metrics["packet_sha256"] == attempt.packet_sha256
             and _integer(metrics["decisions"], len(rows), len(rows)) and _integer(metrics["batch_size"], 1, 1)
             and metrics["hardware"] == "A100" and type(metrics["parity"]) is dict
             and _integer(metrics["total_device_bytes"], 1, 2**52)
             and all(type(metrics[k]) in (int, float) and math.isfinite(metrics[k]) and metrics[k] > 0
                     for k in ("warmup_load_and_parity_s", "wall_including_persistence_s", "decisions_per_wall_s"))
             and set(metrics["warm_latency_ms"]) == {"50", "95", "99"}
             and metrics.get("compact_manifest_sha256") == packet.get("compact_manifest_sha256"))
    for (key, (digest, count)), p, t in zip(rows.items(), io.BytesIO(files["predictions.jsonl"]),
                                             io.BytesIO(files["trace.jsonl"]), strict=True):
        _remaining(attempt.compute_stop)
        pred, trace = _json(p), _json(t)
        _require(set(pred) == {"id", "probs", "input_sha256"} and pred["id"] == trace["id"] == key
                 and pred["input_sha256"] == digest and type(pred["probs"]) is list
                 and len(pred["probs"]) == count and all(type(x) in (float, int) and 0 <= x <= 1
                                                         for x in pred["probs"])
                 and abs(math.fsum(pred["probs"]) - 1) <= 1e-6
                 and {"tokens", "layer_calls", "readouts", "wall_ms", "cuda_event_ms",
                      "callback_counts"} <= set(trace))
    return files


def _child_env(installed):
    # No parent credentials, service env, pip/user config, bearer nonce, or
    # caller-controlled executable/import/library paths enter the child.
    prefix = Path(sys.executable).parent
    env = {"PATH": f"{prefix}:/usr/local/cuda/bin:/usr/bin:/bin", "HOME": str(WORK),
           "PYTHONPATH": str(SOURCE) + (f":{WORK}/site:{STAGED}" if installed else ""),
           "LD_LIBRARY_PATH": f"{prefix.parent}/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PIP_CONFIG_FILE": os.devnull,
           "TMPDIR": str(WORK / "tmp"), "XDG_CACHE_HOME": str(WORK / "cache"),
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false"}
    for name in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def _linux():
    _require(sys.platform == "linux" and os.uname().machine == "x86_64" and ctypes.sizeof(ctypes.c_void_p) == 8)
    return ctypes.CDLL(None, use_errno=True)


def _hard_alarm(libc, stop):
    # A relative wall timer alone misses forward CLOCK_REALTIME adjustments.
    # A kernel absolute UTC timer and a non-extending elapsed timer cover both
    # clock directions, even during native code or a stalled socket client.
    class Timespec(ctypes.Structure):
        _fields_ = [("sec", ctypes.c_long), ("nsec", ctypes.c_long)]

    class Timer(ctypes.Structure):
        _fields_ = [("interval", Timespec), ("value", Timespec)]

    # The namespace init (often this very interpreter) ignores default-action
    # signals. Install an async-signal-safe *native* handler, not a deferred
    # Python handler. Unacknowledged kernel expiry thus exits nonzero (SIGALRM).
    libc.signal.argtypes = (ctypes.c_int, ctypes.c_void_p)
    libc.signal.restype = ctypes.c_void_p
    _require(libc.signal(signal.SIGALRM, ctypes.cast(libc._exit, ctypes.c_void_p)) != ctypes.c_void_p(-1).value)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM})
    signal.setitimer(signal.ITIMER_REAL, _remaining(stop))
    tid = ctypes.c_void_p()
    _require(libc.timer_create(time.CLOCK_REALTIME, None, ctypes.byref(tid)) == 0)
    _require(libc.timer_settime(tid, 1, ctypes.byref(Timer(Timespec(), Timespec(stop, 0))), None) == 0)


def _child_guard(libc, parent, attempt):
    # Run single-threaded, after setsid and before exec. Linux/amd64 is the
    # only verified base. No libseccomp package, elevated caps or netns needed.
    _require(libc.prctl(1, signal.SIGKILL, 0, 0, 0) == 0 and os.getppid() == parent)  # PDEATHSIG
    _require(libc.prctl(38, 1, 0, 0, 0) == 0)  # NO_NEW_PRIVS

    class Insn(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                    ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Insn))]

    load = lambda offset: Insn(0x20, 0, 0, offset)  # BPF_LD | BPF_W | BPF_ABS
    equal = lambda k, t, f: Insn(0x15, t, f, k)
    deny, allow = Insn(6, 0, 0, 0x50000 | errno.EPERM), Insn(6, 0, 0, 0x7fff0000)
    code = [load(4), equal(0xc000003e, 1, 0), deny, load(0),
            Insn(0x35, 0, 1, 0x40000000), deny]  # exclude non-amd64 and x32 ABIs
    # A child and its descendants cannot leave the killable process group,
    # obtain another process's FDs, or use io_uring as a network syscall path.
    for number in (109, 112, 101, 272, 308, 310, 311, 425, 438):  # setpgid, setsid, ptrace, ...
        code += [equal(number, 0, 1), deny]
    # socket and socketpair: only AF_UNIX for local CUDA/compiler IPC.
    code += [equal(41, 1, 0), equal(53, 0, 3), load(16), equal(socket.AF_UNIX, 1, 0), deny, allow]
    instructions = (Insn * len(code))(*code)
    program = Program(len(code), instructions)
    _require(libc.prctl(22, 2, ctypes.byref(program), 0, 0) == 0)  # SET_SECCOMP, FILTER
    _remaining(attempt.admit_before)
    signal.setitimer(signal.ITIMER_REAL, _remaining(attempt.compute_stop))


def _supervise(child, stop):
    try:
        # Observe without reaping: the PGID cannot be recycled under killpg.
        while not os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT):
            time.sleep(min(0.1, _remaining(stop)))
    finally:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        rc = child.wait()
        # The parent is a subreaper; drain any orphaned compiler/loader
        # descendants in this group before publishing or entering another phase.
        while True:
            try:
                os.waitpid(-child.pid, 0)
            except ChildProcessError:
                break
    _remaining(stop)
    if rc == -signal.SIGALRM:
        raise TimeoutError
    _require(rc == 0)


def _phase(attempt, argv, *, cwd, installed):
    libc = _linux()
    parent = os.getpid()
    _remaining(min(attempt.admit_before, attempt.compute_stop))
    with subprocess.Popen(argv, env=_child_env(installed), cwd=cwd, stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
                          start_new_session=True, preexec_fn=lambda: _child_guard(libc, parent, attempt)) as child:
        _supervise(child, attempt.compute_stop)


def _result(attempt):
    _remaining(min(attempt.admit_before, attempt.compute_stop))
    with _directory(WORK) as work:
        for name in ("run", "tmp"):
            os.mkdir(name, 0o700, dir_fd=work)
        with _directory("run", work) as run:
            if attempt.mode == "probe":
                _phase(attempt, [sys.executable, "-sPB", "-c",
                                 "from joey_research.transport import _probe_child; _probe_child()"],
                       cwd="/", installed=False)
                doc = _json(_read(run, "base.json", _LIMIT["probe.json"]))
                _validate_probe(doc)
                challenge = _challenge(attempt)
                _persist(run, "challenge.bin", challenge)
                os.fsync(run)
                verified = _read(run, "challenge.bin", len(challenge))
                _require(verified == challenge)
                return "setup_probe", {"probe.json": _encoded(dict(doc, nonce_sha256=attempt.nonce_sha256)),
                                       "challenge.bin": verified}

            packet, rows = _discovery(attempt)
            _verify_lock(attempt)
            os.mkdir("site", 0o700, dir_fd=work)
            _phase(attempt, [sys.executable, "-IB", "-m", "pip", "--isolated", "--disable-pip-version-check",
                             "--no-input", "--no-cache-dir", "install", "--no-index", "--no-deps",
                             "--only-binary=:all:", "--require-hashes", "--no-compile", "--target", str(WORK / "site"),
                             "-r", str(STAGED / "requirements-linux.txt")], cwd=STAGED, installed=False)
            argv = [sys.executable, "-sPB", "-m", "joey_research.worker", "--packet", str(ADMISSION / "packet.json"),
                    "--packet-sha256", attempt.packet_sha256, "--model-dir", str(STAGED / "model"),
                    "--inputs", str(STAGED / "inputs/discovery.jsonl"), "--output", str(WORK / "run/out")]
            if "compact_manifest_sha256" in packet and "compact_recipe" not in packet:
                argv += ["--compact-dir", str(STAGED / "compact")]
            _phase(attempt, argv, cwd="/", installed=True)
            return "complete", _outputs(run, packet, rows, attempt)


def _validate_manifest(doc, attempt):
    bindings = attempt.bindings()
    _require(type(doc) is dict and set(doc) == set(bindings) | {"status", "files"}
             and all(type(doc[k]) is type(v) and doc[k] == v for k, v in bindings.items()))
    success = "setup_probe" if attempt.mode == "probe" else "complete"
    status, files = doc["status"], doc["files"]
    _require(status in (success, "failed", "timeout") and type(files) is dict and set(files) == set(_FILES[status]))
    for name, record in files.items():
        _require(type(record) is dict and set(record) == {"bytes", "sha256"} and _pin(record["sha256"])
                 and _integer(record["bytes"], 1, _LIMIT[name]))
    if status == "setup_probe":
        _require(files["challenge.bin"]["bytes"] == 65536)


def _publication(attempt, status, files):
    # All payloads are bounded immutable bytes, not paths/files to reopen on GET.
    _require(all(type(v) is bytes for v in files.values()))
    index = {n: {"bytes": len(v), "sha256": _sha(v)} for n, v in files.items()}
    manifest = dict(attempt.bindings(), status=status, files=index)
    _validate_manifest(manifest, attempt)
    raw = _encoded(manifest)
    routes = {b"/file/" + n.encode("ascii"): (v, index[n]["sha256"]) for n, v in files.items()}
    routes[b"/manifest"] = (raw, _sha(raw))
    return MappingProxyType(routes)


def _header(sock, end):
    raw = bytearray()
    while b"\r\n\r\n" not in raw:
        _require(len(raw) < 8192)
        sock.settimeout(min(3, _remaining(end)))
        chunk = sock.recv(8192 - len(raw))
        _require(chunk)
        raw += chunk
    head, tail = bytes(raw).split(b"\r\n\r\n", 1)
    first, *lines = head.split(b"\r\n")
    _require(len(first) <= 256 and len(lines) <= 64)
    fields = {}
    for line in lines:
        name, sep, value = line.partition(b":")
        key = name.lower()
        _require(sep and re.fullmatch(rb"[a-z0-9-]{1,64}", key)
                 and re.fullmatch(rb"[\x20-\x7e]{0,4096}", value) and key not in fields)
        fields[key] = value.strip(b" ")
    _require(not {b"transfer-encoding", b"expect", b"range", b"upgrade"} & fields.keys())
    return first, fields, tail


def _length(headers, maximum, *, default=None):
    value = headers.get(b"content-length", default)
    _require(value is not None and re.fullmatch(rb"0|[1-9][0-9]{0,9}", value) and int(value) <= maximum)
    return int(value)


def _body(sock, tail, length, end):
    _require(len(tail) <= length)
    raw = bytearray(tail)
    while len(raw) < length:
        sock.settimeout(min(3, _remaining(end)))
        piece = sock.recv(min(256 * 1024, length - len(raw)))
        _require(piece)
        raw.extend(piece)
    return bytes(raw)


def _send(sock, data, end):
    buf = memoryview(data)
    while buf:
        sock.settimeout(min(3, _remaining(end)))
        sent = sock.send(buf[:256 * 1024])
        _require(sent > 0)
        buf = buf[sent:]


def _reply(sock, code, end, data=b"", digest=None, *, json_type=False):
    reason = {200: "OK", 204: "No Content", 400: "Bad Request", 403: "Forbidden", 404: "Not Found"}[code]
    headers = (f"HTTP/1.1 {code} {reason}\r\nConnection: close\r\n"
               "Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n")
    if code != 204:
        headers += f"Content-Length: {len(data)}\r\n"
    if digest is not None:
        typ = "application/json" if json_type else "application/octet-stream"
        headers += f"Content-Type: {typ}\r\nX-SHA256: {digest}\r\n"
    _send(sock, (headers + "\r\n").encode("ascii"), end)
    _send(sock, data, end)


def _request(sock, attempt, routes):
    short_end = min(attempt.hard_stop, time.time() + 3)
    try:
        first, headers, tail = _header(sock, short_end)
        method, target, version = first.split(b" ")
        _require(version in (b"HTTP/1.0", b"HTTP/1.1"))
        if not hmac.compare_digest(headers.get(b"x-joey-nonce", b""), attempt.nonce.encode("ascii")):
            _reply(sock, 403, short_end)
            return False
        if method == b"GET" and target in routes:
            _require(_length(headers, 0, default=b"0") == 0 and not tail)
            data, digest = routes[target]
            _reply(sock, 200, attempt.hard_stop, data, digest,
                   json_type=target == b"/manifest" or target.endswith(b".json"))
            return False
        if method == b"POST" and target == b"/ack":
            ack = _json(_body(sock, tail, _length(headers, 256), short_end))
            _require(type(ack) is dict and set(ack) == {"nonce", "receipt_sha256"}
                     and _pin(ack["nonce"]) and _pin(ack["receipt_sha256"]))
            if (hmac.compare_digest(attempt.nonce, ack["nonce"])
                    and hmac.compare_digest(routes[b"/manifest"][1], ack["receipt_sha256"])):
                _reply(sock, 204, short_end)
                return True
            _reply(sock, 403, short_end)
            return False
        _reply(sock, 404, short_end)
    except (ValueError, RecursionError, TimeoutError):
        with suppress(OSError, TransportError):
            _reply(sock, 400, short_end)
    return False


def _serve(listener, attempt, routes, status):
    while time.time() < attempt.hard_stop:
        listener.settimeout(_remaining(attempt.hard_stop))
        try:
            client, _ = listener.accept()
        except TimeoutError:
            break
        with client, suppress(OSError):
            if _request(client, attempt, routes):
                return _EXIT[status]
    return 124  # Running out of retrieval time is never a successful export.


def _export(attempt):
    try:
        status, files = _result(attempt)
        _remaining(attempt.compute_stop)
    except TimeoutError:
        status, files = "timeout", {}
    except Exception:
        # No exception text, native streams, diagnostics, partials or prior results.
        status, files = ("timeout" if time.time() >= attempt.compute_stop else "failed"), {}
    routes = _publication(attempt, status, files)
    _remaining(attempt.hard_stop)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((HOST, PORT))
        listener.listen(4)
        return _serve(listener, attempt, routes, status)


def _exchange(port, attempt, target, limit, *, ack=None):
    # Numeric IPv4 socket; no URL parser, redirect following, DNS, proxy,
    # ambient auth, urllib opener, or TLS/host configuration.
    method = "GET" if ack is None else "POST"
    head = (f"{method} {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            f"X-Joey-Nonce: {attempt.nonce}\r\nConnection: close\r\n")
    if ack is not None:
        head += f"Content-Length: {len(ack)}\r\nContent-Type: application/json\r\n"
    end = attempt.hard_stop
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(min(3, _remaining(end)))
        sock.connect((HOST, port))
        _send(sock, (head + "\r\n").encode("ascii") + (ack or b""), end)
        line, headers, tail = _header(sock, end)
        if ack is not None:
            _require(line == b"HTTP/1.1 204 No Content" and _length(headers, 0, default=b"0") == 0 and not tail)
            return b""
        _require(line == b"HTTP/1.1 200 OK")
        body = _body(sock, tail, _length(headers, limit), end)
        _require(headers.get(b"x-sha256") == _sha(body).encode("ascii"))
        return body


def _verified_bytes(port, attempt, destination):
    destination = Path(destination).absolute()
    parent = destination.parent
    _require(parent.resolve(strict=True) == parent and destination.name not in ("", ".", ".."))
    stage = "." + destination.name + ".receiving"
    with _directory(parent) as base:
        st = os.fstat(base)
        _require(st.st_uid == os.geteuid() and not st.st_mode & 0o077)
        raw = _exchange(port, attempt, "/manifest", 8192)
        doc = _json(raw)
        _validate_manifest(doc, attempt)
        os.mkdir(stage, 0o700, dir_fd=base)  # Exclusive per-destination writer.
        published = False
        try:
            with _directory(stage, base) as local:
                _persist(local, "manifest.json", raw)
                for name, record in doc["files"].items():
                    data = _exchange(port, attempt, "/file/" + name, record["bytes"])
                    _require(len(data) == record["bytes"] and _sha(data) == record["sha256"])
                    _persist(local, name, data)
                os.fsync(local)
                _require(_read(local, "manifest.json", 8192) == raw)
                for name, record in doc["files"].items():
                    data = _read(local, name, record["bytes"])
                    _require(len(data) == record["bytes"] and _sha(data) == record["sha256"])
                    if name == "challenge.bin":
                        _require(data == _challenge(attempt))
                    elif name == "probe.json":
                        probe = _json(data)
                        _require(probe.pop("nonce_sha256") == attempt.nonce_sha256)
                        _validate_probe(probe)
                    elif name == "complete.json":
                        _require(_json(data)["packet_sha256"] == attempt.packet_sha256)
            try:
                os.stat(destination.name, dir_fd=base, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise TransportError("destination exists")
            os.rename(stage, destination.name, src_dir_fd=base, dst_dir_fd=base)
            published = True
            os.fsync(base)
        finally:
            if not published:
                with _directory(stage, base) as local:
                    for name in ("manifest.json", *doc["files"]):
                        with suppress(FileNotFoundError):
                            os.unlink(name, dir_fd=local)
                os.rmdir(stage, dir_fd=base)
    ack = _encoded({"nonce": attempt.nonce, "receipt_sha256": _sha(raw)})
    _exchange(port, attempt, "/ack", 0, ack=ack)
    return doc


def collect(port, destination, *, mode, nonce, admit_before, hard_stop, packet_sha256=None, requirements_sha256=None):
    """Collect from an already-authorized local forward; no workload API or credentials.

    Use a new destination in a private single-writer directory. Returns the
    verified manifest after local durability and ACK. On ACK loss the published
    directory remains; a second collection cannot overwrite it.
    """
    try:
        attempt = Attempt(mode, nonce, admit_before, hard_stop, packet_sha256, requirements_sha256)
        _require(_integer(port, 1, 65535) and _remaining(hard_stop) <= 3600)
        return _verified_bytes(port, attempt, destination)
    except Exception:
        raise TransportError("collection not confirmed") from None


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise TransportError("invalid launch")


def main(argv=None):
    try:
        parser = _Parser(prog="python -m joey_research.transport", allow_abbrev=False)
        parser.add_argument("mode", choices=("probe", "discovery"))
        parser.add_argument("--admit-before", type=int, required=True)
        parser.add_argument("--hard-stop", type=int, required=True)
        parser.add_argument("--packet-sha256")
        parser.add_argument("--requirements-sha256")
        args = parser.parse_args(argv)
        attempt = Attempt(nonce=os.environ.pop("JOEY_EXPORT_NONCE", ""), **vars(args))
        _require(_remaining(attempt.hard_stop) <= 3600)
        libc = _linux()
        os.umask(0o077)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        _require(libc.prctl(4, 0, 0, 0, 0) == 0)  # Parent nonce/FDs are not ptrace-readable.
        _require(libc.prctl(36, 1, 0, 0, 0) == 0)  # PR_SET_CHILD_SUBREAPER
        _hard_alarm(libc, attempt.hard_stop)
        return _export(attempt)
    except TimeoutError:
        return 124
    except Exception:
        return 2


if __name__ == "__main__":
    sys.exit(main())
