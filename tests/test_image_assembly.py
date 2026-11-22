"""Synthetic OCI readback/streaming tests; never load a model, wheel, or transferred module."""
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import assemble_worker_image as image
finally:
    sys.path.pop(0)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stream_hash(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(65536), b""):
        h.update(block)
    return h.hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return digest(data)


def document(path, obj):
    return put(path, json.dumps(obj, indent=2).encode() + b"\n")


def read_json(path):
    return json.loads(path.read_bytes())


def archive(path, records):
    with tarfile.open(path, "w:") as a:
        for name, data, kind, mode in records:
            m = tarfile.TarInfo(name)
            m.type, m.mode = kind, mode
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                m.linkname = "another.py"
            m.size = len(data)
            a.addfile(m, io.BytesIO(data))


class ImageAssembly(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix=".test-image-", dir=Path.cwd())
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / "out"
        self.capsule = self.root / "capsule"
        self.public = self.root / "reviewed"
        self.base = self.root / "manifest.json"
        self.config = self.root / "config.json"
        self.base_lock = self.root / "base-lock.json"
        self.wheel_lock = self.root / "wheel-lock.json"
        self.mock(image, "BASE_LOCK", self.base_lock)
        self.mock(image, "WHEEL_LOCK", self.wheel_lock)
        self.mock(socket, "socket", side_effect=AssertionError("no network"))
        self.mock(subprocess, "Popen", side_effect=AssertionError("no subprocess or host install"))
        layer = {"mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                 "digest": "sha256:" + digest(b"synthetic compressed layer"), "size": 26}
        cfg = {"architecture": "amd64", "os": "linux", "rootfs": {
            "type": "layers", "diff_ids": ["sha256:" + digest(b"synthetic uncompressed layer")]},
            "config": {"User": "1001", "Entrypoint": ["/base-native-init"], "WorkingDir": "/workspace",
                       "Env": ["PATH=/base/bin", "LD_LIBRARY_PATH=/native/lib", "CUDA_VERSION=test",
                               "HF_HUB_OFFLINE=0", "PYTHONPATH=/base/python"], "Cmd": ["original"]},
            "history": [{"created_by": "native"}]}
        cd = document(self.config, cfg)
        self.base_doc = {"schemaVersion": 2, "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                         "config": {"mediaType": "application/vnd.docker.container.image.v1+json",
                                    "size": self.config.stat().st_size, "digest": "sha256:" + cd}, "layers": [layer]}
        self.pin_base()
        self.source_pin = put(self.public / image.TRANSPORT, b"raise AssertionError('do not import on host')\n")

    def mock(self, obj, name, *args, **kwargs):
        m = patch.object(obj, name, *args, **kwargs)
        self.addCleanup(m.stop)
        return m.start()

    def pin_base(self):
        self.base_digest = "sha256:" + document(self.base, self.base_doc)
        document(self.base_lock, {"platform": "linux/amd64", "manifest_digest": self.base_digest})

    def flags(self, mode, *more):
        flags = [mode, "--base-manifest", str(self.base), "--base-config", str(self.config),
                 "--base-digest", self.base_digest, "--output", str(self.out),
                 "--source-root", str(self.public), "--source", image.TRANSPORT + "=" + self.source_pin]
        if mode == "worker":
            flags += ["--capsule", str(self.capsule), "--pins-sha256", self.pins_digest]
        return flags + list(more)

    def args(self, mode, *more):
        return image.parser().parse_args(self.flags(mode, *more))

    def object(self, descriptor):
        path = self.out / "blobs/sha256" / descriptor["digest"].removeprefix("sha256:")
        with path.open("rb") as f:
            self.assertEqual("sha256:" + stream_hash(f), descriptor["digest"])
        self.assertEqual(path.stat().st_size, descriptor["size"])
        return path

    def readback(self, mode, plan):
        self.assertEqual(read_json(self.out / "publication-plan.json"), plan)
        self.assertEqual(read_json(self.out / "index.json")["manifests"], [plan["manifest"]])
        self.assertEqual(read_json(self.out / "oci-layout"), {"imageLayoutVersion": "1.0.0"})
        self.assertEqual(plan["base_manifest"]["digest"], self.base_digest)
        self.assertEqual(plan["cross_mount"], self.base_doc["layers"])
        self.assertNotIn(str(self.root), json.dumps(plan))
        manifest = read_json(self.object(plan["manifest"]))
        config = read_json(self.object(manifest["config"]))
        old = read_json(self.config)
        added = manifest["layers"][len(plan["cross_mount"]):]
        self.assertEqual(manifest["layers"], self.base_doc["layers"] + added)
        self.assertEqual(plan["upload_blobs"], added + [manifest["config"]])
        self.assertEqual(config["rootfs"]["diff_ids"], old["rootfs"]["diff_ids"] + [d["digest"] for d in added])
        self.assertEqual(config["history"][:1], old["history"])
        self.assertEqual(len(config["history"]), len(manifest["layers"]))
        for name in ("User", "WorkingDir", "Entrypoint"):
            self.assertEqual(config["config"][name], old["config"][name])
        env = dict(s.split("=", 1) for s in config["config"]["Env"])
        self.assertEqual(env["CUDA_VERSION"], "test")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/native/lib")
        self.assertEqual(env["PYTHONPATH"], ("/opt/joey/src:" + ("/staged:/work/site:" if mode == "worker" else "")
                                             + "/base/python"))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["HF_DATASETS_OFFLINE"], "1")
        self.assertEqual(config["config"]["Cmd"], ["python", "-m", "joey_research.transport"])
        current = {d["digest"][7:] for d in plan["upload_blobs"] + [plan["manifest"]]}
        self.assertEqual({p.name for p in (self.out / "blobs/sha256").iterdir()}, current)
        groups = []
        for layer in added:
            files = {}
            with tarfile.open(self.object(layer), "r:") as a:
                for m in a:
                    self.assertEqual((m.uid, m.gid, m.mtime, m.uname, m.gname), (0, 0, 0, "", ""))
                    self.assertEqual(m.mode, 0o755 if m.isdir() else 0o444)
                    if not m.isdir():
                        self.assertTrue(m.isreg())
                        self.assertNotIn(m.name, files)
                        self.assertTrue(all(m.name not in previous for previous in groups))
                        with a.extractfile(m) as stream:
                            files[m.name] = stream_hash(stream)
            groups.append(files)
        return groups

    def worker_fixture(self, tensor_bytes=47):
        """Use the public verifier, with fixture model/qev provenance and a label-free fixture validator."""
        self.verifier = image.trusted_staging()
        payload = self.capsule / "payload"
        tensor = payload / "model/stock.bin"
        tensor.parent.mkdir(parents=True)
        with tensor.open("wb") as f:
            block = bytes(range(256)) * 256
            for _ in range(tensor_bytes // len(block)):
                f.write(block)
            f.write(block[:tensor_bytes % len(block)])
        self.model = tensor
        with tensor.open("rb") as f:
            tensor_digest = stream_hash(f)
        qev = {"qev/__init__.py": b"raise AssertionError('no host import')\n"}
        notices = {"notices/" + n: n.encode() for n in ("model-LICENSE", "model-NOTICE", "OneJev-LICENSE")}
        self.wheelname = "example-1.0-py3-none-any.whl"
        wheeldata = b"opaque synthetic wheel bytes"
        lic = b"separate synthetic upstream license\n"
        self.license_digest = put(self.capsule / "notices/detached/LICENSE", lic)
        self.rows = b'{"id":"fixture","prompt":"Choose.","choices":["a","b"]}\n'
        other = {"model/tokenizer.json": b'{"fixture":true}', "inputs/discovery.jsonl": self.rows,
                 "wheelhouse/" + self.wheelname: wheeldata, **qev, **notices}
        for name, data in other.items():
            put(payload / name, data)
        package = {"package": "example", "filename": self.wheelname, "version": "1.0", "sha256": digest(wheeldata),
                   "bytes": len(wheeldata), "source": "https://pypi.org/project/example/1.0/",
                   "reported_license": "synthetic", "license_sha256": self.license_digest}
        wh = document(self.wheel_lock, {"kind": "candidate-metadata-only", "base_manifest": self.base_digest,
                                        "packages": [package], "resolution_gaps": []})
        self.requirements = f"./wheelhouse/{self.wheelname} --hash=sha256:{digest(wheeldata)}\n".encode()
        rh = put(self.capsule / "requirements-linux.txt", self.requirements)
        self.mock(self.verifier, "BASE_DIGEST", self.base_digest)
        self.mock(self.verifier, "MODEL_SHA256", {"stock.bin": tensor_digest,
                                                  "tokenizer.json": digest(other["model/tokenizer.json"])})
        self.mock(self.verifier, "QEV_SHA256", {"__init__.py": digest(qev["qev/__init__.py"])})
        validate_original = self.verifier._asset

        def fixture_asset(record):
            group, name = record["path"].split("/")
            if group in ("qev", "model", "notices"):
                allowed = {"qev": self.verifier.QEV_SHA256, "model": self.verifier.MODEL_SHA256,
                           "notices": {k.split("/")[1]: digest(v) for k, v in notices.items()}}
                if allowed[group][name] != record["sha256"]:
                    raise ValueError("fixture provenance mismatch")
                return record["path"]
            return validate_original(record)

        def fixture_row(row):
            if set(row) != {"id", "prompt", "choices"}:
                raise ValueError("not a label-free fixture")

        self.mock(self.verifier, "_asset", fixture_asset)
        self.mock(self.verifier, "input_sha256", fixture_row)
        entries = []
        for name in ["model/stock.bin", *other]:
            h = tensor_digest if name == "model/stock.bin" else digest(other[name])
            source = package["source"] if name.startswith("wheelhouse/") else "study:generated"
            entries.append({"path": name, "sha256": h, "bytes": (payload / name).stat().st_size,
                            "source": source, "license": "synthetic", "revision": "1.0"})
        receipt = document(self.capsule / "receipt.json", {"schema": "joey-discovery-bundle/v1", "assets": entries})
        sources = {"src/joey_research/" + m + ".py": ("# approved " + m).encode()
                   for m in ("__init__", "evaluation", "native", "staging", "worker")}
        self.tar_data = {**sources, **qev, **notices}
        self.records = [(k, v, tarfile.REGTYPE, 0o644) for k, v in self.tar_data.items()]
        self.members = {"entrypoint": image.WORKER, "files": [
            {"path": k, "bytes": len(v), "sha256": digest(v)} for k, v in self.tar_data.items()]}
        self.pins = {"receipt_sha256": receipt, "requirements_sha256": rh, "wheel_lock_sha256": wh,
                     "discovery_sha256": digest(self.rows)}
        self.repin_tar()
        self.expected = [{"staged/model/stock.bin": tensor_digest, "staged/model/tokenizer.json":
                          digest(other["model/tokenizer.json"])},
                         {"staged/wheelhouse/" + self.wheelname: digest(wheeldata), "staged/requirements-linux.txt": rh},
                         {"opt/joey/" + k: digest(v) for k, v in {**sources, **notices}.items()},
                         {"staged/inputs/discovery.jsonl": digest(self.rows)}]
        self.expected[2].update({"staged/qev/__init__.py": digest(qev["qev/__init__.py"]),
                                 "opt/joey/notices/licenses/" + self.license_digest: self.license_digest,
                                 "opt/joey/" + image.TRANSPORT: self.source_pin})

    def repin_tar(self):
        archive(self.capsule / "runtime.tar", self.records)
        self.pins["runtime_files_sha256"] = document(self.capsule / "runtime-files.json", self.members)
        with (self.capsule / "runtime.tar").open("rb") as f:
            self.pins["runtime_tar_sha256"] = stream_hash(f)
        self.pins_digest = document(self.capsule / "pins.json", self.pins)

    def test_setup_cli_and_determinism(self):
        self.mock(image, "trusted_staging", side_effect=AssertionError("setup must not load verifier"))
        put(self.public / "src/joey_research/unlisted.py", b"never packaged")
        with patch.object(sys, "argv", ["assemble_worker_image.py", *self.flags("setup")]), \
                redirect_stdout(io.StringIO()) as output:
            image.main()
        plan = read_json(self.out / "publication-plan.json")
        self.assertEqual(output.getvalue().strip(), plan["manifest"]["digest"])
        self.assertEqual(self.readback("setup", plan), [{"opt/joey/" + image.TRANSPORT: self.source_pin}])
        (self.public / image.TRANSPORT).chmod(0o400)
        os.utime(self.public / image.TRANSPORT, (13, 13))
        self.out = self.root / "fresh"
        self.assertEqual(image.assemble(self.args("setup")), plan)

    def test_stream_worker_and_resume_interrupted_blob_set(self):
        self.worker_fixture(9 * image.CHUNK + 37)
        opened = image.opened
        reads = []
        resuming = False
        out = self.out
        case = self

        @contextmanager
        def guarded(path):
            with opened(path) as f:
                if path != self.model:
                    yield f
                    return

                class Bounded:
                    def fileno(self):
                        return f.fileno()

                    def read(self, n=-1):
                        case.assertTrue(0 < n <= image.CHUNK, "unbounded model read")
                        if resuming:
                            case.assertFalse((out / ".pending").exists(), "duplicate model allocation on resume")
                        reads.append(n)
                        return f.read(n)

                yield Bounded()

        self.mock(image, "opened", guarded)
        normal = image.Layout.blob

        def interrupt(store, media, size, emit):
            if reads:  # first model layer made durable, next layer not written
                raise InterruptedError("synthetic power loss")
            return normal(store, media, size, emit)

        with patch.object(image.Layout, "blob", interrupt), self.assertRaises(InterruptedError):
            image.assemble(self.args("worker"))
        self.assertGreater(len(reads), 1)
        self.assertFalse((self.out / "publication-plan.json").exists())
        blob, = (self.out / "blobs/sha256").iterdir()
        identity = blob.stat()
        put(self.out / ".pending", b"incomplete next blob")
        resuming = True
        plan = image.assemble(self.args("worker"))
        self.assertEqual(self.readback("worker", plan), self.expected)
        self.assertEqual((identity.st_ino, identity.st_mtime_ns), (blob.stat().st_ino, blob.stat().st_mtime_ns))
        self.assertEqual(image.assemble(self.args("worker")), plan)
        manifest = self.object(plan["manifest"]).read_bytes()
        config = self.object(json.loads(manifest)["config"]).read_bytes()
        ancestry = self.verifier.verify_image_pair(self.base.read_bytes(), self.config.read_bytes(), manifest, config,
                                                   worker_digest=plan["manifest"]["digest"])
        self.assertEqual(ancestry["added_layers"], 4)

    def test_corrupt_cas_refused_not_overwritten(self):
        plan = image.assemble(self.args("setup"))
        blob = self.object(plan["upload_blobs"][0])
        with blob.open("r+b") as f:
            f.write(b"!")
        with self.assertRaisesRegex(image.AssemblyError, "corrupt existing"):
            image.assemble(self.args("setup"))
        self.assertFalse((self.out / "publication-plan.json").exists())

    def test_runtime_change_preserves_model_wheels_and_discovery(self):
        self.worker_fixture()
        old = image.assemble(self.args("worker"))
        self.source_pin = put(self.public / image.TRANSPORT, b"# separately reviewed update\n")
        new = image.assemble(self.args("worker"))
        self.assertNotEqual(old["manifest"], new["manifest"])
        self.assertEqual(old["upload_blobs"][:2], new["upload_blobs"][:2])
        self.assertNotEqual(old["upload_blobs"][2], new["upload_blobs"][2])
        self.assertEqual(old["upload_blobs"][3], new["upload_blobs"][3])

    def test_requires_raw_single_platform_pinned_base(self):
        self.base.write_bytes(self.base.read_bytes().strip())
        with self.assertRaises(image.AssemblyError):
            image.assemble(self.args("setup"))
        self.pin_base()
        self.config.write_bytes(self.config.read_bytes().strip())
        with self.assertRaises(image.AssemblyError):
            image.assemble(self.args("setup"))
        self.base_doc.update(mediaType=image.INDEX, manifests=[])
        self.pin_base()
        with self.assertRaisesRegex(image.AssemblyError, "single-platform"):
            image.assemble(self.args("setup"))
        self.assertFalse(self.out.exists())

    def test_unlisted_paths_and_links_refused(self):
        for name in ("../escape.py", "src/joey_research/.hidden.py", "src/joey_research/nested/module.py"):
            with self.subTest(name=name), self.assertRaises(image.AssemblyError):
                image.assemble(self.args("setup", "--source", name + "=" + self.source_pin))
        source = self.public / image.TRANSPORT
        source.rename(self.root / "elsewhere")
        source.symlink_to(self.root / "elsewhere")
        with self.assertRaises(OSError):
            image.assemble(self.args("setup"))
        source.unlink()
        os.link(self.root / "elsewhere", source)
        with self.assertRaises(image.AssemblyError):
            image.assemble(self.args("setup"))
        alias = self.root / "alias"
        alias.symlink_to(self.public, target_is_directory=True)
        with self.assertRaises(OSError):
            image.assemble(self.args("setup", "--source-root", str(alias)))
        self.assertFalse(self.out.exists())

    def test_capsule_pins_verifier_and_requirement_refusals(self):
        self.worker_fixture()
        args = self.args("worker")
        args.pins_sha256 = digest(b"unrelated")
        with self.assertRaises(image.AssemblyError):
            image.assemble(args)
        with self.assertRaisesRegex(image.AssemblyError, "conflicting pin"):
            image.assemble(self.args("worker", "--pin", "runtime.tar=" + digest(b"other tar")))
        wheel = self.capsule / "payload/wheelhouse" / self.wheelname
        before = wheel.read_bytes()
        wheel.write_bytes(before[::-1])
        with self.assertRaisesRegex(image.AssemblyError, "trusted coordinator verifier"):
            image.assemble(self.args("worker"))
        wheel.write_bytes(before)
        self.pins["requirements_sha256"] = put(self.capsule / "requirements-linux.txt",
                                               b"--find-links https://example.invalid\n" + self.requirements)
        self.pins_digest = document(self.capsule / "pins.json", self.pins)
        with self.assertRaisesRegex(image.AssemblyError, "local wheel requirement"):
            image.assemble(self.args("worker"))
        self.assertFalse(self.out.exists())

    def test_runtime_members_not_just_an_authenticated_tar(self):
        self.worker_fixture()
        original = self.records[:]
        for name, kind, mode in (("src/joey_research/native.py", tarfile.SYMTYPE, 0o644),
                                 ("src/joey_research/native.py", tarfile.LNKTYPE, 0o644),
                                 ("src/joey_research/native.py", tarfile.REGTYPE, 0o4644),
                                 ("src/../elsewhere.py", tarfile.REGTYPE, 0o644)):
            self.records = [(k, b"" if kind != tarfile.REGTYPE else v, kind, mode)
                            if k == "src/joey_research/native.py" else (k, v, t, m) for k, v, t, m in original]
            if name.startswith("src/.."):
                self.records.append((name, b"ignored?", kind, mode))
            self.repin_tar()
            with self.subTest(name=name, kind=kind, mode=mode), self.assertRaises(image.AssemblyError):
                image.assemble(self.args("worker"))
        self.records = original + [original[0]]
        self.repin_tar()
        with self.assertRaisesRegex(image.AssemblyError, "duplicate runtime"):
            image.assemble(self.args("worker"))
        self.records = original
        self.members["files"][0]["sha256"] = digest(b"different source")
        self.repin_tar()
        with self.assertRaisesRegex(image.AssemblyError, "layer member checksum"):
            image.assemble(self.args("worker"))
        self.assertFalse((self.out / "publication-plan.json").exists())

    def test_labelled_fixture_refused_even_with_consistent_pins(self):
        self.worker_fixture()
        row = json.loads(self.rows)
        row["answer"] = "no labels permitted"
        data = json.dumps(row).encode() + b"\n"
        self.pins["discovery_sha256"] = put(self.capsule / "payload/inputs/discovery.jsonl", data)
        receipt = read_json(self.capsule / "receipt.json")
        asset = next(a for a in receipt["assets"] if a["path"] == "inputs/discovery.jsonl")
        asset.update(sha256=digest(data), bytes=len(data))
        self.pins["receipt_sha256"] = document(self.capsule / "receipt.json", receipt)
        self.pins_digest = document(self.capsule / "pins.json", self.pins)
        with self.assertRaisesRegex(image.AssemblyError, "trusted coordinator verifier"):
            image.assemble(self.args("worker"))
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
