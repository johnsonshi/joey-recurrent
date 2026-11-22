import io
import hashlib
import os
import shutil
import sys
import tarfile
import types
import unittest
import uuid
import zipfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

SURFACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURFACE / "scripts"))
import capsule_io as disk
import capsule_net as web
import capsule_build as build
import prepare_capsule as cli
from capsule_patch import new_files, patched
from capsule_policy import PINS, WHEEL_LOCK, Policy, metadata_url, review_refs, wheel_url


class Response(io.BytesIO):
    def __init__(self, body, status=200, **headers):
        super().__init__(body)
        self.status, self.headers = status, {"Content-Length": str(len(body)), **headers}

    def getheader(self, name, default=None):
        return self.headers.get(name, default)


class FixtureNet:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    @contextmanager
    def get(self, url, *, offset=0, accept=None):
        self.calls.append((url, offset, accept))
        with self.responses.pop(0) as r:
            yield r


class Fixture(unittest.TestCase):
    def setUp(self):
        self.root = SURFACE / (".fixture-" + uuid.uuid4().hex)
        self.root.mkdir(0o700)
        def cleanup():
            for current, _, _ in os.walk(self.root):
                os.chmod(current, 0o700)
            shutil.rmtree(self.root)
        self.addCleanup(cleanup)
        old_mask = os.umask(0o077)
        self.addCleanup(os.umask, old_mask)
        block = patch("socket.socket.connect", side_effect=AssertionError("network disabled in fixture"))
        block.start()
        self.addCleanup(block.stop)
        self.parts = disk.directory(self.root / "parts")
        self.out = self.root / "test.bin"

    def transfer(self, net, buf=b"abcdef", size=6):
        return disk.download(net, "fixture", self.out, self.parts, disk.sha(buf), 16, size)

    def test_interrupted_resume_then_verified_reuse(self):
        net = FixtureNet(Response(b"abc", **{"Content-Length": "6"}),
                         Response(b"def", 206, **{"Content-Range": "bytes 3-5/6"}))
        with self.assertRaises(disk.Refused):
            self.transfer(net)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.transfer(net), 6)
        self.assertEqual(self.transfer(net), 6)
        self.assertEqual([offset for _, offset, _ in net.calls], [0, 3])
        self.assertEqual(self.out.read_bytes(), b"abcdef")
        self.assertEqual(self.out.stat().st_mode & 0o777, 0o600)
        self.assertFalse(list(self.parts.iterdir()))

    def test_wrong_range_refused_and_full_fallback_replaces_prefix(self):
        (self.parts / (disk.sha(b"abcdef") + ".part")).write_bytes(b"ab")
        (self.parts / (disk.sha(b"abcdef") + ".part")).chmod(0o600)
        net = FixtureNet(Response(b"abc", 206, **{"Content-Range": "bytes 0-2/6"}), Response(b"abcdef"))
        with self.assertRaises(disk.Refused):
            self.transfer(net)
        self.transfer(net)
        self.assertEqual(self.out.read_bytes(), b"abcdef")

    def test_bad_hash_or_unsafe_destination_never_promoted(self):
        with self.assertRaises(disk.Refused):
            self.transfer(FixtureNet(Response(b"abcdeg")))
        self.assertFalse(self.out.exists())
        other = self.root / "other"
        other.write_bytes(b"unchanged")
        part = next(self.parts.iterdir())
        part.unlink()
        os.link(other, part)
        with self.assertRaises(disk.Refused):
            self.transfer(FixtureNet())
        self.assertEqual(other.read_bytes(), b"unchanged")
        self.out.symlink_to(other)
        with self.assertRaises(OSError):
            self.transfer(FixtureNet())

    def test_sources_and_redirects_do_not_enable_other_reads(self):
        p = Policy(SURFACE)
        start = p.hf("model.safetensors")
        self.assertNotIn(p.hf("setup.sh"), p.sources)
        self.assertNotIn(p.hf("test.jsonl"), p.sources)
        self.assertTrue(web.redirect_ok(start, "https://cas-bridge.xethub.hf.co/xet-bridge-us/a/b?fixture=1"))
        self.assertTrue(web.redirect_ok(start, "https://us.aws.cdn.hf.co/xet-bridge-us/a/b?fixture=1"))
        self.assertFalse(web.redirect_ok(start, "https://us.aws.cdn.hf.co/other/a/b"))
        self.assertFalse(web.redirect_ok(start, "https://huggingface.co/baselabs/joey-1/resolve/main/NOTICE"))
        from urllib.parse import urlunsplit
        with_userinfo = urlunsplit(("https", "reader@huggingface.co", "/model", "", ""))
        for url in ("http://cas-bridge.xethub.hf.co/x", with_userinfo,
                    "https://huggingface.co:443/model", "https://huggingface.co/%2e%2e/test"):
            with self.assertRaises(disk.Refused):
                web.redirect_ok(start, url)

    def test_model_phase_with_small_fixtures(self):
        p = Policy(SURFACE)
        p.model = {n: disk.sha(n.encode()) for n in p.model}
        names = sorted(p.model, key=lambda n: (n == "model.safetensors", n))
        net = FixtureNet(*(Response(n.encode()) for n in names))
        rows = cli.checkpoint(net, p, self.root, self.parts)
        self.assertEqual([r["path"] for r in rows], ["model/" + n for n in names])
        self.assertEqual([u for u, _, _ in net.calls], [p.hf(n) for n in names])
        self.assertFalse((self.root / "receipt.json").exists())

    def test_explicit_tensor_fails_closed_without_any_get(self):
        p = Policy(SURFACE)
        original = b"pinned tensor fixture"
        p.model["model.safetensors"] = disk.sha(original)
        cache = disk.directory(self.root / "cache")
        src = cache / "model.safetensors"
        dest = self.root / "payload/model/model.safetensors"
        net = FixtureNet()
        for kind in ("missing", "wrong", "symlink", "hardlink"):
            if kind == "wrong":
                disk.atomic(src, b"bad tensor fixture")
            if kind == "symlink":
                disk.atomic(cache / "other", original)
                src.symlink_to(cache / "other")
            if kind == "hardlink":
                os.link(cache / "other", src)
            with self.assertRaises((OSError, disk.Refused)):
                cli.checkpoint(net, p, self.root, self.parts, model_file=src)
            src.unlink(missing_ok=True)
            self.assertFalse(dest.exists())
            self.assertEqual(net.calls, [])

    def test_cache_copy_is_independent_and_rechecks_for_mutation(self):
        data = b"approved tensor"
        src = disk.directory(self.root / "cache") / "model.safetensors"
        disk.atomic(src, data)
        pin = disk.sha(data)
        digest = disk.digest_open
        def change_after_hash(f, limit):
            result = digest(f, limit)
            src.write_bytes(b"x" * len(data))
            return result
        with patch.object(disk, "digest_open", change_after_hash), self.assertRaises(disk.Refused):
            disk.copy_cached_tensor(src, self.out, self.parts, pin, 64)
        self.assertFalse(self.out.exists())
        src.write_bytes(data)
        self.assertEqual(disk.copy_cached_tensor(src, self.out, self.parts, pin, 64), len(data))
        self.assertEqual(self.out.read_bytes(), data)
        self.assertFalse(self.out.samefile(src))
        self.assertEqual(src.stat().st_nlink, 1)
        src.write_bytes(b"different")
        self.assertEqual(self.out.read_bytes(), data)

    def test_complete_partial_needs_no_request_and_416_restarts_safely(self):
        pin = disk.sha(b"abcdef")
        disk.atomic(self.parts / (pin + ".part"), b"abcdef")
        self.assertEqual(self.transfer(FixtureNet()), 6)
        self.out.unlink()
        disk.atomic(self.parts / (pin + ".part"), b"old")
        net = FixtureNet(Response(b"", 416), Response(b"abcdef"))
        self.transfer(net)
        self.assertEqual([o for _, o, _ in net.calls], [3, 0])

    def test_only_one_writer_and_no_unrequested_reuse(self):
        with disk.run_directory(self.root, "fixture", False):
            for resume in (False, True):
                with self.assertRaises(disk.Refused), disk.run_directory(self.root, "fixture", resume):
                    self.fail("second writer")
        with self.assertRaises(disk.Refused), disk.run_directory(self.root, "../other", False):
            self.fail("traversal")


class MapNet:
    def __init__(self, policy):
        self.allowed = policy.sources
        self.data, self.calls = {}, []

    @contextmanager
    def get(self, url, *, offset=0, accept=None):
        if url not in self.allowed:
            raise AssertionError("unexpected origin")
        self.calls.append(url)
        assert not offset
        with Response(self.data[url]) as r:
            yield r


def example_wheel(name):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(zipfile.ZipInfo(name + ".dist-info/licenses/LICENSE"), b"Fixture license\n")
        z.writestr(zipfile.ZipInfo("do_not_import.py"), b"raise RuntimeError\n")
    return out.getvalue()


def blob_oid(b):
    return hashlib.sha1(b"blob %d\0" % len(b) + b, usedforsecurity=False).hexdigest()


def miniature():
    """Same logical 59 paths/provenance as the real lock; no real payloads."""
    p = Policy(SURFACE)
    n = MapNet(p)
    for name in p.model:
        data = b"model fixture " + name.encode()
        n.data[p.hf(name)] = data
        p.model[name] = disk.sha(data)
    for name in p.qev:
        data = b"# " + name.encode() + b"\n"
        n.data[p.one("qev/" + name)] = data
        p.qev[name] = disk.sha(data)
    p.qev["engine.py"] = disk.sha(b"# patched engine\n")
    diff = (b"diff --git a/qev/engine.py b/qev/engine.py\nindex 1234567..7654321 100644\n"
            b"--- a/qev/engine.py\n+++ b/qev/engine.py\n@@ -1 +1 @@\n-# engine.py\n+# patched engine\n"
            b"diff --git a/not-in-closure.py b/not-in-closure.py\n--- a/not-in-closure.py\n"
            b"+++ b/not-in-closure.py\n@@ -1 +1 @@\n-old\n+new\n")
    n.data[p.hf("serve.patch")] = diff
    for name in ("LICENSE", "NOTICE"):
        n.data[p.hf(name)] = b"Public model fixture: " + name.encode()
    n.data[p.one("LICENSE")] = b"OneJev fixture notice"
    n.data[p.model_tree] = disk.canonical([
        {"path": k, "type": "file", "oid": blob_oid(n.data[p.hf(k)]), "size": len(n.data[p.hf(k)])}
        for k in ("serve.patch", "LICENSE", "NOTICE")])
    n.data[p.license_meta] = disk.canonical({"sha": blob_oid(n.data[p.one("LICENSE")]),
                                             "type": "file", "path": "LICENSE", "download_url": p.one("LICENSE"),
                                             "size": len(n.data[p.one("LICENSE")])})
    license_data = b"Separate fixture package license"
    n.data[web.FLA_LICENSE] = license_data
    for w in p.wheels:
        name, ver = w["package"], w["version"]
        body = example_wheel(name)
        w["bytes"], w["sha256"] = len(body), disk.sha(body)
        if "license_sha256" in w:
            w["license_sha256"] = disk.sha(license_data)
        if name == "causal-conv1d":
            location = web.KERNEL + "/assets/42"
            rec = {"name": w["filename"], "size": len(body), "digest": "sha256:" + w["sha256"], "id": 42,
                   "state": "uploaded", "browser_download_url":
                   "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/" + w["filename"]}
            meta = {"tag_name": "v1.7.0", "draft": False, "assets": [rec]}
        else:
            blob = hashlib.blake2b(body, digest_size=32).hexdigest()
            location = f"https://files.pythonhosted.org/packages/{blob[:2]}/{blob[2:4]}/{blob[4:]}/{w['filename']}"
            rec = {"filename": w["filename"], "size": len(body), "digests": {"sha256": w["sha256"], "blake2b_256": blob},
                   "packagetype": "bdist_wheel", "url": location}
            meta = {"info": {"name": name, "version": ver}, "urls": [rec]}
        n.data[metadata_url(w)], n.data[location] = disk.canonical(meta), body
    fixture_lock = disk.document(p.public[WHEEL_LOCK])
    fixture_lock["packages"] = p.wheels
    p.public[WHEEL_LOCK] = disk.canonical(fixture_lock)
    return p, n


def inbox(root, refs):
    dest = disk.directory(root / ".capsule-input")
    src = disk.directory(disk.directory(dest / "src") / "joey_research")
    sources = {"__init__.py": b"", "evaluation.py": b"# fixture validator placeholder, never imported by ingress\n",
               "worker.py": b"raise RuntimeError('ingress must not execute runtime source')\n"}
    for k, v in sources.items():
        with (src / k).open("xb") as f:
            f.write(v)
    shard = disk.canonical({"id": "fixture-1", "state": "public fixture", "split": "discovery"})
    disk.atomic(dest / "discovery.jsonl", shard)
    spec = {"schema": "joey-ingress-handoff/v1", "sources": {"src/joey_research/" + k: disk.sha(v)
                                                             for k, v in sources.items()},
            "entrypoint": "src/joey_research/worker.py", "discovery_sha256": disk.sha(shard),
            "discovery_fields": ["id", "state", "split"], **refs}
    raw = disk.canonical(spec)
    disk.atomic(dest / "handoff.json", raw)
    return disk.sha(raw)


class Assembly(unittest.TestCase):
    setUp = Fixture.setUp

    def test_complete_prepare_is_reproducible_and_matches_public_verifier(self):
        p, net = miniature()
        with disk.run_directory(self.root, "review", False) as run:
            refs = review_refs(net, p, run)
        handoff_pin = inbox(self.root, refs)
        tensor_url = p.hf("model.safetensors")
        cache = disk.directory(self.root / "cache")
        disk.atomic(cache / "model.safetensors", net.data[tensor_url])
        pins = []
        with (patch.object(cli, "SURFACE", self.root), patch.object(cli, "Policy", return_value=p),
              patch.object(cli, "Network", return_value=net), patch.dict(PINS, {WHEEL_LOCK: disk.sha(p.public[WHEEL_LOCK])})):
            for name in ("first", "second"):
                stdout = io.StringIO()
                args = ["prepare", name, "--handoff-sha256", handoff_pin]
                if name == "second":
                    del net.data[tensor_url]
                    args += ["--model-cache", str(cache)]
                with redirect_stdout(stdout):
                    self.assertEqual(cli.main(args), 0)
                pins.append(disk.document(stdout.getvalue().encode()))
        self.assertEqual(pins[0], pins[1])
        self.assertEqual(net.calls.count(tensor_url), 1)
        root = self.root / ".capsules/first/capsule"
        self.assertFalse((root.parent / "work").exists())
        self.assertEqual(root.stat().st_mode & 0o777, 0o500)
        self.assertEqual((root / "receipt.json").stat().st_mode & 0o777, 0o400)
        inventory = disk.document((root / "runtime-files.json").read_bytes())
        with tarfile.open(root / "runtime.tar") as t:
            self.assertEqual(t.getnames(), sorted(inventory))
            for m in t.getmembers():
                self.assertTrue(m.isreg() and m.uid == m.gid == m.mtime == 0)
                self.assertEqual(disk.sha(t.extractfile(m).read()), inventory[m.name])
        self.assertNotIn("setup.sh", "\n".join(net.calls))
        self.assertNotIn("not-in-closure.py", "\n".join(net.calls))
        self.assertEqual(sum(x.startswith("https://files.pythonhosted.org/") for x in set(net.calls)), 39)
        self.assertIn(web.KERNEL + "/assets/42", net.calls)
        self.assertTrue(all(rec["notices"] for rec in disk.document((root / "wheel-notices.json").read_bytes()).values()))
        # Only the two permitted public modules are available here. Stand in for the study's
        # row validator; run the real receipt, wheel, filesystem and digest checks unmodified.
        sys.path.insert(0, str(SURFACE / "src"))
        self.addCleanup(sys.path.remove, str(SURFACE / "src"))
        ev = types.ModuleType("joey_research.evaluation")
        def validate(row):
            self.assertEqual(set(row), {"id", "state", "split"})
            return disk.sha(disk.canonical(row))
        ev.input_sha256 = validate
        with patch.dict(sys.modules, {"joey_research.evaluation": ev}):
            from joey_research import staging
            with patch.object(staging, "MODEL_SHA256", p.model), patch.object(staging, "QEV_SHA256", p.qev):
                args = {k: pins[0][k] for k in ("receipt_sha256", "wheel_lock_sha256", "discovery_sha256")}
                got = staging.verify_payload(root / "payload", (root / "receipt.json").read_bytes(),
                                            wheel_lock_bytes=(root / WHEEL_LOCK).read_bytes(), **args)
                self.assertEqual((got["file_count"], got["decisions"]), (59, 1))

    def test_refuses_labelled_or_unapproved_shards_without_fetching(self):
        for row, fields in [({"id": "x", "state": {"gold_label": "x"}}, ["id", "state"]),
                            ({"id": "x", "split": "test"}, ["id", "split"]),
                            ({"id": "x", "state": "a", "extra": "x"}, ["id", "state"])]:
            with self.assertRaises(disk.Refused):
                build.discovery(disk.canonical(row), fields)
        with self.assertRaises(disk.Refused):
            build.discovery(b'{"id":"x","state":"a","state":"b"}\n', ["id", "state"])
        p, net = miniature()
        with disk.run_directory(self.root, "review", False) as run:
            pin = inbox(self.root, review_refs(net, p, run))
        with (patch.object(cli, "SURFACE", self.root), patch.object(cli, "Policy", return_value=p),
              patch.object(cli, "Network", side_effect=AssertionError) as constructor, redirect_stderr(io.StringIO())):
            self.assertEqual(cli.main(["prepare", "new", "--handoff-sha256", "0" * 64]), 1)
            (self.root / ".capsule-input/discovery.jsonl").write_bytes(b"unexpected")
            self.assertEqual(cli.main(["prepare", "new", "--handoff-sha256", pin]), 1)
            constructor.assert_not_called()
        self.assertFalse((self.root / ".capsules/new").exists())

    def test_failed_patch_or_blob_proof_cannot_publish(self):
        p, net = miniature()
        with disk.run_directory(self.root, "review", False) as run:
            refs = review_refs(net, p, run)
            net.data[p.hf("serve.patch")] += b"unexpected\n"
            with self.assertRaises(disk.Refused):
                review_refs(net, p, run)
        pin = inbox(self.root, refs)
        with disk.run_directory(self.root, "bad", False) as run, self.assertRaises(disk.Refused):
            build.assemble(net, p, run, build.read_handoff(self.root, pin), pin, cli.checkpoint)
        self.assertFalse((run / "capsule").exists())
        self.assertNotIn(p.hf("model.safetensors"), net.calls)

    def test_coordinator_export_validates_before_writing_and_preserves_bytes(self):
        pin = "f" * 64
        refs = {"serve_patch_sha256": pin, "notices": dict.fromkeys(build.NOTICE_FILES, pin)}
        source = {"src/joey_research/" + n: (disk.sha(b), b) for n, b in
                  (("__init__.py", b""), ("evaluation.py", b"# fixture\n"), ("worker.py", b"# fixture entry point\n"))}
        entrypoint = "src/joey_research/worker.py"
        raw = b'{"id":"one","questions":{"q":{"criteria":{"z":"first","a":"second"}}}}\n'
        with self.assertRaises(disk.Refused):
            build._export(self.root, raw, disk.sha(raw), lambda row: None, source, entrypoint, refs)
        for bad in ("../config.py", "src/joey_research/auth.py", "scripts/setup.sh", "src/joey_research/native.py"):
            with self.assertRaises(disk.Refused):
                build._export(self.root, raw, disk.sha(raw), lambda row: pin, source | {bad: (pin, b"")}, entrypoint, refs)
        self.assertFalse((self.root / ".capsule-input").exists())
        seen = []
        def validate(row):
            seen.extend(row["questions"]["q"]["criteria"])
            return pin
        accepted = build._export(self.root, raw, disk.sha(raw), validate, source, entrypoint, refs)
        self.assertEqual(seen, ["z", "a"])
        self.assertEqual(build.read_handoff(self.root, accepted)[1], raw)
        with self.assertRaises(FileExistsError):
            build._export(self.root, raw, disk.sha(raw), validate, source, entrypoint, refs)

    def test_postimage_mismatch_leaves_no_importable_files(self):
        p, net = miniature()
        with disk.run_directory(self.root, "bad", False) as run:
            refs = review_refs(net, p, run)
            net.data[p.one("qev/prompt.py")] += b"# not pinned\n"
            work = disk.directory(run / "work")
            with self.assertRaises(disk.Refused):
                build.native_files(net, p, work, self.parts, refs)
            self.assertEqual(list((work / "payload/qev").iterdir()), [])

    def test_transport_checks_every_hop_and_does_not_forward_ambient_auth(self):
        p = Policy(SURFACE)
        start = p.hf("model.safetensors")
        net = object.__new__(web.Network)
        net.sources, net.tls = p.sources, None
        cdn = "https://cas-bridge.xethub.hf.co/xet-bridge-us/a/b?opaque=fixture"
        with patch.object(web.http.client, "HTTPSConnection") as connect:
            conn = connect.return_value
            conn.getresponse.side_effect = [Response(b"", 302, Location=cdn), Response(b"ok")]
            with net.get(start, offset=2) as r:
                self.assertEqual(r.read(), b"ok")
            for call in conn.request.call_args_list:
                self.assertEqual(set(call.kwargs["headers"]), {"Range", "Accept", "User-Agent", "Accept-Encoding"})
            connect.reset_mock()
            conn.getresponse.side_effect = [Response(b"", 302, Location=cdn),
                                            Response(b"", 302, Location="https://unapproved.invalid/?secret=fixture")]
            with self.assertRaises(disk.Refused) as caught, net.get(start):
                self.fail("followed foreign redirect")
            self.assertNotIn("fixture", str(caught.exception))
            self.assertEqual(connect.call_count, 2)
            connect.reset_mock()
            with self.assertRaises(disk.Refused), net.get(p.hf("setup.sh")):
                self.fail("requested a non-allowlisted source")
            connect.assert_not_called()

    def test_publisher_addition_has_no_upstream_preimage(self):
        diff = (b"diff --git a/qev/readout.py b/qev/readout.py\nnew file mode 100644\n"
                b"index 0000000..0123abc\n@@ -0,0 +1,2 @@\n+# data fixture\n+VALUE = 3\n")
        added, normalized = new_files(diff, {"qev/readout.py"})
        self.assertEqual(added, {"qev/readout.py"})
        self.assertEqual(patched({}, normalized, added), {"qev/readout.py": b"# data fixture\nVALUE = 3\n"})
        with self.assertRaises(disk.Refused):
            patched({"qev/readout.py": b"existing"}, normalized, added)

    def test_patch_checks_context_offsets_and_scoped_paths(self):
        orig = {"qev/engine.py": b"first\nold\nlast"}
        head = b"--- a/qev/engine.py\n+++ b/qev/engine.py\n"
        diff = head + b"@@ -2,2 +2,2 @@ fn\n-old\n+new\n last\n\\ No newline at end of file\n"
        self.assertEqual(patched(orig, diff)["qev/engine.py"], b"first\nnew\nlast")
        for bad in (diff.replace(b"-old", b"-other"), diff.replace(b"+2,2", b"+1,2"),
                    diff.replace(b"a/qev/", b"a/../qev/")):
            with self.assertRaises(disk.Refused):
                patched(orig, bad)

    def test_exact_release_metadata_is_not_a_resolver(self):
        p, net = miniature()
        rec = p.wheels[0]
        meta = disk.document(net.data[metadata_url(rec)])
        for variation in ({"size": rec["bytes"]+1}, {"packagetype": "sdist"}, {"url": "https://other.invalid/x"},
                          {"digests": {"sha256": "0" * 64}}, {"filename": "different.whl"}):
            with self.assertRaises(disk.Refused):
                wheel_url(rec, disk.canonical(dict(meta, urls=[meta["urls"][0] | variation])))
        with self.assertRaises(disk.Refused):
            wheel_url(rec, disk.canonical(dict(meta, urls=meta["urls"] * 2)))
        kernel = next(w for w in p.wheels if w["package"] == "causal-conv1d")
        meta = disk.document(net.data[metadata_url(kernel)])
        meta["assets"][0]["digest"] = "sha256:" + "0"*64
        with self.assertRaises(disk.Refused):
            wheel_url(kernel, disk.canonical(meta))

    def test_seal_rejects_extra_entry_without_publishing(self):
        work = disk.directory(self.root / "work")
        disk.atomic(work / "extra", b"extra")
        with self.assertRaises(disk.Refused):
            build.seal(work, self.root, {})
        self.assertFalse((self.root / "capsule").exists())
