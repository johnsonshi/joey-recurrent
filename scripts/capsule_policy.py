"""The public trust roots. Read literals, never import the source to be packaged."""
from __future__ import annotations

import ast
import hashlib
import re
from urllib.parse import unquote

from capsule_io import atomic, canonical, directory, document, is_sha, require, sha, small_file
from capsule_net import FLA_LICENSE, HF, KERNEL, RAW, fetch, parsed


WHEEL_LOCK = "locks/linux-inference-candidates.json"
NATIVE = "src/joey_research/native.py"
STAGING = "src/joey_research/staging.py"
PINS = {
    "locks/model.json": "97cf276fc13f69836cc3841118b1a20fe4f5cce4a76794710072812e03a0af23",
    "locks/worker-base.json": "c6997f731f79768ec0137e02633108bf93398ad301aa621e26de72dbb269b620",
    WHEEL_LOCK: "503b666e08a919fefcd01fb98e23a3916200c3d4f1cba4cf148726c4247e3eea",
    NATIVE: "1e700279aaa709c3a6084cf10316c6e84099639db6e0f753c68cfc2d89fd5e1c",
    STAGING: "76dd7f4b319ff9fa5ff10a814bee299c885efbe15064573f6d7611c67ee7b314",
}


class Policy:
    def __init__(self, surface):
        self.public = {p: small_file(surface / p, 1024 * 1024, pin=s) for p, s in PINS.items()}
        keys = {"MODEL_REVISION", "ONEJEV_REVISION", "MODEL_SHA256", "QEV_SHA256"}
        v = {s.targets[0].id: ast.literal_eval(s.value) for s in ast.parse(self.public[NATIVE]).body
             if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and s.targets[0].id in keys}
        self.model, self.qev = v["MODEL_SHA256"], v["QEV_SHA256"]
        self.model_rev, self.one_rev = v["MODEL_REVISION"], v["ONEJEV_REVISION"]
        model = document(self.public["locks/model.json"])
        base = document(self.public["locks/worker-base.json"])
        wheels = document(self.public[WHEEL_LOCK])
        self.wheels = wheels["packages"]
        require(len(self.model) == 7 and len(self.qev) == 8 and model["revision"] == self.model_rev
                and model["source"] == HF and base["platform"] == "linux/amd64"
                and wheels["kind"] == "candidate-metadata-only" and wheels["resolution_gaps"] == []
                and wheels["base_manifest"] == base["manifest_digest"] and len(self.wheels) == 40,
                "incompatible public contracts")
        self.model_license = "; ".join(model["license"])
        self.base_digest = base["manifest_digest"]
        self.sources = {self.hf(n) for n in (*self.model, "LICENSE", "NOTICE", "serve.patch")}
        self.model_tree = "https://huggingface.co/api/models/baselabs/joey-1/tree/" + self.model_rev
        self.license_meta = "https://api.github.com/repos/OmniJev/OneJev/contents/LICENSE?ref=" + self.one_rev
        self.sources.update(self.one("qev/" + n) for n in self.qev)
        self.sources.update((self.one("LICENSE"), FLA_LICENSE, self.model_tree, self.license_meta))
        self.sources.update(metadata_url(p) for p in self.wheels)

    def hf(self, filename):
        return f"{HF}/resolve/{self.model_rev}/{filename}"

    def one(self, path):
        return f"{RAW}/OmniJev/OneJev/{self.one_rev}/{path}"


def review_refs(net, policy, run):
    """Bootstrap the four small review inputs missing from the public SHA-256 tables."""
    out = directory(run / "review")
    tree = document(fetch(net, policy.model_tree, 2 * 1024**2))
    require(isinstance(tree, list), "unexpected revision metadata")
    meta = document(fetch(net, policy.license_meta, 2 * 1024**2))
    require(meta["type"] == "file" and meta["path"] == "LICENSE"
            and meta["download_url"] == policy.one("LICENSE"), "unexpected license metadata")
    selected = {"OneJev-LICENSE": (meta["sha"], meta["size"], policy.one("LICENSE"))}
    for name in ("serve.patch", "LICENSE", "NOTICE"):
        matches = [x for x in tree if x["path"] == name]
        require(len(matches) == 1 and matches[0]["type"] == "file" and not matches[0].get("lfs"),
                "missing or unsupported reference blob")
        rec = matches[0]
        selected[name if name == "serve.patch" else "model-" + name] = (rec["oid"], rec["size"], policy.hf(name))
    pins = {}
    for name, (oid, size, url) in selected.items():
        require(isinstance(oid, str) and re.fullmatch("[0-9a-f]{40}", oid)
                and type(size) is int and 0 < size <= 4 * 1024**2, "invalid revision blob metadata")
        data = fetch(net, url, size)
        blob = hashlib.sha1(b"blob %d\0" % size + data, usedforsecurity=False).hexdigest()
        require(len(data) == size and blob == oid, "bytes differ from immutable revision metadata")
        atomic(out / name, data)
        pins[name] = sha(data)
    refs = {"serve_patch_sha256": pins.pop("serve.patch"), "notices": pins}
    atomic(out / "reference-pins.json", canonical(refs))
    return refs


def metadata_url(p):
    if p["package"] == "causal-conv1d":
        require(p["source"] == "https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0",
                "unexpected kernel release")
        return KERNEL + "/tags/v1.7.0"
    name, version = p["package"], p["version"]
    require(p["source"] == f"https://pypi.org/project/{name}/{version}/", "unexpected wheel project")
    return f"https://pypi.org/pypi/{name}/{version}/json"


def wheel_url(p, raw):
    """Release metadata locates bytes, but cannot change the independent filename/size/SHA."""
    meta = document(raw)
    filename = p["filename"]
    require(is_sha(p["sha256"]) and re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename),
            "invalid wheel lock entry")
    if p["package"] == "causal-conv1d":
        require(meta["tag_name"] == "v1.7.0" and not meta["draft"], "unexpected kernel release metadata")
        matches = [a for a in meta["assets"] if a["name"] == filename]
        require(len(matches) == 1, "missing or ambiguous kernel asset")
        asset = matches[0]
        require(asset["size"] == p["bytes"] and asset.get("digest") == "sha256:" + p["sha256"]
                and asset["state"] == "uploaded" and type(asset["id"]) is int and asset["id"] > 0,
                "kernel asset metadata differs from lock")
        link = parsed(asset["browser_download_url"])
        require(link.hostname == "github.com" and not link.query and unquote(link.path) ==
                "/Dao-AILab/causal-conv1d/releases/download/v1.7.0/" + filename,
                "unexpected kernel download location")
        # The tag may be moved; the API asset ID cannot be re-used for replacement bytes.
        return KERNEL + "/assets/" + str(asset["id"])
    name = re.sub("[-_.]+", "-", meta["info"]["name"]).lower()
    require(name == p["package"] and meta["info"]["version"] == p["version"],
            "unexpected PyPI release")
    matches = [a for a in meta["urls"] if a["filename"] == filename]
    require(len(matches) == 1, "missing or ambiguous exact wheel")
    asset = matches[0]
    require(asset["size"] == p["bytes"] and asset["digests"]["sha256"] == p["sha256"]
            and asset["packagetype"] == "bdist_wheel" and not asset.get("yanked", False),
            "wheel release metadata differs from lock")
    u = parsed(asset["url"])
    # Warehouse shards its 64-hex BLAKE2b-256 path as 2 / 2 / 60.
    match = re.fullmatch(r"/packages/([a-f0-9]{2})/([a-f0-9]{2})/([a-f0-9]{60})/" + re.escape(filename),
                         unquote(u.path))
    require(u.hostname == "files.pythonhosted.org" and not u.query and
            match and asset["digests"].get("blake2b_256", "".join(match.groups())) == "".join(match.groups()),
            "unexpected wheel distribution URL")
    return asset["url"]
