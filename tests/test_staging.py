import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from joey_research import staging as s


def encoded(o):
    b = json.dumps(o, separators=(",", ":")).encode()
    return b, hashlib.sha256(b).hexdigest()


def image(layers, arch="amd64", foreign=False):
    cfg, cid = encoded({"rootfs":{"type":"layers","diff_ids":["sha256:"+v*64 for v in layers]},
                        "os":"linux","architecture":arch})
    desc = [{"digest":"sha256:"+v*64,"size":5,"mediaType":"application/vnd.oci.image.layer.v1.tar+gzip"} for v in layers]
    if foreign:
        desc[-1]["urls"] = ["https://example.invalid/external"]
    manifest, digest = encoded({"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json",
        "config":{"digest":"sha256:"+cid,"mediaType":"application/vnd.oci.image.config.v1+json","size":len(cfg)},"layers":desc})
    return manifest, cfg, "sha256:"+digest


class StagingTest(unittest.TestCase):
    def test_image_pin_platform_and_base_ancestry(self):
        bm,bc,bd = image(["a"])
        with patch.object(s,"BASE_DIGEST",bd):
            wm,wc,wd = image(["a","b"])
            self.assertEqual(s.verify_image_pair(bm,bc,wm,wc,worker_digest=wd)["added_layers"],1)
            for layers,arch,foreign in [(["b","a"],"amd64",False),(["a","b"],"arm64",False),(["a","b"],"amd64",True)]:
                wm,wc,wd = image(layers,arch,foreign)
                with self.assertRaises(s.StageError):
                    s.verify_image_pair(bm,bc,wm,wc,worker_digest=wd)
            with self.assertRaises(s.StageError):
                s.verify_image_pair(bm,bc+b"x",bm,bc,worker_digest=bd)

    def test_sealed_bytes_cannot_substitute_files_or_extra_labels(self):
        blob=b"generated-by-a-test"
        sha=hashlib.sha256(blob).hexdigest()
        with tempfile.TemporaryDirectory() as d, patch.object(s,"MODEL_SHA256",{"config.json":sha}), patch.object(s,"QEV_SHA256",{"engine.py":sha}):
            root=Path(d).resolve()
            items=[]
            for name in ("model/config.json","qev/engine.py","notices/model-LICENSE","notices/model-NOTICE","notices/OneJev-LICENSE","inputs/discovery.jsonl","wheelhouse/sample-1-py3-none-any.whl"):
                data=blob
                if name.startswith(("model/","notices/model-")):
                    rev,source=s.MODEL_REVISION,"https://huggingface.co/baselabs/joey-1"
                elif name.startswith(("qev/","notices/OneJev")):
                    rev,source=s.ONEJEV_REVISION,"https://github.com/OmniJev/OneJev"
                elif name.startswith("inputs/"):
                    rev,source="v1","study:generated"
                    data,discovery_sha=encoded({"id":"r1","group_id":"g1","task":"route","prompt":"Which?","choices":["A","B"]})
                else:
                    rev,source="1.0.0","https://pypi.org/project/sample/"
                p=root/name; p.parent.mkdir(exist_ok=True); p.write_bytes(data)
                items.append({"path":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),
                              "revision":rev,"license":"MIT","source":source})
            receipt,pin=encoded({"schema":"joey-discovery-bundle/v1","assets":items})
            def verify():return s.verify_payload(root,receipt,receipt_sha256=pin,discovery_sha256=discovery_sha)
            outcome=verify()
            self.assertEqual((outcome["file_count"],outcome["decisions"]),(7,1))
            extra=root/"inputs/labels.jsonl";extra.write_bytes(blob)
            with self.assertRaisesRegex(s.StageError,"unexpected"):verify()
            extra.unlink()
            p=root/"inputs/discovery.jsonl";p.write_bytes(b"X"*p.stat().st_size)
            with self.assertRaisesRegex(s.StageError,"checksum"):verify()
            p.unlink();p.symlink_to(root/"model/config.json")
            with self.assertRaises((s.StageError,OSError)):verify()
