from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from joey_research import compact as cp
from joey_research import pass_norms as pn


class SecondVisitNorms(unittest.TestCase):
    def test_byte_exact_vectors_and_independent_base_and_bundle_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); bundle=root/"side"; source=root/"fixture"; width=3
            chunks={k:struct.pack("<3H",0x3f80+i,0xbf00+i,0) for i,(k,_,_) in enumerate(pn._pairs())}
            source.write_bytes(b"".join(chunks.values()))
            records={k:cp.OutputTensor((cp.Tensor(source,i*6,"BF16",(width,),6),)) for i,k in enumerate(chunks)}
            wp=root/"w";cp._write_weights(wp,records,{});payload=wp.read_bytes()
            base={"compact_manifest_sha256":"a"*64,"weight_sha256":"b"*64,"weight_file_bytes":5123,"stored_elements":2500}
            with patch.object(pn,"WIDTH",width):
                m=pn._manifest(base,10000,payload,chunks);encoded=cp._encode(m,pretty=True);h=pn._sha(encoded)
                pn._publish_new(bundle,{pn.MANIFEST:encoded,pn.WEIGHTS:payload})
                self.assertTrue(all(not f.stat().st_mode & 0o077 for f in bundle.iterdir()))
                got,raw=pn.read_vectors(bundle,manifest_sha256=h,compact_manifest_sha256="a"*64)
                self.assertEqual(raw,chunks);self.assertEqual(got["combined_weight_file_bytes"],5123+len(payload))
                self.assertGreater(len(payload),sum(map(len,chunks.values())))
                with self.assertRaises(ValueError):pn.read_vectors(bundle,manifest_sha256=h,compact_manifest_sha256="c"*64)
                with self.assertRaises(FileExistsError):pn._publish_new(bundle,{pn.MANIFEST:encoded,pn.WEIGHTS:payload})
                altered=bytearray(payload);altered[-1]^=1;(bundle/pn.WEIGHTS).write_bytes(altered)
                with self.assertRaises(ValueError):pn.read_vectors(bundle,manifest_sha256=h,compact_manifest_sha256="a"*64)

    def test_malformed_norms_and_unpinned_source_do_not_create_an_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();source=root/"original";base=root/"base";source.mkdir();base.mkdir()
            (source/"config.json").write_text("{}");(source/"model.safetensors").write_bytes(b"unexpected")
            for n in ("config.json","model.safetensors","compact.json"):(base/n).write_bytes(b"fixture")
            with self.assertRaises(ValueError):pn.source_readback(source,base,root/"out",compact_manifest_sha256="f"*64,create=True)
            self.assertFalse((root/"out").exists())
            bad=root/"bad";bad.write_bytes(struct.pack("<H",0x7f80))
            with self.assertRaisesRegex(ValueError,"nonfinite"):pn._vector(cp.Tensor(bad,0,"BF16",(1,),2),1)
            with self.assertRaisesRegex(ValueError,"layout"):pn._vector(cp.Tensor(bad,0,"F16",(1,),2),1)
