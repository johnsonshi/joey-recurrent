"""Generated Joey-shaped weights only. The small readout below is a serialization witness, not Joey."""

import copy
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

from joey_research.compact import CompactError, export_compact, layer_schedule, measure_text_storage, verify_compact


ROOT = Path(__file__).resolve().parents[1]
SELECT = ROOT / "configs" / "compact" / "select-16.json"
MERGE = ROOT / "configs" / "compact" / "merge-16.json"
PREFIX = "model.language_model.layers."
EMBED = "model.language_model.embed_tokens.weight"
HEAD = "lm_head.weight"
NORM = "model.language_model.norm.weight"


def config():
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
        "joey_generated_fixture": True, "tie_word_embeddings": True,
        "vision_config": {"model_type": "qwen3_5_vision", "depth": 1},
        "modeling_overrides": {"attention": {"full_attention": "block_bidirectional",
                                             "linear_attention": "causal"}, "readout": "slot_lm"},
        "text_config": {
            "model_type": "qwen3_5_text", "num_hidden_layers": 24, "mlp_only_layers": [],
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 6,
            "attn_output_gate": True, "attention_bias": False, "full_attention_interval": 4,
            "hidden_size": 2, "intermediate_size": 3, "vocab_size": 7, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 1, "linear_conv_kernel_dim": 2,
            "linear_num_key_heads": 1, "linear_key_head_dim": 2,
            "linear_num_value_heads": 2, "linear_value_head_dim": 1, "tie_word_embeddings": True,
        },
    }


def encode(values, dtype):
    if dtype == "BF16":
        out = bytearray()
        for v in values:
            word = int.from_bytes(struct.pack("<f", v), "little")
            hi, lo = divmod(word, 65536)
            hi += lo > 32768 or (lo == 32768 and hi % 2)
            out.extend(hi.to_bytes(2, "little"))
        return bytes(out)
    return struct.pack(f"<{len(values)}" + {"F16": "e", "F32": "f"}[dtype], *values)


def decode(record):
    dtype, _, data = record
    if dtype == "BF16":
        return tuple(struct.unpack("<f", b"\0\0" + data[i:i+2])[0] for i in range(0, len(data), 2))
    return tuple(v for (v,) in struct.iter_unpack({"F16": "<e", "F32": "<f"}[dtype], data))


def parameters(dtype="BF16", head="equal"):
    common = {
        "input_layernorm.weight": (2,), "post_attention_layernorm.weight": (2,),
        "mlp.down_proj.weight": (2, 3), "mlp.up_proj.weight": (3, 2), "mlp.gate_proj.weight": (3, 2),
    }
    linear = {
        "linear_attn.A_log": (2,), "linear_attn.dt_bias": (2,), "linear_attn.conv1d.weight": (6, 1, 2),
        "linear_attn.in_proj_a.weight": (2, 2), "linear_attn.in_proj_b.weight": (2, 2),
        "linear_attn.in_proj_qkv.weight": (6, 2), "linear_attn.in_proj_z.weight": (2, 2),
        "linear_attn.out_proj.weight": (2, 2), "linear_attn.norm.weight": (1,),
    }
    full = {
        "self_attn.k_norm.weight": (1,), "self_attn.q_norm.weight": (1,),
        "self_attn.q_proj.weight": (4, 2), "self_attn.k_proj.weight": (1, 2),
        "self_attn.v_proj.weight": (1, 2), "self_attn.o_proj.weight": (2, 2),
    }
    shapes = {EMBED: (7, 2), NORM: (2,), "model.visual.patch_embed.proj.weight": (2, 3, 1, 2, 2),
              "model.visual.merger.norm.bias": (2,)}
    for layer in range(24):
        names = {**common, **(full if layer % 4 == 3 else linear)}
        shapes.update({f"{PREFIX}{layer}.{key}": shape for key, shape in names.items()})
    tensors = {}
    for j, (key, shape) in enumerate(shapes.items(), 1):
        vals = [(1 + ((11*j + 7*i) % 97))/128 for i in range(math.prod(shape))]
        tensors[key] = dtype, shape, encode(vals, dtype)
    if head != "absent":
        tensors[HEAD] = tensors[EMBED] if head == "equal" else (dtype, (7, 2), encode([0.25]*14, dtype))
    return tensors


def shard(path, tensors, metadata=None):
    header = {"__metadata__": {"format": "pt", **(metadata or {})}}
    data = bytearray()
    for key in reversed(tensors):
        dtype, shape, payload = tensors[key]
        start = len(data)
        data.extend(payload)
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(data)]}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + data)


def independent_read(path):
    blob = path.read_bytes()
    n = int.from_bytes(blob[:8], "little")
    header = json.loads(blob[8:8+n])
    header.pop("__metadata__")
    payload = blob[8+n:]
    return {k: (r["dtype"], tuple(r["shape"]), payload[slice(*r["data_offsets"])]) for k, r in header.items()}


def fixture(path, params=None, cfg=None, indexed=False, alias=False):
    path.mkdir()
    cfg = config() if cfg is None else cfg
    tensors = parameters() if params is None else params
    (path / "config.json").write_text(json.dumps(cfg))
    (path / "readout.json").write_text('{"readout":"fixture only"}\n')
    (path / "NOTICE").write_text("Generated weights for a serialization test\n")
    metadata = {HEAD: EMBED} if alias else {}
    if not indexed:
        shard(path / "model.safetensors", tensors, metadata)
        return tensors
    weight_map = {}
    for i in range(2):
        name = f"part-{i}.safetensors"
        part = {key: tensors[key] for key in list(tensors)[i::2]}
        shard(path / name, part, metadata if i == 0 else {})
        weight_map.update(dict.fromkeys(part, name))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return tensors


def snapshot(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}


def witness(tensors, visits, token=2):
    """A nonlinear CPU checksum of phase-specific matrices, MLP, embedding, norm and readout."""
    embed = decode(tensors[EMBED])
    x = list(embed[2*token:2*token+2])
    for i in visits:
        base = f"{PREFIX}{i}."
        bridge = "self_attn.o_proj.weight" if i % 4 == 3 else "linear_attn.out_proj.weight"
        attention = decode(tensors[base + bridge])
        up = decode(tensors[base + "mlp.up_proj.weight"])
        down = decode(tensors[base + "mlp.down_proj.weight"])
        gate = decode(tensors[base + "mlp.gate_proj.weight"])
        mid = [math.tanh(up[2*j]*x[0] + up[2*j+1]*x[1]) * (gate[2*j]*x[0] + gate[2*j+1]*x[1])
               for j in range(3)]
        x = [a + 0.02*(attention[2*j]*x[0] + attention[2*j+1]*x[1] + sum(
            down[3*j+s]*mid[s] for s in range(3))) for j, a in enumerate(x)]
    norm = decode(tensors[NORM])
    head = decode(tensors.get(HEAD, tensors[EMBED]))
    return tuple(sum(head[j+i] * x[i] * norm[i] for i in (0, 1)) for j in range(0, len(head), 2))


class CompactTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT, prefix=".test-compact-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / "src"
        self.out = self.root / "out"

    def export(self, plan=SELECT):
        return export_compact(self.src, self.out, plan, fixture=True)

    def test_physical_export_and_two_same_weight_reloaded_controls(self):
        tensors = fixture(self.src)
        initial = snapshot(self.src)
        manifest = self.export()
        self.assertEqual(verify_compact(self.out), manifest)
        self.assertEqual(snapshot(self.src), initial)
        self.assertTrue(manifest["ties"]["configured_tied"])
        self.assertTrue(manifest["ties"]["both_separately_stored"])
        self.assertTrue(manifest["ties"]["stored_values_bit_identical"])
        self.assertEqual(snapshot(self.out)["model.safetensors"], manifest["weights_sha256"])
        compact = independent_read(self.out / "model.safetensors")
        anchors = [*range(4), *range(8,16), *range(20,24)]
        expected = {}
        for key, record in tensors.items():
            if key.startswith(PREFIX):
                i, suffix = key[len(PREFIX):].split(".", 1)
                if int(i) in anchors:
                    expected[f"{PREFIX}{anchors.index(int(i))}.{suffix}"] = record
            else:
                expected[key] = record
        self.assertEqual(compact, expected)  # every survivor, including independently serialized head and vision
        self.assertEqual(manifest["anchor_source_layers"], anchors)
        cfg = json.loads((self.out / "config.json").read_bytes())
        original = config()
        original["text_config"]["num_hidden_layers"] = 16
        original["text_config"]["layer_types"] = original["text_config"]["layer_types"][:16]
        self.assertEqual(cfg, original)
        for name in ("readout.json", "NOTICE"):
            self.assertEqual((self.out / name).read_bytes(), (self.src / name).read_bytes())
        before, after = manifest["source"]["storage"], manifest["storage"]
        self.assertEqual(before["vision"], after["vision"])
        counts = []
        byte_totals = []
        for params, report, directory in [(tensors, before, self.src), (compact, after, self.out)]:
            elements = sum(math.prod(r[1]) for r in params.values())
            self.assertEqual(report["full"]["stored_parameter_elements"], elements)
            text = sum(math.prod(r[1]) for k,r in params.items() if not k.startswith("model.visual."))
            self.assertEqual(report["text"]["stored_parameter_elements"], text)
            self.assertEqual(report["text"]["parameter_elements_assuming_configured_ties"], text - 14)
            self.assertEqual(report["full"]["tensor_bytes"], elements * 2)
            self.assertEqual(report["weight_file_bytes"], (directory / "model.safetensors").stat().st_size)
            counts.append(elements)
            byte_totals.append(sum(p.stat().st_size for p in directory.iterdir()))
        self.assertLess(counts[1], counts[0])
        self.assertLess(byte_totals[1], byte_totals[0])  # includes the added manifest and all copied sidecars
        self.assertLess(after["weight_file_bytes"], before["weight_file_bytes"])
        visits = layer_schedule(manifest, k=1)
        repeated = layer_schedule(manifest, k=2)
        self.assertEqual(visits, tuple(range(16)))
        self.assertEqual(repeated, tuple([*range(4), *range(4,12), *range(4,12), *range(12,16)]))
        before_source_ids = [*range(4), *range(8,16), *range(8,16), *range(20,24)]
        saved = snapshot(self.out)
        for token in [0, 2, 5]:
            self.assertEqual(witness(compact, visits, token), witness(tensors, anchors, token))
            self.assertEqual(witness(compact, repeated, token), witness(tensors, before_source_ids, token))
            self.assertNotEqual(witness(compact, repeated, token), witness(compact, visits, token))
        self.assertEqual(snapshot(self.out), saved)
        another = self.root / "again"
        export_compact(self.src, another, SELECT, fixture=True)
        self.assertEqual(snapshot(another), saved)

    def test_indexed_weighted_export_and_semantic_exclusions(self):
        for dtype in ("BF16", "F16", "F32"):
            with self.subTest(dtype=dtype):
                source, out = self.root / f"src-{dtype}", self.root / f"out-{dtype}"
                original = fixture(source, parameters(dtype, head="different"), indexed=True)
                export_compact(source, out, MERGE, fixture=True)
                m = verify_compact(out)
                self.assertTrue(m["ties"]["configured_tied"])
                self.assertFalse(m["ties"]["stored_values_bit_identical"])
                compact = independent_read(out / "model.safetensors")
                reference = {}
                eligible = {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}
                for key, rec in original.items():
                    if not key.startswith(PREFIX):
                        reference[key] = rec
                        continue
                    number, suffix = key[len(PREFIX):].split(".", 1)
                    cycle, phase = divmod(int(number), 4)
                    if cycle in (1, 4):
                        continue
                    compact_cycle = {0: 0, 2: 1, 3: 2, 5: 3}[cycle]
                    outkey = f"{PREFIX}{4*compact_cycle+phase}.{suffix}"
                    if suffix in eligible and cycle in (2, 3):
                        neighbor = int(number) - 4 if cycle == 2 else int(number) + 4
                        adjacent = original[f"{PREFIX}{neighbor}.{suffix}"]
                        vals = [0.75*x+0.25*y for x,y in zip(decode(rec), decode(adjacent))]
                        rec = dtype, rec[1], encode(vals, dtype)
                    reference[outkey] = rec
                self.assertEqual(compact, reference)
                self.assertNotEqual(compact[PREFIX+"4.mlp.up_proj.weight"][2], original[PREFIX+"8.mlp.up_proj.weight"][2])
                self.assertEqual(witness(compact, layer_schedule(m, 2)),
                                 witness(reference, [*range(12), *range(4,16)]))
                self.assertEqual(m["source"]["storage"]["weight_file_bytes"],
                                 sum(p.stat().st_size for p in source.glob("*.safetensors")))

    def test_fp32_accumulation_not_double_and_round_to_even_on_cast(self):
        p = {"format": "joey-compact/v1", "groups": [
            {"cycles": [0,1,2], "weights": [0.25,0.5,0.25]}, {"cycles": [5]}], "repeat_cycles": [0,1]}
        for dtype, step in [("F32", None), ("BF16", 1/128), ("F16", 1/1024)]:
            with self.subTest(dtype=dtype):
                records = parameters(dtype)
                vals = [(2**26, 1), (1, 5), (-2**26, 9)] if step is None else [
                    (1, 1+step), (1+step, 1+2*step), (1, 1+step)]
                for cycle, pair in enumerate(vals):
                    key = f"{PREFIX}{cycle*4}.mlp.down_proj.weight"
                    records[key] = (dtype, (2, 3), encode(list(pair)*3, dtype))
                src, out = self.root / (dtype+"src"), self.root / (dtype+"out")
                fixture(src, records)
                export_compact(src, out, p, fixture=True)
                actual = decode(independent_read(out / "model.safetensors")[PREFIX+"0.mlp.down_proj.weight"])
                expected = (0,5) if step is None else (1,1+2*step)
                self.assertEqual(actual, expected*3)

    def test_tie_is_preserved_not_invented(self):
        for tied, head, alias in [(True, "absent", True), (True, "absent", False), (False, "different", False)]:
            with self.subTest(tied=tied, head=head, alias=alias):
                src, out = self.root / f"{tied}-{alias}", self.root / f"out-{tied}-{alias}"
                cfg = config()
                cfg["tie_word_embeddings"] = cfg["text_config"]["tie_word_embeddings"] = tied
                original = fixture(src, parameters(head=head), cfg, alias=alias)
                export_compact(src, out, SELECT, fixture=True)
                m = verify_compact(out)
                compact = independent_read(out / "model.safetensors")
                self.assertEqual(HEAD in compact, HEAD in original)
                self.assertEqual(m["ties"]["configured_tied"], tied)
                self.assertEqual(m["ties"]["stored_values_bit_identical"], None if head == "absent" else False)
                self.assertEqual(witness(compact, range(16)), witness(original, m["anchor_source_layers"]))

    def test_malformed_checkpoint_fails_without_partial_export(self):
        for case in ("shape", "missing_key", "extra_text", "phase", "dtype", "absent_head", "alias", "truncation",
                     "offset", "duplicate", "index", "orphan", "nonfinite"):
            with self.subTest(case=case):
                src, out = self.root / case, self.root / (case+"-out")
                data, cfg = parameters("F32"), config()
                key = PREFIX + "4.mlp.up_proj.weight"
                dtype, shape, buf = data[key]
                if case == "shape":
                    data[key] = dtype, tuple(reversed(shape)), buf
                elif case == "missing_key":
                    del data[PREFIX+"4.linear_attn.A_log"]
                elif case == "extra_text":
                    data[PREFIX+"4.rotary_emb.unknown"] = data[key]
                elif case == "phase":
                    cfg["text_config"]["layer_types"][3] = "linear_attention"
                elif case == "dtype":
                    data[key] = "F16", shape, encode([1.0]*6, "F16")
                elif case == "absent_head":
                    del data[HEAD]
                    cfg["text_config"]["tie_word_embeddings"] = cfg["tie_word_embeddings"] = False
                elif case == "nonfinite":
                    data[key] = dtype, shape, encode([float("nan")]*6, dtype)
                fixture(src, data, cfg, indexed=(case == "index"), alias=(case == "alias"))
                weights = src / "model.safetensors"
                if case == "truncation":
                    weights.write_bytes(weights.read_bytes()[:-1])
                elif case == "offset":
                    raw = weights.read_bytes().replace(b'"data_offsets":[0,', b'"data_offsets":[1,', 1)
                    weights.write_bytes(raw)
                elif case == "duplicate":
                    raw = b'{"one":{},"one":{}}'
                    weights.write_bytes(struct.pack("<Q", len(raw)) + raw)
                elif case == "index":
                    ix = src / "model.safetensors.index.json"
                    indexed = json.loads(ix.read_bytes())
                    indexed["weight_map"][EMBED] = "../missing.safetensors"
                    ix.write_text(json.dumps(indexed))
                elif case == "orphan":
                    shard(src / "unused.safetensors", {HEAD: data[HEAD]})
                with self.assertRaises(CompactError):
                    export_compact(src, out, MERGE, fixture=True)
                self.assertFalse(out.exists())
                self.assertEqual(list(self.root.glob(".joey-export-*")), [])

    def test_rejects_invalid_plan_unsafe_destination_and_unpinned_input(self):
        fixture(self.src)
        with self.assertRaisesRegex(CompactError, "pinned"):
            export_compact(self.src, self.out, SELECT)
        for groups, repeat in [
            ([{"cycles": [0]}, {"cycles": [0]}], [0,1]), ([{"cycles": [6]}], [0,1]),
            ([{"cycles": [True]}], [0,1]), ([{"cycles": [1,2]}], [0,1]),
            ([{"cycles": [1,2], "weights": [0.1,0.1]}], [0,1]),
            ([{"cycles": [0]}], [0,2]), ([{"cycles": [0]}], [False,1]),
            ([{"cycles": [i]} for i in range(6)], [0,6]),
        ]:
            with self.subTest(groups=groups, repeat=repeat), self.assertRaises(CompactError):
                self.export({"format": "joey-compact/v1", "groups": groups, "repeat_cycles": repeat})
        before = snapshot(self.src)
        for dest in (self.src, self.src / "nested", self.root):
            with self.subTest(dest=dest), self.assertRaises(CompactError):
                export_compact(self.src, dest, SELECT, fixture=True)
        self.assertEqual(snapshot(self.src), before)
        self.assertFalse(self.out.exists())
        cfg = config()
        del cfg["joey_generated_fixture"]
        (self.src / "config.json").write_text(json.dumps(cfg))
        with self.assertRaisesRegex(CompactError, "marked"):
            self.export()

    def test_verifier_detects_tampering_and_bad_k(self):
        fixture(self.src)
        manifest = self.export()
        for k in (0, -1, 33, 2.0, True):
            with self.subTest(k=k), self.assertRaises(CompactError):
                layer_schedule(manifest, k)
        with self.assertRaises(CompactError):
            self.export()  # never overwrite
        for name in ("model.safetensors", "config.json", "NOTICE", "compact.json"):
            path = self.out / name
            old = path.read_bytes()
            if name == "model.safetensors":
                new = old[:-1] + bytes([old[-1] ^ 1])
            elif name == "compact.json":
                bad = copy.deepcopy(manifest)
                bad["repeat_layers"] = [0,4]
                new = json.dumps(bad).encode()
            else:
                new = old + b" "
            path.write_bytes(new)
            with self.subTest(name=name), self.assertRaises(CompactError):
                verify_compact(self.out)
            path.write_bytes(old)
        self.assertEqual(verify_compact(self.out), manifest)

    def test_deployed_storage_accounts_for_aliases_views_and_fp32_clone(self):
        class View:
            device = "cpu"

            def __init__(self, allocation, offset, count):
                self.allocation, self.offset, self.count = allocation, offset, count
                self.dtype = allocation.typecode

            def is_contiguous(self):
                return True

            def untyped_storage(self):
                return self

            def nbytes(self):
                return self.allocation.itemsize * len(self.allocation)

            def data_ptr(self):
                return self.allocation.buffer_info()[0]

            def element_size(self):
                return self.allocation.itemsize

            def numel(self):
                return self.count

            def storage_offset(self):
                return self.offset

        bank = array("H", range(16))
        embed = View(bank, 0, 12)
        clone = View(array("f", [1.0]*4), 0, 4)
        report = measure_text_storage({"embed": embed, "head": embed, "view": View(bank, 8, 6), "clone": clone})
        self.assertEqual(report["exact_alias_groups"], [["embed", "head"]])
        self.assertEqual(report["named_elements"], 34)
        self.assertEqual(report["unique_referenced_elements"], 18)
        self.assertEqual(report["unique_referenced_bytes"], 44)
        self.assertEqual(report["backing_storage_bytes"], 48)


if __name__ == "__main__":
    unittest.main()
