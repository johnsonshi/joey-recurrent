"""Compact-adapter fixtures, not weight or GPU parity results."""

import hashlib
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from joey_research import native_compact as c
from joey_research import norm_parity as npv


class Qwen3_5RMSNorm:
    def __init__(self, value):
        self.weight = SimpleNamespace(value=value, shape=(2048,), device="cpu", dtype="torch.bfloat16", requires_grad=False)


class NormSourceCanaryContracts(unittest.TestCase):
    def layers(self):
        return [SimpleNamespace(input_layernorm=Qwen3_5RMSNorm(j), post_attention_layernorm=Qwen3_5RMSNorm(j+.5))
                for j in range(24)]

    def test_original_parameters_restored_after_reference_failure(self):
        layers = self.layers(); keep = [vars(x).copy() for x in layers]
        old = [getattr(x,n).weight for x in layers for n in npv._NAMES]
        with self.assertRaises(InterruptedError):
            with npv._source_donor(layers):
                for i in range(8,12):
                    for n in npv._NAMES:
                        self.assertIs(getattr(layers[i],n).weight, getattr(layers[i+4],n).weight)
                raise InterruptedError()
        self.assertEqual([vars(x) for x in layers], keep)
        self.assertTrue(all(a is b for a,b in zip(old,(getattr(x,n).weight for x in layers for n in npv._NAMES))))

    def test_source_oracle_skips_removed_matrices_and_switches_only_second_visit(self):
        layers = self.layers(); calls = []
        class Hidden(float):
            def clone(self):return Hidden(self)
        def step(h, a, b):
            calls.append((a,b))
            return Hidden(h + sum(1000+i+layers[i].input_layernorm.weight.value
                                  + layers[i].post_attention_layernorm.weight.value for i in range(a,b)))
        branch=SimpleNamespace(_window=(0,24),layout="fixture",labels=["x","y"],_prelude=lambda:Hidden(0),_layers=step,
                               _coda=lambda x:[x,-x],probabilities=lambda z:(.75,.25))
        with patch.object(npv,"_hidden_pin",return_value="fixture"):
            d=npv._source_case(branch,layers)
        self.assertEqual(calls,[(0,8),(8,12),(16,24),(8,12),(16,24),(8,12),(16,24)])
        self.assertEqual(d["donor_two"][0][0]-d["identity_two"][0][0],32)
        self.assertLess(d["one"][0][0],d["identity_two"][0][0])

    def test_equal_top_choice_does_not_excuse_wrong_logits_or_missing_norm_calls(self):
        with self.assertRaisesRegex(ValueError,"independent source"):
            npv._compare(((2.,1.),(.7,.3)), ((2.01,1.),(.7,.3)))
        seen={k:1 for k,_,_ in npv.pass_norms._pairs()}
        result=SimpleNamespace(counts=npv.LoopCounts(0,1,2,1),norm_calls=seen)
        npv._check_counts(result,2);seen[next(iter(seen))]=0
        with self.assertRaisesRegex(ValueError,"normalization calls"):npv._check_counts(result,2)


class CompactContracts(unittest.TestCase):
    def test_verified_fixture_or_wrong_manifest_pin_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, source = Path(tmp) / "compact", Path(tmp) / "source"
            root.mkdir()
            source.mkdir()
            raw = b'{"format":"joey-compact/v1"}'
            (root / "compact.json").write_bytes(raw)
            verify = Mock(return_value={"source": {"kind": "generated-fixture"}})
            with patch.dict(sys.modules, {"joey_research.compact": SimpleNamespace(verify_compact=verify)}):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    c._verified_export(root, source, "0" * 64)
                verify.assert_not_called()
                with self.assertRaisesRegex(ValueError, "never a generated fixture"):
                    c._verified_export(root, source, hashlib.sha256(raw).hexdigest())

    def test_merge_reference_is_not_selection_and_preserves_attention_and_phase(self):
        kept = (0, 1, 2, 3, 8, 9, 10, 11)
        groups = [{"cycles": [0]}, {"cycles": [1, 2], "anchor": 2, "weights": [.25, .75]}]
        stem = "model.language_model.layers."
        source = ["lm_head.weight", stem + "1.mlp.up_proj.weight", stem + "7.mlp.down_proj.weight",
                  stem + "11.mlp.down_proj.weight", stem + "11.self_attn.k_proj.weight"]
        plan = {name: (inputs, weights, replacement)
                for name, inputs, weights, replacement in c._tensor_plan(source, kept, groups)}
        self.assertEqual(plan[stem + "7.mlp.down_proj.weight"],
                         ((stem + "7.mlp.down_proj.weight", stem + "11.mlp.down_proj.weight"),
                          (.25, .75), (11, "down_proj")))
        self.assertEqual(plan[stem + "7.self_attn.k_proj.weight"], ((stem + "11.self_attn.k_proj.weight",), (), None))
        self.assertEqual(plan[stem + "1.mlp.up_proj.weight"], ((stem + "1.mlp.up_proj.weight",), (), None))
        self.assertEqual(plan["lm_head.weight"], (("lm_head.weight",), (), None))
        self.assertEqual(len(plan), 4)

    def test_reloaded_one_pass_must_agree_with_separately_computed_witness(self):
        model = c.CompactJoey(None)
        model.physical_to_source = (0, 1, 2, 3)
        branch = SimpleNamespace(layout="stable-tokenization", temperature=1.,
                                 _full_logits=lambda: [1., 1.], _one_pass=lambda: [1., 1.])
        witness = [{"q": (branch.layout, [1.01, 1.])}]
        with (patch.dict(sys.modules, {"qev.calibrate": SimpleNamespace(softmax=lambda z, t: [.5, .5])}),
              patch.object(model, "inference_context", nullcontext),
              patch.object(model, "_request", return_value=None),
              patch.object(model, "_plan", return_value={"q": branch})):
            report = model._compare_reload([("s", {})], witness, "d" * 64)
            self.assertFalse(report.passed)  # the two reloaded paths agree with each other
            model.reload_report = report
            with self.assertRaises(c.ReloadParityError):
                model.verify_parity(window=(0, 4))

    def test_physical_schedule_uses_fresh_branch_state_for_each_real_visit(self):
        model = c.CompactJoey(None)
        model.physical_to_source = (0, 1, 2, 3, 8, 9, 10, 11)
        model.repeat_window = (4, 8)
        schedule = tuple(range(8)) + tuple(range(4, 8))
        used = []
        b = SimpleNamespace(prelude=lambda: 1, fresh_state=object,
                            block=lambda h, state: (used.append(state), h * 2)[1],
                            coda=lambda h, state: (used.append(state), h + 1)[1], decode=lambda h: h)
        with (patch.object(model, "physical_ids", return_value=schedule),
              patch.object(model, "branches", return_value={"a": b})):
            self.assertEqual(model.score_physical("s", {}, physical_ids=schedule), {"a": 5})
            self.assertEqual(len(set(used)), 3)
            with self.assertRaisesRegex(ValueError, "physical layer schedule"):
                model.score_physical("s", {}, physical_ids=model.physical_to_source * 2)
            with self.assertRaisesRegex(ValueError, "physical depth"):
                c._physical_window((8, 12), 8)


@unittest.skipUnless(os.environ.get("JOEY_TEST_COMPACT_DIR"), "real, verified compact export not requested")
class StagedCompactIntegration(unittest.TestCase):
    def test_exact_export_counterpart_then_physical_recurrence(self):
        model = c.CompactJoey.load(os.environ["JOEY_TEST_COMPACT_DIR"],
                                   source_model_dir=os.environ["JOEY_TEST_MODEL_DIR"],
                                   expected_manifest_sha256=os.environ["JOEY_TEST_COMPACT_MANIFEST_SHA256"],
                                   device=os.environ.get("JOEY_TEST_DEVICE", "cuda:0"))
        self.assertTrue(model.reload_report.passed)
        state, questions = c.stock._parity_requests()[0]
        official = model.counterpart(state, questions)
        once = model.score_physical(state, questions, physical_ids=model.physical_ids(1))
        twice = model.score_physical(state, questions, physical_ids=model.physical_ids(2))
        self.assertEqual(official.keys(), twice.keys())
        self.assertEqual(official["team"]["choice"], once["team"]["choice"])
        for name, p in once["team"]["probabilities"].items():
            self.assertLessEqual(abs(p - official["team"]["probabilities"][name]), .005001)


if __name__ == "__main__":
    unittest.main()
