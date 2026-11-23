"""Contract checks for the public Hub entry point. No checkpoint or GPU required."""

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


path = Path(__file__).resolve().parents[1] / "scripts" / "hub" / "run_joey.py"
spec = importlib.util.spec_from_file_location("hub_run", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class HubRunnerTests(unittest.TestCase):
    def test_two_visits_update_the_same_branch_before_one_readout(self):
        branch = Mock()
        branch.inference_context.side_effect = nullcontext
        branch.fresh_state.side_effect = object
        branch.prelude.return_value = 7
        visit_inputs, states = [], []

        def update(hidden, state):
            visit_inputs.append(hidden)
            states.append(state)
            return 2 * hidden

        branch.block.side_effect = update
        branch.coda.side_effect = lambda hidden, state: (hidden, state)
        branch.decode.side_effect = lambda result: result[0]
        model = Mock()
        model.branches.return_value = {"team": branch}
        questions = {"team": {"type": "choice"}}

        self.assertEqual(runner.run_two_pass(model, "some state", questions), {"team": 28})
        model.branches.assert_called_once_with("some state", questions, window=(8, 12))
        self.assertEqual(visit_inputs, [7, 14])
        self.assertEqual(branch.coda.call_count, 1)
        coda_input, coda_state = branch.coda.call_args.args
        self.assertEqual(coda_input, 28)
        self.assertEqual(len(set([*states, coda_state])), 3)

    def test_plain_one_pass_config_is_rejected_before_any_weight_read(self):
        native = SimpleNamespace(MODEL_REVISION="source", _verify_hashes=Mock())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            wrong = {"variant": "Joey-1-Compact-Loop", "source_revision": "source", "stored_text_layers": 20,
                     "middle_passes": 2, "physical_layer_visits": list(range(20)), "readout": "slot_lm"}
            (folder / "loop_config.json").write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, "two-pass schedule"):
                runner.check_snapshot(folder, native)
            native._verify_hashes.assert_not_called()
