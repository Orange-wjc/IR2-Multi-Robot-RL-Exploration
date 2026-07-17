import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_message_loss_sweep
import train_rlmr_v2


class RLMRV2ProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_parameters = (train_rlmr_v2.ROOT / "test_parameter.py").read_text(
            encoding="utf-8"
        )

    def test_training_parameters_select_v2_1_table(self):
        with tempfile.TemporaryDirectory() as directory:
            parameter_path = Path(directory) / "test_parameter.py"
            args = SimpleNamespace(
                map_offset=0,
                episodes_per_setting=2,
                no_map_shield=False,
                profile="v2_1",
            )
            setting = {
                "map": "corridor",
                "loss": 0.3,
                "message_seed": 10,
                "retransmission_seed": 20,
            }
            q_table = Path(directory) / "v2_1.json"

            with patch.object(train_rlmr_v2, "PARAM_PATH", parameter_path):
                train_rlmr_v2.write_training_parameters(
                    self.original_parameters, setting, args, q_table
                )

            text = parameter_path.read_text(encoding="utf-8")
            self.assertIn("RLMR_V2_PROFILE='v2_1'", text)
            self.assertIn(f"RLMR_V2_1_Q_TABLE_PATH='{q_table}'", text)
            self.assertIn("RLMR_V2_TRAIN=True", text)

    def test_sweep_parameters_forward_profile_table_and_worker_count(self):
        with tempfile.TemporaryDirectory() as directory:
            parameter_path = Path(directory) / "test_parameter.py"
            with patch.object(run_message_loss_sweep, "PARAM_PATH", parameter_path):
                run_message_loss_sweep.write_test_parameters(
                    self.original_parameters,
                    "corridor",
                    "random",
                    0.5,
                    30,
                    40,
                    "rlmr_v2",
                    3,
                    100,
                    1234,
                    True,
                    "v2_1",
                    2,
                    "v1.json",
                    "v2.json",
                    "v2_1.json",
                )

            text = parameter_path.read_text(encoding="utf-8")
            self.assertIn("NUM_META_AGENT=2", text)
            self.assertIn("TEST_RANDOM_SEED=1234", text)
            self.assertIn("RLMR_V2_PROFILE='v2_1'", text)
            self.assertIn("RLMR_V2_1_Q_TABLE_PATH='v2_1.json'", text)
            self.assertIn("RLMR_V2_TRAIN=False", text)


if __name__ == "__main__":
    unittest.main()
