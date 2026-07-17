import unittest
from types import SimpleNamespace

import run_rlmr_v2_1_pipeline


class RLMRV21PipelineTest(unittest.TestCase):
    def make_args(self, **overrides):
        values = {
            "train_map": "corridor",
            "eval_map": "corridor",
            "losses": [0.1, 0.2, 0.3, 0.5],
            "episodes_per_setting": 50,
            "epochs": 1,
            "train_map_offset": 0,
            "train_seed": 0,
            "train_retransmission_seed": 100000,
            "eval_episodes": 20,
            "eval_map_offset": 100,
            "eval_seed": 3000,
            "eval_retransmission_seed": 4000,
            "meta_agents": 2,
            "q_table": "mar_inference/rlmr_v2_1_q_table.json",
            "output_tag": None,
            "overwrite_eval": False,
            "dry_run": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_train_command_uses_v2_1_profile(self):
        command = run_rlmr_v2_1_pipeline.build_train_command(self.make_args())
        self.assertIn("v2_1", command)
        self.assertIn("mar_inference/rlmr_v2_1_q_table.json", command)
        self.assertEqual(command[command.index("--episodes-per-setting") + 1], "50")

    def test_eval_command_uses_frozen_table_and_two_workers(self):
        command = run_rlmr_v2_1_pipeline.build_eval_command(self.make_args())
        self.assertEqual(command[command.index("--profile") + 1], "v2_1")
        self.assertEqual(command[command.index("--meta-agents") + 1], "2")
        self.assertEqual(
            command[command.index("--output-tag") + 1],
            "v2_1_dev20_seed3000",
        )

    def test_dry_run_is_forwarded_to_both_stages(self):
        args = self.make_args(dry_run=True)
        self.assertEqual(run_rlmr_v2_1_pipeline.build_train_command(args)[-1], "--dry-run")
        self.assertEqual(run_rlmr_v2_1_pipeline.build_eval_command(args)[-1], "--dry-run")


if __name__ == "__main__":
    unittest.main()
