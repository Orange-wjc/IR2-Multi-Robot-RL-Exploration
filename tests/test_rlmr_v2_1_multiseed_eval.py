import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import run_rlmr_v2_1_multiseed_eval as multiseed


class RLMRV21MultiSeedEvalTest(unittest.TestCase):
    def make_args(self, **overrides):
        values = {
            "map": "corridor",
            "losses": [0.1, 0.2, 0.3, 0.5],
            "policies": ["priority", "v2_1"],
            "episodes": 3,
            "map_offset": 200,
            "simulation_seed": 7000,
            "message_seed_bases": [15000, 25000],
            "retransmission_seed_bases": [16000, 26000],
            "meta_agents": 1,
            "output_prefix": "v2_1_multiseed",
            "v2_q_table": "mar_inference/rlmr_v2_q_table.json",
            "v2_1_q_table": "mar_inference/rlmr_v2_1_q_table.json",
            "overwrite": False,
            "dry_run": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_build_command_forwards_matched_seeds_and_frozen_profile(self):
        args = self.make_args()
        command = multiseed.build_command(
            args,
            "v2_1",
            0.3,
            15002,
            16002,
            "v2_1_multiseed_seed15000",
        )
        self.assertEqual(command[command.index("--seed") + 1], "15002")
        self.assertEqual(command[command.index("--retransmission-seed") + 1], "16002")
        self.assertEqual(command[command.index("--simulation-seed") + 1], "7000")
        self.assertEqual(command[command.index("--rlmr-v2-profile") + 1], "v2_1")
        self.assertEqual(command[command.index("--meta-agents") + 1], "1")

    def test_result_paths_separate_priority_and_v2_1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            priority, _ = multiseed.result_paths(
                "corridor", 0.5, "priority", "seed_tag", root
            )
            v2_1, _ = multiseed.result_paths(
                "corridor", 0.5, "v2_1", "seed_tag", root
            )
            self.assertIn("priority_retrans", priority.name)
            self.assertIn("rlmr_v2_1_retrans", v2_1.name)
            self.assertNotEqual(priority, v2_1)

    def test_validate_result_checks_coverage_seeds_and_skipped_total(self):
        args = self.make_args()
        with tempfile.TemporaryDirectory() as directory:
            main = Path(directory) / "main.csv"
            skipped = Path(directory) / "skipped.csv"
            fieldnames = [
                "map_index",
                "message_loss_seed",
                "retransmission_loss_seed",
                "simulation_seed",
                "message_loss_prob",
                "rlmr_version",
                "retrans_attempts",
                "map_retrans_attempts",
                "pose_retrans_attempts",
                "graph_retrans_attempts",
            ]
            with main.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for map_index in (200, 201):
                    writer.writerow({
                        "map_index": map_index,
                        "message_loss_seed": 15002,
                        "retransmission_loss_seed": 16002,
                        "simulation_seed": 7000 + map_index,
                        "message_loss_prob": 0.3,
                        "rlmr_version": "v2_1",
                        "retrans_attempts": 5,
                    })
            with skipped.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    "map_index": 202,
                    "message_loss_seed": 15002,
                    "retransmission_loss_seed": 16002,
                    "simulation_seed": 7202,
                    "message_loss_prob": 0.3,
                    "rlmr_version": "v2_1",
                    "retrans_attempts": 6,
                    "map_retrans_attempts": 3,
                    "pose_retrans_attempts": 1,
                    "graph_retrans_attempts": 2,
                })

            valid, skipped_count = multiseed.validate_result(
                main,
                skipped,
                args,
                "v2_1",
                0.3,
                15002,
                16002,
            )
            self.assertEqual((valid, skipped_count), (2, 1))

    def test_validate_args_rejects_unpaired_seed_lists(self):
        args = self.make_args(retransmission_seed_bases=[16000])
        with self.assertRaises(ValueError):
            multiseed.validate_args(args)


if __name__ == "__main__":
    unittest.main()
