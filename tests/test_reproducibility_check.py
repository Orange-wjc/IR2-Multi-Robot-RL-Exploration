import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import run_reproducibility_check


class ReproducibilityCheckTest(unittest.TestCase):
    def make_args(self, **overrides):
        values = {
            "map": "corridor",
            "loss": 0.3,
            "episodes": 10,
            "map_offset": 200,
            "message_seed": 5000,
            "retransmission_seed": 6000,
            "simulation_seed": 7000,
            "meta_agents": 2,
            "q_table": "mar_inference/rlmr_v2_1_q_table.json",
            "output_tag": "v2_1_repro",
            "overwrite": False,
            "dry_run": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_command_forwards_all_three_seeds(self):
        command = run_reproducibility_check.build_command(self.make_args(), "a")
        self.assertEqual(command[command.index("--seed") + 1], "5000")
        self.assertEqual(command[command.index("--retransmission-seed") + 1], "6000")
        self.assertEqual(command[command.index("--simulation-seed") + 1], "7000")

    def test_loader_ignores_meta_agent_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            main = Path(directory) / "main.csv"
            skipped = Path(directory) / "skipped.csv"
            with main.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["eps", "success", "meta_agent_id"])
                writer.writeheader()
                writer.writerow({"eps": 0, "success": True, "meta_agent_id": 0})
            with skipped.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["eps", "skip_reason", "meta_agent_id"])
                writer.writeheader()

            records = run_reproducibility_check.load_episode_records(main, skipped)
            self.assertEqual(records[0][0], "completed")
            self.assertNotIn("meta_agent_id", records[0][1])


if __name__ == "__main__":
    unittest.main()
