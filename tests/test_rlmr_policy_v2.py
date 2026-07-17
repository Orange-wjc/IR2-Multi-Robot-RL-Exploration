import json
import tempfile
import unittest
from pathlib import Path

from rlmr_policy_v2 import TabularRLMRPolicyV2


class TabularRLMRPolicyV2Test(unittest.TestCase):
    def make_policy(self, path, **kwargs):
        defaults = {
            "train": True,
            "alpha": 1.0,
            "gamma": 0.9,
            "epsilon": 0.0,
            "seed": 1,
            "initial_q": 0.0,
        }
        defaults.update(kwargs)
        return TabularRLMRPolicyV2(path, **defaults)

    def test_td_update_bootstraps_from_next_valid_action(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = self.make_policy(Path(directory) / "q.json", alpha=0.5)
            first_state = (0, 1)
            next_state = (1, 2)
            policy.values_for_state(next_state)[policy.action_index("graph")] = 2.0

            self.assertEqual(policy.select_action((0, 1), first_state, ["map"]), "map")
            policy.add_reward((0, 1), 1.0)
            policy.select_action((0, 1), next_state, ["graph"])

            value = policy.values_for_state(first_state)[policy.action_index("map")]
            self.assertAlmostEqual(value, 1.4)
            self.assertEqual(policy.episode_td_updates, 1)

    def test_terminal_update_has_no_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = self.make_policy(Path(directory) / "q.json")
            state = (0, 1)
            policy.select_action((0, 1), state, ["map"])
            policy.add_reward((0, 1), 0.5)
            policy.finish_episode(terminal_reward=-1.0)

            value = policy.values_for_state(state)[policy.action_index("map")]
            self.assertAlmostEqual(value, -0.5)
            self.assertEqual(policy.episodes, 1)

    def test_targeted_terminal_reward_only_affects_matching_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = self.make_policy(Path(directory) / "q.json")
            first_state = (0, 1)
            second_state = (0, 2)
            policy.select_action((0, 1), first_state, ["map"])
            policy.select_action((0, 2), second_state, ["map"])
            policy.finish_episode(
                terminal_reward=-1.0,
                targeted_rewards={(0, 1): -4.0},
            )

            first_value = policy.values_for_state(first_state)[policy.action_index("map")]
            second_value = policy.values_for_state(second_state)[policy.action_index("map")]
            self.assertAlmostEqual(first_value, -5.0)
            self.assertAlmostEqual(second_value, -1.0)

    def test_invalid_actions_are_never_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = self.make_policy(Path(directory) / "q.json")
            state = (0, 1)
            values = policy.values_for_state(state)
            values[policy.action_index("map")] = 100.0
            values[policy.action_index("pose")] = 1.0
            self.assertEqual(policy.choose_action(state, ["none", "pose"]), "pose")

    def test_frozen_evaluation_does_not_add_unseen_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.json"
            training_policy = self.make_policy(path)
            training_policy.save()
            evaluation_policy = self.make_policy(path, train=False)

            self.assertEqual(evaluation_policy.choose_action((9, 9), ["none", "graph"]), "none")
            self.assertEqual(evaluation_policy.num_states(), 0)
            self.assertEqual(evaluation_policy.unseen_state_count(), 1)

    def test_save_and_resume_preserve_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.json"
            policy = self.make_policy(path)
            policy.select_action((0, 1), (1, 2), ["map"])
            policy.finish_episode(terminal_reward=2.0)
            policy.save()

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 2)
            self.assertEqual(payload["algorithm"], "td_q_learning")

            resumed = self.make_policy(path)
            self.assertEqual(resumed.episodes, 1)
            self.assertEqual(resumed.td_updates, 1)
            self.assertEqual(resumed.q_table, policy.q_table)


if __name__ == "__main__":
    unittest.main()
