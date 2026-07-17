import unittest
import sys
import random

import numpy as np
import torch

sys.modules["TRAINING"] = False
from env import Env
from test_driver import seed_episode


class FixedRandom:
    def random(self):
        return 1.0


class RetransmissionPolicyIsolationTest(unittest.TestCase):
    def test_episode_seed_resets_python_numpy_and_torch(self):
        self.assertEqual(seed_episode(207), 207)
        first = (random.random(), np.random.random(), torch.rand(1).item())
        self.assertEqual(seed_episode(207), 207)
        second = (random.random(), np.random.random(), torch.rand(1).item())
        self.assertEqual(first, second)

    def test_priority_retransmission_does_not_read_v2_attributes(self):
        env = Env.__new__(Env)
        env.rlmr_policy = None
        env.rlmr_policy_version = "off"
        env.pending_retransmissions = [[{}, {"map": {"created_step": 1, "retry_count": 0}}]]
        env.retransmission_attempts = {"map": 0}
        env.retransmission_successes = {"map": 0}
        env.retransmission_dropped = {"map": 0}
        env.retransmission_delay_sum = 0.0
        env.retransmission_delay_count = 0
        env.retransmission_delay_max = 0.0
        env.retransmission_loss_rng = FixedRandom()
        env.retransmission_enabled = lambda: True
        env.expire_pending_retransmission = lambda *args: False
        env.has_higher_priority_pending = lambda *args: False
        env.adaptive_retransmission_ready = lambda *args: True
        env.can_attempt_retransmission = lambda *args: True
        env.mark_retransmission_budget_used = lambda *args: None
        env.get_message_loss_prob = lambda message_type: 0.1

        delivered = env.try_pending_retransmission(0, 1, "map", sim_step=5)

        self.assertTrue(delivered)
        self.assertEqual(env.retransmission_attempts["map"], 1)
        self.assertEqual(env.retransmission_successes["map"], 1)
        self.assertIsNone(env.pending_retransmissions[0][1]["map"])


if __name__ == "__main__":
    unittest.main()
