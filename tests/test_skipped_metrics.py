import unittest

from test_multi_robot_worker import TestWorker


class FakeEnv:
    comm_attempts = 10
    comm_successes = 8
    comm_dropped = 2
    message_attempts = {"map": 10, "pose": 8, "graph": 6}
    message_successes = {"map": 7, "pose": 6, "graph": 5}
    message_dropped = {"map": 3, "pose": 2, "graph": 1}
    retransmission_attempts = {"map": 4, "pose": 1, "graph": 2}
    retransmission_successes = {"map": 3, "pose": 1, "graph": 1}
    retransmission_dropped = {"map": 1, "pose": 0, "graph": 1}
    retransmission_expired = {"map": 1, "pose": 0, "graph": 0}
    retransmission_delay_sum = 12
    retransmission_delay_count = 4
    retransmission_delay_max = 5
    pose_staleness_sum = 15
    pose_staleness_count = 3
    pose_staleness_max = 9
    rlmr_decision_count = 6
    rlmr_action_counts = {"none": 1, "map": 3, "graph": 1, "pose": 1}
    rlmr_policy = None
    rlmr_policy_version = "off"
    rlmr_training = False
    rlmr_forced_map_actions = 0

    def pending_message_count(self):
        return 2


class SkippedMetricsTest(unittest.TestCase):
    def test_skipped_episode_records_aggregate_communication_metrics(self):
        worker = TestWorker.__new__(TestWorker)
        worker.env = FakeEnv()
        worker.skip_info = {}
        worker.perf_metrics = {}

        worker.finalize_skip_metrics()

        metrics = worker.perf_metrics["skip_info"]
        self.assertEqual(metrics["comm_successes"], 8)
        self.assertAlmostEqual(metrics["actual_packet_loss_rate"], 0.2)
        self.assertEqual(metrics["map_msg_successes"], 7)
        self.assertAlmostEqual(metrics["actual_map_msg_loss_rate"], 0.3)
        self.assertEqual(metrics["retrans_attempts"], 7)
        self.assertEqual(metrics["retrans_successes"], 5)
        self.assertEqual(metrics["retrans_dropped"], 2)
        self.assertEqual(metrics["retrans_expired"], 1)
        self.assertAlmostEqual(metrics["retrans_success_rate"], 5 / 7)
        self.assertEqual(metrics["retrans_delay_mean"], 3)
        self.assertEqual(metrics["pending_retransmissions"], 2)
        self.assertEqual(metrics["pose_staleness_mean"], 5)


if __name__ == "__main__":
    unittest.main()
