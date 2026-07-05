##############################################################################
# Name: test_driver.py
# [Inference] Driver of training program, maintain & update the global network.
###############################################################################

from test_parameter import *
import ray
import numpy as np
import os
import torch
import csv
import pandas as pd
from model import PolicyNet
from test_multi_robot_worker import TestWorker
from datetime import datetime

def run_test():
    fieldnames = ['eps', 'num_robots', 'max_dist', 'steps', 'explored', 'success', 'connectivity', \
                  'packet_loss_enabled', 'packet_loss_prob', 'comm_attempts', 'comm_successes', \
                  'comm_dropped', 'actual_packet_loss_rate', \
                  'message_loss_enabled', 'message_loss_mode', 'message_loss_prob', \
                  'map_msg_attempts', 'map_msg_successes', 'map_msg_dropped', 'actual_map_msg_loss_rate', \
                  'pose_msg_attempts', 'pose_msg_successes', 'pose_msg_dropped', 'actual_pose_msg_loss_rate', \
                  'graph_msg_attempts', 'graph_msg_successes', 'graph_msg_dropped', 'actual_graph_msg_loss_rate', \
                  'retransmission_enabled', 'retransmission_policy', 'retransmission_budget', \
                  'retrans_attempts', 'retrans_successes', 'retrans_dropped', 'retrans_expired', \
                  'retrans_success_rate', 'retrans_delay_mean', 'retrans_delay_max', 'pending_retransmissions', \
                  'map_retrans_attempts', 'map_retrans_successes', 'map_retrans_dropped', 'map_retrans_expired', \
                  'pose_retrans_attempts', 'pose_retrans_successes', 'pose_retrans_dropped', 'pose_retrans_expired', \
                  'graph_retrans_attempts', 'graph_retrans_successes', 'graph_retrans_dropped', 'graph_retrans_expired', \
                  'pose_staleness_mean', 'pose_staleness_max']
    skipped_fieldnames = ['eps', 'num_robots', 'map_name', 'map_index', 'meta_agent_id', \
                          'skip_stage', 'skip_reason', 'robot_id', 'target_robot_id', 'step', \
                          'current_position', 'destination_position', \
                          'message_loss_enabled', 'message_loss_mode', 'message_loss_prob', \
                          'packet_loss_enabled', 'packet_loss_prob', \
                          'comm_attempts', 'comm_dropped', \
                          'map_msg_attempts', 'map_msg_dropped', \
                          'pose_msg_attempts', 'pose_msg_dropped', \
                          'graph_msg_attempts', 'graph_msg_dropped', \
                          'retransmission_enabled', 'retransmission_policy', 'retransmission_budget', \
                          'pending_retransmissions', \
                          'map_retrans_attempts', 'map_retrans_successes', 'map_retrans_dropped', 'map_retrans_expired', \
                          'pose_retrans_attempts', 'pose_retrans_successes', 'pose_retrans_dropped', 'pose_retrans_expired', \
                          'graph_retrans_attempts', 'graph_retrans_successes', 'graph_retrans_dropped', 'graph_retrans_expired']

    # Create .csv file for data collection
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    csv_file_name = "data_{}.csv".format(current_datetime)
    csv_file_path = os.path.join(log_path, csv_file_name)
    skipped_csv_file_name = "skipped_{}.csv".format(current_datetime)
    skipped_csv_file_path = os.path.join(log_path, skipped_csv_file_name)

    # Create CSV file
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
    if not os.path.exists(skipped_csv_file_path):
        with open(skipped_csv_file_path, mode='w') as skipped_csv_file:
            writer = csv.DictWriter(skipped_csv_file, fieldnames=skipped_fieldnames)
            writer.writeheader()


    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    if device == 'cuda':
        checkpoint = torch.load(MODEL_PATH)
    else:
        checkpoint = torch.load(MODEL_PATH, map_location = torch.device('cpu'))

    global_network.load_state_dict(checkpoint['policy_model'])

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()
    curr_test = 0

    dist_history = []

    job_list = []
    for i, meta_agent in enumerate(meta_agents[:NUM_TEST]):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        eps_skipped = []
        completed_tests = 0
        while completed_tests < NUM_TEST and job_list:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                completed_tests += 1
                success, metrics, info = job
                if success:
                    dist_history.append(metrics['travel_dist'])

                    # Populate CSV file
                    with open(csv_file_path, mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writer.writerow({'eps': info['episode_number'], \
                                        'num_robots': info['n_agent'], \
                                        'max_dist': metrics['travel_dist'], \
                                        'steps': metrics['travel_steps'], \
                                        'explored': metrics['explored_rate'], \
                                        'success': metrics['success_rate'], \
                                        'connectivity': metrics['connectivity_rate'], \
                                        'packet_loss_enabled': ENABLE_PACKET_LOSS, \
                                        'packet_loss_prob': PACKET_LOSS_PROB, \
                                        'comm_attempts': metrics['comm_attempts'], \
                                        'comm_successes': metrics['comm_successes'], \
                                        'comm_dropped': metrics['comm_dropped'], \
                                        'actual_packet_loss_rate': metrics['actual_packet_loss_rate'], \
                                        'message_loss_enabled': ENABLE_MESSAGE_LOSS, \
                                        'message_loss_mode': MESSAGE_LOSS_MODE, \
                                        'message_loss_prob': MESSAGE_LOSS_PROB, \
                                        'map_msg_attempts': metrics['map_msg_attempts'], \
                                        'map_msg_successes': metrics['map_msg_successes'], \
                                        'map_msg_dropped': metrics['map_msg_dropped'], \
                                        'actual_map_msg_loss_rate': metrics['actual_map_msg_loss_rate'], \
                                        'pose_msg_attempts': metrics['pose_msg_attempts'], \
                                        'pose_msg_successes': metrics['pose_msg_successes'], \
                                        'pose_msg_dropped': metrics['pose_msg_dropped'], \
                                        'actual_pose_msg_loss_rate': metrics['actual_pose_msg_loss_rate'], \
                                        'graph_msg_attempts': metrics['graph_msg_attempts'], \
                                        'graph_msg_successes': metrics['graph_msg_successes'], \
                                        'graph_msg_dropped': metrics['graph_msg_dropped'], \
                                        'actual_graph_msg_loss_rate': metrics['actual_graph_msg_loss_rate'], \
                                        'retransmission_enabled': ENABLE_PRIORITY_RETRANSMISSION, \
                                        'retransmission_policy': RETRANSMISSION_POLICY, \
                                        'retransmission_budget': RETRANSMISSION_BUDGET_PER_PAIR, \
                                        'retrans_attempts': metrics['retrans_attempts'], \
                                        'retrans_successes': metrics['retrans_successes'], \
                                        'retrans_dropped': metrics['retrans_dropped'], \
                                        'retrans_expired': metrics['retrans_expired'], \
                                        'retrans_success_rate': metrics['retrans_success_rate'], \
                                        'retrans_delay_mean': metrics['retrans_delay_mean'], \
                                        'retrans_delay_max': metrics['retrans_delay_max'], \
                                        'pending_retransmissions': metrics['pending_retransmissions'], \
                                        'map_retrans_attempts': metrics['map_retrans_attempts'], \
                                        'map_retrans_successes': metrics['map_retrans_successes'], \
                                        'map_retrans_dropped': metrics['map_retrans_dropped'], \
                                        'map_retrans_expired': metrics['map_retrans_expired'], \
                                        'pose_retrans_attempts': metrics['pose_retrans_attempts'], \
                                        'pose_retrans_successes': metrics['pose_retrans_successes'], \
                                        'pose_retrans_dropped': metrics['pose_retrans_dropped'], \
                                        'pose_retrans_expired': metrics['pose_retrans_expired'], \
                                        'graph_retrans_attempts': metrics['graph_retrans_attempts'], \
                                        'graph_retrans_successes': metrics['graph_retrans_successes'], \
                                        'graph_retrans_dropped': metrics['graph_retrans_dropped'], \
                                        'graph_retrans_expired': metrics['graph_retrans_expired'], \
                                        'pose_staleness_mean': metrics['pose_staleness_mean'], \
                                        'pose_staleness_max': metrics['pose_staleness_max'] })
                else:
                    eps_skipped.append(info['episode_number'])
                    skip_info = metrics.get('skip_info', {})
                    with open(skipped_csv_file_path, mode='a') as skipped_csv_file:
                        writer = csv.DictWriter(skipped_csv_file, fieldnames=skipped_fieldnames)
                        writer.writerow({'eps': info['episode_number'], \
                                        'num_robots': info['n_agent'], \
                                        'map_name': skip_info.get('map_name', ''), \
                                        'map_index': skip_info.get('map_index', ''), \
                                        'meta_agent_id': skip_info.get('meta_agent_id', info['id']), \
                                        'skip_stage': skip_info.get('skip_stage', ''), \
                                        'skip_reason': skip_info.get('skip_reason', ''), \
                                        'robot_id': skip_info.get('robot_id', ''), \
                                        'target_robot_id': skip_info.get('target_robot_id', ''), \
                                        'step': skip_info.get('step', ''), \
                                        'current_position': skip_info.get('current_position', ''), \
                                        'destination_position': skip_info.get('destination_position', ''), \
                                        'message_loss_enabled': ENABLE_MESSAGE_LOSS, \
                                        'message_loss_mode': MESSAGE_LOSS_MODE, \
                                        'message_loss_prob': MESSAGE_LOSS_PROB, \
                                        'packet_loss_enabled': ENABLE_PACKET_LOSS, \
                                        'packet_loss_prob': PACKET_LOSS_PROB, \
                                        'comm_attempts': skip_info.get('comm_attempts', 0), \
                                        'comm_dropped': skip_info.get('comm_dropped', 0), \
                                        'map_msg_attempts': skip_info.get('map_msg_attempts', 0), \
                                        'map_msg_dropped': skip_info.get('map_msg_dropped', 0), \
                                        'pose_msg_attempts': skip_info.get('pose_msg_attempts', 0), \
                                        'pose_msg_dropped': skip_info.get('pose_msg_dropped', 0), \
                                        'graph_msg_attempts': skip_info.get('graph_msg_attempts', 0), \
                                        'graph_msg_dropped': skip_info.get('graph_msg_dropped', 0), \
                                        'retransmission_enabled': ENABLE_PRIORITY_RETRANSMISSION, \
                                        'retransmission_policy': RETRANSMISSION_POLICY, \
                                        'retransmission_budget': RETRANSMISSION_BUDGET_PER_PAIR, \
                                        'pending_retransmissions': skip_info.get('pending_retransmissions', 0), \
                                        'map_retrans_attempts': skip_info.get('map_retrans_attempts', 0), \
                                        'map_retrans_successes': skip_info.get('map_retrans_successes', 0), \
                                        'map_retrans_dropped': skip_info.get('map_retrans_dropped', 0), \
                                        'map_retrans_expired': skip_info.get('map_retrans_expired', 0), \
                                        'pose_retrans_attempts': skip_info.get('pose_retrans_attempts', 0), \
                                        'pose_retrans_successes': skip_info.get('pose_retrans_successes', 0), \
                                        'pose_retrans_dropped': skip_info.get('pose_retrans_dropped', 0), \
                                        'pose_retrans_expired': skip_info.get('pose_retrans_expired', 0), \
                                        'graph_retrans_attempts': skip_info.get('graph_retrans_attempts', 0), \
                                        'graph_retrans_successes': skip_info.get('graph_retrans_successes', 0), \
                                        'graph_retrans_dropped': skip_info.get('graph_retrans_dropped', 0), \
                                        'graph_retrans_expired': skip_info.get('graph_retrans_expired', 0) })
                if curr_test < NUM_TEST:
                    job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                    curr_test += 1

        # Sort CSV file by episode number
        df = pd.read_csv(csv_file_path)
        sorted_df = df.sort_values(by='eps')
        sorted_df.to_csv(csv_file_path, index=False)
        skipped_df = pd.read_csv(skipped_csv_file_path)
        skipped_df = skipped_df.sort_values(by='eps')
        skipped_df.to_csv(skipped_csv_file_path, index=False)

        dist_array = np.array(dist_history)
        print('|#Total attempted:', NUM_TEST)
        print('|#Valid episodes:', len(dist_history))
        print('|#Skipped episodes:', len(eps_skipped))
        print('|#Average (Max) length:', dist_array.mean() if len(dist_array) > 0 else np.nan)
        print('|#Length std:', dist_array.std() if len(dist_array) > 0 else np.nan)
        print('|#Eps skipped:', eps_skipped)


    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
 

@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number):
        """ Execute simulation episode and gather experience tuples & metrics """
        n_agent = np.random.randint(NUM_ROBOTS_MIN, NUM_ROBOTS_MAX+1, 1)[0]     
        worker = TestWorker(self.meta_agent_id, n_agent, self.local_network, episode_number, device=self.device, save_image=SAVE_GIFS, greedy=True)
        success = worker.work(episode_number)

        perf_metrics = worker.perf_metrics
        return success, perf_metrics, n_agent

    def job(self, weights, episode_number):
        """ Executes simulation episode """
        print(GREEN, "starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id), NC)
        
        # Set the local weights to the global weight values from the master network
        self.set_weights(weights)

        success, metrics, n_agent = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
            "n_agent": n_agent
        }

        return success, metrics, info


if __name__ == '__main__':
    ray.init()
    print("Welcome to IR2-MARL Exploration Inference Sim!")
    for i in range(NUM_RUN):
        run_test()
