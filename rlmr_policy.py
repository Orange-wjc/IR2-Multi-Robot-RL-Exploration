import json
import os
from pathlib import Path

import numpy as np


class TabularRLMRPolicy:
    ACTIONS = ("none", "map", "graph", "pose")
    TIE_BREAK_ORDER = ("map", "graph", "pose", "none")

    def __init__(self, q_table_path, train=False, alpha=0.1, epsilon=0.1, seed=0, initial_q=0.0):
        self.q_table_path = Path(q_table_path)
        self.train = train
        self.alpha = alpha
        self.epsilon = epsilon if train else 0.0
        self.initial_q = initial_q
        self.rng = np.random.default_rng(seed)
        self.q_table = self.load()
        self.episode_transitions = []

    def load(self):
        if not self.q_table_path.exists():
            return {}
        with self.q_table_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(self):
        os.makedirs(self.q_table_path.parent, exist_ok=True)
        with self.q_table_path.open("w", encoding="utf-8") as f:
            json.dump(self.q_table, f, indent=2, sort_keys=True)

    def state_key(self, state):
        return "|".join(str(item) for item in state)

    def action_index(self, action):
        return self.ACTIONS.index(action)

    def values_for_state(self, state):
        key = self.state_key(state)
        if key not in self.q_table:
            self.q_table[key] = [self.initial_q for _ in self.ACTIONS]
        return self.q_table[key]

    def select_action(self, state, valid_actions):
        valid_actions = [action for action in valid_actions if action in self.ACTIONS]
        if not valid_actions:
            return "none"
        if self.train and self.rng.random() < self.epsilon:
            return valid_actions[int(self.rng.integers(len(valid_actions)))]

        values = self.values_for_state(state)
        best_value = max(values[self.action_index(action)] for action in valid_actions)
        best_actions = [action for action in valid_actions if values[self.action_index(action)] == best_value]
        for action in self.TIE_BREAK_ORDER:
            if action in best_actions:
                return action
        return best_actions[0]

    def record_decision(self, state, action, reward=0.0):
        if not self.train:
            return None
        self.episode_transitions.append({
            "state": state,
            "action": action,
            "reward": reward,
        })
        return len(self.episode_transitions) - 1

    def add_reward(self, transition_index, reward):
        if transition_index is None:
            return
        self.episode_transitions[transition_index]["reward"] += reward

    def finish_episode(self, terminal_reward):
        if not self.train:
            self.episode_transitions = []
            return
        for transition in self.episode_transitions:
            values = self.values_for_state(transition["state"])
            action_idx = self.action_index(transition["action"])
            target = terminal_reward + transition["reward"]
            values[action_idx] += self.alpha * (target - values[action_idx])
        self.episode_transitions = []

    def num_states(self):
        return len(self.q_table)
