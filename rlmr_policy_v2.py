import json
import os
from pathlib import Path

import numpy as np


class TabularRLMRPolicyV2:
    VERSION = 2
    ACTIONS = ("none", "map", "graph", "pose")
    EVAL_TIE_BREAK_ORDER = ("none", "map", "graph", "pose")

    def __init__(
        self,
        q_table_path,
        train=False,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.2,
        seed=0,
        initial_q=0.0,
    ):
        self.q_table_path = Path(q_table_path)
        self.train = train
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon if train else 0.0
        self.initial_q = initial_q
        self.rng = np.random.default_rng(seed)
        self.q_table = {}
        self.episodes = 0
        self.td_updates = 0
        self.active_transitions = {}
        self.episode_td_updates = 0
        self.episode_abs_td_error_sum = 0.0
        self.unseen_state_keys = set()
        self.load()

    def load(self):
        if not self.q_table_path.exists():
            return
        with self.q_table_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("version") != self.VERSION or "q_table" not in payload:
            raise ValueError(f"Not an RLMR-v2 Q table: {self.q_table_path}")
        self.q_table = payload["q_table"]
        self.episodes = int(payload.get("episodes", 0))
        self.td_updates = int(payload.get("td_updates", 0))

    def save(self):
        os.makedirs(self.q_table_path.parent, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "algorithm": "td_q_learning",
            "episodes": self.episodes,
            "td_updates": self.td_updates,
            "q_table": self.q_table,
        }
        temp_path = self.q_table_path.with_suffix(self.q_table_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        temp_path.replace(self.q_table_path)

    def state_key(self, state):
        return "|".join(str(item) for item in state)

    def action_index(self, action):
        return self.ACTIONS.index(action)

    def values_for_state(self, state, create=None):
        key = self.state_key(state)
        if key in self.q_table:
            return self.q_table[key]
        should_create = self.train if create is None else create
        if should_create:
            self.q_table[key] = [self.initial_q for _ in self.ACTIONS]
            return self.q_table[key]
        self.unseen_state_keys.add(key)
        return [self.initial_q for _ in self.ACTIONS]

    def choose_action(self, state, valid_actions):
        valid_actions = [action for action in valid_actions if action in self.ACTIONS]
        if not valid_actions:
            return "none"
        if self.train and self.rng.random() < self.epsilon:
            return valid_actions[int(self.rng.integers(len(valid_actions)))]

        values = self.values_for_state(state)
        best_value = max(values[self.action_index(action)] for action in valid_actions)
        best_actions = [action for action in valid_actions if values[self.action_index(action)] == best_value]
        if self.train and len(best_actions) > 1:
            return best_actions[int(self.rng.integers(len(best_actions)))]
        for action in self.EVAL_TIE_BREAK_ORDER:
            if action in best_actions:
                return action
        return best_actions[0]

    def select_action(self, pair_key, state, valid_actions):
        pair_key = tuple(pair_key)
        if self.train and pair_key in self.active_transitions:
            self._update_transition(pair_key, next_state=state, next_valid_actions=valid_actions, done=False)

        action = self.choose_action(state, valid_actions)
        if self.train:
            self.active_transitions[pair_key] = {
                "state": tuple(state),
                "action": action,
                "reward": 0.0,
            }
        return action

    def add_reward(self, pair_key, reward):
        if not self.train:
            return
        transition = self.active_transitions.get(tuple(pair_key))
        if transition is not None:
            transition["reward"] += float(reward)

    def add_global_reward(self, reward):
        if not self.train:
            return
        for transition in self.active_transitions.values():
            transition["reward"] += float(reward)

    def _update_transition(self, pair_key, next_state=None, next_valid_actions=None, done=False):
        transition = self.active_transitions.pop(tuple(pair_key), None)
        if transition is None:
            return

        values = self.values_for_state(transition["state"], create=True)
        action_index = self.action_index(transition["action"])
        target = transition["reward"]
        if not done:
            next_valid_actions = [
                action for action in (next_valid_actions or ["none"])
                if action in self.ACTIONS
            ]
            if not next_valid_actions:
                next_valid_actions = ["none"]
            next_values = self.values_for_state(next_state, create=True)
            target += self.gamma * max(next_values[self.action_index(action)] for action in next_valid_actions)

        td_error = target - values[action_index]
        values[action_index] += self.alpha * td_error
        self.td_updates += 1
        self.episode_td_updates += 1
        self.episode_abs_td_error_sum += abs(td_error)

    def finish_episode(self, terminal_reward=0.0, targeted_rewards=None):
        if not self.train:
            self.active_transitions = {}
            return
        targeted_rewards = targeted_rewards or {}
        for pair_key in list(self.active_transitions):
            self.add_reward(pair_key, terminal_reward + targeted_rewards.get(pair_key, 0.0))
            self._update_transition(pair_key, done=True)
        self.episodes += 1

    def num_states(self):
        return len(self.q_table)

    def mean_abs_td_error(self):
        if self.episode_td_updates == 0:
            return 0.0
        return self.episode_abs_td_error_sum / self.episode_td_updates

    def unseen_state_count(self):
        return len(self.unseen_state_keys)
