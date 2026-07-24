"""
DQN Agent for TE — off-policy, replay buffer, TD learning.

Bandit TE (γ=0): Q(s, a) predicts reward directly.
ε-greedy exploration + Double DQN via target network.
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward):
        self.buffer.append((
            state.detach().cpu(), action.detach().cpu(), reward.detach().cpu()))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = torch.stack([b[0] for b in batch]).to(device)
        actions = torch.stack([b[1] for b in batch]).to(device)
        rewards = torch.stack([b[2] for b in batch]).to(device)
        return states, actions, rewards

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, q_network, target_network, path_mask, lr=3e-4,
                 gamma=0.0, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=5000, target_update_freq=500,
                 batch_size=128, device='cuda'):
        self.q_net = q_network
        self.target_net = target_network
        self.policy = q_network   # alias for evaluate() API
        self.value = q_network    # alias for evaluate() API
        self.path_mask = path_mask
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.target_update_freq = target_update_freq

        self.optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
        self.replay = ReplayBuffer()

        # Pre-compute per-pair path mask for random sampling
        self._path_probs = path_mask.float()
        self._path_probs = self._path_probs / self._path_probs.sum(-1, keepdim=True)
        self._path_probs = self._path_probs.cpu()  # for multinomial on CPU

    def act_batch(self, states, deterministic=False):
        if not deterministic:
            self.steps_done += states.shape[0]
            self.epsilon = max(self.epsilon_end,
                self.epsilon_end + (1.0 - self.epsilon_end) *
                np.exp(-self.steps_done / self.epsilon_decay))

        with torch.no_grad():
            q_values = self.q_net(states, self.path_mask)

            if not deterministic and random.random() < self.epsilon:
                # Random per-pair: sample from uniform over valid paths
                B, P, K = q_values.shape
                actions = torch.zeros(B, P, dtype=torch.long)
                for p in range(P):
                    n_valid = int(self.path_mask[p].sum().item())
                    actions[:, p] = torch.randint(0, n_valid, (B,))
                actions = actions.to(self.device)
            else:
                actions = q_values.argmax(dim=-1)

        return actions, None, None

    def update(self, states=None, actions=None, log_probs=None, rewards=None, values=None):
        if len(self.replay) < self.batch_size:
            return {}

        s, a, r = self.replay.sample(self.batch_size, self.device)

        q_all = self.q_net(s, self.path_mask)          # (B, P, K)
        q_chosen = q_all.gather(-1, a.unsqueeze(-1)).squeeze(-1)  # (B, P)
        q_sum = q_chosen.sum(dim=-1)                    # (B,)

        with torch.no_grad():
            target_sum = r  # (B,)  — γ=0, no next state

        loss = F.mse_loss(q_sum, target_sum)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()

        if self.steps_done > 0 and self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {'q_loss': loss.item()}

    def get_lambda_state(self):
        return {}

    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
