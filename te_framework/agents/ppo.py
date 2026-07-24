"""
PPO Agent for path-level TE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PPOBuffer:
    """Simple rollout buffer for PPO."""

    def __init__(self, buffer_size: int, num_pairs: int, num_nodes: int):
        self.states = np.zeros((buffer_size, num_nodes, num_nodes), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_pairs), dtype=np.int64)
        self.log_probs = np.zeros((buffer_size, num_pairs), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = buffer_size

    def store(self, state, action, log_prob, reward, value):
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.values[idx] = value
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_batch(self, batch_size: int, device: str):
        """Sample a random batch."""
        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        return (
            torch.tensor(self.states[indices], dtype=torch.float32, device=device),
            torch.tensor(self.actions[indices], dtype=torch.long, device=device),
            torch.tensor(self.log_probs[indices], dtype=torch.float32, device=device),
            torch.tensor(self.rewards[indices], dtype=torch.float32, device=device),
            torch.tensor(self.values[indices], dtype=torch.float32, device=device),
        )

    def clear(self):
        self.ptr = 0
        self.size = 0

    def is_full(self):
        return self.size >= self.max_size


class PPOAgent:
    def __init__(self, policy_net, value_net, path_mask,
                 lr: float = 3e-4, gamma: float = 0.99,
                 clip_ratio: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10, batch_size: int = 64,
                 device: str = 'cpu'):
        """
        Args:
            policy_net: PathSelectionNetwork
            value_net: ValueNetwork
            path_mask: (num_pairs, max_k) boolean tensor
        """
        self.policy = policy_net
        self.value = value_net
        self.path_mask = path_mask
        self.device = device

        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(
            list(policy_net.parameters()) + list(value_net.parameters()),
            lr=lr
        )

    def act(self, state, deterministic=False):
        """
        Select actions for a single state.

        Returns:
            actions: (num_pairs,) numpy array
            log_probs: (num_pairs,) numpy array
            value: float
        """
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                                   device=self.device).unsqueeze(0)

        with torch.no_grad():
            actions, log_probs, _ = self.policy.get_action(
                state_t, self.path_mask, deterministic=deterministic)
            value = self.value(state_t)

        return (actions.squeeze(0).cpu().numpy(),
                log_probs.squeeze(0).cpu().numpy(),
                value.item())

    def update(self, buffer: PPOBuffer):
        """PPO update using buffer data."""
        states, actions, old_log_probs, rewards, values = buffer.get_batch(
            self.batch_size, self.device)

        # Standardize rewards
        advantages = rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            # Forward
            logits = self.policy(states, self.path_mask)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            new_log_probs = dist.log_prob(actions)  # (batch, num_pairs)
            entropy = dist.entropy()                 # (batch, num_pairs)

            # Sum over pairs
            new_log_prob = new_log_probs.sum(dim=-1)  # (batch,)
            old_log_prob = old_log_probs.sum(dim=-1)
            entropy_sum = entropy.sum(dim=-1)

            # PPO ratio
            ratio = torch.exp(new_log_prob - old_log_prob)

            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            new_values = self.value(states)
            value_loss = F.mse_loss(new_values, returns)

            # Entropy bonus
            entropy_loss = -entropy_sum.mean()

            loss = (policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value.parameters()),
                self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_sum.mean().item()
            n_updates += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }

    def save(self, path: str):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy'])
        self.value.load_state_dict(ckpt['value'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
