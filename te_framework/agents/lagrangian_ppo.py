"""
Lagrangian PPO Agent for Constrained MDP Traffic Engineering.

Key idea: maintain one Lagrange multiplier per constraint.
The "reward" for PPO becomes:
    r_lagrangian = -MLU - Σ λ_i * max(0, cost_i - threshold_i)

After each PPO update, λ_i is adjusted:
    λ_i ← max(0, λ_i + lr_λ * (mean(cost_i) - threshold_i))
"""

import torch
import torch.nn.functional as F


class LagrangianPPOAgent:
    def __init__(self, policy_net, value_net, path_mask,
                 constraint_names, constraint_thresholds,
                 lr=3e-4, lr_lambda=0.01,
                 clip_ratio=0.2, entropy_coef=0.01,
                 value_coef=0.5, max_grad_norm=0.5,
                 ppo_epochs=8, device='cuda'):
        """
        Args:
            constraint_names: list of str, e.g. ['mean_delay', 'overload_ratio']
            constraint_thresholds: dict {name: threshold_value}
            lr_lambda: learning rate for Lagrange multiplier updates
        """
        self.policy = policy_net
        self.value = value_net
        self.path_mask = path_mask
        self.device = device

        self.constraint_names = constraint_names
        self.thresholds = constraint_thresholds
        self.lr_lambda = lr_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

        # Lagrange multipliers: one per constraint, initialized to 0
        self.lambdas = {name: torch.tensor(0.0, device=device)
                        for name in constraint_names}

        self.optimizer = torch.optim.Adam(
            list(policy_net.parameters()) + list(value_net.parameters()), lr=lr)

    def act_batch(self, states, deterministic=False):
        with torch.no_grad():
            actions, log_probs, _ = self.policy.get_action(
                states, self.path_mask, deterministic=deterministic)
            values = self.value(states)
        return actions, log_probs, values

    def compute_lagrangian_reward(self, rewards, constraint_costs):
        """
        Compute Lagrangian-modified reward.

        Args:
            rewards: (B,) tensor, original reward (-MLU)
            constraint_costs: dict of (B,) tensors

        Returns:
            lag_rewards: (B,) tensor
            constraint_violations: dict of float
        """
        lag_rewards = rewards.clone()
        violations = {}

        for name in self.constraint_names:
            cost = constraint_costs[name]
            threshold = self.thresholds[name]
            lam = self.lambdas[name]

            # Penalty: λ * max(0, cost - threshold)
            excess = torch.relu(cost - threshold)
            penalty = lam * excess

            lag_rewards = lag_rewards - penalty
            violations[name] = excess.mean().item()

        return lag_rewards, violations

    def update_multipliers(self, constraint_costs):
        """
        Update Lagrange multipliers after each batch.

        λ ← max(0, λ + lr_λ * (mean(cost) - threshold))
        """
        for name in self.constraint_names:
            mean_cost = constraint_costs[name].mean()
            threshold = self.thresholds[name]
            new_lam = self.lambdas[name] + self.lr_lambda * (mean_cost - threshold)
            self.lambdas[name] = torch.clamp(new_lam, min=0.0)

    def update(self, states, actions, old_log_probs, lag_rewards, values):
        """Standard PPO update (same as baseline, but with Lagrangian reward)."""
        B = states.shape[0]
        batch_size = min(256, B)
        advantages = lag_rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values

        stats = {'p_loss': 0, 'v_loss': 0, 'entropy': 0}
        n_up = 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(B, device=self.device)
            for start in range(0, B, batch_size):
                idx = perm[start:start + batch_size]
                s_b, a_b = states[idx], actions[idx]
                old_lp_b = old_log_probs[idx]
                adv_b, ret_b = advantages[idx], returns[idx]

                logits = self.policy(s_b, self.path_mask)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                new_lp = dist.log_prob(a_b)      # (bs, P)
                old_lp = old_lp_b                # (bs, P)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv_b.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio,
                                    1 + self.clip_ratio) * adv_b.unsqueeze(-1)
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(self.value(s_b), ret_b)
                loss = (policy_loss + self.value_coef * value_loss
                        + self.entropy_coef * (-entropy))

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm)
                self.optimizer.step()

                stats['p_loss'] += policy_loss.item()
                stats['v_loss'] += value_loss.item()
                stats['entropy'] += entropy.item()
                n_up += 1

        return {k: v / max(n_up, 1) for k, v in stats.items()}

    def get_lambda_state(self):
        """Return current Lagrange multiplier values for logging."""
        return {name: self.lambdas[name].item()
                for name in self.constraint_names}

    def save(self, path):
        data = {
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lambdas': {k: v.item() for k, v in self.lambdas.items()},
        }
        torch.save(data, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy'])
        self.value.load_state_dict(ckpt['value'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'lambdas' in ckpt:
            for k, v in ckpt['lambdas'].items():
                self.lambdas[k] = torch.tensor(v, device=self.device)
