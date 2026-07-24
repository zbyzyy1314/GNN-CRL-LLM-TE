"""
CVaR PPO Agent — optimizes tail risk directly.

Objective: minimize CVaR_0.95 (mean utilization of top 5% links)
Reward: r = -cvar_util

This shifts focus from the single worst link (MLU) to the worst 5% of links.
"""

import torch
import torch.nn.functional as F


class CVaRPPOAgent:
    def __init__(self, policy_net, value_net, path_mask,
                 lr=3e-4, entropy_coef=0.01, ppo_epochs=8, device='cuda'):
        self.policy = policy_net
        self.value = value_net
        self.path_mask = path_mask
        self.device = device
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.optimizer = torch.optim.Adam(
            list(policy_net.parameters()) + list(value_net.parameters()), lr=lr)

    def act_batch(self, states, deterministic=False):
        with torch.no_grad():
            actions, log_probs, _ = self.policy.get_action(
                states, self.path_mask, deterministic=deterministic)
            values = self.value(states)
        return actions, log_probs, values

    def update(self, states, actions, old_log_probs, rewards, values):
        B = states.shape[0]; bs = min(256, B)
        adv = rewards - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = adv + values
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(B, device=self.device)
            for start in range(0, B, bs):
                idx = perm[start:start+bs]
                s_b, a_b = states[idx], actions[idx]
                old_lp_b = old_log_probs[idx]
                adv_b, ret_b = adv[idx], ret[idx]
                logits = self.policy(s_b, self.path_mask)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(a_b).sum(dim=-1)
                old_lp = old_lp_b.sum(dim=-1)
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_b
                loss = (-torch.min(surr1, surr2).mean()
                        + 0.5 * F.mse_loss(self.value(s_b), ret_b)
                        + self.entropy_coef * (-dist.entropy().mean()))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())+list(self.value.parameters()), 0.5)
                self.optimizer.step()
        return {}

    def get_lambda_state(self):
        return {}

    def save(self, p):
        torch.save({'policy': self.policy.state_dict(), 'value': self.value.state_dict()}, p)
