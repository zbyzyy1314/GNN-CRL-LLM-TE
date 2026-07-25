"""
Combined CMDP Agent: CVaR objective + Lagrangian constraints + Safety projection.

Reward:  r = -cvar_util - Σ λ_i * max(0, cost_i - threshold_i)
Safety:  post-hoc action projection when MLU > 1.0 (only if safety_layer provided)
"""

import torch
import torch.nn.functional as F


class CombinedCMDPAgent:
    def __init__(self, policy_net, value_net, path_mask,
                 constraint_names, constraint_thresholds,
                 safety_layer=None,
                 lr=3e-4, lr_lambda=0.01,
                 clip_ratio=0.2, entropy_coef=0.01,
                 value_coef=0.5, max_grad_norm=0.5,
                 ppo_epochs=8, device='cuda'):
        self.policy = policy_net
        self.value = value_net
        self.path_mask = path_mask
        self.safety = safety_layer
        self.device = device
        self.constraint_names = constraint_names
        self.thresholds = constraint_thresholds
        self.lr_lambda = lr_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

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

    def compute_combined_reward(self, constraint_costs):
        """
        Combined reward: -cvar_util - Σ λ_i * max(0, cost_i - threshold_i)
        """
        cvar = constraint_costs['cvar_util']
        lag_reward = -cvar
        violations = {}

        for name in self.constraint_names:
            cost = constraint_costs[name]
            excess = torch.relu(cost - self.thresholds[name])
            penalty = self.lambdas[name] * excess
            lag_reward = lag_reward - penalty
            violations[name] = excess.mean().item()

        return lag_reward, violations

    def update_multipliers(self, constraint_costs):
        for name in self.constraint_names:
            mean_cost = constraint_costs[name].mean()
            new_lam = self.lambdas[name] + self.lr_lambda * (mean_cost - self.thresholds[name])
            self.lambdas[name] = torch.clamp(new_lam, min=0.0)

    def update(self, states, actions, old_log_probs, combined_reward, values):
        B = states.shape[0]
        bs = min(256, B)
        adv = combined_reward - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = adv + values
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(B, device=self.device)
            for start in range(0, B, bs):
                idx = perm[start:start + bs]
                s_b, a_b = states[idx], actions[idx]
                old_lp_b = old_log_probs[idx]
                adv_b, ret_b = adv[idx], ret[idx]
                logits = self.policy(s_b, self.path_mask)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(a_b)      # (bs, P)
                old_lp = old_lp_b                # (bs, P)
                ratio = torch.exp(new_lp - old_lp)  # (bs, P)
                surr1 = ratio * adv_b.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv_b.unsqueeze(-1)
                loss = (-torch.min(surr1, surr2).mean()
                        + self.value_coef * F.mse_loss(self.value(s_b), ret_b)
                        - self.entropy_coef * dist.entropy().mean())
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())+list(self.value.parameters()), self.max_grad_norm)
                self.optimizer.step()
        return {}

    def get_lambda_state(self):
        return {name: self.lambdas[name].item() for name in self.constraint_names}

    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lambdas': {k: v.item() for k, v in self.lambdas.items()},
        }, path)


    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        # Skip topology-specific keys, load only compatible ones
        policy_state = {k: v for k, v in ckpt['policy'].items()
                        if k not in ('pair_src', 'pair_dst', 'edge_index', 'edge_feat')
                        and not k.startswith('enc.')
                        and not k.startswith('dec.')}
        missing, unexpected = self.policy.load_state_dict(policy_state, strict=False)
        if missing:
            print(f'  Skipped {len(missing)} topology-specific params')
        if ckpt.get('value') is not None:
            self.value.load_state_dict(ckpt['value'], strict=False)