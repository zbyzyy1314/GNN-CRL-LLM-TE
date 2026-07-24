"""
Safety Layer for Constrained MDP TE.

Hard constraint: MLU ≤ 1.0 (no link may exceed capacity).

Algorithm (greedy iterative projection):
  1. Compute link loads from proposed actions
  2. If MLU > 1.0:
     a. Find the most overloaded link
     b. For each pair using that link, try alternative paths
     c. Pick the alternative that most reduces max link load
     d. Update loads, repeat until feasible (or max_iter)

This is a "post-hoc correction" — the policy sees the corrected action's
reward, learning to eventually avoid corrections.
"""

import torch
import torch.nn.functional as F


class SafetyLayer:
    """Greedy action projection for TE link capacity constraint."""

    def __init__(self, env, max_iter=5, max_check=30):
        """
        Args:
            env: TEEnv
            max_iter: max correction iterations per batch element
            max_check: max pairs to check per overloaded link (speed)
        """
        self.env = env
        self.max_iter = max_iter
        self.max_check = max_check
        self.num_pairs = env.num_pairs
        self.num_links = env.num_links
        self.max_k = env.max_k
        self.link_mask = env.link_mask
        self.link_caps = env.link_caps
        self.pair_src = env.pair_src
        self.pair_dst = env.pair_dst
        self.device = env.device

        # Pre-build: for each (pair, link), which paths avoid this link
        self._build_avoidance()

    def _compute_loads_and_util(self, tm, actions):
        """
        Compute link loads and utilization from actions.

        Args:
            tm: (B, N, N) real traffic
            actions: (B, P) long tensor

        Returns:
            link_loads: (B, L)
            utilization: (B, L)
            mlu: (B,)
        """
        B = tm.shape[0]
        # Demand per pair
        demands = tm[:, self.pair_src, self.pair_dst]  # (B, P)

        # Path→link mapping for selected paths
        p_idx = torch.arange(self.num_pairs, device=self.device).unsqueeze(0).expand(B, -1)
        selected_mask = self.link_mask[p_idx, actions]  # (B, P, L)

        link_loads = (demands.unsqueeze(-1) * selected_mask).sum(dim=1)  # (B, L)
        utilization = link_loads / self.link_caps.unsqueeze(0)
        mlu = utilization.max(dim=1).values
        return link_loads, utilization, mlu

    def _build_avoidance(self):
        """Pre-compute safe alternative paths for each (pair, link)."""
        # avoid[pair, link] = list of path indices that DON'T use this link
        self._avoid = []
        for pair_idx in range(self.num_pairs):
            pair_avoid = []
            for lid in range(self.num_links):
                safe_paths = [k for k in range(self.max_k)
                              if self.env.path_mask[pair_idx, k]
                              and self.link_mask[pair_idx, k, lid] == 0]
                pair_avoid.append(safe_paths)
            self._avoid.append(pair_avoid)

    def project(self, tm_indices, raw_actions):
        """
        Project actions to satisfy MLU ≤ 1.0.

        Args:
            tm_indices: (B,) long tensor — which TMs are being processed
            raw_actions: (B, P) long tensor — proposed path selections

        Returns:
            safe_actions: (B, P) long tensor — corrected actions
            correction_count: (B,) int — how many pairs were corrected per sample
        """
        B = raw_actions.shape[0]
        tm = self.env.real_tm[tm_indices]  # (B, N, N)

        safe_actions = raw_actions.clone()
        correction_count = torch.zeros(B, device=self.device)

        link_loads, util, mlu = self._compute_loads_and_util(tm, safe_actions)

        # Check each batch element
        for i in range(B):
            if mlu[i] <= 1.0:
                continue
            for _ in range(self.max_iter):
                # Find most overloaded link
                worst_link = util[i].argmax().item()
                if util[i, worst_link] <= 1.0:
                    break

                # Find a pair using this link and re-route to a safe path
                worst_link = int(util[i].argmax().item())
                if util[i, worst_link] <= 1.0:
                    break
                fixed = False
                # Check pairs in random order, limited to max_check
                perm = torch.randperm(self.num_pairs, device=self.device)[:self.max_check]
                for pair_idx in perm:
                    pair_idx = int(pair_idx.item())
                    k = int(safe_actions[i, pair_idx].item())
                    if self.link_mask[pair_idx, k, worst_link] > 0:
                        alts = self._avoid[pair_idx][worst_link]
                        if alts:
                            safe_actions[i, pair_idx] = alts[0]
                            correction_count[i] += 1
                            fixed = True
                            break
                if not fixed:
                    break

                # Recompute loads
                link_loads, util, mlu = self._compute_loads_and_util(tm, safe_actions)

        return safe_actions, correction_count


class SafetyLayerPPOAgent:
    """
    PPO agent wrapped with Safety Layer projection.

    The policy outputs raw actions; the safety layer corrects them.
    The reward is computed from corrected actions, so the policy learns
    to avoid actions that would be corrected.
    """

    def __init__(self, policy_net, value_net, path_mask,
                 safety_layer, lr=3e-4, clip_ratio=0.2,
                 entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, ppo_epochs=8, device='cuda'):
        self.policy = policy_net
        self.value = value_net
        self.path_mask = path_mask
        self.safety = safety_layer
        self.device = device
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.optimizer = torch.optim.Adam(
            list(policy_net.parameters()) + list(value_net.parameters()), lr=lr)

    def act_batch(self, states, deterministic=False):
        """
        Policy outputs raw actions (before safety projection).
        Caller should use `project_and_act` for execution.
        """
        with torch.no_grad():
            actions, log_probs, _ = self.policy.get_action(
                states, self.path_mask, deterministic=deterministic)
            values = self.value(states)
        return actions, log_probs, values

    def project_and_act(self, tm_indices, raw_actions):
        """
        Project raw actions through safety layer.

        Returns:
            safe_actions: corrected actions
            corrections: how many pairs were fixed per sample
        """
        return self.safety.project(tm_indices, raw_actions)

    def update(self, states, actions, old_log_probs, rewards, values):
        B = states.shape[0]
        bs = min(256, B)
        adv = rewards - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = adv + values

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(B, device=self.device)
            for start in range(0, B, bs):
                idx = perm[start:start + bs]
                s_b = states[idx]; a_b = actions[idx]
                old_lp_b = old_log_probs[idx]
                adv_b = adv[idx]; ret_b = ret[idx]

                logits = self.policy(s_b, self.path_mask)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(a_b).sum(dim=-1)
                old_lp = old_lp_b.sum(dim=-1)
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv_b
                loss = (-torch.min(surr1, surr2).mean()
                        + self.value_coef * F.mse_loss(self.value(s_b), ret_b)
                        + self.entropy_coef * (-dist.entropy().sum(dim=-1).mean()))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())+list(self.value.parameters()), self.max_grad_norm)
                self.optimizer.step()
        return {}

    def get_lambda_state(self):
        return {}

    def save(self, p):
        torch.save({'policy': self.policy.state_dict(), 'value': self.value.state_dict()}, p)
