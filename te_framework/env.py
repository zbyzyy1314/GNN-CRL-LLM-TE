"""
Gym-style Traffic Engineering environment with batched GPU support.

State:  normalized traffic matrix (N, N) tensor on device
Action: per-pair path selection indices (num_pairs,)
Reward: -MLU (negative max link utilization)
"""

import numpy as np
import torch
from te_framework.topology import Topology
from te_framework.traffic import TrafficLoader
from te_framework.routing import ecmp_traffic_distribution, compute_mlu


class TEEnv:
    def __init__(self, topo: Topology, traffic: TrafficLoader,
                 device: str = 'cpu'):
        self.topo = topo
        self.traffic = traffic
        self.device = device

        self.num_pairs = topo.num_pairs
        self.num_nodes = topo.num_nodes
        self.num_links = topo.num_links
        self.max_k = topo.max_k

        # Traffic: normalized (state) + real (reward)
        norm = traffic.get_real_traffic(normalize=True)
        real = traffic.get_real_traffic(normalize=False)
        self.norm_tm = torch.tensor(norm, dtype=torch.float32, device=device)
        self.real_tm = torch.tensor(real, dtype=torch.float32, device=device)
        self.num_tms = self.traffic.num_tms

        # Pre-computed tensors on device
        self.path_mask = torch.tensor(topo.path_mask, dtype=torch.bool,
                                       device=device)
        self.link_mask = torch.tensor(topo.link_mask, dtype=torch.float32,
                                       device=device)  # (P, K, L)
        self.link_caps = torch.tensor(topo.link_capacities, dtype=torch.float32,
                                       device=device)

        # Source-destination for each pair
        sd = np.array(topo.pair_idx_to_sd, dtype=np.int64)
        self.pair_src = torch.tensor(sd[:, 0], device=device)  # (P,)
        self.pair_dst = torch.tensor(sd[:, 1], device=device)  # (P,)

    def get_states(self, indices):
        """Get normalized states for given TM indices. Returns (B, N, N)."""
        return self.norm_tm[indices]

    def get_temporal_states(self, batch_start, history_len=12):
        """Get sliding window of TMs for temporal causality.
        
        Returns state based on TM_{t-H} to TM_{t-1} (history),
        and evaluates on TM_t (current).
        
        Args:
            batch_start: current time step t (must be >= history_len)
            history_len: number of past TMs to use as state
            
        Returns:
            state: (B, history_len, N, N) normalized historical TMs
            target_idx: (B,) indices of current TM for reward computation
        """
        B = min(batch_start - history_len + 1, self.num_tms - batch_start)
        state = torch.stack([
            self.norm_tm[batch_start - history_len + i]
            for i in range(B)
        ], dim=0)  # (B, H, N, N)
        target_idx = torch.arange(batch_start, batch_start + B, device=self.device)
        return state, target_idx

    def precompute_ecmp(self):
        """Pre-compute ECMP loads once for fast evaluation."""
        import numpy as np
        from te_framework.routing import ecmp_traffic_distribution
        real = self.traffic.get_real_traffic(normalize=False)
        loads = [ecmp_traffic_distribution(self.topo, real[i]) for i in range(self.num_tms)]
        self._ecmp_loads = torch.tensor(np.stack(loads), dtype=torch.float32, device=self.device)
        self._ecmp_mlu = (self._ecmp_loads / self.link_caps.unsqueeze(0)).max(dim=1).values
        return self.compute_constraints(self._ecmp_loads)

    def step_batch(self, states, actions):
        """
        Batched step: compute rewards for a batch of TMs and actions.

        Args:
            states: (B, N, N) normalized traffic matrices
            actions: (B, num_pairs) path indices per pair

        Returns:
            rewards: (B,) float tensor, -MLU for each TM
            mlus: (B,) float tensor
        """
        B = states.shape[0]

        # Get real traffic for these indices
        # states are normalized; we need real traffic at the same indices
        # We receive states directly so need to match them to real traffic
        # Easiest: compute from states (which are max-normalized) and real_tm
        # Actually: we need to map from normalized back. Let's pass indices.
        # But the current interface passes states... let me add a step_batch_idx

        # For now, use a simpler approach: the agent passes indices, not states
        raise NotImplementedError("Use step_batch_idx instead")

    def step_batch_idx(self, tm_indices, actions):
        """
        Batched step with TM indices for efficiency.

        Args:
            tm_indices: (B,) long tensor, indices into traffic matrices
            actions: (B, num_pairs) long tensor, path indices per pair

        Returns:
            rewards: (B,) float tensor
            mlus: (B,) float tensor
        """
        B = tm_indices.shape[0]

        # Get real traffic: (B, N, N)
        tm = self.real_tm[tm_indices]

        # Get demand per pair: tm[s, d] → (B, num_pairs)
        demands = tm[:, self.pair_src, self.pair_dst]  # (B, P)

        # For each (batch, pair), get the link mask for selected path k
        # link_mask: (P, K, L), actions: (B, P)
        # Expand pair indices to match batch: (B, P)
        p_idx = torch.arange(self.num_pairs, device=self.device).unsqueeze(0).expand(B, -1)
        selected_mask = self.link_mask[p_idx, actions]  # (B, P, L)

        # Link loads: sum over pairs of demand * mask
        link_loads = (demands.unsqueeze(-1) * selected_mask).sum(dim=1)  # (B, L)

        # MLU
        utilization = link_loads / self.link_caps.unsqueeze(0)  # (B, L)
        mlu = utilization.max(dim=1).values  # (B,)

        return -mlu, mlu, link_loads

    def compute_constraints(self, link_loads):
        """Compute utilization-based constraint costs (all in [0, ~2])."""
        B, L = link_loads.shape
        util = link_loads / self.link_caps.unsqueeze(0)
        util = torch.clamp(util, 0.0, 2.0)
        sorted_u = util.sort(dim=1).values
        p95_idx = max(int(0.95 * L), L - 1)
        # CVaR_0.95 = mean of top 5% most utilized links
        cvar_idx = max(int(0.95 * L), 1)
        cvar = sorted_u[:, cvar_idx:].mean(dim=1)
        return {
            'max_util': util.max(dim=1).values,
            'mean_util': util.mean(dim=1),
            'overload_ratio': (util > 0.8).float().mean(dim=1),
            'p95_util': sorted_u[:, p95_idx],
            'cvar_util': cvar,
        }

    def get_ecmp_mlu_batch(self, tm_indices):
        """Fast indexed ECMP MLU (if precomputed) or CPU fallback."""
        if hasattr(self, '_ecmp_mlu'):
            return self._ecmp_mlu[tm_indices]
        mlus = []
        real_np = self.traffic.get_real_traffic(normalize=False)
        for idx in tm_indices.cpu().numpy():
            loads = ecmp_traffic_distribution(self.topo, real_np[int(idx)])
            mlus.append(compute_mlu(self.topo, loads))
        return torch.tensor(mlus, dtype=torch.float32, device=self.device)
