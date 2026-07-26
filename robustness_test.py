"""
LMTE-style robustness tests for GEANT.
Tests: link failures, traffic bursts, natural drift.

Usage: python robustness_test.py [--checkpoint checkpoints/combined_best.pt]
"""

import sys, torch, os, numpy as np
sys.path.insert(0, '.')
from te_framework.topology import Topology
from te_framework.networks.gnn import TEGNNPolicy, GNNValueNetwork
from te_framework.traffic import TrafficLoader
from te_framework.env import TEEnv
from te_framework.agents.combined_cmdp import CombinedCMDPAgent
from train_cmdp import evaluate

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/combined_best.pt'
assert os.path.exists(CKPT), f'Checkpoint not found: {CKPT}'
TOPO = Topology('data/GEANT', max_paths_per_pair=8)
CKPT_D = torch.load(CKPT, map_location='cuda')
# Detect if checkpoint is from temporal model
IS_TEMPORAL = any('temp_enc' in k for k in CKPT_D['policy'])
if IS_TEMPORAL:
    print('  Detected temporal checkpoint, using temporal mode')
else:
    print('  Detected non-temporal checkpoint')

def make_agent_env(topo=TOPO, traffic=None):
    if traffic is None:
        traffic = TrafficLoader('data/GEANTTM2', topo.num_nodes)
    env = TEEnv(topo, traffic, device='cuda')
    env.precompute_ecmp()
    if IS_TEMPORAL:
        policy = TEGNNPolicy(topo, hidden_dim=128, max_k=topo.max_k, temporal=True, history_len=12).cuda()
    else:
        policy = TEGNNPolicy(topo, hidden_dim=128, max_k=topo.max_k).cuda()
    value = GNNValueNetwork(topo, hidden_dim=128).cuda()
    policy.load_state_dict(CKPT_D['policy'], strict=False)
    value.load_state_dict(CKPT_D['value'], strict=False)
    agent = CombinedCMDPAgent(policy, value, env.path_mask,
        ['mean_util','overload_ratio','p95_util'],
        {'mean_util':0.3,'overload_ratio':0.1,'p95_util':0.5}, device='cuda')
    return agent, env

print(f'Loading checkpoint: {CKPT}')
print('=' * 60)

# ─── 1. Baseline ───
agent, env = make_agent_env()
res = evaluate(agent, env, 512)
print(f'1. Baseline (no fault): MLU={res["avg_mlu"]:.4f}')
baseline_mlu = res["avg_mlu"]

# ─── 2. Link Failures ───
print(f'\n{"="*60}')
print(f'2. Link Failures')
print(f'{"="*60}')
for num_failures in [1, 3, 5, 10]:
    traffic = TrafficLoader('data/GEANTTM2', TOPO.num_nodes)
    env = TEEnv(TOPO, traffic, device='cuda')
    rng = np.random.RandomState(num_failures * 100)
    faulty = rng.choice(env.num_links, num_failures, replace=False)
    env.link_caps[faulty] *= 0.1
    env.precompute_ecmp()
    policy = TEGNNPolicy(TOPO, hidden_dim=128, max_k=TOPO.max_k).cuda()
    value = GNNValueNetwork(TOPO, hidden_dim=128).cuda()
    policy.load_state_dict(CKPT_D['policy'], strict=False)
    value.load_state_dict(CKPT_D['value'], strict=False)
    agent = CombinedCMDPAgent(policy, value, env.path_mask,
        ['mean_util','overload_ratio','p95_util'],
        {'mean_util':0.3,'overload_ratio':0.1,'p95_util':0.5}, device='cuda')
    res = evaluate(agent, env, 512)
    degrad = (res["avg_mlu"] - baseline_mlu) / baseline_mlu * 100
    print(f'  {num_failures} link(s) down: MLU={res["avg_mlu"]:.4f} ({degrad:+.1f}% vs baseline)')

# ─── 3. Traffic Bursts ───
# Fix: amplify traffic (not add noise) to avoid normalization masking
print(f'\n{"="*60}')
print(f'3. Traffic Bursts')
print(f'{"="*60}')
for scale in [2, 5, 10, 20, 30]:
    traffic = TrafficLoader('data/GEANTTM2', TOPO.num_nodes)
    real = traffic.get_real_traffic(normalize=False)  # (T, N, N) in kbps
    # LMTE-style: per-OD-pair zero-mean Gaussian noise
    rng = np.random.RandomState(42)
    od_std = real.std(axis=0)  # (N, N) std per OD pair over time
    noise = rng.randn(*real.shape) * od_std * (scale / 10.0)
    for t in range(len(noise)):  # zero out self-pair noise per TM
        np.fill_diagonal(noise[t], 0.0)
    CONV = 100.0 * 8 / 300 / 1000  # TrafficLoader conversion factor
    traffic.raw_matrices = (real + noise) / CONV
    agent, env = make_agent_env(traffic=traffic)
    res = evaluate(agent, env, 512)
    degrad = (res["avg_mlu"] - baseline_mlu) / baseline_mlu * 100
    print(f'  Burst scale={scale:2d}:      MLU={res["avg_mlu"]:.4f} ({degrad:+.1f}% vs baseline)')

# ─── 4. Natural Drift ───
# Fix: use GEANTTM2 (test set), split into time segments
print(f'\n{"="*60}')
print(f'4. Natural Drift')
print(f'{"="*60}')
traffic_test = TrafficLoader('data/GEANTTM2', TOPO.num_nodes)
total_tms = traffic_test.num_tms
for start_pct, label in [(0, "0-25%"), (25, "25-50%"), (50, "50-75%")]:
    start = int(total_tms * start_pct / 100)
    end = int(total_tms * (start_pct + 25) / 100)
    tm_slice = traffic_test.get_real_traffic(normalize=False)[start:end].copy()
    class SliceLoader:
        def __init__(self, data):
            self.real_tm = data
            self.num_tms = len(data)
        def get_real_traffic(self, normalize=False):
            t = torch.tensor(self.real_tm, dtype=torch.float32)
            if normalize:
                return (t / t.max()).numpy()
            return self.real_tm
    traffic = SliceLoader(tm_slice)
    agent, env = make_agent_env(traffic=traffic)
    res = evaluate(agent, env, 512)
    degrad = (res["avg_mlu"] - baseline_mlu) / baseline_mlu * 100
    print(f'  Segment {label}:       MLU={res["avg_mlu"]:.4f} ({degrad:+.1f}% vs baseline)')

print(f'\n{"="*60}')
print("Done. Compare with LMTE paper Figs 10-11 and Table 1.")
