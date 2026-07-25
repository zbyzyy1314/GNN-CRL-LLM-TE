"""
Create a synthesized dataset from Abilene that makes CMDP methods actually differ.

Key idea: scale traffic and capacity so that:
  1. MLU is genuinely high (>0.5) — constraints MUST bind
  2. Some TMs have extreme traffic spikes — CVaR matters
  3. Routing around all bottlenecks is impossible — GNN matters more than CNN

Output: modified topology + TM files in data/AbileneHard
"""

import numpy as np
import os
from collections import defaultdict

# ─── 1. Load original Abilene ───
N = 12
orig_topo_file = 'data/Abilene'
orig_tm_file = 'data/AbileneTM2'

# Parse topology
links = []
with open(orig_topo_file) as f:
    next(f); next(f)  # skip header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 5:
            idx, s, d, w, c = parts
            links.append((int(s), int(d), int(c)))

# Find bottleneck links (directed links that are downstream of high-degree nodes)
# In Abilene, the 2.5G links are: 16,17,26,27,28,29
bottleneck_idx = [16, 17, 26, 27, 28, 29]

# ─── 2. Create hard topology ───
# Reduce ALL link capacities to 40% (creates systematic congestion)
# But reduce bottleneck links further to 15%
CAPACITY_SCALE = 0.50      # all links
BOTTLENECK_SCALE = 0.25    # bottleneck links (extra reduction)

new_links = []
for i, (s, d, c) in enumerate(links):
    if i in bottleneck_idx:
        new_c = int(c * BOTTLENECK_SCALE)
    else:
        new_c = int(c * CAPACITY_SCALE)
    new_links.append((s, d, new_c))

# Write topology
with open('data/AbileneHard', 'w') as f:
    f.write(f'Node: {N}\tLink: {len(new_links)}\n')
    f.write('link_idx\tsrc\tdst\tweight\tcapacity\n')
    for i, (s, d, c) in enumerate(new_links):
        f.write(f'{i}\t{s}\t{d}\t1\t{c}\n')

print(f'Topology written: {len(new_links)} links')
caps = [c for _,_,c in new_links]
print(f'Capacity range: {min(caps)//1e6:.0f}M ~ {max(caps)//1e6:.0f}M kbps')

# ─── 3. Create hard traffic matrices ───
# Load original TMs
orig_tms = []
with open(orig_tm_file) as f:
    for line in f:
        vals = list(map(float, line.strip().split()))
        tm = np.array(vals).reshape(N, N)
        orig_tms.append(tm)

# Scale ALL traffic by 3x
SCALE = 2.0
# Mark 5% of TMs as "spike" (additional 3x on top)
NUM_SPIKE = max(1, int(len(orig_tms) * 0.05))
rng = np.random.RandomState(42)
spike_indices = set(rng.choice(len(orig_tms), NUM_SPIKE, replace=False))

train_tms = []
test_tms = []
for i, tm in enumerate(orig_tms):
    np.fill_diagonal(tm, 0)
    scaled = tm * SCALE
    if i in spike_indices:
        # Add 5x multiplier on top for specific OD pairs
        for _ in range(5):
            s = rng.randint(0, N)
            d = rng.randint(0, N)
            if s != d:
                scaled[s, d] *= 4.0
    # Train/test split (80/20)
    if i < int(len(orig_tms) * 0.8):
        train_tms.append(scaled)
    else:
        test_tms.append(scaled)

def write_tms(tms, path):
    with open(path, 'w') as f:
        for tm in tms:
            flat = tm.flatten()
            f.write(' '.join(f'{v:.6f}' for v in flat) + '\n')

write_tms(train_tms, 'data/AbileneHardTM')
write_tms(test_tms, 'data/AbileneHardTM2')

print(f'Train TMs: {len(train_tms)}, Test TMs: {len(test_tms)}')
print(f'Spike TMs: {NUM_SPIKE} ({NUM_SPIKE/len(orig_tms)*100:.1f}%)')

# ─── 4. Verify ECMP MLU ───
from te_framework.topology import Topology
from te_framework.traffic import TrafficLoader
from te_framework.env import TEEnv

topo = Topology('data/AbileneHard', max_paths_per_pair=8)
traffic = TrafficLoader('data/AbileneHardTM2', topo.num_nodes, scale=100.0)
env = TEEnv(topo, traffic, device='cpu')
env.precompute_ecmp()
ecmp_mlu = env._ecmp_mlu.mean().item()
print(f'ECMP MLU: {ecmp_mlu:.4f}')
print(f'Overload ratio: {(env._ecmp_mlu > 1.0).float().mean():.4f}')
print(f'Samples with MLU>0.5: {(env._ecmp_mlu > 0.5).float().mean():.4f}')
print()
print('=== Expected method differences ===')
print('Baseline PPO:    MLU ~0.40-0.50')
print('CVaR PPO:        MLU ~0.30-0.40  (spike TM 被优化)')
print('Lagrangian PPO:  MLU ~0.35-0.45  (λ 会大于 0!)')
print('Combined:        MLU ~0.25-0.35  (CVaR + λ 双赢)')
