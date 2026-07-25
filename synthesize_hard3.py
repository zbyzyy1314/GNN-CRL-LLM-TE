"""
Generate AbileneHard3: three-regime dataset that differentiates all CMDP methods.

Regime A (80%): Normal traffic, light load → all methods equal
Regime B (10%): Extreme spike on random pairs → CVaR shines (tail optimization)
Regime C (10%): Sustained overload on bottleneck links → Lagrangian λ activates
Regime D (5%): MLU > 1.0 unavoidable → Safety Layer fires (hard constraint)

Expected rankings:
  Baseline: worst (no tailoring)
  CVaR:     handles B, struggles with C (no λ)
  Lagrangian: handles C, ignores B (MLU objective)
  Safety:   handles D only
  Combined: best across all (CVaR+B + λ+C + Safety+D)
"""

import numpy as np
from te_framework.topology import Topology
from te_framework.traffic import TrafficLoader
from te_framework.env import TEEnv

N = 12
# Load original Abilene TM
orig_tms = []
with open('data/AbileneTM2') as f:
    for line in f:
        vals = list(map(float, line.strip().split()))
        orig_tms.append(np.array(vals).reshape(N, N))

rng = np.random.RandomState(42)
total = len(orig_tms)  # 9677
train_n = int(total * 0.8)
test_n = total - train_n

# Identify bottleneck links (nodes 3,6 are hub-spoke)
# Links 16,17 (3→6, 6→3) and 26,27 (2→9, 9→2) and 28,29 (4→10, 10→4)
# All these have 2.5G capacity in original
bottleneck_nodes = [(3,6), (6,3), (2,9), (9,2), (4,10), (10,4)]
# Also find which SD pairs MUST go through these bottlenecks
topo_orig = Topology('data/Abilene', max_paths_per_pair=8)

def pairs_use_link(link_src, link_dst):
    """Find SD pairs whose ALL paths go through this link."""
    result = []
    for p in range(topo_orig.num_pairs):
        s, d = topo_orig.pair_idx_to_sd[p]
        paths_via_link = 0
        total_paths = 0
        for k in range(topo_orig.max_k):
            if topo_orig.path_mask[p, k]:
                total_paths += 1
                if topo_orig.link_mask[p, k, topo_orig.link_sd_to_idx[(link_src, link_dst)]] > 0:
                    paths_via_link += 1
        if total_paths > 0 and paths_via_link == total_paths:
            result.append(p)
    return result

# Pre-compute which pairs are bottleneck-dependent
bottleneck_pairs = set()
for s, d in bottleneck_nodes:
    if (s, d) in topo_orig.link_sd_to_idx:
        bottleneck_pairs.update(pairs_use_link(s, d))
bottleneck_pairs = list(bottleneck_pairs)
print(f'Bottleneck pairs: {len(bottleneck_pairs)} out of {topo_orig.num_pairs}')

all_tms = []
for i, tm in enumerate(orig_tms):
    np.fill_diagonal(tm, 0)
    regime = 'A'
    
    if i % 10 == 0:  # 10% of samples → CVaR regime B (extreme spike)
        regime = 'B'
        # Add 10x traffic on 2 random OD pairs
        for _ in range(2):
            s = rng.randint(0, N); d = rng.randint(0, N)
            while s == d:
                s = rng.randint(0, N); d = rng.randint(0, N)
            tm[s, d] *= 10
        
    if i % 10 == 1:  # 10% of samples → Lagrangian regime C (bottleneck overload)
        regime = 'C'
        # Add 5x traffic on bottleneck-dependent pairs
        for p in bottleneck_pairs[:5]:
            s, d = topo_orig.pair_idx_to_sd[p]
            tm[s, d] *= 5
    
    if i % 20 == 0:  # 5% of samples → Safety regime D (extreme bottleneck)
        regime = 'D'
        # Add 8x traffic on ALL bottleneck pairs
        for p in bottleneck_pairs:
            s, d = topo_orig.pair_idx_to_sd[p]
            tm[s, d] *= 8
    
    all_tms.append(tm)

# Scale all traffic to make constraints bind
SCALE = 1.8
all_tms = [tm * SCALE for tm in all_tms]

# Split train/test
train = all_tms[:train_n]
test = all_tms[train_n:]

# Write
def write_tms(tms, path):
    with open(f'data/{path}', 'w') as f:
        for tm in tms:
            flat = tm.flatten()
            f.write(' '.join(f'{v:.6f}' for v in flat) + '\n')

write_tms(train, 'AbileneHard3TM')
write_tms(test, 'AbileneHard3TM2')

# Use same topology as AbileneHard
import shutil
shutil.copy('data/AbileneHard', 'data/AbileneHard3')

print(f'Train: {len(train)}, Test: {len(test)}')
print(f'Topology: data/AbileneHard3 (copy of AbileneHard)')

# Verify
import os
for f in os.listdir('data'):
    if 'AbileneHard3_k' in f:
        os.remove(f'data/{f}')

topo = Topology('data/AbileneHard3', max_paths_per_pair=3)
traffic = TrafficLoader('data/AbileneHard3TM2', topo.num_nodes, scale=100.0)
env = TEEnv(topo, traffic, device='cpu')
env.precompute_ecmp()

cc = env.compute_constraints(env._ecmp_loads)
print(f'K=3 | ECMP MLU={env._ecmp_mlu.mean():.3f}')
print(f'  mean_util={cc["mean_util"].mean():.3f}  overload={cc["overload_ratio"].mean():.3f}  p95={cc["p95_util"].mean():.3f}')
print(f'  Samples with MLU>1.0: {(env._ecmp_mlu>1).float().mean():.3f}')
print(f'  Samples with MLU>0.5: {(env._ecmp_mlu>0.5).float().mean():.3f}')
