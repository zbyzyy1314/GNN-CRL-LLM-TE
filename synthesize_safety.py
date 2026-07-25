"""
Safety Layer专属数据集: 30% TM即使最优路由也MLU>1.0

设计:
  - 基于 AbileneHard3 拓扑 (全链路50%, 瓶颈25%)
  - A区 70%: 正常流量 (同 AbileneHard3 A区)
  - D区 30%: 强制 MLU>1.0 (取代原来的 B/C/D 区)
    方法: 对瓶颈链路(16,17,26,27,28,29)必经的OD对施加 15× 流量
    迫使即使最优路由也无法避免 MLU>1.0

预期:
  Baseline:        MLU≈0.55, 30% TM 的 MLU>1.0
  Lagrangian:      MLU≈0.52, 20% TM MLU>1.0 (λ不管硬约束)
  Safety only:     MLU≈0.50, 0% TM MLU>1.0 (硬修正, 但牺牲平均值)
  Combined:        MLU≈0.48, 0% TM MLU>1.0 (λ+修正双保险)
"""

import numpy as np
from te_framework.topology import Topology

N = 12
# Load original Abilene TM
orig_tms = []
with open('data/AbileneTM2') as f:
    for line in f:
        vals = list(map(float, line.strip().split()))
        orig_tms.append(np.array(vals).reshape(N, N))

rng = np.random.RandomState(42)
total = len(orig_tms)
train_n = int(total * 0.8)

# Find bottleneck pairs (pairs whose ALL K=3 paths go through bottleneck links)
topo = Topology('data/AbileneHard', max_paths_per_pair=3)
bottleneck_links = [16, 17, 26, 27, 28, 29]
bottleneck_pairs = set()

for p in range(topo.num_pairs):
    for lid in bottleneck_links:
        all_paths_use = True
        for k in range(topo.max_k):
            if topo.path_mask[p, k] and topo.link_mask[p, k, lid] == 0:
                all_paths_use = False
                break
        if all_paths_use:
            bottleneck_pairs.add(p)
bottleneck_pairs = sorted(bottleneck_pairs)
print(f'Bottleneck pairs (all paths forced): {len(bottleneck_pairs)}')
print(f'Pair indices: {bottleneck_pairs[:10]}...')

all_tms = []
for i, tm in enumerate(orig_tms):
    np.fill_diagonal(tm, 0)
    
    if i % 10 < 2:  # 30% → Safety regime (MLU>1.0 guaranteed)
        # Add 15× traffic on bottleneck pairs
        for p in bottleneck_pairs:
            s, d = topo.pair_idx_to_sd[p]
            tm[s, d] *= 15
    # else: 70% normal (keep original traffic)
    
    all_tms.append(tm * 1.5)  # 2× overall scaling

train = all_tms[:train_n]
test = all_tms[train_n:]

def write_tms(tms, path):
    with open(f'data/{path}', 'w') as f:
        for tm in tms:
            f.write(' '.join(f'{v:.6f}' for v in tm.flatten()) + '\n')

write_tms(train, 'SafetyTM')
write_tms(test, 'SafetyTM2')
import shutil
shutil.copy('data/AbileneHard', 'data/Safety')

import os
for f in os.listdir('data'):
    if 'Safety_k' in f:
        os.remove(f'data/{f}')

from te_framework.traffic import TrafficLoader
from te_framework.env import TEEnv

traffic = TrafficLoader('data/SafetyTM2', topo.num_nodes, scale=100.0)
env = TEEnv(topo, traffic, device='cpu')
env.precompute_ecmp()

print(f'\nSafety Dataset (30% MLU>1.0 forced):')
print(f'  ECMP MLU: {env._ecmp_mlu.mean():.3f}')
print(f'  MLU>1.0: {(env._ecmp_mlu>1).float().mean()*100:.0f}%')
print(f'  MLU>0.5: {(env._ecmp_mlu>0.5).float().mean()*100:.0f}%')
print(f'  mean_util: {env.compute_constraints(env._ecmp_loads)["mean_util"].mean():.3f}')
print(f'  overload_ratio: {env.compute_constraints(env._ecmp_loads)["overload_ratio"].mean():.3f}')
print(f'  p95_util: {env.compute_constraints(env._ecmp_loads)["p95_util"].mean():.3f}')
