"""
Revised Safety dataset: 15% correctable TMs + mild overall load.
Safety can bring MLU>1.0 from ~20% down to 0%.
"""

import numpy as np
from te_framework.topology import Topology
from te_framework.traffic import TrafficLoader
from te_framework.env import TEEnv
import shutil, os

N = 12
orig_tms = []
with open('data/AbileneTM2') as f:
    for line in f:
        v = list(map(float, line.strip().split()))
        orig_tms.append(np.array(v).reshape(N, N))

topo = Topology('data/AbileneHard', max_paths_per_pair=3)
rng = np.random.RandomState(42)
total = len(orig_tms)
train_n = int(total * 0.8)
bottleneck_links = [16, 17, 26, 27, 28, 29]

correctable = []
for p in range(topo.num_pairs):
    has_bottleneck = False
    has_alternative = False
    for k in range(topo.max_k):
        if topo.path_mask[p, k]:
            uses = any(topo.link_mask[p, k, lid] for lid in bottleneck_links)
            if uses: has_bottleneck = True
            else: has_alternative = True
    if has_bottleneck and has_alternative:
        correctable.append(p)
print(f'Correctable pairs: {len(correctable)}')

all_tms = []
for i, tm in enumerate(orig_tms):
    np.fill_diagonal(tm, 0)
    if i % 10 == 0:
        for p in rng.choice(correctable, min(10, len(correctable)), replace=False):
            s, d = topo.pair_idx_to_sd[p]
            tm[s, d] *= 3
    all_tms.append(tm * 1.2)

def write_tms(tms, path):
    with open(path, 'w') as f:
        for tm in tms:
            f.write(' '.join(f'{v:.6f}' for v in tm.flatten()) + '\n')

write_tms(all_tms[:train_n], 'data/Safety2TM')
write_tms(all_tms[train_n:], 'data/Safety2TM2')
shutil.copy('data/AbileneHard', 'data/Safety2')
for f in os.listdir('data'):
    if 'Safety2_k' in f: os.remove('data/' + f)

traffic = TrafficLoader('data/Safety2TM2', topo.num_nodes, scale=100.0)
env = TEEnv(topo, traffic, device='cpu')
env.precompute_ecmp()
cc = env.compute_constraints(env._ecmp_loads)

print(f'Safety2 Dataset:')
print(f'  ECMP MLU: {env._ecmp_mlu.mean():.3f}')
print(f'  MLU>1.0: {(env._ecmp_mlu>1).float().mean()*100:.0f}%')
print(f'  mean_util={cc["mean_util"].mean():.3f}  overload={cc["overload_ratio"].mean():.3f}  p95={cc["p95_util"].mean():.3f}')
