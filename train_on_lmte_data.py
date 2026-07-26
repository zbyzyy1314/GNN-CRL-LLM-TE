"""
Train our model on LMTE's GEANT dataset for direct comparison.
Trains on first 75% time segments, tests on last 25%.
"""
import sys, torch, os, numpy as np, json
sys.path.insert(0, '.')
from te_framework.topology import Topology
from te_framework.env import TEEnv
from te_framework.networks.gnn import TEGNNPolicy, GNNValueNetwork
from te_framework.agents.combined_cmdp import CombinedCMDPAgent
from train_cmdp import evaluate
import time

TRAIN_RATIO = 0.75
TOPOLOGY_JSON = 'LMTE_repo/data/GEANT/topology.json'
TM_CSV = 'LMTE_repo/data/GEANT/GEANT.csv'

# ─── 1. Build topology and TM in our format ───
with open(TOPOLOGY_JSON) as f:
    topo_data = json.load(f)

N = len(topo_data['nodes'])
links = topo_data['links']

# Convert LMTE topology → our format
with open('data/LMTE_GEANT_topo', 'w') as f:
    f.write(f'Node: {N}\tLink: {len(links)}\n')
    f.write('link_idx\tsrc\tdst\tweight\tcapacity\n')
    for i, l in enumerate(links):
        cap_kbps = l['capacity'] / 1000  # bps → kbps
        f.write(f'{i}\t{l["source"]}\t{l["target"]}\t1\t{cap_kbps:.0f}\n')

# Convert LMTE TMs → our format  
tms = np.loadtxt(TM_CSV, delimiter=',')
# LMTE uses scale=1e9. We need to match: divide by 1e9×375
# LMTE data: bps / 1e9 = scaled value
# Our data: file_value * 0.002667 = kbps
# So: file_value = (bps / 1e9) / 0.002667 * 1000 = bps / 1e9 * 375,000
# File: file_value * 0.002667 = kbps
# bps → kbps: /1000
# We want: our_repr / 0.002667 = kbps = bps / 1000
# So: our_repr = bps / 1000 * 0.002667... no
# LMTE normalizes: train_tms / 1e9
# LMTE capacity: capacity_bps / 1e9 
# So MLU = (load/1e9) / (cap/1e9) = load/cap → same
# We just need raw bps converted to our file format

# Our TrafficLoader expects: raw_file_value * 0.002667 = kbps
# bps → kbps: /1000
# raw_file_value = kbps / 0.002667 = (bps/1000) / 0.002667 = bps * 0.375

total = len(tms)
split = int(total * TRAIN_RATIO)

for name, indices in [('LMTE_GEANT_train.txt', slice(0, split)), 
                        ('LMTE_GEANT_test.txt', slice(split, total))]:
    subset = tms[indices]
    with open(f'data/{name}', 'w') as f:
        for tm in subset * 0.375:  # bps → our format
            f.write(' '.join(f'{v:.6f}' for v in tm) + '\n')
    print(f'{name}: {len(subset)} TMs, range [{subset.min():.0f}, {subset.max():.0f}] bps')

# ─── 2. Load into our env ───
topo = Topology('data/LMTE_GEANT_topo', max_paths_per_pair=8)
print(f'\nTopology: {topo.num_nodes} nodes, {topo.num_links} links, {topo.num_pairs} pairs')



from te_framework.traffic import TrafficLoader
train_traffic = TrafficLoader('data/LMTE_GEANT_train.txt', topo.num_nodes)
test_traffic = TrafficLoader('data/LMTE_GEANT_test.txt', topo.num_nodes)

env = TEEnv(topo, train_traffic, device='cuda')
env.precompute_ecmp()
ecmp_c = env.compute_constraints(env._ecmp_loads)
print(f'ECMP MLU={env._ecmp_mlu.mean():.3f}, p95={ecmp_c["p95_util"].mean():.3f}')

test_env = TEEnv(topo, test_traffic, device='cuda')
test_env.precompute_ecmp()

# ─── 3. Train ───
policy = TEGNNPolicy(topo, hidden_dim=128, max_k=topo.max_k).cuda()
value = GNNValueNetwork(topo, hidden_dim=128).cuda()
agent = CombinedCMDPAgent(policy, value, env.path_mask,
    ['mean_util','overload_ratio','p95_util'],
    {'mean_util':0.3,'overload_ratio':0.1,'p95_util':0.5}, device='cuda')

print(f'\nTraining 40 epochs on LMTE GEANT (first 75% time)...')
cb = 128
best_mlu = float('inf')

for epoch in range(1, 41):
    perm = torch.randperm(env.num_tms, device='cuda')
    ep_m = []
    t0 = time.time()
    for start in range(0, env.num_tms, cb):
        end = min(start + cb, env.num_tms)
        idx_b = perm[start:end]
        states = env.get_states(idx_b)
        raw_actions, log_probs, values = agent.act_batch(states)
        actions = raw_actions
        rewards, mlus, loads = env.step_batch_idx(idx_b, actions)
        train_r = rewards
        agent.update(states, actions, log_probs, train_r, values)
        ep_m.append(mlus.mean().item())

    if epoch % 5 == 0 or epoch == 1:
        res = evaluate(agent, test_env, cb)
        imp = '+' if res['improvement'] >= 0 else ''
        if res['avg_mlu'] < best_mlu:
            best_mlu = res['avg_mlu']; tag = ' *BEST*'
        else:
            tag = ''
        print(f'Epoch {epoch:4d} | train_MLU={np.mean(ep_m):.4f} | '
              f'test_MLU={res["avg_mlu"]:.4f} ({imp}{res["improvement"]:.1f}%){tag} | {time.time()-t0:.0f}s')

res = evaluate(agent, test_env, cb)
print(f'\nFinal: MLU={res["avg_mlu"]:.4f} | p95={res["p95_util"]:.4f}')
print(f'Compare LMTE reported: ~0.07-0.10 on GEANT')
print(f'Training data: first 75% time, Test data: last 25% time')
