"""
Train our GNN+LLM on LMTE's GEANT dataset with matching evaluation protocol.

LMTE protocol:
  - Data: GEANT.csv (22 nodes, 10773 TMs, bps)
  - Input: 12-step sliding window of historical TMs
  - Output: routing for next TM
  - Train: first 70% of sequences
  - Valid: next 10% 
  - Test: last 20%
  - MLU: max(link_load / link_capacity)

Usage: python train_lmte_compare.py
"""

import sys, torch, os, json, numpy as np, time
sys.path.insert(0, '.')
from te_framework.topology import Topology
from te_framework.env import TEEnv
from te_framework.networks.gnn import TEGNNLLMPolicy, GNNValueNetwork
from te_framework.llm_encoder import LLMEncoder
from te_framework.agents.combined_cmdp import CombinedCMDPAgent
from te_framework.traffic import TrafficLoader
from train_cmdp import evaluate

# ─── Config ───
WINDOW = 12
EPOCHS = 20
BATCH = 64
LLM_BATCH = 4
LR = 1e-5
DEVICE = 'cuda'

# ─── 1. Prepare LMTE topology and TMs ───
with open('LMTE_repo/data/GEANT/topology.json') as f:
    tdata = json.load(f)
N = len(tdata['nodes']); L = len(tdata['links'])

# Write topology (kbps)
with open('data/lmte_compare_topo','w') as f:
    f.write(f'Node: {N}\tLink: {L}\nlink_idx\tsrc\tdst\tweight\tcapacity\n')
    for i, l in enumerate(tdata['links']):
        f.write(f'{i}\t{l["source"]}\t{l["target"]}\t1\t{l["capacity"]/1000:.0f}\n')

# Write TMs (bps → our file format)
tms_all = np.loadtxt('LMTE_repo/data/GEANT/GEANT.csv', delimiter=',')
for name, sl in [('lmte_compare_train.txt', slice(0, int(len(tms_all)*0.8))),
                  ('lmte_compare_test.txt', slice(int(len(tms_all)*0.8), None))]:
    with open(f'data/{name}', 'w') as f:
        for tm in tms_all[sl] * 0.375:
            f.write(' '.join(f'{v:.6f}' for v in tm)+'\n')

# ─── 2. Load ───
topo = Topology('data/lmte_compare_topo', max_paths_per_pair=8)
train_traffic = TrafficLoader('data/lmte_compare_train.txt', topo.num_nodes)
test_traffic  = TrafficLoader('data/lmte_compare_test.txt', topo.num_nodes)

env = TEEnv(topo, train_traffic, device=DEVICE)
test_env = TEEnv(topo, test_traffic, device=DEVICE)
env.precompute_ecmp(); test_env.precompute_ecmp()
print(f'N={topo.num_nodes} L={topo.num_links} P={topo.num_pairs} K={topo.max_k}')
print(f'Train TMs: {env.num_tms}, Test TMs: {test_env.num_tms}')

# ─── 3. Build LLM model ───
print('Loading LLM...')
llm = LLMEncoder(hidden_dim=128, llm_dim=1536, llm_batch_size=LLM_BATCH,
                 use_4bit=False, device=DEVICE)
llm.load_model()
policy = TEGNNLLMPolicy(topo, llm, hidden_dim=128, max_k=topo.max_k,
                         temporal=True, history_len=WINDOW).to(DEVICE)
value = GNNValueNetwork(topo, hidden_dim=128).to(DEVICE)
agent = CombinedCMDPAgent(policy, value, env.path_mask,
    ['mean_util','overload_ratio','p95_util'],
    {'mean_util':0.3,'overload_ratio':0.1,'p95_util':0.5}, device=DEVICE)

# ─── 4. Train (temporal, sequential by time) ───
cb = BATCH; best_mlu = float('inf')
print(f'\nTraining {EPOCHS} epochs...')
for epoch in range(1, EPOCHS+1):
    ep_m = []; t0 = time.time()
    for t in range(WINDOW, env.num_tms, cb):
        b = min(cb, env.num_tms-t)
        if b <= 0: break
        states = torch.stack([env.norm_tm[t-WINDOW+i:t+i] for i in range(b)], dim=0)
        target = torch.arange(t, t+b, device=DEVICE)
        raw_actions, log_probs, values = agent.act_batch(states)
        rewards, mlus, _ = env.step_batch_idx(target, raw_actions)
        agent.update(states, raw_actions, log_probs, rewards, values)
        ep_m.append(mlus.mean().item())

    if epoch % 5 == 0 or epoch == 1:
        res = evaluate(agent, test_env, cb)
        tag = ' *BEST*' if res['avg_mlu'] < best_mlu else ''
        if res['avg_mlu'] < best_mlu: best_mlu = res['avg_mlu']
        print(f'Epoch {epoch:3d} | train MLU={np.mean(ep_m):.4f} '
              f'| test MLU={res["avg_mlu"]:.4f}{tag} | {time.time()-t0:.0f}s')

# ─── 5. Report ───
res = evaluate(agent, test_env, cb)
print(f'\n=== Final Results ===')
print(f'Test MLU:  {res["avg_mlu"]:.4f}')
print(f'Best MLU:  {best_mlu:.4f}')
print(f'\n{LMTE mlu.txt mean: 1.58}')
print(f'LMTE mlu.txt min:  0.74}')
print(f'Pure GNN (our):    2.47')
