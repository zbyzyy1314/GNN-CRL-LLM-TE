import sys, torch, numpy as np, time
sys.path.insert(0, '.')
from te_framework.topology import Topology
from te_framework.env import TEEnv
from te_framework.networks.gnn import TEGNNLLMPolicy, GNNValueNetwork
from te_framework.llm_encoder import LLMEncoder
from te_framework.agents.combined_cmdp import CombinedCMDPAgent
from te_framework.traffic import TrafficLoader
from train_cmdp import evaluate

WINDOW, EPOCHS, BATCH, LLM_BATCH, LR = 12, 20, 64, 4, 1e-5
DEVICE = 'cuda'

topo = Topology('data/LMTE_correct_topo', max_paths_per_pair=8)
traffic = TrafficLoader('data/LMTE_correct_tm.txt', topo.num_nodes)
tms = traffic.get_real_traffic(normalize=False)
split = int(len(tms) * 0.8)

def make_env(tm_slice):
    raw = tm_slice / (100.0 * 8 / 300 / 1000)
    with open('/tmp/lmte_tmp.txt', 'w') as f:
        for tm in raw:
            f.write(' '.join(f'{v:.6f}' for v in tm.flat) + '\n')
    t = TrafficLoader('/tmp/lmte_tmp.txt', topo.num_nodes)
    e = TEEnv(topo, t, device=DEVICE)
    e.precompute_ecmp()
    return e

env = make_env(tms[:split])
test_env = make_env(tms[split:])
print(f'Train: {env.num_tms}, Test: {test_env.num_tms}')

print('Loading LLM...')
llm = LLMEncoder(hidden_dim=128, llm_dim=1536, llm_batch_size=LLM_BATCH,
                 use_4bit=False, device=DEVICE)
llm.load_model()
policy = TEGNNLLMPolicy(topo, llm, hidden_dim=128, max_k=topo.max_k,
                         temporal=True, history_len=WINDOW).to(DEVICE)
value = GNNValueNetwork(topo, hidden_dim=128).to(DEVICE)
agent = CombinedCMDPAgent(policy, value, env.path_mask,
    ['mean_util', 'overload_ratio', 'p95_util'],
    {'mean_util': 0.3, 'overload_ratio': 0.1, 'p95_util': 0.5}, device=DEVICE)

cb = BATCH; best_mlu = float('inf')
for epoch in range(1, EPOCHS + 1):
    ep_m = []; t0 = time.time()
    for t in range(WINDOW, env.num_tms, cb):
        b = min(cb, env.num_tms - t)
        if b <= 0:
            break
        states = torch.stack(
            [env.norm_tm[t - WINDOW + i:t + i] for i in range(b)], dim=0)
        target = torch.arange(t, t + b, device=DEVICE)
        raw_actions, log_probs, values = agent.act_batch(states)
        rewards, mlus, _ = env.step_batch_idx(target, raw_actions)
        agent.update(states, raw_actions, log_probs, rewards, values)
        ep_m.append(mlus.mean().item())

    if epoch % 5 == 0 or epoch == 1:
        res = evaluate(agent, test_env, cb)
        tag = ' *BEST*' if res['avg_mlu'] < best_mlu else ''
        if res['avg_mlu'] < best_mlu:
            best_mlu = res['avg_mlu']
        print(f'Epoch {epoch:3d} | MLU={np.mean(ep_m):.4f} | '
              f'test={res["avg_mlu"]:.4f}{tag} | {time.time() - t0:.0f}s')

res = evaluate(agent, test_env, cb)
print(f'\nFinal test MLU: {res["avg_mlu"]:.4f}')
print(f'Best MLU: {best_mlu:.4f}')
print('LMTE mlu.txt mean: 1.58')
print('Pure GNN (our):    2.47')
res = evaluate(agent, test_env, cb)
import torch
torch.save({"policy": policy.state_dict(), "value": value.state_dict()}, "checkpoints/lmte_llm.pt")
print(f"Checkpoint saved to checkpoints/lmte_llm.pt")
