import sys, torch, numpy as np, time, argparse
sys.path.insert(0, '.')
from te_framework.topology import Topology
from te_framework.env import TEEnv
from te_framework.networks.gnn import TEGNNPolicy
from te_framework.traffic import TrafficLoader

p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, default=42)
args, _ = p.parse_known_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)

device = 'cuda'
topo = Topology('data/LMTE_correct_topo', max_paths_per_pair=8)
train = TrafficLoader('data/LMTE_train.txt', topo.num_nodes)
test = TrafficLoader('data/LMTE_test.txt', topo.num_nodes)
env = TEEnv(topo, train, device=device)
test_env = TEEnv(topo, test, device=device)
env.precompute_ecmp(); test_env.precompute_ecmp()
print(f'N={topo.num_nodes} L={topo.num_links} P={topo.num_pairs}')
print(f'Train: {env.num_tms}, Test: {test_env.num_tms}')

policy = TEGNNPolicy(topo, hidden_dim=128, max_k=topo.max_k).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
lm = env.link_mask; lc = env.link_caps
sd = torch.tensor(topo.pair_idx_to_sd, device=device)
cb = 128; best_mlu = float('inf')

def evaluate_dir(test_env, policy, cb):
    policy.eval()
    all_m = []
    with torch.no_grad():
        idx = torch.arange(test_env.num_tms, device=device)
        for start in range(0, test_env.num_tms, cb):
            end = min(start+cb, test_env.num_tms)
            idx_b = idx[start:end]
            states = test_env.get_states(idx_b)
            B = states.shape[0]
            logits = policy(states, test_env.path_mask)
            ratios = torch.softmax(logits.view(B, topo.num_pairs, -1), dim=-1)
            demands = test_env.real_tm[idx_b][:, sd[:,0], sd[:,1]]
            loads = torch.einsum('bp,bpk,pkl->bl', demands, ratios, lm)
            mlu = (loads / lc.unsqueeze(0)).max(dim=1).values
            all_m.append(mlu.cpu())
    all_m = torch.cat(all_m)
    policy.train()
    return {'avg_mlu': all_m.mean().item()}

# Train
for epoch in range(1, 41):
    ep_m = []; t0 = time.time()
    perm = torch.randperm(env.num_tms, device=device)
    for start in range(0, env.num_tms, cb):
        end = min(start+cb, env.num_tms)
        idx_b = perm[start:end]
        states = env.get_states(idx_b)
        B = states.shape[0]
        logits = policy(states, env.path_mask)
        ratios = torch.softmax(logits.view(B, topo.num_pairs, -1), dim=-1)
        ratios = ratios * env.path_mask.unsqueeze(0)
        ratios = ratios / (ratios.sum(dim=-1, keepdim=True) + 1e-8)
        demands = env.real_tm[idx_b][:, sd[:,0], sd[:,1]]
        loads = torch.einsum('bp,bpk,pkl->bl', demands, ratios, lm)
        mlu = (loads / lc.unsqueeze(0)).max(dim=1).values.mean()
        optimizer.zero_grad(); mlu.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        ep_m.append(mlu.item())
    
    if epoch % 5 == 0 or epoch == 1:
        res = evaluate_dir(test_env, policy, cb)
        tag = ' *BEST*' if res['avg_mlu'] < best_mlu else ''
        if res['avg_mlu'] < best_mlu: best_mlu = res['avg_mlu']
        print(f'Epoch {epoch:3d} | train={np.mean(ep_m):.4f} | test={res["avg_mlu"]:.4f}{tag} | {time.time()-t0:.0f}s')

res = evaluate_dir(test_env, policy, cb)
print(f'\nFinal test MLU: {res["avg_mlu"]:.4f} (best={best_mlu:.4f})')
print(f'Compare: baseline=1.98, CVaR=2.08')
