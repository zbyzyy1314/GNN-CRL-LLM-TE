import sys, torch, numpy as np, time, argparse
sys.path.insert(0, '.')
from te_framework.topology import Topology
from te_framework.env import TEEnv
from te_framework.networks.gnn import TEGNNPolicy
from te_framework.traffic import TrafficLoader

p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, default=42)
p.add_argument('--lam', type=float, default=0.0, help='Lagrangian: 0=MLU only, >0 = Lagrangian weight')
p.add_argument('--mean-util-threshold', type=float, default=0.5)
p.add_argument('--overload-threshold', type=float, default=0.2)
p.add_argument('--p95-util-threshold', type=float, default=1.5)
p.add_argument('--lr-lambda', type=float, default=0.02)
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

# Lagrangian multipliers
lambdas = {
    'mean_util': torch.tensor(args.lam, device=device),
    'overload_ratio': torch.tensor(args.lam, device=device),
    'p95_util': torch.tensor(args.lam, device=device),
}
thresholds = {
    'mean_util': args.mean_util_threshold,
    'overload_ratio': args.overload_threshold,
    'p95_util': args.p95_util_threshold,
}
constraint_names = ['mean_util', 'overload_ratio', 'p95_util']
use_lagrangian = args.lam > 0

if use_lagrangian:
    print(f'Lagrangian mode: thresholds={thresholds}, lr_lambda={args.lr_lambda}')
def evaluate_dir_full(test_env, policy, cb):
    policy.eval()
    all_m, all_u, all_o, all_p = [], [], [], []
    with torch.no_grad():
        idx = torch.arange(test_env.num_tms, device=device)
        for start in range(0, test_env.num_tms, cb):
            end = min(start+cb, test_env.num_tms)
            idx_b = idx[start:end]
            states = test_env.get_states(idx_b); B = states.shape[0]
            logits = policy(states, test_env.path_mask)
            ratios = torch.softmax(logits.view(B, topo.num_pairs, -1), dim=-1)
            ratios = ratios * test_env.path_mask.unsqueeze(0)
            ratios = ratios / (ratios.sum(dim=-1, keepdim=True) + 1e-8)
            demands = test_env.real_tm[idx_b][:, sd[:,0], sd[:,1]]
            loads = torch.einsum('bp,bpk,pkl->bl', demands, ratios, lm)
            util = loads / lc.unsqueeze(0)
            su = util.sort(dim=1).values
            all_m.append(util.max(dim=1).values.cpu())
            all_u.append(util.mean(dim=1).cpu())
            all_o.append((util > 0.8).float().mean(dim=1).cpu())
            # Match LP: numpy percentile with interpolation
            cpu_u = util.cpu().numpy()
            all_p.append(torch.tensor([np.percentile(u, 95) for u in cpu_u]))
    a = lambda x: torch.cat(x).mean().item()
    policy.train()
    return {'avg_mlu': a(all_m), 'mean_util': a(all_u),
            'overload_ratio': a(all_o), 'p95_util': a(all_p)}

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
        loss = mlu
        if use_lagrangian:
            costs = env.compute_constraints(loads)
            for name in constraint_names:
                excess = torch.relu(costs[name] - thresholds[name])
                loss = loss + (lambdas[name] * excess).mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        if use_lagrangian:
            for name in constraint_names:
                mc = costs[name].mean().detach()
                lambdas[name] = torch.clamp(lambdas[name] + args.lr_lambda * (mc - thresholds[name]), min=0.0)
        ep_m.append(mlu.item())
    
    if epoch % 5 == 0 or epoch == 1:
        res = evaluate_dir_full(test_env, policy, cb)
        tag = ' *BEST*' if res['avg_mlu'] < best_mlu else ''
        if res['avg_mlu'] < best_mlu: best_mlu = res['avg_mlu']
        print(f'Epoch {epoch:3d} | train={np.mean(ep_m):.4f} test={res["avg_mlu"]:.4f} util={res["mean_util"]:.4f} overload={res["overload_ratio"]:.4f} p95={res["p95_util"]:.4f}{tag}')

res = evaluate_dir_full(test_env, policy, cb)
print(f'\nFinal train={np.mean(ep_m):.4f} test={res["avg_mlu"]:.4f} util={res["mean_util"]:.4f} overload={res["overload_ratio"]:.4f} p95={res["p95_util"]:.4f}')
