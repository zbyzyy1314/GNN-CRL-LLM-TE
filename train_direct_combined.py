import sys, torch, numpy as np, time, argparse
sys.path.insert(0, '.')
from te_framework.topology import Topology
from te_framework.env import TEEnv
from te_framework.networks.gnn import TEGNNPolicy
from te_framework.traffic import TrafficLoader

p = argparse.ArgumentParser()
# 通用
p.add_argument('--seed', type=int, default=42)
p.add_argument('--epochs', type=int, default=40)
# CVaR 相关
p.add_argument('--cvar-beta', type=float, default=2.0,
               help='Exponential weight factor (0=uniform, >0=weighted CVaR)')
p.add_argument('--cvar-k-frac', type=float, default=0.05,
               help='Fraction of links to take as the tail (top 5%% => 4 links)')
# Lagrangian 相关
p.add_argument('--lam', type=float, default=0.5, help='Initial Lagrange multiplier')
p.add_argument('--mean-util-threshold', type=float, default=0.25)
p.add_argument('--overload-threshold', type=float, default=0.1)
p.add_argument('--p95-util-threshold', type=float, default=1.33)
p.add_argument('--lr-lambda', type=float, default=0.02)
# 主 loss 混合系数: cvar_mix=1 -> 纯 CVaR, cvar_mix=0 -> 纯 mlu
p.add_argument('--cvar-mix', type=float, default=1.0,
               help='Mix between CVaR (1) and plain MLU (0) for the primary loss')
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

topo = Topology('data/LMTE_correct_topo', max_paths_per_pair=8)
train = TrafficLoader('data/LMTE_train.txt', topo.num_nodes)
test = TrafficLoader('data/LMTE_test.txt', topo.num_nodes)
env = TEEnv(topo, train, device=device)
test_env = TEEnv(topo, test, device=device)
env.precompute_ecmp(); test_env.precompute_ecmp()
print(f'N={topo.num_nodes} L={topo.num_links} P={topo.num_pairs}')

policy = TEGNNPolicy(topo, hidden_dim=128, max_k=topo.max_k).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
lm = env.link_mask; lc = env.link_caps
sd = torch.tensor(topo.pair_idx_to_sd, device=device)
cb = 128; best_mlu = float('inf')

# --- CVaR 设置 ---
K = max(1, int(args.cvar_k_frac * topo.num_links))  # 尾部链接数
beta = args.cvar_beta
cvar_mix = args.cvar_mix

# --- Lagrangian 设置 ---
CNS = ['mean_util', 'overload_ratio', 'p95_util']
lambdas = {k: torch.tensor(args.lam, device=device) for k in CNS}
thresholds = {
    'mean_util': args.mean_util_threshold,
    'overload_ratio': args.overload_threshold,
    'p95_util': args.p95_util_threshold,
}


def compute_ratios(logits, path_mask):
    B = logits.shape[0]
    num_pairs = path_mask.shape[0]
    ratios = torch.softmax(logits.view(B, num_pairs, -1), dim=-1)
    ratios = ratios * path_mask.unsqueeze(0)
    ratios = ratios / (ratios.sum(dim=-1, keepdim=True) + 1e-8)
    return ratios


def evaluate(test_env, policy):
    policy.eval()
    all_m, all_u, all_o, all_p = [], [], [], []
    with torch.no_grad():
        idx = torch.arange(test_env.num_tms, device=device)
        for s in range(0, test_env.num_tms, cb):
            e = min(s + cb, test_env.num_tms)
            idx_b = idx[s:e]
            states = test_env.get_states(idx_b)
            logits = policy(states, test_env.path_mask)
            ratios = compute_ratios(logits, test_env.path_mask)
            demands = test_env.real_tm[idx_b][:, sd[:, 0], sd[:, 1]]
            loads = torch.einsum('bp,bpk,pkl->bl', demands, ratios, lm)
            u = loads / lc.unsqueeze(0)
            all_m.append(u.max(dim=1).values.cpu())
            all_u.append(u.mean(dim=1).cpu())
            all_o.append((u > 0.8).float().mean(dim=1).cpu())
            all_p.append(torch.quantile(u, 0.95, dim=1).cpu())
    a = lambda x: torch.cat(x).mean().item()
    policy.train()
    return {'avg_mlu': a(all_m), 'mean_util': a(all_u),
            'overload_ratio': a(all_o), 'p95_util': a(all_p)}


for epoch in range(1, args.epochs + 1):
    ep_loss = []
    ep_mlu = []
    c_sum = {k: 0.0 for k in CNS}
    c_cnt = 0
    perm = torch.randperm(env.num_tms, device=device)

    for start in range(0, env.num_tms, cb):
        end = min(start + cb, env.num_tms)
        idx_b = perm[start:end]
        states = env.get_states(idx_b)
        B = states.shape[0]
        logits = policy(states, env.path_mask)
        ratios = compute_ratios(logits, env.path_mask)
        demands = env.real_tm[idx_b][:, sd[:, 0], sd[:, 1]]
        loads = torch.einsum('bp,bpk,pkl->bl', demands, ratios, lm)
        util = loads / lc.unsqueeze(0)

        # ---- CVaR 风格的主 loss: 关注最差 top-K 链接 ----
        su, _ = util.sort(dim=1, descending=True)
        topk = su[:, :K]  # top K worst links
        if beta > 0:
            w = torch.softmax(topk * beta, dim=1)  # exponential weights
            cvar_val = (topk * w).sum(dim=1).mean()
        else:
            cvar_val = topk.mean(dim=1).mean()  # uniform CVaR
        mlu = util.max(dim=1).values.mean()
        # CVaR 与 plain MLU 线性混合
        primary = cvar_mix * cvar_val + (1.0 - cvar_mix) * mlu

        # ---- Lagrangian 约束惩罚 ----
        loss = primary
        costs = env.compute_constraints(loads)
        for name in CNS:
            c_val = costs[name].mean()
            loss = loss + lambdas[name] * torch.relu(c_val - thresholds[name])
            c_sum[name] += c_val.item()
        c_cnt += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        ep_loss.append(loss.item())
        ep_mlu.append(mlu.item())

    # 每个 epoch 结束后,按平均违反量更新乘子
    for name in CNS:
        avg_violation = c_sum[name] / c_cnt
        lambdas[name] = torch.clamp(
            lambdas[name] + args.lr_lambda * (avg_violation - thresholds[name]),
            min=0.0,
        )

    if epoch % 5 == 0 or epoch == 1:
        res = evaluate(test_env, policy)
        tag = ' *BEST*' if res['avg_mlu'] < best_mlu else ''
        if res['avg_mlu'] < best_mlu:
            best_mlu = res['avg_mlu']
        print(f'Epoch {epoch:3d} | train={np.mean(ep_loss):.4f} '
              f'mlu={np.mean(ep_mlu):.4f} test={res["avg_mlu"]:.4f} '
              f'util={res["mean_util"]:.4f} overload={res["overload_ratio"]:.4f} '
              f'p95={res["p95_util"]:.4f}{tag}')
        ls = {k: f'{lambdas[k].item():.3f}' for k in CNS}
        print(f'         lambda: {ls["mean_util"]} {ls["overload_ratio"]} {ls["p95_util"]}')

res = evaluate(test_env, policy)
print(f'\nFinal train={np.mean(ep_loss):.4f} mlu={np.mean(ep_mlu):.4f} '
      f'test={res["avg_mlu"]:.4f} util={res["mean_util"]:.4f} '
      f'overload={res["overload_ratio"]:.4f} p95={res["p95_util"]:.4f}')
