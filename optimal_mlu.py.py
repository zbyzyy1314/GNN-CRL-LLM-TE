"""
optimal_mlu.py — 计算给定流量矩阵的理论最优 mlu(LP 下界)

用多商品流(Multi-Commodity Flow)线性规划:
    minimize  α
    subject to:
        sum_k x[p, k] == 1                 ∀ pair p
        x[p, k] >= 0, x[无效路径] == 0
        load[e] <= α * cap[e]              ∀ link e
        load[e] = sum_{p,k} demand[p] * x[p,k] * link_mask[p,k,e]

输出:每个流量矩阵的 optimal mlu,可作为 RL 训练的下界(baseline)

Usage:
    pip install cvxpy
    python optimal_mlu.py --topo data/large200_topo \\
        --traffic data/large200_train.txt --num 10

要求:te_framework 可正常 import
"""

import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, '.')
from te_framework.topology import Topology
from te_framework.traffic import TrafficLoader


def solve_optimal_mlu(lm, lc, path_mask, demands, solver='SCS', verbose=False):
    """单次 LP 求解。

    Args:
        lm: (P, K, L) link_mask
        lc: (L,) link capacities
        path_mask: (P, K) bool,有效路径
        demands: (P,) 每对 (s,d) 的需求
    Returns:
        optimal_mlu: float
        load: (L,) 最优负载分配
    """
    import cvxpy as cp
    P, K, L = lm.shape

    x = cp.Variable((P, K), nonneg=True)
    alpha = cp.Variable()

    # 1) 路径比例和为 1,无效路径强制为 0
    constraints = [cp.sum(x, axis=1) == 1]
    if not path_mask.all():
        constraints.append(cp.multiply(x, (~path_mask).astype(np.float32)) == 0)

    # 2) 链路负载
    y = cp.multiply(x, demands[:, None])
    y_flat = cp.reshape(y, (P * K,), order='C')
    lm_flat = lm.reshape(P * K, L)
    load = lm_flat.T @ y_flat  # (L,)

    # 3) mlu 约束
    constraints.append(load <= alpha * lc)
    constraints.append(alpha >= 0)

    prob = cp.Problem(cp.Minimize(alpha), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        return None, None
    return float(alpha.value), load.value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--topo', required=True)
    p.add_argument('--traffic', required=True)
    p.add_argument('--max-paths', type=int, default=8)
    p.add_argument('--num', type=int, default=10,
                   help='计算前 num 个流量矩阵(全量太慢)')
    p.add_argument('--solver', default='SCS',
                   choices=['SCS', 'ECOS', 'ECOS_BB', 'GUROBI', 'CLARABEL'])
    p.add_argument('--out', default='optimal_mlu.txt')
    args = p.parse_args()

    print(f'[*] 加载拓扑 {args.topo}...')
    topo = Topology(args.topo, max_paths_per_pair=args.max_paths)
    print(f'[*] 加载流量 {args.traffic}...')
    tl = TrafficLoader(args.traffic, topo.num_nodes)

    lm = topo.link_mask
    lc = topo.link_capacities
    path_mask = topo.path_mask
    sd = np.array(topo.pair_idx_to_sd)

    tms = tl.raw_matrices[:args.num]
    print(f'[*] 计算 {len(tms)} 个流量矩阵的最优 mlu (solver={args.solver})...')

    results = []
    for i, tm in enumerate(tms):
        demands = tm[sd[:, 0], sd[:, 1]].astype(np.float64)
        t0 = time.time()
        opt, load = solve_optimal_mlu(lm, lc, path_mask, demands, args.solver)
        dt = time.time() - t0
        if opt is None:
            print(f'  TM {i:3d}: solver failed ({dt:.1f}s)')
            continue
        max_load = load.max()
        avg_load = load.mean()
        print(f'  TM {i:3d}: optimal_mlu = {opt:.4f}  '
              f'(max_load={max_load:.2f}, avg_load={avg_load:.2f}, {dt:.1f}s)')
        results.append((i, opt, max_load, avg_load, dt))

    with open(args.out, 'w') as f:
        f.write('tm_idx\toptimal_mlu\tmax_load\tavg_load\tsolve_time_s\n')
        for r in results:
            f.write('\t'.join(f'{x:.6f}' if isinstance(x, float) else str(x) for x in r) + '\n')

    if results:
        mlu_arr = np.array([r[1] for r in results])
        print(f'\n[Summary]')
        print(f'  mean optimal_mlu = {mlu_arr.mean():.4f}')
        print(f'  min  optimal_mlu = {mlu_arr.min():.4f}')
        print(f'  max  optimal_mlu = {mlu_arr.max():.4f}')
        print(f'  saved to {args.out}')


if __name__ == '__main__':
    main()