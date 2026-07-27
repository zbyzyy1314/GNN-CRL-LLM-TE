"""Compute LP optimal constraint metrics (mean_util, overload_ratio, p95_util)
for all LMTE test TMs. Saves to data/lmte_lp_constraints.txt.
Runs ~30-60 min for 2152 TMs."""
import sys, numpy as np, time
sys.path.insert(0, '.')
from te_framework.topology import Topology
from te_framework.traffic import TrafficLoader
from pulp import (LpMinimize, LpProblem, LpStatus, lpSum, LpVariable,
                  value, GLPK, PULP_CBC_CMD)

topo = Topology('data/LMTE_correct_topo', max_paths_per_pair=8)
traffic = TrafficLoader('data/LMTE_correct_tm.txt', topo.num_nodes)
real = traffic.get_real_traffic(normalize=False)
split = int(len(real) * 0.8)
test = real[split:]
N = topo.num_nodes; L = topo.num_links
pairs = list(range(topo.num_pairs))
links = [(s, d) for (s, d) in topo.link_sd_to_idx]
nodes = list(range(N))

print(f'Computing LP constraints for {len(test)} TMs...')
t0 = time.time()
results = []

for i, tm in enumerate(test):
    model = LpProblem("TE", LpMinimize)
    ratio = {}
    for p in pairs:
        s, d = topo.pair_idx_to_sd[p]
        demand = float(tm[s, d])
        if demand <= 0:
            continue
        for (l_src, l_dst) in links:
            ratio[p, l_src, l_dst] = LpVariable(
                f"r_{p}_{l_src}_{l_dst}", lowBound=0, upBound=1)

    link_load = {l_idx: LpVariable(f"load_{l_idx}", lowBound=0)
                 for l_idx in range(L)}
    r = LpVariable("MLU", lowBound=0)

    for p in pairs:
        s, d = topo.pair_idx_to_sd[p]
        demand = float(tm[s, d])
        if demand <= 0:
            continue
        model += (lpSum(ratio.get((p, l_src, l_dst), 0)
                        for (l_src, l_dst) in links if l_src == s) -
                  lpSum(ratio.get((p, l_src, l_dst), 0)
                        for (l_src, l_dst) in links if l_dst == s) == 1,
                  f"src_{p}")
        model += (lpSum(ratio.get((p, l_src, l_dst), 0)
                        for (l_src, l_dst) in links if l_src == d) -
                  lpSum(ratio.get((p, l_src, l_dst), 0)
                        for (l_src, l_dst) in links if l_dst == d) == -1,
                  f"dst_{p}")
        for n in nodes:
            if n == s or n == d:
                continue
            model += (lpSum(ratio.get((p, l_src, l_dst), 0)
                            for (l_src, l_dst) in links if l_src == n) -
                      lpSum(ratio.get((p, l_src, l_dst), 0)
                            for (l_src, l_dst) in links if l_dst == n) == 0,
                      f"mid_{p}_{n}")

    for l_idx, (l_src, l_dst) in enumerate(links):
        model += (link_load[l_idx] == lpSum(
            float(tm[topo.pair_idx_to_sd[p][0], topo.pair_idx_to_sd[p][1]])
            * ratio.get((p, l_src, l_dst), 0) for p in pairs),
            f"load_{l_idx}")
        model += (link_load[l_idx] <= topo.link_capacities[l_idx] * r,
                  f"cap_{l_idx}")

    EPS = 1e-6
    model += r + EPS * lpSum(link_load[l_idx] for l_idx in range(L))

    try:
        model.solve(GLPK(msg=False, timeLimit=10))
    except Exception:
        model.solve(PULP_CBC_CMD(msg=False, timeLimit=10))

    if LpStatus[model.status] != 'Optimal':
        results.append((r'NA', 0, 0, 0, 0))
        continue

    mlu = value(r)
    loads = np.array([value(link_load[l_idx]) for l_idx in range(L)])
    utils = loads / topo.link_capacities
    mean_util = utils.mean()
    overload_ratio = (utils > 0.8).mean()
    p95_util = np.percentile(utils, 95)
    results.append((mlu, mean_util, overload_ratio, p95_util))

    if i % 200 == 0:
        print(f'  {i}/{len(test)}, elapsed {time.time()-t0:.0f}s')

results = np.array(results)
header = 'mlu mean_util overload_ratio p95_util'
np.savetxt('data/lmte_lp_constraints.txt', results,
           header=header, fmt='%.6f', comments='')
print(f'\nDone in {time.time()-t0:.0f}s')
print(f'LP constraint metrics (mean of {len(results)} TMs):')
print(f'  MLU:            {results[:,0].mean():.4f}')
print(f'  mean_util:       {results[:,1].mean():.4f}')
print(f'  overload_ratio:  {results[:,2].mean():.4f}')
print(f'  p95_util:        {results[:,3].mean():.4f}')
print(f'Saved to data/lmte_lp_constraints.txt')
