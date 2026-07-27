"""8-path constrained LP: compute MLU + constraint metrics (mean_util, overload, p95).
Same path constraint as our RL models (K=8 KSP), for fair comparison.
Saves to data/lmte_lp_8path.txt."""
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

P = topo.num_pairs; L = topo.num_links; K = topo.max_k
links = [(s,d) for (s,d) in topo.link_sd_to_idx]
mask = topo.link_mask  # (P, K, L) binary: does path k for pair p use link l?

print(f'8-path LP on {len(test)} test TMs (~10 s/TM)...')
t0 = time.time()
results = []

for i, tm in enumerate(test):
    model = LpProblem('lp8', LpMinimize)
    w = {}
    for pi in range(P):
        s,d = topo.pair_idx_to_sd[pi]; dem = float(tm[s,d])
        if dem <= 0: continue
        for ki in range(K):
            if not topo.path_mask[pi,ki]: continue
            w[pi,ki] = LpVariable(f'w_{pi}_{ki}', 0, 1)
        model += lpSum(w.get((pi,ki),0) for ki in range(K)) == 1
    
    r = LpVariable('r', 0)
    link_vars = {li: LpVariable(f'l_{li}', 0) for li in range(L)}
    
    for li in range(L):
        load = lpSum(
            float(tm[topo.pair_idx_to_sd[pi][0], topo.pair_idx_to_sd[pi][1]])
            * w.get((pi,ki), 0)
            for pi in range(P) for ki in range(K)
            if (pi,ki) in w and mask[pi,ki,li] > 0
        )
        model += link_vars[li] == load
        model += load <= topo.link_capacities[li] * r
    
    model += r  # minimize MLU
    
    try: model.solve(GLPK(msg=False, timeLimit=15))
    except: model.solve(PULP_CBC_CMD(msg=False, timeLimit=15))
    
    if LpStatus[model.status] == 'Optimal':
        mlu = value(r)
        loads = np.array([value(link_vars[li]) for li in range(L)])
        utils = loads / topo.link_capacities
        mu = utils.mean()
        ov = (utils > 0.8).mean()
        p95 = np.percentile(utils, 95)
        results.append((mlu, mu, ov, p95))
    else:
        results.append((np.nan, np.nan, np.nan, np.nan))
    
    if (i+1) % 200 == 0:
        print(f'  {i+1}/{len(test)}, {time.time()-t0:.0f}s')

results = np.array(results)
header = 'mlu mean_util overload_ratio p95_util'
np.savetxt('data/lmte_lp_8path.txt', results, header=header, fmt='%.6f', comments='')

print(f'\nDone in {time.time()-t0:.0f}s')
print(f'8-path constrained LP ({int(np.sum(~np.isnan(results[:,0])))} TMs solved):')
print(f'  MLU:            {np.nanmean(results[:,0]):.4f}')
print(f'  mean_util:      {np.nanmean(results[:,1]):.4f}')
print(f'  overload_ratio: {np.nanmean(results[:,2]):.4f}')
print(f'  p95_util:       {np.nanmean(results[:,3]):.4f}')

# Compare with free LP
free = np.loadtxt('data/lmte_lp_constraints.txt')
print(f'\nFree flow LP:')
print(f'  MLU:            {free[:,0].mean():.4f}')
print(f'  mean_util:      {free[:,1].mean():.4f}')
print(f'  overload_ratio: {free[:,2].mean():.4f}')
print(f'  p95_util:       {free[:,3].mean():.4f}')
print(f'\n8-path vs free: MLU gap = {np.nanmean(results[:,0])/free[:,0].mean():.3f}x')
