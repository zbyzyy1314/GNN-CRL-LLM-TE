"""
8-path constrained LP solver.
Usage:
  python compute_lp_8path.py --topo data/GEANT --tm data/GEANTTM --samples 5
  python compute_lp_8path.py --topo data/LMTE_correct_topo --tm data/LMTE_correct_tm.txt

Output: data/lp_8path_results.txt (mlu, mean_util, overload_ratio, p95_util)
For 200 nodes: P*K ≈ 200*199*8 ≈ 318k variables, ~2-5 min/TM with CBC.
"""
import sys, numpy as np, time, argparse
sys.path.insert(0, '.')
from te_framework.topology import Topology
from te_framework.traffic import TrafficLoader
from pulp import (LpMinimize, LpProblem, LpStatus, lpSum, LpVariable,
                  value, GLPK, PULP_CBC_CMD)

p = argparse.ArgumentParser()
p.add_argument('--topo', default='data/LMTE_correct_topo')
p.add_argument('--tm', default='data/LMTE_correct_tm.txt')
p.add_argument('--samples', type=int, default=0, help='0 = all TMs')
p.add_argument('--test-ratio', type=float, default=0.2, help='Use last test_ratio of TMs')
p.add_argument('--max-paths', type=int, default=8)
p.add_argument('--time-limit', type=int, default=30)
args = p.parse_args()

topo = Topology(args.topo, max_paths_per_pair=args.max_paths)
traffic = TrafficLoader(args.tm, topo.num_nodes)
real = traffic.get_real_traffic(normalize=False)

if args.test_ratio > 0:
    split = int(len(real) * (1 - args.test_ratio))
    real = real[split:]
    print(f'Using last {args.test_ratio*100:.0f}% TMs ({len(real)} samples)')
if args.samples > 0:
    real = real[:min(args.samples, len(real))]

P = topo.num_pairs; L = topo.num_links; K = topo.max_k
mask = topo.link_mask
print(f'N={topo.num_nodes} L={L} P={P} K={K}')
if P*K > 500000:
    print(f'WARNING: {P*K} variables - may be very slow!')
print(f'Solving {len(real)} TMs...')
t0 = time.time()
results = []

for i, tm in enumerate(real):
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
    
    model += r
    
    try: model.solve(GLPK(msg=False, timeLimit=args.time_limit))
    except: model.solve(PULP_CBC_CMD(msg=False, timeLimit=args.time_limit))
    
    if LpStatus[model.status] == 'Optimal':
        mlu = value(r)
        loads = np.array([value(link_vars[li]) for li in range(L)])
        utils = loads / topo.link_capacities
        results.append((mlu, utils.mean(), (utils>0.8).mean(), np.percentile(utils,95)))
    else:
        results.append((np.nan,)*4)
    
    if (i+1) % 10 == 0:
        print(f'  {i+1}/{len(real)}, {time.time()-t0:.0f}s (avg {(time.time()-t0)/(i+1):.1f}s/TM)')

results = np.array(results)
np.savetxt('data/lp_8path_results.txt', results, fmt='%.6f',
           header='mlu mean_util overload_ratio p95_util', comments='')
valid = ~np.isnan(results[:,0])
print(f'\nDone in {time.time()-t0:.0f}s, solved {int(valid.sum())}/{len(results)} TMs')
if valid.any():
    print(f'  MLU:       {np.nanmean(results[:,0]):.4f}')
    print(f'  mean_util: {np.nanmean(results[:,1]):.4f}')
    print(f'  overload:  {np.nanmean(results[:,2]):.4f}')
    print(f'  p95_util:  {np.nanmean(results[:,3]):.4f}')
