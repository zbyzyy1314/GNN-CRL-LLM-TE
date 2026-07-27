"""8-path constrained LP vs free LP comparison on LMTE test TMs."""
import sys, numpy as np, time
sys.path.insert(0, '.')
from te_framework.topology import Topology
from te_framework.traffic import TrafficLoader
from pulp import (LpMinimize, LpProblem, LpStatus, lpSum, LpVariable,
                  value, GLPK, PULP_CBC_CMD)

topo = Topology('data/LMTE_correct_topo', max_paths_per_pair=8)
traffic = TrafficLoader('data/LMTE_correct_tm.txt', topo.num_nodes)
real = traffic.get_real_traffic(normalize=False)
split = int(len(real)*0.8)
test = real[split:]

P=topo.num_pairs; L=topo.num_links; K=topo.max_k
links = [(s,d) for (s,d) in topo.link_sd_to_idx]
# Pre-build: path p, path k → link mask
# link_mask_from_path[p,k] = [(l_src,l_dst), ...] (not a great format)
# Use existing topo.link_mask if available
path_link_mask = topo.link_mask  # (P, K, L)

print(f'Computing 8-path LP of 10 test TMs...')
t0=time.time()
free_lps = np.loadtxt('data/lmte_lp_results.txt')[:10]
eight_lps=[]

for i in range(10):
    tm=test[i]
    model=LpProblem('path8',LpMinimize)
    # Weight per (path_idx) — total P*K variables
    w_vars={}
    for pi in range(P):
        s,d=topo.pair_idx_to_sd[pi]
        dem=float(tm[s,d])
        if dem<=0: continue
        for ki in range(K):
            if not topo.path_mask[pi,ki]: continue  # invalid path
            w_vars[pi,ki]=LpVariable(f'w_{pi}_{ki}',0,1)
        # sum of weights = 1 for this OD pair
        model+=lpSum(w_vars.get((pi,ki),0) for ki in range(K))==1
    
    # Link load = sum over paths using this link of demand * weight
    r=LpVariable('r',0)
    for li,(ls,ld) in enumerate(links):
        load=lpSum(
            float(tm[topo.pair_idx_to_sd[pi][0], topo.pair_idx_to_sd[pi][1]])
            * w_vars.get((pi,ki),0)
            for pi in range(P)
            for ki in range(K)
            if (pi,ki) in w_vars and path_link_mask[pi,ki,li]>0
        )
        model+=load<=topo.link_capacities[li]*r
    
    EPS=1e-6
    # Also compute total load as tiebreaker
    total_l=lpSum(
        float(tm[topo.pair_idx_to_sd[pi][0], topo.pair_idx_to_sd[pi][1]])
        * w_vars.get((pi,ki),0)
        for pi in range(P) for ki in range(K) if (pi,ki) in w_vars
        for li in range(L) if path_link_mask[pi,ki,li]>0
    )
    model+=r+EPS*total_l
    
    try: model.solve(GLPK(msg=False,timeLimit=30))
    except: model.solve(PULP_CBC_CMD(msg=False,timeLimit=30))
    
    if LpStatus[model.status]=='Optimal':
        eight_lps.append(value(r))
    else:
        eight_lps.append(None)
    
    if (i+1)%5==0: print(f'  {i+1}/10')

eight_lps=np.array(eight_lps)
print(f'\nDone in {time.time()-t0:.0f}s')
print(f'Free LP (flow):    {free_lps.mean():.4f}')
print(f'8-path LP (path):  {eight_lps.mean():.4f}')
print(f'Ratio:             {eight_lps.mean()/free_lps.mean():.3f}x')
print(f'\nThe 8-path LP shows how much the KSP constraint costs vs free flow.')
print(f'Your model MLU (baseline=1.98) vs 8-path LP = answer is the model gap.')
