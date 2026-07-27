"""Compute LP optimal MLU on all LMTE test TMs. Runs ~30-60 min."""
import sys, numpy as np, time
sys.path.insert(0, '.')
from te_framework.optimal_mlu import solve_optimal_mlu
from te_framework.topology import Topology
from te_framework.traffic import TrafficLoader

topo = Topology('data/LMTE_correct_topo', max_paths_per_pair=8)
traffic = TrafficLoader('data/LMTE_correct_tm.txt', topo.num_nodes)
real = traffic.get_real_traffic(normalize=False)
split = int(len(real) * 0.8)
test = real[split:]

print(f'Computing LP on {len(test)} test TMs...')
t0 = time.time()
lps = []
for i in range(len(test)):
    opt = solve_optimal_mlu(topo, test[i], verbose=False, time_limit=10)
    lps.append(opt)
    if i % 200 == 0:
        print(f'{i}/{len(test)}, elapsed {time.time()-t0:.0f}s')

lps = np.array(lps)
np.savetxt('data/lmte_lp_results.txt', lps, fmt='%.4f')
print(f'\nDone in {time.time()-t0:.0f}s')
print(f'Mean: {lps.mean():.4f}')
print(f'Min:  {lps.min():.4f}')
print(f'Max:  {lps.max():.4f}')
print(f'Std:  {lps.std():.4f}')
print('Results saved to data/lmte_lp_results.txt')
