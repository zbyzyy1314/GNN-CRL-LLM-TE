"""
Cross-topology transfer: train on GEANT, test on Abilene.
Loads only topology-independent MPBlock layers.
"""

import sys, torch, os
sys.path.insert(0, '.')
from te_framework.topology import Topology
from te_framework.networks.gnn import TEGNNPolicy, GNNValueNetwork
from te_framework.traffic import TrafficLoader
from te_framework.env import TEEnv
from te_framework.agents.combined_cmdp import CombinedCMDPAgent
from train_cmdp import evaluate

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/combined_best.pt'
if not os.path.exists(ckpt_path):
    print(f'Training GEANT first...')
    os.system('python train_cmdp.py --device cuda --network gnn --method combined --epochs 40')
    ckpt_path = 'checkpoints/combined_best.pt'

ckpt = torch.load(ckpt_path, map_location='cuda')
policy_state = ckpt['policy']

topo = Topology('data/Abilene', max_paths_per_pair=8)
policy = TEGNNPolicy(topo, hidden_dim=128).cuda()
value = GNNValueNetwork(topo, hidden_dim=128).cuda()

transferred = 0
total = 0
for name, param in policy.named_parameters():
    if name.startswith('blocks.'):
        total += 1
        if name in policy_state and policy_state[name].shape == param.shape:
            param.data.copy_(policy_state[name])
            transferred += 1

print(f'Transferred {transferred}/{total} MPBlock layers from GEANT')

traffic = TrafficLoader('data/AbileneTM2', topo.num_nodes)
env = TEEnv(topo, traffic, device='cuda')
env.precompute_ecmp()

agent = CombinedCMDPAgent(policy, value, env.path_mask,
    ['mean_util', 'overload_ratio', 'p95_util'],
    {'mean_util': 0.3, 'overload_ratio': 0.1, 'p95_util': 0.5}, device='cuda')
res = evaluate(agent, env, 512)

print(f'\nCross-topology (GEANT->Abilene):')
print(f'  MLU={res["avg_mlu"]:.4f} ({res["improvement"]:.1f}% vs ECMP)')
print(f'  p95={res["p95_util"]:.4f}  violation_rate={res["violation_rate"]:.1f}%')
print(f'\nCompare:')
print(f'  Abilene trained from scratch: MLU=0.087')
print(f'  Random policy:                MLU~0.50')
if res['avg_mlu'] < 0.20:
    print('Result: Transfer works! GNN message passing generalizes.')
else:
    print('Result: Transfer barely helps. Need LLM for cross-topology.')
