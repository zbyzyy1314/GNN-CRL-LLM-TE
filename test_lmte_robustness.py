"""Test trained LLM model with LMTE-style robustness tests."""
import sys, torch, numpy as np, time, json
sys.path.insert(0, '.')
from te_framework.topology import Topology
from te_framework.env import TEEnv
from te_framework.networks.gnn import TEGNNLLMPolicy, GNNValueNetwork
from te_framework.llm_encoder import LLMEncoder
from te_framework.agents.combined_cmdp import CombinedCMDPAgent
from te_framework.traffic import TrafficLoader
from train_cmdp import evaluate

WINDOW, LLM_BATCH = 12, 64; DEVICE = 'cuda'
CKPT = 'checkpoints/lmte_llm.pt'

topo = Topology('data/LMTE_correct_topo', max_paths_per_pair=8)
traffic = TrafficLoader('data/LMTE_correct_tm.txt', topo.num_nodes)
tms = traffic.get_real_traffic(normalize=False)
split = int(len(tms) * 0.8)

def make_env(tm_slice):
    raw = tm_slice / (100.0 * 8 / 300 / 1000)
    with open('/tmp/lmte_tmp.txt', 'w') as f:
        for tm in raw: f.write(' '.join(f'{v:.6f}' for v in tm.flat) + '\n')
    t = TrafficLoader('/tmp/lmte_tmp.txt', topo.num_nodes)
    e = TEEnv(topo, t, device=DEVICE); e.precompute_ecmp()
    return e

test_env = make_env(tms[split:])

# Load model
print(f'Loading checkpoint: {CKPT}')
ckpt = torch.load(CKPT, map_location=DEVICE)
llm = LLMEncoder(hidden_dim=128, llm_dim=1536, llm_batch_size=LLM_BATCH, use_4bit=False, device=DEVICE)
llm.load_model()
policy = TEGNNLLMPolicy(topo, llm, hidden_dim=128, max_k=topo.max_k, temporal=True, history_len=WINDOW).to(DEVICE)
value = GNNValueNetwork(topo, hidden_dim=128).to(DEVICE)
policy.load_state_dict(ckpt['policy'], strict=False)
value.load_state_dict(ckpt['value'], strict=False)
agent = CombinedCMDPAgent(policy, value, test_env.path_mask,
    ['mean_util','overload_ratio','p95_util'],
    {'mean_util':0.3,'overload_ratio':0.1,'p95_util':0.5}, device=DEVICE)

# ─── 1. Baseline ───
res = evaluate(agent, test_env, 64)
baseline = res['avg_mlu']
print(f'1. Baseline: MLU={baseline:.4f}')

# ─── 2. Link Failures ───
print('2. Link Failures:')
for nf in [1, 3, 5, 10]:
    e = make_env(tms[split:])
    rng = np.random.RandomState(nf * 100)
    faulty = rng.choice(e.num_links, nf, replace=False)
    e.link_caps[faulty] *= 0.1; e.precompute_ecmp()
    r = evaluate(agent, e, 64)
    print(f'  {nf} links: MLU={r["avg_mlu"]:.4f} ({(r["avg_mlu"]/baseline-1)*100:+.0f}%)')

# ─── 3. Traffic Bursts ───
print('3. Traffic Bursts:')
for scale in [2, 5, 10, 20, 30]:
    e = make_env(tms[split:])
    rng = np.random.RandomState(42)
    od_std = e.real_tm.cpu().numpy().std(axis=0)
    noise = torch.tensor(rng.randn(*e.real_tm.shape) * od_std * (scale/10.0), device=DEVICE)
    for t in range(len(noise)): noise[t].fill_diagonal_(0)
    e.real_tm = e.real_tm + noise
    r = evaluate(agent, e, 64)
    print(f'  scale={scale:2d}: MLU={r["avg_mlu"]:.4f} ({(r["avg_mlu"]/baseline-1)*100:+.0f}%)')

# ─── 4. Natural Drift ───
print('4. Natural Drift:')
for pct, label in [(0,'0-25%'),(25,'25-50%'),(50,'50-75%')]:
    s = int(len(tms[split:])*pct/100); end = int(len(tms[split:])*(pct+25)/100)
    e = make_env(tms[split:][s:end])
    r = evaluate(agent, e, 64)
    print(f'  {label}: MLU={r["avg_mlu"]:.4f} ({(r["avg_mlu"]/baseline-1)*100:+.0f}%)')

print('\nCompare: LMTE reports <5% burst degradation, <10% link failure degradation')
