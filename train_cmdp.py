"""
CMDP Training for Constrained RL Traffic Engineering.

Supports:
  - baseline:  standard PPO (no constraints)
  - lagrangian: Lagrangian PPO
  - shaping:   fixed-weight reward shaping (coming soon)

Usage:
    python train_cmdp.py --device cuda --method lagrangian --epochs 100
"""

import os, sys, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from te_framework.topology import Topology
from te_framework.traffic import TrafficLoader
from te_framework.env import TEEnv
from te_framework.networks import PathSelectionNetwork, ValueNetwork
from te_framework.networks.gnn import TEGNNPolicy, TEGNNLLMPolicy, GNNValueNetwork


# ─── Reward Shaping Agent (simple baseline) ───
class ShapingPPOAgent:
    """PPO with fixed-weight reward shaping: r = -MLU - Σ w_i * cost_i."""
    def __init__(self, policy_net, value_net, path_mask,
                 weights, lr=3e-4, clip_ratio=0.2, entropy_coef=0.01,
                 value_coef=0.5, max_grad_norm=0.5, ppo_epochs=8, device='cuda'):
        self.policy = policy_net
        self.value = value_net
        self.path_mask = path_mask
        self.device = device
        self.weights = weights
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(
            list(policy_net.parameters()) + list(value_net.parameters()), lr=lr)
        import torch.nn.functional as F
        self.F = F

    def act_batch(self, states, deterministic=False):
        with torch.no_grad():
            actions, log_probs, _ = self.policy.get_action(
                states, self.path_mask, deterministic=deterministic)
            values = self.value(states)
        return actions, log_probs, values

    def shape_reward(self, rewards, constraint_costs):
        shaped = rewards.clone()
        for name, w in self.weights.items():
            shaped = shaped - w * constraint_costs[name]
        return shaped

    def update(self, states, actions, old_log_probs, shaped_r, values):
        B = states.shape[0]
        bs = min(256, B)
        adv = shaped_r - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = adv + values
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(B, device=self.device)
            for start in range(0, B, bs):
                idx = perm[start:start + bs]
                s_b, a_b = states[idx], actions[idx]
                old_lp_b = old_log_probs[idx]
                adv_b, ret_b = adv[idx], ret[idx]
                logits = self.policy(s_b, self.path_mask)
                probs = self.F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(a_b)
                old_lp = old_lp_b
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv_b.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv_b.unsqueeze(-1)
                loss = (-torch.min(surr1, surr2).mean()
                        + self.value_coef * self.F.mse_loss(self.value(s_b), ret_b)
                        + self.entropy_coef * (-dist.entropy().mean()))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())+list(self.value.parameters()), 0.5)
                self.optimizer.step()
        return {}

    def save(self, p): torch.save({'policy': self.policy.state_dict(), 'value': self.value.state_dict()}, p)


# ─── Lagrangian PPO ───
from te_framework.agents.lagrangian_ppo import LagrangianPPOAgent

# ─── Safety Layer ───
from te_framework.agents.safety_layer import SafetyLayer, SafetyLayerPPOAgent

# ─── CVaR PPO ───
from te_framework.agents.cvar_ppo import CVaRPPOAgent
from te_framework.agents.dqn import DQNAgent

# ─── Combined (CVaR + Lagrangian + Safety) ───
from te_framework.agents.combined_cmdp import CombinedCMDPAgent


# ─── Baselines ───
from te_framework.agents.ppo import PPOBuffer  # not used but keep for reference


class BaselinePPOAgent:
    """Standard PPO wrapper matching CMDP API."""
    def __init__(self, policy_net, value_net, path_mask,
                 lr=3e-4, entropy_coef=0.01, ppo_epochs=8, device='cuda'):
        self.policy = policy_net
        self.value = value_net
        self.path_mask = path_mask
        self.device = device
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.optimizer = torch.optim.Adam(
            list(policy_net.parameters()) + list(value_net.parameters()), lr=lr)

    def act_batch(self, states, deterministic=False):
        with torch.no_grad():
            actions, log_probs, _ = self.policy.get_action(
                states, self.path_mask, deterministic=deterministic)
            values = self.value(states)
        return actions, log_probs, values

    def update(self, states, actions, old_log_probs, rewards, values):
        B = states.shape[0]; bs = min(256, B)
        adv = rewards - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = adv + values
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(B, device=self.device)
            for start in range(0, B, bs):
                idx = perm[start:start+bs]
                s_b, a_b = states[idx], actions[idx]
                old_lp_b = old_log_probs[idx]
                adv_b, ret_b = adv[idx], ret[idx]
                logits = self.policy(s_b, self.path_mask)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(a_b).sum(dim=-1)
                old_lp = old_lp_b.sum(dim=-1)
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_b
                loss = (-torch.min(surr1, surr2).mean()
                        + 0.5 * F.mse_loss(self.value(s_b), ret_b)
                        + self.entropy_coef * (-dist.entropy().mean()))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())+list(self.value.parameters()), 0.5)
                self.optimizer.step()
        return {}

    def get_lambda_state(self):
        return {}

    def save(self, p): torch.save({'policy': self.policy.state_dict(), 'value': self.value.state_dict()}, p)

import torch.nn.functional as F


# ─── Shared evaluation ───
def evaluate(agent, env, cb=512):
    agent.policy.eval(); agent.value.eval()
    try:
        all_m, all_e, all_costs = [], [], []
        indices = torch.arange(env.num_tms, device=env.device)
        for start in range(0, env.num_tms, cb):
            end = min(start+cb, env.num_tms)
            idx_b = indices[start:end]
            states = env.get_states(idx_b)
            actions, _, _ = agent.act_batch(states, deterministic=True)
            _, mlus, loads = env.step_batch_idx(idx_b, actions)
            all_m.append(mlus.cpu())
            all_e.append(env.get_ecmp_mlu_batch(idx_b).cpu())
            all_costs.append(env.compute_constraints(loads))
        all_m = torch.cat(all_m); all_e = torch.cat(all_e)
        agg_costs = {}
        for k in all_costs[0]:
            agg_costs[k] = torch.cat([c[k].cpu() for c in all_costs]).mean().item()
    finally:
        agent.policy.train(); agent.value.train()
    return {
        'avg_mlu': all_m.mean().item(), 'max_mlu': all_m.max().item(),
        'avg_ecmp': all_e.mean().item(),
        'improvement': (all_e.mean()-all_m.mean()).item()/all_e.mean().item()*100,
        'violation_rate': (all_m > 1.0).float().mean().item() * 100,
        **agg_costs
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', default='lagrangian',
                   choices=['baseline','lagrangian','shaping','safety','cvar','combined','dqn'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--collection-batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--entropy-coef', type=float, default=0.05)
    p.add_argument('--network', default='cnn', choices=['cnn','gnn','gnn_llm'],
                   help='Network architecture (gnn_llm = GNN + LLM enhanced)')
    p.add_argument('--llm-model', default='Qwen/Qwen2.5-1.5B-Instruct',
                   help='HuggingFace model name for LLM encoder')
    p.add_argument('--llm-batch-size', type=int, default=8,
                   help='Batch size per LLM forward pass (reduce if OOM)')
    p.add_argument('--llm-fp16', action='store_true',
                   help='Load LLM in FP16 instead of 4-bit (faster, needs 24GB)')
    p.add_argument('--llm-dim', type=int, default=1536,
                   help='LLM hidden dimension')
    p.add_argument('--ppo-epochs', type=int, default=8,
                   help='PPO update epochs per batch')
    p.add_argument('--ckpt-dir', default='checkpoints')
    p.add_argument('--max-paths', type=int, default=8, help='Max candidate paths per SD pair')
    p.add_argument('--eval-interval', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mean-util-threshold', type=float, default=0.3,
                   help='Mean utilization constraint threshold')
    p.add_argument('--overload-threshold', type=float, default=0.1,
                   help='Max fraction of links with util > 0.8')
    p.add_argument('--p95-util-threshold', type=float, default=0.5,
                   help='P95 utilization constraint threshold')
    p.add_argument('--lr-lambda', type=float, default=0.01,
                   help='Lagrange multiplier learning rate')
    p.add_argument('--topo', default='data/GEANT')
    p.add_argument('--traffic', default='data/GEANTTM')
    p.add_argument('--test', default='data/GEANTTM2')
    p.add_argument('--capacity-scale', type=float, default=1.0,
                   help='Scale link capacities (lower → trigger safety layer)')
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[*] Device: {device}  Method: {args.method}')
    if device.type == 'cuda':
        print(f'[*] GPU: {torch.cuda.get_device_name(0)}')

    topo = Topology(args.topo, max_paths_per_pair=args.max_paths)
    train_traf = TrafficLoader(args.traffic, topo.num_nodes)
    test_traf = TrafficLoader(args.test, topo.num_nodes)
    env = TEEnv(topo, train_traf, device=device)
    test_env = TEEnv(topo, test_traf, device=device)

    # Scale capacities (for safety layer testing)
    if args.capacity_scale != 1.0:
        print(f'[*] Scaling link capacities by {args.capacity_scale}')
        env.link_caps = env.link_caps * args.capacity_scale
        test_env.link_caps = test_env.link_caps * args.capacity_scale

    # ECMP reference: pre-compute once
    print('[*] Precomputing ECMP loads (one-time, ~5s)...')
    ecmp_costs = test_env.precompute_ecmp()
    env.precompute_ecmp()  # for training-side ECMP queries
    print(f'  ECMP: util={ecmp_costs["mean_util"].mean():.4f}  '
          f'overload={ecmp_costs["overload_ratio"].mean():.4f}  '
          f'p95={ecmp_costs["p95_util"].mean():.4f}')

    print(f'[*] N={topo.num_nodes} L={topo.num_links} P={topo.num_pairs} K={topo.max_k}')

    if args.network == "gnn_llm":
        from te_framework.llm_encoder import LLMEncoder
        llm_enc = LLMEncoder(hidden_dim=128, llm_dim=args.llm_dim, llm_batch_size=args.llm_batch_size, use_4bit=not args.llm_fp16, model_name=args.llm_model, device=device)
        llm_enc.load_model()
        policy = TEGNNLLMPolicy(topo, llm_enc, hidden_dim=128, max_k=topo.max_k).to(device)
        value = GNNValueNetwork(topo, hidden_dim=128).to(device)
    elif args.network == "gnn":
        policy = TEGNNPolicy(topo, hidden_dim=128, max_k=topo.max_k).to(device)
        value = GNNValueNetwork(topo, hidden_dim=128).to(device)
    else:
        policy = PathSelectionNetwork(topo.num_nodes, topo.num_pairs, topo.max_k,
                                       fc_dims=[512, 512]).to(device)
        value = ValueNetwork(topo.num_nodes, fc_dims=[512, 256]).to(device)
    print(f'[*] Network: {args.network.upper()} | Params: {sum(p.numel() for p in policy.parameters()):,}')

    # Constraint thresholds
    thresholds = {
        'mean_util': args.mean_util_threshold,
        'overload_ratio': args.overload_threshold,
        'p95_util': args.p95_util_threshold,
    }

    if args.method == 'lagrangian':
        agent = LagrangianPPOAgent(
            policy, value, env.path_mask,
            constraint_names=['mean_util', 'overload_ratio', 'p95_util'],
            constraint_thresholds=thresholds,
            lr=args.lr, lr_lambda=args.lr_lambda,
            entropy_coef=args.entropy_coef, ppo_epochs=args.ppo_epochs, device=device)
    elif args.method == 'shaping':
        agent = ShapingPPOAgent(
            policy, value, env.path_mask,
            weights={'mean_util': 0.1, 'overload_ratio': 0.5, 'p95_util': 0.05},
            lr=args.lr, entropy_coef=args.entropy_coef, ppo_epochs=args.ppo_epochs, device=device)
    elif args.method == 'safety':
        safety_layer = SafetyLayer(env, max_iter=3, max_check=30)
        agent = SafetyLayerPPOAgent(
            policy, value, env.path_mask, safety_layer,
            lr=args.lr, entropy_coef=args.entropy_coef, ppo_epochs=args.ppo_epochs, device=device)
    elif args.method == 'cvar':
        agent = CVaRPPOAgent(
            policy, value, env.path_mask,
            lr=args.lr, entropy_coef=args.entropy_coef, ppo_epochs=args.ppo_epochs, device=device)
    elif args.method == 'combined':
        safety_layer = SafetyLayer(env, max_iter=3, max_check=20)
        agent = CombinedCMDPAgent(
            policy, value, env.path_mask,
            constraint_names=['mean_util', 'overload_ratio', 'p95_util'],
            constraint_thresholds=thresholds,
            safety_layer=safety_layer,
            lr=args.lr, lr_lambda=args.lr_lambda,
            entropy_coef=args.entropy_coef, ppo_epochs=args.ppo_epochs, device=device)
    elif args.method == 'dqn':
        # Create a second network as target
        if args.network == 'gnn':
            target_net = TEGNNPolicy(topo, hidden_dim=128, max_k=topo.max_k).to(device)
        else:
            target_net = PathSelectionNetwork(topo.num_nodes, topo.num_pairs, topo.max_k,
                                               fc_dims=[512,512]).to(device)
        target_net.load_state_dict(policy.state_dict())  # init target = main
        agent = DQNAgent(policy, target_net, env.path_mask,
                         lr=args.lr, batch_size=128, device=device)
    else:
        agent = BaselinePPOAgent(
            policy, value, env.path_mask,
            lr=args.lr, entropy_coef=args.entropy_coef, ppo_epochs=args.ppo_epochs, device=device)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    cb = args.collection_batch

    print(f'\n{"="*60}')
    print(f'Training {args.method} ({args.epochs} epochs, batch={cb})')
    print(f'Thresholds: {thresholds}')
    print(f'{"="*60}')

    best_mlu = float('inf')

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(env.num_tms, device=device)
        ep_r, ep_m = [], []
        ep_violations = {k: [] for k in thresholds}
        ep_corrections = []
        t0 = time.time()

        for start in range(0, env.num_tms, cb):
            end = min(start+cb, env.num_tms)
            idx_b = perm[start:end]

            states = env.get_states(idx_b)
            raw_actions, log_probs, values = agent.act_batch(states)

            if args.method == 'safety':
                safe_actions, corrections = agent.project_and_act(idx_b, raw_actions)
                actions = safe_actions
                ep_corrections.append(corrections.float().mean().item())
            elif args.method == 'combined':
                safe_actions, corrections = agent.safety.project(idx_b, raw_actions)
                actions = safe_actions
                ep_corrections.append(corrections.float().mean().item())
            else:
                actions = raw_actions

            rewards, mlus, loads = env.step_batch_idx(idx_b, actions)
            constraint_costs = env.compute_constraints(loads)

            if args.method == 'lagrangian':
                train_r, violations = agent.compute_lagrangian_reward(
                    rewards, constraint_costs)
                agent.update(states, raw_actions, log_probs, train_r, values)
                agent.update_multipliers(constraint_costs)
            elif args.method == 'shaping':
                train_r = agent.shape_reward(rewards, constraint_costs)
                agent.update(states, raw_actions, log_probs, train_r, values)
            elif args.method == 'safety':
                # Train on RAW actions, but reward from CORRECTED actions
                agent.update(states, raw_actions, log_probs, rewards, values)
            elif args.method == 'cvar':
                # Minimize CVaR tail risk
                cvar_r = -constraint_costs['cvar_util']
                agent.update(states, raw_actions, log_probs, cvar_r, values)
            elif args.method == 'combined':
                train_r, _ = agent.compute_combined_reward(constraint_costs)
                agent.update(states, raw_actions, log_probs, train_r, values)
                agent.update_multipliers(constraint_costs)
            elif args.method == 'dqn':
                [agent.replay.push(states[i], raw_actions[i], rewards[i]) for i in range(states.shape[0])]
                agent.update()
            else:
                agent.update(states, raw_actions, log_probs, rewards, values)

            ep_r.append(rewards.mean().item())
            ep_m.append(mlus.mean().item())
            for k in thresholds:
                ep_violations[k].append(constraint_costs[k].mean().item())

        dt = time.time() - t0
        if epoch % 5 == 0 or epoch == 1:
            vstr = ' '.join(f'{k}={np.mean(v):.3f}' for k, v in ep_violations.items())
            print(f'Epoch {epoch:4d} | MLU={np.mean(ep_m):.4f} '
                  f'R={np.mean(ep_r):.4f} | {dt:.1f}s | {vstr}')
            if args.method == 'lagrangian':
                ls = agent.get_lambda_state()
                print(f'  λ: {" ".join(f"{k}={v:.3f}" for k,v in ls.items())}')
            if args.method in ('safety', 'combined') and ep_corrections:
                print(f'  Safety corrections/step: {np.mean(ep_corrections):.1f}')

        if epoch % args.eval_interval == 0:
            res = evaluate(agent, test_env, cb)
            imp = '+' if res['improvement'] > 0 else ''
            tag = ''
            if res['avg_mlu'] < best_mlu:
                best_mlu = res['avg_mlu']; tag = ' *BEST*'
                agent.save(os.path.join(args.ckpt_dir, f'{args.method}_best.pt'))
            print(f'  EVAL | MLU={res["avg_mlu"]:.4f} ({imp}{res["improvement"]:.1f}% vs ECMP)'
                  f' | util={res["mean_util"]:.4f} overload={res["overload_ratio"]:.4f}'
                  f' p95={res["p95_util"]:.4f}{tag}')

    print(f'\n{"="*60}')
    res = evaluate(agent, test_env, cb)
    print(f'Final ({args.method}): MLU={res["avg_mlu"]:.4f} '
          f'util={res["mean_util"]:.4f} overload={res["overload_ratio"]:.4f} '
          f'p95={res["p95_util"]:.4f}')
    agent.save(os.path.join(args.ckpt_dir, f'{args.method}_final.pt'))


if __name__ == '__main__':
    main()
