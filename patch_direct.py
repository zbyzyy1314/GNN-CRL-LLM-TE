import pathlib
f = pathlib.Path('train_cmdp.py')
t = f.read_text('utf-8')

# Replace the direct method block: move BEFORE act_batch to avoid double backward
old = """            raw_actions, log_probs, values = agent.act_batch(states)
            if args.method == 'safety':
                safe_actions, corrections = agent.project_and_act(idx_b, raw_actions)
                actions = safe_actions
                ep_corrections.append(corrections.float().mean().item())
            elif args.method == 'combined':
                if agent.safety is not None:
                    safe_actions, corrections = agent.safety.project(idx_b, raw_actions)
                    actions = safe_actions
                    ep_corrections.append(corrections.float().mean().item())
                else:
                    actions = raw_actions
            if args.method == 'direct':
                # Direct optimization: ratios -> loads -> MLU -> backward
                B = states.shape[0]
                logits = policy(states, env.path_mask)  # (B, P*K)
                ratios = F.softmax(logits.view(B, topo.num_pairs, -1), dim=-1)  # (B, P, K)
                p_idx = torch.arange(topo.num_pairs, device=device)
                s, d = topo.pair_idx_to_sd[:, 0], topo.pair_idx_to_sd[:, 1]
                demands = env.real_tm[idx_b][:, s, d]  # (B, P)
                ratios = ratios * env.path_mask.unsqueeze(0)  # zero out invalid paths
                ratios = ratios / (ratios.sum(dim=-1, keepdim=True) + 1e-8)
                # Link loads via einsum
                link_loads = torch.einsum('bp,bpk,pkl->bl', demands, ratios, env.link_mask)
                mlus = (link_loads / env.link_caps.unsqueeze(0)).max(dim=1).values
                constraint_costs = env.compute_constraints(link_loads)
                # Loss = MLU + Lagrangian penalties
                loss = mlus.mean()
                for name in agent.constraint_names:
                    cost = constraint_costs[name]
                    excess = torch.relu(cost - agent.thresholds[name])
                    loss = loss + (agent.lambdas[name] * excess).mean()
                agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                agent.optimizer.step()
                # Update lambdas
                for name in agent.constraint_names:
                    mc = constraint_costs[name].mean()
                    agent.lambdas[name] = torch.clamp(
                        agent.lambdas[name] + agent.lr_lambda * (mc - agent.thresholds[name]), min=0.0)
                rewards = -mlus
                loads = link_loads"""

new = """            if args.method == 'direct':
                B = states.shape[0]
                logits = policy(states, env.path_mask)
                ratios = F.softmax(logits.view(B, topo.num_pairs, -1), dim=-1)
                sd = torch.tensor(topo.pair_idx_to_sd, device=device)
                demands = env.real_tm[idx_b][:, sd[:, 0], sd[:, 1]]
                ratios = ratios * env.path_mask.unsqueeze(0)
                ratios = ratios / (ratios.sum(dim=-1, keepdim=True) + 1e-8)
                link_loads = torch.einsum('bp,bpk,pkl->bl', demands, ratios, env.link_mask)
                mlus = (link_loads / env.link_caps.unsqueeze(0)).max(dim=1).values
                constraint_costs = env.compute_constraints(link_loads)
                loss = mlus.mean()
                for name in agent.constraint_names:
                    loss = loss + (agent.lambdas[name] * torch.relu(constraint_costs[name] - agent.thresholds[name])).mean()
                agent.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); agent.optimizer.step()
                for name in agent.constraint_names:
                    mc = constraint_costs[name].mean()
                    agent.lambdas[name] = torch.clamp(agent.lambdas[name] + agent.lr_lambda * (mc - agent.thresholds[name]), min=0.0)
                rewards = -mlus; loads = link_loads
            else:
                raw_actions, log_probs, values = agent.act_batch(states)
                if args.method == 'safety':
                    safe_actions, corrections = agent.project_and_act(idx_b, raw_actions)
                    actions = safe_actions; ep_corrections.append(corrections.float().mean().item())
                elif args.method == 'combined':
                    if agent.safety is not None:
                        safe_actions, corrections = agent.safety.project(idx_b, raw_actions)
                        actions = safe_actions; ep_corrections.append(corrections.float().mean().item())
                    else:
                        actions = raw_actions
                else:
                    actions = raw_actions
                rewards, mlus, loads = env.step_batch_idx(idx_b, actions)
                constraint_costs = env.compute_constraints(loads)"""

if old not in t:
    # Try with exact whitespace from the file
    print('Pattern not found, trying to locate exact content...')
    # Find the line with "raw_actions, log_probs, values = agent.act_batch(states)"
    idx = t.find("raw_actions, log_probs, values = agent.act_batch(states)")
    if idx >= 0:
        print(f'Found at position {idx}')
        # Find the end of the direct block
        end_idx = t.find("            rewards = -mlus\n                loads = link_loads", idx)
        if end_idx >= 0:
            print(f'End at position {end_idx + len(chr(10))}')
        else:
            end_idx = t.find("rewards = -mlus\n                loads = link_loads", idx)
            if end_idx >= 0:
                print(f'End at position {end_idx}')
    # Try less strict matching
    old2 = "raw_actions, log_probs, values = agent.act_batch(states)"
    new2 = "if args.method == 'direct':\n                B = states.shape[0]\n                logits = policy(states, env.path_mask)\n                ratios = F.softmax(logits.view(B, topo.num_pairs, -1), dim=-1)\n                sd128 = torch.tensor(topo.pair_idx_to_sd, device=device)\n                demands = env.real_tm[idx_b][:, sd128[:, 0], sd128[:, 1]]\n                ratios = ratios * env.path_mask.unsqueeze(0)\n                ratios = ratios / (ratios.sum(dim=-1, keepdim=True) + 1e-8)\n                link_loads = torch.einsum('bp,bpk,pkl->bl', demands, ratios, env.link_mask)\n                mlus = (link_loads / env.link_caps.unsqueeze(0)).max(dim=1).values\n                constraint_costs = env.compute_constraints(link_loads)\n                loss = mlus.mean()\n                for name in agent.constraint_names:\n                    loss = loss + (agent.lambdas[name] * torch.relu(constraint_costs[name] - agent.thresholds[name])).mean()\n                agent.optimizer.zero_grad(); loss.backward()\n                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); agent.optimizer.step()\n                for name in agent.constraint_names:\n                    mc = constraint_costs[name].mean()\n                    agent.lambdas[name] = torch.clamp(agent.lambdas[name] + agent.lr_lambda * (mc - agent.thresholds[name]), min=0.0)\n                rewards = -mlus; loads = link_loads\n            else:\n                " + old2
    t = t.replace(old2, new2)
    f.write_text(t, 'utf-8')
    print('Patched via alternate method')
else:
    t = t.replace(old, new)
    f.write_text(t, 'utf-8')
    print('OK')
