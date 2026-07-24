# TE-CRDP: Constrained Reinforcement Learning for Traffic Engineering

Constrained DRL framework for traffic engineering with **7 optimization methods × 2 network architectures × 2 datasets**.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# GEANT — best method
python train_cmdp.py --device cuda --network gnn --method combined --epochs 40 --ppo-epochs 4

# Abilene
python train_cmdp.py --device cuda --network gnn --method combined \
    --topo data/Abilene --traffic data/AbileneTM --test data/AbileneTM2 \
    --epochs 40 --ppo-epochs 4
```

## Methods

| Method | Flag | Description |
|---|---|---|
| Baseline PPO | `--method baseline` | Unconstrained PPO with MLU objective |
| Reward Shaping | `--method shaping` | Fixed penalty weights |
| Lagrangian PPO | `--method lagrangian` | Adaptive Lagrange multipliers |
| CVaR PPO | `--method cvar` | Conditional Value-at-Risk objective |
| Safety Layer | `--method safety` | Hard constraint projection |
| **Constrained PPO** | `--method combined` | **CVaR + Lagrangian + Safety (best)** |
| DQN | `--method dqn` | Deep Q-Network baseline |

## Networks

| Network | Params | Speed |
|---|---|---|
| `--network gnn` | 220K | Fast |
| `--network cnn` | 3-4.5M | Faster |

## Datasets

| Topology | Nodes | Links | ECMP MLU |
|---|---|---|---|
| GEANT | 23 | 74 | 0.099 |
| Abilene | 12 | 34 | 0.180 |

Placed in `data/`. Convert from raw with `convert_geant.py` / `convert_abilene.py`.

## Key Results

| Topology | ECMP | Combined GNN | Improvement |
|---|---|---|---|
| GEANT | 0.099 | **0.081** | +17.8% |
| Abilene | 0.180 | **0.087** | +51.9% |

**220K-param GNN achieves same or better MLU than 4.5M-param CNN.**

## Directory

```
te_framework/
├── agents/        # 7 RL methods
├── networks/      # CNN + GNN architectures
├── env.py         # Environment + constraints
├── topology.py    # Graph + shortest paths
├── traffic.py     # TM loader
├── routing.py     # ECMP distribution
├── optimal_mlu.py # LP optimal MLU solver
train_cmdp.py      # Main training entry
convert_geant.py   # GEANT data converter
convert_abilene.py # Abilene data converter
data/              # Topology + traffic matrices
```

## Citation

Based on CFR-RL (Zhang et al., IEEE JSAC 2020).
