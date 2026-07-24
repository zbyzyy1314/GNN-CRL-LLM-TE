"""
Optimal MLU solver via Linear Programming (MILP relaxation).

Minimizes max link utilization by allowing arbitrary traffic splitting
across all possible links for each SD pair. Produces the theoretical
lower bound that no routing strategy can beat.

Uses PuLP + GLPK (or CBC as fallback).
"""

import numpy as np
from pulp import (LpMinimize, LpProblem, LpStatus, lpSum, LpVariable,
                  value, GLPK, PULP_CBC_CMD)


def solve_optimal_mlu(topo, traffic_matrix, verbose=False, time_limit=30):
    """
    Solve the optimal MLU for a given traffic matrix.

    Args:
        topo: Topology object
        traffic_matrix: (N, N) numpy array

    Returns:
        optimal_mlu: float — theoretical minimum MLU
    """
    N = topo.num_nodes
    L = topo.num_links

    # Build SD pair → link variable index
    pairs = list(range(topo.num_pairs))
    links = [(s, d) for (s, d) in topo.link_sd_to_idx]
    nodes = list(range(N))

    # Create variables
    model = LpProblem("TE_Optimal", LpMinimize)

    # Traffic split ratio per pair per link
    ratio = {}
    for p in pairs:
        s, d = topo.pair_idx_to_sd[p]
        demand = float(traffic_matrix[s, d])
        if demand <= 0:
            continue
        for (l_src, l_dst) in links:
            ratio[p, l_src, l_dst] = LpVariable(
                f"r_{p}_{l_src}_{l_dst}", lowBound=0, upBound=1)

    # Link load variables
    link_load = {l_idx: LpVariable(f"load_{l_idx}", lowBound=0)
                 for l_idx in range(L)}

    # Congestion ratio (MLU)
    r = LpVariable("MLU", lowBound=0)

    # Flow conservation constraints
    for p in pairs:
        s, d = topo.pair_idx_to_sd[p]
        demand = float(traffic_matrix[s, d])
        if demand <= 0:
            continue

        # Source node: outflow - inflow = 1
        model += (
            lpSum(ratio.get((p, l_src, l_dst), 0)
                  for (l_src, l_dst) in links if l_src == s) -
            lpSum(ratio.get((p, l_src, l_dst), 0)
                  for (l_src, l_dst) in links if l_dst == s) == 1,
            f"flow_src_{p}")

        # Destination node: outflow - inflow = -1
        model += (
            lpSum(ratio.get((p, l_src, l_dst), 0)
                  for (l_src, l_dst) in links if l_src == d) -
            lpSum(ratio.get((p, l_src, l_dst), 0)
                  for (l_src, l_dst) in links if l_dst == d) == -1,
            f"flow_dst_{p}")

        # Intermediate nodes: outflow = inflow
        for n in nodes:
            if n == s or n == d:
                continue
            model += (
                lpSum(ratio.get((p, l_src, l_dst), 0)
                      for (l_src, l_dst) in links if l_src == n) -
                lpSum(ratio.get((p, l_src, l_dst), 0)
                      for (l_src, l_dst) in links if l_dst == n) == 0,
                f"flow_mid_{p}_{n}")

    # Link load = sum of demands * ratios
    for l_idx, (l_src, l_dst) in enumerate(links):
        model += (
            link_load[l_idx] == lpSum(
                float(traffic_matrix[topo.pair_idx_to_sd[p][0],
                                     topo.pair_idx_to_sd[p][1]])
                * ratio.get((p, l_src, l_dst), 0)
                for p in pairs),
            f"link_load_{l_idx}")

        # Capacity constraint: load ≤ capacity * r
        model += (
            link_load[l_idx] <= topo.link_capacities[l_idx] * r,
            f"cap_{l_idx}")

    # Objective: minimize r + tiny penalty on total load (break ties)
    EPS = 1e-6
    model += r + EPS * lpSum(link_load[l_idx] for l_idx in range(L))

    # Solve
    try:
        model.solve(GLPK(msg=verbose, timeLimit=time_limit))
    except Exception:
        model.solve(PULP_CBC_CMD(msg=verbose, timeLimit=time_limit))

    if LpStatus[model.status] != 'Optimal':
        return None

    return value(r)


def compute_optimal_mlu_batch(topo, traffic_matrices, indices, verbose=False):
    """
    Compute optimal MLU for a batch of TMs (sequential LP solves).

    Args:
        topo: Topology
        traffic_matrices: (T, N, N) real traffic
        indices: list of indices to solve

    Returns:
        list of (idx, optimal_mlu) tuples
    """
    results = []
    total = len(indices)
    for i, idx in enumerate(indices):
        mlu = solve_optimal_mlu(topo, traffic_matrices[idx], verbose=verbose)
        if mlu is not None:
            results.append((idx, mlu))
        if (i + 1) % 10 == 0:
            print(f'  LP solved {i+1}/{total} (latest MLU={mlu:.4f})')
    return results
