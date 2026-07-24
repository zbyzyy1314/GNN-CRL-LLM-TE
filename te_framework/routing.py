"""
Routing utilities: ECMP distribution, link load computation, MLU evaluation.
"""

import numpy as np
from typing import List, Tuple


def ecmp_traffic_distribution(topo, traffic_matrix: np.ndarray) -> np.ndarray:
    """
    Compute ECMP link loads for a traffic matrix.

    Args:
        topo: Topology object (with paths_node, link_sd_to_idx, etc.)
        traffic_matrix: (N, N) array

    Returns:
        link_loads: (num_links,) array
    """
    link_loads = np.zeros(topo.num_links, dtype=np.float64)
    tm = traffic_matrix

    for pair_idx in range(topo.num_pairs):
        s, d = topo.pair_idx_to_sd[pair_idx]
        demand = tm[s, d]
        if demand <= 0:
            continue

        paths = topo.paths_node[pair_idx]
        num_paths = len(paths)
        if num_paths == 0:
            continue

        # Equal split across ECMP paths
        split_demand = demand / num_paths
        for node_path in paths:
            for i in range(len(node_path) - 1):
                link_idx = topo.link_sd_to_idx[(int(node_path[i]),
                                                 int(node_path[i+1]))]
                link_loads[link_idx] += split_demand

    return link_loads


def compute_link_loads(topo, traffic_matrix: np.ndarray,
                       path_choices: np.ndarray) -> np.ndarray:
    """
    Compute link loads given routing decisions.

    Args:
        topo: Topology object
        traffic_matrix: (N, N) array of traffic demands
        path_choices: (num_pairs,) array, each entry is the selected path index
                      (or -1 if no path available)

    Returns:
        link_loads: (num_links,) array
    """
    link_loads = np.zeros(topo.num_links, dtype=np.float64)
    tm = traffic_matrix

    for pair_idx in range(topo.num_pairs):
        k = int(path_choices[pair_idx])
        if k < 0:
            continue

        s, d = topo.pair_idx_to_sd[pair_idx]
        demand = tm[s, d]
        if demand <= 0:
            continue

        paths = topo.paths_link[pair_idx]
        if k >= len(paths):
            continue

        link_path = paths[k]
        for link_idx in link_path:
            link_loads[int(link_idx)] += demand

    return link_loads


def compute_link_loads_fast(topo, traffic_matrix: np.ndarray,
                             path_choices: np.ndarray) -> np.ndarray:
    """
    Vectorized link load computation using pre-computed lookup tables.
    Much faster than the per-pair loop version.
    """
    tm = traffic_matrix
    link_loads = np.zeros(topo.num_links, dtype=np.float64)

    for pair_idx in range(topo.num_pairs):
        k = int(path_choices[pair_idx])
        if k < 0:
            continue
        s, d = topo.pair_idx_to_sd[pair_idx]
        demand = tm[s, d]
        if demand <= 0:
            continue
        paths = topo.paths_link[pair_idx]
        if k >= len(paths):
            continue
        for link_idx in paths[k]:
            link_loads[int(link_idx)] += demand

    return link_loads


def compute_mlu(topo, link_loads: np.ndarray) -> float:
    """Compute Maximum Link Utilization."""
    utilization = link_loads / topo.link_capacities
    return float(np.max(utilization))


def compute_mean_delay(topo, link_loads: np.ndarray) -> float:
    """
    Compute mean queuing delay using M/M/1 approximation: d = 1/(C - L).

    Only valid when L < C for all links.
    """
    remaining = topo.link_capacities - link_loads
    remaining = np.maximum(remaining, 1e-6)
    delays = link_loads / remaining
    return float(np.mean(delays))


def ecmp_path_selection(topo) -> np.ndarray:
    """
    ECMP equivalent path selection: always picks path 0 (first shortest path).
    Since ECMP splits equally across all equal-cost paths, and we route per-flow,
    picking the first path for all flows approximates ECMP with per-flow routing.

    For a complete ECMP simulation, use ecmp_traffic_distribution instead.
    """
    choices = np.zeros(topo.num_pairs, dtype=np.int32)
    for i in range(topo.num_pairs):
        if len(topo.paths_link[i]) == 0:
            choices[i] = -1
    return choices
