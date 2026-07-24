"""
Topology: loads network topology and pre-computes shortest paths.

Supports the existing GEANT topology format:
  Line 1: "Node: <N>\tLink: <L>"
  Line 2: header
  Lines 3+: "link_idx\tsrc\tdst\tweight\tcapacity"
"""

import os
import pickle
import numpy as np
import networkx as nx


class Topology:
    def __init__(self, topo_file: str, cache_dir: str = None,
                 max_paths_per_pair: int = 8):
        self.topo_file = topo_file
        self.max_paths = max_paths_per_pair

        if cache_dir is None:
            cache_dir = os.path.dirname(topo_file)
        cache_name = os.path.basename(topo_file) + f'_k{max_paths_per_pair}_v2.pkl'
        self.cache_file = os.path.join(cache_dir, cache_name)

        self._load_topology()
        self._compute_paths()

    def _load_topology(self):
        with open(self.topo_file, 'r') as f:
            header = f.readline()
            self.num_nodes = int(header.split(':')[1].split('\t')[0].strip())
            self.num_links = int(header.split(':')[2].strip())
            f.readline()

            self.link_idx_to_sd = {}
            self.link_sd_to_idx = {}
            self.link_capacities = np.empty(self.num_links)
            self.link_weights = np.empty(self.num_links)
            self.graph = nx.DiGraph()

            for line in f:
                parts = line.strip().split('\t')
                idx, src, dst, w, cap = parts
                idx = int(idx); src = int(src); dst = int(dst)
                w = int(w); cap = float(cap)
                self.link_idx_to_sd[idx] = (src, dst)
                self.link_sd_to_idx[(src, dst)] = idx
                self.link_capacities[idx] = cap
                self.link_weights[idx] = w
                self.graph.add_edge(src, dst, weight=w)

        assert len(self.graph.nodes) == self.num_nodes
        assert len(self.graph.edges) == self.num_links

    def _compute_paths(self):
        if os.path.exists(self.cache_file):
            print(f'[*] Loading cached paths from {self.cache_file}')
            with open(self.cache_file, 'rb') as f:
                cached = pickle.load(f)
            self.__dict__.update(cached)
            return

        print(f'[*] Computing up to {self.max_paths} shortest paths per pair...')
        self.num_pairs = 0
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        self.paths_node = []
        self.paths_link = []

        for s in range(self.num_nodes):
            for d in range(self.num_nodes):
                if s == d:
                    continue
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.pair_idx_to_sd.append((s, d))
                self.num_pairs += 1

                try:
                    node_paths = list(nx.shortest_simple_paths(
                        self.graph, s, d, weight='weight'))
                    node_paths = node_paths[:self.max_paths]
                except nx.NetworkXNoPath:
                    node_paths = []

                node_list, link_list = [], []
                for np_nodes in node_paths:
                    links = [self.link_sd_to_idx[(np_nodes[i], np_nodes[i+1])]
                             for i in range(len(np_nodes) - 1)]
                    node_list.append(np.array(np_nodes, dtype=np.int32))
                    link_list.append(np.array(links, dtype=np.int32))
                self.paths_node.append(node_list)
                self.paths_link.append(link_list)

        self.max_k = min(max(len(pl) for pl in self.paths_link), self.max_paths)
        self.path_mask = np.zeros((self.num_pairs, self.max_k), dtype=bool)
        for i, pl in enumerate(self.paths_link):
            n = min(len(pl), self.max_k)
            self.path_mask[i, :n] = True

        # --- GPU-optimized: link_mask[pair, k, link] = 1 if used ---
        self.link_mask = np.zeros((self.num_pairs, self.max_k, self.num_links),
                                  dtype=np.float32)
        for pi in range(self.num_pairs):
            for k, lp in enumerate(self.paths_link[pi]):
                if k >= self.max_k:
                    break
                for lid in lp:
                    self.link_mask[pi, k, int(lid)] = 1.0

        print(f'  Pairs: {self.num_pairs}, Nodes: {self.num_nodes}, '
              f'Links: {self.num_links}, Max paths/pair: {self.max_k}')

        cache_data = {k: v for k, v in self.__dict__.items()
                      if not k.startswith('_') and k != 'graph'}
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
