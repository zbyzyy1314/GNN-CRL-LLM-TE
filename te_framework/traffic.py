"""
Traffic: loads traffic matrices from TXT files.

Supports the format produced by convert_geant.py:
  - Each line is a flattened N×N matrix (space-separated)
  - Values are scaled: file_value = original_kbps * 375 (for scale=100)
"""

import os
import numpy as np


class TrafficLoader:
    def __init__(self, traffic_file: str, num_nodes: int, scale: float = 100.0):
        """
        Args:
            traffic_file: path to traffic matrix file
            num_nodes: number of nodes in topology
            scale: the scale factor used for file values
                   (file_value = original_kbps * 300 * 1000 / (scale * 8))
                   default: 100 → factor 375
        """
        self.traffic_file = traffic_file
        self.num_nodes = num_nodes
        self.scale = scale
        self._load()

    def _load(self):
        print(f'[*] Loading traffic matrices from {self.traffic_file}')
        matrices = []
        with open(self.traffic_file, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) != self.num_nodes * self.num_nodes:
                    continue
                mat = np.array(values, dtype=np.float32).reshape(
                    self.num_nodes, self.num_nodes)
                # Zero out diagonal (self-traffic)
                np.fill_diagonal(mat, 0)
                matrices.append(mat)

        self.raw_matrices = np.stack(matrices, axis=0)  # (T, N, N)
        self.num_tms = self.raw_matrices.shape[0]
        print(f'  Loaded {self.num_tms} traffic matrices, '
              f'shape: {self.raw_matrices.shape}')

    def get_real_traffic(self, normalize: bool = False):
        """
        Convert file values back to real kbps and optionally normalize.

        Args:
            normalize: if True, divide each matrix by its max value

        Returns:
            np.ndarray of shape (T, N, N)
        """
        # Reverse the scaling: file_value / 375 * 100 → original_kbps
        # Actually the env.py does: self.traffic_matrices*100*8/300/1000
        # So real_kbps = file_value * 100 * 8 / 300 / 1000 = file_value / 375
        conversion = self.scale * 8 / 300 / 1000  # = 100*8/300000 ≈ 0.002667
        real = self.raw_matrices * conversion  # now in kbps

        if normalize:
            max_vals = np.max(real, axis=(1, 2), keepdims=True)
            max_vals = np.maximum(max_vals, 1e-10)  # avoid division by zero
            real = real / max_vals

        return real

    def split(self, train_ratio=0.8):
        """Split into train/test sets."""
        split = int(self.num_tms * train_ratio)
        return self.raw_matrices[:split], self.raw_matrices[split:]
