"""
Neural network architectures for TE policy and value functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PathSelectionNetwork(nn.Module):
    """
    CNN-based network for path-level TE decisions.

    Input:  (batch, 1, N, N) traffic matrix
    Output: (batch, num_pairs, max_k) path logits (masked)
    """

    def __init__(self, num_nodes: int, num_pairs: int, max_k: int,
                 conv_channels: list = [32, 64],
                 fc_dims: list = [256, 256],
                 pool_size: int = 8):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_pairs = num_pairs
        self.max_k = max_k

        # CNN feature extractor with pooling to reduce parameters
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2))
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d((pool_size, pool_size)))
        self.conv = nn.Sequential(*layers)

        # Flattened size after pooling
        conv_out_size = conv_channels[-1] * pool_size * pool_size

        # FC layers
        fc_layers = []
        in_dim = conv_out_size
        for dim in fc_dims:
            fc_layers.append(nn.Linear(in_dim, dim))
            fc_layers.append(nn.LeakyReLU(0.2))
            in_dim = dim
        self.fc = nn.Sequential(*fc_layers)

        # Path selection head
        self.path_head = nn.Linear(in_dim, num_pairs * max_k)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, path_mask: torch.Tensor = None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            x = x[:, -1:]      # temporal: take last frame (closest to target)

        batch = x.shape[0]
        feat = self.conv(x)
        feat = feat.reshape(batch, -1)
        feat = self.fc(feat)
        logits = self.path_head(feat).reshape(batch, self.num_pairs, self.max_k)

        if path_mask is not None:
            mask = path_mask.unsqueeze(0)
            logits = torch.where(mask, logits,
                                 torch.full_like(logits, -1e9))
        return logits

    def get_action(self, x: torch.Tensor, path_mask: torch.Tensor,
                   deterministic: bool = False):
        logits = self.forward(x, path_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            actions = probs.argmax(dim=-1)
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        return actions, log_probs, dist.entropy()


class ValueNetwork(nn.Module):
    """Value (critic) network with same architecture as policy."""

    def __init__(self, num_nodes: int,
                 conv_channels: list = [32, 64],
                 fc_dims: list = [256, 128],
                 pool_size: int = 8):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(0.2))
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d((pool_size, pool_size)))
        self.conv = nn.Sequential(*layers)

        conv_out_size = conv_channels[-1] * pool_size * pool_size

        fc_layers = []
        in_dim = conv_out_size
        for dim in fc_dims:
            fc_layers.append(nn.Linear(in_dim, dim))
            fc_layers.append(nn.LeakyReLU(0.2))
            in_dim = dim
        fc_layers.append(nn.Linear(in_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            x = x[:, -1:]      # temporal: take last frame
        batch = x.shape[0]
        feat = self.conv(x)
        feat = feat.reshape(batch, -1)
        value = self.fc(feat)
        return value.squeeze(-1)
