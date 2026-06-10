"""UV-grid geometry encoders (UV-Net style, written from scratch).

Two small convolutional encoders turn the sampled B-Rep geometry into fixed-size
embeddings that are fused with the topological features:

* :class:`UVNetSurfaceEncoder` -- a 2D CNN over a face's ``[7, U, V]`` UV-grid
  (xyz + normal + trimming mask) -> ``[out_dim]``.
* :class:`UVNetCurveEncoder` -- a 1D CNN over an edge's ``[6, U]`` U-grid
  (xyz + tangent) -> ``[out_dim]``.

These are standard convolutional stacks (the generic UV-Net idea of running a
CNN over the parametric grid), implemented independently for ``ll_brepnet``.
"""

from __future__ import annotations

import torch
from torch import nn


class UVNetSurfaceEncoder(nn.Module):
    """2D CNN encoder for face UV-grids ``[B, in_channels, U, V]``.

    Args:
        in_channels: Input channels (7 = xyz + normal + trimming mask).
        out_dim: Output embedding dimension.
    """

    def __init__(self, in_channels: int = 7, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, in_channels, U, V] -> [B, out_dim]``."""
        if x.shape[0] == 0:
            return x.new_zeros((0, self.out_dim))
        return self.net(x).flatten(1)


class UVNetCurveEncoder(nn.Module):
    """1D CNN encoder for edge U-grids ``[B, in_channels, U]``.

    Args:
        in_channels: Input channels (6 = xyz + tangent).
        out_dim: Output embedding dimension.
    """

    def __init__(self, in_channels: int = 6, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, in_channels, U] -> [B, out_dim]``."""
        if x.shape[0] == 0:
            return x.new_zeros((0, self.out_dim))
        return self.net(x).flatten(1)
