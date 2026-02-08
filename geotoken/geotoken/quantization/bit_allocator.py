"""Adaptive bit allocation based on geometric complexity.

Maps per-vertex complexity scores to bit widths using
percentile-based allocation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import AdaptiveBitAllocationConfig

_log = logging.getLogger(__name__)


@dataclass
class BitAllocationResult:
    """Per-vertex bit allocation results."""
    bits_per_vertex: np.ndarray     # Bit width per vertex (N,) int
    min_bits: int
    max_bits: int
    mean_bits: float


class BitAllocator:
    """Allocates bits per vertex based on complexity scores."""

    def __init__(self, config: Optional[AdaptiveBitAllocationConfig] = None):
        self.config = config or AdaptiveBitAllocationConfig()

    def allocate(self, complexity: np.ndarray) -> BitAllocationResult:
        """Allocate bits based on per-vertex complexity.

        Complexity is mapped to bits via percentile-based interpolation:
        - Below percentile_low -> base_bits
        - Above percentile_high -> base_bits + max_additional_bits
        - Between -> linear interpolation

        Args:
            complexity: (N,) per-vertex complexity scores

        Returns:
            BitAllocationResult with per-vertex bit widths
        """
        n = len(complexity)
        if n == 0:
            return BitAllocationResult(
                bits_per_vertex=np.array([], dtype=int),
                min_bits=0,
                max_bits=0,
                mean_bits=0.0,
            )

        # Handle constant complexity
        if np.max(complexity) - np.min(complexity) < 1e-12:
            bits = np.full(n, self.config.base_bits, dtype=int)
            return BitAllocationResult(
                bits_per_vertex=bits,
                min_bits=self.config.base_bits,
                max_bits=self.config.base_bits,
                mean_bits=float(self.config.base_bits),
            )

        # Percentile thresholds
        low_thresh = np.percentile(complexity, self.config.percentile_low)
        high_thresh = np.percentile(complexity, self.config.percentile_high)

        if high_thresh - low_thresh < 1e-12:
            bits = np.full(n, self.config.base_bits, dtype=int)
        else:
            # Linear interpolation between thresholds
            t = np.clip(
                (complexity - low_thresh) / (high_thresh - low_thresh),
                0.0, 1.0
            )
            bits_float = self.config.base_bits + t * self.config.max_additional_bits
            bits = np.round(bits_float).astype(int)

        # Clamp to [min_bits, max_bits]
        bits = np.clip(bits, self.config.min_bits, self.config.max_bits)

        return BitAllocationResult(
            bits_per_vertex=bits,
            min_bits=int(np.min(bits)),
            max_bits=int(np.max(bits)),
            mean_bits=float(np.mean(bits)),
        )
