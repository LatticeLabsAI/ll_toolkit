"""Uniform (fixed-precision) quantization.

Baseline quantizer using the same bit width for all vertices,
matching the DeepCAD approach.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import QuantizationConfig, PrecisionTier

_log = logging.getLogger(__name__)


class UniformQuantizer:
    """Fixed-precision quantizer (DeepCAD baseline).

    Quantizes all coordinates to the same number of levels.
    """

    def __init__(self, bits: int = 8):
        """Initialize with fixed bit width.

        Args:
            bits: Bit width for all coordinates (default 8 = 256 levels)
        """
        if not (1 <= bits <= 32):
            raise ValueError(f"bits must be 1-32, got {bits}")
        self.bits = bits
        self.levels = 2 ** bits

    @classmethod
    def from_tier(cls, tier: PrecisionTier) -> "UniformQuantizer":
        """Create quantizer from precision tier."""
        return cls(bits=tier.bits)

    def quantize(self, values: np.ndarray) -> np.ndarray:
        """Quantize normalized values [0, 1] to integer levels.

        Args:
            values: Normalized values in [0, 1]

        Returns:
            Integer quantized values in [0, levels-1]
        """
        clamped = np.clip(values, 0.0, 1.0)
        quantized = np.round(clamped * (self.levels - 1)).astype(int)
        return np.clip(quantized, 0, self.levels - 1)

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """Dequantize integer levels back to [0, 1] range.

        Args:
            quantized: Integer values in [0, levels-1]

        Returns:
            Reconstructed values in [0, 1]
        """
        return quantized.astype(float) / (self.levels - 1)
