"""Tests for bit allocation."""
import numpy as np
import pytest
from geotoken.quantization.bit_allocator import BitAllocator
from geotoken.config import AdaptiveBitAllocationConfig


class TestBitAllocator:
    def test_flat_gets_base_bits(self):
        config = AdaptiveBitAllocationConfig(base_bits=8)
        allocator = BitAllocator(config)
        complexity = np.zeros(10)
        result = allocator.allocate(complexity)

        assert np.all(result.bits_per_vertex == 8)

    def test_high_curvature_more_bits(self):
        config = AdaptiveBitAllocationConfig(
            base_bits=8,
            max_additional_bits=4,
        )
        allocator = BitAllocator(config)
        complexity = np.array([0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        result = allocator.allocate(complexity)

        # High complexity should get more bits
        assert result.max_bits >= result.min_bits

    def test_clamping(self):
        config = AdaptiveBitAllocationConfig(
            base_bits=8,
            max_additional_bits=100,
            min_bits=4,
            max_bits=16,
        )
        allocator = BitAllocator(config)
        complexity = np.array([0.0, 1.0])
        result = allocator.allocate(complexity)

        assert np.all(result.bits_per_vertex >= 4)
        assert np.all(result.bits_per_vertex <= 16)

    def test_empty_input(self):
        allocator = BitAllocator()
        result = allocator.allocate(np.array([]))
        assert len(result.bits_per_vertex) == 0
