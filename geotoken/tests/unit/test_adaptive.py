"""Tests for adaptive quantizer."""
import numpy as np
import pytest
from geotoken.config import QuantizationConfig, PrecisionTier
from geotoken.quantization.adaptive import AdaptiveQuantizer


class TestAdaptiveQuantizer:
    def test_cube_quantization(self, cube_mesh):
        vertices, faces = cube_mesh
        quantizer = AdaptiveQuantizer()
        result = quantizer.quantize(vertices, faces)

        assert result.quantized_vertices.shape == (8, 3)
        assert result.bits_per_vertex.shape == (8,)
        assert result.total_bits > 0

    def test_roundtrip_fidelity(self, cube_mesh):
        vertices, faces = cube_mesh
        quantizer = AdaptiveQuantizer()
        result = quantizer.quantize(vertices, faces)
        reconstructed = quantizer.dequantize(result)

        # Reconstruction should be close to original
        errors = np.linalg.norm(vertices - reconstructed, axis=1)
        assert np.max(errors) < 0.05  # within 5% of a unit

    def test_sphere_variable_bits(self, sphere_mesh):
        vertices, faces = sphere_mesh
        config = QuantizationConfig(
            tier=PrecisionTier.STANDARD,
            adaptive=True,
        )
        quantizer = AdaptiveQuantizer(config)
        result = quantizer.quantize(vertices, faces)

        # Sphere should have some variation in bits
        assert result.bits_per_vertex.shape == (len(vertices),)

    def test_empty_input(self):
        vertices = np.array([]).reshape(0, 3)
        quantizer = AdaptiveQuantizer()
        result = quantizer.quantize(vertices)
        assert result.quantized_vertices.shape == (0, 3)

    def test_feature_collapse_prevention(self):
        # Two close but distinct vertices
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [0.06, 0.0, 0.0],  # Just above default threshold 0.05
            [1.0, 1.0, 1.0],
        ])
        config = QuantizationConfig(
            tier=PrecisionTier.DRAFT,  # Low precision, more likely to collapse
            minimum_feature_threshold=0.05,
        )
        quantizer = AdaptiveQuantizer(config)
        result = quantizer.quantize(vertices)

        # After collapse prevention, close vertices should still be distinct
        q = result.quantized_vertices
        # At least check that the function runs without error
        assert q.shape == (3, 3)
        # The two close vertices must remain distinct after quantization
        assert not np.array_equal(q[0], q[1])
