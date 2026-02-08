"""Tests for uniform quantizer."""
import numpy as np
from geotoken.quantization.uniform import UniformQuantizer
from geotoken.config import PrecisionTier


class TestUniformQuantizer:
    def test_quantize_range(self):
        q = UniformQuantizer(bits=8)
        values = np.array([[0.0, 0.5, 1.0]])
        result = q.quantize(values)
        assert result[0, 0] == 0
        assert result[0, 1] == 128  # round(0.5 * 255)
        assert result[0, 2] == 255

    def test_dequantize_roundtrip(self):
        q = UniformQuantizer(bits=8)
        values = np.array([[0.0, 0.5, 1.0]])
        quantized = q.quantize(values)
        recovered = q.dequantize(quantized)
        np.testing.assert_allclose(recovered, values, atol=1.0 / 255)

    def test_from_tier(self):
        q = UniformQuantizer.from_tier(PrecisionTier.DRAFT)
        assert q.bits == 6
        assert q.levels == 64

    def test_clamp(self):
        q = UniformQuantizer(bits=8)
        values = np.array([[-0.5, 1.5, 0.5]])
        result = q.quantize(values)
        assert result[0, 0] == 0
        assert result[0, 1] == 255
