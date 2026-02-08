"""Tests for configuration."""
from geotoken.config import PrecisionTier, QuantizationConfig, NormalizationConfig, AdaptiveBitAllocationConfig


class TestPrecisionTier:
    def test_tier_bits(self):
        assert PrecisionTier.DRAFT.bits == 6
        assert PrecisionTier.STANDARD.bits == 8
        assert PrecisionTier.PRECISION.bits == 10

    def test_tier_levels(self):
        assert PrecisionTier.DRAFT.levels == 64
        assert PrecisionTier.STANDARD.levels == 256
        assert PrecisionTier.PRECISION.levels == 1024


class TestQuantizationConfig:
    def test_default_config(self):
        config = QuantizationConfig()
        assert config.tier == PrecisionTier.STANDARD
        assert config.adaptive is True
        assert config.minimum_feature_threshold == 0.05

    def test_custom_config(self):
        config = QuantizationConfig(tier=PrecisionTier.PRECISION, adaptive=False)
        assert config.tier.bits == 10
        assert config.adaptive is False
