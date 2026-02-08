"""End-to-end integration tests."""
import numpy as np
import pytest
from geotoken import GeoTokenizer, QuantizationConfig, PrecisionTier


class TestEndToEnd:
    def test_tokenize_detokenize_cube(self, cube_mesh):
        vertices, faces = cube_mesh
        tokenizer = GeoTokenizer()
        tokens = tokenizer.tokenize(vertices, faces)
        reconstructed = tokenizer.detokenize(tokens)

        assert reconstructed.shape == vertices.shape
        errors = np.linalg.norm(vertices - reconstructed, axis=1)
        assert np.max(errors) < 1.0

    def test_all_tiers_monotonic_quality(self, cube_mesh):
        vertices, faces = cube_mesh
        errors_by_tier = {}

        for tier in PrecisionTier:
            config = QuantizationConfig(tier=tier, adaptive=False)
            tokenizer = GeoTokenizer(config)
            tokens = tokenizer.tokenize(vertices, faces)
            reconstructed = tokenizer.detokenize(tokens)
            mean_error = float(np.mean(np.linalg.norm(
                vertices - reconstructed, axis=1
            )))
            errors_by_tier[tier.value] = mean_error

        # Higher precision should give lower or equal error
        assert errors_by_tier["precision"] <= errors_by_tier["standard"] + 1e-6
        assert errors_by_tier["standard"] <= errors_by_tier["draft"] + 1e-6

    def test_sphere_tokenization(self, sphere_mesh):
        vertices, faces = sphere_mesh
        tokenizer = GeoTokenizer()
        tokens = tokenizer.tokenize(vertices, faces)

        assert tokens.total_tokens > 0
        assert len(tokens.coordinate_tokens) == len(vertices)

    def test_impact_analysis(self, cube_mesh):
        vertices, faces = cube_mesh
        tokenizer = GeoTokenizer()
        report = tokenizer.analyze_impact(vertices, faces)

        assert report.hausdorff_distance >= 0
        assert report.mean_error >= 0

    def test_uniform_mode(self, cube_mesh):
        vertices, faces = cube_mesh
        config = QuantizationConfig(adaptive=False)
        tokenizer = GeoTokenizer(config)
        tokens = tokenizer.tokenize(vertices, faces)

        # All tokens should have same bit width
        bits = set(t.bits for t in tokens.coordinate_tokens)
        assert len(bits) == 1
