"""Unit tests for GeoTokenizer."""
from __future__ import annotations

import numpy as np
import pytest

from geotoken.tokenizer.geo_tokenizer import GeoTokenizer


@pytest.fixture
def tokenizer():
    return GeoTokenizer()


class TestGeoTokenizer:
    def test_tokenize_cube(self, tokenizer, cube_mesh):
        vertices, faces = cube_mesh
        result = tokenizer.tokenize(vertices, faces)
        assert len(result.coordinate_tokens) == 8
        assert len(result.geometry_tokens) == 12

    def test_tokenize_empty(self, tokenizer):
        vertices = np.zeros((0, 3), dtype=float)
        result = tokenizer.tokenize(vertices)
        assert len(result.coordinate_tokens) == 0
        assert len(result.geometry_tokens) == 0

    def test_detokenize_roundtrip(self, tokenizer, cube_mesh):
        vertices, faces = cube_mesh
        tokens = tokenizer.tokenize(vertices, faces)
        reconstructed = tokenizer.detokenize(tokens)
        assert reconstructed.shape == vertices.shape
        # Quantization introduces error, but it should be small
        error = np.max(np.abs(reconstructed - vertices))
        assert error < 0.1, f"Max reconstruction error {error} too large"

    def test_analyze_impact(self, tokenizer, cube_mesh):
        vertices, faces = cube_mesh
        report = tokenizer.analyze_impact(vertices, faces)
        assert hasattr(report, "hausdorff_distance")
        assert hasattr(report, "mean_error")
        assert hasattr(report, "max_error")
        assert hasattr(report, "relationship_preservation_rate")

    def test_detokenize_with_list_scale_metadata(self, tokenizer, cube_mesh):
        """Regression: scale serialized as list (from JSON) must roundtrip correctly."""
        vertices, faces = cube_mesh
        tokens = tokenizer.tokenize(vertices, faces)
        # Simulate JSON serialization: convert ndarray scale to list
        if isinstance(tokens.metadata.get("norm_scale"), np.ndarray):
            tokens.metadata["norm_scale"] = tokens.metadata["norm_scale"].tolist()
        elif isinstance(tokens.metadata.get("norm_scale"), (int, float)):
            tokens.metadata["norm_scale"] = [float(tokens.metadata["norm_scale"])] * 3
        reconstructed = tokenizer.detokenize(tokens)
        assert reconstructed.shape == vertices.shape
        error = np.max(np.abs(reconstructed - vertices))
        assert error < 0.1, f"Max reconstruction error {error} too large with list scale"

    def test_invalid_input_shape(self, tokenizer):
        bad = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match=r"must be \(N, 3\)"):
            tokenizer.tokenize(bad)
