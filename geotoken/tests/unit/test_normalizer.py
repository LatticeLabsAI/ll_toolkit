"""Tests for normalization."""
import numpy as np
import pytest
from geotoken.quantization.normalizer import RelationshipPreservingNormalizer


class TestNormalizer:
    def test_fits_bounding_cube(self, cube_mesh):
        vertices, _ = cube_mesh
        normalizer = RelationshipPreservingNormalizer()
        result = normalizer.normalize(vertices)

        norm_v = result.normalized_vertices
        assert np.all(norm_v >= -0.01)
        assert np.all(norm_v <= 1.01)

    def test_denormalize_roundtrip(self, cube_mesh):
        vertices, _ = cube_mesh
        normalizer = RelationshipPreservingNormalizer()
        result = normalizer.normalize(vertices)
        recovered = normalizer.denormalize(result.normalized_vertices, result)

        np.testing.assert_allclose(recovered, vertices, atol=1e-10)

    def test_preserves_proportions(self):
        # Rectangular box - should preserve aspect ratio
        vertices = np.array([
            [0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0],
            [0, 0, 3], [2, 0, 3], [2, 1, 3], [0, 1, 3],
        ], dtype=float)
        normalizer = RelationshipPreservingNormalizer()
        result = normalizer.normalize(vertices)

        # Uniform scale preserves aspect ratio
        norm_v = result.normalized_vertices
        x_range = norm_v[:, 0].max() - norm_v[:, 0].min()
        y_range = norm_v[:, 1].max() - norm_v[:, 1].min()
        z_range = norm_v[:, 2].max() - norm_v[:, 2].min()

        # Original ratio is 2:1:3, should be preserved
        np.testing.assert_allclose(x_range / z_range, 2.0 / 3.0, atol=1e-10)
        np.testing.assert_allclose(y_range / z_range, 1.0 / 3.0, atol=1e-10)

    def test_empty_vertices(self):
        vertices = np.array([]).reshape(0, 3)
        normalizer = RelationshipPreservingNormalizer()
        result = normalizer.normalize(vertices)
        assert result.scale == 1.0

    def test_denormalize_single_axis_degenerate(self):
        """Denormalize must not produce inf when one axis has zero scale.

        This covers the case where a NormalizationResult is deserialized
        or constructed with a zero in a per-axis scale array.
        """
        from geotoken.quantization.normalizer import NormalizationResult

        normalizer = RelationshipPreservingNormalizer()
        norm_verts = np.array([[0.5, 0.5, 0.5], [1.0, 0.5, 1.0]], dtype=float)

        # Simulate a result where Y-axis scale is zero (degenerate)
        result = NormalizationResult(
            normalized_vertices=norm_verts,
            center=np.array([2.0, 5.0, 1.5]),
            scale=np.array([4.0, 0.0, 3.0]),  # Y is degenerate
            original_bbox_min=np.array([0.0, 5.0, 0.0]),
            original_bbox_max=np.array([4.0, 5.0, 3.0]),
        )

        recovered = normalizer.denormalize(norm_verts, result)
        assert np.all(np.isfinite(recovered)), "denormalize produced inf/nan on single-axis zero scale"
