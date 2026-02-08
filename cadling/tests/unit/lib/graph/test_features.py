"""Unit tests for geometric feature computation utilities."""

import numpy as np
import pytest

from cadling.lib.graph.features import (
    compute_bounding_box,
    compute_dihedral_angle,
    compute_edge_length,
    compute_edge_midpoint,
    compute_face_area,
    compute_face_bounding_box,
    compute_face_centroid,
    compute_face_normal,
    compute_face_normal_normalized,
    normalize_features,
    standardize_features,
)


class TestFaceGeometry:
    """Test face geometric computation functions."""

    def test_compute_face_centroid(self):
        """Test centroid computation for a triangle."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        face = np.array([0, 1, 2])

        centroid = compute_face_centroid(vertices, face)

        # Centroid should be mean of three vertices
        expected = np.array([1/3, 1/3, 0.0])
        np.testing.assert_array_almost_equal(centroid, expected)

    def test_compute_face_centroid_different_positions(self):
        """Test centroid with different vertex positions."""
        vertices = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        face = np.array([0, 1, 2])

        centroid = compute_face_centroid(vertices, face)

        expected = np.array([4.0, 5.0, 6.0])  # Mean of the three points
        np.testing.assert_array_almost_equal(centroid, expected)

    def test_compute_face_normal(self):
        """Test normal vector computation."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        face = np.array([0, 1, 2])

        normal = compute_face_normal(vertices, face)

        # Normal should point in +z direction
        # Cross product of (1,0,0) and (0,1,0) is (0,0,1)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(normal, expected)

    def test_compute_face_normal_different_orientation(self):
        """Test normal with reversed winding order."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        face = np.array([0, 1, 2])

        normal = compute_face_normal(vertices, face)

        # Normal should point in -z direction
        expected = np.array([0.0, 0.0, -1.0])
        np.testing.assert_array_almost_equal(normal, expected)

    def test_compute_face_normal_normalized(self):
        """Test normalized normal vector computation."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ])
        face = np.array([0, 1, 2])

        normal = compute_face_normal_normalized(vertices, face)

        # Should be unit vector in +z direction
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(normal, expected)

        # Should have unit length
        assert np.isclose(np.linalg.norm(normal), 1.0)

    def test_compute_face_normal_normalized_degenerate(self):
        """Test normalized normal for degenerate face."""
        # Degenerate triangle (all points collinear)
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        face = np.array([0, 1, 2])

        normal = compute_face_normal_normalized(vertices, face)

        # Should return default z-axis for degenerate face
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(normal, expected)

    def test_compute_face_area(self):
        """Test face area computation."""
        # Right triangle with legs of length 1
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        face = np.array([0, 1, 2])

        area = compute_face_area(vertices, face)

        # Area should be 0.5
        assert np.isclose(area, 0.5)

    def test_compute_face_area_larger_triangle(self):
        """Test area for larger triangle."""
        # Triangle with base 4, height 3
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ])
        face = np.array([0, 1, 2])

        area = compute_face_area(vertices, face)

        # Area should be 6.0
        assert np.isclose(area, 6.0)

    def test_compute_face_bounding_box(self):
        """Test bounding box computation for a face."""
        vertices = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        face = np.array([0, 1, 2])

        min_corner, max_corner, dimensions = compute_face_bounding_box(vertices, face)

        np.testing.assert_array_almost_equal(min_corner, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(max_corner, [7.0, 8.0, 9.0])
        np.testing.assert_array_almost_equal(dimensions, [6.0, 6.0, 6.0])


class TestDihedralAngle:
    """Test dihedral angle computation."""

    def test_compute_dihedral_angle_parallel(self):
        """Test dihedral angle for parallel normals."""
        normal1 = np.array([0.0, 0.0, 1.0])
        normal2 = np.array([0.0, 0.0, 1.0])

        angle = compute_dihedral_angle(normal1, normal2)

        # Parallel normals should have angle 0
        assert np.isclose(angle, 0.0)

    def test_compute_dihedral_angle_opposite(self):
        """Test dihedral angle for opposite normals."""
        normal1 = np.array([0.0, 0.0, 1.0])
        normal2 = np.array([0.0, 0.0, -1.0])

        angle = compute_dihedral_angle(normal1, normal2)

        # Opposite normals should have angle π
        assert np.isclose(angle, np.pi)

    def test_compute_dihedral_angle_perpendicular(self):
        """Test dihedral angle for perpendicular normals."""
        normal1 = np.array([1.0, 0.0, 0.0])
        normal2 = np.array([0.0, 1.0, 0.0])

        angle = compute_dihedral_angle(normal1, normal2)

        # Perpendicular normals should have angle π/2
        assert np.isclose(angle, np.pi / 2)

    def test_compute_dihedral_angle_45_degrees(self):
        """Test dihedral angle for 45 degree angle."""
        normal1 = np.array([1.0, 0.0, 0.0])
        normal2 = np.array([1.0, 1.0, 0.0])

        angle = compute_dihedral_angle(normal1, normal2)

        # Should be π/4 radians (45 degrees)
        assert np.isclose(angle, np.pi / 4)

    def test_compute_dihedral_angle_degenerate(self):
        """Test dihedral angle with zero-length normal."""
        normal1 = np.array([0.0, 0.0, 0.0])
        normal2 = np.array([1.0, 0.0, 0.0])

        angle = compute_dihedral_angle(normal1, normal2)

        # Degenerate case should return 0
        assert np.isclose(angle, 0.0)

    def test_compute_dihedral_angle_non_unit_vectors(self):
        """Test dihedral angle with non-unit vectors."""
        normal1 = np.array([2.0, 0.0, 0.0])
        normal2 = np.array([0.0, 3.0, 0.0])

        angle = compute_dihedral_angle(normal1, normal2)

        # Should still be π/2 (perpendicular)
        assert np.isclose(angle, np.pi / 2)


class TestEdgeGeometry:
    """Test edge geometric computation functions."""

    def test_compute_edge_length(self):
        """Test edge length computation."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ])
        edge = (0, 1)

        length = compute_edge_length(vertices, edge)

        # Length should be 5 (3-4-5 triangle)
        assert np.isclose(length, 5.0)

    def test_compute_edge_length_unit(self):
        """Test unit edge length."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        edge = (0, 1)

        length = compute_edge_length(vertices, edge)

        assert np.isclose(length, 1.0)

    def test_compute_edge_length_3d(self):
        """Test edge length in 3D."""
        vertices = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 6.0, 8.0],
        ])
        edge = (0, 1)

        length = compute_edge_length(vertices, edge)

        # sqrt((4-1)^2 + (6-2)^2 + (8-3)^2) = sqrt(9 + 16 + 25) = sqrt(50)
        expected = np.sqrt(50)
        assert np.isclose(length, expected)

    def test_compute_edge_midpoint(self):
        """Test edge midpoint computation."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 4.0, 6.0],
        ])
        edge = (0, 1)

        midpoint = compute_edge_midpoint(vertices, edge)

        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(midpoint, expected)

    def test_compute_edge_midpoint_negative_coords(self):
        """Test midpoint with negative coordinates."""
        vertices = np.array([
            [-2.0, -4.0, -6.0],
            [2.0, 4.0, 6.0],
        ])
        edge = (0, 1)

        midpoint = compute_edge_midpoint(vertices, edge)

        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(midpoint, expected)


class TestBoundingBox:
    """Test bounding box computation."""

    def test_compute_bounding_box_simple(self):
        """Test bounding box for simple point set."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])

        min_corner, max_corner, dimensions = compute_bounding_box(points)

        np.testing.assert_array_almost_equal(min_corner, [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(max_corner, [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(dimensions, [1.0, 1.0, 1.0])

    def test_compute_bounding_box_multiple_points(self):
        """Test bounding box for multiple points."""
        points = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [0.5, 1.5, 2.5],
        ])

        min_corner, max_corner, dimensions = compute_bounding_box(points)

        np.testing.assert_array_almost_equal(min_corner, [0.5, 1.5, 2.5])
        np.testing.assert_array_almost_equal(max_corner, [7.0, 8.0, 9.0])
        np.testing.assert_array_almost_equal(dimensions, [6.5, 6.5, 6.5])

    def test_compute_bounding_box_negative_coords(self):
        """Test bounding box with negative coordinates."""
        points = np.array([
            [-1.0, -2.0, -3.0],
            [1.0, 2.0, 3.0],
        ])

        min_corner, max_corner, dimensions = compute_bounding_box(points)

        np.testing.assert_array_almost_equal(min_corner, [-1.0, -2.0, -3.0])
        np.testing.assert_array_almost_equal(max_corner, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(dimensions, [2.0, 4.0, 6.0])


class TestFeatureNormalization:
    """Test feature normalization functions."""

    def test_normalize_features(self):
        """Test feature normalization to zero mean and unit variance."""
        features = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])

        normalized = normalize_features(features)

        # Check zero mean
        mean = normalized.mean(axis=0)
        np.testing.assert_array_almost_equal(mean, [0.0, 0.0, 0.0], decimal=10)

        # Check unit variance (std should be 1)
        std = normalized.std(axis=0)
        np.testing.assert_array_almost_equal(std, [1.0, 1.0, 1.0], decimal=10)

    def test_normalize_features_single_value(self):
        """Test normalization with constant feature."""
        features = np.array([
            [5.0, 2.0],
            [5.0, 3.0],
            [5.0, 4.0],
        ])

        normalized = normalize_features(features)

        # First column should be all zeros (constant value)
        np.testing.assert_array_almost_equal(normalized[:, 0], [0.0, 0.0, 0.0])

        # Second column should be normalized
        assert not np.allclose(normalized[:, 1], 0.0)

    def test_standardize_features(self):
        """Test feature standardization to [0, 1] range."""
        features = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ])

        standardized = standardize_features(features)

        # Check range [0, 1]
        assert np.all(standardized >= 0.0)
        assert np.all(standardized <= 1.0)

        # Check min is 0, max is 1
        np.testing.assert_array_almost_equal(standardized.min(axis=0), [0.0, 0.0])
        np.testing.assert_array_almost_equal(standardized.max(axis=0), [1.0, 1.0])

    def test_standardize_features_with_precomputed_minmax(self):
        """Test standardization with pre-computed min/max."""
        features = np.array([
            [0.5, 15.0],
        ])

        feature_min = np.array([[0.0, 10.0]])
        feature_max = np.array([[1.0, 20.0]])

        standardized = standardize_features(features, feature_min, feature_max)

        expected = np.array([[0.5, 0.5]])
        np.testing.assert_array_almost_equal(standardized, expected)

    def test_standardize_features_constant_feature(self):
        """Test standardization with constant feature."""
        features = np.array([
            [5.0, 2.0],
            [5.0, 3.0],
            [5.0, 4.0],
        ])

        standardized = standardize_features(features)

        # First column should be all zeros (constant value)
        # Due to epsilon, it won't be exactly 0 but very small
        assert np.all(np.abs(standardized[:, 0]) < 1.0)

        # Second column should be standardized to [0, 1]
        assert np.isclose(standardized[0, 1], 0.0)
        assert np.isclose(standardized[2, 1], 1.0)


class TestFeatureComputationIntegration:
    """Integration tests for feature computations."""

    def test_compute_multiple_face_features(self):
        """Test computing multiple features for the same face."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        face = np.array([0, 1, 2])

        # Compute all features
        centroid = compute_face_centroid(vertices, face)
        normal = compute_face_normal_normalized(vertices, face)
        area = compute_face_area(vertices, face)
        bbox = compute_face_bounding_box(vertices, face)

        # Verify all return valid results
        assert centroid.shape == (3,)
        assert normal.shape == (3,)
        assert np.isscalar(area) or area.shape == ()
        assert len(bbox) == 3

        # Check consistency
        assert np.isclose(np.linalg.norm(normal), 1.0)  # Normal should be unit
        assert area > 0  # Area should be positive

    def test_edge_and_face_features_consistency(self):
        """Test that edge and face features are consistent."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        face = np.array([0, 1, 2])
        edge = (0, 1)

        face_centroid = compute_face_centroid(vertices, face)
        edge_midpoint = compute_edge_midpoint(vertices, edge)
        edge_length = compute_edge_length(vertices, edge)

        # All should return valid numpy arrays or scalars
        assert isinstance(face_centroid, np.ndarray)
        assert isinstance(edge_midpoint, np.ndarray)
        assert isinstance(edge_length, (float, np.floating))

        # Edge length should be 1.0
        assert np.isclose(edge_length, 1.0)
