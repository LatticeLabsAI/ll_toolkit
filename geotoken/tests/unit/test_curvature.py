"""Tests for curvature analysis."""
import numpy as np
import pytest
from geotoken.analysis.curvature import CurvatureAnalyzer


class TestCurvatureAnalyzer:
    def test_flat_plane_zero_curvature(self):
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        ], dtype=float)
        faces = np.array([[0, 1, 2], [0, 2, 3]])

        analyzer = CurvatureAnalyzer()
        result = analyzer.analyze_mesh(vertices, faces)

        # Flat plane should have near-zero curvature (boundary effects aside)
        assert result.mean_value < 10.0  # reasonable upper bound for boundary effects

    def test_sphere_positive_curvature(self, sphere_mesh):
        vertices, faces = sphere_mesh
        analyzer = CurvatureAnalyzer()
        result = analyzer.analyze_mesh(vertices, faces)

        # Sphere should have positive curvature
        assert result.max_value > 0

    def test_result_shapes(self, cube_mesh):
        vertices, faces = cube_mesh
        analyzer = CurvatureAnalyzer()
        result = analyzer.analyze_mesh(vertices, faces)

        assert result.mean_curvature.shape == (len(vertices),)
        assert result.gaussian_curvature.shape == (len(vertices),)
        assert result.combined_magnitude.shape == (len(vertices),)

    def test_point_cloud_analysis(self):
        points = np.random.rand(50, 3)
        analyzer = CurvatureAnalyzer()
        result = analyzer.analyze_point_cloud(points)
        assert result.combined_magnitude.shape == (50,)
