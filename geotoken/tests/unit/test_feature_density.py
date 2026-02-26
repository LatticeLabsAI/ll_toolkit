"""Tests for FeatureDensityAnalyzer."""
from __future__ import annotations

import numpy as np
import pytest

from geotoken.analysis.feature_density import FeatureDensityAnalyzer, FeatureDensityResult


class TestFeatureDensityAnalyzerInit:
    """Tests for FeatureDensityAnalyzer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        analyzer = FeatureDensityAnalyzer()
        assert analyzer.n_neighbors == 12

    def test_custom_n_neighbors(self):
        """Test initialization with custom n_neighbors."""
        analyzer = FeatureDensityAnalyzer(n_neighbors=20)
        assert analyzer.n_neighbors == 20


class TestAnalyzeMesh:
    """Tests for analyze() method on meshes."""

    def test_analyze_cube_mesh(self, cube_mesh):
        """Test analysis on cube mesh produces low variance."""
        vertices, faces = cube_mesh
        analyzer = FeatureDensityAnalyzer()
        result = analyzer.analyze(vertices, faces)

        assert isinstance(result, FeatureDensityResult)
        assert len(result.density) == len(vertices)
        assert len(result.edge_length_variance) == len(vertices)
        assert len(result.face_area_variance) == len(vertices)

    def test_analyze_sphere_mesh(self, sphere_mesh):
        """Test analysis on sphere mesh."""
        vertices, faces = sphere_mesh
        analyzer = FeatureDensityAnalyzer()
        result = analyzer.analyze(vertices, faces)

        assert len(result.density) == len(vertices)
        # Sphere should have relatively uniform density
        assert result.mean_value >= 0.0
        assert result.max_value >= result.min_value

    def test_analyze_cylinder_mesh(self, cylinder_mesh):
        """Test analysis on cylinder mesh."""
        vertices, faces = cylinder_mesh
        analyzer = FeatureDensityAnalyzer()
        result = analyzer.analyze(vertices, faces)

        assert len(result.density) == len(vertices)

    def test_density_values_normalized(self, cube_mesh):
        """Test that density values are in [0, 1]."""
        vertices, faces = cube_mesh
        analyzer = FeatureDensityAnalyzer()
        result = analyzer.analyze(vertices, faces)

        assert np.all(result.density >= 0.0)
        assert np.all(result.density <= 1.0)

    def test_variable_density_mesh(self):
        """Test analysis on mesh with variable density."""
        # Create a mesh with intentionally variable face sizes
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 0.1, 0],  # Small triangle
            [2, 0, 0], [3, 0, 0], [2.5, 1.0, 0],  # Larger triangle
            [4, 0, 0], [5, 0, 0], [4.5, 0.05, 0],  # Very small triangle
        ], dtype=float)
        faces = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ])
        analyzer = FeatureDensityAnalyzer()
        result = analyzer.analyze(vertices, faces)

        # Should have some variance in density
        assert len(result.density) == 9


class TestAnalyzePointCloud:
    """Tests for analyze_point_cloud() method."""

    def test_analyze_point_cloud_empty(self):
        """Test analysis on empty point cloud."""
        analyzer = FeatureDensityAnalyzer()
        result = analyzer.analyze_point_cloud(np.zeros((0, 3)))

        assert len(result.density) == 0
        assert result.min_value == 0.0
        assert result.max_value == 0.0
        assert result.mean_value == 0.0

    def test_analyze_point_cloud_uniform(self):
        """Test analysis on uniformly distributed points."""
        # Create uniform grid of points
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        z = np.linspace(0, 1, 10)
        xx, yy, zz = np.meshgrid(x, y, z)
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        analyzer = FeatureDensityAnalyzer()
        result = analyzer.analyze_point_cloud(points)

        assert len(result.density) == len(points)
        # Uniform grid should have relatively uniform density
        # (low variance in density values)
        density_std = np.std(result.density)
        # Allow some variance but should be relatively uniform
        assert density_std < 0.15

    def test_analyze_point_cloud_variable_density(self):
        """Test analysis on points with variable density."""
        # Create clusters with different densities
        np.random.seed(42)
        # Dense cluster
        cluster1 = np.random.randn(50, 3) * 0.1
        # Sparse cluster
        cluster2 = np.random.randn(10, 3) * 0.5 + np.array([5, 0, 0])
        points = np.vstack([cluster1, cluster2])

        analyzer = FeatureDensityAnalyzer()
        result = analyzer.analyze_point_cloud(points)

        assert len(result.density) == len(points)
        # Should have varying density values
        assert result.max_value > result.min_value

    def test_analyze_point_cloud_shapes(self):
        """Test that result arrays have correct shapes."""
        np.random.seed(42)
        points = np.random.randn(100, 3)
        analyzer = FeatureDensityAnalyzer()
        result = analyzer.analyze_point_cloud(points)

        assert result.density.shape == (100,)
        assert result.edge_length_variance.shape == (100,)
        # Face area variance is None for point clouds (no faces)
        assert result.face_area_variance is None

    def test_analyze_point_cloud_few_points(self):
        """Test analysis with fewer points than neighbors."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        analyzer = FeatureDensityAnalyzer(n_neighbors=12)
        result = analyzer.analyze_point_cloud(points)

        assert len(result.density) == 3

    def test_analyze_point_cloud_single_point(self):
        """Test analysis with single point."""
        points = np.array([[0, 0, 0]])
        analyzer = FeatureDensityAnalyzer()
        result = analyzer.analyze_point_cloud(points)

        assert len(result.density) == 1


class TestFeatureDensityResult:
    """Tests for FeatureDensityResult dataclass."""

    def test_result_attributes(self):
        """Test result has expected attributes."""
        result = FeatureDensityResult(
            density=np.array([0.1, 0.2, 0.3]),
            edge_length_variance=np.array([0.01, 0.02, 0.03]),
            face_area_variance=np.array([0.001, 0.002, 0.003]),
            min_value=0.1,
            max_value=0.3,
            mean_value=0.2,
        )
        assert len(result.density) == 3
        assert result.min_value == 0.1
        assert result.max_value == 0.3
        assert result.mean_value == 0.2
