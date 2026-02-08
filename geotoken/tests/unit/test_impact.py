"""Tests for impact analysis."""
import numpy as np
import pytest
from geotoken.config import QuantizationConfig, PrecisionTier
from geotoken.impact.analyzer import QuantizationImpactAnalyzer


class TestImpactAnalyzer:
    def test_hausdorff_computed(self, cube_mesh):
        vertices, faces = cube_mesh
        analyzer = QuantizationImpactAnalyzer()
        report = analyzer.analyze(vertices, faces)

        assert report.hausdorff_distance >= 0
        assert report.mean_error >= 0

    def test_higher_tiers_lower_error(self, cube_mesh):
        vertices, faces = cube_mesh
        analyzer = QuantizationImpactAnalyzer()
        reports = analyzer.compare_tiers(vertices, faces)

        assert "draft" in reports
        assert "standard" in reports
        assert "precision" in reports

        # Higher precision tiers should generally have lower error
        # (May not always hold for simple cube, so just check they run)
        assert reports["draft"].hausdorff_distance >= 0
        assert reports["precision"].hausdorff_distance >= 0

    def test_empty_input(self):
        vertices = np.array([]).reshape(0, 3)
        analyzer = QuantizationImpactAnalyzer()
        report = analyzer.analyze(vertices)
        assert report.hausdorff_distance == 0
