"""Geometric analysis modules for adaptive quantization."""
from __future__ import annotations

from .curvature import CurvatureAnalyzer, CurvatureResult
from .feature_density import FeatureDensityAnalyzer, FeatureDensityResult
from .geometric_relationships import RelationshipDetector, GeometricRelationship

__all__ = [
    "CurvatureAnalyzer",
    "CurvatureResult",
    "FeatureDensityAnalyzer",
    "FeatureDensityResult",
    "RelationshipDetector",
    "GeometricRelationship",
]
