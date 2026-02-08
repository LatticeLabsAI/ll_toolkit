"""Geometry utilities and analysis for CAD models.

Submodules:
    distribution_analyzer: Statistical analysis of geometric feature distributions
        (dihedral angles, curvature, surface types, B-Rep hierarchy)
    face_geometry: Face-level geometric feature extraction (curvature, normals, area)
    uv_grid_extractor: UV parametric grid sampling for B-Rep surfaces and edges
"""

from cadling.lib.geometry.distribution_analyzer import (
    BRepHierarchyAnalyzer,
    CurvatureAnalyzer,
    DihedralAngleAnalyzer,
    SurfaceTypeAnalyzer,
)
from cadling.lib.geometry.face_geometry import FaceGeometryExtractor
from cadling.lib.geometry.uv_grid_extractor import EdgeUVGridExtractor, FaceUVGridExtractor

__all__ = [
    "DihedralAngleAnalyzer",
    "CurvatureAnalyzer",
    "SurfaceTypeAnalyzer",
    "BRepHierarchyAnalyzer",
    "FaceGeometryExtractor",
    "FaceUVGridExtractor",
    "EdgeUVGridExtractor",
]
