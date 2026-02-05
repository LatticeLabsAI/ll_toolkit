"""Experimental pipelines for CADling.

This module provides experimental pipelines for advanced CAD processing workflows,
including threaded geometry-VLM analysis, multi-view fusion, and assembly hierarchy.
"""

from .assembly_hierarchy_pipeline import AssemblyHierarchyPipeline, AssemblyNode
from .multi_view_fusion_pipeline import MultiViewFusionPipeline
from .threaded_geometry_vlm_pipeline import ThreadedGeometryVlmPipeline

__all__ = [
    # Threaded Geometry-VLM Pipeline (Feature 1)
    "ThreadedGeometryVlmPipeline",
    # Multi-View Fusion Pipeline (Feature 2)
    "MultiViewFusionPipeline",
    # Assembly Hierarchy Pipeline (Feature 3)
    "AssemblyHierarchyPipeline",
    "AssemblyNode",
]
