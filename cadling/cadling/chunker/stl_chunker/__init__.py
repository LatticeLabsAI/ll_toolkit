"""STL mesh segmentation chunker for CAD documents.

This module provides geometric mesh segmentation for STL files,
chunking the actual 3D triangle mesh geometry using mesh processing algorithms.

Classes:
    STLMeshChunker: Main mesh segmentation chunker for STL files
    RegionGrowingChunker: Segments mesh by region growing based on normals
    WatershedChunker: Segments mesh using watershed algorithm
    CurvatureChunker: Segments mesh based on curvature analysis
    ConnectedComponentsChunker: Segments disconnected mesh components
"""

from cadling.chunker.stl_chunker.stl_chunker import (
    ConnectedComponentsChunker,
    CurvatureChunker,
    RegionGrowingChunker,
    STLMeshChunker,
    WatershedChunker,
)

__all__ = [
    "STLMeshChunker",
    "RegionGrowingChunker",
    "WatershedChunker",
    "CurvatureChunker",
    "ConnectedComponentsChunker",
]
