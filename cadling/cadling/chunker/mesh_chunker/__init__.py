"""Generic mesh chunking utilities for CAD documents.

This module provides format-agnostic mesh segmentation algorithms
that can be applied to various mesh representations.

Classes:
    MeshChunker: Main generic mesh chunker
    MeshData: Generic mesh data structure
    OctreeChunker: Octree-based spatial partitioning
    KMeansChunker: K-means clustering-based segmentation
    GraphChunker: Graph-based segmentation
    FeatureChunker: Feature-based segmentation
"""

from cadling.chunker.mesh_chunker.mesh_chunker import (
    FeatureChunker,
    GraphChunker,
    KMeansChunker,
    MeshChunker,
    MeshData,
    OctreeChunker,
)

__all__ = [
    "MeshChunker",
    "MeshData",
    "OctreeChunker",
    "KMeansChunker",
    "GraphChunker",
    "FeatureChunker",
]
