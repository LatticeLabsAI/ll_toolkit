"""CAD chunking module.

This module provides chunking strategies for CAD documents,
including format-specific chunkers and utilities.

Format-Specific Chunkers:
    - step_chunker: Text-based chunking for STEP files
    - stl_chunker: Mesh segmentation for STL files
    - brep_chunker: Text-based chunking for BRep files
    - mesh_chunker: Generic mesh chunking utilities
    - dfs_chunker: DFS traversal-based chunking for CAD documents

Utilities:
    - tokenizer: Token counting and text splitting
    - serializer: Chunk serialization to various formats
    - visualizer: Chunk visualization and analysis

Base Classes:
    - BaseCADChunker: Abstract base chunker
    - HybridChunker: Hybrid chunking strategy
    - HierarchicalChunker: Hierarchical chunking strategy
    - DFSChunker: DFS traversal-based chunker
"""

from cadling.chunker.base_chunker import BaseCADChunker, CADChunk, CADChunkMeta
from cadling.chunker.dfs_chunker import DFSChunker
from cadling.chunker.hierarchical_chunker import CADHierarchicalChunker
from cadling.chunker.hybrid_chunker import CADHybridChunker

# Format-specific chunkers
from cadling.chunker import step_chunker, stl_chunker, brep_chunker, mesh_chunker, dfs_chunker

# Utilities
from cadling.chunker import tokenizer, serializer, visualizer

__all__ = [
    # Base classes
    "BaseCADChunker",
    "CADChunk",
    "CADChunkMeta",
    "CADHierarchicalChunker",
    "CADHybridChunker",
    "DFSChunker",
    # Submodules
    "step_chunker",
    "stl_chunker",
    "brep_chunker",
    "mesh_chunker",
    "dfs_chunker",
    "tokenizer",
    "serializer",
    "visualizer",
]
