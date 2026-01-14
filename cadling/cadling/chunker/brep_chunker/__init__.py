"""BRep-specific text-based chunker for CAD documents.

This module provides chunking strategies tailored for BRep (Boundary Representation) files,
which have hierarchical entity-based structure with topology relationships.

Classes:
    BRepChunker: BRep-specific text-based chunker
    EntityTypeChunker: Chunks by entity type (VERTEX, EDGE, FACE, etc.)
    TopologyChunker: Chunks by topological connectivity
    HierarchyChunker: Chunks by hierarchical structure
"""

from cadling.chunker.brep_chunker.brep_chunker import (
    BRepChunker,
    EntityTypeChunker,
    HierarchyChunker,
    TopologyChunker,
)

__all__ = [
    "BRepChunker",
    "EntityTypeChunker",
    "TopologyChunker",
    "HierarchyChunker",
]
