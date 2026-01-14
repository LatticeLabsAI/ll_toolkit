"""STEP-specific chunker for CAD documents.

This module provides chunking strategies tailored for STEP (ISO-10303-21) files,
which have entity-based structure with topology relationships.

Classes:
    STEPChunker: STEP-specific chunker
    EntityGroupChunker: Chunks by entity type groups
    TopologyChunker: Chunks by topological connectivity
"""

from cadling.chunker.step_chunker.step_chunker import (
    EntityGroupChunker,
    STEPChunker,
    TopologyChunker,
)

__all__ = [
    "STEPChunker",
    "EntityGroupChunker",
    "TopologyChunker",
]
