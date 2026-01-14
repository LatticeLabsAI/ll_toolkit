"""Base chunker for CAD documents.

This module provides the foundational classes for chunking CADlingDocument
into pieces suitable for RAG (Retrieval-Augmented Generation) systems.

Classes:
    BaseCADChunker: Abstract base class for all chunkers
    CADChunk: Single chunk with text and metadata
    CADChunkMeta: Metadata for chunks
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator, Optional

from pydantic import BaseModel, Field
from stepnet import STEPTokenizer

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADlingDocument, TopologyGraph

_log = logging.getLogger(__name__)


class CADChunkMeta(BaseModel):
    """Metadata for a CAD chunk.

    Contains CAD-specific metadata that helps with retrieval and
    understanding of chunks in RAG systems.

    Attributes:
        entity_types: Types of STEP entities in this chunk
        entity_ids: IDs of STEP entities in this chunk
        topology_subgraph: Topology subgraph for entities in chunk
        embedding: Averaged embedding for chunk (if available)
        properties: Additional chunk properties
        bbox_center: Center of bounding box
        bbox_volume: Volume of bounding box
    """

    # Entity information
    entity_types: list[str] = Field(default_factory=list)
    entity_ids: list[int] = Field(default_factory=list)

    # Topology
    topology_subgraph: Optional[dict[str, Any]] = None

    # Embeddings
    embedding: Optional[list[float]] = None

    # Properties
    properties: dict[str, Any] = Field(default_factory=dict)

    # Geometric properties
    bbox_center: Optional[list[float]] = None
    bbox_volume: Optional[float] = None


class CADChunk(BaseModel):
    """Single chunk of a CAD document.

    A chunk represents a semantically or topologically coherent piece
    of a CAD document, suitable for RAG systems.

    Attributes:
        text: Text representation of this chunk
        meta: Metadata about this chunk
        chunk_id: Unique identifier for this chunk
        doc_name: Name of source document
        items: List of CADItem objects in this chunk
        metadata: Dictionary metadata for this chunk
    """

    text: str
    meta: CADChunkMeta
    chunk_id: Optional[str] = None
    doc_name: Optional[str] = None
    items: list = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class BaseCADChunker(ABC):
    """Base class for CAD document chunkers.

    Chunkers break down CADlingDocument into smaller pieces suitable for
    RAG systems. This is similar to docling's chunkers but adapted for
    CAD-specific requirements:
    - Topology-aware chunking (respect entity relationships)
    - Embedding-aware chunking (use neural embeddings)
    - Property-aware chunking (group by geometric properties)

    Subclasses must implement:
    - chunk(doc): Generate chunks from document

    Attributes:
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of overlapping tokens between chunks

    Example:
        chunker = MyChunker(max_tokens=512)
        for chunk in chunker.chunk(doc):
            print(chunk.text)
            print(chunk.meta.entity_types)
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        vocab_size: int = 50000,
    ):
        """Initialize chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            vocab_size: Vocabulary size for tokenizer
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = STEPTokenizer(vocab_size=vocab_size)
        _log.debug(
            f"Initialized {self.__class__.__name__} "
            f"(max_tokens={max_tokens}, overlap={overlap_tokens})"
        )

    @abstractmethod
    def chunk(self, doc: CADlingDocument) -> Iterator[CADChunk]:
        """Generate chunks from a CAD document.

        Args:
            doc: CADlingDocument to chunk

        Yields:
            CADChunk objects

        Example:
            def chunk(self, doc):
                for item in doc.items:
                    chunk = CADChunk(
                        text=item.text,
                        meta=CADChunkMeta(entity_types=[item.item_type])
                    )
                    yield chunk
        """
        pass

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using ll_stepnet tokenizer.

        Args:
            text: Text to count tokens in

        Returns:
            Token count
        """
        token_ids = self.tokenizer.encode(text)
        return len(token_ids)

    def _extract_topology_subgraph(
        self,
        entity_ids: list[int],
        full_topology: Optional[TopologyGraph],
    ) -> Optional[dict[str, Any]]:
        """Extract topology subgraph for given entities.

        Args:
            entity_ids: Entity IDs to include
            full_topology: Full topology graph from document

        Returns:
            Topology subgraph as dictionary, or None
        """
        if not full_topology:
            return None

        # Build subgraph containing only specified entities
        subgraph = {
            "num_nodes": len(entity_ids),
            "num_edges": 0,
            "adjacency_list": {},
        }

        entity_set = set(entity_ids)

        # Extract edges within subgraph
        for from_node in entity_ids:
            neighbors = full_topology.get_neighbors(from_node)
            # Only include edges to other nodes in subgraph
            subgraph_neighbors = [n for n in neighbors if n in entity_set]
            if subgraph_neighbors:
                subgraph["adjacency_list"][from_node] = subgraph_neighbors
                subgraph["num_edges"] += len(subgraph_neighbors)

        return subgraph
