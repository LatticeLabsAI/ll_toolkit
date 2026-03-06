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

from cadling.chunker.tokenizer.tokenizer import (
    CADTokenizer,
    SimpleTokenizer,
    get_tokenizer,
)

try:
    from stepnet import STEPTokenizer
    _has_stepnet = True
except ImportError:
    STEPTokenizer = None
    _has_stepnet = False

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
        token_count: Approximate token count for this chunk
    """

    text: str
    meta: CADChunkMeta
    chunk_id: Optional[str] = None
    doc_name: Optional[str] = None
    items: list = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    token_count: int = 0


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
        tokenizer_type: str = "auto",
        tokenizer_model: str | None = None,
    ):
        """Initialize chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            vocab_size: Vocabulary size for tokenizer
            tokenizer_type: Tokenizer type ("auto", "simple", "gpt", "huggingface").
                "auto" tries STEPTokenizer first, then falls back to SimpleTokenizer.
            tokenizer_model: Model name for gpt/huggingface tokenizers.
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = self._create_tokenizer(tokenizer_type, tokenizer_model, vocab_size)
        _log.debug(
            f"Initialized {self.__class__.__name__} "
            f"(max_tokens={max_tokens}, overlap={overlap_tokens}, "
            f"tokenizer={self.tokenizer.__class__.__name__})"
        )

    @staticmethod
    def _create_tokenizer(
        tokenizer_type: str, tokenizer_model: str | None, vocab_size: int
    ) -> Any:
        """Create the appropriate tokenizer instance.

        Tries STEPTokenizer first when tokenizer_type is "auto", falling back
        to SimpleTokenizer. For explicit types ("simple", "gpt", "huggingface"),
        delegates to the get_tokenizer factory.

        Args:
            tokenizer_type: Tokenizer selection strategy.
            tokenizer_model: Model name for gpt/huggingface tokenizers.
            vocab_size: Vocabulary size for STEPTokenizer.

        Returns:
            A tokenizer instance (STEPTokenizer or CADTokenizer subclass).
        """
        if tokenizer_type == "auto":
            if _has_stepnet and STEPTokenizer is not None:
                try:
                    return STEPTokenizer(vocab_size=vocab_size)
                except Exception as exc:
                    _log.debug("STEPTokenizer init failed (%s), using SimpleTokenizer", exc)
            return SimpleTokenizer()

        # Explicit tokenizer type requested
        return get_tokenizer(tokenizer_type, model=tokenizer_model)

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
        """Count tokens in text using the configured tokenizer.

        Supports both STEPTokenizer (encode -> list of IDs) and
        CADTokenizer subclasses (count_tokens method).

        Args:
            text: Text to count tokens in

        Returns:
            Token count
        """
        if isinstance(self.tokenizer, CADTokenizer):
            return self.tokenizer.count_tokens(text)
        # STEPTokenizer path
        token_ids = self.tokenizer.encode(text)
        return len(token_ids)

    def _item_to_text(self, item: Any) -> str:
        """Convert a CAD item/entity to text representation.

        Shared utility used by all chunker subclasses. Handles multiple
        item types with format-specific logic for STEP entities and
        generic CADItem fallbacks.

        Args:
            item: CAD item or entity to convert.

        Returns:
            Text representation of the item.
        """
        from cadling.datamodel.step import STEPEntityItem

        # Handle STEPEntityItem specifically
        if isinstance(item, STEPEntityItem):
            parts = [f"#{item.entity_id} {item.entity_type}"]
            if item.text:
                parts.append(item.text)
            return "\n".join(parts)

        # Try raw_line attribute (e.g. parsed STEP lines)
        if hasattr(item, "raw_line") and item.raw_line:
            return str(item.raw_line)

        # Build from entity_id and type attributes
        parts: list[str] = []
        entity_id = getattr(item, "entity_id", getattr(item, "id", None))
        entity_type = getattr(item, "type", getattr(item, "entity_type", None))

        if entity_id is not None:
            parts.append(f"#{entity_id}")
        if entity_type is not None:
            parts.append(f"={entity_type}")

        # Add parameters if available
        params = getattr(item, "parameters", getattr(item, "params", None))
        if params:
            if isinstance(params, dict):
                param_str = ",".join(f"{k}={v}" for k, v in params.items())
            elif isinstance(params, (list, tuple)):
                param_str = ",".join(str(p) for p in params)
            else:
                param_str = str(params)
            parts.append(f"({param_str})")

        if parts:
            return "".join(parts) + ";"

        # Try CADItem standard attributes (label + text + properties + bbox)
        if hasattr(item, "label") and item.label is not None:
            label_parts = [f"[{item.label.text}]"]

            if hasattr(item, "text") and item.text:
                label_parts.append(item.text)

            # Add properties (excluding large objects)
            if hasattr(item, "properties") and item.properties:
                prop_lines = []
                for key, value in item.properties.items():
                    if key in ["embedding", "children"]:
                        continue
                    prop_lines.append(f"{key}: {value}")
                if prop_lines:
                    label_parts.append("Properties: " + ", ".join(prop_lines))

            # Add bounding box info
            if hasattr(item, "bbox") and item.bbox is not None:
                center = item.bbox.center
                size = item.bbox.size
                label_parts.append(
                    f"BBox: center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), "
                    f"size=({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})"
                )

            return "\n".join(label_parts)

        # Fallback to string representation
        return str(item)

    def _get_overlap_items(self, items: list) -> list:
        """Get items for overlap with next chunk.

        Takes items from the end of the current chunk such that their
        total token count does not exceed ``self.overlap_tokens``.

        Args:
            items: Current chunk items

        Returns:
            Items to include in next chunk for overlap
        """
        if not items or self.overlap_tokens <= 0:
            return []

        overlap_items: list = []
        overlap_tokens_count = 0

        for item in reversed(items):
            item_tokens = self._count_tokens(self._item_to_text(item))
            if overlap_tokens_count + item_tokens > self.overlap_tokens:
                break
            overlap_items.insert(0, item)
            overlap_tokens_count += item_tokens

        return overlap_items

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
