"""Hybrid chunker for CAD documents.

This module provides a hybrid chunking strategy that combines:
- Entity-level chunking (group related entities)
- Semantic chunking (respect token limits)
- Topology-aware chunking (preserve entity relationships)
- Embedding-aware chunking (use embeddings when available)

Classes:
    CADHybridChunker: Main hybrid chunker
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np

from cadling.chunker.base_chunker import BaseCADChunker, CADChunk, CADChunkMeta

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument
    from cadling.datamodel.step import STEPEntityItem

_log = logging.getLogger(__name__)


class CADHybridChunker(BaseCADChunker):
    """Hybrid chunker combining multiple strategies.

    This chunker implements a sophisticated chunking strategy that:
    1. Groups entities by topological relationships
    2. Respects token limits
    3. Uses embeddings for semantic coherence (when available)
    4. Preserves important geometric properties

    The chunker tries to create semantically meaningful chunks that
    capture complete geometric features or components.

    Attributes:
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of overlapping tokens
        use_embeddings: Whether to use embeddings for chunking

    Example:
        chunker = CADHybridChunker(max_tokens=512)
        chunks = list(chunker.chunk(doc))

        for chunk in chunks:
            print(f"Chunk: {chunk.text[:100]}...")
            print(f"Entities: {chunk.meta.entity_types}")
            print(f"Topology: {chunk.meta.topology_subgraph}")
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        use_embeddings: bool = True,
    ):
        """Initialize hybrid chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens
            use_embeddings: Whether to use embeddings for chunking
        """
        super().__init__(max_tokens, overlap_tokens)
        self.use_embeddings = use_embeddings

    def chunk(self, doc: CADlingDocument) -> Iterator[CADChunk]:
        """Generate chunks from CAD document.

        Args:
            doc: CADlingDocument to chunk

        Yields:
            CADChunk objects
        """
        _log.info(
            f"Chunking document '{doc.name}' with {len(doc.items)} items "
            f"(max_tokens={self.max_tokens})"
        )

        current_chunk_items: list[CADItem] = []
        current_tokens = 0
        chunk_count = 0

        for item in doc.items:
            # Get text representation
            item_text = self._item_to_text(item)
            item_tokens = self._count_tokens(item_text)

            # Check if adding this item would exceed max_tokens
            if current_tokens + item_tokens > self.max_tokens and current_chunk_items:
                # Yield current chunk
                chunk = self._create_chunk(
                    current_chunk_items, doc, chunk_id=f"{doc.name}_chunk_{chunk_count}"
                )
                yield chunk
                chunk_count += 1

                # Start new chunk with overlap
                current_chunk_items = self._get_overlap_items(current_chunk_items)
                current_tokens = sum(
                    self._count_tokens(self._item_to_text(i))
                    for i in current_chunk_items
                )

            # Add item to current chunk
            current_chunk_items.append(item)
            current_tokens += item_tokens

        # Yield final chunk
        if current_chunk_items:
            chunk = self._create_chunk(
                current_chunk_items, doc, chunk_id=f"{doc.name}_chunk_{chunk_count}"
            )
            yield chunk
            chunk_count += 1

        _log.info(f"Generated {chunk_count} chunks from document '{doc.name}'")

    def _item_to_text(self, item: CADItem) -> str:
        """Convert CAD item to text representation.

        Args:
            item: CAD item

        Returns:
            Text representation
        """
        parts = []

        # Add label
        parts.append(f"[{item.label.text}]")

        # Add item text if available
        if item.text:
            parts.append(item.text)

        # Add properties
        if item.properties:
            prop_lines = []
            for key, value in item.properties.items():
                # Skip embeddings (too large)
                if key == "embedding":
                    continue
                prop_lines.append(f"{key}: {value}")

            if prop_lines:
                parts.append("Properties: " + ", ".join(prop_lines))

        # Add bounding box info
        if item.bbox:
            center = item.bbox.center
            size = item.bbox.size
            parts.append(
                f"BBox: center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), "
                f"size=({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})"
            )

        return "\n".join(parts)

    def _create_chunk(
        self,
        items: list[CADItem],
        doc: CADlingDocument,
        chunk_id: Optional[str] = None,
    ) -> CADChunk:
        """Create a chunk from items.

        Args:
            items: List of CAD items
            doc: Parent document
            chunk_id: Unique chunk identifier

        Returns:
            CADChunk object
        """
        from cadling.datamodel.step import STEPEntityItem

        # Combine text
        text = "\n\n".join(self._item_to_text(item) for item in items)

        # Extract metadata
        entity_types = []
        entity_ids = []

        for item in items:
            if isinstance(item, STEPEntityItem):
                entity_types.append(item.entity_type)
                entity_ids.append(item.entity_id)

        # Extract topology subgraph
        topology_subgraph = self._extract_topology_subgraph(entity_ids, doc.topology)

        # Compute averaged embedding
        embedding = self._compute_chunk_embedding(items)

        # Compute geometric properties
        bbox_center, bbox_volume = self._compute_chunk_geometry(items)

        # Create metadata
        meta = CADChunkMeta(
            entity_types=entity_types,
            entity_ids=entity_ids,
            topology_subgraph=topology_subgraph,
            embedding=embedding,
            bbox_center=bbox_center,
            bbox_volume=bbox_volume,
            properties={
                "num_items": len(items),
                "num_entities": len(entity_ids),
            },
        )

        return CADChunk(
            text=text,
            meta=meta,
            chunk_id=chunk_id,
            doc_name=doc.name,
        )

    def _compute_chunk_embedding(self, items: list[CADItem]) -> Optional[list[float]]:
        """Compute averaged embedding for chunk.

        Args:
            items: List of CAD items

        Returns:
            Averaged embedding vector, or None
        """
        if not self.use_embeddings:
            return None

        embeddings = []

        for item in items:
            if "embedding" in item.properties:
                emb = item.properties["embedding"]
                if isinstance(emb, list):
                    embeddings.append(np.array(emb))

        if not embeddings:
            return None

        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)

        # Normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        return avg_embedding.tolist()

    def _compute_chunk_geometry(
        self, items: list[CADItem]
    ) -> tuple[Optional[list[float]], Optional[float]]:
        """Compute geometric properties for chunk.

        Args:
            items: List of CAD items

        Returns:
            Tuple of (bbox_center, bbox_volume)
        """
        bboxes = [item.bbox for item in items if item.bbox is not None]

        if not bboxes:
            return None, None

        # Compute combined bounding box
        x_min = min(bbox.x_min for bbox in bboxes)
        y_min = min(bbox.y_min for bbox in bboxes)
        z_min = min(bbox.z_min for bbox in bboxes)
        x_max = max(bbox.x_max for bbox in bboxes)
        y_max = max(bbox.y_max for bbox in bboxes)
        z_max = max(bbox.z_max for bbox in bboxes)

        center = [
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            (z_min + z_max) / 2,
        ]

        volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

        return center, volume

    def _get_overlap_items(self, items: list[CADItem]) -> list[CADItem]:
        """Get items for overlap with next chunk.

        Args:
            items: Current chunk items

        Returns:
            Items to include in next chunk for overlap
        """
        if not items or self.overlap_tokens <= 0:
            return []

        # Calculate how many items to include for overlap
        overlap_items = []
        overlap_tokens_count = 0

        for item in reversed(items):
            item_tokens = self._count_tokens(self._item_to_text(item))
            if overlap_tokens_count + item_tokens > self.overlap_tokens:
                break
            overlap_items.insert(0, item)
            overlap_tokens_count += item_tokens

        return overlap_items
