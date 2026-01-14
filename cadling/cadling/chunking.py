"""Chunking strategies for CAD documents.

This module provides various chunking strategies for breaking down
CADlingDocument objects into smaller chunks suitable for RAG systems.

Classes:
    SequentialChunker: Sequential chunking with overlap support
    EntityTypeChunker: Chunk by entity type
    SpatialChunker: Spatial chunking using octree subdivision
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Iterator, List
from collections import defaultdict

from cadling.chunker.base_chunker import BaseCADChunker, CADChunk, CADChunkMeta

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADlingDocument, CADItem, BoundingBox3D

_log = logging.getLogger(__name__)


class SequentialChunker(BaseCADChunker):
    """Sequential chunker with overlap support.

    Chunks documents sequentially, grouping consecutive items together
    with optional overlap between chunks.

    Args:
        chunk_size: Number of items per chunk
        chunk_overlap: Number of items to overlap between chunks
        max_tokens: Maximum tokens per chunk (inherited)
        overlap_tokens: Token-level overlap (inherited)
    """

    def __init__(
        self,
        chunk_size: int = 10,
        chunk_overlap: int = 0,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
    ):
        """Initialize sequential chunker.

        Args:
            chunk_size: Number of items per chunk
            chunk_overlap: Number of items to overlap between chunks
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Token-level overlap
        """
        super().__init__(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        _log.debug(
            f"Initialized SequentialChunker "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
        )

    def chunk(self, doc: CADlingDocument) -> List[CADChunk]:
        """Generate sequential chunks from document.

        Args:
            doc: CADlingDocument to chunk

        Returns:
            List of CADChunk objects
        """
        chunks = []
        items = doc.items

        if not items:
            _log.warning(f"Document {doc.name} has no items to chunk")
            return chunks

        # Calculate step size (accounting for overlap)
        step = max(1, self.chunk_size - self.chunk_overlap)

        # Create chunks
        for i in range(0, len(items), step):
            chunk_items = items[i:i + self.chunk_size]

            if not chunk_items:
                continue

            # Skip if this would be a tiny chunk that's already covered by overlap
            # (avoid creating redundant final chunks)
            if i > 0 and len(chunk_items) <= self.chunk_overlap:
                continue

            # Generate chunk ID
            chunk_id = f"chunk_{i // step}_{uuid.uuid4().hex[:8]}"

            # Generate text representation
            text_lines = [
                f"CAD Chunk: {chunk_id}",
                f"Document: {doc.name}",
                f"Items: {len(chunk_items)}",
                "",
            ]

            for idx, item in enumerate(chunk_items, 1):
                item_text = f"Item {idx}: {item.label.text if hasattr(item.label, 'text') else item.label}"
                if item.text:
                    item_text += f"\n  {item.text}"
                text_lines.append(item_text)

            text = "\n".join(text_lines)

            # Create metadata
            metadata = {
                "chunk_type": "sequential",
                "chunk_id": chunk_id,
                "start_index": i,
                "end_index": min(i + self.chunk_size, len(items)),
                "num_items": len(chunk_items),
            }

            # Create chunk meta
            meta = CADChunkMeta(
                entity_types=[item.item_type for item in chunk_items],
                properties=metadata.copy(),
            )

            # Create chunk
            chunk = CADChunk(
                text=text,
                meta=meta,
                chunk_id=chunk_id,
                doc_name=doc.name,
                items=chunk_items,
                metadata=metadata,
            )

            chunks.append(chunk)
            _log.debug(f"Created chunk {chunk_id} with {len(chunk_items)} items")

        return chunks


class EntityTypeChunker(BaseCADChunker):
    """Entity type chunker.

    Groups items by their entity type, creating one chunk per type.

    Args:
        chunk_size: Maximum items per chunk (splits large type groups)
        max_tokens: Maximum tokens per chunk (inherited)
    """

    def __init__(
        self,
        chunk_size: int = 100,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
    ):
        """Initialize entity type chunker.

        Args:
            chunk_size: Maximum items per chunk
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Token-level overlap
        """
        super().__init__(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        self.chunk_size = chunk_size
        _log.debug(f"Initialized EntityTypeChunker (chunk_size={chunk_size})")

    def chunk(self, doc: CADlingDocument) -> List[CADChunk]:
        """Generate chunks grouped by entity type.

        Args:
            doc: CADlingDocument to chunk

        Returns:
            List of CADChunk objects
        """
        chunks = []

        if not doc.items:
            _log.warning(f"Document {doc.name} has no items to chunk")
            return chunks

        # Group items by type
        items_by_type = defaultdict(list)
        for item in doc.items:
            # Get entity type from item
            entity_type = getattr(item, 'entity_type', item.item_type)
            items_by_type[entity_type].append(item)

        # Create chunks for each type
        for entity_type, type_items in items_by_type.items():
            # Split large groups into multiple chunks
            for i in range(0, len(type_items), self.chunk_size):
                chunk_items = type_items[i:i + self.chunk_size]

                # Generate chunk ID
                chunk_id = f"chunk_{entity_type}_{i // self.chunk_size}_{uuid.uuid4().hex[:8]}"

                # Generate text representation
                text_lines = [
                    f"CAD Chunk: {chunk_id}",
                    f"Document: {doc.name}",
                    f"Entity Type: {entity_type}",
                    f"Items: {len(chunk_items)}",
                    "",
                ]

                for idx, item in enumerate(chunk_items, 1):
                    item_text = f"Item {idx}: {item.label.text if hasattr(item.label, 'text') else item.label}"
                    if item.text:
                        item_text += f"\n  {item.text}"
                    text_lines.append(item_text)

                text = "\n".join(text_lines)

                # Create metadata
                metadata = {
                    "chunk_type": "entity_type",
                    "chunk_id": chunk_id,
                    "entity_type": entity_type,
                    "num_items": len(chunk_items),
                }

                # Create chunk meta
                meta = CADChunkMeta(
                    entity_types=[entity_type],
                    properties=metadata.copy(),
                )

                # Create chunk
                chunk = CADChunk(
                    text=text,
                    meta=meta,
                    chunk_id=chunk_id,
                    doc_name=doc.name,
                    items=chunk_items,
                    metadata=metadata,
                )

                chunks.append(chunk)
                _log.debug(
                    f"Created chunk {chunk_id} for type {entity_type} "
                    f"with {len(chunk_items)} items"
                )

        return chunks


class SpatialChunker(BaseCADChunker):
    """Spatial chunker using octree subdivision.

    Subdivides space using an octree-like structure and groups items
    based on their spatial location (bounding boxes).

    Args:
        chunk_size: Target items per chunk (triggers subdivision)
        max_depth: Maximum subdivision depth
        max_tokens: Maximum tokens per chunk (inherited)
    """

    def __init__(
        self,
        chunk_size: int = 10,
        max_depth: int = 3,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
    ):
        """Initialize spatial chunker.

        Args:
            chunk_size: Target items per chunk
            max_depth: Maximum subdivision depth
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Token-level overlap
        """
        super().__init__(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        self.chunk_size = chunk_size
        self.max_depth = max_depth
        _log.debug(
            f"Initialized SpatialChunker "
            f"(chunk_size={chunk_size}, max_depth={max_depth})"
        )

    def chunk(self, doc: CADlingDocument) -> List[CADChunk]:
        """Generate spatial chunks from document.

        Args:
            doc: CADlingDocument to chunk

        Returns:
            List of CADChunk objects
        """
        chunks = []

        if not doc.items:
            _log.warning(f"Document {doc.name} has no items to chunk")
            return chunks

        # Get document bounding box
        doc_bbox = doc.bounding_box if hasattr(doc, 'bounding_box') else None

        if not doc_bbox:
            # Fall back to sequential chunking
            _log.warning(
                f"Document {doc.name} has no bounding box, "
                "falling back to sequential chunking"
            )
            sequential = SequentialChunker(
                chunk_size=self.chunk_size,
                max_tokens=self.max_tokens
            )
            return sequential.chunk(doc)

        # Filter items with bounding boxes
        items_with_bbox = [item for item in doc.items if hasattr(item, 'bbox') and item.bbox]

        if not items_with_bbox:
            _log.warning(f"No items with bounding boxes in {doc.name}")
            return chunks

        # Recursively subdivide space
        self._subdivide_space(
            items=items_with_bbox,
            bbox=doc_bbox,
            depth=0,
            chunks=chunks,
            doc_name=doc.name,
            octant_path="root",
        )

        return chunks

    def _subdivide_space(
        self,
        items: List,
        bbox: BoundingBox3D,
        depth: int,
        chunks: List[CADChunk],
        doc_name: str,
        octant_path: str,
    ):
        """Recursively subdivide space and create chunks.

        Args:
            items: Items in this region
            bbox: Bounding box of this region
            depth: Current subdivision depth
            chunks: List to append chunks to
            doc_name: Document name
            octant_path: Path through octree (for metadata)
        """
        # Base case: create chunk if below threshold or max depth
        if len(items) <= self.chunk_size or depth >= self.max_depth:
            if not items:
                return

            # Generate chunk ID
            chunk_id = f"chunk_spatial_{octant_path}_{uuid.uuid4().hex[:8]}"

            # Generate text representation
            text_lines = [
                f"CAD Chunk: {chunk_id}",
                f"Document: {doc_name}",
                f"Spatial Region: {octant_path}",
                f"Items: {len(items)}",
                "",
            ]

            for idx, item in enumerate(items, 1):
                item_text = f"Item {idx}: {item.label.text if hasattr(item.label, 'text') else item.label}"
                if item.text:
                    item_text += f"\n  {item.text}"
                text_lines.append(item_text)

            text = "\n".join(text_lines)

            # Create metadata
            metadata = {
                "chunk_type": "spatial",
                "chunk_id": chunk_id,
                "octant": octant_path,
                "depth": depth,
                "num_items": len(items),
                "bbox_center": list(bbox.center),
                "bbox_volume": bbox.volume,
            }

            # Create chunk meta
            meta = CADChunkMeta(
                entity_types=[item.item_type for item in items],
                properties=metadata.copy(),
                bbox_center=list(bbox.center),
                bbox_volume=bbox.volume,
            )

            # Create chunk
            chunk = CADChunk(
                text=text,
                meta=meta,
                chunk_id=chunk_id,
                doc_name=doc_name,
                items=items,
                metadata=metadata,
            )

            chunks.append(chunk)
            _log.debug(
                f"Created spatial chunk {chunk_id} at {octant_path} "
                f"with {len(items)} items"
            )
            return

        # Subdivide into 8 octants
        center = bbox.center

        # Define 8 octants
        octants = []
        for i in range(8):
            # Octant indexing: i = 4*z + 2*y + x (where x,y,z are 0 or 1)
            x_bit = i % 2
            y_bit = (i // 2) % 2
            z_bit = (i // 4) % 2

            octant_bbox_dict = {
                'x_min': bbox.x_min if x_bit == 0 else center[0],
                'x_max': center[0] if x_bit == 0 else bbox.x_max,
                'y_min': bbox.y_min if y_bit == 0 else center[1],
                'y_max': center[1] if y_bit == 0 else bbox.y_max,
                'z_min': bbox.z_min if z_bit == 0 else center[2],
                'z_max': center[2] if z_bit == 0 else bbox.z_max,
            }

            # Import BoundingBox3D here to avoid circular import
            from cadling.datamodel.base_models import BoundingBox3D
            octant_bbox = BoundingBox3D(**octant_bbox_dict)
            octants.append((i, octant_bbox))

        # Distribute items into octants
        for octant_idx, octant_bbox in octants:
            octant_items = []

            for item in items:
                if self._bbox_intersects(item.bbox, octant_bbox):
                    octant_items.append(item)

            # Recursively process octant
            if octant_items:
                new_path = f"{octant_path}.{octant_idx}"
                self._subdivide_space(
                    items=octant_items,
                    bbox=octant_bbox,
                    depth=depth + 1,
                    chunks=chunks,
                    doc_name=doc_name,
                    octant_path=new_path,
                )

    def _bbox_intersects(self, bbox1: BoundingBox3D, bbox2: BoundingBox3D) -> bool:
        """Check if two bounding boxes intersect.

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box

        Returns:
            True if bounding boxes intersect
        """
        return not (
            bbox1.x_max < bbox2.x_min or bbox1.x_min > bbox2.x_max or
            bbox1.y_max < bbox2.y_min or bbox1.y_min > bbox2.y_max or
            bbox1.z_max < bbox2.z_min or bbox1.z_min > bbox2.z_max
        )


__all__ = [
    "SequentialChunker",
    "EntityTypeChunker",
    "SpatialChunker",
]
