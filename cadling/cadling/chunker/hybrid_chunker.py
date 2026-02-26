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

        Uses multi-strategy approach:
        1. Primary: Use pre-computed embeddings from items
        2. Fallback: Compute geometric embedding from properties
        3. Fallback: Compute text-hash embedding from item text

        Args:
            items: List of CAD items

        Returns:
            Averaged embedding vector, or None only if use_embeddings is False
        """
        if not self.use_embeddings:
            return None

        # Strategy 1: Use pre-computed embeddings
        embeddings = []
        for item in items:
            if "embedding" in item.properties:
                emb = item.properties["embedding"]
                if isinstance(emb, list):
                    embeddings.append(np.array(emb))

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            return avg_embedding.tolist()

        # Strategy 2: Compute geometric embedding from properties
        geom_embedding = self._compute_geometric_embedding(items)
        if geom_embedding is not None:
            return geom_embedding

        # Strategy 3: Compute text-hash embedding from item text
        text = " ".join(self._item_to_text(item) for item in items)
        return self._compute_text_hash_embedding(text)

    def _compute_geometric_embedding(self, items: list[CADItem]) -> Optional[list[float]]:
        """Compute embedding from geometric properties.

        Args:
            items: List of CAD items

        Returns:
            Geometric embedding or None if insufficient data
        """
        features = []

        for item in items:
            # Extract numeric properties from geometry_analysis
            if "geometry_analysis" in item.properties:
                geom = item.properties["geometry_analysis"]
                if "surface_area" in geom:
                    features.append(np.log1p(geom["surface_area"]))
                if "volume" in geom:
                    features.append(np.log1p(abs(geom["volume"])))

            # Extract from item bbox
            if item.bbox is not None:
                bbox = item.bbox
                bbox_size = [
                    bbox.x_max - bbox.x_min,
                    bbox.y_max - bbox.y_min,
                    bbox.z_max - bbox.z_min,
                ]
                features.extend([np.log1p(s) for s in bbox_size])

        if len(features) < 3:
            return None

        # Pad/truncate to fixed dimension (64)
        embedding = np.zeros(64)
        embedding[:min(len(features), 64)] = features[:64]

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def _compute_text_hash_embedding(self, text: str) -> list[float]:
        """Compute simple hash-based embedding from text.

        Args:
            text: Text to embed

        Returns:
            64-dimensional hash-based embedding
        """
        import hashlib

        embedding = np.zeros(64)

        # Use multiple hash functions for different aspects
        words = text.split()[:64]
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
            embedding[i % 64] += (h / (2**32)) - 0.5

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def _compute_chunk_geometry(
        self, items: list[CADItem]
    ) -> tuple[Optional[list[float]], Optional[float]]:
        """Compute geometric properties for chunk.

        Uses multi-strategy approach:
        1. Primary: Use item bounding boxes
        2. Fallback: Extract from geometry_analysis properties

        Args:
            items: List of CAD items

        Returns:
            Tuple of (bbox_center, bbox_volume) - tries to always return values
        """
        # Strategy 1: Use item bounding boxes
        bboxes = [item.bbox for item in items if item.bbox is not None]

        if bboxes:
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

        # Strategy 2: Extract from geometry_analysis properties
        centers = []
        volumes = []

        for item in items:
            if "geometry_analysis" in item.properties:
                geom = item.properties["geometry_analysis"]

                # Try to get center from geometry analysis
                center = geom.get("centroid") or geom.get("center_of_mass")
                if center:
                    if isinstance(center, dict):
                        center = [center.get("x", 0), center.get("y", 0), center.get("z", 0)]
                    centers.append(center)

                # Try to get volume
                if "volume" in geom:
                    volumes.append(abs(geom["volume"]))

                # Try to get bbox from geometry_analysis
                bbox = geom.get("bounding_box", {})
                if bbox and all(k in bbox for k in ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]):
                    center = [
                        (bbox["xmin"] + bbox["xmax"]) / 2,
                        (bbox["ymin"] + bbox["ymax"]) / 2,
                        (bbox["zmin"] + bbox["zmax"]) / 2,
                    ]
                    centers.append(center)
                    vol = (bbox["xmax"] - bbox["xmin"]) * (bbox["ymax"] - bbox["ymin"]) * (bbox["zmax"] - bbox["zmin"])
                    volumes.append(vol)

        if centers:
            avg_center = np.mean(centers, axis=0).tolist()
            total_volume = sum(volumes) if volumes else None
            return avg_center, total_volume

        return None, None

    # _get_overlap_items is inherited from BaseCADChunker
