"""Hierarchical chunker for CAD assemblies.

This module provides a hierarchical chunking strategy that:
- Respects assembly hierarchies (assembly -> part -> feature)
- Groups items by their hierarchical relationships
- Maintains parent-child context in chunks
- Handles both single parts and multi-part assemblies

Classes:
    CADHierarchicalChunker: Hierarchy-aware chunker for assemblies
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Optional

from cadling.chunker.base_chunker import BaseCADChunker, CADChunk, CADChunkMeta

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument
    from cadling.datamodel.step import STEPEntityItem
    from cadling.datamodel.stl import AssemblyItem

_log = logging.getLogger(__name__)


class CADHierarchicalChunker(BaseCADChunker):
    """Hierarchical chunker for CAD assemblies.

    This chunker creates chunks based on the hierarchical structure of
    CAD assemblies. It groups entities by their position in the hierarchy:
    1. Assembly-level chunks (top-level components)
    2. Part-level chunks (individual parts within assemblies)
    3. Feature-level chunks (features within parts)

    The chunker preserves parent-child relationships and includes
    context from parent levels in each chunk.

    Attributes:
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of overlapping tokens
        include_parent_context: Whether to include parent info in chunks
        max_depth: Maximum hierarchy depth to process

    Example:
        chunker = CADHierarchicalChunker(max_tokens=512)
        chunks = list(chunker.chunk(assembly_doc))

        for chunk in chunks:
            print(f"Level: {chunk.meta.properties['hierarchy_level']}")
            print(f"Parent: {chunk.meta.properties.get('parent_id')}")
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        include_parent_context: bool = True,
        max_depth: int = 5,
    ):
        """Initialize hierarchical chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens
            include_parent_context: Include parent context in chunks
            max_depth: Maximum hierarchy depth to process
        """
        super().__init__(max_tokens, overlap_tokens)
        self.include_parent_context = include_parent_context
        self.max_depth = max_depth

    def chunk(self, doc: CADlingDocument) -> Iterator[CADChunk]:
        """Generate hierarchy-aware chunks from CAD document.

        Args:
            doc: CADlingDocument to chunk

        Yields:
            CADChunk objects organized by hierarchy
        """
        _log.info(
            f"Hierarchical chunking document '{doc.name}' with {len(doc.items)} items"
        )

        # Build hierarchy tree from items
        hierarchy_tree = self._build_hierarchy_tree(doc)

        # Generate chunks from hierarchy
        chunk_count = 0
        for chunk in self._chunk_hierarchy(hierarchy_tree, doc):
            yield chunk
            chunk_count += 1

        _log.info(f"Generated {chunk_count} hierarchical chunks from '{doc.name}'")

    def _build_hierarchy_tree(
        self, doc: CADlingDocument
    ) -> dict[str, list[CADItem]]:
        """Build hierarchical tree from document items.

        Args:
            doc: CADlingDocument

        Returns:
            Dictionary mapping parent IDs to child items
        """
        from cadling.datamodel.stl import AssemblyItem

        hierarchy: dict[str, list[CADItem]] = {"root": []}

        for item in doc.items:
            # Check if item is an assembly with children
            if isinstance(item, AssemblyItem) and item.children:
                hierarchy[item.label.text] = list(item.children)
            else:
                # Add to root level
                hierarchy["root"].append(item)

        # Also check topology for hierarchical relationships
        if doc.topology:
            hierarchy = self._augment_with_topology(hierarchy, doc)

        return hierarchy

    def _augment_with_topology(
        self, hierarchy: dict[str, list[CADItem]], doc: CADlingDocument
    ) -> dict[str, list[CADItem]]:
        """Augment hierarchy tree with topology relationships.

        Args:
            hierarchy: Current hierarchy tree
            doc: CADlingDocument

        Returns:
            Updated hierarchy tree
        """
        from cadling.datamodel.step import STEPEntityItem

        # Build entity ID to item mapping
        entity_map: dict[int, CADItem] = {}
        for item in doc.items:
            if isinstance(item, STEPEntityItem):
                entity_map[item.entity_id] = item

        # Use topology to find parent-child relationships
        if doc.topology:
            for entity_id, item in entity_map.items():
                # Get neighbors from topology
                neighbors = doc.topology.get_neighbors(entity_id)

                # Heuristic: if entity has many outgoing edges, it may be a parent
                if len(neighbors) > 3:  # Threshold for "parent-like" entity
                    parent_key = f"entity_{entity_id}"
                    if parent_key not in hierarchy:
                        hierarchy[parent_key] = []

                    # Add neighbors as children
                    for neighbor_id in neighbors:
                        if neighbor_id in entity_map:
                            child_item = entity_map[neighbor_id]
                            if child_item not in hierarchy[parent_key]:
                                hierarchy[parent_key].append(child_item)

        return hierarchy

    def _chunk_hierarchy(
        self,
        hierarchy_tree: dict[str, list[CADItem]],
        doc: CADlingDocument,
        parent_id: Optional[str] = None,
        level: int = 0,
    ) -> Iterator[CADChunk]:
        """Recursively chunk hierarchy tree.

        Args:
            hierarchy_tree: Hierarchy tree
            doc: Parent document
            parent_id: ID of parent node
            level: Current hierarchy level

        Yields:
            CADChunk objects
        """
        if level > self.max_depth:
            return

        # Process root level
        if parent_id is None:
            parent_id = "root"

        if parent_id not in hierarchy_tree:
            return

        items = hierarchy_tree[parent_id]

        # Group items into chunks based on token limits
        current_chunk_items: list[CADItem] = []
        current_tokens = 0
        chunk_index = 0

        for item in items:
            item_text = self._item_to_text(item)
            item_tokens = self._count_tokens(item_text)

            # Check if adding this item exceeds max_tokens
            if current_tokens + item_tokens > self.max_tokens and current_chunk_items:
                # Yield current chunk
                chunk = self._create_hierarchical_chunk(
                    current_chunk_items,
                    doc,
                    parent_id=parent_id,
                    level=level,
                    chunk_index=chunk_index,
                )
                yield chunk
                chunk_index += 1

                # Start new chunk with overlap
                current_chunk_items = self._get_overlap_items(current_chunk_items)
                current_tokens = sum(
                    self._count_tokens(self._item_to_text(i))
                    for i in current_chunk_items
                )

            # Add item to current chunk
            current_chunk_items.append(item)
            current_tokens += item_tokens

        # Yield final chunk at this level
        if current_chunk_items:
            chunk = self._create_hierarchical_chunk(
                current_chunk_items,
                doc,
                parent_id=parent_id,
                level=level,
                chunk_index=chunk_index,
            )
            yield chunk

        # Recursively process children
        for item in items:
            item_key = item.label.text
            if item_key in hierarchy_tree:
                yield from self._chunk_hierarchy(
                    hierarchy_tree, doc, parent_id=item_key, level=level + 1
                )

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

        # Add properties (excluding large objects)
        if item.properties:
            prop_lines = []
            for key, value in item.properties.items():
                if key in ["embedding", "children"]:
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

    def _create_hierarchical_chunk(
        self,
        items: list[CADItem],
        doc: CADlingDocument,
        parent_id: str,
        level: int,
        chunk_index: int,
    ) -> CADChunk:
        """Create a hierarchical chunk from items.

        Args:
            items: List of CAD items
            doc: Parent document
            parent_id: Parent node ID
            level: Hierarchy level
            chunk_index: Index within this level

        Returns:
            CADChunk object with hierarchy metadata
        """
        from cadling.datamodel.step import STEPEntityItem

        # Add parent context if enabled
        text_parts = []
        if self.include_parent_context and parent_id != "root":
            text_parts.append(f"[Parent: {parent_id}]")
            text_parts.append(f"[Hierarchy Level: {level}]")
            text_parts.append("")

        # Add item text
        text_parts.extend(self._item_to_text(item) for item in items)
        text = "\n\n".join(text_parts)

        # Extract metadata
        entity_types = []
        entity_ids = []

        for item in items:
            if isinstance(item, STEPEntityItem):
                entity_types.append(item.entity_type)
                entity_ids.append(item.entity_id)

        # Extract topology subgraph
        topology_subgraph = self._extract_topology_subgraph(entity_ids, doc.topology)

        # Compute geometric properties
        bbox_center, bbox_volume = self._compute_chunk_geometry(items)

        # Create metadata with hierarchy info
        meta = CADChunkMeta(
            entity_types=entity_types,
            entity_ids=entity_ids,
            topology_subgraph=topology_subgraph,
            bbox_center=bbox_center,
            bbox_volume=bbox_volume,
            properties={
                "num_items": len(items),
                "num_entities": len(entity_ids),
                "hierarchy_level": level,
                "parent_id": parent_id,
                "chunk_index": chunk_index,
            },
        )

        chunk_id = f"{doc.name}_L{level}_{parent_id}_chunk_{chunk_index}"

        return CADChunk(
            text=text,
            meta=meta,
            chunk_id=chunk_id,
            doc_name=doc.name,
        )

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
