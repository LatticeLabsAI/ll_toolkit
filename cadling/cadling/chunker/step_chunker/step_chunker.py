"""STEP-specific chunking strategies.

STEP files have entity-based structure with topology graphs, making them
ideal for entity-aware and topology-aware chunking.

Classes:
    STEPChunker: Main STEP chunker with configurable strategies
    EntityGroupChunker: Groups entities by type
    TopologyChunker: Groups entities by topological connectivity
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Iterator, Optional

from cadling.chunker.base_chunker import BaseCADChunker, CADChunk, CADChunkMeta
from cadling.datamodel.step import STEPDocument, STEPEntityItem

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADlingDocument

_log = logging.getLogger(__name__)


class STEPChunker(BaseCADChunker):
    """STEP-specific chunker.

    Provides entity-aware chunking for STEP files, with options for:
    - Entity type grouping
    - Topology-based grouping
    - Token-limited chunking

    Attributes:
        strategy: Chunking strategy ("entity_type", "topology", "hybrid")
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks
    """

    def __init__(
        self,
        strategy: str = "hybrid",
        max_tokens: int = 512,
        overlap_tokens: int = 50,
    ):
        """Initialize STEP chunker.

        Args:
            strategy: Chunking strategy
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap tokens
        """
        super().__init__(max_tokens, overlap_tokens)
        self.strategy = strategy

    def chunk(self, doc: CADlingDocument) -> Iterator[CADChunk]:
        """Generate chunks from STEP document.

        Args:
            doc: CADlingDocument (should be STEPDocument)

        Yields:
            CADChunk objects
        """
        if not isinstance(doc, STEPDocument):
            _log.warning(f"STEPChunker used on non-STEP document: {type(doc)}")
            # Fallback to basic chunking
            yield from self._basic_chunk(doc)
            return

        if self.strategy == "entity_type":
            yield from self._chunk_by_entity_type(doc)
        elif self.strategy == "topology":
            yield from self._chunk_by_topology(doc)
        else:  # hybrid
            yield from self._chunk_hybrid(doc)

    def _chunk_by_entity_type(self, doc: STEPDocument) -> Iterator[CADChunk]:
        """Chunk by grouping similar entity types.

        Args:
            doc: STEP document

        Yields:
            CADChunk objects
        """
        # Group entities by type
        entity_groups = defaultdict(list)

        for item in doc.items:
            if isinstance(item, STEPEntityItem):
                entity_groups[item.entity_type].append(item)

        _log.info(f"Grouped {len(doc.items)} entities into {len(entity_groups)} type groups")

        # Pre-compute token counts for all entities (avoid repeated computation in loop)
        entity_token_cache = {}
        for entity_type, entities in entity_groups.items():
            for entity in entities:
                entity_text = self._item_to_text(entity)
                entity_token_cache[entity.entity_id] = self._count_tokens(entity_text)

        # Create chunks for each group
        chunk_count = 0
        for entity_type, entities in entity_groups.items():
            # Further split if exceeds token limit
            current_group = []
            current_tokens = 0

            for entity in entities:
                entity_tokens = entity_token_cache[entity.entity_id]  # O(1) lookup

                if current_tokens + entity_tokens > self.max_tokens and current_group:
                    # Yield chunk
                    yield self._create_chunk_from_entities(
                        current_group,
                        doc,
                        chunk_id=f"{doc.name}_{entity_type}_{chunk_count}",
                    )
                    chunk_count += 1

                    # Start new group
                    current_group = []
                    current_tokens = 0

                current_group.append(entity)
                current_tokens += entity_tokens

            # Yield final chunk for this type
            if current_group:
                yield self._create_chunk_from_entities(
                    current_group,
                    doc,
                    chunk_id=f"{doc.name}_{entity_type}_{chunk_count}",
                )
                chunk_count += 1

    def _chunk_by_topology(self, doc: STEPDocument) -> Iterator[CADChunk]:
        """Chunk by topological connectivity.

        Groups entities that are topologically connected.

        Args:
            doc: STEP document

        Yields:
            CADChunk objects
        """
        if not doc.topology:
            _log.warning("No topology available, falling back to entity type chunking")
            yield from self._chunk_by_entity_type(doc)
            return

        # Find connected components in topology graph
        components = self._find_connected_components(doc)

        _log.info(f"Found {len(components)} connected components")

        # Create chunk for each component
        for i, component_ids in enumerate(components):
            # Get entities for this component
            component_entities = [
                item
                for item in doc.items
                if isinstance(item, STEPEntityItem) and item.entity_id in component_ids
            ]

            if component_entities:
                yield self._create_chunk_from_entities(
                    component_entities,
                    doc,
                    chunk_id=f"{doc.name}_topo_{i}",
                )

    def _chunk_hybrid(self, doc: STEPDocument) -> Iterator[CADChunk]:
        """Hybrid chunking: topology-aware + token-limited.

        Args:
            doc: STEP document

        Yields:
            CADChunk objects
        """
        # Start with topology-based components
        if doc.topology:
            components = self._find_connected_components(doc)
        else:
            # Fallback: treat each entity as its own component
            components = [[item.entity_id] for item in doc.items if isinstance(item, STEPEntityItem)]

        # Pre-compute token counts for all entities
        entity_token_cache = {}
        for item in doc.items:
            if isinstance(item, STEPEntityItem):
                entity_text = self._item_to_text(item)
                entity_token_cache[item.entity_id] = self._count_tokens(entity_text)

        chunk_count = 0

        for component_ids in components:
            # Get entities for this component
            component_ids_set = set(component_ids)  # O(1) lookup
            component_entities = [
                item
                for item in doc.items
                if isinstance(item, STEPEntityItem) and item.entity_id in component_ids_set
            ]

            # Split component if too large
            current_batch = []
            current_tokens = 0

            for entity in component_entities:
                entity_tokens = entity_token_cache[entity.entity_id]  # O(1) lookup

                if current_tokens + entity_tokens > self.max_tokens and current_batch:
                    # Yield chunk
                    yield self._create_chunk_from_entities(
                        current_batch,
                        doc,
                        chunk_id=f"{doc.name}_hybrid_{chunk_count}",
                    )
                    chunk_count += 1

                    # Start new batch with token-count-aware overlap
                    current_batch = self._get_overlap_items(current_batch)
                    current_tokens = sum(
                        entity_token_cache[item.entity_id] for item in current_batch
                    )

                current_batch.append(entity)
                current_tokens += entity_tokens

            # Yield final chunk
            if current_batch:
                yield self._create_chunk_from_entities(
                    current_batch,
                    doc,
                    chunk_id=f"{doc.name}_hybrid_{chunk_count}",
                )
                chunk_count += 1

    def _basic_chunk(self, doc: CADlingDocument) -> Iterator[CADChunk]:
        """Basic chunking fallback.

        Args:
            doc: CAD document

        Yields:
            CADChunk objects
        """
        current_items = []
        current_tokens = 0
        chunk_count = 0

        for item in doc.items:
            item_text = self._item_to_text(item)
            item_tokens = self._count_tokens(item_text)

            if current_tokens + item_tokens > self.max_tokens and current_items:
                yield self._create_chunk_from_items(
                    current_items,
                    doc,
                    chunk_id=f"{doc.name}_basic_{chunk_count}",
                )
                chunk_count += 1
                current_items = []
                current_tokens = 0

            current_items.append(item)
            current_tokens += item_tokens

        if current_items:
            yield self._create_chunk_from_items(
                current_items,
                doc,
                chunk_id=f"{doc.name}_basic_{chunk_count}",
            )

    def _find_connected_components(self, doc: STEPDocument) -> list[list[int]]:
        """Find connected components in topology graph.

        Args:
            doc: STEP document

        Returns:
            List of entity ID lists (one per component)
        """
        if not doc.topology:
            return []

        visited = set()
        components = []

        # Get all entity IDs
        entity_ids = [
            item.entity_id for item in doc.items if isinstance(item, STEPEntityItem)
        ]

        for entity_id in entity_ids:
            if entity_id in visited:
                continue

            # BFS to find connected component
            component = []
            queue = deque([entity_id])  # Use deque for O(1) popleft
            visited.add(entity_id)

            while queue:
                current_id = queue.popleft()  # O(1) instead of O(n)
                component.append(current_id)

                # Get neighbors
                neighbors = doc.topology.get_neighbors(current_id)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            components.append(component)

        return components

    def _create_chunk_from_entities(
        self,
        entities: list[STEPEntityItem],
        doc: STEPDocument,
        chunk_id: Optional[str] = None,
    ) -> CADChunk:
        """Create chunk from STEP entities.

        Args:
            entities: List of STEP entities
            doc: Parent document
            chunk_id: Chunk identifier

        Returns:
            CADChunk
        """
        # Combine text
        text = "\n\n".join(self._item_to_text(entity) for entity in entities)

        # Extract metadata
        entity_types = [e.entity_type for e in entities]
        entity_ids = [e.entity_id for e in entities]

        # Extract topology subgraph
        topology_subgraph = self._extract_topology_subgraph(entity_ids, doc.topology)

        meta = CADChunkMeta(
            entity_types=entity_types,
            entity_ids=entity_ids,
            topology_subgraph=topology_subgraph,
            properties={
                "num_entities": len(entities),
                "unique_types": len(set(entity_types)),
            },
        )

        return CADChunk(
            text=text,
            meta=meta,
            chunk_id=chunk_id,
            doc_name=doc.name,
        )

    def _create_chunk_from_items(self, items, doc, chunk_id=None) -> CADChunk:
        """Create chunk from generic items.

        Args:
            items: List of CAD items
            doc: Parent document
            chunk_id: Chunk identifier

        Returns:
            CADChunk
        """
        text = "\n\n".join(self._item_to_text(item) for item in items)

        meta = CADChunkMeta(
            properties={"num_items": len(items)},
        )

        return CADChunk(
            text=text,
            meta=meta,
            chunk_id=chunk_id,
            doc_name=doc.name,
        )


# Convenience aliases
def EntityGroupChunker(**kwargs) -> STEPChunker:
    """Create a STEPChunker with entity_type strategy."""
    return STEPChunker(strategy="entity_type", **kwargs)


def TopologyChunker(**kwargs) -> STEPChunker:
    """Create a STEPChunker with topology strategy."""
    return STEPChunker(strategy="topology", **kwargs)
