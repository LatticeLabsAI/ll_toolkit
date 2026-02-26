"""BRep-specific text-based chunking strategies.

BRep (Boundary Representation) files have hierarchical structure with
topological entities (vertices, edges, faces, shells, solids), making them
ideal for topology-aware and entity-based chunking.

Classes:
    BRepChunker: Main BRep chunker with configurable strategies
    EntityTypeChunker: Groups entities by type (vertex, edge, face, etc.)
    TopologyChunker: Groups entities by topological relationships
    HierarchyChunker: Groups entities by hierarchical structure
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Iterator, Optional, Set, List

from cadling.chunker.base_chunker import BaseCADChunker, CADChunk, CADChunkMeta

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADlingDocument

_log = logging.getLogger(__name__)


class BRepEntity:
    """Represents a BRep entity.

    Attributes:
        entity_id: Entity identifier
        entity_type: Type of entity (VERTEX, EDGE, FACE, SHELL, SOLID, etc.)
        text: Text representation of entity
        references: Entity IDs referenced by this entity
        properties: Entity properties
    """

    def __init__(
        self,
        entity_id: int,
        entity_type: str,
        text: str,
        references: Optional[List[int]] = None,
    ):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.text = text
        self.references = references or []
        self.properties: dict = {}


class BRepChunker(BaseCADChunker):
    """BRep-specific text-based chunker.

    Provides entity-aware chunking for BRep files, with options for:
    - Entity type grouping
    - Topology-based grouping
    - Hierarchical grouping
    - Token-limited chunking

    Attributes:
        strategy: Chunking strategy ("entity_type", "topology", "hierarchy", "hybrid")
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks
    """

    def __init__(
        self,
        strategy: str = "hybrid",
        max_tokens: int = 512,
        overlap_tokens: int = 50,
    ):
        """Initialize BRep chunker.

        Args:
            strategy: Chunking strategy
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap tokens
        """
        super().__init__(max_tokens, overlap_tokens)
        self.strategy = strategy

    def chunk(self, doc: CADlingDocument) -> Iterator[CADChunk]:
        """Generate chunks from BRep document.

        Args:
            doc: CADlingDocument (should contain BRep data)

        Yields:
            CADChunk objects
        """
        # Extract BRep entities from document
        entities = self._extract_brep_entities(doc)

        if not entities:
            _log.warning("No BRep entities found in document")
            return

        _log.info(f"Chunking {len(entities)} BRep entities using strategy: {self.strategy}")

        if self.strategy == "entity_type":
            yield from self._chunk_by_entity_type(entities, doc)
        elif self.strategy == "topology":
            yield from self._chunk_by_topology(entities, doc)
        elif self.strategy == "hierarchy":
            yield from self._chunk_by_hierarchy(entities, doc)
        else:  # hybrid
            yield from self._chunk_hybrid(entities, doc)

    def _extract_brep_entities(self, doc: CADlingDocument) -> List[BRepEntity]:
        """Extract BRep entities from document.

        Args:
            doc: CADling document

        Returns:
            List of BRep entities
        """
        entities = []

        # Parse document items as BRep entities
        entity_id = 0
        for item in doc.items:
            # Determine entity type from item text or label
            entity_type = self._infer_entity_type(item)

            # Extract references (entity IDs mentioned in text)
            references = self._extract_references(item.text or "")

            entity = BRepEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                text=item.text or item.label.text,
                references=references,
            )

            # Copy item properties
            if hasattr(item, "properties"):
                entity.properties = dict(item.properties)

            entities.append(entity)
            entity_id += 1

        return entities

    def _infer_entity_type(self, item) -> str:
        """Infer entity type from item.

        Args:
            item: CAD item

        Returns:
            Entity type string
        """
        text = (item.text or item.label.text).upper()

        # Common BRep entity types
        if "VERTEX" in text:
            return "VERTEX"
        elif "EDGE" in text:
            return "EDGE"
        elif "WIRE" in text:
            return "WIRE"
        elif "FACE" in text:
            return "FACE"
        elif "SHELL" in text:
            return "SHELL"
        elif "SOLID" in text:
            return "SOLID"
        elif "COMPOUND" in text:
            return "COMPOUND"
        elif "CURVE" in text:
            return "CURVE"
        elif "SURFACE" in text:
            return "SURFACE"
        elif "POINT" in text:
            return "POINT"

        return "UNKNOWN"

    def _extract_references(self, text: str) -> List[int]:
        """Extract entity ID references from text.

        Args:
            text: Entity text

        Returns:
            List of referenced entity IDs
        """
        import re

        # Look for entity references like #123, @45, etc.
        references = []
        patterns = [
            r'#(\d+)',  # #123
            r'@(\d+)',  # @123
            r'entity[_\s](\d+)',  # entity_123 or entity 123
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(int(m) for m in matches)

        return list(set(references))  # Remove duplicates

    def _chunk_by_entity_type(
        self,
        entities: List[BRepEntity],
        doc: CADlingDocument
    ) -> Iterator[CADChunk]:
        """Chunk by grouping similar entity types.

        Args:
            entities: List of BRep entities
            doc: Parent document

        Yields:
            CADChunk objects
        """
        # Group entities by type
        type_groups = defaultdict(list)

        for entity in entities:
            type_groups[entity.entity_type].append(entity)

        _log.info(f"Grouped {len(entities)} entities into {len(type_groups)} type groups")

        # Create chunks for each type group
        chunk_count = 0
        for entity_type, type_entities in type_groups.items():
            # Further split if exceeds token limit
            current_batch = []
            current_tokens = 0

            for entity in type_entities:
                entity_tokens = self._count_tokens(entity.text)

                if current_tokens + entity_tokens > self.max_tokens and current_batch:
                    # Yield chunk
                    yield self._create_chunk_from_entities(
                        current_batch,
                        doc,
                        chunk_id=f"{doc.name}_{entity_type}_{chunk_count}",
                    )
                    chunk_count += 1

                    # Start new batch
                    current_batch = []
                    current_tokens = 0

                current_batch.append(entity)
                current_tokens += entity_tokens

            # Yield final chunk for this type
            if current_batch:
                yield self._create_chunk_from_entities(
                    current_batch,
                    doc,
                    chunk_id=f"{doc.name}_{entity_type}_{chunk_count}",
                )
                chunk_count += 1

    def _chunk_by_topology(
        self,
        entities: List[BRepEntity],
        doc: CADlingDocument
    ) -> Iterator[CADChunk]:
        """Chunk by topological connectivity.

        Groups entities that are topologically connected via references.

        Args:
            entities: List of BRep entities
            doc: Parent document

        Yields:
            CADChunk objects
        """
        # Build reference graph
        entity_map = {e.entity_id: e for e in entities}
        graph = self._build_reference_graph(entities)

        # Find connected components
        components = self._find_connected_components_in_graph(graph, entities)

        _log.info(f"Found {len(components)} topological components")

        # Create chunk for each component
        for i, component_ids in enumerate(components):
            component_entities = [entity_map[eid] for eid in component_ids if eid in entity_map]

            if component_entities:
                # Split if too large
                if len(component_entities) * 50 > self.max_tokens:  # Rough estimate
                    # Split into sub-chunks
                    sub_chunk_count = 0
                    current_batch = []
                    current_tokens = 0

                    for entity in component_entities:
                        entity_tokens = self._count_tokens(entity.text)

                        if current_tokens + entity_tokens > self.max_tokens and current_batch:
                            yield self._create_chunk_from_entities(
                                current_batch,
                                doc,
                                chunk_id=f"{doc.name}_topo_{i}_{sub_chunk_count}",
                            )
                            sub_chunk_count += 1
                            current_batch = []
                            current_tokens = 0

                        current_batch.append(entity)
                        current_tokens += entity_tokens

                    if current_batch:
                        yield self._create_chunk_from_entities(
                            current_batch,
                            doc,
                            chunk_id=f"{doc.name}_topo_{i}_{sub_chunk_count}",
                        )
                else:
                    yield self._create_chunk_from_entities(
                        component_entities,
                        doc,
                        chunk_id=f"{doc.name}_topo_{i}",
                    )

    def _chunk_by_hierarchy(
        self,
        entities: List[BRepEntity],
        doc: CADlingDocument
    ) -> Iterator[CADChunk]:
        """Chunk by hierarchical structure.

        Groups entities bottom-up: vertices -> edges -> faces -> shells -> solids.

        Args:
            entities: List of BRep entities
            doc: Parent document

        Yields:
            CADChunk objects
        """
        # Define hierarchy levels
        hierarchy = [
            "POINT",
            "VERTEX",
            "CURVE",
            "EDGE",
            "WIRE",
            "SURFACE",
            "FACE",
            "SHELL",
            "SOLID",
            "COMPOUND",
        ]

        # Group by hierarchy level
        level_groups = defaultdict(list)

        for entity in entities:
            # Find hierarchy level
            if entity.entity_type in hierarchy:
                level = hierarchy.index(entity.entity_type)
            else:
                level = len(hierarchy)  # Unknown types at the end

            level_groups[level].append(entity)

        # Create chunks for each level
        chunk_count = 0
        for level in sorted(level_groups.keys()):
            level_entities = level_groups[level]

            # Split level if too large
            current_batch = []
            current_tokens = 0

            for entity in level_entities:
                entity_tokens = self._count_tokens(entity.text)

                if current_tokens + entity_tokens > self.max_tokens and current_batch:
                    yield self._create_chunk_from_entities(
                        current_batch,
                        doc,
                        chunk_id=f"{doc.name}_level_{level}_{chunk_count}",
                    )
                    chunk_count += 1
                    current_batch = []
                    current_tokens = 0

                current_batch.append(entity)
                current_tokens += entity_tokens

            if current_batch:
                yield self._create_chunk_from_entities(
                    current_batch,
                    doc,
                    chunk_id=f"{doc.name}_level_{level}_{chunk_count}",
                )
                chunk_count += 1

    def _chunk_hybrid(
        self,
        entities: List[BRepEntity],
        doc: CADlingDocument
    ) -> Iterator[CADChunk]:
        """Hybrid chunking: topology-aware + hierarchy-aware + token-limited.

        Args:
            entities: List of BRep entities
            doc: Parent document

        Yields:
            CADChunk objects
        """
        # Build reference graph
        graph = self._build_reference_graph(entities)

        # Find connected components
        components = self._find_connected_components_in_graph(graph, entities)

        chunk_count = 0

        for component_ids in components:
            # Get entities for this component
            component_entities = [e for e in entities if e.entity_id in component_ids]

            # Sort by hierarchy
            component_entities.sort(key=lambda e: self._get_hierarchy_order(e.entity_type))

            # Split into token-limited chunks with overlap
            current_batch = []
            current_tokens = 0

            for entity in component_entities:
                entity_tokens = self._count_tokens(entity.text)

                if current_tokens + entity_tokens > self.max_tokens and current_batch:
                    # Yield chunk
                    yield self._create_chunk_from_entities(
                        current_batch,
                        doc,
                        chunk_id=f"{doc.name}_hybrid_{chunk_count}",
                    )
                    chunk_count += 1

                    # Start new batch with overlap (keep last entity for context)
                    overlap_items = current_batch[-1:]
                    current_batch = overlap_items
                    current_tokens = sum(
                        self._count_tokens(e.text) for e in overlap_items
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

    def _build_reference_graph(self, entities: List[BRepEntity]) -> dict:
        """Build entity reference graph.

        Args:
            entities: List of entities

        Returns:
            Graph dictionary mapping entity_id to set of referenced entity_ids
        """
        graph = defaultdict(set)

        for entity in entities:
            # Add bidirectional edges for references
            for ref_id in entity.references:
                graph[entity.entity_id].add(ref_id)
                graph[ref_id].add(entity.entity_id)

        return graph

    def _find_connected_components_in_graph(
        self,
        graph: dict,
        entities: List[BRepEntity]
    ) -> List[Set[int]]:
        """Find connected components in reference graph.

        Args:
            graph: Reference graph
            entities: List of entities

        Returns:
            List of sets of entity IDs (one per component)
        """
        visited = set()
        components = []

        for entity in entities:
            entity_id = entity.entity_id

            if entity_id in visited:
                continue

            # BFS to find connected component using deque for O(1) popleft
            component = set()
            queue = deque([entity_id])
            visited.add(entity_id)

            while queue:
                current_id = queue.popleft()  # O(1) instead of O(n)
                component.add(current_id)

                # Add neighbors
                for neighbor_id in graph.get(current_id, []):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)

            components.append(component)

        return components

    def _get_hierarchy_order(self, entity_type: str) -> int:
        """Get hierarchy order for entity type.

        Args:
            entity_type: Entity type

        Returns:
            Hierarchy level (lower is more fundamental)
        """
        hierarchy = [
            "POINT",
            "VERTEX",
            "CURVE",
            "EDGE",
            "WIRE",
            "SURFACE",
            "FACE",
            "SHELL",
            "SOLID",
            "COMPOUND",
        ]

        if entity_type in hierarchy:
            return hierarchy.index(entity_type)
        return len(hierarchy)

    def _create_chunk_from_entities(
        self,
        entities: List[BRepEntity],
        doc: CADlingDocument,
        chunk_id: Optional[str] = None,
    ) -> CADChunk:
        """Create chunk from BRep entities.

        Args:
            entities: List of BRep entities
            doc: Parent document
            chunk_id: Chunk identifier

        Returns:
            CADChunk
        """
        # Combine text
        text_parts = []
        for entity in entities:
            text_parts.append(f"#{entity.entity_id} {entity.entity_type}")
            if entity.text:
                text_parts.append(entity.text)
            text_parts.append("")  # Blank line between entities

        text = "\n".join(text_parts)

        # Extract metadata
        entity_types = [e.entity_type for e in entities]
        entity_ids = [e.entity_id for e in entities]

        meta = CADChunkMeta(
            entity_types=entity_types,
            entity_ids=entity_ids,
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


# Convenience aliases
def EntityTypeChunker(**kwargs) -> BRepChunker:
    """Create a BRepChunker with entity_type strategy."""
    return BRepChunker(strategy="entity_type", **kwargs)


def BRepTopologyChunker(**kwargs) -> BRepChunker:
    """Create a BRepChunker with topology strategy."""
    return BRepChunker(strategy="topology", **kwargs)


# Keep backward-compatible alias
TopologyChunker = BRepTopologyChunker


def HierarchyChunker(**kwargs) -> BRepChunker:
    """Create a BRepChunker with hierarchy strategy."""
    return BRepChunker(strategy="hierarchy", **kwargs)
