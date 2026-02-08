"""DFS-ordered chunker for CAD documents.

Chunks CAD entities in depth-first traversal order, keeping subtree-related
entities together for better semantic coherence in RAG retrieval.

Uses the DFS reserialization from ll_stepnet when available, with fallback
to document topology when unavailable.

Classes:
    DFSChunker: DFS traversal-based chunker for CAD documents
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterator, Optional

from cadling.chunker.base_chunker import BaseCADChunker, CADChunk, CADChunkMeta

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument

_log = logging.getLogger(__name__)

# Try to import reserialization for DFS ordering
try:
    from stepnet.reserialization import (
        STEPDFSSerializer,
        STEPEntityGraph,
        STEPReserializationConfig,
    )

    _HAS_RESERIALIZATION = True
except ImportError:
    _HAS_RESERIALIZATION = False
    _log.debug("stepnet reserialization not available, using fallback DFS")


class DFSChunker(BaseCADChunker):
    """Chunks CAD documents using DFS traversal ordering.

    Entities are ordered by depth-first traversal of the reference graph,
    then partitioned into chunks respecting token limits and subtree boundaries.
    Falls back to document topology when stepnet reserialization is unavailable.

    The DFS ordering ensures that related entities (parent and children) appear
    together in the same chunk, improving semantic coherence for RAG retrieval.

    Args:
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Number of overlapping tokens between chunks.
        min_chunk_entities: Minimum entities to include in a chunk.
        max_cross_references: Maximum forward cross-chunk references before forcing a split.
        vocab_size: Vocabulary size for token estimation.

    Example:
        chunker = DFSChunker(max_tokens=512)
        for chunk in chunker.chunk(doc):
            print(chunk.text)
            print(chunk.meta.properties["traversal_strategy"])
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        min_chunk_entities: int = 3,
        max_cross_references: int = 5,
        vocab_size: int = 50000,
    ):
        """Initialize DFS chunker.

        Args:
            max_tokens: Maximum tokens per chunk.
            overlap_tokens: Number of overlapping tokens between chunks.
            min_chunk_entities: Minimum entities to include in a chunk.
            max_cross_references: Maximum forward cross-chunk references
                before forcing a split.
            vocab_size: Vocabulary size for token estimation.
        """
        super().__init__(max_tokens=max_tokens, overlap_tokens=overlap_tokens, vocab_size=vocab_size)
        self.min_chunk_entities = min_chunk_entities
        self.max_cross_references = max_cross_references

    def chunk(self, doc: CADlingDocument) -> Iterator[CADChunk]:
        """Chunk document using DFS ordering.

        Steps:
        1. Extract entities from document
        2. Build DAG from topology or entity references
        3. Perform DFS traversal
        4. Find subtree boundaries
        5. Partition into chunks respecting token limits
        6. Count cross-references
        7. Yield CADChunk objects

        Args:
            doc: CADlingDocument to chunk.

        Yields:
            CADChunk objects with DFS-ordered entities.
        """
        # Get entities from document
        entities = self._get_entities(doc)
        if not entities:
            _log.debug("No entities found in document '%s', skipping DFS chunking", doc.name)
            return

        entity_map = {self._get_entity_id(e): e for e in entities}
        entity_ids = list(entity_map.keys())

        _log.info(
            "DFS chunking document '%s' with %d entities (max_tokens=%d)",
            doc.name,
            len(entity_ids),
            self.max_tokens,
        )

        # Build DAG
        forward_adj, reverse_adj, roots = self._build_dag(doc, entity_map)

        # DFS traversal
        dfs_order = self._dfs_traversal(roots, forward_adj, set(entity_ids))

        # Find subtree boundaries
        boundaries = self._find_subtree_boundaries(dfs_order)

        # Partition into chunks
        raw_chunks = self._partition_into_chunks(dfs_order, boundaries, entity_map)

        if not raw_chunks:
            _log.debug("No chunks produced for document '%s'", doc.name)
            return

        # Count cross-references
        cross_refs = self._count_cross_references(raw_chunks, forward_adj)

        # Create and yield chunk objects
        chunk_count = 0
        for idx, (chunk_entities, chunk_dfs_info) in enumerate(raw_chunks):
            chunk = self._create_dfs_chunk(
                chunk_entities, doc, entity_map, idx, cross_refs[idx], chunk_dfs_info
            )
            yield chunk
            chunk_count += 1

        _log.info(
            "Generated %d DFS chunks from document '%s' (%d subtree boundaries)",
            chunk_count,
            doc.name,
            len(boundaries),
        )

    def _get_entities(self, doc: CADlingDocument) -> list[CADItem]:
        """Extract entities from the document.

        Looks for items in the document, filtering to those with
        identifiable IDs for graph construction.

        Args:
            doc: CADlingDocument to extract entities from.

        Returns:
            List of CADItem objects from the document.
        """
        if not doc.items:
            return []
        return list(doc.items)

    def _get_entity_id(self, entity: CADItem) -> str:
        """Get a string ID for an entity.

        Tries multiple attribute patterns to find a unique identifier.

        Args:
            entity: CADItem to get ID for.

        Returns:
            String identifier for the entity.
        """
        # Try STEP entity_id first
        if hasattr(entity, "entity_id") and entity.entity_id is not None:
            return str(entity.entity_id)

        # Try generic item_id
        if hasattr(entity, "item_id") and entity.item_id is not None:
            return str(entity.item_id)

        # Try label text as fallback
        if hasattr(entity, "label") and entity.label is not None:
            return entity.label.text

        # Ultimate fallback: use id()
        return str(id(entity))

    def _build_dag(
        self, doc: CADlingDocument, entity_map: dict[str, CADItem]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
        """Build DAG from document topology or entity references.

        Attempts to build the directed acyclic graph from:
        1. Document topology (preferred)
        2. Entity reference parameters (fallback)

        Args:
            doc: CADlingDocument with topology information.
            entity_map: Mapping of entity ID to entity object.

        Returns:
            Tuple of (forward_adj, reverse_adj, roots) where adj maps are
            {entity_id: [referenced_entity_ids]}.
        """
        forward_adj: dict[str, list[str]] = defaultdict(list)
        reverse_adj: dict[str, list[str]] = defaultdict(list)
        all_ids = set(entity_map.keys())

        # Try to use document topology first
        if hasattr(doc, "topology") and doc.topology:
            topology = doc.topology
            # Build from topology adjacency_list (TopologyGraph uses this)
            if hasattr(topology, "adjacency_list") and topology.adjacency_list:
                for src, targets in topology.adjacency_list.items():
                    src_str = str(src)
                    for tgt in targets:
                        tgt_str = str(tgt)
                        if src_str in all_ids and tgt_str in all_ids:
                            forward_adj[src_str].append(tgt_str)
                            reverse_adj[tgt_str].append(src_str)
            elif hasattr(topology, "edges"):
                for edge in topology.edges:
                    src = str(getattr(edge, "source", getattr(edge, "from_id", "")))
                    tgt = str(getattr(edge, "target", getattr(edge, "to_id", "")))
                    if src in all_ids and tgt in all_ids:
                        forward_adj[src].append(tgt)
                        reverse_adj[tgt].append(src)

        # Fallback: build from entity reference_params or children
        if not forward_adj:
            for eid, entity in entity_map.items():
                refs: list = []
                if hasattr(entity, "reference_params") and entity.reference_params:
                    refs = entity.reference_params
                elif hasattr(entity, "references") and entity.references:
                    refs = entity.references
                elif hasattr(entity, "children") and entity.children:
                    refs = entity.children

                for ref in refs:
                    ref_id = str(ref)
                    if ref_id in all_ids and ref_id != eid:
                        forward_adj[eid].append(ref_id)
                        reverse_adj[ref_id].append(eid)

        # Find roots (no incoming edges)
        roots = [eid for eid in all_ids if eid not in reverse_adj or not reverse_adj[eid]]
        if not roots:
            # Fallback: use all entities sorted by ID
            roots = sorted(all_ids)
        else:
            roots = sorted(roots)

        _log.debug(
            "Built DAG: %d forward edges, %d roots",
            sum(len(v) for v in forward_adj.values()),
            len(roots),
        )

        return dict(forward_adj), dict(reverse_adj), roots

    def _dfs_traversal(
        self,
        roots: list[str],
        forward_adj: dict[str, list[str]],
        all_ids: set[str],
    ) -> list[tuple[str, int]]:
        """Iterative DFS traversal with depth tracking.

        Uses an explicit stack to avoid recursion limits. Each entity
        is visited exactly once (branch pruning). Orphan entities
        (unreachable from roots) are appended at the end.

        Args:
            roots: Root entity IDs to start traversal from.
            forward_adj: Forward adjacency mapping.
            all_ids: Set of all entity IDs in the graph.

        Returns:
            List of (entity_id, depth) in DFS order.
        """
        visited: set[str] = set()
        order: list[tuple[str, int]] = []

        for root_id in roots:
            if root_id in visited:
                continue

            stack = [(root_id, 0)]

            while stack:
                eid, depth = stack.pop()

                if eid in visited:
                    continue

                visited.add(eid)
                order.append((eid, depth))

                # Push children in reverse order for correct DFS ordering
                children = forward_adj.get(eid, [])
                for child_id in reversed(children):
                    if child_id not in visited and child_id in all_ids:
                        stack.append((child_id, depth + 1))

        # Add unvisited entities (orphans)
        for eid in sorted(all_ids - visited):
            order.append((eid, 0))

        return order

    def _find_subtree_boundaries(self, dfs_order: list[tuple[str, int]]) -> list[int]:
        """Find subtree boundary indices in DFS order.

        Boundaries occur where depth returns to root level (0),
        indicating the start of a new top-level subtree.

        Args:
            dfs_order: List of (entity_id, depth) in DFS order.

        Returns:
            List of indices marking subtree boundaries.
        """
        boundaries = []
        for i in range(1, len(dfs_order)):
            _, depth = dfs_order[i]
            if depth == 0:
                boundaries.append(i)
        return boundaries

    def _partition_into_chunks(
        self,
        dfs_order: list[tuple[str, int]],
        boundaries: list[int],
        entity_map: dict[str, Any],
    ) -> list[tuple[list[tuple[str, int, Any]], list[tuple[str, int]]]]:
        """Partition DFS-ordered entities into chunks.

        Splits at subtree boundaries while respecting max_tokens.
        Merges undersized chunks with neighbors.

        Args:
            dfs_order: DFS-ordered list of (entity_id, depth).
            boundaries: Subtree boundary indices.
            entity_map: Mapping of entity ID to entity object.

        Returns:
            List of (chunk_entities, chunk_dfs_info) tuples where
            chunk_entities is [(eid, depth, entity), ...] and
            chunk_dfs_info is [(eid, depth), ...].
        """
        if not dfs_order:
            return []

        # Create initial segments at subtree boundaries
        all_boundaries = [0] + boundaries + [len(dfs_order)]
        segments: list[list[tuple[str, int]]] = []
        for i in range(len(all_boundaries) - 1):
            start = all_boundaries[i]
            end = all_boundaries[i + 1]
            segment = dfs_order[start:end]
            if segment:
                segments.append(segment)

        # Split segments that exceed max_tokens
        chunks: list[list[tuple[str, int, Any]]] = []
        for segment in segments:
            current_chunk: list[tuple[str, int, Any]] = []
            current_tokens = 0

            for eid, depth in segment:
                entity = entity_map.get(eid)
                if entity is None:
                    continue

                entity_text = self._item_to_text(entity)
                entity_tokens = self._estimate_tokens(entity_text)

                if (
                    current_tokens + entity_tokens > self.max_tokens
                    and len(current_chunk) >= self.min_chunk_entities
                ):
                    chunks.append(list(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                current_chunk.append((eid, depth, entity))
                current_tokens += entity_tokens

            if current_chunk:
                chunks.append(current_chunk)

        # Merge undersized chunks
        merged: list[list[tuple[str, int, Any]]] = []
        i = 0
        while i < len(chunks):
            if len(chunks[i]) < self.min_chunk_entities and merged:
                # Merge with previous chunk
                merged[-1].extend(chunks[i])
            else:
                merged.append(chunks[i])
            i += 1

        # Convert to (entities, dfs_info) format
        result: list[tuple[list[tuple[str, int, Any]], list[tuple[str, int]]]] = []
        for chunk in merged:
            entities = [(eid, depth, entity) for eid, depth, entity in chunk]
            dfs_info = [(eid, depth) for eid, depth, _ in chunk]
            result.append((entities, dfs_info))

        return result

    def _count_cross_references(
        self,
        chunks: list[tuple[list[tuple[str, int, Any]], list[tuple[str, int]]]],
        forward_adj: dict[str, list[str]],
    ) -> list[int]:
        """Count cross-chunk forward references for each chunk.

        A cross-reference occurs when an entity in one chunk references
        an entity in a different chunk.

        Args:
            chunks: List of (chunk_entities, chunk_dfs_info) tuples.
            forward_adj: Forward adjacency mapping.

        Returns:
            List of cross-reference counts, one per chunk.
        """
        # Build chunk membership map
        entity_to_chunk: dict[str, int] = {}
        for chunk_idx, (entities, _) in enumerate(chunks):
            for eid, _, _ in entities:
                entity_to_chunk[eid] = chunk_idx

        cross_refs = []
        for chunk_idx, (entities, _) in enumerate(chunks):
            count = 0
            for eid, _, _ in entities:
                for ref_id in forward_adj.get(eid, []):
                    ref_chunk = entity_to_chunk.get(ref_id)
                    if ref_chunk is not None and ref_chunk != chunk_idx:
                        count += 1
            cross_refs.append(count)

        return cross_refs

    def _create_dfs_chunk(
        self,
        chunk_entities: list[tuple[str, int, Any]],
        doc: CADlingDocument,
        entity_map: dict[str, Any],
        idx: int,
        cross_refs: int,
        dfs_info: list[tuple[str, int]],
    ) -> CADChunk:
        """Create a CADChunk with DFS structural metadata.

        Follows the pattern established by STEPChunker and
        CADHierarchicalChunker for chunk creation.

        Args:
            chunk_entities: List of (eid, depth, entity) tuples.
            doc: Parent CADlingDocument.
            entity_map: Full entity map for lookups.
            idx: Chunk index in the sequence.
            cross_refs: Number of cross-chunk forward references.
            dfs_info: DFS traversal info [(eid, depth), ...].

        Returns:
            CADChunk with DFS metadata in properties.
        """
        from cadling.datamodel.step import STEPEntityItem

        # Build text from entities
        texts: list[str] = []
        entity_types: list[str] = []
        entity_ids: list[int] = []
        type_counts: dict[str, int] = defaultdict(int)
        depths: list[int] = []

        for eid, depth, entity in chunk_entities:
            text = self._item_to_text(entity)
            texts.append(text)
            depths.append(depth)

            # Extract entity type information
            if isinstance(entity, STEPEntityItem):
                entity_types.append(entity.entity_type)
                entity_ids.append(entity.entity_id)
                type_counts[entity.entity_type] += 1
            else:
                entity_type = getattr(
                    entity, "type", getattr(entity, "entity_type", getattr(entity, "item_type", "unknown"))
                )
                type_counts[str(entity_type)] += 1

        chunk_text = "\n".join(texts)

        # Extract topology subgraph for STEP entities
        topology_subgraph = self._extract_topology_subgraph(entity_ids, doc.topology)

        # Build DFS-specific metadata
        properties: dict[str, Any] = {
            "num_entities": len(chunk_entities),
            "unique_types": len(type_counts),
            "subtree_depth": max(depths) - min(depths) if depths else 0,
            "min_depth": min(depths) if depths else 0,
            "entity_type_distribution": dict(type_counts),
            "cross_chunk_forward_refs": cross_refs,
            "dfs_chunk_index": idx,
            "traversal_strategy": "dfs",
        }

        meta = CADChunkMeta(
            entity_types=entity_types,
            entity_ids=entity_ids,
            topology_subgraph=topology_subgraph,
            properties=properties,
        )

        chunk_id = f"{doc.name}_dfs_{idx}"

        return CADChunk(
            text=chunk_text,
            meta=meta,
            chunk_id=chunk_id,
            doc_name=doc.name,
        )

    def _item_to_text(self, item: Any) -> str:
        """Convert a CAD item/entity to text representation.

        Attempts multiple attribute patterns to produce a meaningful
        text representation of the entity.

        Args:
            item: CAD item or entity to convert.

        Returns:
            Text representation of the item.
        """
        from cadling.datamodel.step import STEPEntityItem

        # Handle STEPEntityItem specifically (matches STEPChunker pattern)
        if isinstance(item, STEPEntityItem):
            parts = [f"#{item.entity_id} {item.entity_type}"]
            if item.text:
                parts.append(item.text)
            return "\n".join(parts)

        # Try raw_line attribute
        if hasattr(item, "raw_line") and item.raw_line:
            return str(item.raw_line)

        # Build from entity_id and type
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

        # Try CADItem standard attributes (label + text)
        if hasattr(item, "label") and item.label is not None:
            label_parts = [f"[{item.label.text}]"]
            if hasattr(item, "text") and item.text:
                label_parts.append(item.text)
            return "\n".join(label_parts)

        # Fallback to string representation
        return str(item)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple character-based heuristic. STEP text averages
        roughly 4 characters per token.

        Args:
            text: Text to estimate token count for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        # Rough estimate: ~4 chars per token for STEP text
        return max(1, len(text) // 4)
