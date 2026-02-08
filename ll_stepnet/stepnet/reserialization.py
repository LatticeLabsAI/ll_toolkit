"""DFS-based reserialization of STEP files.

Reorders STEP entity definitions by depth-first traversal of the entity
reference graph, producing a semantically-grouped output where related
entities appear contiguously.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .config import STEPReserializationConfig

_log = logging.getLogger(__name__)

# STEP entity line pattern: #<id>=<TYPE>(...);\n
_ENTITY_PATTERN = re.compile(
    r"#(\d+)\s*=\s*([A-Z_][A-Z0-9_]*)\s*\(([^;]*)\)\s*;",
    re.DOTALL,
)
# Reference pattern: #<id> within parameter text
_REFERENCE_PATTERN = re.compile(r"#(\d+)")

# B-Rep type hierarchy for root scoring (higher = more likely root)
_BREP_TYPE_WEIGHTS = {
    "MANIFOLD_SOLID_BREP": 100,
    "BREP_WITH_VOIDS": 95,
    "FACETED_BREP": 90,
    "SHELL_BASED_SURFACE_MODEL": 85,
    "ADVANCED_BREP_SHAPE_REPRESENTATION": 80,
    "SHAPE_REPRESENTATION": 75,
    "SHAPE_DEFINITION_REPRESENTATION": 70,
    "PRODUCT_DEFINITION_SHAPE": 65,
    "PRODUCT_DEFINITION": 60,
    "PRODUCT": 55,
    "CLOSED_SHELL": 50,
    "OPEN_SHELL": 45,
    "GEOMETRIC_SET": 40,
}

# Float normalization pattern
_FLOAT_PATTERN = re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?")


@dataclass
class STEPEntityNode:
    """A single STEP entity with its references."""
    entity_id: int
    entity_type: str
    parameters: str            # raw parameter text
    children: list[int] = field(default_factory=list)   # forward references
    parents: list[int] = field(default_factory=list)     # back references
    raw_line: str = ""


@dataclass
class STEPEntityGraph:
    """Graph of STEP entities with parent/child relationships.

    Parses raw STEP text into an entity reference graph suitable for
    DFS traversal. Each entity node stores its forward references
    (children) and back references (parents). See also
    ``STEPFeatureExtractor.extract_entity_info`` and
    ``STEPFeatureExtractor.extract_references`` in ``features.py``
    for similar single-entity parsing — the regex approach used here
    is optimised for bulk graph construction.
    """
    nodes: dict[int, STEPEntityNode] = field(default_factory=dict)

    @classmethod
    def parse(cls, step_text: str) -> "STEPEntityGraph":
        """Parse STEP text into entity graph.

        Extracts entity definitions and builds parent/child reference graph.

        Args:
            step_text: Raw STEP text containing entity definitions.

        Returns:
            Populated STEPEntityGraph instance.
        """
        graph = cls()

        for match in _ENTITY_PATTERN.finditer(step_text):
            entity_id = int(match.group(1))
            entity_type = match.group(2)
            parameters = match.group(3)
            raw_line = match.group(0)

            # Find all references in parameters
            children = []
            for ref_match in _REFERENCE_PATTERN.finditer(parameters):
                ref_id = int(ref_match.group(1))
                if ref_id != entity_id:  # skip self-references
                    children.append(ref_id)

            node = STEPEntityNode(
                entity_id=entity_id,
                entity_type=entity_type,
                parameters=parameters,
                children=children,
                raw_line=raw_line,
            )
            graph.nodes[entity_id] = node

        # Build parent (back-reference) links
        for node_id, node in graph.nodes.items():
            for child_id in node.children:
                if child_id in graph.nodes:
                    graph.nodes[child_id].parents.append(node_id)

        _log.debug(
            "Parsed STEP entity graph: %d nodes, %d edges",
            len(graph.nodes),
            sum(len(n.children) for n in graph.nodes.values()),
        )

        return graph

    @property
    def roots(self) -> list[int]:
        """Find root entities (no parents within graph)."""
        return [nid for nid, node in self.nodes.items() if not node.parents]

    def roots_by_strategy(self, strategy: str = "both") -> list[int]:
        """Find roots using specified strategy.

        Strategies:
            no_incoming: Entities with no parent references
            type_hierarchy: Entities with highest B-Rep type weight
            both: Combine and deduplicate, no_incoming first

        Args:
            strategy: Root-finding strategy name.

        Returns:
            Ordered list of root entity IDs.
        """
        if strategy == "no_incoming":
            return sorted(self.roots)

        if strategy == "type_hierarchy":
            scored = []
            for nid, node in self.nodes.items():
                weight = _BREP_TYPE_WEIGHTS.get(node.entity_type, 0)
                if weight > 0:
                    scored.append((weight, nid))
            scored.sort(reverse=True)
            return [nid for _, nid in scored]

        # "both" strategy
        no_incoming = set(self.roots)
        type_roots = []
        for nid, node in self.nodes.items():
            weight = _BREP_TYPE_WEIGHTS.get(node.entity_type, 0)
            if weight > 0:
                type_roots.append((weight, nid))
        type_roots.sort(reverse=True)

        result = sorted(no_incoming)
        for _, nid in type_roots:
            if nid not in no_incoming:
                result.append(nid)

        return result


@dataclass
class STEPReserializedOutput:
    """Output of DFS reserialization."""
    text: str                           # reserialized STEP text
    traversal_order: list[tuple[int, int]]  # [(entity_id, depth), ...]
    entity_count: int
    orphan_count: int
    max_depth_reached: int
    id_mapping: dict[int, int] = field(default_factory=dict)  # old_id -> new_id


class STEPDFSSerializer:
    """DFS-based STEP entity serializer.

    Reorders STEP entities by depth-first traversal of the reference graph,
    producing output where related entities appear contiguously. Each entity
    is expanded exactly once (branch pruning).

    Args:
        config: Reserialization configuration. Uses defaults if None.
    """

    def __init__(self, config: Optional[STEPReserializationConfig] = None):
        self.config = config or STEPReserializationConfig()

    def serialize(self, graph: STEPEntityGraph) -> STEPReserializedOutput:
        """Perform DFS reserialization of entity graph.

        Algorithm:
        1. Find roots using configured strategy
        2. DFS traverse, visiting each entity exactly once
        3. Append orphans (unreachable entities)
        4. Optionally renumber IDs sequentially
        5. Optionally normalize floats

        Args:
            graph: Parsed STEP entity graph.

        Returns:
            STEPReserializedOutput with reserialized text and metadata.
        """
        if not graph.nodes:
            return STEPReserializedOutput(
                text="",
                traversal_order=[],
                entity_count=0,
                orphan_count=0,
                max_depth_reached=0,
            )

        # Step 1: Find roots
        roots = graph.roots_by_strategy(self.config.root_strategy)
        if not roots:
            # Fallback: use all nodes sorted by ID
            roots = sorted(graph.nodes.keys())

        # Step 2: DFS traverse
        traversal_order = self._dfs_traverse(graph, roots)

        # Step 3: Collect orphans
        visited_ids = {eid for eid, _ in traversal_order}
        orphan_ids = sorted(set(graph.nodes.keys()) - visited_ids)
        orphan_count = len(orphan_ids)

        if self.config.include_orphans:
            for oid in orphan_ids:
                traversal_order.append((oid, 0))

        max_depth = max((d for _, d in traversal_order), default=0)

        # Step 4: Build output lines
        lines = []
        id_mapping = {}

        if self.config.renumber_ids:
            # Create sequential mapping
            for new_id, (old_id, _depth) in enumerate(traversal_order, start=1):
                id_mapping[old_id] = new_id

            # Rewrite each entity with new IDs
            for old_id, _depth in traversal_order:
                node = graph.nodes[old_id]
                new_id = id_mapping[old_id]
                new_params = self._rewrite_references(node.parameters, id_mapping)

                if self.config.normalize_floats:
                    new_params = self._normalize_floats(new_params)

                lines.append(f"#{new_id}={node.entity_type}({new_params});")
        else:
            for old_id, _depth in traversal_order:
                node = graph.nodes[old_id]
                params = node.parameters

                if self.config.normalize_floats:
                    params = self._normalize_floats(params)

                lines.append(f"#{old_id}={node.entity_type}({params});")

        text = "\n".join(lines)

        _log.debug(
            "DFS reserialization complete: %d entities, %d orphans, max depth %d",
            len(traversal_order),
            orphan_count,
            max_depth,
        )

        return STEPReserializedOutput(
            text=text,
            traversal_order=traversal_order,
            entity_count=len(traversal_order),
            orphan_count=orphan_count,
            max_depth_reached=max_depth,
            id_mapping=id_mapping,
        )

    def _dfs_traverse(
        self,
        graph: STEPEntityGraph,
        roots: list[int],
    ) -> list[tuple[int, int]]:
        """Iterative DFS with explicit stack.

        Each entity visited exactly once (branch pruning).
        Respects max_depth configuration.

        Args:
            graph: The entity graph to traverse.
            roots: Ordered list of root entity IDs to start traversal from.

        Returns:
            List of (entity_id, depth) in DFS order.
        """
        visited = set()
        order = []

        for root_id in roots:
            if root_id in visited or root_id not in graph.nodes:
                continue

            # Explicit stack: (entity_id, depth)
            stack = [(root_id, 0)]

            while stack:
                eid, depth = stack.pop()

                if eid in visited:
                    continue
                if eid not in graph.nodes:
                    continue
                if depth > self.config.max_depth:
                    continue

                visited.add(eid)
                order.append((eid, depth))

                # Push children in reverse order so first child is processed first
                node = graph.nodes[eid]
                children = [
                    c for c in reversed(node.children)
                    if c not in visited and c in graph.nodes
                ]
                for child_id in children:
                    stack.append((child_id, depth + 1))

        return order

    def _rewrite_references(self, params: str, id_mapping: dict[int, int]) -> str:
        """Rewrite entity references using new ID mapping.

        Args:
            params: Raw parameter text containing #<id> references.
            id_mapping: Mapping from old entity IDs to new sequential IDs.

        Returns:
            Parameter text with references rewritten.
        """
        def replace_ref(match):
            old_id = int(match.group(1))
            new_id = id_mapping.get(old_id, old_id)
            return f"#{new_id}"

        return _REFERENCE_PATTERN.sub(replace_ref, params)

    def _normalize_floats(self, text: str) -> str:
        """Normalize float values to configured precision.

        Removes trailing zeros and normalizes scientific notation.

        Examples:
            0.000000000 -> 0.0
            8.001000001 -> 8.001
            1.23456789012 -> 1.23457 (at precision=6)

        Args:
            text: Text containing float literals.

        Returns:
            Text with floats normalized to configured precision.
        """
        precision = self.config.float_precision

        def normalize_float(match):
            try:
                value = float(match.group(0))
                # Use g-specifier to remove trailing zeros
                formatted = f"{value:.{precision}g}"
                # Ensure there's always a decimal point for floats
                if "." not in formatted and "e" not in formatted.lower():
                    formatted += ".0"
                return formatted
            except ValueError:
                return match.group(0)

        return _FLOAT_PATTERN.sub(normalize_float, text)


def reserialize_step(
    step_text: str,
    config: Optional[STEPReserializationConfig] = None,
) -> STEPReserializedOutput:
    """Convenience function to reserialize STEP text via DFS.

    Parses the raw STEP text into an entity graph, then performs
    DFS reserialization to produce semantically-grouped output.

    Args:
        step_text: Raw STEP file text (DATA section entities).
        config: Optional reserialization configuration.

    Returns:
        STEPReserializedOutput with reserialized text and metadata.
    """
    graph = STEPEntityGraph.parse(step_text)
    serializer = STEPDFSSerializer(config)
    return serializer.serialize(graph)
