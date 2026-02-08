"""Structural annotations for DFS-reserialized STEP files.

Generates summary and branch annotations that describe the high-level
structure of STEP entity graphs, intended for prepending to reserialized
output to aid language model comprehension.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .config import STEPAnnotationConfig
from .reserialization import STEPEntityGraph, STEPReserializedOutput, _BREP_TYPE_WEIGHTS

_log = logging.getLogger(__name__)


@dataclass
class BranchAnnotation:
    """Annotation for a single DFS branch (subtree rooted at a root entity).

    Attributes:
        root_id: Entity ID of the branch root.
        root_type: STEP entity type of the branch root.
        descendant_count: Number of descendants (excluding root).
        max_depth: Maximum depth reached in this branch.
        type_distribution: Counts of each entity type in the branch.
    """
    root_id: int
    root_type: str
    descendant_count: int
    max_depth: int
    type_distribution: dict[str, int] = field(default_factory=dict)

    def format(self, max_types: int = 5) -> str:
        """Format branch annotation as text string.

        Args:
            max_types: Maximum number of type counts to include.

        Returns:
            Formatted annotation string with [BRANCH] delimiters.
        """
        top_types = sorted(
            self.type_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:max_types]
        type_summary = ", ".join(
            f"{count} {typ}" for typ, count in top_types
        )
        return (
            f"[BRANCH #{self.root_id} {self.root_type}] "
            f"{self.descendant_count} descendants, depth {self.max_depth}. "
            f"{type_summary}. [/BRANCH]"
        )


@dataclass
class StructuralSummary:
    """File-level structural summary.

    Attributes:
        total_entities: Total number of entities in the graph.
        root_count: Number of root entities identified.
        max_depth: Maximum DFS depth reached.
        type_distribution: Counts of each entity type.
        dominant_category: Classified category (B-Rep, Geometry, Assembly, Mixed).
    """
    total_entities: int
    root_count: int
    max_depth: int
    type_distribution: dict[str, int] = field(default_factory=dict)
    dominant_category: str = "unknown"  # B-Rep, Geometry, Assembly, etc.

    def format(self, max_types: int = 5) -> str:
        """Format summary as text string.

        Args:
            max_types: Maximum number of type counts to include.

        Returns:
            Formatted summary string with [SUMMARY] delimiters.
        """
        top_types = sorted(
            self.type_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:max_types]
        type_summary = ", ".join(
            f"{count} {typ.lower().replace('_', ' ')}" for typ, count in top_types
        )
        return (
            f"[SUMMARY] {self.total_entities} entities, "
            f"{self.root_count} root(s). "
            f"{self.dominant_category}, depth {self.max_depth}. "
            f"{type_summary}. [/SUMMARY]"
        )


@dataclass
class STEPAnnotatedOutput:
    """Combined reserialized output with structural annotations.

    Attributes:
        summary: File-level structural summary (if generated).
        branches: Per-branch annotations for each root subtree.
        reserialized_text: The DFS-reserialized entity text.
        annotated_text: Full output combining summary, branches, and entities.
    """
    summary: Optional[StructuralSummary] = None
    branches: list[BranchAnnotation] = field(default_factory=list)
    reserialized_text: str = ""
    annotated_text: str = ""  # summary + branches + reserialized

    def format(self) -> str:
        """Format full annotated output.

        Combines summary, branch annotations, and reserialized text
        into a single string separated by newlines.

        Returns:
            Complete annotated output string.
        """
        parts = []
        if self.summary:
            parts.append(self.summary.format())
        for branch in self.branches:
            parts.append(branch.format())
        if self.reserialized_text:
            parts.append(self.reserialized_text)
        self.annotated_text = "\n".join(parts)
        return self.annotated_text


class STEPStructuralAnnotator:
    """Generates structural annotations for STEP entity graphs.

    Analyzes DFS-reserialized output and the underlying entity graph
    to produce human/LLM-readable summaries of the file structure.

    Args:
        config: Annotation configuration. Uses defaults if None.
    """

    def __init__(self, config: Optional[STEPAnnotationConfig] = None):
        self.config = config or STEPAnnotationConfig()

    def annotate(
        self,
        graph: STEPEntityGraph,
        reserialized: STEPReserializedOutput,
    ) -> STEPAnnotatedOutput:
        """Generate annotations for reserialized output.

        Args:
            graph: The entity graph (pre-reserialization).
            reserialized: The DFS reserialization output.

        Returns:
            STEPAnnotatedOutput with summary, branch annotations, and full text.
        """
        output = STEPAnnotatedOutput(reserialized_text=reserialized.text)

        if self.config.include_file_summary:
            output.summary = self._generate_summary(graph, reserialized)

        if self.config.include_branch_annotations:
            output.branches = self._generate_branch_annotations(
                graph, reserialized
            )

        output.format()

        _log.debug(
            "Generated annotations: summary=%s, %d branches",
            output.summary is not None,
            len(output.branches),
        )

        return output

    def _generate_summary(
        self,
        graph: STEPEntityGraph,
        reserialized: STEPReserializedOutput,
    ) -> StructuralSummary:
        """Generate file-level structural summary.

        Args:
            graph: The entity graph.
            reserialized: The reserialization output (for depth info).

        Returns:
            StructuralSummary with entity counts and category classification.
        """
        type_counts = Counter()
        for node in graph.nodes.values():
            type_counts[node.entity_type] += 1

        # Determine dominant category
        category = self._classify_category(type_counts)

        roots = graph.roots_by_strategy("both")

        return StructuralSummary(
            total_entities=len(graph.nodes),
            root_count=len(roots),
            max_depth=reserialized.max_depth_reached,
            type_distribution=dict(type_counts),
            dominant_category=category,
        )

    def _generate_branch_annotations(
        self,
        graph: STEPEntityGraph,
        reserialized: STEPReserializedOutput,
    ) -> list[BranchAnnotation]:
        """Generate per-branch annotations for each root subtree.

        Uses DFS to explore each root's subtree independently and
        collects type distribution and depth statistics.

        Args:
            graph: The entity graph.
            reserialized: The reserialization output.

        Returns:
            List of BranchAnnotation, one per non-trivial root subtree.
        """
        branches = []
        roots = graph.roots_by_strategy("both")

        # Track which entities belong to which root's subtree
        for root_id in roots:
            if root_id not in graph.nodes:
                continue

            root_node = graph.nodes[root_id]

            # DFS to count descendants
            visited = set()
            type_counts = Counter()
            max_depth = 0

            depth_stack = [(root_id, 0)]
            while depth_stack:
                eid, depth = depth_stack.pop()
                if eid in visited or eid not in graph.nodes:
                    continue
                visited.add(eid)
                node = graph.nodes[eid]
                type_counts[node.entity_type] += 1
                max_depth = max(max_depth, depth)

                for child_id in node.children:
                    if child_id not in visited and child_id in graph.nodes:
                        depth_stack.append((child_id, depth + 1))

            if len(visited) > 1:  # Only annotate non-trivial branches
                branches.append(BranchAnnotation(
                    root_id=root_id,
                    root_type=root_node.entity_type,
                    descendant_count=len(visited) - 1,  # exclude root
                    max_depth=max_depth,
                    type_distribution=dict(type_counts),
                ))

        _log.debug("Generated %d branch annotations", len(branches))

        return branches

    def _classify_category(self, type_counts: Counter) -> str:
        """Classify the dominant entity category.

        Examines the distribution of entity types to determine whether
        the file primarily represents B-Rep topology, raw geometry,
        an assembly structure, or a mix.

        Args:
            type_counts: Counter of entity type occurrences.

        Returns:
            Category string: 'B-Rep', 'Geometry', 'Assembly', 'Mixed', or 'unknown'.
        """
        brep_types = {
            "MANIFOLD_SOLID_BREP", "CLOSED_SHELL", "OPEN_SHELL",
            "ADVANCED_FACE", "FACE_OUTER_BOUND", "FACE_BOUND",
            "EDGE_LOOP", "ORIENTED_EDGE", "EDGE_CURVE",
            "VERTEX_POINT",
        }
        geom_types = {
            "CARTESIAN_POINT", "DIRECTION", "AXIS2_PLACEMENT_3D",
            "LINE", "CIRCLE", "ELLIPSE", "B_SPLINE_CURVE_WITH_KNOTS",
            "B_SPLINE_SURFACE_WITH_KNOTS", "CYLINDRICAL_SURFACE",
            "CONICAL_SURFACE", "SPHERICAL_SURFACE", "TOROIDAL_SURFACE",
            "PLANE",
        }
        assembly_types = {
            "NEXT_ASSEMBLY_USAGE_OCCURRENCE", "PRODUCT_DEFINITION",
            "PRODUCT", "PRODUCT_DEFINITION_FORMATION",
        }

        brep_count = sum(type_counts.get(t, 0) for t in brep_types)
        geom_count = sum(type_counts.get(t, 0) for t in geom_types)
        assembly_count = sum(type_counts.get(t, 0) for t in assembly_types)

        total = sum(type_counts.values())
        if total == 0:
            return "unknown"

        if assembly_count > 5 and assembly_count / total > 0.1:
            return "Assembly"
        elif brep_count > geom_count:
            return "B-Rep"
        elif geom_count > 0:
            return "Geometry"
        else:
            return "Mixed"
