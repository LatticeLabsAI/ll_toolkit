"""
STEP Topology Builder - Constructs entity reference graphs and analyzes topology.

Implements topology analysis from scratch without external dependencies.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque
import re


class TopologyBuilder:
    """Builds and analyzes topological relationships in STEP files."""

    def __init__(self):
        """Initialize the topology builder."""
        self.adjacency_list: Dict[int, List[int]] = defaultdict(list)
        self.reverse_adjacency_list: Dict[int, List[int]] = defaultdict(list)
        self.entity_levels: Dict[int, int] = {}
        self.connected_components: List[Set[int]] = []

    def build_topology_graph(
        self, entities: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build topology graph from entity references.

        Args:
            entities: Dictionary mapping entity_id to parsed entity data

        Returns:
            Dictionary containing topology graph information
        """
        # Reset state
        self.adjacency_list = defaultdict(list)
        self.reverse_adjacency_list = defaultdict(list)
        self.entity_levels = {}

        # Build adjacency lists
        for entity_id, entity_data in entities.items():
            refs = self._extract_all_references(entity_data)
            for ref_id in refs:
                if ref_id in entities:
                    # Add edge from entity to referenced entity
                    self.adjacency_list[entity_id].append(ref_id)
                    # Add reverse edge for traversal
                    self.reverse_adjacency_list[ref_id].append(entity_id)

        # Compute entity levels (topological sort levels)
        self._compute_entity_levels(entities)

        # Find connected components
        self.connected_components = self._find_connected_components(entities)

        # Compute topology statistics
        topology_stats = self._compute_topology_statistics(entities)

        return {
            "num_nodes": len(entities),
            "num_edges": sum(len(refs) for refs in self.adjacency_list.values()),
            "adjacency_list": dict(self.adjacency_list),
            "reverse_adjacency_list": dict(self.reverse_adjacency_list),
            "entity_levels": self.entity_levels,
            "num_connected_components": len(self.connected_components),
            "connected_components": [list(comp) for comp in self.connected_components],
            "topology_statistics": topology_stats,
        }

    def _extract_all_references(self, entity_data: Dict[str, Any]) -> List[int]:
        """Extract all entity references from entity data."""
        refs = []
        params = entity_data.get("params", [])

        for param in params:
            if isinstance(param, str):
                # Extract references like #123
                matches = re.findall(r"#(\d+)", param)
                refs.extend(int(m) for m in matches)
            elif isinstance(param, list):
                # Recursively extract from lists
                refs.extend(self._extract_refs_from_list(param))

        return refs

    def _extract_refs_from_list(self, param_list: List[Any]) -> List[int]:
        """Recursively extract references from nested lists."""
        refs = []
        for item in param_list:
            if isinstance(item, str):
                matches = re.findall(r"#(\d+)", item)
                refs.extend(int(m) for m in matches)
            elif isinstance(item, list):
                refs.extend(self._extract_refs_from_list(item))
        return refs

    def _compute_entity_levels(self, entities: Dict[int, Dict[str, Any]]) -> None:
        """
        Compute entity levels using topological sort.
        Level 0: entities with no dependencies
        Level n: entities that depend only on entities at levels < n
        """
        # Compute in-degree for each entity
        in_degree = defaultdict(int)
        for entity_id in entities:
            in_degree[entity_id] = 0

        for entity_id, refs in self.adjacency_list.items():
            for ref_id in refs:
                in_degree[ref_id] += 1

        # BFS to compute levels
        queue = deque()
        for entity_id in entities:
            if in_degree[entity_id] == 0:
                self.entity_levels[entity_id] = 0
                queue.append(entity_id)

        while queue:
            current_id = queue.popleft()
            current_level = self.entity_levels[current_id]

            # Process all entities that reference this one
            for dependent_id in self.reverse_adjacency_list[current_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    self.entity_levels[dependent_id] = current_level + 1
                    queue.append(dependent_id)

        # Handle cycles: entities not assigned a level
        for entity_id in entities:
            if entity_id not in self.entity_levels:
                self.entity_levels[entity_id] = -1  # Mark as part of cycle

    def _find_connected_components(
        self, entities: Dict[int, Dict[str, Any]]
    ) -> List[Set[int]]:
        """Find connected components in the entity graph."""
        visited = set()
        components = []

        for entity_id in entities:
            if entity_id not in visited:
                component = self._dfs_component(entity_id, entities, visited)
                components.append(component)

        return components

    def _dfs_component(
        self, start_id: int, entities: Dict[int, Dict[str, Any]], visited: Set[int]
    ) -> Set[int]:
        """DFS to find all entities in a connected component."""
        component = set()
        stack = [start_id]

        while stack:
            entity_id = stack.pop()
            if entity_id in visited:
                continue

            visited.add(entity_id)
            component.add(entity_id)

            # Add neighbors (both forward and backward edges)
            for neighbor_id in self.adjacency_list[entity_id]:
                if neighbor_id not in visited and neighbor_id in entities:
                    stack.append(neighbor_id)

            for neighbor_id in self.reverse_adjacency_list[entity_id]:
                if neighbor_id not in visited and neighbor_id in entities:
                    stack.append(neighbor_id)

        return component

    def _compute_topology_statistics(
        self, entities: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute various topology statistics."""
        stats = {}

        # Degree distribution
        out_degrees = [len(refs) for refs in self.adjacency_list.values()]
        in_degrees = [len(refs) for refs in self.reverse_adjacency_list.values()]

        if out_degrees:
            stats["avg_out_degree"] = sum(out_degrees) / len(entities)
            stats["max_out_degree"] = max(out_degrees)
        else:
            stats["avg_out_degree"] = 0
            stats["max_out_degree"] = 0

        if in_degrees:
            stats["avg_in_degree"] = sum(in_degrees) / len(entities)
            stats["max_in_degree"] = max(in_degrees)
        else:
            stats["avg_in_degree"] = 0
            stats["max_in_degree"] = 0

        # Level distribution
        level_counts = defaultdict(int)
        for level in self.entity_levels.values():
            level_counts[level] += 1

        stats["num_levels"] = len([l for l in level_counts if l >= 0])
        stats["num_cycles"] = level_counts.get(-1, 0)
        stats["level_distribution"] = dict(level_counts)

        # Component statistics
        if self.connected_components:
            component_sizes = [len(comp) for comp in self.connected_components]
            stats["largest_component_size"] = max(component_sizes)
            stats["smallest_component_size"] = min(component_sizes)
            stats["avg_component_size"] = sum(component_sizes) / len(component_sizes)

        # Find root entities (level 0, no incoming edges)
        root_entities = [
            entity_id
            for entity_id, level in self.entity_levels.items()
            if level == 0 and not self.reverse_adjacency_list[entity_id]
        ]
        stats["num_root_entities"] = len(root_entities)
        stats["root_entity_ids"] = root_entities[:10]  # Limit to first 10

        # Find leaf entities (no outgoing edges)
        leaf_entities = [
            entity_id
            for entity_id in entities
            if not self.adjacency_list[entity_id]
        ]
        stats["num_leaf_entities"] = len(leaf_entities)

        return stats

    def get_entity_neighborhood(
        self, entity_id: int, depth: int = 1, direction: str = "both"
    ) -> Set[int]:
        """
        Get all entities within a certain graph distance.

        Args:
            entity_id: The starting entity ID
            depth: Maximum distance to traverse
            direction: "forward", "backward", or "both"

        Returns:
            Set of entity IDs in the neighborhood
        """
        neighborhood = set()
        queue = deque([(entity_id, 0)])
        visited = {entity_id}

        while queue:
            current_id, current_depth = queue.popleft()
            neighborhood.add(current_id)

            if current_depth >= depth:
                continue

            neighbors = []
            if direction in ("forward", "both"):
                neighbors.extend(self.adjacency_list[current_id])
            if direction in ("backward", "both"):
                neighbors.extend(self.reverse_adjacency_list[current_id])

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, current_depth + 1))

        return neighborhood

    def find_shortest_path(
        self, start_id: int, end_id: int
    ) -> Optional[List[int]]:
        """
        Find shortest path between two entities.

        Args:
            start_id: Starting entity ID
            end_id: Target entity ID

        Returns:
            List of entity IDs forming the path, or None if no path exists
        """
        if start_id == end_id:
            return [start_id]

        queue = deque([(start_id, [start_id])])
        visited = {start_id}

        while queue:
            current_id, path = queue.popleft()

            # Check neighbors (forward edges only)
            for neighbor_id in self.adjacency_list[current_id]:
                if neighbor_id == end_id:
                    return path + [end_id]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def get_dependency_chain(self, entity_id: int) -> List[int]:
        """
        Get the full dependency chain for an entity.

        Args:
            entity_id: The entity ID to analyze

        Returns:
            List of entity IDs that this entity depends on (directly or indirectly)
        """
        dependencies = set()
        queue = deque([entity_id])
        visited = {entity_id}

        while queue:
            current_id = queue.popleft()

            for ref_id in self.adjacency_list[current_id]:
                if ref_id not in visited:
                    visited.add(ref_id)
                    dependencies.add(ref_id)
                    queue.append(ref_id)

        return list(dependencies)

    def get_dependents(self, entity_id: int) -> List[int]:
        """
        Get all entities that depend on this entity.

        Args:
            entity_id: The entity ID to analyze

        Returns:
            List of entity IDs that depend on this entity (directly or indirectly)
        """
        dependents = set()
        queue = deque([entity_id])
        visited = {entity_id}

        while queue:
            current_id = queue.popleft()

            for dependent_id in self.reverse_adjacency_list[current_id]:
                if dependent_id not in visited:
                    visited.add(dependent_id)
                    dependents.add(dependent_id)
                    queue.append(dependent_id)

        return list(dependents)

    def analyze_topology_type(
        self, entities: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the topological structure of the STEP file.

        Args:
            entities: Dictionary of all entities

        Returns:
            Dictionary describing the topology type
        """
        analysis = {
            "has_brep": False,
            "has_mesh": False,
            "has_wireframe": False,
            "has_surfaces": False,
            "has_curves": False,
            "representation_type": "unknown",
        }

        # Count entity types
        entity_types = [entity["type"] for entity in entities.values()]

        # Check for B-Rep entities
        brep_indicators = [
            "MANIFOLD_SOLID_BREP",
            "BREP_WITH_VOIDS",
            "FACETED_BREP",
            "CLOSED_SHELL",
            "OPEN_SHELL",
            "FACE_SURFACE",
            "EDGE_CURVE",
        ]
        analysis["has_brep"] = any(et in brep_indicators for et in entity_types)

        # Check for surface entities
        surface_indicators = [
            "PLANE",
            "CYLINDRICAL_SURFACE",
            "CONICAL_SURFACE",
            "SPHERICAL_SURFACE",
            "TOROIDAL_SURFACE",
            "B_SPLINE_SURFACE",
        ]
        analysis["has_surfaces"] = any(et in surface_indicators for et in entity_types)

        # Check for curve entities
        curve_indicators = [
            "LINE",
            "CIRCLE",
            "ELLIPSE",
            "B_SPLINE_CURVE",
            "POLYLINE",
        ]
        analysis["has_curves"] = any(et in curve_indicators for et in entity_types)

        # Check for wireframe
        wireframe_indicators = [
            "POLYLINE",
            "COMPOSITE_CURVE",
            "GEOMETRIC_CURVE_SET",
        ]
        analysis["has_wireframe"] = any(
            et in wireframe_indicators for et in entity_types
        )

        # Determine representation type
        if analysis["has_brep"]:
            analysis["representation_type"] = "brep"
        elif analysis["has_surfaces"]:
            analysis["representation_type"] = "surface"
        elif analysis["has_wireframe"]:
            analysis["representation_type"] = "wireframe"
        elif analysis["has_curves"]:
            analysis["representation_type"] = "curve"

        # Count topology hierarchy levels
        shell_count = sum(
            1
            for et in entity_types
            if et in ("CLOSED_SHELL", "OPEN_SHELL")
        )
        face_count = sum(
            1
            for et in entity_types
            if et in ("FACE_SURFACE", "FACE_BOUND", "ADVANCED_FACE")
        )
        edge_count = sum(
            1
            for et in entity_types
            if et in ("EDGE_CURVE", "ORIENTED_EDGE")
        )
        vertex_count = sum(1 for et in entity_types if et == "VERTEX_POINT")

        analysis["topology_counts"] = {
            "shells": shell_count,
            "faces": face_count,
            "edges": edge_count,
            "vertices": vertex_count,
        }

        # Compute Euler characteristic if we have a manifold
        if analysis["has_brep"] and vertex_count > 0:
            euler_characteristic = vertex_count - edge_count + face_count
            analysis["euler_characteristic"] = euler_characteristic

            # For a closed solid: χ = 2 - 2g where g is genus (number of holes)
            # For genus 0 (sphere topology): χ = 2
            # For genus 1 (torus topology): χ = 0
            # For genus 2: χ = -2, etc.
            if shell_count > 0:
                genus = (2 - euler_characteristic) // 2
                analysis["estimated_genus"] = max(0, genus)

        return analysis

    def extract_topology_hierarchy(
        self, entities: Dict[int, Dict[str, Any]]
    ) -> Dict[str, List[int]]:
        """
        Extract the topological hierarchy (solids -> shells -> faces -> edges -> vertices).

        Args:
            entities: Dictionary of all entities

        Returns:
            Dictionary mapping hierarchy levels to entity IDs
        """
        hierarchy = {
            "solids": [],
            "shells": [],
            "faces": [],
            "edges": [],
            "vertices": [],
        }

        for entity_id, entity_data in entities.items():
            entity_type = entity_data["type"]

            if entity_type in ("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS"):
                hierarchy["solids"].append(entity_id)
            elif entity_type in ("CLOSED_SHELL", "OPEN_SHELL"):
                hierarchy["shells"].append(entity_id)
            elif entity_type in ("FACE_SURFACE", "ADVANCED_FACE", "FACE_BOUND"):
                hierarchy["faces"].append(entity_id)
            elif entity_type in ("EDGE_CURVE", "ORIENTED_EDGE"):
                hierarchy["edges"].append(entity_id)
            elif entity_type == "VERTEX_POINT":
                hierarchy["vertices"].append(entity_id)

        return hierarchy

    def compute_entity_importance(
        self, entities: Dict[int, Dict[str, Any]]
    ) -> Dict[int, float]:
        """
        Compute importance scores for entities based on topology.

        Uses a combination of:
        - In-degree (how many entities reference this one)
        - Out-degree (how many entities this one references)
        - Level in hierarchy
        - Entity type importance

        Args:
            entities: Dictionary of all entities

        Returns:
            Dictionary mapping entity_id to importance score (0-1)
        """
        importance_scores = {}

        # Type-based importance weights
        type_weights = {
            "MANIFOLD_SOLID_BREP": 1.0,
            "CLOSED_SHELL": 0.9,
            "OPEN_SHELL": 0.85,
            "FACE_SURFACE": 0.7,
            "ADVANCED_FACE": 0.7,
            "EDGE_CURVE": 0.5,
            "ORIENTED_EDGE": 0.5,
            "VERTEX_POINT": 0.3,
            "CARTESIAN_POINT": 0.2,
        }

        max_in_degree = max(
            (len(refs) for refs in self.reverse_adjacency_list.values()),
            default=1,
        )
        max_out_degree = max(
            (len(refs) for refs in self.adjacency_list.values()),
            default=1,
        )

        for entity_id, entity_data in entities.items():
            entity_type = entity_data["type"]

            # Type-based score
            type_score = type_weights.get(entity_type, 0.4)

            # Connectivity score (normalized)
            in_degree = len(self.reverse_adjacency_list[entity_id])
            out_degree = len(self.adjacency_list[entity_id])
            connectivity_score = (
                0.6 * (in_degree / max_in_degree) + 0.4 * (out_degree / max_out_degree)
            )

            # Level score (higher level = more important)
            level = self.entity_levels.get(entity_id, 0)
            max_level = max(self.entity_levels.values(), default=1)
            if max_level > 0:
                level_score = level / max_level
            else:
                level_score = 0.5

            # Combined score
            importance_scores[entity_id] = (
                0.5 * type_score + 0.3 * connectivity_score + 0.2 * level_score
            )

        return importance_scores
