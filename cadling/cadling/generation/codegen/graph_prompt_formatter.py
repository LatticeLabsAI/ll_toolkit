"""Graph to Prompt formatter for LLM-based CAD code generation.

This module converts decoded graph features (from GNN or tokenized B-Rep) into
structured prompts for LLM-based CadQuery code generation.

The formatter translates:
- Node features (face surface types, curvatures, areas) → geometry description
- Edge index (adjacency) → topology relationships
- Edge features (convexity, angles) → joining instructions

Example:
    from cadling.generation.codegen.graph_prompt_formatter import GraphToPromptFormatter

    formatter = GraphToPromptFormatter()

    prompt = formatter.format_for_cadquery(
        node_features=node_features,  # [N, 48] from decoder
        edge_index=edge_index,        # [2, M]
        adjacency=adjacency_dict,
    )

    # Use with CadQueryGenerator
    generator = CadQueryGenerator()
    script = generator.generate(prompt)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

_log = logging.getLogger(__name__)

# Entity type names for decoding one-hot vectors
SURFACE_TYPES = [
    "PLANE",
    "CYLINDER",
    "CONE",
    "SPHERE",
    "TORUS",
    "BEZIER",
    "BSPLINE",
    "REVOLUTION",
    "EXTRUSION",
    "OTHER",
]

STEP_ENTITY_TYPES = [
    "ADVANCED_FACE",
    "FACE_BOUND",
    "CYLINDRICAL_SURFACE",
    "PLANE",
    "CONICAL_SURFACE",
    "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "B_SPLINE_SURFACE",
    "EDGE_LOOP",
    "ORIENTED_EDGE",
    "EDGE_CURVE",
    "LINE",
    "CIRCLE",
    "VERTEX_POINT",
    "CARTESIAN_POINT",
    "OTHER",
]

CURVE_TYPES = [
    "LINE",
    "CIRCLE",
    "ELLIPSE",
    "BSPLINE",
    "PARABOLA",
    "OTHER",
]


class GraphToPromptFormatter:
    """Format decoded graph features as LLM prompt for CadQuery generation.

    Converts node features, edge indices, and adjacency information into
    a structured natural language description that an LLM can use to
    generate CadQuery code.

    Attributes:
        include_coordinates: Whether to include exact coordinates in prompt
        include_dimensions: Whether to include computed dimensions
        max_faces_detailed: Maximum faces to describe in detail
        decimal_places: Precision for numeric values
    """

    def __init__(
        self,
        include_coordinates: bool = True,
        include_dimensions: bool = True,
        max_faces_detailed: int = 20,
        decimal_places: int = 3,
    ):
        """Initialize graph to prompt formatter.

        Args:
            include_coordinates: Include centroid/bbox coordinates
            include_dimensions: Include computed dimensions
            max_faces_detailed: Max faces to describe individually
            decimal_places: Numeric precision
        """
        self.include_coordinates = include_coordinates
        self.include_dimensions = include_dimensions
        self.max_faces_detailed = max_faces_detailed
        self.decimal_places = decimal_places

    def format_for_cadquery(
        self,
        node_features: np.ndarray,
        edge_index: Optional[np.ndarray] = None,
        adjacency: Optional[Dict[int, List[int]]] = None,
        edge_features: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate prompt describing geometry for LLM code generation.

        Args:
            node_features: Face features [N, feature_dim]
            edge_index: Adjacency in COO format [2, M]
            adjacency: Adjacency as dict (alternative to edge_index)
            edge_features: Optional edge features [M, edge_dim]
            metadata: Optional metadata dict (volume, bbox, etc.)

        Returns:
            Structured prompt string for CadQuery generation
        """
        num_faces = node_features.shape[0]

        # Build adjacency dict if only edge_index provided
        if adjacency is None and edge_index is not None:
            adjacency = self._edge_index_to_adjacency(edge_index)

        sections = []

        # Header
        sections.append(self._format_header(num_faces, metadata))

        # Entity summary
        sections.append(self._format_entity_summary(node_features))

        # Face details
        sections.append(self._format_face_details(node_features))

        # Topology relationships
        if adjacency:
            sections.append(self._format_adjacency(adjacency, node_features))

        # Edge features if available
        if edge_features is not None and edge_index is not None:
            sections.append(self._format_edge_features(edge_features, edge_index, node_features))

        # Generation instructions
        sections.append(self._format_instructions())

        return "\n\n".join(sections)

    def _format_header(
        self,
        num_faces: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format header section with overview."""
        lines = [
            "You are generating CadQuery code to reconstruct a 3D CAD model.",
            "",
            "## Overview",
            f"- Number of faces: {num_faces}",
        ]

        if metadata:
            if "volume" in metadata:
                lines.append(f"- Approximate volume: {metadata['volume']:.{self.decimal_places}f}")
            if "surface_area" in metadata:
                lines.append(f"- Surface area: {metadata['surface_area']:.{self.decimal_places}f}")
            if "bbox_dimensions" in metadata:
                dims = metadata["bbox_dimensions"]
                lines.append(
                    f"- Bounding box dimensions: "
                    f"{dims[0]:.{self.decimal_places}f} x "
                    f"{dims[1]:.{self.decimal_places}f} x "
                    f"{dims[2]:.{self.decimal_places}f}"
                )

        return "\n".join(lines)

    def _format_entity_summary(self, node_features: np.ndarray) -> str:
        """Format summary of entity types."""
        num_faces = node_features.shape[0]
        feature_dim = node_features.shape[1]

        # Count surface types
        type_counts: Dict[str, int] = {}

        for i in range(num_faces):
            surface_type = self._decode_surface_type(node_features[i])
            type_counts[surface_type] = type_counts.get(surface_type, 0) + 1

        lines = ["## Entity Summary"]

        # Sort by count descending
        sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])

        for surface_type, count in sorted_types:
            percentage = count / num_faces * 100
            lines.append(f"- {surface_type}: {count} faces ({percentage:.1f}%)")

        return "\n".join(lines)

    def _format_face_details(self, node_features: np.ndarray) -> str:
        """Format detailed face descriptions."""
        num_faces = node_features.shape[0]
        lines = ["## Face Details"]

        # Limit to max_faces_detailed
        faces_to_describe = min(num_faces, self.max_faces_detailed)

        for i in range(faces_to_describe):
            face_desc = self._describe_face(i, node_features[i])
            lines.append(face_desc)

        if num_faces > self.max_faces_detailed:
            lines.append(f"\n... and {num_faces - self.max_faces_detailed} more faces with similar patterns")

        return "\n".join(lines)

    def _describe_face(self, index: int, features: np.ndarray) -> str:
        """Generate description for a single face."""
        surface_type = self._decode_surface_type(features)

        parts = [f"Face {index}: {surface_type}"]

        feature_dim = len(features)

        # Extract area if available (index 10 in standard 24-dim format)
        if feature_dim >= 11:
            area = features[10]
            if area > 0:
                parts.append(f"area={area:.{self.decimal_places}f}")

        # Extract curvature if available (indices 11-12)
        if feature_dim >= 13:
            gauss_curv = features[11]
            mean_curv = features[12]
            if abs(gauss_curv) > 1e-6 or abs(mean_curv) > 1e-6:
                parts.append(f"K={gauss_curv:.4f}, H={mean_curv:.4f}")

        # Extract centroid if available (indices 16-18)
        if self.include_coordinates and feature_dim >= 19:
            centroid = features[16:19]
            if np.any(centroid != 0):
                parts.append(
                    f"center=({centroid[0]:.{self.decimal_places}f}, "
                    f"{centroid[1]:.{self.decimal_places}f}, "
                    f"{centroid[2]:.{self.decimal_places}f})"
                )

        # Extract bbox dimensions if available (indices 19-21)
        if self.include_dimensions and feature_dim >= 22:
            dims = features[19:22]
            if np.any(dims > 0):
                parts.append(
                    f"size=({dims[0]:.{self.decimal_places}f}, "
                    f"{dims[1]:.{self.decimal_places}f}, "
                    f"{dims[2]:.{self.decimal_places}f})"
                )

        return "- " + ", ".join(parts)

    def _decode_surface_type(self, features: np.ndarray) -> str:
        """Decode surface type from one-hot encoded features."""
        # First 10 features are typically surface type one-hot
        if len(features) >= 10:
            type_probs = features[:10]
            type_idx = int(np.argmax(type_probs))
            if type_probs[type_idx] > 0.5 and type_idx < len(SURFACE_TYPES):
                return SURFACE_TYPES[type_idx]

        return "OTHER"

    def _format_adjacency(
        self,
        adjacency: Dict[int, List[int]],
        node_features: np.ndarray,
    ) -> str:
        """Format topology relationships."""
        lines = ["## Topology Relationships"]

        if not adjacency:
            lines.append("- No adjacency information available")
            return "\n".join(lines)

        # Group faces by surface type
        face_types: Dict[str, List[int]] = {}
        for i in range(node_features.shape[0]):
            surface_type = self._decode_surface_type(node_features[i])
            if surface_type not in face_types:
                face_types[surface_type] = []
            face_types[surface_type].append(i)

        # Describe connectivity patterns
        lines.append("Face adjacency patterns:")

        # For each face type, describe what it connects to
        for surface_type, face_indices in face_types.items():
            connected_types: Dict[str, int] = {}

            for face_idx in face_indices:
                neighbors = adjacency.get(face_idx, [])
                for neighbor_idx in neighbors:
                    if neighbor_idx < node_features.shape[0]:
                        neighbor_type = self._decode_surface_type(node_features[neighbor_idx])
                        connected_types[neighbor_type] = connected_types.get(neighbor_type, 0) + 1

            if connected_types:
                connections = ", ".join(
                    f"{count} {t}" for t, count in sorted(connected_types.items(), key=lambda x: -x[1])
                )
                lines.append(f"- {surface_type} faces connect to: {connections}")

        return "\n".join(lines)

    def _format_edge_features(
        self,
        edge_features: np.ndarray,
        edge_index: np.ndarray,
        node_features: np.ndarray,
    ) -> str:
        """Format edge features (convexity, angles)."""
        lines = ["## Edge Characteristics"]

        num_edges = edge_features.shape[0]

        # Summarize edge types
        convex_count = 0
        concave_count = 0
        tangent_count = 0

        # Assuming edge_features has convexity in last dim
        edge_dim = edge_features.shape[1]

        for i in range(num_edges):
            # Check for convexity (often in last feature)
            if edge_dim >= 8:
                convexity = edge_features[i, 7] if edge_dim > 7 else 0.5

                if convexity > 0.7:
                    convex_count += 1
                elif convexity < 0.3:
                    concave_count += 1
                else:
                    tangent_count += 1

        lines.append(f"- Convex edges: {convex_count}")
        lines.append(f"- Concave edges: {concave_count}")
        lines.append(f"- Tangent/smooth edges: {tangent_count}")

        return "\n".join(lines)

    def _format_instructions(self) -> str:
        """Format CadQuery generation instructions."""
        return """## CadQuery Generation Instructions

Generate CadQuery code that creates this geometry. Follow these guidelines:

1. **Start with a base shape**: Use the most appropriate primitive (box, cylinder, sphere) based on the dominant surface types.

2. **Apply operations in order**:
   - Extrusions for planar faces
   - Revolves for cylindrical/conical surfaces
   - Fillets for smooth edges (concave edges)
   - Chamfers for sharp edges (convex edges)

3. **Use the centroid and dimension information** to position and size features correctly.

4. **Combine features using boolean operations** (union, cut, intersect) based on the topology.

5. **The final result must be assigned to a variable named `result`**.

Generate complete, executable CadQuery Python code."""

    def _edge_index_to_adjacency(self, edge_index: np.ndarray) -> Dict[int, List[int]]:
        """Convert COO edge index to adjacency dictionary."""
        adjacency: Dict[int, List[int]] = {}

        if edge_index.shape[0] != 2:
            return adjacency

        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i])
            dst = int(edge_index[1, i])

            if src not in adjacency:
                adjacency[src] = []
            if dst not in adjacency[src]:
                adjacency[src].append(dst)

        return adjacency

    def format_from_step_features(
        self,
        entity_features: List[Dict[str, Any]],
    ) -> str:
        """Format prompt from STEP entity features (alternative input).

        Args:
            entity_features: List of dicts with entity_type, properties, references

        Returns:
            Structured prompt string
        """
        sections = []

        # Header
        num_entities = len(entity_features)
        sections.append(
            "You are generating CadQuery code to reconstruct a 3D CAD model.\n\n"
            f"## Overview\n"
            f"- Number of STEP entities: {num_entities}"
        )

        # Entity summary
        type_counts: Dict[str, int] = {}
        for entity in entity_features:
            entity_type = entity.get("entity_type", "UNKNOWN")
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

        lines = ["## Entity Summary"]
        for entity_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {entity_type}: {count}")
        sections.append("\n".join(lines))

        # Entity details (limited)
        face_entities = [e for e in entity_features if "FACE" in e.get("entity_type", "")]
        if face_entities:
            lines = ["## Face Entities"]
            for i, entity in enumerate(face_entities[:self.max_faces_detailed]):
                entity_type = entity.get("entity_type", "UNKNOWN")
                props = entity.get("properties", {})
                desc = f"- Face {i}: {entity_type}"
                if "surface_type" in props:
                    desc += f", surface={props['surface_type']}"
                if "area" in props:
                    desc += f", area={props['area']:.3f}"
                lines.append(desc)
            sections.append("\n".join(lines))

        # Instructions
        sections.append(self._format_instructions())

        return "\n\n".join(sections)
