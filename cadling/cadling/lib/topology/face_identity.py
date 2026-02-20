"""Stable shape identity registry for B-Rep processing.

This module provides stable identity tracking for OCC shapes (faces, edges, vertices)
using the HashCode method with fallback to Python's hash() for compatibility across
pythonocc versions.

The registry maintains a mapping from shape IDs to shape objects, enabling
consistent referencing across processing stages.

Example:
    from cadling.lib.topology.face_identity import ShapeIdentityRegistry

    registry = ShapeIdentityRegistry()

    # Register all faces from a shape
    face_map = registry.register_all_faces(shape)

    # Get stable ID for any shape
    face_id = registry.get_id(face)

    # Look up shape by ID
    face = registry.get_face(face_id)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex

_log = logging.getLogger(__name__)

# Availability flag
HAS_OCC = False

try:
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core import topods

    HAS_OCC = True
except ImportError:
    _log.debug("pythonocc-core not available. ShapeIdentityRegistry will have limited functionality.")


class ShapeIdentityRegistry:
    """Stable shape identity using HashCode with fallback.

    Provides consistent shape identification across processing stages by
    maintaining registries of faces, edges, and vertices indexed by stable IDs.

    The ID is computed using OCC's HashCode method when available, falling
    back to Python's hash() for compatibility across pythonocc versions.

    Attributes:
        _faces: Dictionary mapping face_id -> TopoDS_Face
        _edges: Dictionary mapping edge_id -> TopoDS_Edge
        _vertices: Dictionary mapping vertex_id -> TopoDS_Vertex
        _face_indices: Dictionary mapping face_id -> sequential index
        _edge_indices: Dictionary mapping edge_id -> sequential index
    """

    def __init__(self):
        """Initialize shape identity registry."""
        self._faces: Dict[str, Any] = {}
        self._edges: Dict[str, Any] = {}
        self._vertices: Dict[str, Any] = {}
        self._face_indices: Dict[str, int] = {}
        self._edge_indices: Dict[str, int] = {}
        self._vertex_indices: Dict[str, int] = {}

    def get_id(self, shape: Any) -> str:
        """Get stable ID for a shape.

        Uses OCCT's HashCode method if available, falling back to Python's
        built-in hash() for compatibility across pythonocc versions.

        Args:
            shape: Any TopoDS_Shape object (Face, Edge, Vertex, etc.)

        Returns:
            String representation of the shape's hash code
        """
        try:
            return str(shape.HashCode(2**31 - 1))
        except AttributeError:
            # Fallback for pythonocc versions where HashCode is not exposed
            return str(hash(shape))

    def register_face(self, face: "TopoDS_Face") -> str:
        """Register a face and return its stable ID.

        Args:
            face: TopoDS_Face to register

        Returns:
            Stable face ID string
        """
        face_id = self.get_id(face)

        if face_id not in self._faces:
            self._faces[face_id] = face
            self._face_indices[face_id] = len(self._face_indices)

        return face_id

    def register_edge(self, edge: "TopoDS_Edge") -> str:
        """Register an edge and return its stable ID.

        Args:
            edge: TopoDS_Edge to register

        Returns:
            Stable edge ID string
        """
        edge_id = self.get_id(edge)

        if edge_id not in self._edges:
            self._edges[edge_id] = edge
            self._edge_indices[edge_id] = len(self._edge_indices)

        return edge_id

    def register_vertex(self, vertex: "TopoDS_Vertex") -> str:
        """Register a vertex and return its stable ID.

        Args:
            vertex: TopoDS_Vertex to register

        Returns:
            Stable vertex ID string
        """
        vertex_id = self.get_id(vertex)

        if vertex_id not in self._vertices:
            self._vertices[vertex_id] = vertex
            self._vertex_indices[vertex_id] = len(self._vertex_indices)

        return vertex_id

    def register_all_faces(self, shape: "TopoDS_Shape") -> Dict[str, Any]:
        """Register all faces from a shape.

        Args:
            shape: TopoDS_Shape containing faces

        Returns:
            Dictionary mapping face_id -> TopoDS_Face
        """
        if not HAS_OCC:
            return {}

        try:
            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                face = topods.Face(explorer.Current())
                self.register_face(face)
                explorer.Next()

            _log.debug(f"Registered {len(self._faces)} faces")
            return self._faces.copy()

        except Exception as e:
            _log.warning(f"Failed to register faces: {e}")
            return {}

    def register_all_edges(self, shape: "TopoDS_Shape") -> Dict[str, Any]:
        """Register all edges from a shape.

        Args:
            shape: TopoDS_Shape containing edges

        Returns:
            Dictionary mapping edge_id -> TopoDS_Edge
        """
        if not HAS_OCC:
            return {}

        try:
            explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            while explorer.More():
                edge = topods.Edge(explorer.Current())
                self.register_edge(edge)
                explorer.Next()

            _log.debug(f"Registered {len(self._edges)} edges")
            return self._edges.copy()

        except Exception as e:
            _log.warning(f"Failed to register edges: {e}")
            return {}

    def register_all_vertices(self, shape: "TopoDS_Shape") -> Dict[str, Any]:
        """Register all vertices from a shape.

        Args:
            shape: TopoDS_Shape containing vertices

        Returns:
            Dictionary mapping vertex_id -> TopoDS_Vertex
        """
        if not HAS_OCC:
            return {}

        try:
            explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
            while explorer.More():
                vertex = topods.Vertex(explorer.Current())
                self.register_vertex(vertex)
                explorer.Next()

            _log.debug(f"Registered {len(self._vertices)} vertices")
            return self._vertices.copy()

        except Exception as e:
            _log.warning(f"Failed to register vertices: {e}")
            return {}

    def register_all(self, shape: "TopoDS_Shape") -> None:
        """Register all faces, edges, and vertices from a shape.

        Args:
            shape: TopoDS_Shape to register all elements from
        """
        self.register_all_faces(shape)
        self.register_all_edges(shape)
        self.register_all_vertices(shape)

    def get_face(self, face_id: str) -> Optional[Any]:
        """Get face by ID.

        Args:
            face_id: Stable face ID

        Returns:
            TopoDS_Face or None if not registered
        """
        return self._faces.get(face_id)

    def get_edge(self, edge_id: str) -> Optional[Any]:
        """Get edge by ID.

        Args:
            edge_id: Stable edge ID

        Returns:
            TopoDS_Edge or None if not registered
        """
        return self._edges.get(edge_id)

    def get_vertex(self, vertex_id: str) -> Optional[Any]:
        """Get vertex by ID.

        Args:
            vertex_id: Stable vertex ID

        Returns:
            TopoDS_Vertex or None if not registered
        """
        return self._vertices.get(vertex_id)

    def get_face_index(self, face_id: str) -> int:
        """Get sequential index for a face.

        Args:
            face_id: Stable face ID

        Returns:
            Sequential index (0-based), or -1 if not registered
        """
        return self._face_indices.get(face_id, -1)

    def get_edge_index(self, edge_id: str) -> int:
        """Get sequential index for an edge.

        Args:
            edge_id: Stable edge ID

        Returns:
            Sequential index (0-based), or -1 if not registered
        """
        return self._edge_indices.get(edge_id, -1)

    def get_face_by_index(self, index: int) -> Optional[Tuple[str, Any]]:
        """Get face by sequential index.

        Args:
            index: Sequential index

        Returns:
            Tuple of (face_id, face) or None if not found
        """
        for face_id, idx in self._face_indices.items():
            if idx == index:
                return (face_id, self._faces[face_id])
        return None

    def get_edge_by_index(self, index: int) -> Optional[Tuple[str, Any]]:
        """Get edge by sequential index.

        Args:
            index: Sequential index

        Returns:
            Tuple of (edge_id, edge) or None if not found
        """
        for edge_id, idx in self._edge_indices.items():
            if idx == index:
                return (edge_id, self._edges[edge_id])
        return None

    def faces(self) -> Iterator[Tuple[str, Any]]:
        """Iterate over registered faces.

        Yields:
            Tuples of (face_id, face)
        """
        yield from self._faces.items()

    def edges(self) -> Iterator[Tuple[str, Any]]:
        """Iterate over registered edges.

        Yields:
            Tuples of (edge_id, edge)
        """
        yield from self._edges.items()

    def vertices(self) -> Iterator[Tuple[str, Any]]:
        """Iterate over registered vertices.

        Yields:
            Tuples of (vertex_id, vertex)
        """
        yield from self._vertices.items()

    @property
    def num_faces(self) -> int:
        """Get number of registered faces."""
        return len(self._faces)

    @property
    def num_edges(self) -> int:
        """Get number of registered edges."""
        return len(self._edges)

    @property
    def num_vertices(self) -> int:
        """Get number of registered vertices."""
        return len(self._vertices)

    def clear(self) -> None:
        """Clear all registries."""
        self._faces.clear()
        self._edges.clear()
        self._vertices.clear()
        self._face_indices.clear()
        self._edge_indices.clear()
        self._vertex_indices.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export registry state to dictionary.

        Returns:
            Dictionary with face/edge/vertex counts and ID lists
        """
        return {
            "num_faces": self.num_faces,
            "num_edges": self.num_edges,
            "num_vertices": self.num_vertices,
            "face_ids": list(self._faces.keys()),
            "edge_ids": list(self._edges.keys()),
            "vertex_ids": list(self._vertices.keys()),
        }
