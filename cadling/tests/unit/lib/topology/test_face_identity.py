"""Unit tests for face identity registry module.

Tests the ShapeIdentityRegistry class that provides stable shape identity
tracking using HashCode with fallback.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestShapeIdentityRegistryImports:
    """Test module imports."""

    def test_module_imports(self):
        """Test that the module imports successfully."""
        from cadling.lib.topology import ShapeIdentityRegistry
        from cadling.lib.topology.face_identity import (
            ShapeIdentityRegistry as FIRegistry,
            HAS_OCC,
        )

        assert ShapeIdentityRegistry is not None
        assert FIRegistry is ShapeIdentityRegistry


class TestShapeIdentityRegistry:
    """Test ShapeIdentityRegistry class."""

    def test_init(self):
        """Test initialization."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()

        assert registry.num_faces == 0
        assert registry.num_edges == 0
        assert registry.num_vertices == 0

    def test_get_id_with_hashcode(self):
        """Test get_id uses HashCode when available."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()

        mock_shape = MagicMock()
        mock_shape.HashCode.return_value = 12345

        shape_id = registry.get_id(mock_shape)

        assert shape_id == "12345"
        mock_shape.HashCode.assert_called_once_with(2**31 - 1)

    def test_get_id_fallback(self):
        """Test get_id falls back to hash() when HashCode unavailable."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()

        mock_shape = MagicMock(spec=[])  # No HashCode method
        # MagicMock has __hash__ by default

        shape_id = registry.get_id(mock_shape)

        # Should be a string
        assert isinstance(shape_id, str)

    def test_register_face(self):
        """Test registering a face."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()

        mock_face = MagicMock()
        mock_face.HashCode.return_value = 111

        face_id = registry.register_face(mock_face)

        assert face_id == "111"
        assert registry.num_faces == 1
        assert registry.get_face("111") is mock_face
        assert registry.get_face_index("111") == 0

    def test_register_face_duplicate(self):
        """Test registering same face twice."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()

        mock_face = MagicMock()
        mock_face.HashCode.return_value = 111

        face_id1 = registry.register_face(mock_face)
        face_id2 = registry.register_face(mock_face)

        # Should only be registered once
        assert face_id1 == face_id2
        assert registry.num_faces == 1

    def test_register_edge(self):
        """Test registering an edge."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()

        mock_edge = MagicMock()
        mock_edge.HashCode.return_value = 222

        edge_id = registry.register_edge(mock_edge)

        assert edge_id == "222"
        assert registry.num_edges == 1
        assert registry.get_edge("222") is mock_edge
        assert registry.get_edge_index("222") == 0

    def test_register_vertex(self):
        """Test registering a vertex."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()

        mock_vertex = MagicMock()
        mock_vertex.HashCode.return_value = 333

        vertex_id = registry.register_vertex(mock_vertex)

        assert vertex_id == "333"
        assert registry.num_vertices == 1
        assert registry.get_vertex("333") is mock_vertex

    def test_get_face_not_found(self):
        """Test getting non-existent face returns None."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()
        result = registry.get_face("nonexistent")
        assert result is None

    def test_get_edge_not_found(self):
        """Test getting non-existent edge returns None."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()
        result = registry.get_edge("nonexistent")
        assert result is None

    def test_get_vertex_not_found(self):
        """Test getting non-existent vertex returns None."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()
        result = registry.get_vertex("nonexistent")
        assert result is None

    def test_get_face_index_not_found(self):
        """Test getting index for non-existent face returns -1."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()
        result = registry.get_face_index("nonexistent")
        assert result == -1

    def test_get_edge_index_not_found(self):
        """Test getting index for non-existent edge returns -1."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()
        result = registry.get_edge_index("nonexistent")
        assert result == -1

    def test_get_face_by_index(self):
        """Test getting face by index."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()

        mock_face1 = MagicMock()
        mock_face1.HashCode.return_value = 111
        mock_face2 = MagicMock()
        mock_face2.HashCode.return_value = 222

        registry.register_face(mock_face1)
        registry.register_face(mock_face2)

        result = registry.get_face_by_index(0)
        assert result is not None
        assert result[0] == "111"
        assert result[1] is mock_face1

        result = registry.get_face_by_index(1)
        assert result is not None
        assert result[0] == "222"
        assert result[1] is mock_face2

        result = registry.get_face_by_index(99)
        assert result is None

    def test_get_edge_by_index(self):
        """Test getting edge by index."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()

        mock_edge = MagicMock()
        mock_edge.HashCode.return_value = 111

        registry.register_edge(mock_edge)

        result = registry.get_edge_by_index(0)
        assert result is not None
        assert result[0] == "111"

        result = registry.get_edge_by_index(99)
        assert result is None

    def test_faces_iterator(self):
        """Test iterating over registered faces."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()

        mock_face1 = MagicMock()
        mock_face1.HashCode.return_value = 111
        mock_face2 = MagicMock()
        mock_face2.HashCode.return_value = 222

        registry.register_face(mock_face1)
        registry.register_face(mock_face2)

        faces = list(registry.faces())
        assert len(faces) == 2
        assert ("111", mock_face1) in faces
        assert ("222", mock_face2) in faces

    def test_edges_iterator(self):
        """Test iterating over registered edges."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()

        mock_edge = MagicMock()
        mock_edge.HashCode.return_value = 111

        registry.register_edge(mock_edge)

        edges = list(registry.edges())
        assert len(edges) == 1
        assert edges[0] == ("111", mock_edge)

    def test_vertices_iterator(self):
        """Test iterating over registered vertices."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()

        mock_vertex = MagicMock()
        mock_vertex.HashCode.return_value = 111

        registry.register_vertex(mock_vertex)

        vertices = list(registry.vertices())
        assert len(vertices) == 1
        assert vertices[0] == ("111", mock_vertex)

    def test_clear(self):
        """Test clearing all registries."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()

        mock_face = MagicMock()
        mock_face.HashCode.return_value = 111
        mock_edge = MagicMock()
        mock_edge.HashCode.return_value = 222
        mock_vertex = MagicMock()
        mock_vertex.HashCode.return_value = 333

        registry.register_face(mock_face)
        registry.register_edge(mock_edge)
        registry.register_vertex(mock_vertex)

        assert registry.num_faces == 1
        assert registry.num_edges == 1
        assert registry.num_vertices == 1

        registry.clear()

        assert registry.num_faces == 0
        assert registry.num_edges == 0
        assert registry.num_vertices == 0

    def test_to_dict(self):
        """Test exporting registry to dictionary."""
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()

        mock_face = MagicMock()
        mock_face.HashCode.return_value = 111
        mock_edge = MagicMock()
        mock_edge.HashCode.return_value = 222

        registry.register_face(mock_face)
        registry.register_edge(mock_edge)

        result = registry.to_dict()

        assert result["num_faces"] == 1
        assert result["num_edges"] == 1
        assert result["num_vertices"] == 0
        assert "111" in result["face_ids"]
        assert "222" in result["edge_ids"]

    def test_register_all_faces_no_occ(self):
        """Test register_all_faces returns empty dict when no OCC."""
        from cadling.lib.topology.face_identity import (
            ShapeIdentityRegistry,
            HAS_OCC,
        )

        if not HAS_OCC:
            registry = ShapeIdentityRegistry()
            mock_shape = MagicMock()

            result = registry.register_all_faces(mock_shape)
            assert result == {}

    def test_register_all_edges_no_occ(self):
        """Test register_all_edges returns empty dict when no OCC."""
        from cadling.lib.topology.face_identity import (
            ShapeIdentityRegistry,
            HAS_OCC,
        )

        if not HAS_OCC:
            registry = ShapeIdentityRegistry()
            mock_shape = MagicMock()

            result = registry.register_all_edges(mock_shape)
            assert result == {}

    def test_register_all_vertices_no_occ(self):
        """Test register_all_vertices returns empty dict when no OCC."""
        from cadling.lib.topology.face_identity import (
            ShapeIdentityRegistry,
            HAS_OCC,
        )

        if not HAS_OCC:
            registry = ShapeIdentityRegistry()
            mock_shape = MagicMock()

            result = registry.register_all_vertices(mock_shape)
            assert result == {}
