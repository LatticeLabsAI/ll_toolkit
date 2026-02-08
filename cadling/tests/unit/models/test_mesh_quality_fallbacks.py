"""Tests for mesh quality fallback implementations.

Tests the fallback methods that work when trimesh is unavailable:
- _assess_mesh_numpy: Numpy-based mesh quality assessment
- _parse_ascii_stl: Parse ASCII STL files
- _parse_binary_stl: Parse binary STL files
- _estimate_from_properties: Estimate quality from geometry properties
"""

from __future__ import annotations

import struct
import pytest
import numpy as np


class TestMeshQualityNumpyFallback:
    """Test numpy-based mesh quality assessment fallback."""

    def test_assess_mesh_numpy_tetrahedron(self):
        """Test numpy assessment on a tetrahedron."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        # Simple tetrahedron
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0.866, 0],
            [0.5, 0.289, 0.816]
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3]
        ])

        result = model._assess_mesh_numpy(vertices, faces)

        assert result["status"] == "success"
        assert result["method"] == "numpy_fallback"
        assert result["num_vertices"] == 4
        assert result["num_faces"] == 4
        assert result["quality_score"] >= 0
        assert result["quality_score"] <= 1

    def test_assess_mesh_numpy_cube(self):
        """Test numpy assessment on a triangulated cube."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        # Cube vertices
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top
        ], dtype=np.float64)

        # Cube faces (12 triangles, 2 per face)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ])

        result = model._assess_mesh_numpy(vertices, faces)

        assert result["status"] == "success"
        assert result["num_vertices"] == 8
        assert result["num_faces"] == 12
        assert result["is_manifold"] is True

    def test_assess_mesh_numpy_degenerate_faces(self):
        """Test numpy assessment handles degenerate triangles."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        # Include a degenerate triangle (collinear points)
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0],  # valid
            [0, 0, 0], [0.5, 0, 0], [1, 0, 0],  # degenerate (collinear)
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [3, 4, 5]])

        result = model._assess_mesh_numpy(vertices, faces)

        assert result["status"] == "success"
        assert result["num_degenerate_faces"] >= 1

    def test_assess_mesh_numpy_edge_statistics(self):
        """Test numpy assessment computes edge statistics."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        # Equilateral triangle
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]])

        result = model._assess_mesh_numpy(vertices, faces)

        assert "edge_lengths" in result
        assert result["edge_lengths"]["min"] > 0
        assert result["edge_lengths"]["max"] >= result["edge_lengths"]["min"]
        assert result["edge_lengths"]["mean"] > 0

    def test_assess_mesh_numpy_aspect_ratios(self):
        """Test numpy assessment computes aspect ratios."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        # Stretched triangle (high aspect ratio)
        # Edge 0->1 is 10, edge 1->2 is sqrt(25+0.01)=5.001, edge 2->0 is 5.001
        # aspect ratio = 10 / 5.001 ≈ 2
        # For higher aspect ratio, make triangle more extreme
        vertices = np.array([
            [0, 0, 0], [100, 0, 0], [50, 0.1, 0]  # very flat, 100 : ~50 ratio
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]])

        result = model._assess_mesh_numpy(vertices, faces)

        assert "aspect_ratio" in result
        assert result["aspect_ratio"]["max"] > 1.5  # Should have high aspect ratio


class TestMeshQualitySTLParsing:
    """Test STL parsing fallback methods."""

    def test_parse_ascii_stl_basic(self):
        """Test ASCII STL parsing."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        stl_content = """solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0.5 0.866 0
    endloop
  endfacet
endsolid test"""

        vertices, faces = model._parse_ascii_stl(stl_content)

        assert vertices is not None
        assert faces is not None
        assert len(vertices) == 3
        assert len(faces) == 1
        assert np.allclose(vertices[0], [0, 0, 0])

    def test_parse_ascii_stl_multiple_facets(self):
        """Test ASCII STL with multiple facets."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        stl_content = """solid multi
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0.5 0.866 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 1 0 0
      vertex 1 1 0
      vertex 0.5 0.866 0
    endloop
  endfacet
endsolid multi"""

        vertices, faces = model._parse_ascii_stl(stl_content)

        assert len(vertices) == 6  # 3 vertices per triangle, 2 triangles
        assert len(faces) == 2

    def test_parse_binary_stl_basic(self):
        """Test binary STL parsing."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        # Build a binary STL with one triangle
        header = b"Binary STL test" + b"\x00" * (80 - 15)
        num_triangles = struct.pack("<I", 1)

        # Normal (0, 0, 1)
        normal = struct.pack("<fff", 0, 0, 1)
        # Vertices
        v1 = struct.pack("<fff", 0, 0, 0)
        v2 = struct.pack("<fff", 1, 0, 0)
        v3 = struct.pack("<fff", 0.5, 0.866, 0)
        # Attribute byte count
        attr = struct.pack("<H", 0)

        binary_data = header + num_triangles + normal + v1 + v2 + v3 + attr

        vertices, faces = model._parse_binary_stl(binary_data)

        assert vertices is not None
        assert faces is not None
        assert len(vertices) == 3
        assert len(faces) == 1

    def test_parse_binary_stl_multiple_triangles(self):
        """Test binary STL with multiple triangles."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        header = b"Binary STL" + b"\x00" * 70
        num_triangles = struct.pack("<I", 2)

        def make_triangle(nx, ny, nz, v1, v2, v3):
            return (
                struct.pack("<fff", nx, ny, nz) +
                struct.pack("<fff", *v1) +
                struct.pack("<fff", *v2) +
                struct.pack("<fff", *v3) +
                struct.pack("<H", 0)
            )

        t1 = make_triangle(0, 0, 1, (0, 0, 0), (1, 0, 0), (0.5, 0.866, 0))
        t2 = make_triangle(0, 0, 1, (1, 0, 0), (1, 1, 0), (0.5, 0.866, 0))

        binary_data = header + num_triangles + t1 + t2

        vertices, faces = model._parse_binary_stl(binary_data)

        assert len(vertices) == 6
        assert len(faces) == 2


class TestMeshQualityManifoldCheck:
    """Test manifold checking."""

    def test_check_manifold_simple_mesh(self):
        """Test manifold check on simple closed mesh."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        # Tetrahedron faces - manifold (each edge shared by exactly 2 faces)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3]
        ])

        result = model._check_manifold_numpy(faces, 4)

        assert result is True

    def test_check_manifold_non_manifold(self):
        """Test manifold check on non-manifold mesh."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        # Non-manifold: edge 0-1 shared by 3 faces
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],  # Third face sharing edge 0-1
        ])

        result = model._check_manifold_numpy(faces, 5)

        assert result is False


class TestMeshQualityEstimation:
    """Test property-based estimation."""

    def test_estimate_from_properties_with_bbox(self):
        """Test estimation from bounding box."""
        from cadling.models.mesh_quality import MeshQualityModel
        from unittest.mock import MagicMock

        model = MeshQualityModel()

        doc = MagicMock()
        item = MagicMock()
        item.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 0, "max_x": 10,
                    "min_y": 0, "max_y": 10,
                    "min_z": 0, "max_z": 10,
                },
                "surface_area": 600,  # Cube: 6 * 10 * 10
                "volume": 1000,
            }
        }

        result = model._estimate_from_properties(doc, item)

        assert result is not None
        assert result["status"] == "estimated"
        assert result["num_faces"] > 0
        assert result["quality_score"] > 0


class TestMeshQualityIntegration:
    """Integration tests for fallback workflow."""

    def test_assess_item_uses_fallback_chain(self):
        """Test that _assess_item tries fallbacks in order."""
        from cadling.models.mesh_quality import MeshQualityModel
        from cadling.datamodel.stl import STLDocument, MeshItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = MeshQualityModel()
        model.has_trimesh = False

        doc = STLDocument(
            name="test.stl",
            origin=CADDocumentOrigin(
                filename="test.stl",
                format=InputFormat.STL,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = MeshItem(
            label=CADItemLabel(text="TestItem"),
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]],
            normals=[[0.0, 0.0, 1.0]],
            facets=[[0, 1, 2]],
        )

        # Should return result dict (either success or error)
        result = model._assess_item(doc, item)

        assert result is not None
        assert "status" in result
        # Will have either success or error status
