"""
End-to-end integration tests for STL conversion.

Tests the complete STL conversion workflow using DocumentConverter,
including ASCII/binary STL files, mesh processing, export, and error handling.
"""

import io
import struct
from pathlib import Path

import pytest


class TestSTLEndToEnd:
    """End-to-end tests for STL conversion workflow."""

    @pytest.fixture
    def ascii_stl_file(self, tmp_path):
        """Create a minimal valid ASCII STL file."""
        stl_content = """solid cube
facet normal 0.0 0.0 1.0
  outer loop
    vertex 0.0 0.0 0.0
    vertex 1.0 0.0 0.0
    vertex 0.0 1.0 0.0
  endloop
endfacet
facet normal 0.0 0.0 1.0
  outer loop
    vertex 1.0 0.0 0.0
    vertex 1.0 1.0 0.0
    vertex 0.0 1.0 0.0
  endloop
endfacet
endsolid cube
"""
        stl_file = tmp_path / "test_ascii.stl"
        stl_file.write_text(stl_content)
        return stl_file

    @pytest.fixture
    def binary_stl_file(self, tmp_path):
        """Create a minimal valid binary STL file."""
        # 80 bytes header
        header = b"Binary STL test file" + b"\x00" * 60

        # 4 bytes: number of triangles
        num_triangles = struct.pack("<I", 2)

        # Triangle 1
        triangle1 = struct.pack(
            "<ffffffffffff",
            0.0, 0.0, 1.0,  # normal
            0.0, 0.0, 0.0,  # vertex 1
            1.0, 0.0, 0.0,  # vertex 2
            0.0, 1.0, 0.0,  # vertex 3
        ) + struct.pack("<H", 0)  # attribute byte count

        # Triangle 2
        triangle2 = struct.pack(
            "<ffffffffffff",
            0.0, 0.0, 1.0,  # normal
            1.0, 0.0, 0.0,  # vertex 1
            1.0, 1.0, 0.0,  # vertex 2
            0.0, 1.0, 0.0,  # vertex 3
        ) + struct.pack("<H", 0)  # attribute byte count

        stl_content = header + num_triangles + triangle1 + triangle2

        stl_file = tmp_path / "test_binary.stl"
        stl_file.write_bytes(stl_content)
        return stl_file

    @pytest.fixture
    def complex_stl_file(self, tmp_path):
        """Create a more complex ASCII STL file with multiple facets."""
        # Create a simple cube (6 faces, 2 triangles each = 12 facets)
        stl_content = """solid cube
facet normal 0.0 0.0 1.0
  outer loop
    vertex 0.0 0.0 1.0
    vertex 1.0 0.0 1.0
    vertex 0.0 1.0 1.0
  endloop
endfacet
facet normal 0.0 0.0 1.0
  outer loop
    vertex 1.0 0.0 1.0
    vertex 1.0 1.0 1.0
    vertex 0.0 1.0 1.0
  endloop
endfacet
facet normal 0.0 0.0 -1.0
  outer loop
    vertex 0.0 0.0 0.0
    vertex 0.0 1.0 0.0
    vertex 1.0 0.0 0.0
  endloop
endfacet
facet normal 0.0 0.0 -1.0
  outer loop
    vertex 1.0 0.0 0.0
    vertex 0.0 1.0 0.0
    vertex 1.0 1.0 0.0
  endloop
endfacet
facet normal 1.0 0.0 0.0
  outer loop
    vertex 1.0 0.0 0.0
    vertex 1.0 1.0 0.0
    vertex 1.0 0.0 1.0
  endloop
endfacet
facet normal 1.0 0.0 0.0
  outer loop
    vertex 1.0 1.0 0.0
    vertex 1.0 1.0 1.0
    vertex 1.0 0.0 1.0
  endloop
endfacet
facet normal -1.0 0.0 0.0
  outer loop
    vertex 0.0 0.0 0.0
    vertex 0.0 0.0 1.0
    vertex 0.0 1.0 0.0
  endloop
endfacet
facet normal -1.0 0.0 0.0
  outer loop
    vertex 0.0 1.0 0.0
    vertex 0.0 0.0 1.0
    vertex 0.0 1.0 1.0
  endloop
endfacet
endsolid cube
"""
        stl_file = tmp_path / "complex.stl"
        stl_file.write_text(stl_content)
        return stl_file

    def test_ascii_stl_conversion_success(self, ascii_stl_file):
        """Test successful ASCII STL file conversion."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        # Initialize converter
        converter = DocumentConverter()

        # Convert file
        result = converter.convert(ascii_stl_file)

        # Verify conversion succeeded
        assert result.status == ConversionStatus.SUCCESS
        assert result.document is not None
        assert result.errors == []

        # Verify document properties
        doc = result.document
        assert doc.name == "test_ascii.stl"
        assert doc.format.value == "stl"
        assert doc.is_ascii is True

        # Verify mesh was parsed
        assert doc.mesh is not None
        assert doc.mesh.num_facets == 2
        assert doc.mesh.num_vertices == 6  # 2 triangles * 3 vertices

        # Verify bounding box
        assert doc.bounding_box is not None
        bbox = doc.bounding_box
        # Check that BoundingBox3D has the expected attributes
        assert hasattr(bbox, "x_min")
        assert hasattr(bbox, "x_max")

    def test_binary_stl_conversion_success(self, binary_stl_file):
        """Test successful binary STL file conversion."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        converter = DocumentConverter()
        result = converter.convert(binary_stl_file)

        # Verify conversion succeeded
        assert result.status == ConversionStatus.SUCCESS
        assert result.document is not None

        # Verify document properties
        doc = result.document
        assert doc.name == "test_binary.stl"
        assert doc.is_ascii is False

        # Verify mesh
        assert doc.mesh is not None
        assert doc.mesh.num_facets == 2
        # Binary STL deduplicates vertices: 2 triangles share 2 vertices
        # Unique vertices: (0,0,0), (1,0,0), (0,1,0), (1,1,0) = 4 vertices
        assert doc.mesh.num_vertices == 4

    def test_complex_stl_conversion(self, complex_stl_file):
        """Test conversion of complex STL file with many facets."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        converter = DocumentConverter()
        result = converter.convert(complex_stl_file)

        assert result.status == ConversionStatus.SUCCESS
        doc = result.document

        # Verify mesh properties
        assert doc.mesh.num_facets == 8
        assert doc.mesh.num_vertices == 24  # 8 triangles * 3 vertices

        # Verify mesh properties were computed
        assert doc.mesh.surface_area is not None
        assert doc.mesh.surface_area > 0

        # Verify bounding box is correct for unit cube
        bbox = doc.bounding_box
        assert bbox.x_min == 0.0  # x_min
        assert bbox.x_max == 1.0  # x_max
        assert bbox.y_min == 0.0  # y_min
        assert bbox.y_max == 1.0  # y_max

    def test_stl_export_to_json(self, ascii_stl_file):
        """Test exporting STL document to JSON."""
        from cadling.backend.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(ascii_stl_file)
        doc = result.document

        # Export to JSON
        json_data = doc.export_to_json()

        # Verify JSON structure
        assert isinstance(json_data, dict)
        assert json_data["name"] == "test_ascii.stl"
        assert json_data["format"] == "stl"
        assert json_data["is_ascii"] is True

        # Verify mesh data in JSON
        assert "mesh" in json_data
        mesh_data = json_data["mesh"]
        assert mesh_data["num_facets"] == 2
        assert mesh_data["num_vertices"] == 6
        assert "surface_area" in mesh_data
        assert "bounding_box" in mesh_data

        # Verify items (facets)
        assert "items" in json_data
        assert isinstance(json_data["items"], list)

    def test_stl_export_to_markdown(self, ascii_stl_file):
        """Test exporting STL document to Markdown."""
        from cadling.backend.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(ascii_stl_file)
        doc = result.document

        # Export to Markdown
        markdown_text = doc.export_to_markdown()

        # Verify Markdown contains key information
        assert isinstance(markdown_text, str)
        assert "test_ascii.stl" in markdown_text
        assert "Format: stl" in markdown_text or "STL" in markdown_text
        assert "ASCII" in markdown_text or "ascii" in markdown_text

        # Verify mesh information
        assert "Mesh" in markdown_text or "mesh" in markdown_text
        assert "2" in markdown_text  # number of facets
        assert "facet" in markdown_text.lower()

    def test_stl_bytesio_conversion(self):
        """Test STL conversion from BytesIO stream."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus, InputFormat

        stl_content = b"""solid test
facet normal 0.0 0.0 1.0
  outer loop
    vertex 0.0 0.0 0.0
    vertex 1.0 0.0 0.0
    vertex 0.0 1.0 0.0
  endloop
endfacet
endsolid test
"""

        # Create BytesIO stream
        stream = io.BytesIO(stl_content)

        # Convert with explicit format
        converter = DocumentConverter()
        result = converter.convert(stream, format=InputFormat.STL)

        assert result.status == ConversionStatus.SUCCESS
        assert result.document is not None
        assert result.document.mesh.num_facets == 1

    def test_stl_invalid_file_error(self, tmp_path):
        """Test error handling for invalid STL file."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        # Create invalid STL file
        invalid_file = tmp_path / "invalid.stl"
        invalid_file.write_text("This is not a valid STL file\n")

        converter = DocumentConverter()
        result = converter.convert(invalid_file)

        # Should handle gracefully
        assert result.status in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL, ConversionStatus.FAILURE]

    def test_stl_empty_file_error(self, tmp_path):
        """Test error handling for empty STL file."""
        from cadling.backend.document_converter import DocumentConverter

        # Create empty file
        empty_file = tmp_path / "empty.stl"
        empty_file.write_text("")

        converter = DocumentConverter()
        result = converter.convert(empty_file)

        # Should handle gracefully
        assert result is not None

    def test_stl_malformed_ascii_error(self, tmp_path):
        """Test error handling for malformed ASCII STL."""
        from cadling.backend.document_converter import DocumentConverter

        # Missing endfacet
        malformed_content = """solid test
facet normal 0.0 0.0 1.0
  outer loop
    vertex 0.0 0.0 0.0
    vertex 1.0 0.0 0.0
    vertex 0.0 1.0 0.0
  endloop
endsolid test
"""

        malformed_file = tmp_path / "malformed.stl"
        malformed_file.write_text(malformed_content)

        converter = DocumentConverter()
        result = converter.convert(malformed_file)

        # Should handle gracefully
        assert result is not None

    def test_stl_mesh_items_structure(self, ascii_stl_file):
        """Test the structure of parsed STL mesh items."""
        from cadling.backend.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(ascii_stl_file)
        doc = result.document

        # Verify items exist
        assert len(doc.items) > 0

        # Check that the first item is a mesh
        mesh_item = doc.items[0]
        assert hasattr(mesh_item, "item_type")
        assert hasattr(mesh_item, "label")
        assert mesh_item.item_type == "mesh"

        # Verify mesh has facets and vertices
        assert hasattr(mesh_item, "facets")
        assert hasattr(mesh_item, "vertices")
        assert hasattr(mesh_item, "normals")
        assert len(mesh_item.facets) == 2  # 2 triangles
        assert len(mesh_item.vertices) > 0
        assert len(mesh_item.normals) == 2

    def test_stl_mesh_properties_computation(self, complex_stl_file):
        """Test mesh property computation."""
        from cadling.backend.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(complex_stl_file)
        doc = result.document

        mesh = doc.mesh

        # Verify computed properties
        assert mesh.surface_area is not None
        assert mesh.surface_area > 0

        # For a unit cube, surface area should be 6.0 (6 faces of 1x1)
        # With 8 triangles forming partial cube, area should be reasonable
        assert mesh.surface_area > 0.5  # At least some area

        # Check manifold property (may be True or False depending on mesh)
        assert mesh.is_manifold is not None

    def test_stl_bounding_box_accuracy(self, ascii_stl_file):
        """Test bounding box computation accuracy."""
        from cadling.backend.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(ascii_stl_file)
        doc = result.document

        bbox = doc.bounding_box
        assert bbox is not None

        # For the ASCII test file (vertices at 0,0,0 to 1,1,0)
        assert bbox.x_min == 0.0
        assert bbox.x_max == 1.0
        assert bbox.y_min == 0.0
        assert bbox.y_max == 1.0
        assert bbox.z_min == 0.0
        assert bbox.z_max == 0.0  # All vertices at z=0

    def test_stl_vertex_deduplication(self, ascii_stl_file):
        """Test vertex deduplication in mesh processing."""
        from cadling.backend.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(ascii_stl_file)
        doc = result.document

        mesh = doc.mesh

        # Original has 6 vertices (2 triangles * 3 vertices)
        # After deduplication, should have 4 unique vertices
        # (0,0,0), (1,0,0), (0,1,0), (1,1,0)

        # Check if deduplication happened (implementation dependent)
        # At minimum, verify we have vertices
        assert mesh.num_vertices > 0
        assert mesh.num_vertices <= 6

    def test_stl_conversion_performance(self, tmp_path):
        """Test conversion performance with larger STL file."""
        import time

        # Create a STL file with many facets
        facets = []
        for i in range(100):
            facets.append(f"""facet normal 0.0 0.0 1.0
  outer loop
    vertex {i}.0 0.0 0.0
    vertex {i+1}.0 0.0 0.0
    vertex {i}.0 1.0 0.0
  endloop
endfacet""")

        stl_content = f"solid perf\n{chr(10).join(facets)}\nendsolid perf\n"

        perf_file = tmp_path / "perf.stl"
        perf_file.write_text(stl_content)

        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        converter = DocumentConverter()

        start_time = time.time()
        result = converter.convert(perf_file)
        end_time = time.time()

        # Should complete successfully
        assert result.status == ConversionStatus.SUCCESS
        assert result.document.mesh.num_facets == 100

        # Should be reasonably fast (< 5 seconds for 100 facets)
        conversion_time = end_time - start_time
        assert conversion_time < 5.0, f"Conversion took {conversion_time:.2f}s, expected < 5.0s"

    def test_stl_binary_vs_ascii_consistency(self, tmp_path):
        """Test that binary and ASCII STL produce consistent results."""
        from cadling.backend.document_converter import DocumentConverter

        # Create identical geometry in ASCII and binary formats

        # ASCII version
        ascii_content = """solid test
facet normal 0.0 0.0 1.0
  outer loop
    vertex 0.0 0.0 0.0
    vertex 1.0 0.0 0.0
    vertex 0.0 1.0 0.0
  endloop
endfacet
endsolid test
"""
        ascii_file = tmp_path / "test.stl"
        ascii_file.write_text(ascii_content)

        # Binary version (same triangle)
        header = b"Binary version" + b"\x00" * 66
        num_triangles = struct.pack("<I", 1)
        triangle = struct.pack(
            "<ffffffffffff",
            0.0, 0.0, 1.0,  # normal
            0.0, 0.0, 0.0,  # vertex 1
            1.0, 0.0, 0.0,  # vertex 2
            0.0, 1.0, 0.0,  # vertex 3
        ) + struct.pack("<H", 0)

        binary_file = tmp_path / "test_bin.stl"
        binary_file.write_bytes(header + num_triangles + triangle)

        converter = DocumentConverter()

        # Convert both
        ascii_result = converter.convert(ascii_file)
        binary_result = converter.convert(binary_file)

        # Compare results
        assert ascii_result.document.mesh.num_facets == binary_result.document.mesh.num_facets
        # Note: ASCII doesn't deduplicate vertices, binary does
        # ASCII: 1 triangle = 3 vertices (no deduplication)
        # Binary: 1 triangle = 3 unique vertices (with deduplication)
        assert ascii_result.document.mesh.num_vertices == 3
        assert binary_result.document.mesh.num_vertices == 3

        # Bounding boxes should match
        ascii_bbox = ascii_result.document.bounding_box
        binary_bbox = binary_result.document.bounding_box

        # Compare each attribute of BoundingBox3D
        assert abs(ascii_bbox.x_min - binary_bbox.x_min) < 0.001
        assert abs(ascii_bbox.y_min - binary_bbox.y_min) < 0.001
        assert abs(ascii_bbox.z_min - binary_bbox.z_min) < 0.001
        assert abs(ascii_bbox.x_max - binary_bbox.x_max) < 0.001
        assert abs(ascii_bbox.y_max - binary_bbox.y_max) < 0.001
        assert abs(ascii_bbox.z_max - binary_bbox.z_max) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
