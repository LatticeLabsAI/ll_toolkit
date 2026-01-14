"""
Integration tests for STL pipeline.

Tests the full STL conversion workflow from file to document.
"""

import io
import struct
import pytest


class TestSTLPipeline:
    """Integration tests for STL pipeline."""

    def test_stl_backend_ascii(self):
        """Test STL backend with ASCII STL."""
        from cadling.backend.stl import STLBackend
        from cadling.datamodel.base_models import CADInputDocument, InputFormat
        from pathlib import Path

        # Create minimal ASCII STL
        stl_content = b"""solid test
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
endsolid test
"""

        stream = io.BytesIO(stl_content)

        # Create input document
        input_doc = CADInputDocument(
            file=Path("test.stl"),
            format=InputFormat.STL,
            document_hash="test_hash",
        )

        # Create backend
        backend = STLBackend(input_doc, stream)

        # Test validity
        assert backend.is_valid()

        # Convert
        document = backend.convert()

        assert document.name == "test.stl"
        assert document.is_ascii is True
        assert document.mesh is not None
        assert document.mesh.num_facets == 2
        assert document.mesh.num_vertices == 6  # Before deduplication

    def test_stl_backend_binary(self):
        """Test STL backend with binary STL."""
        from cadling.backend.stl import STLBackend
        from cadling.datamodel.base_models import CADInputDocument, InputFormat
        from pathlib import Path

        # Create minimal binary STL
        # 80 bytes header + 4 bytes triangle count + 50 bytes per triangle
        header = b"Binary STL test file" + b"\x00" * 60
        triangle_count = struct.pack("<I", 1)  # 1 triangle

        # Triangle: normal (12 bytes) + 3 vertices (36 bytes) + attributes (2 bytes)
        triangle = struct.pack(
            "<ffffffffffff",
            0.0,
            0.0,
            1.0,  # normal
            0.0,
            0.0,
            0.0,  # vertex 1
            1.0,
            0.0,
            0.0,  # vertex 2
            0.0,
            1.0,
            0.0,  # vertex 3
        )
        attributes = struct.pack("<H", 0)

        stl_content = header + triangle_count + triangle + attributes

        stream = io.BytesIO(stl_content)

        # Create input document
        input_doc = CADInputDocument(
            file=Path("test.stl"),
            format=InputFormat.STL,
            document_hash="test_hash",
        )

        # Create backend
        backend = STLBackend(input_doc, stream)

        # Test validity
        assert backend.is_valid()

        # Convert
        document = backend.convert()

        assert document.name == "test.stl"
        assert document.is_ascii is False
        assert document.mesh is not None
        assert document.mesh.num_facets == 1
        assert document.mesh.num_vertices == 3

    def test_stl_pipeline_integration(self):
        """Test full STL pipeline integration."""
        from cadling.pipeline.stl_pipeline import STLPipeline
        from cadling.datamodel.base_models import CADInputDocument, InputFormat
        from cadling.backend.stl import STLBackend
        from pathlib import Path

        # Create ASCII STL
        stl_content = b"""solid cube
facet normal 0.0 0.0 1.0
  outer loop
    vertex 0.0 0.0 0.0
    vertex 1.0 0.0 0.0
    vertex 1.0 1.0 0.0
  endloop
endfacet
facet normal 0.0 0.0 1.0
  outer loop
    vertex 0.0 0.0 0.0
    vertex 1.0 1.0 0.0
    vertex 0.0 1.0 0.0
  endloop
endfacet
endsolid cube
"""

        stream = io.BytesIO(stl_content)

        # Create input document with backend
        input_doc = CADInputDocument(
            file=Path("test.stl"),
            format=InputFormat.STL,
            document_hash="test_hash",
        )
        input_doc._backend = STLBackend(input_doc, stream)

        # Create pipeline
        options = STLPipeline.get_default_options()
        pipeline = STLPipeline(options)

        # Execute
        result = pipeline.execute(input_doc)

        assert result.status.value in ["success", "partial"]
        assert result.document is not None
        assert result.document.mesh is not None
        assert result.document.mesh.num_facets == 2
        assert result.document.bounding_box is not None

    def test_mesh_properties(self):
        """Test mesh property computation."""
        from cadling.backend.stl import STLBackend
        from cadling.datamodel.base_models import CADInputDocument, InputFormat
        from pathlib import Path

        # Create closed cube (6 faces, 2 triangles each = 12 triangles)
        # For simplicity, just test with 2 triangles forming a square
        stl_content = b"""solid square
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
endsolid square
"""

        stream = io.BytesIO(stl_content)

        input_doc = CADInputDocument(
            file=Path("test.stl"),
            format=InputFormat.STL,
            document_hash="test_hash",
        )

        backend = STLBackend(input_doc, stream)
        document = backend.convert()

        # Check computed properties
        assert document.mesh.surface_area is not None
        assert document.mesh.surface_area > 0
        # Manifold check might fail for open mesh
        assert document.mesh.is_manifold is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
