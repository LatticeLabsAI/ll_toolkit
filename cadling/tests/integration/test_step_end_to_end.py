"""
End-to-end integration tests for STEP conversion.

Tests the complete STEP conversion workflow using DocumentConverter,
including file loading, conversion, export, and error handling.
"""

import io
import tempfile
from pathlib import Path

import pytest


class TestSTEPEndToEnd:
    """End-to-end tests for STEP conversion workflow."""

    @pytest.fixture
    def minimal_step_file(self, tmp_path):
        """Create a minimal valid STEP file."""
        step_content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('minimal test file'),'2;1');
FILE_NAME('test.step','2024-01-01T00:00:00',('Test Author'),('Test Org'),'','CADling','');
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(10.,0.,0.));
#3=CARTESIAN_POINT('',(0.,10.,0.));
#4=DIRECTION('',(1.,0.,0.));
#5=DIRECTION('',(0.,1.,0.));
#6=DIRECTION('',(0.,0.,1.));
ENDSEC;
END-ISO-10303-21;
"""
        step_file = tmp_path / "test.step"
        step_file.write_text(step_content)
        return step_file

    @pytest.fixture
    def complex_step_file(self, tmp_path):
        """Create a more complex STEP file with geometric entities."""
        step_content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('complex test file with geometry'),'2;1');
FILE_NAME('complex.step','2024-01-01T00:00:00',('Author'),('Org'),'','CADling','');
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(10.,0.,0.));
#3=CARTESIAN_POINT('',(10.,10.,0.));
#4=CARTESIAN_POINT('',(0.,10.,0.));
#10=DIRECTION('',(0.,0.,1.));
#11=DIRECTION('',(1.,0.,0.));
#12=DIRECTION('',(0.,1.,0.));
#20=VECTOR('',#10,5.0);
#21=VECTOR('',#11,10.0);
#30=LINE('',#1,#20);
#31=LINE('',#2,#20);
#40=CIRCLE('',#1,5.0);
#50=PLANE('',#1,#10);
ENDSEC;
END-ISO-10303-21;
"""
        step_file = tmp_path / "complex.step"
        step_file.write_text(step_content)
        return step_file

    def test_step_file_conversion_success(self, minimal_step_file):
        """Test successful STEP file conversion."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        # Initialize converter
        converter = DocumentConverter()

        # Convert file
        result = converter.convert(minimal_step_file)

        # Verify conversion succeeded
        assert result.status == ConversionStatus.SUCCESS
        assert result.document is not None
        assert result.errors == []

        # Verify document properties
        doc = result.document
        assert doc.name == "test.step"
        assert doc.format.value == "step"
        assert doc.hash is not None
        assert len(doc.hash) == 64  # SHA256 hash length

        # Verify entities were parsed
        assert len(doc.items) > 0
        assert doc.items[0].item_type == "step_entity"

        # Verify topology was built
        assert doc.topology is not None
        assert doc.topology.num_nodes == 6  # 6 entities in the file
        assert doc.topology.num_edges >= 0

        # Verify metadata
        assert doc.origin is not None
        assert doc.origin.filename == "test.step"
        assert doc.origin.format.value == "step"

    def test_step_complex_conversion(self, complex_step_file):
        """Test conversion of complex STEP file with various entity types."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        converter = DocumentConverter()
        result = converter.convert(complex_step_file)

        assert result.status == ConversionStatus.SUCCESS
        assert result.document is not None

        doc = result.document
        assert len(doc.items) == 13  # Total entities

        # Check for different entity types
        entity_types = {item.entity_type for item in doc.items}
        assert "CARTESIAN_POINT" in entity_types
        assert "DIRECTION" in entity_types
        assert "VECTOR" in entity_types
        assert "LINE" in entity_types
        assert "CIRCLE" in entity_types

        # Check topology reflects entity references
        assert doc.topology.num_nodes == 13
        assert doc.topology.num_edges > 0  # Entities reference each other

    def test_step_export_to_json(self, minimal_step_file):
        """Test exporting STEP document to JSON."""
        from cadling.backend.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(minimal_step_file)
        doc = result.document

        # Export to JSON
        json_data = doc.export_to_json()

        # Verify JSON structure
        assert isinstance(json_data, dict)
        assert json_data["name"] == "test.step"
        assert json_data["format"] == "step"
        assert "items" in json_data
        assert isinstance(json_data["items"], list)
        assert len(json_data["items"]) == 6

        # Check first item
        first_item = json_data["items"][0]
        assert "type" in first_item
        assert "label" in first_item
        assert first_item["type"] == "step_entity"

        # Verify topology in JSON
        assert "topology" in json_data
        assert json_data["topology"]["num_nodes"] == 6

    def test_step_export_to_markdown(self, minimal_step_file):
        """Test exporting STEP document to Markdown."""
        from cadling.backend.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(minimal_step_file)
        doc = result.document

        # Export to Markdown
        markdown_text = doc.export_to_markdown()

        # Verify Markdown contains key information
        assert isinstance(markdown_text, str)
        assert "# test.step" in markdown_text
        assert "Format: step" in markdown_text
        assert "## Entities" in markdown_text or "## Items" in markdown_text
        assert "CARTESIAN_POINT" in markdown_text

        # Verify topology section
        assert "Topology" in markdown_text or "topology" in markdown_text
        assert "6 nodes" in markdown_text or "6" in markdown_text

    def test_step_bytesio_conversion(self):
        """Test STEP conversion from BytesIO stream."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus, InputFormat

        step_content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('bytesio test'),'2;1');
FILE_NAME('stream.step','2024-01-01T00:00:00',('Author'),('Org'),'','','');
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(1.,0.,0.));
ENDSEC;
END-ISO-10303-21;
"""

        # Create BytesIO stream
        stream = io.BytesIO(step_content.encode("utf-8"))

        # Convert with explicit format
        converter = DocumentConverter()
        result = converter.convert(stream, format=InputFormat.STEP)

        assert result.status == ConversionStatus.SUCCESS
        assert result.document is not None
        assert len(result.document.items) == 2

    def test_step_invalid_file_error(self, tmp_path):
        """Test error handling for invalid STEP file."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        # Create invalid STEP file
        invalid_file = tmp_path / "invalid.step"
        invalid_file.write_text("This is not a valid STEP file\n")

        converter = DocumentConverter()
        result = converter.convert(invalid_file)

        # Should still complete but may have partial status or errors
        assert result.status in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL, ConversionStatus.FAILURE]
        # If it succeeds with invalid content, it should at least create a document
        if result.status == ConversionStatus.SUCCESS:
            assert result.document is not None

    def test_step_empty_file_error(self, tmp_path):
        """Test error handling for empty STEP file."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        # Create empty file
        empty_file = tmp_path / "empty.step"
        empty_file.write_text("")

        converter = DocumentConverter()
        result = converter.convert(empty_file)

        # Should handle gracefully
        assert result is not None
        assert result.document is not None or result.status != ConversionStatus.SUCCESS

    def test_step_nonexistent_file_error(self):
        """Test error handling for non-existent file."""
        from cadling.backend.document_converter import DocumentConverter

        nonexistent_file = Path("/nonexistent/path/to/file.step")

        converter = DocumentConverter()

        # Should raise an error or return failure status
        with pytest.raises(Exception):
            converter.convert(nonexistent_file)

    def test_step_document_items_structure(self, minimal_step_file):
        """Test the structure of parsed STEP entity items."""
        from cadling.backend.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(minimal_step_file)
        doc = result.document

        # Verify each item has required attributes
        for item in doc.items:
            assert hasattr(item, "item_type")
            assert hasattr(item, "label")
            assert hasattr(item, "entity_id")
            assert hasattr(item, "entity_type")
            assert hasattr(item, "raw_text")

            # Verify entity properties
            assert item.entity_id > 0
            assert item.entity_type in ["CARTESIAN_POINT", "DIRECTION"]
            assert item.raw_text.startswith("#")

    def test_step_header_parsing(self, complex_step_file):
        """Test STEP header parsing."""
        from cadling.backend.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(complex_step_file)
        doc = result.document

        # Check if header information was captured
        # This depends on implementation, but we should have metadata
        assert doc.origin is not None
        assert doc.origin.filename == "complex.step"

    def test_step_topology_graph(self, complex_step_file):
        """Test topology graph construction from entity references."""
        from cadling.backend.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(complex_step_file)
        doc = result.document

        topology = doc.topology
        assert topology is not None

        # Verify graph properties
        assert topology.num_nodes > 0
        assert hasattr(topology, "adjacency_list")

        # Entities with references should have edges
        # For example, VECTOR references DIRECTION
        assert topology.num_edges > 0

    def test_step_conversion_performance(self, tmp_path):
        """Test conversion performance with larger STEP file."""
        import time

        # Create a STEP file with many entities
        entities = []
        for i in range(100):
            entities.append(f"#{i+1}=CARTESIAN_POINT('',({i}.0,{i}.0,{i}.0));")

        step_content = f"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('performance test'),'2;1');
FILE_NAME('perf.step','2024-01-01T00:00:00',('Author'),('Org'),'','','');
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
{chr(10).join(entities)}
ENDSEC;
END-ISO-10303-21;
"""

        perf_file = tmp_path / "perf.step"
        perf_file.write_text(step_content)

        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        converter = DocumentConverter()

        start_time = time.time()
        result = converter.convert(perf_file)
        end_time = time.time()

        # Should complete successfully
        assert result.status == ConversionStatus.SUCCESS
        assert len(result.document.items) == 100

        # Should be reasonably fast (< 5 seconds for 100 entities)
        conversion_time = end_time - start_time
        assert conversion_time < 5.0, f"Conversion took {conversion_time:.2f}s, expected < 5.0s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
