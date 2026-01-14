"""
Integration tests for STEP pipeline.

Tests the full STEP conversion workflow from file to document.
"""

import io
import pytest


class TestSTEPPipeline:
    """Integration tests for STEP pipeline."""

    def test_step_tokenizer_basic(self):
        """Test basic STEP tokenizer functionality."""
        from cadling.backend.step.tokenizer import STEPTokenizer

        tokenizer = STEPTokenizer()

        # Create minimal STEP content
        step_content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('test file'),'2;1');
FILE_NAME('test.step','2024-01-01',('Author'),('Organization'),'','','');
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(1.,0.,0.));
#3=CARTESIAN_POINT('',(0.,1.,0.));
ENDSEC;
END-ISO-10303-21;
"""

        result = tokenizer.parse_step_file(step_content)

        assert "header" in result
        assert "entities" in result
        assert len(result["entities"]) == 3
        assert 1 in result["entities"]
        assert result["entities"][1]["type"] == "CARTESIAN_POINT"

    def test_step_feature_extractor_basic(self):
        """Test basic feature extraction."""
        from cadling.backend.step.feature_extractor import STEPFeatureExtractor

        extractor = STEPFeatureExtractor()

        # Create test entities
        entities = {
            1: {
                "type": "CARTESIAN_POINT",
                "params": [["0.0", "0.0", "0.0"]],
            },
            2: {
                "type": "CIRCLE",
                "params": ["#1", "5.0"],
            },
        }

        features = extractor.extract_features(entities)

        assert 1 in features
        assert 2 in features
        assert features[1]["category"] == "point"
        assert features[2]["category"] == "curve"

    def test_step_topology_builder_basic(self):
        """Test basic topology building."""
        from cadling.backend.step.topology_builder import TopologyBuilder

        builder = TopologyBuilder()

        # Create test entities with references
        entities = {
            1: {"type": "CARTESIAN_POINT", "params": []},
            2: {"type": "CARTESIAN_POINT", "params": []},
            3: {"type": "LINE", "params": ["#1", "#2"]},
        }

        topology = builder.build_topology_graph(entities)

        assert topology["num_nodes"] == 3
        assert topology["num_edges"] == 2
        assert 3 in topology["adjacency_list"]
        assert 1 in topology["adjacency_list"][3]
        assert 2 in topology["adjacency_list"][3]

    def test_step_backend_minimal(self):
        """Test STEP backend with minimal file."""
        from cadling.backend.step import STEPBackend
        from cadling.datamodel.base_models import CADInputDocument, InputFormat
        from pathlib import Path

        # Create minimal STEP content
        step_content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('test'),'2;1');
FILE_NAME('test.step','2024-01-01',('Author'),('Org'),'','','');
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(10.,0.,0.));
#3=CARTESIAN_POINT('',(0.,10.,0.));
ENDSEC;
END-ISO-10303-21;
"""

        # Create BytesIO stream
        stream = io.BytesIO(step_content.encode("utf-8"))

        # Create input document
        input_doc = CADInputDocument(
            file=Path("test.step"),
            format=InputFormat.STEP,
            document_hash="test_hash",
        )

        # Create backend
        backend = STEPBackend(input_doc, stream)

        # Test validity
        assert backend.is_valid()

        # Convert
        document = backend.convert()

        assert document.name == "test.step"
        assert len(document.items) == 3
        assert document.topology is not None
        assert document.topology.num_nodes == 3

    def test_step_pipeline_integration(self):
        """Test full STEP pipeline integration."""
        from cadling.pipeline.step_pipeline import STEPPipeline
        from cadling.datamodel.base_models import CADInputDocument, InputFormat, ConversionResult
        from cadling.backend.step import STEPBackend
        from pathlib import Path

        # Create STEP content
        step_content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('integration test'),'2;1');
FILE_NAME('test.step','2024-01-01',('Author'),('Org'),'','','');
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(1.,0.,0.));
#3=VECTOR('',#2,1.0);
#4=LINE('',#1,#3);
ENDSEC;
END-ISO-10303-21;
"""

        stream = io.BytesIO(step_content.encode("utf-8"))

        # Create input document with backend
        input_doc = CADInputDocument(
            file=Path("test.step"),
            format=InputFormat.STEP,
            document_hash="test_hash",
        )
        input_doc._backend = STEPBackend(input_doc, stream)

        # Create pipeline
        options = STEPPipeline.get_default_options()
        pipeline = STEPPipeline(options)

        # Execute
        result = pipeline.execute(input_doc)

        assert result.status.value in ["success", "partial"]
        assert result.document is not None
        assert len(result.document.items) == 4
        assert result.document.topology is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
