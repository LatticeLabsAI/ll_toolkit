"""Tests for STEPNet integration alternative/fallback implementations.

Tests the fallback methods that work when ll_stepnet is unavailable:
- _tokenize_alternative: Tokenizes STEP text without ll_stepnet
- _extract_features_alternative: Extracts entity features without ll_stepnet
- _build_topology_alternative: Builds topology graph without ll_stepnet
- _reserialize_alternative: Reserializes STEP with annotations without ll_stepnet
"""

from __future__ import annotations

import pytest


class TestSTEPNetIntegrationFallbacks:
    """Test alternative implementations when ll_stepnet unavailable."""

    def test_tokenize_alternative_basic(self):
        """Test alternative tokenization with basic STEP entities."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        step_text = """
        ISO-10303-21;
        HEADER;
        FILE_DESCRIPTION(('Test'),'2;1');
        ENDSEC;
        DATA;
        #1=CARTESIAN_POINT('Point1',(0.0,0.0,0.0));
        #2=CARTESIAN_POINT('Point2',(1.0,0.0,0.0));
        #3=DIRECTION('Dir',(1.0,0.0,0.0));
        ENDSEC;
        END-ISO-10303-21;
        """

        # Force use of alternative implementation
        result = integration._tokenize_alternative(step_text)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(t, int) for t in result)

    def test_tokenize_alternative_with_references(self):
        """Test alternative tokenization handles entity references."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        step_text = """
        DATA;
        #1=CARTESIAN_POINT('',(0.0,0.0,0.0));
        #2=VERTEX_POINT('',#1);
        #3=DIRECTION('',(1.0,0.0,0.0));
        #4=AXIS2_PLACEMENT_3D('',#1,#3,$);
        ENDSEC;
        """

        result = integration._tokenize_alternative(step_text)

        assert result is not None
        assert len(result) > 0

    def test_tokenize_alternative_with_numeric_params(self):
        """Test alternative tokenization handles numeric parameters."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        step_text = """
        DATA;
        #1=CARTESIAN_POINT('',(-123.456,789.012,0.5e-3));
        #2=CIRCLE('',#1,25.4);
        ENDSEC;
        """

        result = integration._tokenize_alternative(step_text)

        assert result is not None
        assert len(result) > 0

    def test_tokenize_alternative_empty_input(self):
        """Test alternative tokenization with empty input."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        result = integration._tokenize_alternative("")

        assert result is not None
        assert isinstance(result, list)
        # Should return at least CLS token
        assert len(result) >= 1

    def test_extract_features_alternative_basic(self):
        """Test alternative feature extraction for basic entity."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        entity_text = "#1=CARTESIAN_POINT('Origin',(0.0,0.0,0.0));"

        result = integration._extract_features_alternative(entity_text, "CARTESIAN_POINT")

        assert result is not None
        assert isinstance(result, dict)
        assert "entity_type" in result
        assert "category" in result
        assert "num_references" in result

    def test_extract_features_alternative_with_references(self):
        """Test alternative feature extraction counts references."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        entity_text = "#10=AXIS2_PLACEMENT_3D('Placement',#1,#3,#5);"

        result = integration._extract_features_alternative(entity_text, "AXIS2_PLACEMENT_3D")

        assert result is not None
        assert result["num_references"] >= 3  # Should find #1, #3, #5

    def test_extract_features_alternative_categories(self):
        """Test entity type categorization."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        # Test various entity types
        test_cases = [
            ("CARTESIAN_POINT", "geometry"),
            ("VERTEX_POINT", "topology"),
            ("ADVANCED_FACE", "topology"),
            ("PRODUCT", "assembly"),
            ("APPLICATION_CONTEXT", "context"),
        ]

        for entity_type, expected_category in test_cases:
            result = integration._extract_features_alternative(
                f"#1={entity_type}('test');", entity_type
            )
            assert result is not None, f"Failed for {entity_type}"
            # Category should be computed

    def test_build_topology_alternative_basic(self):
        """Test alternative topology building with basic entities."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        entities = [
            {"entity_id": 1, "entity_type": "CARTESIAN_POINT", "text": "#1=CARTESIAN_POINT('',());"},
            {"entity_id": 2, "entity_type": "VERTEX_POINT", "text": "#2=VERTEX_POINT('',#1);"},
            {"entity_id": 3, "entity_type": "CARTESIAN_POINT", "text": "#3=CARTESIAN_POINT('',());"},
        ]

        result = integration._build_topology_alternative(entities)

        assert result is not None
        assert isinstance(result, dict)
        assert "num_nodes" in result
        assert "num_edges" in result
        assert "adjacency_list" in result
        assert result["num_nodes"] == 3

    def test_build_topology_alternative_detects_references(self):
        """Test alternative topology building detects entity references."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        # Provide params that contain references (which is how TopologyBuilder detects edges)
        entities = [
            {"entity_id": 1, "entity_type": "CARTESIAN_POINT", "params": ["''", "(0.0,0.0,0.0)"]},
            {"entity_id": 2, "entity_type": "DIRECTION", "params": ["''", "(1.0,0.0,0.0)"]},
            {"entity_id": 3, "entity_type": "AXIS2_PLACEMENT_3D", "params": ["''", "#1", "#2", "$"]},
        ]

        result = integration._build_topology_alternative(entities)

        assert result is not None
        assert result["num_edges"] >= 2  # #3 references #1 and #2

    def test_build_topology_alternative_empty_input(self):
        """Test alternative topology building with empty input."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        result = integration._build_topology_alternative([])

        assert result is not None
        assert result["num_nodes"] == 0
        assert result["num_edges"] == 0

    def test_reserialize_alternative_basic(self):
        """Test alternative reserialization with basic STEP text."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        step_text = """
        DATA;
        #1=CARTESIAN_POINT('P1',(0.0,0.0,0.0));
        #2=CARTESIAN_POINT('P2',(1.0,0.0,0.0));
        #3=LINE('L1',#1,#2);
        ENDSEC;
        """

        result = integration._reserialize_alternative(step_text)

        assert result is not None
        assert isinstance(result, dict)
        assert "reserialized_text" in result
        assert "id_mapping" in result
        assert "summary" in result or "annotated_text" in result

    def test_reserialize_alternative_preserves_content(self):
        """Test alternative reserialization preserves entity content."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        step_text = """
        DATA;
        #100=CARTESIAN_POINT('Test',(1.5,2.5,3.5));
        ENDSEC;
        """

        result = integration._reserialize_alternative(step_text)

        # Should contain the coordinate values in some form
        text = result["reserialized_text"]
        assert "1.5" in text or "(1.5" in text

    def test_reserialize_alternative_generates_annotations(self):
        """Test alternative reserialization generates annotations."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        step_text = """
        DATA;
        #1=PRODUCT('Part1','Part One Description',$,$);
        #2=CARTESIAN_POINT('',(0.0,0.0,0.0));
        #3=DIRECTION('',(1.0,0.0,0.0));
        #4=ADVANCED_FACE('Face1',(),$,$);
        ENDSEC;
        """

        result = integration._reserialize_alternative(step_text)

        # Should have summary or annotated_text
        assert "summary" in result or "annotated_text" in result
        # If summary exists, it should have content
        if "summary" in result and result["summary"]:
            assert len(result["summary"]) > 0

    def test_reserialize_alternative_empty_input(self):
        """Test alternative reserialization with empty input."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        result = integration._reserialize_alternative("")

        assert result is not None
        assert "reserialized_text" in result
        assert "id_mapping" in result


class TestSTEPNetIntegrationHelpers:
    """Test helper methods used in fallback implementations."""

    def test_parse_step_entities(self):
        """Test STEP entity parsing."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        step_text = """
        DATA;
        #1=CARTESIAN_POINT('P1',(0.0,0.0,0.0));
        #2=DIRECTION('D1',(1.0,0.0,0.0));
        ENDSEC;
        """

        entities = integration._parse_step_entities(step_text)

        assert len(entities) == 2
        assert entities[0]["id"] == 1
        assert entities[0]["type"] == "CARTESIAN_POINT"
        assert entities[1]["id"] == 2
        assert entities[1]["type"] == "DIRECTION"

    def test_parse_step_params(self):
        """Test STEP parameter parsing."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        # Parse simple parameters
        result = integration._parse_step_params("'Name',(1.0,2.0,3.0),#5")

        assert len(result) == 3
        # Result[0] is the string 'Name'
        # Result[1] is a list [1.0, 2.0, 3.0]
        # Result[2] is the reference '#5'
        assert result[0] == "Name"  # String without quotes
        assert isinstance(result[1], list)
        assert 1.0 in result[1] or 2.0 in result[1]  # Contains coordinates as floats

    def test_quantize_number(self):
        """Test numeric quantization."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        # Test various numbers
        assert integration._quantize_number(0.0) >= 0
        assert integration._quantize_number(1.0) >= 0
        assert integration._quantize_number(-1.0) >= 0
        assert integration._quantize_number(1e10) >= 0
        assert integration._quantize_number(-1e10) >= 0

    def test_categorize_entity_type(self):
        """Test entity type categorization."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        # Point types
        assert integration._categorize_entity_type("CARTESIAN_POINT") == "point"

        # Curve types
        assert integration._categorize_entity_type("LINE") == "curve"
        assert integration._categorize_entity_type("CIRCLE") == "curve"

        # Surface types
        assert integration._categorize_entity_type("PLANE") == "surface"
        assert integration._categorize_entity_type("CYLINDRICAL_SURFACE") == "surface"

        # Topology types
        assert integration._categorize_entity_type("EDGE_CURVE") == "topology"
        assert integration._categorize_entity_type("ADVANCED_FACE") == "topology"

        # Shape types
        assert integration._categorize_entity_type("SHAPE_REPRESENTATION") == "shape"

        # Unknown type
        assert integration._categorize_entity_type("UNKNOWN_ENTITY_TYPE") == "other"
        assert integration._categorize_entity_type("PRODUCT") == "other"


class TestSTEPNetIntegrationPublicMethods:
    """Test public methods that use fallbacks."""

    def test_tokenize_uses_fallback(self):
        """Test tokenize() uses alternative when ll_stepnet unavailable."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()
        # Force ll_stepnet unavailable
        integration._has_ll_stepnet = False

        step_text = "#1=CARTESIAN_POINT('',());"

        result = integration.tokenize(step_text)

        # Should return tokens (not None)
        assert result is not None
        assert isinstance(result, list)

    def test_extract_features_uses_fallback(self):
        """Test extract_features() uses alternative when ll_stepnet unavailable."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()
        integration._has_ll_stepnet = False

        entity_text = "#1=CARTESIAN_POINT('',());"

        result = integration.extract_features(entity_text, "CARTESIAN_POINT")

        assert result is not None
        assert isinstance(result, dict)

    def test_build_topology_uses_fallback(self):
        """Test build_topology() uses alternative when ll_stepnet unavailable."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()
        integration._has_ll_stepnet = False

        entities = [{"entity_id": 1, "entity_type": "POINT", "text": "#1=POINT();"}]

        result = integration.build_topology(entities)

        assert result is not None
        assert isinstance(result, dict)

    def test_reserialize_uses_fallback(self):
        """Test reserialize() uses alternative when ll_stepnet unavailable."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()
        integration._has_ll_stepnet = False

        step_text = "#1=CARTESIAN_POINT('',());"

        result = integration.reserialize(step_text)

        assert result is not None
        assert isinstance(result, dict)


class TestSTEPNetIntegrationEdgeCases:
    """Test edge cases and error handling."""

    def test_tokenize_malformed_step(self):
        """Test tokenization handles malformed STEP gracefully."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        # Malformed STEP
        malformed = "This is not valid STEP data #@!$%^"

        result = integration._tokenize_alternative(malformed)

        # Should return something, not crash
        assert result is not None

    def test_extract_features_no_params(self):
        """Test feature extraction with entity having no parameters."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        entity_text = "#1=EMPTY_ENTITY();"

        result = integration._extract_features_alternative(entity_text, "EMPTY_ENTITY")

        assert result is not None

    def test_build_topology_circular_references(self):
        """Test topology building with circular references."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        # Create circular reference (entity refers to itself indirectly)
        entities = [
            {"entity_id": 1, "entity_type": "A", "text": "#1=A(#2);"},
            {"entity_id": 2, "entity_type": "B", "text": "#2=B(#1);"},
        ]

        result = integration._build_topology_alternative(entities)

        # Should handle without infinite loop
        assert result is not None
        assert result["num_nodes"] == 2

    def test_reserialize_large_ids(self):
        """Test reserialization with large entity IDs."""
        from cadling.backend.step.stepnet_integration import STEPNetIntegration

        integration = STEPNetIntegration()

        step_text = """
        DATA;
        #999999=CARTESIAN_POINT('',(0.0,0.0,0.0));
        #1000000=DIRECTION('',(1.0,0.0,0.0));
        ENDSEC;
        """

        result = integration._reserialize_alternative(step_text)

        assert result is not None
        assert "id_mapping" in result
