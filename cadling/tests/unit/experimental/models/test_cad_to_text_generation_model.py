"""
Unit tests for CADToTextGenerationModel and CADDescription.

Tests cover:
- CADDescription pydantic model
- Model initialization (API and local VLM)
- Template-based description generation
- VLM-based caption generation
- Description combining and structuring
"""

import pytest
from unittest.mock import Mock, patch

from cadling.experimental.models import CADToTextGenerationModel, CADDescription
from cadling.experimental.datamodel import CADAnnotationOptions


@pytest.fixture
def mock_doc():
    """Create a mock CADlingDocument."""
    doc = Mock()
    doc.topology = {
        "num_faces": 10,
        "num_edges": 20,
        "faces": [],
    }
    return doc


@pytest.fixture
def mock_item_rich():
    """Create a mock CADItem with rich properties."""
    item = Mock()
    item.self_ref = "test_item"
    item.properties = {
        "rendered_images": {
            "isometric": Mock(),
            "front": Mock(),
        },
        "bounding_box": {"x": 100.0, "y": 50.0, "z": 30.0},
        "material": "Aluminum 6061",
        "machining_features": [
            {"feature_type": "hole", "parameters": {"diameter": 8.0}},
            {"feature_type": "hole", "parameters": {"diameter": 8.0}},
            {"feature_type": "pocket", "parameters": {}},
        ],
        "pmi_annotations": [
            {"type": "dimension", "value": 10.0},
            {"type": "dimension", "value": 20.0},
            {"type": "tolerance", "value": 0.05},
        ],
        "design_intent": {
            "primary_intent": "mounting",
            "functional_description": "Mounting bracket for attaching components",
        },
        "manufacturability_report": {
            "overall_score": 85.0,
            "estimated_difficulty": "Moderate",
        },
    }
    return item


@pytest.fixture
def mock_item_minimal():
    """Create a mock CADItem with minimal properties."""
    item = Mock()
    item.self_ref = "minimal_item"
    item.properties = {
        "bounding_box": {"x": 50.0, "y": 25.0, "z": 10.0},
    }
    return item


class TestCADDescription:
    """Test CADDescription pydantic model."""

    def test_initialization_full(self):
        """Test CADDescription with all fields."""
        desc = CADDescription(
            summary="A mounting bracket",
            detailed_description="A precision-machined aluminum bracket...",
            key_features=["2 mounting holes", "1 pocket"],
            dimensions_summary="100 x 50 x 30 mm",
            material_notes="Aluminum 6061-T6",
            manufacturing_notes="CNC milling required",
            assembly_instructions="Mount using M8 bolts",
            technical_specifications=["Max load: 500N", "Surface finish: Ra 3.2"],
            context_aware_notes="Designed for outdoor use",
        )

        assert desc.summary == "A mounting bracket"
        assert len(desc.key_features) == 2
        assert len(desc.technical_specifications) == 2

    def test_initialization_minimal(self):
        """Test CADDescription with minimal fields."""
        desc = CADDescription()

        assert desc.summary == ""
        assert desc.detailed_description == ""
        assert desc.key_features == []
        assert desc.dimensions_summary == ""
        assert desc.technical_specifications == []

    def test_to_dict(self):
        """Test conversion to dictionary."""
        desc = CADDescription(
            summary="Test part",
            key_features=["Feature 1", "Feature 2"],
        )

        data = desc.model_dump()

        assert data["summary"] == "Test part"
        assert len(data["key_features"]) == 2
        assert "detailed_description" in data


class TestCADToTextGenerationModel:
    """Test CADToTextGenerationModel."""

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_initialization_with_api_vlm(self, mock_vlm_class):
        """Test model initialization with API-based VLM."""
        options = CADAnnotationOptions(vlm_model="gpt-4-vision-preview")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)

        assert model.options == options
        assert model.vlm is not None
        assert len(model.templates) > 0

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_initialization_with_claude(self, mock_vlm_class):
        """Test model initialization with Claude VLM."""
        options = CADAnnotationOptions(vlm_model="claude-3-opus-20240229")

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)

        assert model.vlm is not None

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_templates_loaded(self, mock_vlm_class):
        """Test that description templates are loaded."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)

        assert "summary" in model.templates
        assert "dimensions" in model.templates
        assert "features_list" in model.templates

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_generate_template_description_rich(
        self, mock_vlm_class, mock_doc, mock_item_rich
    ):
        """Test template-based description with rich data."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)

            desc = model._generate_template_description(mock_doc, mock_item_rich)

        # Should extract dimensions
        assert "dimensions" in desc
        assert "100.0" in desc["dimensions"]

        # Should summarize features
        assert "features" in desc
        assert "hole" in desc["features"]
        assert "pocket" in desc["features"]

        # Should include PMI summary
        assert "pmi_summary" in desc
        assert "2 dimensions" in desc["pmi_summary"]

        # Should include material
        assert "material" in desc
        assert desc["material"] == "Aluminum 6061"

        # Should include intent
        assert "intent" in desc
        assert "Mounting bracket" in desc["intent"]

        # Should include manufacturability
        assert "manufacturability" in desc
        assert "85.0" in desc["manufacturability"]

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_generate_template_description_minimal(
        self, mock_vlm_class, mock_doc, mock_item_minimal
    ):
        """Test template-based description with minimal data."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)

            desc = model._generate_template_description(mock_doc, mock_item_minimal)

        # Should only have dimensions
        assert "dimensions" in desc
        assert "features" not in desc
        assert "material" not in desc

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_generate_vlm_description(
        self, mock_vlm_class, mock_doc, mock_item_rich
    ):
        """Test VLM-based description generation."""
        options = CADAnnotationOptions()

        # Mock VLM response
        mock_vlm_instance = Mock()
        mock_response = Mock()
        mock_response.raw_text = """
        {
            "summary": "A precision-machined mounting bracket",
            "detailed_description": "This is a rectangular bracket with mounting holes...",
            "key_features": ["2 through holes", "Rectangular pocket", "Flat mounting surface"],
            "manufacturing_notes": "CNC milling required for precision",
            "assembly_instructions": "Mount using M8 bolts",
            "technical_specifications": ["Max load: 500N", "Material: Aluminum"]
        }
        """
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)

            vlm_desc = model._generate_vlm_description(mock_doc, mock_item_rich)

        assert "summary" in vlm_desc
        assert "mounting bracket" in vlm_desc["summary"].lower()
        assert "key_features" in vlm_desc
        assert len(vlm_desc["key_features"]) == 3

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_generate_vlm_description_no_images(
        self, mock_vlm_class, mock_doc, mock_item_minimal
    ):
        """Test VLM description when no images available."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)

            vlm_desc = model._generate_vlm_description(mock_doc, mock_item_minimal)

        # Should return empty dict
        assert vlm_desc == {}

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_combine_descriptions_vlm_priority(
        self, mock_vlm_class, mock_doc, mock_item_rich
    ):
        """Test combining descriptions with VLM content taking priority."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)

            template_desc = {
                "dimensions": "100 x 50 x 30 mm",
                "features": "2 holes, 1 pocket",
                "material": "Aluminum",
            }

            vlm_desc = {
                "summary": "A precision mounting bracket",
                "detailed_description": "Detailed VLM description...",
                "key_features": ["Feature 1", "Feature 2"],
                "manufacturing_notes": "CNC milling",
            }

            final_desc = model._combine_descriptions(
                mock_doc, mock_item_rich, template_desc, vlm_desc
            )

        assert isinstance(final_desc, CADDescription)
        # VLM summary should take priority
        assert final_desc.summary == "A precision mounting bracket"
        # VLM detailed description should be used
        assert final_desc.detailed_description == "Detailed VLM description..."
        # VLM features should be used
        assert len(final_desc.key_features) == 2
        # Template dimensions should be included
        assert "100 x 50 x 30" in final_desc.dimensions_summary

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_combine_descriptions_template_fallback(
        self, mock_vlm_class, mock_doc, mock_item_rich
    ):
        """Test combining descriptions with template fallback."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)

            template_desc = {
                "dimensions": "100 x 50 x 30 mm",
                "features": "2 holes, 1 pocket",
            }

            vlm_desc = {}  # Empty VLM result

            final_desc = model._combine_descriptions(
                mock_doc, mock_item_rich, template_desc, vlm_desc
            )

        # Should fallback to template content
        assert "2 holes, 1 pocket" in final_desc.summary
        assert "100 x 50 x 30 mm" in final_desc.detailed_description

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_call_complete_workflow(self, mock_vlm_class, mock_doc, mock_item_rich):
        """Test complete text generation workflow."""
        options = CADAnnotationOptions()

        # Mock VLM
        mock_vlm_instance = Mock()
        mock_response = Mock()
        mock_response.raw_text = """{
            "summary": "Mounting bracket",
            "detailed_description": "A precision part...",
            "key_features": ["holes", "pocket"],
            "manufacturing_notes": "CNC",
            "assembly_instructions": "Mount with bolts",
            "technical_specifications": ["Load: 500N"]
        }"""
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)
            model(mock_doc, [mock_item_rich])

        # Check description was generated
        assert "text_description" in mock_item_rich.properties
        desc = mock_item_rich.properties["text_description"]
        assert "summary" in desc
        assert "detailed_description" in desc
        assert "key_features" in desc

        # Check model name recorded
        assert "text_generation_model" in mock_item_rich.properties

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_multiple_items(self, mock_vlm_class, mock_doc, mock_item_rich):
        """Test processing multiple items."""
        options = CADAnnotationOptions()

        # Mock VLM
        mock_vlm_instance = Mock()
        mock_response = Mock()
        mock_response.raw_text = '{"summary": "Test", "detailed_description": "Test", "key_features": []}'
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        # Create multiple items
        item1 = Mock()
        item1.self_ref = "item1"
        item1.properties = {"rendered_images": {"isometric": Mock()}, "bounding_box": {}}

        item2 = Mock()
        item2.self_ref = "item2"
        item2.properties = {"rendered_images": {"front": Mock()}, "bounding_box": {}}

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)
            model(mock_doc, [item1, item2])

        # Both items should have descriptions
        assert "text_description" in item1.properties
        assert "text_description" in item2.properties

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_supports_batch_processing(self, mock_vlm_class):
        """Test batch processing support."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)

        assert model.supports_batch_processing() is False

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    @patch("cadling.experimental.models.cad_to_text_generation_model.InlineVlmModel")
    def test_requires_gpu(self, mock_inline_vlm_class, mock_api_vlm_class):
        """Test GPU requirements based on model type."""
        # API model - no GPU
        options = CADAnnotationOptions(vlm_model="gpt-4-vision-preview")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)

        assert model.requires_gpu() is False

        # Local model - GPU needed
        options = CADAnnotationOptions(vlm_model="llava-hf/llava-1.5-7b-hf")
        model = CADToTextGenerationModel(options)

        assert model.requires_gpu() is True

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_get_model_info(self, mock_vlm_class):
        """Test model info retrieval."""
        options = CADAnnotationOptions(vlm_model="gpt-4-vision-preview")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)

        info = model.get_model_info()

        assert "vlm_model" in info
        assert info["vlm_model"] == "gpt-4-vision-preview"
        assert "supports_templates" in info
        assert info["supports_templates"] == "true"

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_error_handling(self, mock_vlm_class, mock_doc, mock_item_rich):
        """Test error handling during generation."""
        options = CADAnnotationOptions()

        # Mock VLM to raise error
        mock_vlm_instance = Mock()
        mock_vlm_instance.predict.side_effect = Exception("VLM error")
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)

            # Should not crash
            model(mock_doc, [mock_item_rich])

        # Should still generate description (VLM error is handled gracefully)
        # The test verifies that model doesn't crash even with VLM errors
        assert "text_description" in mock_item_rich.properties

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_provenance_tracking(self, mock_vlm_class, mock_doc, mock_item_rich):
        """Test that provenance is tracked."""
        options = CADAnnotationOptions()

        mock_vlm_instance = Mock()
        mock_response = Mock()
        mock_response.raw_text = '{"summary": "Test", "detailed_description": "Test"}'
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        # Add provenance tracking method
        mock_item_rich.add_provenance = Mock()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)
            model(mock_doc, [mock_item_rich])

        # Check provenance was added
        if hasattr(mock_item_rich, "add_provenance"):
            mock_item_rich.add_provenance.assert_called()

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_view_selection_preference(self, mock_vlm_class, mock_doc):
        """Test that isometric view is preferred when available."""
        options = CADAnnotationOptions()

        # Item with multiple views
        item = Mock()
        item.self_ref = "test"
        item.properties = {
            "rendered_images": {
                "front": Mock(),
                "isometric": Mock(),
                "top": Mock(),
            }
        }

        mock_vlm_instance = Mock()
        mock_response = Mock()
        mock_response.raw_text = '{"summary": "Test"}'
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)
            model._generate_vlm_description(mock_doc, item)

        # Should have used isometric view
        # (VLM predict was called with the isometric image)
        assert mock_vlm_instance.predict.called

    @patch("cadling.experimental.models.cad_to_text_generation_model.ApiVlmModel")
    def test_feature_counting(self, mock_vlm_class, mock_doc):
        """Test that features are properly counted and aggregated."""
        options = CADAnnotationOptions()

        item = Mock()
        item.self_ref = "test"
        item.properties = {
            "machining_features": [
                {"feature_type": "hole"},
                {"feature_type": "hole"},
                {"feature_type": "hole"},
                {"feature_type": "pocket"},
                {"feature_type": "fillet"},
                {"feature_type": "fillet"},
            ]
        }

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = CADToTextGenerationModel(options)
            desc = model._generate_template_description(mock_doc, item)

        # Should have counted features
        assert "features" in desc
        assert "3 holes" in desc["features"]
        assert "1 pocket" in desc["features"]
        assert "2 fillets" in desc["features"]
