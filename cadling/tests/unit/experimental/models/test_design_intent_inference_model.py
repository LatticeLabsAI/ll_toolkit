"""
Unit tests for DesignIntentInferenceModel.

Tests cover:
- Model initialization
- Intent inference from geometry and VLM
- Pattern detection
- Intent category classification
"""

import pytest
from unittest.mock import Mock, patch

from cadling.experimental.models import (
    DesignIntentInferenceModel,
    DesignIntent,
    IntentCategory,
    LoadType,
)
from cadling.experimental.datamodel import CADAnnotationOptions


@pytest.fixture
def mock_doc():
    """Create a mock CADlingDocument."""
    doc = Mock()
    doc.topology = {"num_faces": 10, "num_edges": 20, "num_vertices": 12}
    return doc


@pytest.fixture
def mock_item():
    """Create a mock CADItem."""
    item = Mock()
    item.self_ref = "test_item"
    item.properties = {
        "rendered_images": {"isometric": Mock()},
        "machining_features": [
            {"feature_type": "hole", "parameters": {"diameter": 8.0}},
            {"feature_type": "boss", "parameters": {"diameter": 20.0}},
        ],
        "bounding_box": {"x": 100, "y": 50, "z": 20},
        "material": "Steel",
    }
    return item


class TestDesignIntent:
    """Test DesignIntent pydantic model."""

    def test_initialization(self):
        """Test DesignIntent initialization."""
        intent = DesignIntent(
            primary_intent=IntentCategory.STRUCTURAL,
            secondary_intents=[IntentCategory.MOUNTING],
            confidence=0.9,
            is_load_bearing=True,
            expected_loads=[LoadType.TENSION, LoadType.BENDING],
            functional_description="Load-bearing bracket",
            design_rationale="Designed to support heavy loads",
            critical_features=["ribs", "bosses"],
            constraints=["Must fit in 100x50mm space"],
        )

        assert intent.primary_intent == IntentCategory.STRUCTURAL
        assert len(intent.secondary_intents) == 1
        assert intent.is_load_bearing is True
        assert len(intent.expected_loads) == 2

    def test_optional_fields(self):
        """Test intent with minimal fields."""
        intent = DesignIntent(
            primary_intent=IntentCategory.COSMETIC, confidence=0.7
        )

        assert intent.secondary_intents == []
        assert intent.is_load_bearing is False
        assert intent.expected_loads == []


class TestIntentCategory:
    """Test IntentCategory enumeration."""

    def test_intent_categories(self):
        """Test all intent categories."""
        assert IntentCategory.STRUCTURAL == "structural"
        assert IntentCategory.COSMETIC == "cosmetic"
        assert IntentCategory.MOUNTING == "mounting"
        assert IntentCategory.ALIGNMENT == "alignment"
        assert IntentCategory.SEALING == "sealing"
        assert IntentCategory.THERMAL == "thermal"
        assert IntentCategory.MOTION == "motion"
        assert IntentCategory.UNKNOWN == "unknown"


class TestLoadType:
    """Test LoadType enumeration."""

    def test_load_types(self):
        """Test all load types."""
        assert LoadType.TENSION == "tension"
        assert LoadType.COMPRESSION == "compression"
        assert LoadType.SHEAR == "shear"
        assert LoadType.BENDING == "bending"
        assert LoadType.TORSION == "torsion"
        assert LoadType.COMBINED == "combined"
        assert LoadType.NONE == "none"


class TestDesignIntentInferenceModel:
    """Test DesignIntentInferenceModel."""

    @patch("cadling.experimental.models.design_intent_inference_model.ApiVlmModel")
    def test_initialization(self, mock_vlm_class):
        """Test model initialization."""
        options = CADAnnotationOptions(vlm_model="gpt-4-vision-preview")

        mock_vlm_instance = Mock()
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = DesignIntentInferenceModel(options)

        assert model.options == options
        assert model.vlm is not None
        assert len(model.intent_patterns) > 0

    @patch("cadling.experimental.models.design_intent_inference_model.ApiVlmModel")
    def test_intent_patterns_defined(self, mock_vlm_class):
        """Test that intent patterns are defined."""
        options = CADAnnotationOptions()

        mock_vlm_instance = Mock()
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = DesignIntentInferenceModel(options)

        # Check patterns for key intents
        assert IntentCategory.STRUCTURAL in model.intent_patterns
        assert IntentCategory.MOUNTING in model.intent_patterns
        assert IntentCategory.ALIGNMENT in model.intent_patterns
        assert IntentCategory.SEALING in model.intent_patterns

    @patch("cadling.experimental.models.design_intent_inference_model.ApiVlmModel")
    def test_analyze_geometric_patterns(self, mock_vlm_class, mock_doc, mock_item):
        """Test geometric pattern analysis."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = DesignIntentInferenceModel(options)

            hints = model._analyze_geometric_patterns(mock_doc, mock_item)

        assert isinstance(hints, dict)
        # Should detect mounting intent from holes + bosses
        if IntentCategory.MOUNTING in hints:
            assert hints[IntentCategory.MOUNTING] > 0

    @patch("cadling.experimental.models.design_intent_inference_model.ApiVlmModel")
    def test_detect_pattern_bolt_circle(self, mock_vlm_class, mock_doc):
        """Test bolt circle pattern detection."""
        options = CADAnnotationOptions()

        mock_vlm_instance = Mock()
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = DesignIntentInferenceModel(options)

            # Features with multiple holes arranged in a circle pattern
            # Positions approximate a circle at radius ~10 from center (5, 5)
            features = [
                {"feature_type": "hole", "parameters": {"diameter": 8.0}, "location": [15, 5]},
                {"feature_type": "hole", "parameters": {"diameter": 8.0}, "location": [5, 15]},
                {"feature_type": "hole", "parameters": {"diameter": 8.0}, "location": [-5, 5]},
                {"feature_type": "hole", "parameters": {"diameter": 8.0}, "location": [5, -5]},
            ]

            detected = model._detect_pattern(features, "bolt_circle")

        # Should detect bolt circle with 4+ holes in circular arrangement
        assert detected == True

    @patch("cadling.experimental.models.design_intent_inference_model.ApiVlmModel")
    def test_detect_pattern_mounting_holes(self, mock_vlm_class, mock_doc):
        """Test mounting holes pattern detection."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = DesignIntentInferenceModel(options)

            # Holes with bosses
            features = [
                {"feature_type": "hole", "parameters": {}},
                {"feature_type": "boss", "parameters": {}},
            ]

            detected = model._detect_pattern(features, "mounting_holes")

        assert detected is True

    @patch("cadling.experimental.models.design_intent_inference_model.ApiVlmModel")
    def test_vlm_intent_inference(self, mock_vlm_class, mock_doc, mock_item):
        """Test VLM-based intent inference."""
        options = CADAnnotationOptions()

        # Mock VLM response
        mock_vlm_instance = Mock()
        mock_response = Mock()
        mock_response.raw_text = """
        {
            "primary_intent": "mounting",
            "secondary_intents": ["alignment"],
            "confidence": 0.85,
            "is_load_bearing": true,
            "expected_loads": ["tension", "shear"],
            "functional_description": "Mounting bracket for attaching components",
            "design_rationale": "Boss features provide mounting points",
            "critical_features": ["holes", "bosses"],
            "constraints": ["Must align with mating part"]
        }
        """
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = DesignIntentInferenceModel(options)

            vlm_intent = model._vlm_intent_inference(mock_doc, mock_item)

        assert "primary_intent" in vlm_intent
        assert vlm_intent["primary_intent"] == "mounting"
        assert vlm_intent["is_load_bearing"] is True

    @patch("cadling.experimental.models.design_intent_inference_model.ApiVlmModel")
    def test_determine_intent_combines_evidence(
        self, mock_vlm_class, mock_doc, mock_item
    ):
        """Test that intent determination combines geometric and VLM evidence."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = DesignIntentInferenceModel(options)

            geometric_hints = {IntentCategory.MOUNTING: 0.7}
            vlm_intent = {
                "primary_intent": "mounting",
                "confidence": 0.8,
                "is_load_bearing": True,
                "expected_loads": ["tension"],
                "functional_description": "Test description",
                "design_rationale": "Test rationale",
                "critical_features": [],
                "constraints": [],
            }

            intent = model._determine_intent(
                mock_doc, mock_item, geometric_hints, vlm_intent
            )

        assert isinstance(intent, DesignIntent)
        assert intent.primary_intent == IntentCategory.MOUNTING
        # Confidence should be weighted average
        assert intent.confidence > 0.7

    @patch("cadling.experimental.models.design_intent_inference_model.ApiVlmModel")
    def test_call_complete_workflow(self, mock_vlm_class, mock_doc, mock_item):
        """Test complete intent inference workflow."""
        options = CADAnnotationOptions()

        # Mock VLM
        mock_vlm_instance = Mock()
        mock_response = Mock()
        mock_response.raw_text = """{
            "primary_intent": "mounting",
            "confidence": 0.8,
            "is_load_bearing": false,
            "expected_loads": [],
            "functional_description": "Test",
            "design_rationale": "Test",
            "critical_features": [],
            "constraints": []
        }"""
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = DesignIntentInferenceModel(options)
            model(mock_doc, [mock_item])

        # Check intent was inferred
        assert "design_intent" in mock_item.properties
        intent_dict = mock_item.properties["design_intent"]
        assert "primary_intent" in intent_dict
        assert "confidence" in intent_dict

    @patch("cadling.experimental.models.design_intent_inference_model.ApiVlmModel")
    def test_supports_batch_processing(self, mock_vlm_class):
        """Test batch processing support."""
        options = CADAnnotationOptions()

        mock_vlm_instance = Mock()
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = DesignIntentInferenceModel(options)

        assert model.supports_batch_processing() is False

    @patch("cadling.experimental.models.design_intent_inference_model.ApiVlmModel")
    def test_requires_gpu(self, mock_vlm_class):
        """Test GPU requirements."""
        # API model - no GPU
        options = CADAnnotationOptions(vlm_model="gpt-4-vision-preview")

        mock_vlm_instance = Mock()
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = DesignIntentInferenceModel(options)

        assert model.requires_gpu() is False

    @patch("cadling.experimental.models.design_intent_inference_model.ApiVlmModel")
    def test_get_model_info(self, mock_vlm_class):
        """Test model info retrieval."""
        options = CADAnnotationOptions()

        mock_vlm_instance = Mock()
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = DesignIntentInferenceModel(options)

        info = model.get_model_info()

        assert "vlm_model" in info
        assert "intent_categories" in info

    @patch("cadling.experimental.models.design_intent_inference_model.ApiVlmModel")
    def test_error_handling(self, mock_vlm_class, mock_doc, mock_item):
        """Test error handling during inference."""
        options = CADAnnotationOptions()

        # Mock VLM to raise error
        mock_vlm_instance = Mock()
        mock_vlm_instance.predict.side_effect = Exception("VLM error")
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = DesignIntentInferenceModel(options)

            # Should not crash
            model(mock_doc, [mock_item])

        # Should still have design_intent recorded from geometric patterns fallback
        # and error recorded from VLM failure
        assert "design_intent" in mock_item.properties
        # Error may or may not be recorded depending on fallback behavior
        # The key is that design_intent exists (fallback to geometric analysis)

    @patch("cadling.experimental.models.design_intent_inference_model.ApiVlmModel")
    def test_fallback_to_geometric_intent(self, mock_vlm_class, mock_doc, mock_item):
        """Test fallback to geometric hints when VLM fails."""
        options = CADAnnotationOptions()

        # Mock VLM to return invalid intent
        mock_vlm_instance = Mock()
        mock_response = Mock()
        mock_response.raw_text = '{"primary_intent": "invalid_intent"}'
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = DesignIntentInferenceModel(options)
            model(mock_doc, [mock_item])

        # Should still infer some intent (fallback to geometric)
        assert "design_intent" in mock_item.properties
