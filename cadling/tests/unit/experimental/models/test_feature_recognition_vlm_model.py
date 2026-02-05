"""
Unit tests for FeatureRecognitionVlmModel and MachiningFeature.

Tests cover:
- Model initialization
- Feature recognition from rendered views
- Cross-view validation and fusion
- Feature similarity detection
"""

import pytest
from unittest.mock import MagicMock, Mock, patch

from cadling.experimental.models import FeatureRecognitionVlmModel, MachiningFeature
from cadling.experimental.datamodel import CADAnnotationOptions


@pytest.fixture
def mock_doc():
    """Create a mock CADlingDocument."""
    doc = Mock()
    doc.topology = {
        "num_faces": 10,
        "num_edges": 20,
        "num_vertices": 12,
        "faces": [],
    }
    return doc


@pytest.fixture
def mock_item():
    """Create a mock CADItem with rendered images."""
    item = Mock()
    item.self_ref = "test_item"
    item.properties = {
        "rendered_images": {
            "front": Mock(),
            "top": Mock(),
            "right": Mock(),
        },
        "bounding_box": {"x": 100, "y": 50, "z": 30},
        "material": "Steel",
    }
    return item


class TestMachiningFeature:
    """Test MachiningFeature pydantic model."""

    def test_initialization(self):
        """Test MachiningFeature initialization."""
        feature = MachiningFeature(
            feature_type="hole",
            subtype="through_hole",
            parameters={"diameter": 10.0, "depth": None},
            location=[50.0, 25.0, 15.0],
            confidence=0.9,
            view="front",
            description="10mm through hole",
        )

        assert feature.feature_type == "hole"
        assert feature.subtype == "through_hole"
        assert feature.parameters["diameter"] == 10.0
        assert feature.confidence == 0.9

    def test_optional_fields(self):
        """Test feature with optional fields."""
        feature = MachiningFeature(feature_type="pocket", confidence=0.8)

        assert feature.subtype is None
        assert feature.location is None
        assert feature.view == ""
        assert feature.description == ""


class TestFeatureRecognitionVlmModel:
    """Test FeatureRecognitionVlmModel."""

    def test_initialization_with_api_vlm(self):
        """Test model initialization with API-based VLM."""
        options = CADAnnotationOptions(vlm_model="gpt-4-vision-preview")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = FeatureRecognitionVlmModel(options)

        assert model.options == options
        assert model.vlm is not None
        assert len(model.feature_prompts) > 0

    def test_feature_prompts_created(self):
        """Test that feature prompts are created for all feature types."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = FeatureRecognitionVlmModel(options)

        # Check that prompts exist for key types
        assert "hole" in model.feature_prompts
        assert "pocket" in model.feature_prompts
        assert "fillet" in model.feature_prompts
        assert "chamfer" in model.feature_prompts
        assert "boss" in model.feature_prompts
        assert "thread" in model.feature_prompts

    @patch("cadling.experimental.models.feature_recognition_vlm_model.ApiVlmModel")
    def test_call_with_no_rendered_images(self, mock_vlm_class, mock_doc):
        """Test that items without rendered images are skipped."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = FeatureRecognitionVlmModel(options)

        # Item without rendered images
        item = Mock()
        item.self_ref = "test_item"
        item.properties = {}

        model(mock_doc, [item])

        # Should skip and not crash
        assert "machining_features" not in item.properties or item.properties.get(
            "machining_features"
        ) == []

    @patch("cadling.experimental.models.feature_recognition_vlm_model.ApiVlmModel")
    def test_call_with_rendered_images(self, mock_vlm_class, mock_doc, mock_item):
        """Test feature recognition with rendered images."""
        options = CADAnnotationOptions(
            annotation_types=["hole"],
            views_to_process=["front"],
        )

        # Mock VLM response
        mock_vlm_instance = Mock()
        mock_response = Mock()
        mock_response.raw_text = """
        [{"feature_type": "hole", "subtype": "through_hole",
          "parameters": {"diameter": 10.0}, "location": [50, 25],
          "confidence": 0.9}]
        """
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = FeatureRecognitionVlmModel(options)
            model(mock_doc, [mock_item])

        # Check that features were added
        assert "machining_features" in mock_item.properties

    @patch("cadling.experimental.models.feature_recognition_vlm_model.ApiVlmModel")
    def test_parse_features_from_response(self, mock_vlm_class, mock_doc):
        """Test parsing features from JSON response."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = FeatureRecognitionVlmModel(options)

        response_text = """
        Here are the features:
        [
            {
                "feature_type": "hole",
                "subtype": "blind_hole",
                "parameters": {"diameter": 8.0, "depth": 15.0},
                "confidence": 0.85
            },
            {
                "feature_type": "pocket",
                "subtype": "rectangular_pocket",
                "parameters": {"length": 50.0, "width": 30.0, "depth": 10.0},
                "confidence": 0.9
            }
        ]
        """

        features = model._parse_features_from_response(response_text, "front")

        assert len(features) == 2
        assert features[0]["feature_type"] == "hole"
        assert features[1]["feature_type"] == "pocket"
        assert features[0]["view"] == "front"

    @patch("cadling.experimental.models.feature_recognition_vlm_model.ApiVlmModel")
    def test_features_similar(self, mock_vlm_class, mock_doc):
        """Test feature similarity detection."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = FeatureRecognitionVlmModel(options)

        # Similar features
        feat1 = {
            "feature_type": "hole",
            "subtype": "through_hole",
            "parameters": {"diameter": 10.0},
        }
        feat2 = {
            "feature_type": "hole",
            "subtype": "through_hole",
            "parameters": {"diameter": 10.1},  # Within 10% tolerance
        }

        assert model._features_similar(feat1, feat2) is True

        # Different feature types
        feat3 = {"feature_type": "pocket", "parameters": {}}
        assert model._features_similar(feat1, feat3) is False

        # Same type but different parameters
        feat4 = {
            "feature_type": "hole",
            "subtype": "through_hole",
            "parameters": {"diameter": 20.0},  # >10% different
        }
        assert model._features_similar(feat1, feat4) is False

    @patch("cadling.experimental.models.feature_recognition_vlm_model.ApiVlmModel")
    def test_merge_features(self, mock_vlm_class, mock_doc):
        """Test merging similar features from multiple views."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = FeatureRecognitionVlmModel(options)

        features = [
            {
                "feature_type": "hole",
                "parameters": {"diameter": 10.0},
                "confidence": 0.8,
                "view": "front",
            },
            {
                "feature_type": "hole",
                "parameters": {"diameter": 10.2},
                "confidence": 0.85,
                "view": "top",
            },
        ]

        merged = model._merge_features(features)

        assert merged["confidence"] > 0.8  # Confidence boosted
        assert "views" in merged
        assert len(merged["views"]) == 2
        assert merged["cross_view_validated"] is True

    @patch("cadling.experimental.models.feature_recognition_vlm_model.ApiVlmModel")
    def test_cross_view_validation(self, mock_vlm_class, mock_doc, mock_item):
        """Test cross-view validation of features."""
        options = CADAnnotationOptions(
            annotation_types=["hole"],
            views_to_process=["front", "top"],
            enable_cross_view_validation=True,
        )

        mock_vlm_instance = Mock()
        mock_response = Mock()
        mock_response.raw_text = """
        [{"feature_type": "hole", "parameters": {"diameter": 10.0},
          "confidence": 0.8}]
        """
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = FeatureRecognitionVlmModel(options)
            model(mock_doc, [mock_item])

        # Should have validated features
        assert "machining_features" in mock_item.properties

    def test_supports_batch_processing(self):
        """Test that model reports batch processing support."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = FeatureRecognitionVlmModel(options)

        assert model.supports_batch_processing() is False

    @patch("cadling.experimental.models.feature_recognition_vlm_model.InlineVlmModel")
    @patch("cadling.experimental.models.feature_recognition_vlm_model.ApiVlmModel")
    def test_requires_gpu(self, mock_api_vlm_class, mock_inline_vlm_class):
        """Test GPU requirements based on model type."""
        # API model - no GPU
        mock_api_vlm = Mock()
        mock_api_vlm_class.return_value = mock_api_vlm
        options = CADAnnotationOptions(vlm_model="gpt-4-vision-preview")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = FeatureRecognitionVlmModel(options)

        assert model.requires_gpu() is False

        # Local model - GPU needed
        mock_inline_vlm = Mock()
        mock_inline_vlm_class.return_value = mock_inline_vlm
        options = CADAnnotationOptions(vlm_model="llava-hf/llava-1.5-7b-hf")
        model = FeatureRecognitionVlmModel(options)

        assert model.requires_gpu() is True

    def test_get_model_info(self):
        """Test model info retrieval."""
        options = CADAnnotationOptions(
            vlm_model="gpt-4-vision-preview",
            annotation_types=["hole", "pocket"],
            views_to_process=["front"],
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = FeatureRecognitionVlmModel(options)

        info = model.get_model_info()

        assert "vlm_model" in info
        assert "feature_types" in info
        assert "views" in info

    @patch("cadling.experimental.models.feature_recognition_vlm_model.ApiVlmModel")
    def test_error_handling(self, mock_vlm_class, mock_doc, mock_item):
        """Test error handling during recognition."""
        options = CADAnnotationOptions()

        # Mock VLM to raise error
        mock_vlm_instance = Mock()
        mock_vlm_instance.predict.side_effect = Exception("VLM error")
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = FeatureRecognitionVlmModel(options)

            # Should not crash
            model(mock_doc, [mock_item])

        # Should have error recorded
        assert "feature_recognition_error" in mock_item.properties
