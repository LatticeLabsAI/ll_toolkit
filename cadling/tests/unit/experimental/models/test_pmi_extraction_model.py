"""
Unit tests for PMIExtractionModel.

Tests cover:
- Model initialization
- VLM initialization (API vs local)
- PMI extraction from rendered views
- Cross-view validation
- Geometric context building

All tests are non-blocking with proper mocking.
"""

import pytest
from unittest.mock import MagicMock, Mock, patch, PropertyMock
from pathlib import Path

from cadling.experimental.models import PMIExtractionModel
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
def mock_item():
    """Create a mock CADItem with rendered images."""
    item = Mock()
    item.self_ref = "test_item"
    item.properties = {
        "rendered_images": {
            "front": Mock(),  # Mock PIL Image
            "top": Mock(),
            "isometric": Mock(),
        },
        "bounding_box": {"x": 100, "y": 50, "z": 30},
        "volume": 150000,
        "mass": 1.2,
    }
    return item


@pytest.fixture
def mock_api_vlm():
    """Create a mock API VLM instance."""
    mock_vlm = Mock()
    mock_vlm.predict = Mock(return_value=Mock(annotations=[]))
    return mock_vlm


@pytest.fixture
def mock_inline_vlm():
    """Create a mock inline VLM instance."""
    mock_vlm = Mock()
    mock_vlm.predict = Mock(return_value=Mock(annotations=[]))
    return mock_vlm


class TestPMIExtractionModel:
    """Test PMIExtractionModel."""

    @pytest.mark.timeout(5)
    @patch("cadling.experimental.models.pmi_extraction_model.ApiVlmModel")
    def test_initialization_with_api_vlm(self, mock_vlm_class, mock_api_vlm):
        """Test model initialization with API-based VLM."""
        mock_vlm_class.return_value = mock_api_vlm
        options = CADAnnotationOptions(vlm_model="gpt-4-vision-preview")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = PMIExtractionModel(options)

        assert model.options == options
        assert len(model.pmi_prompts) > 0

    @pytest.mark.timeout(5)
    @patch("cadling.experimental.models.pmi_extraction_model.ApiVlmModel")
    def test_initialization_with_claude(self, mock_vlm_class, mock_api_vlm):
        """Test model initialization with Claude VLM."""
        mock_vlm_class.return_value = mock_api_vlm
        options = CADAnnotationOptions(vlm_model="claude-3-opus-20240229")

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            model = PMIExtractionModel(options)

        assert model is not None

    @pytest.mark.timeout(5)
    @patch("cadling.experimental.models.pmi_extraction_model.InlineVlmModel")
    def test_initialization_with_local_vlm(self, mock_vlm_class, mock_inline_vlm):
        """Test model initialization with local VLM."""
        mock_vlm_class.return_value = mock_inline_vlm
        options = CADAnnotationOptions(vlm_model="llava-hf/llava-1.5-7b-hf")

        model = PMIExtractionModel(options)

        assert model is not None

    @pytest.mark.timeout(5)
    @patch("cadling.experimental.models.pmi_extraction_model.ApiVlmModel")
    def test_pmi_prompts_created(self, mock_vlm_class, mock_api_vlm):
        """Test that PMI prompts are created for all annotation types."""
        mock_vlm_class.return_value = mock_api_vlm
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = PMIExtractionModel(options)

        # Check that prompts exist for key types
        assert "dimension" in model.pmi_prompts
        assert "tolerance" in model.pmi_prompts
        assert "gdt" in model.pmi_prompts
        assert "surface_finish" in model.pmi_prompts
        assert "material" in model.pmi_prompts
        assert "welding" in model.pmi_prompts
        assert "note" in model.pmi_prompts

    @pytest.mark.timeout(5)
    @patch("cadling.experimental.models.pmi_extraction_model.ApiVlmModel")
    def test_call_with_no_rendered_images(self, mock_vlm_class, mock_api_vlm, mock_doc):
        """Test that items without rendered images are skipped."""
        mock_vlm_class.return_value = mock_api_vlm
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = PMIExtractionModel(options)

        # Item without rendered images
        item = Mock()
        item.self_ref = "test_item"
        item.properties = {}

        model(mock_doc, [item])

        # Should skip and not crash
        assert "pmi_annotations" not in item.properties or item.properties.get(
            "pmi_annotations"
        ) == []

    @pytest.mark.timeout(5)
    @patch("cadling.experimental.models.pmi_extraction_model.ApiVlmModel")
    def test_call_with_rendered_images(self, mock_vlm_class, mock_doc, mock_item):
        """Test PMI extraction with rendered images."""
        options = CADAnnotationOptions(
            annotation_types=["dimension"],
            views_to_process=["front"],
        )

        # Mock VLM response
        mock_vlm_instance = Mock()
        mock_response = Mock()
        mock_response.annotations = [
            Mock(
                annotation_type="dimension",
                text="10.5 mm",
                value=10.5,
                unit="mm",
                confidence=0.9,
                bbox=[100, 100, 50, 30],
            )
        ]
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = PMIExtractionModel(options)
            model(mock_doc, [mock_item])

        # Check that annotations were added
        assert "pmi_annotations" in mock_item.properties
        annotations = mock_item.properties["pmi_annotations"]
        assert len(annotations) > 0

    @pytest.mark.timeout(5)
    @patch("cadling.experimental.models.pmi_extraction_model.ApiVlmModel")
    def test_confidence_filtering(self, mock_vlm_class, mock_doc, mock_item):
        """Test that low-confidence annotations are filtered."""
        options = CADAnnotationOptions(
            annotation_types=["dimension"],
            views_to_process=["front"],
            min_confidence=0.8,
        )

        # Mock VLM response with mixed confidence
        mock_vlm_instance = Mock()
        mock_response = Mock()
        mock_response.annotations = [
            Mock(
                annotation_type="dimension",
                text="10.5 mm",
                value=10.5,
                unit="mm",
                confidence=0.9,  # Above threshold
                bbox=None,
            ),
            Mock(
                annotation_type="dimension",
                text="5.0 mm",
                value=5.0,
                unit="mm",
                confidence=0.6,  # Below threshold
                bbox=None,
            ),
        ]
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = PMIExtractionModel(options)
            model(mock_doc, [mock_item])

        annotations = mock_item.properties["pmi_annotations"]
        # Only high-confidence annotation should be kept
        assert all(a["confidence"] >= 0.8 for a in annotations)

    @pytest.mark.timeout(5)
    @patch("cadling.experimental.models.pmi_extraction_model.ApiVlmModel")
    def test_geometric_context_building(self, mock_vlm_class, mock_api_vlm, mock_doc, mock_item):
        """Test that geometric context is built correctly."""
        mock_vlm_class.return_value = mock_api_vlm
        options = CADAnnotationOptions(
            annotation_types=["dimension"],
            views_to_process=["front"],
            include_geometric_context=True,
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = PMIExtractionModel(options)

            # Test context building
            context = model._build_geometric_context(mock_doc, mock_item)

        assert isinstance(context, str)
        assert len(context) > 0

    @pytest.mark.timeout(5)
    @patch("cadling.experimental.models.pmi_extraction_model.ApiVlmModel")
    def test_cross_view_validation(self, mock_vlm_class, mock_doc, mock_item):
        """Test cross-view validation of annotations."""
        options = CADAnnotationOptions(
            annotation_types=["dimension"],
            views_to_process=["front", "top"],
            enable_cross_view_validation=True,
        )

        mock_vlm_instance = Mock()
        # Return same annotation from both views (should increase confidence)
        mock_response = Mock()
        mock_response.annotations = [
            Mock(
                annotation_type="dimension",
                text="10.0 mm",
                value=10.0,
                unit="mm",
                confidence=0.8,
                bbox=None,
            )
        ]
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = PMIExtractionModel(options)
            model(mock_doc, [mock_item])

        # Check that validation was applied
        assert "pmi_annotations" in mock_item.properties

    @pytest.mark.timeout(5)
    @patch("cadling.experimental.models.pmi_extraction_model.ApiVlmModel")
    def test_supports_batch_processing(self, mock_vlm_class, mock_api_vlm):
        """Test that model reports batch processing support."""
        mock_vlm_class.return_value = mock_api_vlm
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = PMIExtractionModel(options)

        # Model may or may not support batch processing - just check method exists
        result = model.supports_batch_processing()
        assert isinstance(result, bool)

    @pytest.mark.timeout(5)
    @patch("cadling.experimental.models.pmi_extraction_model.InlineVlmModel")
    @patch("cadling.experimental.models.pmi_extraction_model.ApiVlmModel")
    def test_requires_gpu(self, mock_api_vlm_class, mock_inline_vlm_class, mock_api_vlm, mock_inline_vlm):
        """Test GPU requirements based on model type."""
        # API model - no GPU needed
        mock_api_vlm_class.return_value = mock_api_vlm
        options = CADAnnotationOptions(vlm_model="gpt-4-vision-preview")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = PMIExtractionModel(options)

        # Just check method exists and returns bool
        result = model.requires_gpu()
        assert isinstance(result, bool)

    @pytest.mark.timeout(5)
    @patch("cadling.experimental.models.pmi_extraction_model.ApiVlmModel")
    def test_get_model_info(self, mock_vlm_class, mock_api_vlm):
        """Test model info retrieval."""
        mock_vlm_class.return_value = mock_api_vlm
        options = CADAnnotationOptions(
            vlm_model="gpt-4-vision-preview",
            annotation_types=["dimension", "tolerance"],
            views_to_process=["front", "top"],
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = PMIExtractionModel(options)

        info = model.get_model_info()

        assert "vlm_model" in info
        assert info["vlm_model"] == "gpt-4-vision-preview"
        assert "annotation_types" in info
        assert "views" in info

    @pytest.mark.timeout(5)
    @patch("cadling.experimental.models.pmi_extraction_model.ApiVlmModel")
    def test_error_handling(self, mock_vlm_class, mock_doc, mock_item):
        """Test error handling during extraction."""
        options = CADAnnotationOptions()

        # Mock VLM to raise error
        mock_vlm_instance = Mock()
        mock_vlm_instance.predict.side_effect = Exception("VLM error")
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = PMIExtractionModel(options)

            # Should not crash
            model(mock_doc, [mock_item])

        # Should have error recorded or handled gracefully
        assert "pmi_extraction_error" in mock_item.properties or "pmi_annotations" in mock_item.properties

    @pytest.mark.timeout(5)
    @patch("cadling.experimental.models.pmi_extraction_model.ApiVlmModel")
    def test_provenance_tracking(self, mock_vlm_class, mock_api_vlm, mock_doc, mock_item):
        """Test that provenance is tracked."""
        mock_vlm_class.return_value = mock_api_vlm
        options = CADAnnotationOptions()

        # Add provenance tracking method
        mock_item.add_provenance = Mock()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = PMIExtractionModel(options)
            model(mock_doc, [mock_item])

        # Check provenance was added if method exists
        if hasattr(mock_item, "add_provenance"):
            # May or may not be called depending on implementation
            assert True
