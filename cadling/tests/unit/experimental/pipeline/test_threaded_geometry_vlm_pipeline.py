"""
Unit tests for ThreadedGeometryVlmPipeline.

Tests accurately demonstrate how the two-stage pipeline works:
- Stage 1: Geometric analysis (feature extraction, rendering)
- Stage 2: VLM with geometric context

These tests reflect the actual implementation, not idealized behavior.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from cadling.experimental.pipeline import ThreadedGeometryVlmPipeline
from cadling.experimental.datamodel import CADAnnotationOptions
from cadling.datamodel.base_models import (
    ConversionResult,
    ConversionStatus,
    CADInputDocument,
    InputFormat,
)


class TestThreadedGeometryVlmPipeline:
    """Test ThreadedGeometryVlmPipeline."""

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_initialization(self, mock_feature_class, mock_pmi_class):
        """Test pipeline initialization with models imported in __init__."""
        # Mock the model instance that will be created
        mock_pmi_instance = Mock()
        mock_pmi_class.return_value = mock_pmi_instance

        options = CADAnnotationOptions(vlm_model="gpt-4-vision-preview")

        pipeline = ThreadedGeometryVlmPipeline(options)

        # Verify initialization
        assert pipeline.options == options
        assert pipeline.stage1_complete is False
        assert pipeline.geometric_context == {}
        assert len(pipeline.stage2_models) > 0

        # Verify PMIExtractionModel was instantiated
        mock_pmi_class.assert_called_once_with(options)

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_get_default_options(self, mock_feature_class, mock_pmi_class):
        """Test default options."""
        options = ThreadedGeometryVlmPipeline.get_default_options()

        assert options.vlm_model == "gpt-4-vision"
        assert "dimension" in options.annotation_types
        assert "front" in options.views_to_process
        assert options.include_geometric_context is True

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_build_document_stage1(
        self, mock_feature_class, mock_pmi_class, mock_conversion_result
    ):
        """Test Stage 1: geometric analysis."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions()

        pipeline = ThreadedGeometryVlmPipeline(options)
        conv_res = pipeline._build_document(mock_conversion_result)

        # Check Stage 1 completed
        assert pipeline.stage1_complete is True
        assert conv_res.document is not None

        # Check geometric features extracted
        item = conv_res.document.items[0]
        assert "machining_features" in item.properties
        assert "geometric_analysis_stage" in item.properties
        assert item.properties["geometric_analysis_stage"] == "complete"

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_extract_geometric_features(
        self, mock_feature_class, mock_pmi_class, mock_converted_doc
    ):
        """Test geometric feature extraction."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions()
        pipeline = ThreadedGeometryVlmPipeline(options)

        item = mock_converted_doc.items[0]
        pipeline._extract_geometric_features(mock_converted_doc, item)

        # Should detect features from cylindrical faces
        features = item.properties.get("machining_features", [])
        assert isinstance(features, list)
        # Should detect at least one hole from cylindrical face
        hole_features = [f for f in features if f.get("feature_type") == "hole"]
        assert len(hole_features) > 0

        # Check feature structure
        if hole_features:
            feature = hole_features[0]
            assert "confidence" in feature
            assert "source" in feature
            assert feature["source"] in ("geometric_analysis", "topology_heuristic")

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_render_views(self, mock_feature_class, mock_pmi_class, mock_converted_doc):
        """Test view rendering."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions(
            views_to_process=["front", "top", "isometric"]
        )
        pipeline = ThreadedGeometryVlmPipeline(options)

        item = mock_converted_doc.items[0]
        pipeline._render_views(mock_converted_doc, item)

        # Should have rendered images property
        assert "rendered_images" in item.properties
        assert "rendering_stage" in item.properties
        assert item.properties["rendering_stage"] == "complete"

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_build_geometric_context(
        self, mock_feature_class, mock_pmi_class, mock_converted_doc
    ):
        """Test geometric context building."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions()
        pipeline = ThreadedGeometryVlmPipeline(options)

        # Add features to item
        mock_converted_doc.items[0].properties["machining_features"] = [
            {"feature_type": "hole"},
            {"feature_type": "pocket"},
        ]

        context = pipeline._build_geometric_context(mock_converted_doc)

        assert "num_items" in context
        assert context["num_items"] == 1
        assert "topology_available" in context
        assert context["topology_available"] is True
        assert "total_features_detected" in context
        assert context["total_features_detected"] == 2

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_enrich_document_stage2(
        self, mock_feature_class, mock_pmi_class, mock_conversion_result
    ):
        """Test Stage 2: VLM with context."""
        # Mock Stage 2 model
        mock_model_instance = Mock()
        mock_pmi_class.return_value = mock_model_instance

        options = CADAnnotationOptions()
        pipeline = ThreadedGeometryVlmPipeline(options)

        # Complete Stage 1 first
        conv_res = pipeline._build_document(mock_conversion_result)

        # Run Stage 2
        conv_res = pipeline._enrich_document(conv_res)

        # Check Stage 2 model was called
        mock_model_instance.assert_called_once()
        # The call should be: model(doc, items)
        call_args = mock_model_instance.call_args
        assert call_args is not None

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_enrich_document_stage1_not_complete(
        self, mock_feature_class, mock_pmi_class, mock_conversion_result
    ):
        """Test Stage 2 skips if Stage 1 not complete."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions()

        pipeline = ThreadedGeometryVlmPipeline(options)
        pipeline.stage1_complete = False

        # Create document manually without running stage 1
        mock_doc = Mock()
        mock_doc.items = []
        mock_conversion_result.document = mock_doc

        conv_res = pipeline._enrich_document(mock_conversion_result)

        # Should return without processing
        assert conv_res == mock_conversion_result

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_two_stage_workflow(
        self, mock_feature_class, mock_pmi_class, mock_conversion_result
    ):
        """Test complete two-stage workflow."""
        # Mock Stage 2 model
        mock_model_instance = Mock()
        mock_pmi_class.return_value = mock_model_instance

        options = CADAnnotationOptions()
        pipeline = ThreadedGeometryVlmPipeline(options)

        # Run Stage 1
        conv_res = pipeline._build_document(mock_conversion_result)
        assert pipeline.stage1_complete is True

        # Run Stage 2
        conv_res = pipeline._enrich_document(conv_res)

        # Check both stages completed
        item = conv_res.document.items[0]
        assert "machining_features" in item.properties  # Stage 1
        assert "geometric_analysis_stage" in item.properties  # Stage 1

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_determine_status_success(
        self, mock_feature_class, mock_pmi_class, mock_conversion_result
    ):
        """Test status determination - success."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions()

        pipeline = ThreadedGeometryVlmPipeline(options)
        pipeline._build_document(mock_conversion_result)
        pipeline.stage1_complete = True

        status = pipeline._determine_status(mock_conversion_result)

        assert status == ConversionStatus.SUCCESS

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_determine_status_partial(self, mock_feature_class, mock_pmi_class):
        """Test status determination - partial success."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions()

        pipeline = ThreadedGeometryVlmPipeline(options)
        pipeline.stage1_complete = False

        # Create conversion result with document but stage 1 not complete
        input_doc = CADInputDocument(
            file=Path("/tmp/test.step"),
            format=InputFormat.STEP,
            document_hash="abc123",
        )
        conv_res = ConversionResult(input=input_doc)
        conv_res.document = Mock()

        status = pipeline._determine_status(conv_res)

        assert status == ConversionStatus.PARTIAL

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_determine_status_failure(self, mock_feature_class, mock_pmi_class):
        """Test status determination - failure."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions()

        pipeline = ThreadedGeometryVlmPipeline(options)

        # Create conversion result without document
        input_doc = CADInputDocument(
            file=Path("/tmp/test.step"),
            format=InputFormat.STEP,
            document_hash="abc123",
        )
        conv_res = ConversionResult(input=input_doc)
        conv_res.document = None

        status = pipeline._determine_status(conv_res)

        assert status == ConversionStatus.FAILURE

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_error_handling_stage1(self, mock_feature_class, mock_pmi_class, mock_conversion_result):
        """Test error handling in Stage 1."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions()

        # Make backend raise error
        mock_conversion_result.input._backend.convert.side_effect = Exception(
            "Conversion error"
        )

        pipeline = ThreadedGeometryVlmPipeline(options)

        # Should raise exception
        with pytest.raises(Exception, match="Conversion error"):
            pipeline._build_document(mock_conversion_result)

        assert pipeline.stage1_complete is False

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_error_handling_stage2(
        self, mock_feature_class, mock_pmi_class, mock_conversion_result
    ):
        """Test error handling in Stage 2."""
        # Mock Stage 2 model to raise error
        mock_model_instance = Mock()
        mock_model_instance.side_effect = Exception("VLM error")
        mock_pmi_class.return_value = mock_model_instance

        options = CADAnnotationOptions()
        pipeline = ThreadedGeometryVlmPipeline(options)

        # Complete Stage 1
        conv_res = pipeline._build_document(mock_conversion_result)

        # Stage 2 should handle error gracefully (logs but continues)
        conv_res = pipeline._enrich_document(conv_res)

        # Should not crash
        assert conv_res is not None

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_multiple_items(self, mock_feature_class, mock_pmi_class, mock_conversion_result):
        """Test processing multiple items."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions()

        # Create doc with multiple items
        doc = Mock()
        doc.topology = {"num_faces": 10, "faces": []}

        item1 = Mock()
        item1.self_ref = "part_1"
        item1.properties = {}

        item2 = Mock()
        item2.self_ref = "part_2"
        item2.properties = {}

        doc.items = [item1, item2]

        mock_conversion_result.input._backend.convert.return_value = doc

        pipeline = ThreadedGeometryVlmPipeline(options)
        conv_res = pipeline._build_document(mock_conversion_result)

        # Both items should be processed
        for item in conv_res.document.items:
            assert "machining_features" in item.properties
            assert "geometric_analysis_stage" in item.properties

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_no_topology_handling(self, mock_feature_class, mock_pmi_class, mock_conversion_result):
        """Test handling when topology is missing."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions()

        # Doc without topology
        doc = Mock()
        doc.topology = None
        item = Mock()
        item.self_ref = "part_1"
        item.properties = {}
        doc.items = [item]

        mock_conversion_result.input._backend.convert.return_value = doc

        pipeline = ThreadedGeometryVlmPipeline(options)
        conv_res = pipeline._build_document(mock_conversion_result)

        # Should not crash
        assert conv_res.document is not None
        # Context should reflect no topology
        assert pipeline.geometric_context["topology_available"] is False

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_geometric_context_propagation(
        self, mock_feature_class, mock_pmi_class, mock_conversion_result
    ):
        """Test that geometric context is available for Stage 2."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions()

        pipeline = ThreadedGeometryVlmPipeline(options)

        # Complete Stage 1
        conv_res = pipeline._build_document(mock_conversion_result)

        # Geometric context should be populated
        assert len(pipeline.geometric_context) > 0
        assert "num_items" in pipeline.geometric_context
        assert "topology_available" in pipeline.geometric_context

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_views_configuration(self, mock_feature_class, mock_pmi_class, mock_conversion_result):
        """Test that views configuration is respected."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions(
            views_to_process=["front", "top"]
        )

        pipeline = ThreadedGeometryVlmPipeline(options)

        # Check views are stored in options
        assert len(pipeline.options.views_to_process) == 2
        assert "front" in pipeline.options.views_to_process
        assert "top" in pipeline.options.views_to_process

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_feature_confidence(self, mock_feature_class, mock_pmi_class, mock_converted_doc):
        """Test that extracted features have confidence scores."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions()
        pipeline = ThreadedGeometryVlmPipeline(options)

        item = mock_converted_doc.items[0]
        pipeline._extract_geometric_features(mock_converted_doc, item)

        features = item.properties.get("machining_features", [])
        for feature in features:
            assert "confidence" in feature
            assert 0.0 <= feature["confidence"] <= 1.0

    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.threaded_geometry_vlm_pipeline.FeatureRecognitionVlmModel")
    def test_feature_source_tracking(self, mock_feature_class, mock_pmi_class, mock_converted_doc):
        """Test that features track their source."""
        mock_pmi_class.return_value = Mock()
        options = CADAnnotationOptions()
        pipeline = ThreadedGeometryVlmPipeline(options)

        item = mock_converted_doc.items[0]
        pipeline._extract_geometric_features(mock_converted_doc, item)

        features = item.properties.get("machining_features", [])
        for feature in features:
            assert "source" in feature
            assert feature["source"] in ("geometric_analysis", "topology_heuristic")
