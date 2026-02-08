"""
Unit tests for MultiViewFusionPipeline.

Tests cover:
- Pipeline initialization
- Multi-view rendering
- Per-view VLM analysis
- Fusion strategies (weighted_consensus, majority_vote, hierarchical)
- Consistency checking
- Error handling
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from cadling.experimental.pipeline import MultiViewFusionPipeline
from cadling.experimental.datamodel import MultiViewOptions, ViewConfig
from cadling.datamodel.base_models import (
    ConversionResult,
    ConversionStatus,
    CADInputDocument,
    InputFormat,
)


@pytest.fixture
def mock_input_doc():
    """Create a mock input document."""
    input_doc = CADInputDocument(
        file=Path("/tmp/test.step"),
        format=InputFormat.STEP,
        document_hash="test123",
    )
    backend = Mock()
    backend.convert = Mock()
    input_doc._backend = backend
    return input_doc


@pytest.fixture
def mock_converted_doc():
    """Create a mock converted CAD document."""
    doc = Mock()
    doc.topology = {"num_faces": 10}

    item = Mock()
    item.self_ref = "part_1"
    item.properties = {}

    doc.items = [item]
    return doc


@pytest.fixture
def mock_conversion_result(mock_input_doc, mock_converted_doc):
    """Create a mock conversion result."""
    conv_res = ConversionResult(input=mock_input_doc)
    mock_input_doc._backend.convert.return_value = mock_converted_doc
    return conv_res


class TestMultiViewFusionPipeline:
    """Test MultiViewFusionPipeline."""

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_initialization(self, mock_feature_model, mock_pmi_model):
        """Test pipeline initialization."""
        options = MultiViewOptions(
            fusion_strategy="weighted_consensus",
            views=[
                ViewConfig(name="front", azimuth=0, elevation=0),
                ViewConfig(name="top", azimuth=0, elevation=90),
            ],
        )

        pipeline = MultiViewFusionPipeline(options)

        assert pipeline.options == options
        assert pipeline.fusion_strategy == "weighted_consensus"
        assert len(pipeline.enrichment_pipe) > 0

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_invalid_fusion_strategy(self, mock_feature_model, mock_pmi_model):
        """Test that invalid fusion strategy raises error."""
        options = MultiViewOptions(fusion_strategy="invalid_strategy")

        with pytest.raises(ValueError, match="Invalid fusion_strategy"):
            MultiViewFusionPipeline(options)

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_get_default_options(self, mock_feature_model, mock_pmi_model):
        """Test default options."""
        options = MultiViewFusionPipeline.get_default_options()

        assert isinstance(options, MultiViewOptions)
        assert len(options.views) > 0

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_build_document(
        self, mock_feature_model, mock_pmi_model, mock_conversion_result
    ):
        """Test build stage with multi-view rendering."""
        options = MultiViewOptions(
            views=[
                ViewConfig(name="front", azimuth=0, elevation=0),
                ViewConfig(name="top", azimuth=0, elevation=90),
            ]
        )

        pipeline = MultiViewFusionPipeline(options)
        conv_res = pipeline._build_document(mock_conversion_result)

        # Check document created
        assert conv_res.document is not None

        # Check views rendered
        item = conv_res.document.items[0]
        assert "rendered_images" in item.properties
        assert "num_views" in item.properties

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_render_all_views(
        self, mock_feature_model, mock_pmi_model, mock_converted_doc
    ):
        """Test rendering all configured views."""
        options = MultiViewOptions(
            views=[
                ViewConfig(name="front", azimuth=0, elevation=0),
                ViewConfig(name="top", azimuth=0, elevation=90),
                ViewConfig(name="isometric", azimuth=45, elevation=35.264),
            ]
        )

        pipeline = MultiViewFusionPipeline(options)
        item = mock_converted_doc.items[0]

        rendered_views = pipeline._render_all_views(item)

        # Should be a dictionary (even if empty placeholder)
        assert isinstance(rendered_views, dict)

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_fuse_weighted_consensus(self, mock_feature_model, mock_pmi_model):
        """Test weighted consensus fusion strategy."""
        options = MultiViewOptions(fusion_strategy="weighted_consensus")
        pipeline = MultiViewFusionPipeline(options)

        # Items appearing in multiple views with consistent values
        items = [
            {
                "text": "10mm",
                "value": 10.0,
                "confidence": 0.8,
                "view": "front",
            },
            {
                "text": "10mm",
                "value": 10.1,  # Slightly different but within tolerance
                "confidence": 0.85,
                "view": "top",
            },
            {
                "text": "5mm",
                "value": 5.0,
                "confidence": 0.7,
                "view": "front",
            },
        ]

        fused = pipeline._fuse_weighted_consensus(items)

        # Should have 2 unique items
        assert len(fused) == 2

        # Check "10mm" was merged with boosted confidence
        merged_10mm = next((f for f in fused if f.get("text") == "10mm"), None)
        assert merged_10mm is not None
        assert merged_10mm["view_count"] == 2
        assert merged_10mm["confidence"] > 0.8  # Boosted
        assert merged_10mm["fusion_method"] == "weighted_consensus"

        # Check "5mm" kept as single view
        single_5mm = next((f for f in fused if f.get("text") == "5mm"), None)
        assert single_5mm is not None
        assert single_5mm["view_count"] == 1
        assert single_5mm["fusion_method"] == "single_view"

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_fuse_majority_vote(self, mock_feature_model, mock_pmi_model):
        """Test majority vote fusion strategy."""
        options = MultiViewOptions(
            fusion_strategy="majority_vote",
            views=[
                ViewConfig(name="front"),
                ViewConfig(name="top"),
                ViewConfig(name="right"),
            ],
        )
        pipeline = MultiViewFusionPipeline(options)

        # Items with different view counts
        items = [
            {"feature_type": "hole", "view": "front"},
            {"feature_type": "hole", "view": "top"},  # Appears in 2/3 views (majority)
            {"feature_type": "pocket", "view": "front"},  # Only 1/3 views (not majority)
        ]

        fused = pipeline._fuse_majority_vote(items)

        # Only "hole" should pass (2 >= majority threshold of 2)
        assert len(fused) == 1
        assert fused[0]["feature_type"] == "hole"
        assert fused[0]["view_count"] == 2
        assert fused[0]["fusion_method"] == "majority_vote"

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_fuse_hierarchical(self, mock_feature_model, mock_pmi_model):
        """Test hierarchical fusion strategy."""
        options = MultiViewOptions(fusion_strategy="hierarchical")
        pipeline = MultiViewFusionPipeline(options)

        # Same item from different views with different confidences
        items = [
            {"feature_type": "hole", "confidence": 0.7, "view": "right"},  # Priority 2
            {"feature_type": "hole", "confidence": 0.8, "view": "front"},  # Priority 3 (higher)
            {"feature_type": "hole", "confidence": 0.9, "view": "back"},  # Priority 1 (lower)
        ]

        fused = pipeline._fuse_hierarchical(items)

        # Should select front view (highest priority * confidence)
        assert len(fused) == 1
        assert fused[0]["view"] == "front"
        assert fused[0]["fusion_method"] == "hierarchical"
        assert "priority_score" in fused[0]

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_check_consistency_consistent(self, mock_feature_model, mock_pmi_model):
        """Test consistency checking with consistent values."""
        options = MultiViewOptions()
        pipeline = MultiViewFusionPipeline(options)

        items = [
            {"parameters": {"diameter": 10.0}},
            {"parameters": {"diameter": 10.05}},  # Within 10% tolerance
        ]

        consistent = pipeline._check_consistency(items)
        assert consistent is True

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_check_consistency_inconsistent(self, mock_feature_model, mock_pmi_model):
        """Test consistency checking with inconsistent values."""
        options = MultiViewOptions()
        pipeline = MultiViewFusionPipeline(options)

        items = [
            {"parameters": {"diameter": 10.0}},
            {"parameters": {"diameter": 20.0}},  # More than 10% different
        ]

        consistent = pipeline._check_consistency(items)
        assert consistent is False

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_fuse_multi_view_results(
        self, mock_feature_model, mock_pmi_model, mock_converted_doc
    ):
        """Test complete multi-view fusion."""
        options = MultiViewOptions(fusion_strategy="weighted_consensus")
        pipeline = MultiViewFusionPipeline(options)

        item = mock_converted_doc.items[0]
        item.properties = {
            "pmi_annotations": [
                {"text": "10mm", "value": 10.0, "view": "front"},
                {"text": "10mm", "value": 10.0, "view": "top"},
            ],
            "machining_features": [
                {"feature_type": "hole", "view": "front"},
            ],
        }

        pipeline._fuse_multi_view_results(item)

        # Check fusion metadata added
        assert "fusion_strategy" in item.properties
        assert item.properties["fusion_strategy"] == "weighted_consensus"
        assert "num_views_processed" in item.properties

        # Check fused results added
        assert "fused_pmi_annotations" in item.properties
        assert "fused_machining_features" in item.properties

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_enrich_document(
        self, mock_feature_model_class, mock_pmi_model_class, mock_conversion_result
    ):
        """Test enrichment stage with fusion."""
        options = MultiViewOptions()

        # Mock enrichment models
        mock_pmi_instance = Mock()
        mock_feature_instance = Mock()
        mock_pmi_model_class.return_value = mock_pmi_instance
        mock_feature_model_class.return_value = mock_feature_instance

        pipeline = MultiViewFusionPipeline(options)

        # Complete build first
        conv_res = pipeline._build_document(mock_conversion_result)

        # Add some mock results
        conv_res.document.items[0].properties["pmi_annotations"] = []
        conv_res.document.items[0].properties["machining_features"] = []

        # Run enrichment
        conv_res = pipeline._enrich_document(conv_res)

        # Check fusion was performed
        item = conv_res.document.items[0]
        assert "fusion_strategy" in item.properties

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_determine_status_success(
        self, mock_feature_model, mock_pmi_model, mock_conversion_result
    ):
        """Test status determination - success."""
        options = MultiViewOptions()
        pipeline = MultiViewFusionPipeline(options)

        pipeline._build_document(mock_conversion_result)

        status = pipeline._determine_status(mock_conversion_result)
        assert status == ConversionStatus.SUCCESS

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_determine_status_failure(self, mock_feature_model, mock_pmi_model):
        """Test status determination - failure."""
        options = MultiViewOptions()
        pipeline = MultiViewFusionPipeline(options)

        mock_input = CADInputDocument(
            file=Path("/tmp/test.step"),
            format=InputFormat.STEP,
            document_hash="test123",
        )
        conv_res = ConversionResult(input=mock_input)
        conv_res.document = None

        status = pipeline._determine_status(conv_res)
        assert status == ConversionStatus.FAILURE

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_conflict_detection(self, mock_feature_model, mock_pmi_model):
        """Test that conflicts are detected and flagged."""
        options = MultiViewOptions(fusion_strategy="weighted_consensus")
        pipeline = MultiViewFusionPipeline(options)

        # Items with same key but very different values (conflict)
        items = [
            {"text": "dimension", "value": 10.0, "view": "front"},
            {"text": "dimension", "value": 20.0, "view": "top"},  # Inconsistent!
        ]

        fused = pipeline._fuse_weighted_consensus(items)

        # Should keep both and flag conflict
        conflict_items = [f for f in fused if f.get("conflict_detected")]
        assert len(conflict_items) > 0

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_error_handling_build(
        self, mock_feature_model, mock_pmi_model, mock_conversion_result
    ):
        """Test error handling in build stage."""
        options = MultiViewOptions()

        # Make backend raise error
        mock_conversion_result.input._backend.convert.side_effect = Exception(
            "Conversion error"
        )

        pipeline = MultiViewFusionPipeline(options)

        with pytest.raises(Exception, match="Conversion error"):
            pipeline._build_document(mock_conversion_result)

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_error_handling_enrich(
        self, mock_feature_model_class, mock_pmi_model_class, mock_conversion_result
    ):
        """Test error handling in enrichment stage."""
        options = MultiViewOptions()

        # Mock enrichment model to raise error
        mock_pmi_instance = Mock()
        mock_pmi_instance.side_effect = Exception("VLM error")
        mock_pmi_model_class.return_value = mock_pmi_instance

        pipeline = MultiViewFusionPipeline(options)

        # Complete build
        conv_res = pipeline._build_document(mock_conversion_result)

        # Enrichment should handle error gracefully
        conv_res = pipeline._enrich_document(conv_res)

        # Should not crash
        assert conv_res is not None

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_multiple_items(
        self, mock_feature_model, mock_pmi_model, mock_conversion_result
    ):
        """Test processing multiple items."""
        options = MultiViewOptions()

        # Create doc with multiple items
        doc = Mock()
        item1 = Mock()
        item1.self_ref = "part_1"
        item1.properties = {}
        item2 = Mock()
        item2.self_ref = "part_2"
        item2.properties = {}
        doc.items = [item1, item2]

        mock_conversion_result.input._backend.convert.return_value = doc

        pipeline = MultiViewFusionPipeline(options)
        conv_res = pipeline._build_document(mock_conversion_result)

        # Both items should be processed
        for item in conv_res.document.items:
            assert "rendered_images" in item.properties
            assert "num_views" in item.properties

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_empty_annotations(self, mock_feature_model, mock_pmi_model):
        """Test fusion with empty annotation lists."""
        options = MultiViewOptions()
        pipeline = MultiViewFusionPipeline(options)

        fused = pipeline._fuse_annotations([])
        assert fused == []

    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.PMIExtractionModel")
    @patch("cadling.experimental.pipeline.multi_view_fusion_pipeline.FeatureRecognitionVlmModel")
    def test_view_count_tracking(self, mock_feature_model, mock_pmi_model):
        """Test that view count is properly tracked in fused results."""
        options = MultiViewOptions(fusion_strategy="weighted_consensus")
        pipeline = MultiViewFusionPipeline(options)

        items = [
            {"feature_type": "hole", "view": "front"},
            {"feature_type": "hole", "view": "top"},
            {"feature_type": "hole", "view": "right"},
        ]

        fused = pipeline._fuse_weighted_consensus(items)

        # Should have view count
        assert len(fused) == 1
        assert fused[0]["view_count"] == 3
        assert len(fused[0]["views"]) == 3
