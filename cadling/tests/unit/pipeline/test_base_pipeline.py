"""Unit tests for base pipeline."""

import pytest
from pathlib import Path

from cadling.datamodel.base_models import (
    CADInputDocument,
    CADlingDocument,
    ConversionResult,
    ConversionStatus,
    InputFormat,
)
from cadling.datamodel.pipeline_options import PipelineOptions
from cadling.pipeline.base_pipeline import BaseCADPipeline, EnrichmentModel


class MockEnrichmentModel(EnrichmentModel):
    """Mock enrichment model for testing."""

    def __call__(self, doc, item_batch):
        """Add mock property to items."""
        for item in item_batch:
            item.properties["mock_enriched"] = True


class MockPipeline(BaseCADPipeline):
    """Mock pipeline for testing."""

    def _build_document(self, conv_res):
        """Mock build that creates a simple document."""
        conv_res.document = CADlingDocument(name="mock_doc.step")
        return conv_res


class TestBaseCADPipeline:
    """Test BaseCADPipeline class."""

    def test_initialization(self):
        """Test pipeline initialization."""
        options = PipelineOptions()
        pipeline = MockPipeline(options)

        assert pipeline.pipeline_options == options
        assert pipeline.enrichment_pipe == []

    def test_initialization_with_enrichment(self):
        """Test pipeline initialization with enrichment models."""
        model = MockEnrichmentModel()
        options = PipelineOptions(enrichment_models=[model])
        pipeline = MockPipeline(options)

        assert len(pipeline.enrichment_pipe) == 1

    def test_execute_success(self):
        """Test successful pipeline execution."""
        options = PipelineOptions()
        pipeline = MockPipeline(options)

        input_doc = CADInputDocument(
            file=Path("test.step"),
            format=InputFormat.STEP,
            document_hash="abc123",
        )

        result = pipeline.execute(input_doc)

        assert result.status == ConversionStatus.SUCCESS
        assert result.document is not None
        assert result.document.name == "mock_doc.step"

    def test_execute_with_enrichment(self):
        """Test pipeline execution with enrichment."""
        from cadling.datamodel.base_models import CADItem, CADItemLabel

        model = MockEnrichmentModel()
        options = PipelineOptions(enrichment_models=[model])
        pipeline = MockPipeline(options)

        # Override build to add items
        def build_with_items(conv_res):
            doc = CADlingDocument(name="mock_doc.step")
            doc.add_item(
                CADItem(
                    item_type="test",
                    label=CADItemLabel(text="Test"),
                )
            )
            conv_res.document = doc
            return conv_res

        pipeline._build_document = build_with_items

        input_doc = CADInputDocument(
            file=Path("test.step"),
            format=InputFormat.STEP,
            document_hash="abc123",
        )

        result = pipeline.execute(input_doc)

        assert result.status == ConversionStatus.SUCCESS
        assert len(result.document.items) == 1
        assert result.document.items[0].properties["mock_enriched"] is True

    def test_determine_status_success(self):
        """Test status determination for success."""
        options = PipelineOptions()
        pipeline = MockPipeline(options)

        input_doc = CADInputDocument(
            file=Path("test.step"),
            format=InputFormat.STEP,
            document_hash="abc123",
        )

        conv_res = ConversionResult(input=input_doc)
        conv_res.document = CADlingDocument(name="test.step")

        status = pipeline._determine_status(conv_res)
        assert status == ConversionStatus.SUCCESS

    def test_determine_status_failure(self):
        """Test status determination for failure."""
        options = PipelineOptions()
        pipeline = MockPipeline(options)

        input_doc = CADInputDocument(
            file=Path("test.step"),
            format=InputFormat.STEP,
            document_hash="abc123",
        )

        conv_res = ConversionResult(input=input_doc)
        # No document means failure

        status = pipeline._determine_status(conv_res)
        assert status == ConversionStatus.FAILURE

    def test_determine_status_partial(self):
        """Test status determination for partial success."""
        options = PipelineOptions()
        pipeline = MockPipeline(options)

        input_doc = CADInputDocument(
            file=Path("test.step"),
            format=InputFormat.STEP,
            document_hash="abc123",
        )

        conv_res = ConversionResult(input=input_doc)
        conv_res.document = CADlingDocument(name="test.step")
        conv_res.add_error("TestComponent", "Test error")

        status = pipeline._determine_status(conv_res)
        assert status == ConversionStatus.PARTIAL
