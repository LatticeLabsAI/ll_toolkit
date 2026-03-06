"""Unit tests for base data models."""

import pytest
from pathlib import Path

from cadling.datamodel.base_models import (
    BoundingBox3D,
    CADDocumentOrigin,
    CADInputDocument,
    CADItem,
    CADItemLabel,
    CADlingDocument,
    ConversionResult,
    ConversionStatus,
    InputFormat,
    TopologyGraph,
)


class TestBoundingBox3D:
    """Test BoundingBox3D model."""

    def test_creation(self):
        """Test creating a 3D bounding box."""
        bbox = BoundingBox3D(
            x_min=0.0,
            y_min=0.0,
            z_min=0.0,
            x_max=10.0,
            y_max=10.0,
            z_max=10.0,
        )

        assert bbox.x_min == 0.0
        assert bbox.x_max == 10.0

    def test_center(self):
        """Test center calculation."""
        bbox = BoundingBox3D(
            x_min=0.0,
            y_min=0.0,
            z_min=0.0,
            x_max=10.0,
            y_max=10.0,
            z_max=10.0,
        )

        assert bbox.center == (5.0, 5.0, 5.0)

    def test_size(self):
        """Test size calculation."""
        bbox = BoundingBox3D(
            x_min=0.0,
            y_min=0.0,
            z_min=0.0,
            x_max=10.0,
            y_max=20.0,
            z_max=30.0,
        )

        assert bbox.size == (10.0, 20.0, 30.0)

    def test_volume(self):
        """Test volume calculation."""
        bbox = BoundingBox3D(
            x_min=0.0,
            y_min=0.0,
            z_min=0.0,
            x_max=10.0,
            y_max=10.0,
            z_max=10.0,
        )

        assert bbox.volume == 1000.0


class TestTopologyGraph:
    """Test TopologyGraph model."""

    def test_creation(self):
        """Test creating a topology graph."""
        graph = TopologyGraph(num_nodes=3)

        assert graph.num_nodes == 3
        assert graph.num_edges == 0
        assert graph.adjacency_list == {}

    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = TopologyGraph(num_nodes=3)

        graph.add_edge(1, 2)
        graph.add_edge(1, 3)

        assert graph.num_edges == 2
        assert graph.adjacency_list[1] == [2, 3]

    def test_get_neighbors(self):
        """Test getting neighbors."""
        graph = TopologyGraph(num_nodes=3)

        graph.add_edge(1, 2)
        graph.add_edge(1, 3)

        neighbors = graph.get_neighbors(1)
        assert neighbors == [2, 3]

        # Non-existent node returns empty list
        assert graph.get_neighbors(99) == []


class TestCADItem:
    """Test CADItem model."""

    def test_creation(self):
        """Test creating a CAD item."""
        item = CADItem(
            item_type="test_item",
            label=CADItemLabel(text="Test Item"),
        )

        assert item.item_type == "test_item"
        assert item.label.text == "Test Item"
        assert item.properties == {}

    def test_add_provenance(self):
        """Test adding provenance."""
        item = CADItem(
            item_type="test_item",
            label=CADItemLabel(text="Test Item"),
        )

        item.add_provenance("backend", "TestBackend")

        assert len(item.prov) == 1
        assert item.prov[0].component_type == "backend"
        assert item.prov[0].component_name == "TestBackend"


class TestCADlingDocument:
    """Test CADlingDocument model."""

    def test_creation(self):
        """Test creating a CAD document."""
        doc = CADlingDocument(name="test.step")

        assert doc.name == "test.step"
        assert doc.items == []
        assert doc.topology is None

    def test_add_item(self):
        """Test adding items to document."""
        doc = CADlingDocument(name="test.step")

        item1 = CADItem(
            item_type="entity",
            label=CADItemLabel(text="Entity 1"),
        )
        item2 = CADItem(
            item_type="entity",
            label=CADItemLabel(text="Entity 2"),
        )

        doc.add_item(item1)
        doc.add_item(item2)

        assert len(doc.items) == 2

    def test_add_processing_step(self):
        """Test adding processing step."""
        doc = CADlingDocument(name="test.step")

        doc.add_processing_step(
            step_name="build",
            component="TestBackend",
            duration_ms=100.0,
        )

        assert len(doc.processing_history) == 1
        assert doc.processing_history[0].step_name == "build"
        assert doc.processing_history[0].duration_ms == 100.0

    def test_export_to_json(self):
        """Test exporting to JSON."""
        doc = CADlingDocument(
            name="test.step",
            format=InputFormat.STEP,
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="abc123",
            ),
        )

        item = CADItem(
            item_type="entity",
            label=CADItemLabel(text="Entity 1"),
        )
        doc.add_item(item)

        json_data = doc.export_to_json()

        assert json_data["name"] == "test.step"
        assert json_data["format"] == "step"
        assert json_data["num_items"] == 1
        assert len(json_data["items"]) == 1

    def test_export_to_markdown(self):
        """Test exporting to Markdown."""
        doc = CADlingDocument(
            name="test.step",
            format=InputFormat.STEP,
        )

        item = CADItem(
            item_type="entity",
            label=CADItemLabel(text="Entity 1"),
        )
        doc.add_item(item)

        markdown = doc.export_to_markdown()

        assert "# test.step" in markdown
        assert "Entity 1" in markdown


class TestConversionResult:
    """Test ConversionResult model."""

    def test_creation(self):
        """Test creating a conversion result."""
        input_doc = CADInputDocument(
            file=Path("test.step"),
            format=InputFormat.STEP,
            document_hash="abc123",
        )

        result = ConversionResult(input=input_doc)

        assert result.input == input_doc
        assert result.status == ConversionStatus.SUCCESS
        assert result.document is None
        assert result.errors == []

    def test_add_error(self):
        """Test adding errors."""
        input_doc = CADInputDocument(
            file=Path("test.step"),
            format=InputFormat.STEP,
            document_hash="abc123",
        )

        result = ConversionResult(input=input_doc)
        result.add_error("TestBackend", "Test error message")

        assert len(result.errors) == 1
        assert result.errors[0].component == "TestBackend"
        assert result.errors[0].error_message == "Test error message"
