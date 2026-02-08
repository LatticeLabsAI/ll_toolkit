"""Unit tests for PyTorch Geometric exporter."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from cadling.lib.graph.pyg_exporter import (
    export_to_pyg_with_uvgrids,
    save_pyg_data,
    validate_pyg_data,
)


@pytest.fixture
def real_graph_data(test_data_path):
    """Create real graph data from actual STEP file."""
    from cadling.backend.document_converter import DocumentConverter
    from cadling.models.segmentation.brep_graph_builder import BRepFaceGraphBuilder

    step_path = test_data_path / "step"
    if not step_path.exists():
        pytest.skip("STEP test data not found")

    # Find a small STEP file
    step_files = sorted(step_path.glob("*.stp"))
    if not step_files:
        pytest.skip("No STEP files found")

    test_file = step_files[0]

    # Convert to document
    converter = DocumentConverter()
    result = converter.convert(test_file)

    if result.document is None or len(result.errors) > 0:
        pytest.skip(f"STEP conversion failed: {result.errors}")

    # Build graph
    builder = BRepFaceGraphBuilder()
    graph_data = builder.build_face_graph(result.document)

    if graph_data is None:
        pytest.skip("Graph building failed")

    return {
        'node_features': graph_data.x.numpy() if hasattr(graph_data.x, 'numpy') else graph_data.x,
        'edge_index': graph_data.edge_index.numpy() if hasattr(graph_data.edge_index, 'numpy') else graph_data.edge_index,
        'edge_features': graph_data.edge_attr.numpy() if hasattr(graph_data.edge_attr, 'numpy') else graph_data.edge_attr,
        'graph_data': graph_data
    }


class TestExportToPyGWithUVGrids:
    """Test export_to_pyg_with_uvgrids function."""

    def test_function_exists(self):
        """Test that function exists."""
        assert export_to_pyg_with_uvgrids is not None
        assert callable(export_to_pyg_with_uvgrids)

    def test_export_with_real_data(self, real_graph_data):
        """Test export with real graph data."""
        result = export_to_pyg_with_uvgrids(
            node_features=real_graph_data['node_features'],
            edge_index=real_graph_data['edge_index'],
            edge_features=real_graph_data['edge_features']
        )

        assert result is not None
        assert hasattr(result, 'x')
        assert hasattr(result, 'edge_index')
        assert hasattr(result, 'edge_attr')

    def test_output_has_required_attributes(self, real_graph_data):
        """Test that output has required PyG attributes."""
        result = export_to_pyg_with_uvgrids(
            node_features=real_graph_data['node_features'],
            edge_index=real_graph_data['edge_index'],
            edge_features=real_graph_data['edge_features']
        )

        if result is not None:
            assert hasattr(result, 'x')
            assert hasattr(result, 'edge_index')
            assert hasattr(result, 'edge_attr')
            assert hasattr(result, 'metadata')

    def test_includes_uv_grids(self, real_graph_data):
        """Test that UV grids are included in output."""
        result = export_to_pyg_with_uvgrids(
            node_features=real_graph_data['node_features'],
            edge_index=real_graph_data['edge_index'],
            edge_features=real_graph_data['edge_features']
        )

        if result is not None:
            assert hasattr(result, 'face_uv_grids')
            assert hasattr(result, 'edge_uv_grids')


class TestSavePyGData:
    """Test save_pyg_data function."""

    def test_function_exists(self):
        """Test that function exists."""
        assert save_pyg_data is not None
        assert callable(save_pyg_data)

    def test_saves_to_file(self, real_graph_data):
        """Test that data is saved to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_graph.pt"

            # Create PyG data
            data = export_to_pyg_with_uvgrids(
                node_features=real_graph_data['node_features'],
                edge_index=real_graph_data['edge_index'],
                edge_features=real_graph_data['edge_features']
            )

            if data is not None:
                save_pyg_data(data, output_path)
                assert output_path.exists()

    def test_creates_parent_directories(self, real_graph_data):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "dir" / "graph.pt"

            # Create PyG data
            data = export_to_pyg_with_uvgrids(
                node_features=real_graph_data['node_features'],
                edge_index=real_graph_data['edge_index'],
                edge_features=real_graph_data['edge_features']
            )

            if data is not None:
                save_pyg_data(data, nested_path)
                assert nested_path.exists()


class TestValidatePyGData:
    """Test validate_pyg_data function."""

    def test_function_exists(self):
        """Test that function exists."""
        assert validate_pyg_data is not None
        assert callable(validate_pyg_data)

    def test_validates_real_data(self, real_graph_data):
        """Test validation of real PyG data."""
        data = export_to_pyg_with_uvgrids(
            node_features=real_graph_data['node_features'],
            edge_index=real_graph_data['edge_index'],
            edge_features=real_graph_data['edge_features']
        )

        if data is not None:
            errors = validate_pyg_data(data)
            assert isinstance(errors, list)


class TestPyGExporterIntegration:
    """Integration tests for PyG exporter."""

    def test_export_and_validate_workflow(self, real_graph_data):
        """Test complete export and validation workflow."""
        # Export data
        data = export_to_pyg_with_uvgrids(
            node_features=real_graph_data['node_features'],
            edge_index=real_graph_data['edge_index'],
            edge_features=real_graph_data['edge_features']
        )

        # Validate exported data
        if data is not None:
            errors = validate_pyg_data(data)
            assert isinstance(errors, list)

    def test_export_save_workflow(self, real_graph_data):
        """Test export and save workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_graph.pt"

            # Export data
            data = export_to_pyg_with_uvgrids(
                node_features=real_graph_data['node_features'],
                edge_index=real_graph_data['edge_index'],
                edge_features=real_graph_data['edge_features']
            )

            # Save data
            if data is not None:
                save_pyg_data(data, output_path, include_metadata=True)

                # Check files exist
                assert output_path.exists()

    def test_enhanced_features_in_export(self, real_graph_data):
        """Test that enhanced features are included in export."""
        data = export_to_pyg_with_uvgrids(
            node_features=real_graph_data['node_features'],
            edge_index=real_graph_data['edge_index'],
            edge_features=real_graph_data['edge_features']
        )

        if data is not None:
            # Node features should be 24-dim (or 48-dim with enhanced features)
            assert data.x.shape[1] in [24, 48]
            # Edge features should be 8-dim (or 16-dim with enhanced features)
            assert data.edge_attr.shape[1] in [8, 16]

    def test_metadata_in_export(self, real_graph_data):
        """Test that metadata is included in export."""
        metadata = {
            'source': 'test',
            'feature_version': '1.0'
        }

        data = export_to_pyg_with_uvgrids(
            node_features=real_graph_data['node_features'],
            edge_index=real_graph_data['edge_index'],
            edge_features=real_graph_data['edge_features'],
            metadata=metadata
        )

        if data is not None:
            assert hasattr(data, 'metadata')
            assert isinstance(data.metadata, dict)
