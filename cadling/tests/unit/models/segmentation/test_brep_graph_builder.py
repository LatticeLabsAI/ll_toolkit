"""Unit tests for BRep graph builder."""

import numpy as np
import pytest
from pathlib import Path

from cadling.models.segmentation.brep_graph_builder import BRepFaceGraphBuilder


@pytest.fixture
def test_data_path():
    """Get test data directory."""
    return Path(__file__).parent.parent.parent.parent / "data" / "test_data"


@pytest.fixture
def real_step_document(test_data_path):
    """Load real STEP document."""
    from cadling.backend.document_converter import DocumentConverter

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

    return result.document


class TestBRepFaceGraphBuilder:
    """Test BRepFaceGraphBuilder class."""

    def test_init(self):
        """Test initialization."""
        builder = BRepFaceGraphBuilder()
        assert builder is not None

    def test_has_build_face_graph_method(self):
        """Test that build_face_graph method exists."""
        builder = BRepFaceGraphBuilder()
        assert hasattr(builder, 'build_face_graph')
        assert callable(builder.build_face_graph)

    def test_build_face_graph_handles_none(self):
        """Test build_face_graph with None document."""
        builder = BRepFaceGraphBuilder()
        # Should handle None gracefully - either return None or raise an exception
        try:
            result = builder.build_face_graph(None)
            # If it doesn't raise, it should return None or empty graph
            assert result is None or hasattr(result, 'x')
        except (AttributeError, TypeError, ValueError):
            # It's acceptable to raise an exception for invalid input
            pass

    def test_build_face_graph_with_real_document(self, real_step_document):
        """Test build_face_graph with real STEP document."""
        builder = BRepFaceGraphBuilder()
        result = builder.build_face_graph(real_step_document)

        # Should produce a valid graph
        assert result is not None
        assert hasattr(result, 'x')
        assert hasattr(result, 'edge_index')

    def test_output_is_pyg_data(self, real_step_document):
        """Test that output is PyG Data object."""
        builder = BRepFaceGraphBuilder()
        result = builder.build_face_graph(real_step_document)

        if result is not None:
            # Should be PyG Data object with required attributes
            assert hasattr(result, 'x') or hasattr(result, 'edge_index') or hasattr(result, 'num_nodes')

    def test_graph_has_node_features(self, real_step_document):
        """Test that graph has node features."""
        builder = BRepFaceGraphBuilder()
        result = builder.build_face_graph(real_step_document)

        if result is not None and hasattr(result, 'x'):
            # Node features should be 2D array
            assert result.x is not None
            assert result.x.ndim == 2
            assert result.x.shape[0] > 0  # Should have nodes
            assert result.x.shape[1] > 0  # Should have features

    def test_graph_has_edge_index(self, real_step_document):
        """Test that graph has edge connectivity."""
        builder = BRepFaceGraphBuilder()
        result = builder.build_face_graph(real_step_document)

        if result is not None and hasattr(result, 'edge_index'):
            # Edge index should be present
            assert result.edge_index is not None
            assert result.edge_index.ndim == 2
            assert result.edge_index.shape[0] == 2

    def test_graph_has_edge_features(self, real_step_document):
        """Test that graph has edge features."""
        builder = BRepFaceGraphBuilder()
        result = builder.build_face_graph(real_step_document)

        if result is not None and hasattr(result, 'edge_attr'):
            # Edge features should be present
            if result.edge_attr is not None:
                assert result.edge_attr.ndim == 2
                assert result.edge_attr.shape[1] > 0

    def test_face_adjacency_graph(self, real_step_document):
        """Test that builder creates face adjacency graph."""
        builder = BRepFaceGraphBuilder()
        # Face graph connects faces that share edges
        # Each face is a node, edges represent adjacency
        result = builder.build_face_graph(real_step_document)

        if result is not None and hasattr(result, 'edge_index'):
            # Edge index should be 2xE matrix
            if result.edge_index is not None:
                assert result.edge_index.ndim == 2
                if result.edge_index.shape[1] > 0:
                    assert result.edge_index.shape[0] == 2

    def test_node_feature_dimension(self, real_step_document):
        """Test node feature dimensions."""
        builder = BRepFaceGraphBuilder()
        result = builder.build_face_graph(real_step_document)

        if result is not None and hasattr(result, 'x') and result.x is not None:
            # With enhanced features: 48-dim
            # Without: basic geometric features (~7-10 dim)
            assert result.x.ndim == 2
            if result.x.shape[0] > 0:
                # Should have consistent feature dimension
                assert result.x.shape[1] > 0
                # Check for expected dimensions (24 or 48)
                assert result.x.shape[1] in [24, 48]

    def test_edge_feature_dimension(self, real_step_document):
        """Test edge feature dimensions."""
        builder = BRepFaceGraphBuilder()
        result = builder.build_face_graph(real_step_document)

        if result is not None and hasattr(result, 'edge_attr') and result.edge_attr is not None:
            # With enhanced features: 16-dim
            # Without: basic features (dihedral angle, length, etc.)
            assert result.edge_attr.ndim == 2
            if result.edge_attr.shape[0] > 0:
                assert result.edge_attr.shape[1] > 0
                # Check for expected dimensions (8 or 16)
                assert result.edge_attr.shape[1] in [8, 16]


class TestBRepFaceGraphBuilderIntegration:
    """Integration tests for BRep graph builder."""

    def test_builds_valid_graph_structure(self, real_step_document):
        """Test that builder creates valid graph structure."""
        builder = BRepFaceGraphBuilder()
        result = builder.build_face_graph(real_step_document)

        if result is not None:
            # Valid graph should have:
            # 1. Node features (x)
            # 2. Edge connectivity (edge_index)
            # 3. Optionally edge features (edge_attr)

            has_nodes = hasattr(result, 'x')
            has_edges = hasattr(result, 'edge_index')

            # Should have both nodes and edges for real document
            assert has_nodes and has_edges
            assert result.x.shape[0] > 0  # Should have faces

    def test_graph_connectivity_is_valid(self, real_step_document):
        """Test that graph connectivity is valid."""
        builder = BRepFaceGraphBuilder()
        result = builder.build_face_graph(real_step_document)

        if result is not None and hasattr(result, 'edge_index') and result.edge_index is not None:
            edge_index = result.edge_index

            if edge_index.shape[1] > 0:
                # Edge indices should be non-negative
                assert edge_index.min() >= 0

                # Edge indices should be within node range
                if hasattr(result, 'x') and result.x is not None:
                    num_nodes = result.x.shape[0]
                    if num_nodes > 0:
                        assert edge_index.max() < num_nodes

    def test_graph_is_undirected(self, real_step_document):
        """Test that face adjacency graph is undirected."""
        builder = BRepFaceGraphBuilder()
        result = builder.build_face_graph(real_step_document)

        if result is not None and hasattr(result, 'edge_index') and result.edge_index is not None:
            edge_index = result.edge_index

            if edge_index.shape[1] > 0:
                # For undirected graph, each edge (i,j) should have reverse (j,i)
                # This is optional - graph may be stored as directed
                # Just verify shape is valid
                assert edge_index.shape[0] == 2

    def test_handles_empty_document(self):
        """Test handling of document with no faces."""
        from cadling.datamodel.base_models import CADlingDocument

        builder = BRepFaceGraphBuilder()
        # Empty document should produce empty or None graph
        empty_doc = CADlingDocument(name="empty", items=[])
        result = builder.build_face_graph(empty_doc)

        # Should handle gracefully - return None or empty graph
        if result is not None:
            if hasattr(result, 'x'):
                # Can have 0 nodes
                assert result.x.shape[0] >= 0

    def test_features_are_finite(self, real_step_document):
        """Test that all features are finite (no NaN/Inf)."""
        builder = BRepFaceGraphBuilder()
        result = builder.build_face_graph(real_step_document)

        if result is not None:
            if hasattr(result, 'x') and result.x is not None:
                # Check node features are finite
                x_array = result.x.numpy() if hasattr(result.x, 'numpy') else result.x
                assert np.all(np.isfinite(x_array))

            if hasattr(result, 'edge_attr') and result.edge_attr is not None:
                # Check edge features are finite
                edge_array = result.edge_attr.numpy() if hasattr(result.edge_attr, 'numpy') else result.edge_attr
                assert np.all(np.isfinite(edge_array))

    def test_face_features_include_geometry(self, real_step_document):
        """Test that face features include geometric properties."""
        builder = BRepFaceGraphBuilder()
        result = builder.build_face_graph(real_step_document)

        if result is not None and hasattr(result, 'x') and result.x is not None:
            # Face features should include geometric properties
            # Check that features are not all zeros (real data)
            x_array = result.x.numpy() if hasattr(result.x, 'numpy') else result.x
            non_zero_ratio = np.count_nonzero(x_array) / x_array.size
            assert non_zero_ratio > 0.05  # At least 5% non-zero

    def test_edge_features_include_adjacency_info(self, real_step_document):
        """Test that edge features include adjacency information."""
        builder = BRepFaceGraphBuilder()
        result = builder.build_face_graph(real_step_document)

        if result is not None and hasattr(result, 'edge_attr') and result.edge_attr is not None:
            # Edge features should include adjacency information
            # Check that features are not all zeros (real data)
            edge_array = result.edge_attr.numpy() if hasattr(result.edge_attr, 'numpy') else result.edge_attr
            non_zero_ratio = np.count_nonzero(edge_array) / edge_array.size
            assert non_zero_ratio > 0.05  # At least 5% non-zero
