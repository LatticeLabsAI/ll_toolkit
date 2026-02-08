"""Unit tests for enhanced feature extraction."""

import numpy as np
import pytest

from cadling.lib.graph.enhanced_features import (
    extract_enhanced_edge_features,
    extract_enhanced_node_features,
    get_curve_type_from_edge,
    get_surface_type_from_face,
)


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


@pytest.fixture
def real_occ_faces(real_step_document):
    """Extract OCC faces from real document."""
    if not hasattr(real_step_document, '_occ_shape') or real_step_document._occ_shape is None:
        pytest.skip("OCC shape not available (pythonocc-core not installed)")

    from OCC.Extend.TopologyUtils import TopologyExplorer
    topo = TopologyExplorer(real_step_document._occ_shape)
    faces = list(topo.faces())

    if len(faces) == 0:
        pytest.skip("No OCC faces found in document")
    return faces


@pytest.fixture
def real_occ_edges(real_step_document):
    """Extract OCC edges from real document."""
    if not hasattr(real_step_document, '_occ_shape') or real_step_document._occ_shape is None:
        pytest.skip("OCC shape not available (pythonocc-core not installed)")

    from OCC.Extend.TopologyUtils import TopologyExplorer
    topo = TopologyExplorer(real_step_document._occ_shape)
    edges = list(topo.edges())

    if len(edges) == 0:
        pytest.skip("No OCC edges found in document")
    return edges


class TestGetSurfaceTypeFromFace:
    """Test get_surface_type_from_face function."""

    def test_returns_string(self, real_occ_faces):
        """Test that function returns a string."""
        result = get_surface_type_from_face(real_occ_faces[0])
        assert isinstance(result, str)

    def test_returns_valid_surface_type(self, real_occ_faces):
        """Test that function returns valid surface type."""
        result = get_surface_type_from_face(real_occ_faces[0])
        valid_types = ['PLANE', 'CYLINDRICAL_SURFACE', 'CONICAL_SURFACE',
                       'SPHERICAL_SURFACE', 'TOROIDAL_SURFACE', 'B_SPLINE_SURFACE',
                       'SURFACE_OF_REVOLUTION', 'SURFACE_OF_LINEAR_EXTRUSION', 'UNKNOWN']
        assert result in valid_types


class TestGetCurveTypeFromEdge:
    """Test get_curve_type_from_edge function."""

    def test_returns_string(self, real_occ_edges):
        """Test that function returns a string."""
        result = get_curve_type_from_edge(real_occ_edges[0])
        assert isinstance(result, str)

    def test_returns_valid_curve_type(self, real_occ_edges):
        """Test that function returns valid curve type."""
        result = get_curve_type_from_edge(real_occ_edges[0])
        valid_types = ['LINE', 'CIRCLE', 'ELLIPSE', 'BSPLINE', 'BEZIER', 'OTHER']
        assert result in valid_types


class TestExtractEnhancedNodeFeatures:
    """Test extract_enhanced_node_features function."""

    def test_returns_numpy_array(self, real_occ_faces):
        """Test that function returns numpy array."""
        face = real_occ_faces[0]
        surface_type = get_surface_type_from_face(face)
        result = extract_enhanced_node_features(face, surface_type)
        assert isinstance(result, np.ndarray)

    def test_returns_correct_dimension(self, real_occ_faces):
        """Test that output has 48 dimensions."""
        face = real_occ_faces[0]
        surface_type = get_surface_type_from_face(face)
        result = extract_enhanced_node_features(face, surface_type)
        assert result.shape[-1] == 48  # 48-dim node features

    def test_features_are_finite(self, real_occ_faces):
        """Test that all features are finite."""
        face = real_occ_faces[0]
        surface_type = get_surface_type_from_face(face)
        result = extract_enhanced_node_features(face, surface_type)
        assert np.all(np.isfinite(result))

    def test_handles_multiple_faces(self, real_occ_faces):
        """Test extraction for multiple faces."""
        for face in real_occ_faces[:5]:  # Test first 5 faces
            surface_type = get_surface_type_from_face(face)
            result = extract_enhanced_node_features(face, surface_type)
            assert result.shape[-1] == 48


class TestExtractEnhancedEdgeFeatures:
    """Test extract_enhanced_edge_features function."""

    def test_returns_numpy_array(self, real_occ_edges):
        """Test that function returns numpy array."""
        edge = real_occ_edges[0]
        curve_type = get_curve_type_from_edge(edge)
        result = extract_enhanced_edge_features(edge, curve_type)
        assert isinstance(result, np.ndarray)

    def test_returns_correct_dimension(self, real_occ_edges):
        """Test that output has 16 dimensions."""
        edge = real_occ_edges[0]
        curve_type = get_curve_type_from_edge(edge)
        result = extract_enhanced_edge_features(edge, curve_type)
        assert result.shape[-1] == 16  # 16-dim edge features

    def test_features_are_finite(self, real_occ_edges):
        """Test that all features are finite."""
        edge = real_occ_edges[0]
        curve_type = get_curve_type_from_edge(edge)
        result = extract_enhanced_edge_features(edge, curve_type)
        assert np.all(np.isfinite(result))

    def test_handles_multiple_edges(self, real_occ_edges):
        """Test extraction for multiple edges."""
        for edge in real_occ_edges[:5]:  # Test first 5 edges
            curve_type = get_curve_type_from_edge(edge)
            result = extract_enhanced_edge_features(edge, curve_type)
            assert result.shape[-1] == 16


class TestEnhancedFeaturesIntegration:
    """Integration tests for enhanced features."""

    def test_node_and_edge_features_compatible(self, real_occ_faces, real_occ_edges):
        """Test that node and edge features are compatible."""
        # Extract node features
        face = real_occ_faces[0]
        surface_type = get_surface_type_from_face(face)
        node_features = extract_enhanced_node_features(face, surface_type)

        # Extract edge features
        edge = real_occ_edges[0]
        curve_type = get_curve_type_from_edge(edge)
        edge_features = extract_enhanced_edge_features(edge, curve_type)

        # Both should be valid arrays
        assert isinstance(node_features, np.ndarray)
        assert isinstance(edge_features, np.ndarray)
        assert node_features.shape[-1] == 48
        assert edge_features.shape[-1] == 16

    def test_features_are_numeric(self, real_occ_faces, real_occ_edges):
        """Test that all features are numeric."""
        # Node features
        face = real_occ_faces[0]
        surface_type = get_surface_type_from_face(face)
        node_features = extract_enhanced_node_features(face, surface_type)
        assert np.issubdtype(node_features.dtype, np.number)

        # Edge features
        edge = real_occ_edges[0]
        curve_type = get_curve_type_from_edge(edge)
        edge_features = extract_enhanced_edge_features(edge, curve_type)
        assert np.issubdtype(edge_features.dtype, np.number)

    def test_batch_extraction(self, real_occ_faces, real_occ_edges):
        """Test batch extraction of features."""
        # Extract features for multiple faces
        node_feature_list = []
        for face in real_occ_faces[:10]:
            surface_type = get_surface_type_from_face(face)
            features = extract_enhanced_node_features(face, surface_type)
            node_feature_list.append(features)

        # Stack into matrix
        node_matrix = np.vstack(node_feature_list)
        assert node_matrix.shape == (len(node_feature_list), 48)

        # Extract features for multiple edges
        edge_feature_list = []
        for edge in real_occ_edges[:10]:
            curve_type = get_curve_type_from_edge(edge)
            features = extract_enhanced_edge_features(edge, curve_type)
            edge_feature_list.append(features)

        # Stack into matrix
        edge_matrix = np.vstack(edge_feature_list)
        assert edge_matrix.shape == (len(edge_feature_list), 16)
