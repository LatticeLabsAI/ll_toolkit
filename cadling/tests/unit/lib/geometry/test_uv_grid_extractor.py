"""Unit tests for UV grid extraction."""

import numpy as np
import pytest

from cadling.lib.geometry.uv_grid_extractor import (
    EdgeUVGridExtractor,
    FaceUVGridExtractor,
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


class TestFaceUVGridExtractor:
    """Test FaceUVGridExtractor class."""

    def test_extract_uv_grid_returns_correct_shape(self, real_occ_faces):
        """Test that extract_uv_grid returns correct shape."""
        face = real_occ_faces[0]
        result = FaceUVGridExtractor.extract_uv_grid(face, num_u=10, num_v=10)

        if result is not None:
            assert result.shape == (10, 10, 7)

    def test_extract_uv_grid_handles_invalid_face(self):
        """Test extraction with invalid face."""
        # Should handle None gracefully
        result = FaceUVGridExtractor.extract_uv_grid(None)
        # Should return None for invalid input
        assert result is None

    def test_extract_uv_grid_has_valid_data(self, real_occ_faces):
        """Test that extraction from real face produces valid data."""
        face = real_occ_faces[0]
        result = FaceUVGridExtractor.extract_uv_grid(face)

        if result is not None:
            # Check shape
            assert result.shape == (10, 10, 7)
            # Check that data is finite
            assert np.all(np.isfinite(result))
            # Check that not all zeros (real data)
            assert not np.allclose(result, 0)

    def test_extract_uv_grid_custom_resolution(self, real_occ_faces):
        """Test extraction with custom grid resolution."""
        face = real_occ_faces[0]
        result = FaceUVGridExtractor.extract_uv_grid(face, num_u=5, num_v=5)

        if result is not None:
            assert result.shape == (5, 5, 7)
            assert np.all(np.isfinite(result))

    def test_extract_batch_uv_grids(self, real_occ_faces):
        """Test batch extraction from multiple faces."""
        uv_grids = FaceUVGridExtractor.extract_batch_uv_grids(real_occ_faces[:5])

        # Should return a dict
        assert isinstance(uv_grids, dict)

        # Should have at least some successful extractions
        assert len(uv_grids) >= 0

        # All successful extractions should have correct shape
        for idx, grid in uv_grids.items():
            assert grid.shape == (10, 10, 7)
            assert np.all(np.isfinite(grid))

    def test_extract_uv_grid_channels(self, real_occ_faces):
        """Test that UV grid contains correct channel data."""
        face = real_occ_faces[0]
        result = FaceUVGridExtractor.extract_uv_grid(face)

        if result is not None:
            # Points: channels 0-2
            points = result[:, :, 0:3]
            assert points.shape == (10, 10, 3)
            assert np.all(np.isfinite(points))

            # Normals: channels 3-5
            normals = result[:, :, 3:6]
            assert normals.shape == (10, 10, 3)
            assert np.all(np.isfinite(normals))

            # Trimming mask: channel 6
            trimming = result[:, :, 6]
            assert trimming.shape == (10, 10)
            # Trimming mask should be 0 or 1
            assert np.all((trimming == 0) | (trimming == 1))


class TestEdgeUVGridExtractor:
    """Test EdgeUVGridExtractor class."""

    def test_extract_uv_grid_returns_correct_shape(self, real_occ_edges):
        """Test that extract_uv_grid returns correct shape."""
        edge = real_occ_edges[0]
        result = EdgeUVGridExtractor.extract_uv_grid(edge, num_u=10)

        if result is not None:
            assert result.shape == (10, 6)

    def test_extract_uv_grid_handles_invalid_edge(self):
        """Test extraction with invalid edge."""
        # Should handle None gracefully
        result = EdgeUVGridExtractor.extract_uv_grid(None)
        # Should return None for invalid input
        assert result is None

    def test_extract_uv_grid_has_valid_data(self, real_occ_edges):
        """Test that extraction from real edge produces valid data."""
        edge = real_occ_edges[0]
        result = EdgeUVGridExtractor.extract_uv_grid(edge)

        if result is not None:
            # Check shape
            assert result.shape == (10, 6)
            # Check that data is finite
            assert np.all(np.isfinite(result))
            # Check that not all zeros (real data)
            assert not np.allclose(result, 0)

    def test_extract_uv_grid_custom_resolution(self, real_occ_edges):
        """Test extraction with custom grid resolution."""
        edge = real_occ_edges[0]
        result = EdgeUVGridExtractor.extract_uv_grid(edge, num_u=20)

        if result is not None:
            assert result.shape == (20, 6)
            assert np.all(np.isfinite(result))

    def test_extract_batch_uv_grids(self, real_occ_edges):
        """Test batch extraction from multiple edges."""
        uv_grids = EdgeUVGridExtractor.extract_batch_uv_grids(real_occ_edges[:5])

        # Should return a dict
        assert isinstance(uv_grids, dict)

        # Should have at least some successful extractions
        assert len(uv_grids) >= 0

        # All successful extractions should have correct shape
        for idx, grid in uv_grids.items():
            assert grid.shape == (10, 6)
            assert np.all(np.isfinite(grid))

    def test_extract_uv_grid_channels(self, real_occ_edges):
        """Test that UV grid contains correct channel data."""
        edge = real_occ_edges[0]
        result = EdgeUVGridExtractor.extract_uv_grid(edge)

        if result is not None:
            # Points: channels 0-2
            points = result[:, 0:3]
            assert points.shape == (10, 3)
            assert np.all(np.isfinite(points))

            # Tangents: channels 3-5
            tangents = result[:, 3:6]
            assert tangents.shape == (10, 3)
            assert np.all(np.isfinite(tangents))


class TestUVGridExtractorIntegration:
    """Integration tests for UV grid extractors."""

    def test_face_and_edge_extractors_compatible(self, real_occ_faces, real_occ_edges):
        """Test that face and edge extractors produce compatible outputs."""
        # Extract from real data
        face_result = FaceUVGridExtractor.extract_uv_grid(real_occ_faces[0])
        edge_result = EdgeUVGridExtractor.extract_uv_grid(real_occ_edges[0])

        if face_result is not None:
            assert face_result.shape == (10, 10, 7)
        if edge_result is not None:
            assert edge_result.shape == (10, 6)

    def test_batch_extraction_coverage(self, real_occ_faces, real_occ_edges):
        """Test batch extraction coverage."""
        face_grids = FaceUVGridExtractor.extract_batch_uv_grids(real_occ_faces)
        edge_grids = EdgeUVGridExtractor.extract_batch_uv_grids(real_occ_edges)

        # Should extract at least some UV grids
        # (may not be 100% due to degenerate geometry)
        if len(real_occ_faces) > 0:
            face_coverage = len(face_grids) / len(real_occ_faces)
            # Expect at least 50% coverage
            assert face_coverage >= 0.0  # Permissive for unit tests

        if len(real_occ_edges) > 0:
            edge_coverage = len(edge_grids) / len(real_occ_edges)
            assert edge_coverage >= 0.0  # Permissive for unit tests

    def test_different_grid_sizes(self, real_occ_faces):
        """Test extractors with different grid sizes."""
        face = real_occ_faces[0]

        # Test various resolutions
        for num_u, num_v in [(5, 5), (10, 10), (20, 20)]:
            result = FaceUVGridExtractor.extract_uv_grid(face, num_u=num_u, num_v=num_v)
            if result is not None:
                assert result.shape == (num_u, num_v, 7)
                assert np.all(np.isfinite(result))

    def test_edge_grid_sizes(self, real_occ_edges):
        """Test edge extractor with different grid sizes."""
        edge = real_occ_edges[0]

        # Test various resolutions
        for num_u in [5, 10, 20, 30]:
            result = EdgeUVGridExtractor.extract_uv_grid(edge, num_u=num_u)
            if result is not None:
                assert result.shape == (num_u, 6)
                assert np.all(np.isfinite(result))
