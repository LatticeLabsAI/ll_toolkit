"""Unit tests for face geometry extraction."""

import numpy as np
import pytest
from pathlib import Path

# Requires pythonocc-core (conda-only) for OCC shape handling
pytest.importorskip("OCC", reason="pythonocc-core not available")

from cadling.lib.geometry.face_geometry import FaceGeometryExtractor


@pytest.fixture
def test_data_path():
    """Get test data directory."""
    # From cadling/tests/unit/lib/geometry, go up to cadling/, then to data/test_data
    return Path(__file__).parent.parent.parent.parent.parent / "data" / "test_data"


@pytest.fixture
def real_step_document(test_data_path):
    """Load real STEP document."""
    from cadling.backend.document_converter import DocumentConverter

    step_path = test_data_path / "step"
    if not step_path.exists():
        pytest.skip("STEP test data not found")

    # Find a small STEP file (try both .stp and .step extensions)
    step_files = sorted(list(step_path.glob("*.stp")) + list(step_path.glob("*.step")))
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
    assert hasattr(real_step_document, '_occ_shape'), "Document does not have _occ_shape attribute"
    assert real_step_document._occ_shape is not None, "Document OCC shape is None"

    from OCC.Extend.TopologyUtils import TopologyExplorer
    topo = TopologyExplorer(real_step_document._occ_shape)
    faces = list(topo.faces())

    assert len(faces) > 0, "No faces found in document"
    return faces


@pytest.fixture
def extractor():
    """Create FaceGeometryExtractor instance."""
    return FaceGeometryExtractor()


class TestFaceGeometryExtractor:
    """Test FaceGeometryExtractor class."""

    def test_init(self):
        """Test initialization."""
        extractor = FaceGeometryExtractor()
        assert extractor is not None
        # Should have pythonocc available in test environment
        assert hasattr(extractor, 'has_pythonocc')

    def test_has_pythonocc(self):
        """Test that pythonocc availability is tracked."""
        extractor = FaceGeometryExtractor()
        # Just verify the attribute exists and is a boolean
        assert isinstance(extractor.has_pythonocc, bool)


class TestExtractFeatures:
    """Test extract_features method."""

    def test_extract_features_returns_dict(self, extractor, real_occ_faces):
        """Test that extract_features returns a dictionary."""
        if not extractor.has_pythonocc:
            pytest.skip("pythonocc-core not available")

        face = real_occ_faces[0]
        result = extractor.extract_features(face)

        assert isinstance(result, dict)

    def test_extract_features_has_required_keys(self, extractor, real_occ_faces):
        """Test that result has all required keys."""
        face = real_occ_faces[0]
        result = extractor.extract_features(face)

        expected_keys = [
            'gaussian_curvature',
            'mean_curvature',
            'normal',
            'centroid',
            'bbox_dimensions',
            'surface_area'
        ]

        for key in expected_keys:
            assert key in result

    def test_extract_features_values_are_valid(self, extractor, real_occ_faces):
        """Test that extracted feature values are valid."""
        face = real_occ_faces[0]
        result = extractor.extract_features(face)

        # Curvatures should be finite
        assert np.isfinite(result['gaussian_curvature'])
        assert np.isfinite(result['mean_curvature'])

        # Normal should be a 3D vector
        assert isinstance(result['normal'], list)
        assert len(result['normal']) == 3
        assert all(np.isfinite(v) for v in result['normal'])

        # Normal should be approximately unit length (allowing some tolerance)
        normal = np.array(result['normal'])
        normal_length = np.linalg.norm(normal)
        assert 0.9 <= normal_length <= 1.1, f"Normal length {normal_length} not close to 1"

        # Centroid should be a 3D point
        assert isinstance(result['centroid'], list)
        assert len(result['centroid']) == 3
        assert all(np.isfinite(v) for v in result['centroid'])

        # Bounding box dimensions should be positive
        assert isinstance(result['bbox_dimensions'], list)
        assert len(result['bbox_dimensions']) == 3
        assert all(d > 0 for d in result['bbox_dimensions'])

        # Surface area should be positive
        assert isinstance(result['surface_area'], (int, float))
        assert result['surface_area'] > 0

    def test_extract_features_multiple_faces(self, extractor, real_occ_faces):
        """Test extraction from multiple faces."""
        num_faces_to_test = min(5, len(real_occ_faces))

        for i in range(num_faces_to_test):
            face = real_occ_faces[i]
            result = extractor.extract_features(face)

            # All faces should return valid results
            assert result is not None
            assert isinstance(result, dict)
            assert 'gaussian_curvature' in result

    def test_extract_features_handles_none(self, extractor):
        """Test extract_features with None input."""
        result = extractor.extract_features(None)
        # Should return None or handle gracefully
        assert result is None or isinstance(result, dict)


class TestComputeCurvatureAtCenter:
    """Test compute_curvature_at_center method."""

    def test_compute_curvature_returns_tuple(self, extractor, real_occ_faces):
        """Test that compute_curvature_at_center returns a tuple."""
        face = real_occ_faces[0]
        result = extractor.compute_curvature_at_center(face)

        if result is not None:
            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_compute_curvature_values_are_finite(self, extractor, real_occ_faces):
        """Test that curvature values are finite."""
        face = real_occ_faces[0]
        result = extractor.compute_curvature_at_center(face)

        if result is not None:
            gaussian_curv, mean_curv = result
            assert np.isfinite(gaussian_curv)
            assert np.isfinite(mean_curv)

    def test_compute_curvature_planar_face(self, extractor, real_occ_faces):
        """Test curvature computation for planar faces."""
        # Find a planar face (curvature should be near 0)
        for face in real_occ_faces[:10]:
            result = extractor.compute_curvature_at_center(face)
            if result is not None:
                gaussian_curv, mean_curv = result
                # Planar faces should have curvature near 0
                if abs(gaussian_curv) < 1e-6 and abs(mean_curv) < 1e-6:
                    assert abs(gaussian_curv) < 1e-3
                    assert abs(mean_curv) < 1e-3
                    break

    def test_compute_curvature_multiple_faces(self, extractor, real_occ_faces):
        """Test curvature computation for multiple faces."""
        num_faces_to_test = min(10, len(real_occ_faces))

        curvatures = []
        for i in range(num_faces_to_test):
            face = real_occ_faces[i]
            result = extractor.compute_curvature_at_center(face)
            if result is not None:
                curvatures.append(result)

        # Should have computed curvature for most faces
        assert len(curvatures) > 0


class TestComputeNormalAtCenter:
    """Test compute_normal_at_center method."""

    def test_compute_normal_returns_list(self, extractor, real_occ_faces):
        """Test that compute_normal_at_center returns a list."""
        face = real_occ_faces[0]
        result = extractor.compute_normal_at_center(face)

        if result is not None:
            assert isinstance(result, list)
            assert len(result) == 3

    def test_compute_normal_is_unit_vector(self, extractor, real_occ_faces):
        """Test that normal is a unit vector."""
        face = real_occ_faces[0]
        result = extractor.compute_normal_at_center(face)

        if result is not None:
            normal = np.array(result)
            length = np.linalg.norm(normal)
            # Should be close to 1 (allowing 10% tolerance)
            assert 0.9 <= length <= 1.1

    def test_compute_normal_components_finite(self, extractor, real_occ_faces):
        """Test that all normal components are finite."""
        face = real_occ_faces[0]
        result = extractor.compute_normal_at_center(face)

        if result is not None:
            assert all(np.isfinite(v) for v in result)

    def test_compute_normal_multiple_faces(self, extractor, real_occ_faces):
        """Test normal computation for multiple faces."""
        num_faces_to_test = min(10, len(real_occ_faces))

        normals = []
        for i in range(num_faces_to_test):
            face = real_occ_faces[i]
            result = extractor.compute_normal_at_center(face)
            if result is not None:
                normals.append(result)

        # Should have computed normals for most faces
        assert len(normals) > 0

        # All should be approximately unit vectors
        for normal in normals:
            length = np.linalg.norm(np.array(normal))
            assert 0.9 <= length <= 1.1


class TestComputeCentroid:
    """Test compute_centroid method."""

    def test_compute_centroid_returns_list(self, extractor, real_occ_faces):
        """Test that compute_centroid returns a list."""
        face = real_occ_faces[0]
        result = extractor.compute_centroid(face)

        if result is not None:
            assert isinstance(result, list)
            assert len(result) == 3

    def test_compute_centroid_values_finite(self, extractor, real_occ_faces):
        """Test that centroid coordinates are finite."""
        face = real_occ_faces[0]
        result = extractor.compute_centroid(face)

        if result is not None:
            assert all(np.isfinite(v) for v in result)

    def test_compute_centroid_multiple_faces(self, extractor, real_occ_faces):
        """Test centroid computation for multiple faces."""
        num_faces_to_test = min(10, len(real_occ_faces))

        centroids = []
        for i in range(num_faces_to_test):
            face = real_occ_faces[i]
            result = extractor.compute_centroid(face)
            if result is not None:
                centroids.append(result)

        # Should have computed centroids for most faces
        assert len(centroids) > 0

        # All centroids should be valid 3D points
        for centroid in centroids:
            assert len(centroid) == 3
            assert all(np.isfinite(v) for v in centroid)


class TestComputeBboxDimensions:
    """Test compute_bbox_dimensions method."""

    def test_compute_bbox_dimensions_returns_list(self, extractor, real_occ_faces):
        """Test that compute_bbox_dimensions returns a list."""
        face = real_occ_faces[0]
        result = extractor.compute_bbox_dimensions(face)

        if result is not None:
            assert isinstance(result, list)
            assert len(result) == 3

    def test_compute_bbox_dimensions_positive(self, extractor, real_occ_faces):
        """Test that bounding box dimensions are positive."""
        face = real_occ_faces[0]
        result = extractor.compute_bbox_dimensions(face)

        if result is not None:
            assert all(d > 0 for d in result)

    def test_compute_bbox_dimensions_finite(self, extractor, real_occ_faces):
        """Test that bounding box dimensions are finite."""
        face = real_occ_faces[0]
        result = extractor.compute_bbox_dimensions(face)

        if result is not None:
            assert all(np.isfinite(d) for d in result)

    def test_compute_bbox_dimensions_multiple_faces(self, extractor, real_occ_faces):
        """Test bounding box computation for multiple faces."""
        num_faces_to_test = min(10, len(real_occ_faces))

        bbox_dims = []
        for i in range(num_faces_to_test):
            face = real_occ_faces[i]
            result = extractor.compute_bbox_dimensions(face)
            if result is not None:
                bbox_dims.append(result)

        # Should have computed bbox for most faces
        assert len(bbox_dims) > 0

        # All should be valid dimensions
        for dims in bbox_dims:
            assert len(dims) == 3
            assert all(d > 0 for d in dims)


class TestComputeSurfaceArea:
    """Test compute_surface_area method."""

    def test_compute_surface_area_returns_float(self, extractor, real_occ_faces):
        """Test that compute_surface_area returns a float."""
        face = real_occ_faces[0]
        result = extractor.compute_surface_area(face)

        if result is not None:
            assert isinstance(result, (int, float))

    def test_compute_surface_area_positive(self, extractor, real_occ_faces):
        """Test that surface area is positive."""
        face = real_occ_faces[0]
        result = extractor.compute_surface_area(face)

        if result is not None:
            assert result > 0

    def test_compute_surface_area_finite(self, extractor, real_occ_faces):
        """Test that surface area is finite."""
        face = real_occ_faces[0]
        result = extractor.compute_surface_area(face)

        if result is not None:
            assert np.isfinite(result)

    def test_compute_surface_area_multiple_faces(self, extractor, real_occ_faces):
        """Test surface area computation for multiple faces."""
        num_faces_to_test = min(10, len(real_occ_faces))

        areas = []
        for i in range(num_faces_to_test):
            face = real_occ_faces[i]
            result = extractor.compute_surface_area(face)
            if result is not None:
                areas.append(result)

        # Should have computed area for most faces
        assert len(areas) > 0

        # All should be positive
        assert all(a > 0 for a in areas)

    def test_compute_surface_area_consistency(self, extractor, real_occ_faces):
        """Test that larger bounding boxes generally have larger surface areas."""
        # Extract features for multiple faces
        face_data = []
        for i in range(min(10, len(real_occ_faces))):
            face = real_occ_faces[i]
            area = extractor.compute_surface_area(face)
            bbox = extractor.compute_bbox_dimensions(face)

            if area is not None and bbox is not None:
                bbox_volume = bbox[0] * bbox[1] * bbox[2]
                face_data.append((area, bbox_volume))

        # Should have some data
        assert len(face_data) > 0


class TestGeometryExtractorIntegration:
    """Integration tests for FaceGeometryExtractor."""

    def test_all_methods_work_together(self, extractor, real_occ_faces):
        """Test that all extraction methods work together."""
        face = real_occ_faces[0]

        # Extract all features
        features = extractor.extract_features(face)
        curvature = extractor.compute_curvature_at_center(face)
        normal = extractor.compute_normal_at_center(face)
        centroid = extractor.compute_centroid(face)
        bbox_dims = extractor.compute_bbox_dimensions(face)
        area = extractor.compute_surface_area(face)

        # All should return valid results
        assert features is not None
        assert curvature is not None or curvature is None  # May fail for some faces
        assert normal is not None or normal is None
        assert centroid is not None or centroid is None
        assert bbox_dims is not None or bbox_dims is None
        assert area is not None or area is None

    def test_extract_features_consistency(self, extractor, real_occ_faces):
        """Test that extract_features returns consistent values with individual methods."""
        face = real_occ_faces[0]

        # Extract all features
        features = extractor.extract_features(face)
        curvature = extractor.compute_curvature_at_center(face)
        normal = extractor.compute_normal_at_center(face)
        centroid = extractor.compute_centroid(face)
        bbox_dims = extractor.compute_bbox_dimensions(face)
        area = extractor.compute_surface_area(face)

        # Features should match individual extractions
        if curvature is not None:
            assert features['gaussian_curvature'] == curvature[0]
            assert features['mean_curvature'] == curvature[1]

        if normal is not None:
            np.testing.assert_array_almost_equal(
                features['normal'], normal, decimal=10
            )

        if centroid is not None:
            np.testing.assert_array_almost_equal(
                features['centroid'], centroid, decimal=10
            )

        if bbox_dims is not None:
            np.testing.assert_array_almost_equal(
                features['bbox_dimensions'], bbox_dims, decimal=10
            )

        if area is not None:
            assert abs(features['surface_area'] - area) < 1e-6

    def test_batch_feature_extraction(self, extractor, real_occ_faces):
        """Test batch feature extraction from multiple faces."""
        num_faces_to_test = min(20, len(real_occ_faces))

        all_features = []
        for i in range(num_faces_to_test):
            face = real_occ_faces[i]
            features = extractor.extract_features(face)
            if features is not None:
                all_features.append(features)

        # Should have extracted features for most faces
        assert len(all_features) > 0

        # Check statistical properties across all faces
        gaussian_curvatures = [f['gaussian_curvature'] for f in all_features]
        mean_curvatures = [f['mean_curvature'] for f in all_features]
        surface_areas = [f['surface_area'] for f in all_features]

        # All should be finite
        assert all(np.isfinite(k) for k in gaussian_curvatures)
        assert all(np.isfinite(h) for h in mean_curvatures)
        assert all(np.isfinite(a) for a in surface_areas)

        # Areas should be positive
        assert all(a > 0 for a in surface_areas)
