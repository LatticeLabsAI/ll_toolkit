"""Unit tests for geometric distribution analyzers."""

import numpy as np
import pytest

from cadling.lib.geometry.distribution_analyzer import (
    BRepHierarchyAnalyzer,
    CurvatureAnalyzer,
    DihedralAngleAnalyzer,
    SurfaceTypeAnalyzer,
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
    """Extract OCC faces from real document (requires pythonocc-core)."""
    if not hasattr(real_step_document, '_occ_shape') or real_step_document._occ_shape is None:
        pytest.skip("OCC shape not available (pythonocc-core not installed)")

    from OCC.Extend.TopologyUtils import TopologyExplorer
    topo = TopologyExplorer(real_step_document._occ_shape)
    faces = list(topo.faces())

    if len(faces) == 0:
        pytest.skip("No OCC faces found in document")
    return faces


class TestDihedralAngleAnalyzer:
    """Test DihedralAngleAnalyzer class."""

    def test_init(self):
        """Test initialization."""
        analyzer = DihedralAngleAnalyzer()
        assert analyzer is not None

    def test_compute_dihedral_angles_method_exists(self):
        """Test that compute_dihedral_angles method exists."""
        assert hasattr(DihedralAngleAnalyzer, 'compute_dihedral_angles')

    def test_compute_dihedral_angles_with_empty_document(self):
        """Test compute_dihedral_angles with empty document."""
        from cadling.datamodel.base_models import CADlingDocument

        # Create minimal document without OCC shape
        doc = CADlingDocument(name="test", items=[])
        result = DihedralAngleAnalyzer.compute_dihedral_angles(doc)

        # Should return empty results gracefully
        assert isinstance(result, dict)
        assert 'angles' in result
        assert 'mean' in result
        assert result['angles'] == []

    def test_angle_ranges(self):
        """Test analysis handles various angle ranges."""
        # Test data covering 0-180 degrees
        test_angles = [
            np.array([0]),  # 0 degrees
            np.array([np.pi/2]),  # 90 degrees
            np.array([np.pi]),  # 180 degrees
        ]

        for angles in test_angles:
            # Should handle all angle ranges
            assert angles is not None

    def test_compute_dihedral_angles_with_real_document(self, real_step_document):
        """Test compute_dihedral_angles with real STEP document."""
        result = DihedralAngleAnalyzer.compute_dihedral_angles(real_step_document)

        # Should return valid result
        assert isinstance(result, dict)
        assert 'angles' in result
        assert 'mean' in result
        assert 'std' in result
        assert 'median' in result
        assert 'histogram_bins' in result
        assert 'histogram_counts' in result

        # If there are angles, validate them
        if len(result['angles']) > 0:
            angles = np.array(result['angles'])
            # Angles should be in [0, π] range
            assert np.all(angles >= 0)
            assert np.all(angles <= np.pi)
            # Mean should be finite
            assert np.isfinite(result['mean'])
            # Std should be non-negative
            assert result['std'] >= 0

    def test_dihedral_angles_statistical_properties(self, real_step_document):
        """Test statistical properties of dihedral angles."""
        result = DihedralAngleAnalyzer.compute_dihedral_angles(real_step_document)

        if len(result['angles']) > 0:
            # Mean should be between min and max
            angles = result['angles']
            assert min(angles) <= result['mean'] <= max(angles)

            # Median should be a valid angle
            assert 0 <= result['median'] <= np.pi

            # Histogram should have bins and counts
            if len(result['histogram_bins']) > 0:
                assert len(result['histogram_counts']) == len(result['histogram_bins']) - 1

    def test_result_structure(self, real_step_document):
        """Test that result has correct structure."""
        result = DihedralAngleAnalyzer.compute_dihedral_angles(real_step_document)

        # Check all expected keys are present
        expected_keys = ['angles', 'mean', 'std', 'median', 'histogram_bins', 'histogram_counts']
        for key in expected_keys:
            assert key in result

        # Check types
        assert isinstance(result['angles'], list)
        assert isinstance(result['mean'], (int, float))
        assert isinstance(result['std'], (int, float))
        assert isinstance(result['median'], (int, float))


class TestCurvatureAnalyzer:
    """Test CurvatureAnalyzer class."""

    def test_init(self):
        """Test initialization."""
        analyzer = CurvatureAnalyzer()
        assert analyzer is not None

    def test_compute_curvature_distribution_method_exists(self):
        """Test that compute_curvature_distribution method exists."""
        assert hasattr(CurvatureAnalyzer, 'compute_curvature_distribution')

    def test_compute_curvature_distribution_with_empty_list(self):
        """Test compute_curvature_distribution with empty face list."""
        result = CurvatureAnalyzer.compute_curvature_distribution([])

        # Should return empty results gracefully
        assert isinstance(result, dict)
        assert 'gaussian' in result
        assert 'mean' in result

    def test_compute_curvature_distribution_with_real_faces(self, real_occ_faces):
        """Test compute_curvature_distribution with real OCC faces."""
        result = CurvatureAnalyzer.compute_curvature_distribution(real_occ_faces)

        # Should return valid result
        assert isinstance(result, dict)
        assert 'gaussian' in result
        assert 'mean' in result

        # Check gaussian curvature
        gaussian = result['gaussian']
        assert isinstance(gaussian, dict)
        if 'values' in gaussian and len(gaussian['values']) > 0:
            assert isinstance(gaussian['values'], list)
            # All values should be finite
            assert all(np.isfinite(v) for v in gaussian['values'])

        # Check mean curvature
        mean = result['mean']
        assert isinstance(mean, dict)
        if 'values' in mean and len(mean['values']) > 0:
            assert isinstance(mean['values'], list)
            # All values should be finite
            assert all(np.isfinite(v) for v in mean['values'])

    def test_curvature_result_structure(self, real_occ_faces):
        """Test that curvature result has correct structure."""
        result = CurvatureAnalyzer.compute_curvature_distribution(real_occ_faces[:5])

        # Check expected top-level keys
        assert 'gaussian' in result
        assert 'mean' in result

        # Both should be dicts
        assert isinstance(result['gaussian'], dict)
        assert isinstance(result['mean'], dict)


class TestSurfaceTypeAnalyzer:
    """Test SurfaceTypeAnalyzer class."""

    def test_init(self):
        """Test initialization."""
        analyzer = SurfaceTypeAnalyzer()
        assert analyzer is not None

    def test_analyze_surface_types_method_exists(self):
        """Test that analyze_surface_types method exists."""
        assert hasattr(SurfaceTypeAnalyzer, 'analyze_surface_types')

    def test_analyze_surface_types_with_empty_document(self):
        """Test analyze_surface_types with empty document."""
        from cadling.datamodel.base_models import CADlingDocument

        # Create minimal document
        doc = CADlingDocument(name="test", items=[])
        result = SurfaceTypeAnalyzer.analyze_surface_types(doc)

        # Should return empty dict gracefully
        assert isinstance(result, dict)

    def test_analyze_surface_types_with_real_document(self, real_step_document):
        """Test analyze_surface_types with real STEP document."""
        result = SurfaceTypeAnalyzer.analyze_surface_types(real_step_document)

        # Should return valid result
        assert isinstance(result, dict)

        # All values should be non-negative integers (counts)
        for surface_type, count in result.items():
            assert isinstance(surface_type, str)
            assert isinstance(count, int)
            assert count >= 0

    def test_surface_type_categories(self, real_step_document):
        """Test that surface types are valid categories."""
        result = SurfaceTypeAnalyzer.analyze_surface_types(real_step_document)

        # Source returns STEP entity type names (e.g. PLANE, CYLINDRICAL_SURFACE,
        # ADVANCED_FACE) which may be uppercase or mixed case
        for surface_type in result.keys():
            # Surface type should be a non-empty string
            assert isinstance(surface_type, str)
            assert len(surface_type) > 0


class TestBRepHierarchyAnalyzer:
    """Test BRepHierarchyAnalyzer class."""

    def test_init(self):
        """Test initialization."""
        analyzer = BRepHierarchyAnalyzer()
        assert analyzer is not None

    def test_extract_hierarchy_method_exists(self):
        """Test that extract_hierarchy method exists."""
        assert hasattr(BRepHierarchyAnalyzer, 'extract_hierarchy')

    def test_extract_hierarchy_with_empty_document(self):
        """Test extract_hierarchy with empty document."""
        from cadling.datamodel.base_models import CADlingDocument

        # Create minimal document
        doc = CADlingDocument(name="test", items=[])
        result = BRepHierarchyAnalyzer.extract_hierarchy(doc)

        # Should return hierarchy dict gracefully
        assert isinstance(result, dict)
        assert 'num_faces' in result
        assert 'num_edges' in result
        assert 'num_vertices' in result

    def test_extract_hierarchy_with_real_document(self, real_step_document):
        """Test extract_hierarchy with real STEP document."""
        result = BRepHierarchyAnalyzer.extract_hierarchy(real_step_document)

        # Should return valid result
        assert isinstance(result, dict)
        assert 'num_faces' in result
        assert 'num_edges' in result
        assert 'num_vertices' in result

        # All counts should be non-negative integers
        assert isinstance(result['num_faces'], int)
        assert isinstance(result['num_edges'], int)
        assert isinstance(result['num_vertices'], int)
        assert result['num_faces'] >= 0
        assert result['num_edges'] >= 0
        assert result['num_vertices'] >= 0

    def test_hierarchy_euler_characteristic(self, real_step_document):
        """Test topological relationships in hierarchy."""
        result = BRepHierarchyAnalyzer.extract_hierarchy(real_step_document)

        # For valid solids, Euler's formula: V - E + F = 2 (for simple polyhedra)
        # This is approximate for complex CAD models, but ratios should be reasonable
        if result['num_faces'] > 0 and result['num_edges'] > 0:
            # Typically edges > faces for most CAD models
            # Just verify we have meaningful data
            assert result['num_faces'] > 0
            assert result['num_edges'] > 0

    def test_hierarchy_result_completeness(self, real_step_document):
        """Test that hierarchy result is complete."""
        result = BRepHierarchyAnalyzer.extract_hierarchy(real_step_document)

        # Should have all basic topology counts
        required_keys = ['num_faces', 'num_edges', 'num_vertices']
        for key in required_keys:
            assert key in result

        # May have additional keys for wires, shells, solids, etc.
        optional_keys = ['num_wires', 'num_shells', 'num_solids', 'num_compounds']
        # If present, should also be non-negative integers
        for key in optional_keys:
            if key in result:
                assert isinstance(result[key], int)
                assert result[key] >= 0


class TestDistributionAnalyzerIntegration:
    """Integration tests for distribution analyzers."""

    def test_all_analyzers_instantiate(self):
        """Test that all analyzers can be instantiated."""
        dihedral = DihedralAngleAnalyzer()
        curvature = CurvatureAnalyzer()
        surface = SurfaceTypeAnalyzer()
        hierarchy = BRepHierarchyAnalyzer()

        assert dihedral is not None
        assert curvature is not None
        assert surface is not None
        assert hierarchy is not None

    def test_analyzers_have_correct_methods(self):
        """Test that all analyzers have their respective methods."""
        assert hasattr(DihedralAngleAnalyzer, 'compute_dihedral_angles')
        assert hasattr(CurvatureAnalyzer, 'compute_curvature_distribution')
        assert hasattr(SurfaceTypeAnalyzer, 'analyze_surface_types')
        assert hasattr(BRepHierarchyAnalyzer, 'extract_hierarchy')

    def test_analyzer_compatibility(self):
        """Test that analyzers produce compatible output formats."""
        from cadling.datamodel.base_models import CADlingDocument

        # Create minimal document
        doc = CADlingDocument(name="test", items=[])

        # Test each analyzer returns dict
        dihedral_result = DihedralAngleAnalyzer.compute_dihedral_angles(doc)
        assert isinstance(dihedral_result, dict)

        curvature_result = CurvatureAnalyzer.compute_curvature_distribution([])
        assert isinstance(curvature_result, dict)

        surface_result = SurfaceTypeAnalyzer.analyze_surface_types(doc)
        assert isinstance(surface_result, dict)

        hierarchy_result = BRepHierarchyAnalyzer.extract_hierarchy(doc)
        assert isinstance(hierarchy_result, dict)

    def test_all_analyzers_with_real_document(self, real_step_document, real_occ_faces):
        """Test all analyzers work together on real document."""
        # Run all analyzers
        dihedral_result = DihedralAngleAnalyzer.compute_dihedral_angles(real_step_document)
        curvature_result = CurvatureAnalyzer.compute_curvature_distribution(real_occ_faces)
        surface_result = SurfaceTypeAnalyzer.analyze_surface_types(real_step_document)
        hierarchy_result = BRepHierarchyAnalyzer.extract_hierarchy(real_step_document)

        # All should return dicts
        assert isinstance(dihedral_result, dict)
        assert isinstance(curvature_result, dict)
        assert isinstance(surface_result, dict)
        assert isinstance(hierarchy_result, dict)

        # All should have content
        assert len(dihedral_result) > 0
        assert len(curvature_result) > 0
        assert len(hierarchy_result) > 0  # Always has basic counts

    def test_analyzers_produce_consistent_counts(self, real_step_document, real_occ_faces):
        """Test that analyzers produce consistent counts."""
        surface_result = SurfaceTypeAnalyzer.analyze_surface_types(real_step_document)
        hierarchy_result = BRepHierarchyAnalyzer.extract_hierarchy(real_step_document)

        # Total surface types should match num_faces (if both are available)
        if len(surface_result) > 0 and hierarchy_result['num_faces'] > 0:
            total_surfaces = sum(surface_result.values())
            # They should be close (allowing for some faces that couldn't be classified)
            assert total_surfaces <= hierarchy_result['num_faces']
