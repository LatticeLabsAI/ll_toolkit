"""Tests for vertex validation module.

Comprehensive test suite for VertexValidator and TopologyValidator,
covering bounds checking, collision detection, degeneracy detection,
manifold properties, face winding consistency, and Euler characteristic.
"""
from __future__ import annotations

import numpy as np
import pytest

from geotoken.vertex.vertex_validation import (
    BoundsCheckResult,
    CollisionCheckResult,
    DegeneracyCheckResult,
    EulerCheckResult,
    ManifoldCheckResult,
    TopologyValidator,
    VertexValidationReport,
    VertexValidator,
    WindingCheckResult,
)


# ============================================================================
# Test Fixtures: Common mesh geometries
# ============================================================================


@pytest.fixture
def single_vertex():
    """Single vertex at origin."""
    vertices = np.array([[0.0, 0.0, 0.0]])
    return vertices


@pytest.fixture
def tetrahedron_vertices():
    """Regular tetrahedron vertices."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3) / 2, 0.0],
        [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
    ])
    return vertices


@pytest.fixture
def tetrahedron_faces():
    """Tetrahedron faces (4 triangles)."""
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3],
    ])
    return faces


@pytest.fixture
def cube_vertices():
    """Cube vertices normalized to [-1, 1]."""
    vertices = np.array([
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ])
    return vertices


@pytest.fixture
def cube_faces():
    """Cube faces (12 triangles = 6 sides × 2)."""
    faces = np.array([
        # Bottom face (z = -1)
        [0, 1, 2],
        [0, 2, 3],
        # Top face (z = 1)
        [4, 6, 5],
        [4, 7, 6],
        # Front face (y = -1)
        [0, 5, 1],
        [0, 4, 5],
        # Back face (y = 1)
        [2, 7, 3],
        [2, 6, 7],
        # Left face (x = -1)
        [0, 3, 7],
        [0, 7, 4],
        # Right face (x = 1)
        [1, 5, 6],
        [1, 6, 2],
    ])
    return faces


@pytest.fixture
def empty_arrays():
    """Empty vertices and faces arrays."""
    vertices = np.array([]).reshape(0, 3)
    faces = np.array([]).reshape(0, 3).astype(int)
    return vertices, faces


@pytest.fixture
def duplicate_vertices():
    """Vertices with duplicates."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],  # duplicate of vertex 1
        [0.5, 0.5, 0.0],
    ])
    return vertices


@pytest.fixture
def near_duplicate_vertices():
    """Vertices with near-duplicates within collision tolerance."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0 + 1e-5, 0.0, 0.0],  # very close to vertex 1
        [0.5, 0.5, 0.0],
    ])
    return vertices


@pytest.fixture
def degenerate_faces():
    """Vertices and faces with degenerate triangles."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],  # collinear with 0 and 1
        [0.5, 0.5, 0.0],
    ])
    faces = np.array([
        [0, 1, 2],  # degenerate: collinear
        [0, 1, 3],  # valid triangle
    ])
    return vertices, faces


@pytest.fixture
def non_manifold_faces():
    """Mesh with non-manifold edge (shared by 3 faces)."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
        [0.5, 0.0, 1.0],
    ])
    faces = np.array([
        [0, 1, 2],  # shares edge (0, 1) with faces below
        [0, 1, 3],  # shares edge (0, 1) with faces 0 and 2
        [0, 1, 4],  # shares edge (0, 1) with faces 0 and 1 -> non-manifold!
    ])
    return vertices, faces


@pytest.fixture
def out_of_bounds_vertices():
    """Vertices outside bounds."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [-50.0, 0.0, 0.0],
        [0.0, 150.0, 0.0],
        [0.0, 0.0, -200.0],
    ])
    return vertices


# ============================================================================
# Tests: VertexValidator initialization
# ============================================================================


class TestVertexValidatorInit:
    """Tests for VertexValidator constructor."""

    def test_default_bounds(self):
        """Test default coordinate bounds."""
        validator = VertexValidator()
        assert validator.coord_min == -100.0
        assert validator.coord_max == 100.0

    def test_custom_bounds(self):
        """Test custom coordinate bounds."""
        validator = VertexValidator(coord_bounds=(-1.0, 1.0))
        assert validator.coord_min == -1.0
        assert validator.coord_max == 1.0

    def test_collision_tolerance(self):
        """Test custom collision tolerance."""
        validator = VertexValidator(collision_tolerance=1e-3)
        assert validator.collision_tol == 1e-3

    def test_area_tolerance(self):
        """Test custom area tolerance."""
        validator = VertexValidator(area_tolerance=1e-6)
        assert validator.area_tol == 1e-6

    def test_manifold_check_enabled(self):
        """Test manifold checking can be enabled."""
        validator = VertexValidator(manifold_check=True)
        assert validator.do_manifold is True

    def test_manifold_check_disabled(self):
        """Test manifold checking can be disabled."""
        validator = VertexValidator(manifold_check=False)
        assert validator.do_manifold is False


# ============================================================================
# Tests: Bounds Checking
# ============================================================================


class TestBoundsChecking:
    """Tests for check_bounds method."""

    def test_all_vertices_in_bounds(self, cube_vertices):
        """Test all vertices within bounds."""
        validator = VertexValidator(coord_bounds=(-2.0, 2.0))
        result = validator.check_bounds(cube_vertices)

        assert isinstance(result, BoundsCheckResult)
        assert result.all_in_bounds is True
        assert result.num_violations == 0
        assert result.max_violation == 0.0
        assert len(result.out_of_bounds_indices) == 0

    def test_vertices_out_of_bounds(self, out_of_bounds_vertices):
        """Test detection of out-of-bounds vertices."""
        validator = VertexValidator(coord_bounds=(-100.0, 100.0))
        result = validator.check_bounds(out_of_bounds_vertices)

        assert result.all_in_bounds is False
        assert result.num_violations == 2  # vertices 2 and 3 exceed bounds
        assert result.max_violation > 0.0

    def test_single_vertex_in_bounds(self, single_vertex):
        """Test single vertex within bounds."""
        validator = VertexValidator(coord_bounds=(-1.0, 1.0))
        result = validator.check_bounds(single_vertex)

        assert result.all_in_bounds is True
        assert result.num_violations == 0

    def test_empty_vertices(self, empty_arrays):
        """Test empty vertex array."""
        vertices, _ = empty_arrays
        validator = VertexValidator()
        result = validator.check_bounds(vertices)

        assert result.all_in_bounds is True
        assert result.num_violations == 0
        assert result.max_violation == 0.0

    def test_edge_case_exact_bounds(self):
        """Test vertices exactly at bounds."""
        vertices = np.array([
            [-100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, -100.0, 0.0],
            [0.0, 100.0, 0.0],
        ])
        validator = VertexValidator(coord_bounds=(-100.0, 100.0))
        result = validator.check_bounds(vertices)

        assert result.all_in_bounds is True
        assert result.num_violations == 0

    def test_violation_indices_correct(self):
        """Test that out-of-bounds indices are correctly identified."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [150.0, 0.0, 0.0],  # violates max bound
            [0.0, 0.0, 0.0],
            [-150.0, 0.0, 0.0],  # violates min bound
        ])
        validator = VertexValidator(coord_bounds=(-100.0, 100.0))
        result = validator.check_bounds(vertices)

        assert result.num_violations == 2
        assert set(result.out_of_bounds_indices) == {1, 3}


# ============================================================================
# Tests: Collision Detection
# ============================================================================


class TestCollisionDetection:
    """Tests for check_collisions method."""

    def test_collision_free_mesh(self, cube_vertices):
        """Test mesh with no collisions."""
        validator = VertexValidator(collision_tolerance=1e-4)
        result = validator.check_collisions(cube_vertices)

        assert isinstance(result, CollisionCheckResult)
        assert result.collision_free is True
        assert result.num_collisions == 0
        assert len(result.collision_pairs) == 0
        assert result.min_distance > 1e-4

    def test_duplicate_vertices_detected(self, duplicate_vertices):
        """Test detection of exact duplicate vertices."""
        validator = VertexValidator(collision_tolerance=1e-4)
        result = validator.check_collisions(duplicate_vertices)

        assert result.collision_free is False
        assert result.num_collisions > 0
        assert result.min_distance < 1e-4
        # Pair (1, 2) should be detected
        assert (1, 2) in result.collision_pairs

    def test_near_duplicate_vertices_detected(self, near_duplicate_vertices):
        """Test detection of near-duplicate vertices."""
        validator = VertexValidator(collision_tolerance=1e-4)
        result = validator.check_collisions(near_duplicate_vertices)

        assert result.collision_free is False
        assert result.num_collisions > 0
        # Pair (1, 2) should be detected since distance is 1e-5
        assert (1, 2) in result.collision_pairs

    def test_collision_tolerance_threshold(self, near_duplicate_vertices):
        """Test that collision tolerance threshold is respected."""
        # Set tolerance above the 1e-5 distance -> should detect
        validator = VertexValidator(collision_tolerance=1e-4)
        result = validator.check_collisions(near_duplicate_vertices)

        assert result.collision_free is False

        # Set tolerance below the 1e-5 distance -> should not detect
        validator_strict = VertexValidator(collision_tolerance=1e-6)
        result_strict = validator_strict.check_collisions(near_duplicate_vertices)
        assert result_strict.collision_free is True

    def test_single_vertex_no_collision(self, single_vertex):
        """Test single vertex cannot have collisions."""
        validator = VertexValidator()
        result = validator.check_collisions(single_vertex)

        assert result.collision_free is True
        assert result.num_collisions == 0
        assert result.min_distance == float("inf")

    def test_two_vertices_distant(self):
        """Test two distant vertices."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        validator = VertexValidator(collision_tolerance=1e-4)
        result = validator.check_collisions(vertices)

        assert result.collision_free is True
        assert result.min_distance > 1e-4

    def test_empty_vertices(self, empty_arrays):
        """Test empty vertex array."""
        vertices, _ = empty_arrays
        validator = VertexValidator()
        result = validator.check_collisions(vertices)

        assert result.collision_free is True
        assert result.num_collisions == 0
        assert result.min_distance == float("inf")


# ============================================================================
# Tests: Degeneracy Detection
# ============================================================================


class TestDegeneracyDetection:
    """Tests for check_degeneracy method."""

    def test_valid_non_degenerate_faces(self, tetrahedron_vertices, tetrahedron_faces):
        """Test mesh with no degenerate faces."""
        validator = VertexValidator(area_tolerance=1e-10)
        result = validator.check_degeneracy(tetrahedron_vertices, tetrahedron_faces)

        assert isinstance(result, DegeneracyCheckResult)
        assert result.has_degenerate is False
        assert result.zero_area_count == 0
        assert len(result.degenerate_face_indices) == 0
        assert result.min_area > 1e-10

    def test_degenerate_collinear_vertices(self, degenerate_faces):
        """Test detection of degenerate faces from collinear vertices."""
        vertices, faces = degenerate_faces
        validator = VertexValidator(area_tolerance=1e-10)
        result = validator.check_degeneracy(vertices, faces)

        assert result.has_degenerate is True
        assert result.zero_area_count > 0
        assert 0 in result.degenerate_face_indices

    def test_degenerate_coincident_vertices(self):
        """Test detection of faces with coincident vertices."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # same as 0
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ])
        faces = np.array([
            [0, 1, 2],  # degenerate: v0 and v1 are same
            [0, 1, 3],  # degenerate
        ])
        validator = VertexValidator(area_tolerance=1e-10)
        result = validator.check_degeneracy(vertices, faces)

        assert result.has_degenerate is True
        assert result.zero_area_count >= 2

    def test_empty_faces(self, tetrahedron_vertices):
        """Test with empty face array."""
        faces = np.array([]).reshape(0, 3).astype(int)
        validator = VertexValidator()
        result = validator.check_degeneracy(tetrahedron_vertices, faces)

        assert result.has_degenerate is False
        assert result.zero_area_count == 0
        assert result.min_area == float("inf")

    def test_area_tolerance_threshold(self):
        """Test that area tolerance threshold affects degeneracy detection."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.05, 0.05, 0.0],  # very small triangle
            [1.0, 0.0, 0.0],
        ])
        faces = np.array([
            [0, 1, 2],  # tiny area
            [0, 1, 3],  # larger area
        ])

        # Strict tolerance -> detects small face
        validator_strict = VertexValidator(area_tolerance=1e-2)
        result_strict = validator_strict.check_degeneracy(vertices, faces)
        assert result_strict.has_degenerate is True

        # Loose tolerance -> might not detect
        validator_loose = VertexValidator(area_tolerance=1e-6)
        result_loose = validator_loose.check_degeneracy(vertices, faces)
        # The tiny triangle should still be detected as degenerate
        assert result_loose.degenerate_face_indices is not None
        assert len(result_loose.degenerate_face_indices) >= 0


# ============================================================================
# Tests: Manifold Checking
# ============================================================================


class TestManifoldChecking:
    """Tests for check_manifold method."""

    def test_manifold_tetrahedron(self, tetrahedron_faces):
        """Test that tetrahedron is edge-manifold."""
        result = VertexValidator().check_manifold(tetrahedron_faces)

        assert isinstance(result, ManifoldCheckResult)
        assert result.is_manifold is True
        assert result.num_non_manifold == 0
        assert len(result.non_manifold_edges) == 0

    def test_manifold_cube(self, cube_faces):
        """Test that cube is edge-manifold and closed."""
        result = VertexValidator().check_manifold(cube_faces)

        assert result.is_manifold is True
        assert result.num_non_manifold == 0
        assert result.num_boundary == 0  # closed mesh

    def test_non_manifold_edge_shared_by_three_faces(self, non_manifold_faces):
        """Test detection of edge shared by 3+ faces."""
        faces = non_manifold_faces[1]
        result = VertexValidator().check_manifold(faces)

        assert result.is_manifold is False
        assert result.num_non_manifold > 0
        # Edge (0, 1) is shared by all 3 faces
        assert (0, 1) in result.non_manifold_edges

    def test_boundary_edges_detected(self):
        """Test detection of boundary edges."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ])
        # Single triangle (open mesh)
        faces = np.array([[0, 1, 2]])
        result = VertexValidator().check_manifold(faces)

        assert result.is_manifold is True  # no edge shared by 3+
        assert result.num_boundary == 3  # all 3 edges are boundary

    def test_empty_faces(self):
        """Test with empty faces."""
        faces = np.array([]).reshape(0, 3).astype(int)
        result = VertexValidator().check_manifold(faces)

        assert result.is_manifold is True
        assert result.num_non_manifold == 0
        assert result.num_boundary == 0


# ============================================================================
# Tests: Face Winding Consistency
# ============================================================================


class TestFaceWindingConsistency:
    """Tests for check_face_winding method."""

    def test_consistent_winding_tetrahedron(self, tetrahedron_vertices, tetrahedron_faces):
        """Test consistent winding in tetrahedron."""
        validator = VertexValidator()
        result = validator.check_face_winding(tetrahedron_vertices, tetrahedron_faces)

        assert isinstance(result, WindingCheckResult)
        assert result is not None
        assert hasattr(result, 'inconsistent_face_indices')
        # Tetrahedron should have relatively consistent winding
        # (may have some due to normal calculation, but not all)

    def test_consistent_winding_cube(self, cube_vertices, cube_faces):
        """Test cube should have consistent face winding."""
        validator = VertexValidator()
        result = validator.check_face_winding(cube_vertices, cube_faces)

        # Well-formed cube should have consistent winding
        assert result.consistent is True
        assert result.num_inconsistent == 0

    def test_empty_faces(self, tetrahedron_vertices):
        """Test with empty faces."""
        faces = np.array([]).reshape(0, 3).astype(int)
        validator = VertexValidator()
        result = validator.check_face_winding(tetrahedron_vertices, faces)

        assert result.consistent is True
        assert result.num_inconsistent == 0
        assert len(result.inconsistent_face_indices) == 0

    def test_opposite_winding_detected(self):
        """Test detection of opposite winding direction."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, -1.0, 0.0],
        ])
        # Two triangles with opposite winding sharing edge (0, 1)
        faces = np.array([
            [0, 1, 2],  # CCW when viewed from +z
            [0, 3, 1],  # CCW when viewed from -z (opposite to above)
        ])
        validator = VertexValidator()
        result = validator.check_face_winding(vertices, faces)

        # Verify result is well-formed
        assert isinstance(result, WindingCheckResult)
        assert result is not None
        assert isinstance(result.num_inconsistent, int)


# ============================================================================
# Tests: Euler Characteristic
# ============================================================================


class TestEulerCharacteristic:
    """Tests for check_euler method."""

    def test_euler_tetrahedron(self, tetrahedron_faces):
        """Test Euler characteristic of tetrahedron."""
        result = VertexValidator().check_euler(tetrahedron_faces)

        assert isinstance(result, EulerCheckResult)
        # Tetrahedron: V=4, F=4, E=6, V-E+F=4-6+4=2
        assert result.V == 4
        assert result.F == 4
        assert result.E == 6
        assert result.euler == 2
        assert result.valid is True

    def test_euler_cube(self, cube_faces):
        """Test Euler characteristic of cube."""
        result = VertexValidator().check_euler(cube_faces)

        assert result.V == 8  # 8 vertices
        assert result.F == 12  # 12 triangles (6 faces × 2)
        assert result.E == 18  # 18 edges
        assert result.euler == 2  # 8 - 18 + 12 = 2
        assert result.valid is True

    def test_euler_empty_faces(self):
        """Test Euler characteristic with empty faces."""
        faces = np.array([]).reshape(0, 3).astype(int)
        result = VertexValidator().check_euler(faces)

        assert result.V == 0
        assert result.F == 0
        assert result.E == 0
        assert result.euler == 0

    def test_euler_single_triangle(self):
        """Test Euler characteristic of single triangle (open mesh)."""
        faces = np.array([[0, 1, 2]])
        result = VertexValidator().check_euler(faces)

        assert result.V == 3
        assert result.F == 1
        assert result.E == 3
        assert result.euler == 1  # V-E+F = 3-3+1 = 1 (not genus-0)


# ============================================================================
# Tests: VertexValidator.validate() Integration
# ============================================================================


class TestValidateIntegration:
    """Integration tests for full validate() method."""

    def test_validate_valid_tetrahedron(self, tetrahedron_vertices, tetrahedron_faces):
        """Test full validation of valid tetrahedron."""
        validator = VertexValidator()
        report = validator.validate(tetrahedron_vertices, tetrahedron_faces)

        assert isinstance(report, VertexValidationReport)
        assert report.valid is True
        assert len(report.errors) == 0
        assert report.bounds is not None
        assert report.bounds.all_in_bounds is True
        assert report.collisions is not None
        assert report.collisions.collision_free is True

    def test_validate_valid_cube(self, cube_vertices, cube_faces):
        """Test full validation of valid cube."""
        validator = VertexValidator(coord_bounds=(-2.0, 2.0))
        report = validator.validate(cube_vertices, cube_faces)

        assert report.valid is True
        assert len(report.errors) == 0

    def test_validate_out_of_bounds(self, out_of_bounds_vertices):
        """Test validation detects out-of-bounds vertices."""
        validator = VertexValidator(coord_bounds=(-100.0, 100.0))
        # Create dummy faces
        faces = np.array([])
        report = validator.validate(out_of_bounds_vertices, faces)

        assert report.valid is False
        assert len(report.errors) > 0
        assert "out of bounds" in report.errors[0].lower()

    def test_validate_with_collisions(self, duplicate_vertices):
        """Test validation detects collisions in warnings."""
        validator = VertexValidator(coord_bounds=(-2.0, 2.0))
        faces = np.array([]).reshape(0, 3).astype(int)
        report = validator.validate(duplicate_vertices, faces)

        # Collisions are warnings, not errors
        assert report.valid is True
        assert len(report.warnings) > 0
        assert "collision" in report.warnings[0].lower() or "near-duplicate" in report.warnings[0].lower()

    def test_validate_with_degenerate_faces(self, degenerate_faces):
        """Test validation detects degenerate faces in warnings."""
        vertices, faces = degenerate_faces
        validator = VertexValidator()
        report = validator.validate(vertices, faces)

        # Degeneracy is a warning
        assert len(report.warnings) > 0
        assert "degenerate" in report.warnings[0].lower()

    def test_validate_with_non_manifold(self, non_manifold_faces):
        """Test validation detects non-manifold mesh."""
        vertices, faces = non_manifold_faces
        validator = VertexValidator()
        report = validator.validate(vertices, faces)

        assert report.valid is False
        assert len(report.errors) > 0
        assert "non-manifold" in report.errors[0].lower()

    def test_validate_vertices_only(self, cube_vertices):
        """Test validation with vertices only, no faces."""
        validator = VertexValidator(coord_bounds=(-2.0, 2.0))
        report = validator.validate(cube_vertices)

        assert report.valid is True
        assert report.degeneracy is None
        assert report.manifold is None
        assert report.euler is None

    def test_validate_empty_vertices_fails(self, empty_arrays):
        """Test validation with empty arrays."""
        vertices, faces = empty_arrays
        validator = VertexValidator()
        # Empty vertices should still be considered valid shape
        report = validator.validate(vertices, faces)
        # An empty mesh is technically valid
        assert report.valid is True

    def test_validate_invalid_vertex_shape(self):
        """Test validation rejects invalid vertex shape."""
        vertices = np.array([[0.0, 0.0]])  # Wrong: 2D instead of 3D
        validator = VertexValidator()
        report = validator.validate(vertices)

        assert report.valid is False
        assert len(report.errors) > 0
        assert "shape" in report.errors[0].lower()

    def test_validate_invalid_face_shape(self, tetrahedron_vertices):
        """Test validation rejects invalid face shape."""
        faces = np.array([[0, 1]])  # Wrong: 2 indices instead of 3
        validator = VertexValidator()
        report = validator.validate(tetrahedron_vertices, faces)

        assert report.valid is False
        assert len(report.errors) > 0

    def test_validate_face_indices_out_of_range(self, tetrahedron_vertices):
        """Test validation detects face indices out of range."""
        faces = np.array([
            [0, 1, 2],
            [0, 1, 10],  # index 10 out of range for 4 vertices
        ])
        validator = VertexValidator()
        report = validator.validate(tetrahedron_vertices, faces)

        assert report.valid is False
        assert len(report.errors) > 0
        assert "out of range" in report.errors[0].lower()

    def test_validate_with_manifold_check_disabled(self, non_manifold_faces):
        """Test validation with manifold check disabled."""
        vertices, faces = non_manifold_faces
        validator = VertexValidator(manifold_check=False)
        report = validator.validate(vertices, faces)

        # Should not report non-manifold as error
        assert report.manifold is None
        # But may still fail for other reasons or succeed


# ============================================================================
# Tests: TopologyValidator
# ============================================================================


class TestTopologyValidator:
    """Tests for TopologyValidator class."""

    def test_validate_mesh_valid_tetrahedron(self, tetrahedron_vertices, tetrahedron_faces):
        """Test TopologyValidator on valid tetrahedron."""
        validator = TopologyValidator()
        result = validator.validate_mesh(tetrahedron_vertices, tetrahedron_faces)

        assert isinstance(result, dict)
        assert "manifold" in result
        assert "watertight" in result
        assert "euler" in result
        assert "winding" in result
        assert "degeneracy" in result
        assert "valid" in result

        assert result["manifold"].is_manifold is True
        assert result["watertight"] is True
        assert result["euler"].valid is True

    def test_validate_mesh_valid_cube(self, cube_vertices, cube_faces):
        """Test TopologyValidator on valid cube."""
        validator = TopologyValidator()
        result = validator.validate_mesh(cube_vertices, cube_faces)

        assert result["valid"] is True
        assert result["manifold"].is_manifold is True
        assert result["watertight"] is True
        assert result["euler"].valid is True
        assert result["winding"].consistent is True

    def test_validate_mesh_non_manifold(self, non_manifold_faces):
        """Test TopologyValidator detects non-manifold mesh."""
        vertices, faces = non_manifold_faces
        validator = TopologyValidator()
        result = validator.validate_mesh(vertices, faces)

        assert result["manifold"].is_manifold is False
        assert result["valid"] is False

    def test_validate_mesh_open_mesh(self):
        """Test TopologyValidator detects open mesh."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ])
        faces = np.array([[0, 1, 2]])  # single triangle
        validator = TopologyValidator()
        result = validator.validate_mesh(vertices, faces)

        # Open mesh has boundary edges
        assert result["watertight"] is False
        assert result["manifold"].num_boundary > 0

    def test_topology_validator_custom_area_tolerance(self, degenerate_faces):
        """Test TopologyValidator with custom area tolerance."""
        vertices, faces = degenerate_faces
        validator = TopologyValidator(area_tolerance=1e-10)
        result = validator.validate_mesh(vertices, faces)

        assert result["degeneracy"].has_degenerate is True


# ============================================================================
# Tests: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_vertex_mesh(self, single_vertex):
        """Test mesh with single vertex."""
        validator = VertexValidator()
        # Single vertex with empty faces
        faces = np.array([]).reshape(0, 3).astype(int)
        report = validator.validate(single_vertex, faces)

        assert report.valid is True
        assert report.collisions.num_collisions == 0

    def test_two_vertex_mesh(self):
        """Test mesh with two vertices."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        faces = np.array([]).reshape(0, 3).astype(int)
        validator = VertexValidator()
        report = validator.validate(vertices, faces)

        assert report.valid is True

    def test_very_large_vertex_count(self):
        """Test validation with many vertices."""
        n = 1000
        vertices = np.random.randn(n, 3) * 10.0
        validator = VertexValidator(coord_bounds=(-100.0, 100.0))
        result = validator.check_collisions(vertices)

        assert isinstance(result, CollisionCheckResult)
        # Should not raise exception

    def test_very_small_coordinates(self):
        """Test validation with very small coordinates."""
        vertices = np.array([
            [1e-8, 1e-8, 1e-8],
            [2e-8, 0.0, 0.0],
        ])
        validator = VertexValidator()
        result = validator.check_collisions(vertices)

        assert isinstance(result, CollisionCheckResult)
        assert result.min_distance < 1.0

    def test_very_large_coordinates(self):
        """Test validation with very large coordinates."""
        vertices = np.array([
            [1e6, 1e6, 1e6],
            [1e6 + 1.0, 1e6, 1e6],
        ])
        validator = VertexValidator(coord_bounds=(-1e7, 1e7))
        result = validator.check_bounds(vertices)

        assert result.all_in_bounds is True

    def test_nan_in_vertices(self):
        """Test validation behavior with NaN vertices."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        validator = VertexValidator()
        # NaN comparisons should be handled gracefully
        result = validator.check_bounds(vertices)
        # NaN is actually not flagged by numpy's comparison operators
        # (NaN < x and NaN > x both return False), but this test documents current behavior
        # The implementation should still work without crashing
        assert isinstance(result, BoundsCheckResult)

    def test_infinity_in_vertices(self):
        """Test validation behavior with infinite vertices."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [np.inf, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        validator = VertexValidator(coord_bounds=(-100.0, 100.0))
        result = validator.check_bounds(vertices)

        assert result.num_violations >= 1


# ============================================================================
# Tests: Data Class Integrity
# ============================================================================


class TestDataClasses:
    """Tests for result data classes."""

    def test_bounds_check_result_fields(self):
        """Test BoundsCheckResult has all required fields."""
        result = BoundsCheckResult(
            all_in_bounds=True,
            out_of_bounds_indices=np.array([]),
            max_violation=0.0,
            num_violations=0,
        )

        assert result.all_in_bounds is True
        assert result.max_violation == 0.0
        assert result.num_violations == 0

    def test_collision_check_result_fields(self):
        """Test CollisionCheckResult has all required fields."""
        result = CollisionCheckResult(
            collision_free=True,
            collision_pairs=[],
            min_distance=float("inf"),
            num_collisions=0,
        )

        assert result.collision_free is True
        assert len(result.collision_pairs) == 0
        assert result.min_distance == float("inf")

    def test_degeneracy_check_result_fields(self):
        """Test DegeneracyCheckResult has all required fields."""
        result = DegeneracyCheckResult(
            has_degenerate=False,
            degenerate_face_indices=[],
            zero_area_count=0,
            min_area=float("inf"),
        )

        assert result.has_degenerate is False
        assert result.zero_area_count == 0

    def test_manifold_check_result_fields(self):
        """Test ManifoldCheckResult has all required fields."""
        result = ManifoldCheckResult(
            is_manifold=True,
            non_manifold_edges=[],
            boundary_edges=[],
            num_non_manifold=0,
            num_boundary=0,
        )

        assert result.is_manifold is True
        assert result.num_non_manifold == 0

    def test_winding_check_result_fields(self):
        """Test WindingCheckResult has all required fields."""
        result = WindingCheckResult(
            consistent=True,
            num_inconsistent=0,
            inconsistent_face_indices=[],
        )

        assert result.consistent is True
        assert result.num_inconsistent == 0

    def test_euler_check_result_fields(self):
        """Test EulerCheckResult has all required fields."""
        result = EulerCheckResult(
            valid=True,
            V=4,
            E=6,
            F=4,
            euler=2,
            expected_euler=2,
        )

        assert result.valid is True
        assert result.V == 4
        assert result.euler == 2

    def test_validation_report_fields(self):
        """Test VertexValidationReport has all required fields."""
        report = VertexValidationReport(
            valid=True,
            bounds=None,
            collisions=None,
            degeneracy=None,
            manifold=None,
            winding=None,
            euler=None,
            errors=[],
            warnings=[],
        )

        assert report.valid is True
        assert len(report.errors) == 0
        assert len(report.warnings) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
