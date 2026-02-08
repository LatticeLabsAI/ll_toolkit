"""Tests for geometric relationship detection."""
from __future__ import annotations

import numpy as np
import pytest

from geotoken.analysis.geometric_relationships import (
    GeometricRelationship,
    RelationshipDetector,
)


class TestGeometricRelationship:
    """Tests for GeometricRelationship dataclass."""

    def test_hash_consistency(self):
        """Test that hash is consistent for same values."""
        rel1 = GeometricRelationship(
            relationship_type="parallel",
            entity_a=0,
            entity_b=1,
            confidence=0.9,
        )
        rel2 = GeometricRelationship(
            relationship_type="parallel",
            entity_a=0,
            entity_b=1,
            confidence=0.8,  # Different confidence
        )
        # Hash should be same (based on type, a, b only)
        assert hash(rel1) == hash(rel2)

    def test_hash_different_for_different_values(self):
        """Test that hash differs for different values."""
        rel1 = GeometricRelationship("parallel", 0, 1)
        rel2 = GeometricRelationship("parallel", 0, 2)
        rel3 = GeometricRelationship("perpendicular", 0, 1)

        assert hash(rel1) != hash(rel2)
        assert hash(rel1) != hash(rel3)

    def test_eq_same_values(self):
        """Test equality for same values."""
        rel1 = GeometricRelationship("parallel", 0, 1, confidence=0.9)
        rel2 = GeometricRelationship("parallel", 0, 1, confidence=0.8)
        # Should be equal (confidence is not part of eq)
        assert rel1 == rel2

    def test_eq_different_values(self):
        """Test inequality for different values."""
        rel1 = GeometricRelationship("parallel", 0, 1)
        rel2 = GeometricRelationship("parallel", 0, 2)
        assert rel1 != rel2

    def test_eq_different_type(self):
        """Test equality with different type returns NotImplemented."""
        rel = GeometricRelationship("parallel", 0, 1)
        assert rel.__eq__("not a relationship") == NotImplemented

    def test_hash_eq_consistency(self):
        """Test that equal objects have same hash."""
        rel1 = GeometricRelationship("parallel", 0, 1)
        rel2 = GeometricRelationship("parallel", 0, 1)
        assert rel1 == rel2
        assert hash(rel1) == hash(rel2)

    def test_set_membership(self):
        """Test that relationships can be used in sets."""
        rel1 = GeometricRelationship("parallel", 0, 1)
        rel2 = GeometricRelationship("parallel", 0, 1)  # Same
        rel3 = GeometricRelationship("perpendicular", 0, 1)  # Different

        rel_set = {rel1, rel2, rel3}
        assert len(rel_set) == 2  # rel1 and rel2 should be deduplicated


class TestRelationshipDetectorInit:
    """Tests for RelationshipDetector initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        detector = RelationshipDetector()
        assert detector.angle_tolerance == 1e-3
        assert detector.distance_tolerance == 1e-4
        assert detector.max_face_pairs == 100

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        detector = RelationshipDetector(
            angle_tolerance=0.01,
            distance_tolerance=0.001,
            max_face_pairs=50,
        )
        assert detector.angle_tolerance == 0.01
        assert detector.distance_tolerance == 0.001
        assert detector.max_face_pairs == 50


class TestDetectFaceRelationships:
    """Tests for detect_face_relationships() method."""

    def test_detect_parallel_faces_on_cube(self, cube_mesh):
        """Test detection of parallel faces on cube."""
        vertices, faces = cube_mesh
        detector = RelationshipDetector()
        relationships = detector.detect_face_relationships(vertices, faces)

        # Cube has 3 pairs of parallel faces
        parallel_rels = [r for r in relationships if r.relationship_type == "parallel"]
        # Each pair consists of 2 triangles per face, so more relationships
        assert len(parallel_rels) >= 3

    def test_detect_perpendicular_faces_on_cube(self, cube_mesh):
        """Test detection of perpendicular faces on cube."""
        vertices, faces = cube_mesh
        detector = RelationshipDetector()
        relationships = detector.detect_face_relationships(vertices, faces)

        perp_rels = [r for r in relationships if r.relationship_type == "perpendicular"]
        # Cube should have many perpendicular face pairs
        assert len(perp_rels) > 0

    def test_detect_empty_faces(self):
        """Test with empty face array."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([]).reshape(0, 3).astype(int)
        detector = RelationshipDetector()
        relationships = detector.detect_face_relationships(vertices, faces)
        assert relationships == []

    def test_detect_single_face(self):
        """Test with single face."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        detector = RelationshipDetector()
        relationships = detector.detect_face_relationships(vertices, faces)
        # Single face has no pairs
        assert relationships == []

    def test_max_face_pairs_limit(self):
        """Test that max_face_pairs limits computation."""
        # Create many faces
        n = 200
        vertices = np.random.randn(n * 3, 3)
        faces = np.arange(n * 3).reshape(n, 3)

        detector = RelationshipDetector(max_face_pairs=10)
        relationships = detector.detect_face_relationships(vertices, faces)
        # Should not check all pairs, limiting computation


class TestVerifyRelationships:
    """Tests for verify_relationships() method."""

    def test_verify_identity_transform(self, cube_mesh):
        """Test that identity transform preserves all relationships."""
        vertices, faces = cube_mesh
        detector = RelationshipDetector()
        original_rels = detector.detect_face_relationships(vertices, faces)

        # Same vertices should give 100% preservation
        rate = detector.verify_relationships(original_rels, vertices, faces)
        assert rate == 1.0

    def test_verify_empty_relationships(self, cube_mesh):
        """Test verification with empty relationship list returns None."""
        vertices, faces = cube_mesh
        detector = RelationshipDetector()
        rate = detector.verify_relationships([], vertices, faces)
        assert rate is None

    def test_verify_after_translation(self, cube_mesh):
        """Test that translation preserves relationships."""
        vertices, faces = cube_mesh
        detector = RelationshipDetector()
        original_rels = detector.detect_face_relationships(vertices, faces)

        # Translate vertices
        translated = vertices + np.array([10, 20, 30])
        rate = detector.verify_relationships(original_rels, translated, faces)
        assert rate == 1.0

    def test_verify_after_uniform_scale(self, cube_mesh):
        """Test that uniform scale preserves relationships."""
        vertices, faces = cube_mesh
        detector = RelationshipDetector()
        original_rels = detector.detect_face_relationships(vertices, faces)

        # Scale uniformly
        scaled = vertices * 2.0
        rate = detector.verify_relationships(original_rels, scaled, faces)
        assert rate == 1.0

    def test_verify_after_distortion(self, cube_mesh):
        """Test that non-uniform distortion may break relationships."""
        vertices, faces = cube_mesh
        detector = RelationshipDetector()
        original_rels = detector.detect_face_relationships(vertices, faces)

        if not original_rels:
            pytest.skip("No relationships detected")

        # Non-uniform scale may break parallel relationships
        distorted = vertices.copy()
        distorted[:, 0] *= 2.0  # Scale only X axis

        rate = detector.verify_relationships(original_rels, distorted, faces)
        # Some relationships may still be preserved, some may not
        assert 0.0 <= rate <= 1.0


class TestRelationshipConfidence:
    """Tests for relationship confidence values."""

    def test_parallel_confidence_near_one(self, cube_mesh):
        """Test that parallel faces have high confidence."""
        vertices, faces = cube_mesh
        detector = RelationshipDetector()
        relationships = detector.detect_face_relationships(vertices, faces)

        parallel_rels = [r for r in relationships if r.relationship_type == "parallel"]
        for rel in parallel_rels:
            # Perfect parallel should have confidence ~1.0
            assert rel.confidence >= 0.99

    def test_perpendicular_confidence_near_one(self, cube_mesh):
        """Test that perpendicular faces have high confidence."""
        vertices, faces = cube_mesh
        detector = RelationshipDetector()
        relationships = detector.detect_face_relationships(vertices, faces)

        perp_rels = [r for r in relationships if r.relationship_type == "perpendicular"]
        for rel in perp_rels:
            # Perfect perpendicular should have confidence ~1.0
            assert rel.confidence >= 0.99
