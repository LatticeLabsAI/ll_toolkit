"""Geometric relationship detection for normalization verification.

Detects and verifies preservation of geometric relationships like
parallel faces, perpendicular edges, and symmetry after normalization.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)


@dataclass
class GeometricRelationship:
    """A detected geometric relationship between entities."""
    relationship_type: str   # "parallel", "perpendicular", "coplanar", "symmetric"
    entity_a: int           # Vertex/face index A
    entity_b: int           # Vertex/face index B
    confidence: float = 1.0 # Detection confidence [0, 1]

    def __hash__(self):
        return hash((self.relationship_type, self.entity_a, self.entity_b))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GeometricRelationship):
            return NotImplemented
        return (
            self.relationship_type == other.relationship_type
            and self.entity_a == other.entity_a
            and self.entity_b == other.entity_b
        )


class RelationshipDetector:
    """Detects geometric relationships in mesh data."""

    def __init__(
        self,
        angle_tolerance: float = 1e-3,
        distance_tolerance: float = 1e-4,
        max_face_pairs: int = 100,
    ):
        """Initialize relationship detector.

        Args:
            angle_tolerance: Tolerance for angle comparisons (radians).
            distance_tolerance: Tolerance for distance comparisons.
            max_face_pairs: Maximum number of face pairs to check to avoid O(n²).
        """
        self.angle_tolerance = angle_tolerance
        self.distance_tolerance = distance_tolerance
        self.max_face_pairs = max_face_pairs

    def detect_face_relationships(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> list[GeometricRelationship]:
        """Detect parallel and perpendicular face relationships.

        Args:
            vertices: (N, 3) vertex positions
            faces: (F, 3) face indices

        Returns:
            List of detected relationships
        """
        relationships = []

        # Compute face normals
        normals = []
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal = normal / norm
            normals.append(normal)
        normals = np.array(normals)

        # Check pairs (limit to avoid O(n^2) for large meshes)
        n_faces = len(faces)
        max_check = min(n_faces, self.max_face_pairs)

        for i in range(max_check):
            for j in range(i + 1, max_check):
                dot = abs(np.dot(normals[i], normals[j]))

                # Parallel: dot product ~= 1
                if abs(dot - 1.0) < self.angle_tolerance:
                    relationships.append(GeometricRelationship(
                        relationship_type="parallel",
                        entity_a=i,
                        entity_b=j,
                        confidence=1.0 - abs(dot - 1.0),
                    ))

                # Perpendicular: dot product ~= 0
                elif dot < self.angle_tolerance:
                    relationships.append(GeometricRelationship(
                        relationship_type="perpendicular",
                        entity_a=i,
                        entity_b=j,
                        confidence=1.0 - dot,
                    ))

        return relationships

    def verify_relationships(
        self,
        original_relationships: list[GeometricRelationship],
        transformed_vertices: np.ndarray,
        transformed_faces: np.ndarray,
    ) -> Optional[float]:
        """Verify that relationships are preserved after transformation.

        Args:
            original_relationships: Relationships detected before transform.
                Must be non-empty for meaningful verification.
            transformed_vertices: Vertices after transformation.
            transformed_faces: Faces (should be same indices).

        Returns:
            Preservation rate [0, 1], or None if verification cannot be
            performed (e.g., empty input). Callers must handle None.
        """
        if not original_relationships:
            _log.warning(
                "verify_relationships called with empty baseline; "
                "cannot compute preservation rate"
            )
            return None

        new_relationships = self.detect_face_relationships(
            transformed_vertices, transformed_faces
        )
        new_set = {(r.relationship_type, r.entity_a, r.entity_b) for r in new_relationships}

        preserved = 0
        for rel in original_relationships:
            key = (rel.relationship_type, rel.entity_a, rel.entity_b)
            if key in new_set:
                preserved += 1

        return preserved / len(original_relationships)
