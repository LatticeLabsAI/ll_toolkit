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
    confidence: float = field(default=1.0)  # Detection confidence [0, 1]
    metadata: Optional[dict] = field(default=None)  # Extra relationship metadata

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
        max_faces: int = 100,
    ):
        """Initialize relationship detector.

        Args:
            angle_tolerance: Tolerance for angle comparisons (radians).
            distance_tolerance: Tolerance for distance comparisons.
            max_faces: Maximum number of faces to sample for pairwise checks
                to avoid O(n²). Faces are randomly sampled with a fixed seed
                for reproducibility.
        """
        self.angle_tolerance = angle_tolerance
        self.distance_tolerance = distance_tolerance
        self.max_faces = max_faces

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

        # Sample faces BEFORE computing normals to avoid unnecessary work
        n_faces = len(faces)
        if n_faces > self.max_faces:
            rng = np.random.RandomState(42)
            sampled_indices = np.sort(rng.choice(n_faces, size=self.max_faces, replace=False))
        else:
            sampled_indices = np.arange(n_faces)

        # Compute face normals only for sampled faces
        sampled_faces = faces[sampled_indices]
        v0 = vertices[sampled_faces[:, 0]]
        v1 = vertices[sampled_faces[:, 1]]
        v2 = vertices[sampled_faces[:, 2]]
        raw_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(raw_normals, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        normals = raw_normals / norms

        for si, i in enumerate(sampled_indices):
            for sj, j in enumerate(sampled_indices):
                if j <= i:
                    continue
                dot = abs(np.dot(normals[si], normals[sj]))

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
    ) -> float:
        """Verify that relationships are preserved after transformation.

        Args:
            original_relationships: Relationships detected before transform.
            transformed_vertices: Vertices after transformation.
            transformed_faces: Faces (should be same indices).

        Returns:
            Preservation rate [0, 1]. Returns 1.0 if baseline is empty
            (no relationships to violate).
        """
        if not original_relationships:
            _log.debug(
                "verify_relationships called with empty baseline; "
                "returning 1.0 (no relationships to violate)"
            )
            return 1.0

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
