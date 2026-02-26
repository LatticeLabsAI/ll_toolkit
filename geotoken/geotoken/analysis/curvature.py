"""Curvature analysis for adaptive bit allocation.

Computes per-vertex curvature using discrete differential geometry.
For meshes, uses the discrete Laplace-Beltrami operator.
For point clouds, uses local PCA-based curvature estimation with KDTree.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

_log = logging.getLogger(__name__)


@dataclass
class CurvatureResult:
    """Per-vertex curvature analysis results."""
    mean_curvature: np.ndarray = field(default_factory=lambda: np.array([]))          # H per vertex (N,)
    gaussian_curvature: np.ndarray = field(default_factory=lambda: np.array([]))      # K per vertex (N,)
    combined_magnitude: np.ndarray = field(default_factory=lambda: np.array([]))      # sqrt(H^2 + |K|) per vertex (N,)
    min_value: float = 0.0
    max_value: float = 0.0
    mean_value: float = 0.0


class CurvatureAnalyzer:
    """Computes discrete curvature on meshes or point clouds."""

    def __init__(self, n_neighbors: int = 12):
        """Initialize curvature analyzer.

        Args:
            n_neighbors: Number of neighbors for point cloud PCA
        """
        self.n_neighbors = n_neighbors

    def analyze_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> CurvatureResult:
        """Compute per-vertex curvature on a triangle mesh.

        Uses discrete Laplace-Beltrami operator for mean curvature
        and angle defect for Gaussian curvature.

        Args:
            vertices: (N, 3) vertex positions
            faces: (F, 3) triangle face indices

        Returns:
            CurvatureResult with per-vertex curvatures

        Raises:
            TypeError: If vertices or faces are not numpy arrays.
            ValueError: If arrays have incorrect shapes.
        """
        if not isinstance(vertices, np.ndarray):
            raise TypeError(
                f"vertices must be numpy array, got {type(vertices).__name__}"
            )
        if not isinstance(faces, np.ndarray):
            raise TypeError(
                f"faces must be numpy array, got {type(faces).__name__}"
            )
        if vertices.ndim != 2 or (len(vertices) > 0 and vertices.shape[1] != 3):
            raise ValueError(
                f"vertices must be (N, 3) array, got shape {vertices.shape}"
            )
        if faces.ndim != 2 or (len(faces) > 0 and faces.shape[1] != 3):
            raise ValueError(
                f"faces must be (F, 3) array, got shape {faces.shape}"
            )

        n_verts = len(vertices)
        mean_curvature = np.zeros(n_verts)
        gaussian_curvature = np.full(n_verts, 2.0 * np.pi)  # Start with 2*pi for angle defect

        # Build vertex adjacency and compute cotangent weights
        # For each face, compute contributions to mean and Gaussian curvature
        vertex_areas = np.zeros(n_verts)
        laplacian = np.zeros((n_verts, 3))

        if len(faces) > 0:
            # Extract all face vertex indices
            fi = faces[:, 0]
            fj = faces[:, 1]
            fk = faces[:, 2]

            # Gather vertex positions for all faces at once: (F, 3)
            vi = vertices[fi]
            vj = vertices[fj]
            vk = vertices[fk]

            # Edge vectors (F, 3)
            eij = vj - vi
            eik = vk - vi
            ejk = vk - vj
            eji = -eij
            eki = -eik
            ekj = -ejk

            # Face areas (F,)
            cross_face = np.cross(eij, eik)
            face_area = 0.5 * np.linalg.norm(cross_face, axis=1)

            # Mask out degenerate faces
            valid = face_area >= 1e-12

            # Cross products at each vertex (F, 3)
            cross_i = cross_face  # eij x eik
            cross_j = np.cross(eji, ejk)
            cross_k = np.cross(eki, ekj)

            # Norms of cross products (F,)
            norm_i = np.linalg.norm(cross_i, axis=1)
            norm_j = np.linalg.norm(cross_j, axis=1)
            norm_k = np.linalg.norm(cross_k, axis=1)

            # Cotangent weights (F,)
            cot_i = np.sum(eij * eik, axis=1) / np.maximum(norm_i, 1e-12)
            cot_j = np.sum(eji * ejk, axis=1) / np.maximum(norm_j, 1e-12)
            cot_k = np.sum(eki * ekj, axis=1) / np.maximum(norm_k, 1e-12)

            # Zero out contributions from degenerate faces
            cot_i[~valid] = 0.0
            cot_j[~valid] = 0.0
            cot_k[~valid] = 0.0
            face_area_valid = face_area.copy()
            face_area_valid[~valid] = 0.0

            # Laplacian contributions (F, 3) for each vertex of each face
            lap_i = cot_k[:, None] * (vj - vi) + cot_j[:, None] * (vk - vi)
            lap_j = cot_i[:, None] * (vk - vj) + cot_k[:, None] * (vi - vj)
            lap_k = cot_j[:, None] * (vi - vk) + cot_i[:, None] * (vj - vk)

            # Accumulate Laplacian per vertex using np.add.at
            np.add.at(laplacian, fi, lap_i)
            np.add.at(laplacian, fj, lap_j)
            np.add.at(laplacian, fk, lap_k)

            # Accumulate vertex areas
            area_third = face_area_valid / 3.0
            np.add.at(vertex_areas, fi, area_third)
            np.add.at(vertex_areas, fj, area_third)
            np.add.at(vertex_areas, fk, area_third)

            # Angle defect for Gaussian curvature
            norm_eij = np.linalg.norm(eij, axis=1)
            norm_eik = np.linalg.norm(eik, axis=1)
            norm_eji = norm_eij  # same length
            norm_ejk = np.linalg.norm(ejk, axis=1)
            norm_eki = norm_eik  # same length
            norm_ekj = norm_ejk  # same length

            cos_i = np.sum(eij * eik, axis=1) / (norm_eij * norm_eik + 1e-12)
            cos_j = np.sum(eji * ejk, axis=1) / (norm_eji * norm_ejk + 1e-12)
            cos_k = np.sum(eki * ekj, axis=1) / (norm_eki * norm_ekj + 1e-12)

            angle_i = np.arccos(np.clip(cos_i, -1.0, 1.0))
            angle_j = np.arccos(np.clip(cos_j, -1.0, 1.0))
            angle_k = np.arccos(np.clip(cos_k, -1.0, 1.0))

            # Zero out degenerate face angles
            angle_i[~valid] = 0.0
            angle_j[~valid] = 0.0
            angle_k[~valid] = 0.0

            np.subtract.at(gaussian_curvature, fi, angle_i)
            np.subtract.at(gaussian_curvature, fj, angle_j)
            np.subtract.at(gaussian_curvature, fk, angle_k)

        # Normalize by area
        safe_areas = np.maximum(vertex_areas, 1e-12)

        # Mean curvature = ||Laplacian|| / (2 * area)
        laplacian_magnitude = np.linalg.norm(laplacian, axis=1)
        mean_curvature = laplacian_magnitude / (2.0 * safe_areas)

        # Gaussian curvature = angle_defect / area
        gaussian_curvature = gaussian_curvature / safe_areas

        # Isolated vertices (no faces) have zero area and invalid curvature — zero them out
        isolated = vertex_areas == 0.0
        mean_curvature[isolated] = 0.0
        gaussian_curvature[isolated] = 0.0

        # Combined magnitude
        combined = np.sqrt(mean_curvature**2 + np.abs(gaussian_curvature))

        return CurvatureResult(
            mean_curvature=mean_curvature,
            gaussian_curvature=gaussian_curvature,
            combined_magnitude=combined,
            min_value=float(np.min(combined)),
            max_value=float(np.max(combined)),
            mean_value=float(np.mean(combined)),
        )

    def analyze_point_cloud(self, points: np.ndarray) -> CurvatureResult:
        """Estimate curvature from point cloud using local PCA with KDTree.

        Uses scipy.spatial.cKDTree for O(n log n) neighbor queries instead
        of O(n²) brute-force search.

        Args:
            points: (N, 3) point positions

        Returns:
            CurvatureResult with estimated curvatures

        Raises:
            TypeError: If points is not a numpy array.
            ValueError: If points is not a 2D array with 3 columns.
        """
        if not isinstance(points, np.ndarray):
            raise TypeError(
                f"points must be numpy array, got {type(points).__name__}"
            )
        if points.ndim != 2 or (len(points) > 0 and points.shape[1] != 3):
            raise ValueError(
                f"points must be (N, 3) array, got shape {points.shape}"
            )

        n_points = len(points)
        mean_curvature = np.zeros(n_points)
        gaussian_curvature = np.zeros(n_points)

        if n_points == 0:
            return CurvatureResult(
                mean_curvature=mean_curvature,
                gaussian_curvature=gaussian_curvature,
                combined_magnitude=np.zeros(0),
                min_value=0.0,
                max_value=0.0,
                mean_value=0.0,
            )

        # Build KD-tree for O(log n) neighbor queries
        tree = cKDTree(points)

        # Query k+1 neighbors (includes self) for all points at once
        k = min(self.n_neighbors + 1, n_points)
        distances, indices = tree.query(points, k=k)

        if k > 1:
            # Build (N, k-1, 3) neighbor coordinate array (skip self at index 0)
            neighbor_coords = points[indices[:, 1:]]  # (N, k-1, 3)
            centered = neighbor_coords - points[:, np.newaxis, :]  # (N, k-1, 3)

            # Build (N, 3, 3) covariance matrix stack
            # cov[i] = centered[i].T @ centered[i] / (k-1)
            n_neighbors_actual = centered.shape[1]
            cov_stack = np.einsum('nki,nkj->nij', centered, centered) / n_neighbors_actual

            # Compute eigenvalues for all points at once: (N, 3)
            all_eigenvalues = np.linalg.eigvalsh(cov_stack)
            all_eigenvalues = np.sort(all_eigenvalues, axis=1)

            # Mask: only update points that had >= 3 neighbors
            valid = n_neighbors_actual >= 3
            if valid:
                total = np.sum(all_eigenvalues, axis=1) + 1e-12
                # Smallest eigenvalue ratio indicates surface variation
                mean_curvature[:] = all_eigenvalues[:, 0] / total
                gaussian_curvature[:] = (all_eigenvalues[:, 0] * all_eigenvalues[:, 1]) / (total**2)

        combined = np.sqrt(mean_curvature**2 + np.abs(gaussian_curvature))

        return CurvatureResult(
            mean_curvature=mean_curvature,
            gaussian_curvature=gaussian_curvature,
            combined_magnitude=combined,
            min_value=float(np.min(combined)) if n_points > 0 else 0.0,
            max_value=float(np.max(combined)) if n_points > 0 else 0.0,
            mean_value=float(np.mean(combined)) if n_points > 0 else 0.0,
        )
