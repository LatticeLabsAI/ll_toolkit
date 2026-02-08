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
    mean_curvature: np.ndarray          # H per vertex (N,)
    gaussian_curvature: np.ndarray      # K per vertex (N,)
    combined_magnitude: np.ndarray      # sqrt(H^2 + |K|) per vertex (N,)
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

        for face in faces:
            i, j, k = face
            vi, vj, vk = vertices[i], vertices[j], vertices[k]

            # Edge vectors
            eij = vj - vi
            eik = vk - vi
            ejk = vk - vj
            eji = -eij
            eki = -eik
            ekj = -ejk

            # Face area (for mixed area computation)
            face_area = 0.5 * np.linalg.norm(np.cross(eij, eik))
            if face_area < 1e-12:
                continue

            # Cotangent weights for Laplace-Beltrami
            # cot(angle at i) = dot(eij, eik) / ||eij x eik||
            cross_i = np.cross(eij, eik)
            cross_j = np.cross(eji, ejk)
            cross_k = np.cross(eki, ekj)

            norm_i = np.linalg.norm(cross_i)
            norm_j = np.linalg.norm(cross_j)
            norm_k = np.linalg.norm(cross_k)

            cot_i = np.dot(eij, eik) / max(norm_i, 1e-12)
            cot_j = np.dot(eji, ejk) / max(norm_j, 1e-12)
            cot_k = np.dot(eki, ekj) / max(norm_k, 1e-12)

            # Accumulate Laplacian (cotangent-weighted)
            laplacian[i] += cot_k * (vj - vi) + cot_j * (vk - vi)
            laplacian[j] += cot_i * (vk - vj) + cot_k * (vi - vj)
            laplacian[k] += cot_j * (vi - vk) + cot_i * (vj - vk)

            # Mixed area (Voronoi area per vertex)
            vertex_areas[i] += face_area / 3.0
            vertex_areas[j] += face_area / 3.0
            vertex_areas[k] += face_area / 3.0

            # Angle defect for Gaussian curvature
            angle_i = np.arccos(np.clip(
                np.dot(eij, eik) / (np.linalg.norm(eij) * np.linalg.norm(eik) + 1e-12),
                -1.0, 1.0
            ))
            angle_j = np.arccos(np.clip(
                np.dot(eji, ejk) / (np.linalg.norm(eji) * np.linalg.norm(ejk) + 1e-12),
                -1.0, 1.0
            ))
            angle_k = np.arccos(np.clip(
                np.dot(eki, ekj) / (np.linalg.norm(eki) * np.linalg.norm(ekj) + 1e-12),
                -1.0, 1.0
            ))

            gaussian_curvature[i] -= angle_i
            gaussian_curvature[j] -= angle_j
            gaussian_curvature[k] -= angle_k

        # Normalize by area
        safe_areas = np.maximum(vertex_areas, 1e-12)

        # Mean curvature = ||Laplacian|| / (2 * area)
        laplacian_magnitude = np.linalg.norm(laplacian, axis=1)
        mean_curvature = laplacian_magnitude / (2.0 * safe_areas)

        # Gaussian curvature = angle_defect / area
        gaussian_curvature = gaussian_curvature / safe_areas

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

        for i in range(n_points):
            # Skip self (first neighbor)
            neighbor_idx = indices[i, 1:] if k > 1 else []

            if len(neighbor_idx) < 3:
                continue

            neighbors = points[neighbor_idx]
            centered = neighbors - points[i]

            # PCA on local neighborhood
            cov = centered.T @ centered / len(centered)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)

            # Curvature from eigenvalue ratios
            total = np.sum(eigenvalues) + 1e-12
            # Smallest eigenvalue ratio indicates surface variation
            mean_curvature[i] = eigenvalues[0] / total
            gaussian_curvature[i] = (eigenvalues[0] * eigenvalues[1]) / (total**2)

        combined = np.sqrt(mean_curvature**2 + np.abs(gaussian_curvature))

        return CurvatureResult(
            mean_curvature=mean_curvature,
            gaussian_curvature=gaussian_curvature,
            combined_magnitude=combined,
            min_value=float(np.min(combined)) if n_points > 0 else 0.0,
            max_value=float(np.max(combined)) if n_points > 0 else 0.0,
            mean_value=float(np.mean(combined)) if n_points > 0 else 0.0,
        )
