"""Feature density analysis for adaptive bit allocation.

Measures local geometric complexity based on edge length variance
and face area variance to identify regions requiring higher precision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

_log = logging.getLogger(__name__)


@dataclass
class FeatureDensityResult:
    """Per-vertex feature density results."""
    density: np.ndarray = field(default_factory=lambda: np.array([]))              # Density score per vertex (N,)
    edge_length_variance: np.ndarray = field(default_factory=lambda: np.array([]))  # Local edge length variance (N,)
    face_area_variance: Optional[np.ndarray] = field(default=None)   # Local face area variance (N,), None for point clouds
    min_value: float = 0.0
    max_value: float = 0.0
    mean_value: float = 0.0


class FeatureDensityAnalyzer:
    """Analyzes local feature density on meshes and point clouds."""

    def __init__(self, n_neighbors: int = 12):
        """Initialize feature density analyzer.

        Args:
            n_neighbors: Number of neighbors for point cloud density estimation.
        """
        self.n_neighbors = n_neighbors

    def analyze(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> FeatureDensityResult:
        """Compute per-vertex feature density.

        Feature density is based on local edge length variance and
        face area variance. High variance indicates complex features.

        Args:
            vertices: (N, 3) vertex positions
            faces: (F, 3) triangle face indices

        Returns:
            FeatureDensityResult with per-vertex density scores
        """
        n_verts = len(vertices)
        _log.debug("Analyzing feature density for mesh with %d vertices", n_verts)

        n_faces = len(faces)

        # Compute face areas vectorized
        if n_faces > 0:
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        else:
            face_areas = np.zeros(0)

        # Build vertex-to-face adjacency using vectorized approach
        vert_faces = [[] for _ in range(n_verts)]
        if n_faces > 0:
            face_indices = np.arange(n_faces)
            for col in range(3):
                for fi, vi in zip(face_indices, faces[:, col]):
                    vert_faces[vi].append(fi)

        # Build edge list vectorized and compute per-vertex edge length variance
        edge_length_var = np.zeros(n_verts)
        face_area_var = np.zeros(n_verts)

        if n_faces > 0:
            # Build all edges from faces: 3 edges per face
            edges = np.vstack([
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [0, 2]],
            ])  # (3F, 2)

            # Build vertex neighbor sets from edge array (vectorized)
            from scipy.sparse import csr_matrix
            data = np.ones(len(edges) * 2, dtype=np.int8)
            row = np.concatenate([edges[:, 0], edges[:, 1]])
            col = np.concatenate([edges[:, 1], edges[:, 0]])
            adj = csr_matrix((data, (row, col)), shape=(n_verts, n_verts))
            vert_neighbors = [set(adj[i].indices) for i in range(n_verts)]
        else:
            vert_neighbors = [set() for _ in range(n_verts)]

        for vi in range(n_verts):
            # Edge lengths to neighbors
            neighbors = list(vert_neighbors[vi])
            if len(neighbors) > 1:
                neighbor_verts = vertices[neighbors]
                edge_lengths = np.linalg.norm(neighbor_verts - vertices[vi], axis=1)
                edge_length_var[vi] = np.var(edge_lengths)

            # Face areas around vertex
            adj_faces = vert_faces[vi]
            if len(adj_faces) > 1:
                local_areas = face_areas[adj_faces]
                face_area_var[vi] = np.var(local_areas)

        # Normalize variances to [0, 1]
        elv_max = np.max(edge_length_var) if np.max(edge_length_var) > 0 else 1.0
        fav_max = np.max(face_area_var) if np.max(face_area_var) > 0 else 1.0

        edge_length_var_norm = edge_length_var / elv_max
        face_area_var_norm = face_area_var / fav_max

        # Combined density: weighted average
        density = 0.5 * edge_length_var_norm + 0.5 * face_area_var_norm

        result = FeatureDensityResult(
            density=density,
            edge_length_variance=edge_length_var,
            face_area_variance=face_area_var,
            min_value=float(np.min(density)) if n_verts > 0 else 0.0,
            max_value=float(np.max(density)) if n_verts > 0 else 0.0,
            mean_value=float(np.mean(density)) if n_verts > 0 else 0.0,
        )
        _log.debug(
            "Feature density analysis complete: min=%.4f, max=%.4f, mean=%.4f",
            result.min_value, result.max_value, result.mean_value,
        )
        return result

    def analyze_point_cloud(self, points: np.ndarray) -> FeatureDensityResult:
        """Estimate feature density for point cloud using k-NN.

        Uses inverse of local k-NN distance as density proxy and
        distance variance as feature complexity indicator.

        Args:
            points: (N, 3) point positions

        Returns:
            FeatureDensityResult with per-point density scores
        """
        n_points = len(points)
        _log.debug("Analyzing feature density for point cloud with %d points", n_points)

        if n_points == 0:
            return FeatureDensityResult(
                density=np.zeros(0),
                edge_length_variance=np.zeros(0),
                face_area_variance=None,
                min_value=0.0,
                max_value=0.0,
                mean_value=0.0,
            )

        # Build KD-tree for efficient neighbor queries
        tree = cKDTree(points)

        # Query k+1 neighbors (includes self)
        k = min(self.n_neighbors + 1, n_points)
        distances, _ = tree.query(points, k=k)

        # Skip self (distance 0) and compute density metrics
        if k > 1:
            neighbor_dists = distances[:, 1:]  # Exclude self

            # Density: inverse of mean k-NN distance (higher density = closer neighbors)
            mean_dists = np.mean(neighbor_dists, axis=1)
            # Avoid division by zero
            density = 1.0 / (mean_dists + 1e-12)

            # Edge length variance: variance of distances to neighbors
            edge_length_var = np.var(neighbor_dists, axis=1)
        else:
            density = np.ones(n_points)
            edge_length_var = np.zeros(n_points)

        # Normalize density to [0, 1]
        d_max = np.max(density) if np.max(density) > 0 else 1.0
        density_norm = density / d_max

        # Normalize edge variance
        elv_max = np.max(edge_length_var) if np.max(edge_length_var) > 0 else 1.0
        edge_var_norm = edge_length_var / elv_max

        # Combined score: weighted average of normalized metrics
        combined = 0.5 * density_norm + 0.5 * edge_var_norm

        return FeatureDensityResult(
            density=combined,
            edge_length_variance=edge_length_var,
            face_area_variance=None,  # No faces for point clouds
            min_value=float(np.min(combined)),
            max_value=float(np.max(combined)),
            mean_value=float(np.mean(combined)),
        )
