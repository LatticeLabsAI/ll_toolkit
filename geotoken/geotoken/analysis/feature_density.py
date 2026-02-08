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
    density: np.ndarray              # Density score per vertex (N,)
    edge_length_variance: np.ndarray # Local edge length variance (N,)
    face_area_variance: np.ndarray   # Local face area variance (N,)
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

        # Build vertex-to-face adjacency
        vert_faces = [[] for _ in range(n_verts)]
        for fi, face in enumerate(faces):
            for vi in face:
                vert_faces[vi].append(fi)

        # Build vertex-to-vertex adjacency (edges)
        vert_neighbors = [set() for _ in range(n_verts)]
        for face in faces:
            i, j, k = face
            vert_neighbors[i].update([j, k])
            vert_neighbors[j].update([i, k])
            vert_neighbors[k].update([i, j])

        # Compute face areas
        face_areas = np.zeros(len(faces))
        for fi, face in enumerate(faces):
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            face_areas[fi] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        # Per-vertex edge length variance and face area variance
        edge_length_var = np.zeros(n_verts)
        face_area_var = np.zeros(n_verts)

        for vi in range(n_verts):
            # Edge lengths to neighbors
            neighbors = list(vert_neighbors[vi])
            if len(neighbors) > 1:
                edge_lengths = np.array([
                    np.linalg.norm(vertices[ni] - vertices[vi])
                    for ni in neighbors
                ])
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

        return FeatureDensityResult(
            density=density,
            edge_length_variance=edge_length_var,
            face_area_variance=face_area_var,
            min_value=float(np.min(density)) if n_verts > 0 else 0.0,
            max_value=float(np.max(density)) if n_verts > 0 else 0.0,
            mean_value=float(np.mean(density)) if n_verts > 0 else 0.0,
        )

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

        if n_points == 0:
            return FeatureDensityResult(
                density=np.zeros(0),
                edge_length_variance=np.zeros(0),
                face_area_variance=np.zeros(0),
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
            face_area_variance=np.zeros(n_points),  # No faces for point clouds
            min_value=float(np.min(combined)),
            max_value=float(np.max(combined)),
            mean_value=float(np.mean(combined)),
        )
