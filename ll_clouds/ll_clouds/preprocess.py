"""Point-cloud preprocessing for ll_clouds.

Normalization, voxel-grid and farthest-point downsampling, and statistical
outlier removal. All operations return a new :class:`PointCloud` and preserve
per-point attributes where the operation makes that meaningful.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .datamodel import PointCloud


def _subset(pc: PointCloud, idx: np.ndarray) -> PointCloud:
    """Index/mask a PointCloud, carrying all per-point attributes along."""
    return PointCloud(
        points=pc.points[idx],
        normals=None if pc.normals is None else pc.normals[idx],
        colors=None if pc.colors is None else pc.colors[idx],
        labels=None if pc.labels is None else pc.labels[idx],
        metadata=dict(pc.metadata),
    )


def normalize(pc: PointCloud) -> PointCloud:
    """Center on the centroid and scale to the unit sphere (max radius 1).

    Idempotent: re-normalizing an already-normalized cloud is a no-op. Normals
    are directions, so they are passed through unchanged. An empty cloud is
    returned unchanged (avoids NaNs from mean/max over zero points).
    """
    if pc.num_points == 0:
        return _subset(pc, np.arange(0))
    points = pc.points
    centroid = points.mean(axis=0)
    centered = points - centroid
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale > 0.0:
        centered = centered / scale
    return PointCloud(
        points=centered,
        normals=pc.normals,
        colors=pc.colors,
        labels=pc.labels,
        metadata=dict(pc.metadata),
    )


def voxel_downsample(pc: PointCloud, voxel_size: float) -> PointCloud:
    """Keep one centroid point per occupied voxel of side ``voxel_size``.

    Normals/colors are averaged per voxel (normals re-normalized); labels are
    dropped since they cannot be meaningfully aggregated.
    """
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")

    keys = np.floor(pc.points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n_groups = int(inverse.max()) + 1

    def _group_mean(values: np.ndarray) -> np.ndarray:
        sums = np.zeros((n_groups, values.shape[1]), dtype=np.float64)
        counts = np.zeros(n_groups, dtype=np.float64)
        np.add.at(sums, inverse, values)
        np.add.at(counts, inverse, 1.0)
        return sums / counts[:, None]

    points = _group_mean(pc.points)
    normals = None
    if pc.normals is not None:
        normals = _group_mean(pc.normals)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 0)
    colors = _group_mean(pc.colors) if pc.colors is not None else None

    return PointCloud(
        points=points, normals=normals, colors=colors, metadata=dict(pc.metadata)
    )


def _farthest_point_indices(points: np.ndarray, k: int) -> np.ndarray:
    """Indices of ``k`` farthest-point samples (deterministic start at point 0).

    Interface-compatible with the FPS used in ll_ocadr's GeometryNet.
    """
    n = points.shape[0]
    if k >= n:
        return np.arange(n)

    selected = np.empty(k, dtype=np.int64)
    distances = np.full(n, np.inf)
    farthest = 0
    for i in range(k):
        selected[i] = farthest
        diff = points - points[farthest]
        dist = np.einsum("ij,ij->i", diff, diff)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))
    return selected


def farthest_point_downsample(pc: PointCloud, k: int) -> PointCloud:
    """Downsample to ``k`` well-spread points via farthest-point sampling.

    Returns the whole cloud unchanged when ``k >= num_points``.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    idx = _farthest_point_indices(pc.points, k)
    return _subset(pc, idx)


def remove_statistical_outliers(
    pc: PointCloud, k: int = 16, std_ratio: float = 2.0
) -> PointCloud:
    """Remove points whose mean distance to their ``k`` nearest neighbours is
    more than ``std_ratio`` standard deviations above the global mean.
    """
    n = pc.num_points
    if n <= k:
        return _subset(pc, np.arange(n))

    tree = cKDTree(pc.points)
    # query k+1 because the nearest neighbour of a point is itself (dist 0).
    dists, _ = tree.query(pc.points, k=k + 1)
    mean_neighbor_dist = dists[:, 1:].mean(axis=1)

    threshold = mean_neighbor_dist.mean() + std_ratio * mean_neighbor_dist.std()
    mask = mean_neighbor_dist <= threshold
    return _subset(pc, mask)
