"""Point-cloud segmentation for ll_clouds.

- ``ransac_plane``: robustly fit the dominant plane and label inliers.
- ``euclidean_cluster``: density-based (DBSCAN-style) clustering with noise.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .datamodel import PointCloud, SegmentationResult


def _plane_from_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    """Plane (a, b, c, d) with unit normal through three points, or None if
    the points are collinear/degenerate."""
    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    normal = normal / norm
    d = -float(normal @ p0)
    return np.array([normal[0], normal[1], normal[2], d])


def _best_fit_plane(points: np.ndarray) -> np.ndarray:
    """Total-least-squares plane through a set of points (SVD)."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    normal = vt[-1]
    d = -float(normal @ centroid)
    return np.array([normal[0], normal[1], normal[2], d])


def _point_plane_distance(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    return np.asarray(np.abs(points @ plane[:3] + plane[3]))


def ransac_plane(
    pc: PointCloud,
    distance_threshold: float = 0.05,
    num_iterations: int = 200,
    seed: int = 0,
    min_inliers: int = 3,
) -> tuple[SegmentationResult, np.ndarray]:
    """RANSAC plane segmentation.

    Returns a SegmentationResult whose labels are ``1`` for plane inliers and
    ``0`` for the rest, plus the refined plane coefficients ``(a, b, c, d)``.

    A plane is only reported when its inlier set is **larger than**
    ``min_inliers`` — the 3 sampled points always lie on their own candidate
    plane, so a genuine consensus must exceed that minimal sample. Clouds with
    fewer than 3 points, or with no consensus beyond the sample, yield
    ``num_segments == 0`` and ``num_inliers == 0`` (labels all 0, default plane).
    """
    points = pc.points
    n = points.shape[0]
    default_plane = np.array([0.0, 0.0, 1.0, 0.0])

    def _no_plane() -> tuple[SegmentationResult, np.ndarray]:
        return (
            SegmentationResult(
                labels=np.zeros(n, dtype=np.int64),
                num_segments=0,
                metadata={"plane": default_plane.tolist(), "num_inliers": 0},
            ),
            default_plane,
        )

    if n < 3:
        return _no_plane()

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = -1
    best_plane = default_plane

    for _ in range(num_iterations):
        i, j, k = rng.choice(n, size=3, replace=False)
        plane = _plane_from_points(points[i], points[j], points[k])
        if plane is None:
            continue
        inliers = _point_plane_distance(points, plane) <= distance_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_plane = plane

    # Refine the plane on its inliers and re-evaluate membership.
    if best_inliers.sum() >= 3:
        refined = _best_fit_plane(points[best_inliers])
        refined_inliers = _point_plane_distance(points, refined) <= distance_threshold
        if int(refined_inliers.sum()) >= best_count:
            best_plane = refined
            best_inliers = refined_inliers

    if int(best_inliers.sum()) <= min_inliers:
        return _no_plane()

    labels = best_inliers.astype(np.int64)  # 1 = inlier, 0 = outlier
    result = SegmentationResult(
        labels=labels,
        num_segments=1,
        metadata={"plane": best_plane.tolist(), "num_inliers": int(best_inliers.sum())},
    )
    return result, best_plane


def euclidean_cluster(
    pc: PointCloud, eps: float = 0.5, min_points: int = 10
) -> SegmentationResult:
    """Density-based (DBSCAN) Euclidean clustering.

    Points are labelled by cluster id (``0..K-1``); isolated points that never
    join a dense region are labelled ``-1`` (noise).
    """
    points = pc.points
    n = points.shape[0]
    tree = cKDTree(points)
    neighbours = tree.query_ball_point(points, eps)

    labels = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbours[i]) < min_points:
            continue  # leave as noise for now (may be absorbed by a cluster)

        labels[i] = cluster_id
        seeds = list(neighbours[i])
        s = 0
        while s < len(seeds):
            q = seeds[s]
            s += 1
            if not visited[q]:
                visited[q] = True
                if len(neighbours[q]) >= min_points:
                    seeds.extend(neighbours[q])
            if labels[q] == -1:
                labels[q] = cluster_id
        cluster_id += 1

    return SegmentationResult(labels=labels, num_segments=cluster_id)
