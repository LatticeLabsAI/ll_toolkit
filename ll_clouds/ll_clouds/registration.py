"""Point-cloud registration for ll_clouds — point-to-point ICP.

Iterative Closest Point: at each step, match each (current) source point to its
nearest target point, solve the optimal rigid transform for those
correspondences via SVD (Umeyama without scaling), apply it, and repeat until
the alignment RMSE stops improving.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .datamodel import PointCloud, RegistrationResult


def _best_rigid_transform(
    src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rotation R and translation t minimizing ||R·src + t − dst||."""
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_c = src - src_centroid
    dst_c = dst - dst_centroid

    h = src_c.T @ dst_c
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    rotation = vt.T @ correction @ u.T
    translation = dst_centroid - rotation @ src_centroid
    return rotation, translation


def icp(
    source: PointCloud,
    target: PointCloud,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
    max_correspondence_distance: float = np.inf,
) -> RegistrationResult:
    """Align ``source`` to ``target`` with point-to-point ICP.

    Args:
        source: cloud to be transformed.
        target: reference cloud.
        max_iterations: hard cap on iterations.
        tolerance: stop when the RMSE improvement between iterations drops below
            this value.
        max_correspondence_distance: correspondences farther than this are
            excluded from the reported ``fitness``.

    Returns:
        RegistrationResult with the cumulative source->target transform.
    """
    tree = cKDTree(target.points)
    current = source.points.copy()
    transform = np.eye(4)

    prev_rmse = np.inf
    converged = False
    iterations = 0

    for i in range(max_iterations):
        iterations = i + 1
        distances, indices = tree.query(current)
        matched = target.points[indices]

        rotation, translation = _best_rigid_transform(current, matched)
        current = current @ rotation.T + translation

        step = np.eye(4)
        step[:3, :3] = rotation
        step[:3, 3] = translation
        transform = step @ transform

        rmse = float(np.sqrt(np.mean(distances**2)))
        if abs(prev_rmse - rmse) < tolerance:
            converged = True
            break
        prev_rmse = rmse

    final_distances, _ = tree.query(current)
    # inlier_rmse is the RMSE over INLIER correspondences only (those within
    # max_correspondence_distance) — matching the field docstring and the
    # Open3D convention — not over all correspondences.
    inlier_mask = final_distances <= max_correspondence_distance
    fitness = float(np.mean(inlier_mask))
    if np.any(inlier_mask):
        inlier_rmse = float(np.sqrt(np.mean(final_distances[inlier_mask] ** 2)))
    else:
        inlier_rmse = 0.0

    return RegistrationResult(
        transformation=transform,
        fitness=fitness,
        inlier_rmse=inlier_rmse,
        iterations=iterations,
        converged=converged,
    )
