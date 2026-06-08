"""Per-point features and geometry statistics for ll_clouds.

Normals and curvature are estimated from local k-nearest-neighbour PCA: for each
point the covariance of its neighbourhood is eigendecomposed; the eigenvector of
the smallest eigenvalue is the surface normal, and the normalized smallest
eigenvalue (surface variation) is a curvature proxy bounded in ``[0, 1/3]``.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

from .datamodel import PointCloud


def _neighbour_covariances(points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ascending eigenvalues [N,3], eigenvectors [N,3,3]) of each point's
    k-NN covariance."""
    n = points.shape[0]
    k = min(k, n)
    tree = cKDTree(points)
    _, nbr_idx = tree.query(points, k=k)
    nbr_idx = np.atleast_2d(nbr_idx)
    neighbours = points[nbr_idx]  # [N, k, 3]
    centered = neighbours - neighbours.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k  # [N, 3, 3]
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    return eigvals, eigvecs


def estimate_normals(
    pc: PointCloud, k: int = 16, as_cloud: bool = False
) -> Union[np.ndarray, PointCloud]:
    """Estimate unit surface normals via k-NN PCA.

    Args:
        pc: input point cloud.
        k: neighbourhood size.
        as_cloud: if True, return a PointCloud with the normals attached;
            otherwise return the ``[N, 3]`` normals array.
    """
    _, eigvecs = _neighbour_covariances(pc.points, k)
    normals = eigvecs[:, :, 0]  # eigenvector of the smallest eigenvalue
    # eigh returns orthonormal vectors; guard against any numerical drift.
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 0)

    if as_cloud:
        return PointCloud(
            points=pc.points,
            normals=normals,
            colors=pc.colors,
            labels=pc.labels,
            metadata=dict(pc.metadata),
        )
    return normals


def estimate_curvature(pc: PointCloud, k: int = 16) -> np.ndarray:
    """Estimate per-point surface variation (curvature proxy) in ``[0, 1/3]``."""
    eigvals, _ = _neighbour_covariances(pc.points, k)
    total = eigvals.sum(axis=1)
    return eigvals[:, 0] / (total + 1e-12)


def bounding_box(pc: PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box as ``(min_xyz, max_xyz)``."""
    return pc.points.min(axis=0), pc.points.max(axis=0)


def centroid(pc: PointCloud) -> np.ndarray:
    """Mean position of the cloud."""
    return pc.points.mean(axis=0)


def extent(pc: PointCloud) -> np.ndarray:
    """Per-axis size (max - min) of the bounding box."""
    bbox_min, bbox_max = bounding_box(pc)
    return bbox_max - bbox_min
