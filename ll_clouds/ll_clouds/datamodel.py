"""Pydantic v2 data models for ll_clouds.

``PointCloud`` is the central container; ``RegistrationResult`` and
``SegmentationResult`` are returned by the registration/segmentation modules.
NumPy arrays are allowed via ``arbitrary_types_allowed`` and validated/coerced
so callers can pass plain lists or arrays of any numeric dtype.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class PointCloud(BaseModel):
    """An (optionally attributed) 3D point cloud.

    Attributes:
        points: ``[N, 3]`` float coordinates (required).
        normals: ``[N, 3]`` per-point unit normals (optional).
        colors: ``[N, 3]`` per-point RGB in [0, 1] (optional).
        labels: ``[N]`` per-point integer labels (optional).
        metadata: free-form dictionary of provenance/extra info.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    points: np.ndarray
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    metadata: dict[str, Any] = {}

    @field_validator("points", mode="before")
    @classmethod
    def _coerce_points(cls, value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"points must have shape [N, 3], got {arr.shape}")
        return arr

    @field_validator("normals", "colors", mode="before")
    @classmethod
    def _coerce_n3(cls, value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"expected shape [N, 3], got {arr.shape}")
        return arr

    @field_validator("labels", mode="before")
    @classmethod
    def _coerce_labels(cls, value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim != 1:
            raise ValueError(f"labels must be 1-D [N], got shape {arr.shape}")
        return arr

    @model_validator(mode="after")
    def _check_lengths(self) -> "PointCloud":
        n = self.points.shape[0]
        for name in ("normals", "colors", "labels"):
            attr = getattr(self, name)
            if attr is not None and attr.shape[0] != n:
                raise ValueError(
                    f"{name} length {attr.shape[0]} does not match "
                    f"number of points {n}"
                )
        return self

    @property
    def num_points(self) -> int:
        """Number of points in the cloud."""
        return int(self.points.shape[0])

    def __len__(self) -> int:
        return self.num_points


class RegistrationResult(BaseModel):
    """Result of a point-cloud registration (e.g. ICP).

    Attributes:
        transformation: ``[4, 4]`` homogeneous transform mapping source -> target.
        fitness: fraction of source points with a correspondence (0..1).
        inlier_rmse: RMSE over inlier correspondences.
        iterations: number of iterations performed.
        converged: whether the convergence criterion was met.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    iterations: int
    converged: bool

    @field_validator("transformation", mode="before")
    @classmethod
    def _coerce_transform(cls, value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (4, 4):
            raise ValueError(f"transformation must be [4, 4], got {arr.shape}")
        return arr


class SegmentationResult(BaseModel):
    """Result of a segmentation/clustering.

    Attributes:
        labels: ``[N]`` per-point integer labels (``-1`` = noise/unassigned).
        num_segments: number of distinct non-noise segments.
        metadata: free-form extra info (e.g. plane coefficients).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    labels: np.ndarray
    num_segments: int
    metadata: dict[str, Any] = {}

    @field_validator("labels", mode="before")
    @classmethod
    def _coerce_labels(cls, value: Any) -> np.ndarray:
        arr = np.asarray(value)
        if arr.ndim != 1:
            raise ValueError(f"labels must be 1-D [N], got shape {arr.shape}")
        return arr
