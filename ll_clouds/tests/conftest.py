"""Shared fixtures for the ll_clouds test suite.

OpenMP safety: ``OMP_NUM_THREADS`` is pinned to 1. The core library is
NumPy/SciPy only (no torch), so there is no libomp double-init risk here, but we
keep the pin for consistency with the rest of the monorepo.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministic NumPy random generator."""
    return np.random.default_rng(0)


@pytest.fixture
def plane_points(rng) -> np.ndarray:
    """A noisy planar point cloud in the z=0 plane: [N, 3]."""
    n = 500
    xy = rng.uniform(-1.0, 1.0, size=(n, 2))
    z = rng.normal(0.0, 1e-3, size=(n, 1))  # near-zero with tiny noise
    return np.concatenate([xy, z], axis=1).astype(np.float64)


@pytest.fixture
def sphere_points(rng) -> np.ndarray:
    """Points sampled on a unit sphere (radius 1): [N, 3]."""
    n = 2000
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype(np.float64)


@pytest.fixture
def two_blobs(rng) -> np.ndarray:
    """Two well-separated Gaussian blobs: [N, 3]."""
    a = rng.normal(loc=(0.0, 0.0, 0.0), scale=0.1, size=(200, 3))
    b = rng.normal(loc=(5.0, 0.0, 0.0), scale=0.1, size=(200, 3))
    return np.concatenate([a, b], axis=0).astype(np.float64)
