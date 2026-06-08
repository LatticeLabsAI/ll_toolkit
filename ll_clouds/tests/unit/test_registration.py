"""Tests for ll_clouds ICP registration (SPEC-1 M5, T5.5)."""

from __future__ import annotations

import numpy as np
import pytest

from ll_clouds.datamodel import PointCloud, RegistrationResult
from ll_clouds.registration import icp


def _rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _transform(points: np.ndarray, rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ rot.T + t


class TestICP:
    def test_recovers_known_transform(self, rng) -> None:
        target = rng.normal(size=(300, 3))
        rot = _rotation_z(0.2)
        t = np.array([0.5, -0.3, 0.1])
        # source maps to target via (rot, t): target ~= source @ rot.T + t
        source = (target - t) @ rot

        result = icp(
            PointCloud(points=source),
            PointCloud(points=target),
            max_iterations=60,
            tolerance=1e-10,
        )
        assert isinstance(result, RegistrationResult)
        assert result.converged is True

        # Applying the recovered transform to source should match target.
        transform = result.transformation
        src_h = np.concatenate([source, np.ones((source.shape[0], 1))], axis=1)
        aligned = (src_h @ transform.T)[:, :3]
        assert np.mean(np.linalg.norm(aligned - target, axis=1)) < 1e-3
        assert result.inlier_rmse < 1e-3

    def test_identity_for_same_cloud(self, rng) -> None:
        pts = rng.normal(size=(200, 3))
        result = icp(PointCloud(points=pts), PointCloud(points=pts), max_iterations=10)
        np.testing.assert_allclose(result.transformation, np.eye(4), atol=1e-6)
        assert result.fitness == pytest.approx(1.0)

    def test_iterations_bounded(self, rng) -> None:
        target = rng.normal(size=(100, 3))
        source = _transform(target, _rotation_z(0.1), np.array([0.2, 0.0, 0.0]))
        result = icp(
            PointCloud(points=source),
            PointCloud(points=target),
            max_iterations=25,
        )
        assert 1 <= result.iterations <= 25


class TestICPFitness:
    def test_max_correspondence_distance_controls_fitness(self, rng) -> None:
        # 1000 inliers near the origin + 3 far outliers (same direction so the
        # rigid fit barely shifts the inliers). Outliers sit ~100 away.
        inliers = rng.normal(scale=0.1, size=(1000, 3))
        outliers = np.array([[100.0, 0.0, 0.0]] * 3)
        target = PointCloud(points=inliers)
        source = PointCloud(points=np.concatenate([inliers, outliers], axis=0))

        loose = icp(source, target, max_iterations=20, max_correspondence_distance=1e9)
        tight = icp(source, target, max_iterations=20, max_correspondence_distance=2.0)

        # With an unbounded gate every correspondence counts -> fitness 1.0.
        assert loose.fitness == pytest.approx(1.0)
        # A tight gate excludes the 3 far outliers -> fitness drops to ~1000/1003.
        assert tight.fitness < loose.fitness
        assert tight.fitness == pytest.approx(1000 / 1003, abs=0.02)
