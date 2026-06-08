"""Tests for ll_clouds segmentation (SPEC-1 M5, T5.6)."""
from __future__ import annotations

import numpy as np
import pytest

from ll_clouds.datamodel import PointCloud, SegmentationResult
from ll_clouds.segmentation import euclidean_cluster, ransac_plane


class TestRansacPlane:
    def test_finds_planted_plane(self, rng) -> None:
        # Plane z=0 with many inliers + a handful of off-plane outliers.
        inliers = np.concatenate(
            [rng.uniform(-1, 1, size=(400, 2)), np.zeros((400, 1))], axis=1
        )
        outliers = rng.uniform(-1, 1, size=(40, 3)) + np.array([0, 0, 5.0])
        pts = np.concatenate([inliers, outliers], axis=0)

        result, coeffs = ransac_plane(
            PointCloud(points=pts), distance_threshold=0.05, num_iterations=200, seed=0
        )
        assert isinstance(result, SegmentationResult)
        # Inlier ratio should reflect the ~400/440 planar points.
        inlier_mask = result.labels == 1
        assert inlier_mask.sum() >= 380

        # The recovered normal must be ±z.
        normal = np.array(coeffs[:3]) / np.linalg.norm(coeffs[:3])
        assert abs(abs(normal[2]) - 1.0) < 0.05

    def test_no_plane_when_threshold_tiny(self, rng) -> None:
        pts = rng.uniform(-1, 1, size=(200, 3))  # volumetric blob, no plane
        result, _ = ransac_plane(
            PointCloud(points=pts), distance_threshold=1e-4, num_iterations=50, seed=0
        )
        # Almost nothing should be an inlier of any single plane.
        assert (result.labels == 1).sum() < 50


class TestEuclideanCluster:
    def test_separates_two_blobs(self, two_blobs) -> None:
        result = euclidean_cluster(
            PointCloud(points=two_blobs), eps=0.5, min_points=10
        )
        assert isinstance(result, SegmentationResult)
        assert result.num_segments == 2
        # Each blob has 200 points; both clusters should be substantial.
        unique, counts = np.unique(result.labels[result.labels >= 0], return_counts=True)
        assert len(unique) == 2
        assert all(c >= 150 for c in counts)

    def test_noise_labeled_minus_one(self, rng) -> None:
        blob = rng.normal(scale=0.05, size=(100, 3))
        far_noise = np.array([[10.0, 10.0, 10.0]])  # isolated -> noise
        pts = np.concatenate([blob, far_noise], axis=0)
        result = euclidean_cluster(PointCloud(points=pts), eps=0.3, min_points=5)
        assert result.labels[-1] == -1
        assert result.num_segments == 1
