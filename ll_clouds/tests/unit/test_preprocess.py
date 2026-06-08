"""Tests for ll_clouds preprocessing (SPEC-1 M5, T5.3)."""

from __future__ import annotations

import numpy as np
import pytest

from ll_clouds.datamodel import PointCloud
from ll_clouds.preprocess import (
    farthest_point_downsample,
    normalize,
    remove_statistical_outliers,
    voxel_downsample,
)


class TestNormalize:
    def test_centers_and_unit_scales(self, rng) -> None:
        pts = rng.normal(loc=(10.0, -5.0, 3.0), scale=4.0, size=(500, 3))
        pc = normalize(PointCloud(points=pts))
        # Centroid at origin.
        np.testing.assert_allclose(pc.points.mean(axis=0), 0.0, atol=1e-9)
        # Max distance from origin is 1 (unit-sphere normalization).
        assert np.isclose(np.linalg.norm(pc.points, axis=1).max(), 1.0)

    def test_is_idempotent(self, rng) -> None:
        pc = PointCloud(points=rng.normal(size=(200, 3)))
        once = normalize(pc)
        twice = normalize(once)
        np.testing.assert_allclose(once.points, twice.points, atol=1e-9)

    def test_preserves_normals(self, rng) -> None:
        pts = rng.normal(size=(50, 3))
        normals = rng.normal(size=(50, 3))
        pc = normalize(PointCloud(points=pts, normals=normals))
        np.testing.assert_allclose(pc.normals, normals)


class TestVoxelDownsample:
    def test_reduces_count_and_snaps_to_grid(self, rng) -> None:
        # Dense cloud in a unit cube; voxel size 0.5 => at most 2^3 = 8 cells.
        pts = rng.uniform(0.0, 1.0, size=(5000, 3))
        out = voxel_downsample(PointCloud(points=pts), voxel_size=0.5)
        assert out.num_points < 5000
        assert out.num_points <= 8

    def test_single_point_per_occupied_voxel(self) -> None:
        # Two points in the same voxel collapse to one; a far point stays.
        pts = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [9.0, 9.0, 9.0]])
        out = voxel_downsample(PointCloud(points=pts), voxel_size=1.0)
        assert out.num_points == 2


class TestFarthestPointDownsample:
    def test_returns_exactly_k(self, rng) -> None:
        pts = rng.normal(size=(1000, 3))
        out = farthest_point_downsample(PointCloud(points=pts), k=64)
        assert out.num_points == 64

    def test_k_larger_than_n_returns_all(self, rng) -> None:
        pts = rng.normal(size=(10, 3))
        out = farthest_point_downsample(PointCloud(points=pts), k=50)
        assert out.num_points == 10

    def test_points_are_well_spread(self) -> None:
        # FPS over a line should pick the extremes first.
        pts = np.stack([np.linspace(0, 10, 100), np.zeros(100), np.zeros(100)], axis=1)
        out = farthest_point_downsample(PointCloud(points=pts), k=2)
        xs = np.sort(out.points[:, 0])
        assert xs[0] < 1.0 and xs[1] > 9.0


class TestOutlierRemoval:
    def test_drops_planted_outliers(self, rng) -> None:
        inliers = rng.normal(scale=0.05, size=(500, 3))
        outliers = np.array([[10.0, 10.0, 10.0], [-12.0, 0.0, 3.0]])
        pts = np.concatenate([inliers, outliers], axis=0)
        out = remove_statistical_outliers(PointCloud(points=pts), k=16, std_ratio=2.0)
        assert out.num_points <= 500
        # No surviving point should be near the planted outlier locations.
        assert np.all(np.linalg.norm(out.points, axis=1) < 5.0)

    def test_preserves_labels_alignment(self, rng) -> None:
        inliers = rng.normal(scale=0.05, size=(100, 3))
        outlier = np.array([[20.0, 20.0, 20.0]])
        pts = np.concatenate([inliers, outlier], axis=0)
        labels = np.arange(101)
        out = remove_statistical_outliers(
            PointCloud(points=pts, labels=labels), k=8, std_ratio=2.0
        )
        # Labels stay aligned with surviving points.
        assert out.labels is not None
        assert out.labels.shape[0] == out.num_points
        assert 100 not in out.labels  # the outlier (label 100) was dropped


class TestVoxelDownsampleExtras:
    def test_invalid_voxel_size_raises(self, rng) -> None:
        pc = PointCloud(points=rng.uniform(0.0, 1.0, size=(10, 3)))
        with pytest.raises(ValueError, match="voxel_size must be positive"):
            voxel_downsample(pc, voxel_size=0.0)
        with pytest.raises(ValueError, match="voxel_size must be positive"):
            voxel_downsample(pc, voxel_size=-0.5)

    def test_attributes_aggregated_and_labels_dropped(self) -> None:
        # Two points in the same voxel: attributes averaged, labels dropped.
        pts = np.array([[0.1, 0.1, 0.1], [0.4, 0.4, 0.4]], dtype=np.float64)
        normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        labels = np.array([1, 2], dtype=np.int64)
        out = voxel_downsample(
            PointCloud(points=pts, normals=normals, colors=colors, labels=labels),
            voxel_size=1.0,
        )
        assert out.num_points == 1
        np.testing.assert_allclose(out.points[0], pts.mean(axis=0))
        assert out.colors is not None
        np.testing.assert_allclose(out.colors[0], colors.mean(axis=0))
        # Normals averaged then re-normalized to unit length.
        expected_normal = normals.mean(axis=0)
        expected_normal /= np.linalg.norm(expected_normal)
        assert out.normals is not None
        np.testing.assert_allclose(out.normals[0], expected_normal)
        assert np.isclose(np.linalg.norm(out.normals[0]), 1.0)
        assert out.labels is None  # labels cannot be aggregated -> dropped


class TestFarthestPointDownsampleExtras:
    def test_invalid_k_raises(self, rng) -> None:
        pc = PointCloud(points=rng.normal(size=(10, 3)))
        with pytest.raises(ValueError, match="k must be positive"):
            farthest_point_downsample(pc, k=0)
        with pytest.raises(ValueError, match="k must be positive"):
            farthest_point_downsample(pc, k=-1)

    def test_attributes_stay_aligned(self) -> None:
        pts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        normals = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
        )
        labels = np.array([10, 20, 30, 40], dtype=np.int64)
        out = farthest_point_downsample(
            PointCloud(points=pts, normals=normals, labels=labels), k=2
        )
        assert out.num_points == 2
        assert out.normals.shape == (2, 3) and out.labels.shape == (2,)
        # Each surviving point keeps its original normal + label.
        for p, nrm, lab in zip(out.points, out.normals, out.labels):
            idx = np.where(np.all(pts == p, axis=1))[0]
            assert idx.size == 1
            i = int(idx[0])
            np.testing.assert_allclose(normals[i], nrm)
            assert labels[i] == lab


class TestNormalizeEmpty:
    def test_empty_cloud_no_nan(self) -> None:
        out = normalize(PointCloud(points=np.zeros((0, 3))))
        assert out.num_points == 0
        assert not np.isnan(out.points).any()
