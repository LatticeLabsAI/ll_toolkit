"""Tests for ll_clouds features (SPEC-1 M5, T5.4)."""

from __future__ import annotations

import numpy as np

from ll_clouds.datamodel import PointCloud
from ll_clouds.features import (
    bounding_box,
    centroid,
    estimate_curvature,
    estimate_normals,
    extent,
)


class TestNormals:
    def test_plane_normals_are_perpendicular(self, plane_points) -> None:
        normals = estimate_normals(PointCloud(points=plane_points), k=16)
        assert normals.shape == plane_points.shape
        # On the z=0 plane every normal must be ±z (|nz| ~ 1, nx,ny ~ 0).
        assert np.all(np.abs(normals[:, 2]) > 0.99)
        assert np.all(np.abs(normals[:, :2]) < 0.1)

    def test_unit_length(self, sphere_points) -> None:
        normals = estimate_normals(PointCloud(points=sphere_points), k=16)
        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-6)

    def test_as_cloud_attaches_normals(self, plane_points) -> None:
        pc = estimate_normals(PointCloud(points=plane_points), k=16, as_cloud=True)
        assert isinstance(pc, PointCloud)
        assert pc.normals is not None and pc.normals.shape == plane_points.shape


class TestCurvature:
    def test_plane_curvature_near_zero(self, plane_points) -> None:
        curv = estimate_curvature(PointCloud(points=plane_points), k=16)
        assert curv.shape == (plane_points.shape[0],)
        assert np.median(curv) < 1e-2

    def test_sphere_curvature_positive(self, sphere_points) -> None:
        # A flat plane has ~0 surface variation; a curved sphere has clearly
        # larger surface variation. (Surface variation is bounded in [0, 1/3].)
        curv = estimate_curvature(PointCloud(points=sphere_points), k=24)
        assert np.median(curv) > 0.0
        assert np.all(curv <= 1.0 / 3.0 + 1e-9)


class TestStats:
    def test_bounding_box_and_extent(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]])
        bbox_min, bbox_max = bounding_box(PointCloud(points=pts))
        np.testing.assert_allclose(bbox_min, [0, 0, 0])
        np.testing.assert_allclose(bbox_max, [2, 4, 6])
        np.testing.assert_allclose(extent(PointCloud(points=pts)), [2, 4, 6])

    def test_centroid(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        np.testing.assert_allclose(centroid(PointCloud(points=pts)), [1, 1, 1])
