"""Tests for ll_clouds Pydantic data models (SPEC-1 M5, T5.1)."""

from __future__ import annotations

import numpy as np
import pytest

from ll_clouds.datamodel import PointCloud, RegistrationResult, SegmentationResult


class TestPointCloud:
    def test_minimal_construction(self) -> None:
        pts = np.zeros((10, 3))
        pc = PointCloud(points=pts)
        assert pc.num_points == 10
        assert len(pc) == 10
        assert pc.normals is None and pc.colors is None and pc.labels is None

    def test_optional_fields(self) -> None:
        pts = np.random.default_rng(0).normal(size=(8, 3))
        pc = PointCloud(
            points=pts,
            normals=np.zeros((8, 3)),
            colors=np.zeros((8, 3)),
            labels=np.arange(8),
            metadata={"source": "test"},
        )
        assert pc.normals.shape == (8, 3)
        assert pc.labels.shape == (8,)
        assert pc.metadata["source"] == "test"

    def test_points_must_be_n_by_3(self) -> None:
        with pytest.raises(ValueError):
            PointCloud(points=np.zeros((10, 2)))

    def test_normals_length_must_match_points(self) -> None:
        with pytest.raises(ValueError):
            PointCloud(points=np.zeros((10, 3)), normals=np.zeros((9, 3)))

    def test_labels_length_must_match_points(self) -> None:
        with pytest.raises(ValueError):
            PointCloud(points=np.zeros((10, 3)), labels=np.zeros(9))

    def test_points_coerced_to_float_ndarray(self) -> None:
        pc = PointCloud(points=[[0, 0, 0], [1, 1, 1]])
        assert isinstance(pc.points, np.ndarray)
        assert pc.points.shape == (2, 3)
        assert np.issubdtype(pc.points.dtype, np.floating)


class TestResultModels:
    def test_registration_result(self) -> None:
        transform = np.eye(4)
        res = RegistrationResult(
            transformation=transform,
            fitness=0.9,
            inlier_rmse=0.01,
            iterations=12,
            converged=True,
        )
        assert res.transformation.shape == (4, 4)
        assert res.converged is True
        assert res.iterations == 12

    def test_registration_transformation_must_be_4x4(self) -> None:
        with pytest.raises(ValueError):
            RegistrationResult(
                transformation=np.eye(3),
                fitness=1.0,
                inlier_rmse=0.0,
                iterations=1,
                converged=True,
            )

    def test_segmentation_result(self) -> None:
        labels = np.array([0, 0, 1, 1, -1])
        res = SegmentationResult(labels=labels, num_segments=2)
        assert res.num_segments == 2
        assert res.labels.shape == (5,)
