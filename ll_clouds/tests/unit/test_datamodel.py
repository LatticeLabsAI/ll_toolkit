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


class TestPointCloudMetadata:
    def test_default_metadata_not_shared_between_instances(self) -> None:
        pts = np.zeros((5, 3))
        pc1 = PointCloud(points=pts)
        pc2 = PointCloud(points=pts)
        pc1.metadata["foo"] = "bar"
        assert "foo" in pc1.metadata
        assert "foo" not in pc2.metadata
        assert pc1.metadata is not pc2.metadata

    def test_metadata_copied_not_aliased_on_normalize(self) -> None:
        from ll_clouds.preprocess import normalize

        pc = PointCloud(
            points=np.random.default_rng(1).normal(size=(16, 3)),
            metadata={"source": "unit-test", "version": 1},
        )
        out = normalize(pc)
        assert out.metadata == pc.metadata
        assert out.metadata is not pc.metadata
        out.metadata["new_key"] = "x"
        assert "new_key" not in pc.metadata

    def test_segmentation_result_default_metadata_not_shared(self) -> None:
        r1 = SegmentationResult(labels=np.zeros(3), num_segments=0)
        r2 = SegmentationResult(labels=np.zeros(3), num_segments=0)
        r1.metadata["k"] = 1
        assert "k" not in r2.metadata
        assert r1.metadata is not r2.metadata
