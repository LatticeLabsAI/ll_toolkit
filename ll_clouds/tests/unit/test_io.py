"""Tests for ll_clouds I/O (SPEC-1 M5, T5.2)."""

from __future__ import annotations

import numpy as np
import pytest

from ll_clouds.datamodel import PointCloud
from ll_clouds.io import read_point_cloud, sample_from_mesh, write_point_cloud


@pytest.mark.parametrize("ext", [".xyz", ".pcd", ".ply"])
def test_write_read_roundtrip_points(tmp_path, rng, ext) -> None:
    pts = rng.normal(size=(50, 3))
    pc = PointCloud(points=pts)
    path = tmp_path / f"cloud{ext}"
    write_point_cloud(pc, str(path))
    loaded = read_point_cloud(str(path))
    assert loaded.num_points == 50
    np.testing.assert_allclose(loaded.points, pts, rtol=0, atol=1e-5)


@pytest.mark.parametrize("ext", [".ply", ".pcd"])
def test_roundtrip_preserves_normals(tmp_path, rng, ext) -> None:
    pts = rng.normal(size=(20, 3))
    normals = rng.normal(size=(20, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    pc = PointCloud(points=pts, normals=normals)
    path = tmp_path / f"cloud{ext}"
    write_point_cloud(pc, str(path))
    loaded = read_point_cloud(str(path))
    assert loaded.normals is not None
    np.testing.assert_allclose(loaded.normals, normals, rtol=0, atol=1e-5)


def test_read_unsupported_extension_raises(tmp_path) -> None:
    p = tmp_path / "cloud.bogus"
    p.write_text("nope")
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        read_point_cloud(str(p))


@pytest.mark.requires_trimesh
def test_sample_from_mesh_returns_requested_count() -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    pc = sample_from_mesh(mesh, n=1024, with_normals=True)
    assert isinstance(pc, PointCloud)
    assert pc.num_points == 1024
    assert pc.normals is not None and pc.normals.shape == (1024, 3)
    # Box half-extent 0.5: sampled points lie within the bounding box.
    assert np.all(pc.points >= -0.5 - 1e-6) and np.all(pc.points <= 0.5 + 1e-6)


@pytest.mark.requires_trimesh
def test_sample_from_mesh_file_path(tmp_path) -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh_path = tmp_path / "sphere.stl"
    mesh.export(str(mesh_path))
    pc = sample_from_mesh(str(mesh_path), n=512)
    assert pc.num_points == 512
    # Points sampled on a unit sphere: radius ~1.
    radii = np.linalg.norm(pc.points, axis=1)
    assert np.allclose(radii, 1.0, atol=0.05)
