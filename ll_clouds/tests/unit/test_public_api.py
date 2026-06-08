"""The documented public API is importable from the package root (M5, T5.8)."""

from __future__ import annotations

import importlib

import ll_clouds


def test_all_exports_are_present() -> None:
    for name in ll_clouds.__all__:
        assert hasattr(ll_clouds, name), f"ll_clouds.__all__ lists missing '{name}'"


def test_key_callables_are_usable() -> None:
    import numpy as np

    pc = ll_clouds.PointCloud(points=np.zeros((4, 3)))
    assert ll_clouds.centroid(pc).shape == (3,)
    assert callable(ll_clouds.icp)
    assert callable(ll_clouds.ransac_plane)
    assert callable(ll_clouds.estimate_normals)


def test_bridges_submodule_is_accessible() -> None:
    bridges = importlib.import_module("ll_clouds.bridges")
    assert hasattr(bridges, "from_mesh")
