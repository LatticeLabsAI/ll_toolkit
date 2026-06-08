"""Tests for ll_clouds bridges to cadling / ll_ocadr (SPEC-1 M5, T5.7).

The bridges lazily import their heavy dependencies, so importing
``ll_clouds.bridges`` must succeed with only numpy installed. The cadling /
ll_ocadr converters are duck-typed on the same attributes those packages
actually expose (cadling ``MeshItem.vertices``/``normals``; ll_ocadr
``MeshData.vertices``/``normals``), verified against their source — so they are
exercised here with light stand-ins instead of pulling those heavy packages in.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from ll_clouds import bridges
from ll_clouds.datamodel import PointCloud


def test_bridges_module_imports_without_heavy_deps() -> None:
    """The module must import even though cadling/ll_ocadr/trimesh may be absent."""
    assert hasattr(bridges, "from_cadling_document")
    assert hasattr(bridges, "from_ll_ocadr_mesh")
    assert hasattr(bridges, "to_ll_ocadr_arrays")
    assert hasattr(bridges, "from_mesh")


@pytest.mark.requires_trimesh
def test_from_mesh() -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    pc = bridges.from_mesh(mesh, n=256, with_normals=True)
    assert isinstance(pc, PointCloud)
    assert pc.num_points == 256
    assert pc.normals is not None


def test_from_cadling_document() -> None:
    # Stand-in mirroring cadling's CADlingDocument.items -> MeshItem.vertices/normals.
    verts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    norms = [[0.0, 0.0, 1.0]] * 4
    mesh_item = types.SimpleNamespace(vertices=verts, normals=norms)
    doc = types.SimpleNamespace(name="test", items=[mesh_item])

    pc = bridges.from_cadling_document(doc)
    assert isinstance(pc, PointCloud)
    assert pc.num_points == 4
    np.testing.assert_allclose(pc.points, np.asarray(verts))
    assert pc.normals is not None and pc.normals.shape == (4, 3)


def test_from_cadling_document_walks_children() -> None:
    leaf = types.SimpleNamespace(vertices=[[1.0, 2.0, 3.0]], normals=[])
    parent = types.SimpleNamespace(vertices=None, children=[leaf])
    doc = types.SimpleNamespace(items=[parent])
    pc = bridges.from_cadling_document(doc)
    assert pc.num_points == 1
    np.testing.assert_allclose(pc.points, [[1.0, 2.0, 3.0]])


def test_from_cadling_document_no_vertices_raises() -> None:
    doc = types.SimpleNamespace(items=[types.SimpleNamespace(vertices=None)])
    with pytest.raises(ValueError, match="no mesh vertices"):
        bridges.from_cadling_document(doc)


def test_from_ll_ocadr_mesh() -> None:
    # Stand-in mirroring ll_ocadr MeshData.vertices / .normals (ndarrays).
    verts = np.random.default_rng(0).normal(size=(20, 3))
    mesh_data = types.SimpleNamespace(vertices=verts, normals=np.zeros((20, 3)))
    pc = bridges.from_ll_ocadr_mesh(mesh_data)
    assert isinstance(pc, PointCloud)
    assert pc.num_points == 20
    np.testing.assert_allclose(pc.points, verts)


def test_from_ll_ocadr_mesh_mismatched_normals_dropped() -> None:
    verts = np.random.default_rng(3).normal(size=(10, 3))
    mesh_data = types.SimpleNamespace(vertices=verts, normals=np.zeros((4, 3)))
    pc = bridges.from_ll_ocadr_mesh(mesh_data)
    assert pc.num_points == 10
    assert pc.normals is None  # mismatched length -> dropped


def test_to_ll_ocadr_arrays_batched() -> None:
    pts = np.random.default_rng(1).normal(size=(15, 3))
    pc = PointCloud(points=pts, normals=np.zeros((15, 3)))
    coords, norms = bridges.to_ll_ocadr_arrays(pc, batched=True)
    assert coords.shape == (1, 15, 3)
    assert norms.shape == (1, 15, 3)
    np.testing.assert_allclose(coords[0], pts)


def test_to_ll_ocadr_arrays_unbatched_no_normals() -> None:
    pts = np.random.default_rng(2).normal(size=(15, 3))
    pc = PointCloud(points=pts)
    coords, norms = bridges.to_ll_ocadr_arrays(pc, batched=False)
    assert coords.shape == (15, 3)
    assert norms is None


def test_from_cadling_document_include_normals_false() -> None:
    verts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    norms = [[0.0, 0.0, 1.0]] * 3
    doc = types.SimpleNamespace(
        items=[types.SimpleNamespace(vertices=verts, normals=norms)]
    )
    pc = bridges.from_cadling_document(doc, include_normals=False)
    assert pc.num_points == 3
    assert pc.normals is None  # normals suppressed despite being present


def test_from_cadling_document_mismatched_normals_dropped() -> None:
    verts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    norms = [[0.0, 0.0, 1.0]]  # length 1 != 3 vertices
    doc = types.SimpleNamespace(
        items=[types.SimpleNamespace(vertices=verts, normals=norms)]
    )
    pc = bridges.from_cadling_document(doc)
    assert pc.num_points == 3
    assert pc.normals is None  # mismatched length -> normals dropped


def test_from_cadling_document_numpy_vertices_do_not_raise() -> None:
    # NumPy-array vertices must not trip the truthiness check.
    verts = np.zeros((4, 3))
    doc = types.SimpleNamespace(
        items=[types.SimpleNamespace(vertices=verts, normals=None)]
    )
    pc = bridges.from_cadling_document(doc)
    assert pc.num_points == 4
