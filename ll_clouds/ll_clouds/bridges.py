"""Lazy bridges between ll_clouds and the rest of the LatticeLabs toolkit.

Every heavy/optional dependency (trimesh, cadling, ll_ocadr, torch) is imported
*inside* the function that needs it, so ``import ll_clouds.bridges`` succeeds
with only numpy + the core ll_clouds package installed. This keeps ll_clouds a
standalone library while still offering first-class interop.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .datamodel import PointCloud


def from_mesh(mesh: Any, n: int = 2048, with_normals: bool = True) -> PointCloud:
    """Sample a PointCloud from a mesh (trimesh object or file path).

    Thin convenience wrapper over :func:`ll_clouds.io.sample_from_mesh`.
    """
    from .io import sample_from_mesh

    return sample_from_mesh(mesh, n=n, with_normals=with_normals)


def from_cadling_document(doc: Any, include_normals: bool = True) -> PointCloud:
    """Build a PointCloud from a cadling ``CADlingDocument``.

    Collects vertices (and, when available and consistent, normals) from every
    mesh item in the document's item hierarchy. Duck-typed on a ``.vertices``
    attribute (cadling ``MeshItem.vertices``), so it needs no cadling import.
    """
    vertices: list[list[float]] = []
    normals: list[list[float]] = []

    def _walk(items: Any) -> None:
        for item in items or []:
            item_verts = getattr(item, "vertices", None)
            if item_verts:
                vertices.extend([list(v) for v in item_verts])
                item_norms = getattr(item, "normals", None)
                if item_norms and len(item_norms) == len(item_verts):
                    normals.extend([list(nrm) for nrm in item_norms])
            _walk(getattr(item, "children", None))

    _walk(getattr(doc, "items", None))

    if not vertices:
        raise ValueError("cadling document contains no mesh vertices")

    points = np.asarray(vertices, dtype=np.float64)
    out_normals = None
    if include_normals and len(normals) == len(vertices):
        out_normals = np.asarray(normals, dtype=np.float64)

    return PointCloud(
        points=points,
        normals=out_normals,
        metadata={"source": "cadling", "document": getattr(doc, "name", None)},
    )


def from_ll_ocadr_mesh(mesh_data: Any) -> PointCloud:
    """Convert an ll_ocadr ``MeshData`` (vertices/normals ndarrays) to a PointCloud."""
    points = np.asarray(mesh_data.vertices, dtype=np.float64)
    normals = getattr(mesh_data, "normals", None)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
        if normals.shape != points.shape:
            normals = None
    return PointCloud(points=points, normals=normals, metadata={"source": "ll_ocadr"})


def to_ll_ocadr_arrays(
    pc: PointCloud, batched: bool = True
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert a PointCloud to ``(vertex_coords, vertex_normals)`` arrays shaped
    for the ll_ocadr model.

    With ``batched=True`` a leading batch dimension is added so the arrays are
    ``[1, N, 3]`` (ready for ``torch.from_numpy``); otherwise they are ``[N, 3]``.
    Normals are ``None`` when the cloud has none.
    """
    coords = np.asarray(pc.points, dtype=np.float64)
    normals = None if pc.normals is None else np.asarray(pc.normals, dtype=np.float64)
    if batched:
        coords = coords[None, ...]
        if normals is not None:
            normals = normals[None, ...]
    return coords, normals
