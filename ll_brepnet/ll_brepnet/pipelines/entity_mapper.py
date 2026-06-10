"""Stable integer indices for the faces and edges of a B-Rep solid.

``BRepEntityMapper`` assigns each face and edge of a ``TopoDS_Shape`` a stable
0-based index and keeps the inverse (index -> ``TopoDS`` entity) so that
per-face / per-edge feature and UV-grid arrays can be built in a consistent
order. Coedges produced by ``cadling``'s ``CoedgeExtractor`` carry string
``face_id`` / ``edge_id`` identifiers; this mapper computes the **same** stable
identifier (OCC ``HashCode`` with a ``hash()`` fallback) so a coedge can be
resolved to the index of its parent face/edge.

This module is deliberately self-contained: it does not rely on
``cadling``'s ``ShapeIdentityRegistry``, whose ``register_all_*`` helpers can
silently register nothing on some pythonocc builds (a bad ``topods`` import),
which would leave every ``face_index`` at ``-1``.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

HAS_OCC = False
try:
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Face, TopoDS_Shape, topods

    HAS_OCC = True
except ImportError:  # pragma: no cover - exercised only without pythonocc
    _log.debug("pythonocc-core not available; BRepEntityMapper is inert.")


def stable_shape_id(shape: TopoDS_Shape) -> str:
    """Return a stable identifier for an OCC shape.

    Uses OCCT ``HashCode`` when exposed (stable across the different Python
    wrapper objects OCC hands back for the same topological entity), falling
    back to Python's ``hash`` on pythonocc builds that drop ``HashCode``. This
    matches ``cadling.lib.topology.face_identity.ShapeIdentityRegistry.get_id``
    so coedge ``face_id`` / ``edge_id`` strings line up with this mapper.
    """
    try:
        return str(shape.HashCode(2**31 - 1))
    except (AttributeError, TypeError):
        return str(hash(shape))


class BRepEntityMapper:
    """Assigns stable 0-based indices to the faces and edges of a solid.

    Args:
        shape: The ``TopoDS_Shape`` (typically a solid) to index.
    """

    def __init__(self, shape: TopoDS_Shape):
        self._faces: list[TopoDS_Face] = []
        self._edges: list[TopoDS_Edge] = []
        self._face_id_to_index: dict[str, int] = {}
        self._edge_id_to_index: dict[str, int] = {}
        if HAS_OCC and shape is not None:
            self._build(shape)

    def _build(self, shape: TopoDS_Shape) -> None:
        face_exp = TopExp_Explorer(shape, TopAbs_FACE)
        while face_exp.More():
            face = topods.Face(face_exp.Current())
            fid = stable_shape_id(face)
            if fid not in self._face_id_to_index:
                self._face_id_to_index[fid] = len(self._faces)
                self._faces.append(face)
            face_exp.Next()

        edge_exp = TopExp_Explorer(shape, TopAbs_EDGE)
        while edge_exp.More():
            edge = topods.Edge(edge_exp.Current())
            eid = stable_shape_id(edge)
            if eid not in self._edge_id_to_index:
                self._edge_id_to_index[eid] = len(self._edges)
                self._edges.append(edge)
            edge_exp.Next()

    # -- counts -----------------------------------------------------------
    @property
    def num_faces(self) -> int:
        return len(self._faces)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    # -- index -> entity --------------------------------------------------
    def face_by_index(self, index: int) -> TopoDS_Face:
        return self._faces[index]

    def edge_by_index(self, index: int) -> TopoDS_Edge:
        return self._edges[index]

    def faces(self) -> list[TopoDS_Face]:
        """Faces in index order."""
        return list(self._faces)

    def edges(self) -> list[TopoDS_Edge]:
        """Edges in index order."""
        return list(self._edges)

    # -- id -> index ------------------------------------------------------
    def face_index(self, face_id: str) -> int | None:
        """Index of the face with the given stable id, or ``None``."""
        return self._face_id_to_index.get(face_id)

    def edge_index(self, edge_id: str) -> int | None:
        """Index of the edge with the given stable id, or ``None``."""
        return self._edge_id_to_index.get(edge_id)
