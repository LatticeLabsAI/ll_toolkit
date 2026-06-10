"""Real B-Rep face-adjacency graph construction from OCC shapes.

This module builds the face-to-face adjacency graph that GNN dataset builders
consume, derived directly from B-Rep topology rather than from array order.

Two faces are adjacent **iff they share a topological edge** — computed with
OCC's ``MapShapesAndAncestors`` (the same primitive used by
``cadling.lib.occ_wrapper._face_adjacency_pythonocc`` and
``cadling.models.topology_validation``). Face and edge features
(surface/curve type, area, centroid, outward normal, Gaussian/mean curvature,
dihedral angle, convexity) are computed from the real geometry.

This replaces the previous per-builder placeholders that connected each face to
the next four faces by index (``range(i+1, min(i+5, num_faces))``) and emitted
zeroed normals/curvatures/convexity/dihedral. It is the single source of truth
shared by:

  - ``cadling/data/hf_builders/brep_graph_builder.py``
  - ``cadling/data/hf_builders/arrow_brep_builder.py``
  - ``cadling/data/webdataset.py``

The returned dict matches the schema those builders already serialize::

    {
        "faces": [
            {"idx", "surface_type", "area", "centroid", "normal", "curvatures"},
            ...
        ],
        "edges": [
            {"idx", "curve_type", "length", "convexity", "dihedral_angle"},
            ...
        ],
        "edge_index": [[src...], [dst...]],   # bidirectional face adjacency
    }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from cadling.lib.geometry.face_geometry import FaceGeometryExtractor
from cadling.lib.graph.features import compute_dihedral_angle

_log = logging.getLogger(__name__)

HAS_OCC = False
try:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GeomAbs import (
        GeomAbs_BSplineCurve,
        GeomAbs_BSplineSurface,
        GeomAbs_Circle,
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Ellipse,
        GeomAbs_Line,
        GeomAbs_Plane,
        GeomAbs_Sphere,
        GeomAbs_Torus,
    )
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.gp import gp_Pnt, gp_Vec
    from OCC.Core.TopAbs import (
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_FORWARD,
        TopAbs_REVERSED,
    )
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopTools import (
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopTools_IndexedMapOfShape,
        TopTools_ListIteratorOfListOfShape,
    )

    HAS_OCC = True

    _SURFACE_TYPE_MAP = {
        GeomAbs_Plane: "plane",
        GeomAbs_Cylinder: "cylinder",
        GeomAbs_Cone: "cone",
        GeomAbs_Sphere: "sphere",
        GeomAbs_Torus: "torus",
        GeomAbs_BSplineSurface: "bspline",
    }
    _CURVE_TYPE_MAP = {
        GeomAbs_Line: "line",
        GeomAbs_Circle: "circle",
        GeomAbs_Ellipse: "ellipse",
        GeomAbs_BSplineCurve: "bspline",
    }
except ImportError:  # pragma: no cover - exercised only without pythonocc
    _log.debug(
        "pythonocc-core not available; build_brep_face_graph will raise RuntimeError."
    )
    _SURFACE_TYPE_MAP = {}
    _CURVE_TYPE_MAP = {}


# Stateless; reused across calls. Guarded so importing without OCC stays quiet.
_FACE_EXTRACTOR = FaceGeometryExtractor() if HAS_OCC else None


def build_brep_face_graph(shape: Any) -> Dict[str, Any]:
    """Build a real face-adjacency graph with geometric features from a B-Rep.

    Args:
        shape: A non-null ``OCC.Core.TopoDS.TopoDS_Shape`` (typically the result
            of ``STEPControl_Reader.OneShape()``).

    Returns:
        Dict with ``"faces"``, ``"edges"`` and ``"edge_index"`` (see module
        docstring for the exact schema). ``edge_index`` is a bidirectional
        adjacency: for every pair of faces sharing at least one edge, both
        ``(a, b)`` and ``(b, a)`` are present.

    Raises:
        RuntimeError: if pythonocc-core is not importable.
        ValueError: if ``shape`` is None or null.
    """
    if not HAS_OCC:
        raise RuntimeError(
            "pythonocc-core is required to build B-Rep face graphs but is not installed."
        )
    if shape is None or shape.IsNull():
        raise ValueError("Cannot build a face graph from a null shape.")

    # 1. Index every face with a stable 1-based OCC index. The same TopoDS_Face
    #    objects are returned as ancestors below, so FindIndex maps them back.
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    num_faces = face_map.Size()

    # Enumerate faces with TopExp_Explorer so each face carries the orientation
    # it has *in the solid* — required for a correct solid-outward normal (e.g.
    # a hole wall is REVERSED relative to its natural surface normal). FindIndex
    # maps each oriented face back to its stable face_map index (orientation-
    # insensitive), keeping faces_data and the adjacency in the same indexing.
    faces_data: List[Any] = [None] * num_faces
    face_normals: List[Any] = [None] * num_faces
    face_shapes: List[Any] = [None] * num_faces
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        idx = face_map.FindIndex(face) - 1
        if 0 <= idx < num_faces and faces_data[idx] is None:
            record = _build_face_record(face, idx)
            faces_data[idx] = record
            face_normals[idx] = np.asarray(record["normal"], dtype=float)
            face_shapes[idx] = face
        explorer.Next()
    # Defensive: fill any face the explorer somehow missed (non-solid shells).
    for i in range(num_faces):
        if faces_data[i] is None:
            face = topods.Face(face_map.FindKey(i + 1))
            record = _build_face_record(face, i)
            faces_data[i] = record
            face_normals[i] = np.asarray(record["normal"], dtype=float)
            face_shapes[i] = face

    # 2. Map each edge to its adjacent (ancestor) faces — the real topology.
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    edges_data: List[Dict[str, Any]] = []
    adjacency: set[Tuple[int, int]] = set()

    for i in range(1, edge_face_map.Size() + 1):
        edge = topods.Edge(edge_face_map.FindKey(i))
        face_list = edge_face_map.FindFromIndex(i)

        # Distinct adjacent face indices. A seam edge (e.g. on a cylinder)
        # lists the same face twice; dict.fromkeys preserves order and dedupes.
        adj_indices: List[int] = []
        list_it = TopTools_ListIteratorOfListOfShape(face_list)
        while list_it.More():
            f = topods.Face(list_it.Value())
            idx = face_map.FindIndex(f) - 1
            if idx >= 0:
                adj_indices.append(idx)
            list_it.Next()
        adj_unique = list(dict.fromkeys(adj_indices))

        dihedral, convexity = _edge_dihedral_convexity(
            edge, adj_unique, face_normals, face_shapes
        )
        edges_data.append(_build_edge_record(edge, i - 1, convexity, dihedral))

        # Connect every distinct pair of faces meeting at this edge.
        for a in range(len(adj_unique)):
            for b in range(a + 1, len(adj_unique)):
                fa, fb = adj_unique[a], adj_unique[b]
                if fa != fb:
                    adjacency.add((fa, fb))
                    adjacency.add((fb, fa))

    ordered = sorted(adjacency)
    edge_index_src = [pair[0] for pair in ordered]
    edge_index_dst = [pair[1] for pair in ordered]

    return {
        "faces": faces_data,
        "edges": edges_data,
        "edge_index": [edge_index_src, edge_index_dst],
    }


def _build_face_record(face: Any, idx: int) -> Dict[str, Any]:
    """Build one face node with real surface type, area, centroid, normal, curvature."""
    adaptor = BRepAdaptor_Surface(face)
    surface_type = _SURFACE_TYPE_MAP.get(adaptor.GetType(), "other")

    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    area = props.Mass()
    center = props.CentreOfMass()
    centroid = [center.X(), center.Y(), center.Z()]

    feats = _FACE_EXTRACTOR.extract_features(face) or {}
    normal = list(feats.get("normal", [0.0, 0.0, 1.0]))
    # OCC's surface normal ignores face orientation; flip so it points outward.
    if face.Orientation() == TopAbs_REVERSED:
        normal = [-normal[0], -normal[1], -normal[2]]

    gaussian = float(feats.get("gaussian_curvature", 0.0))
    mean = float(feats.get("mean_curvature", 0.0))

    return {
        "idx": idx,
        "surface_type": surface_type,
        "area": float(area),
        "centroid": [float(c) for c in centroid],
        "normal": [float(n) for n in normal],
        "curvatures": [gaussian, mean],
    }


def _build_edge_record(
    edge: Any, idx: int, convexity: float, dihedral: float
) -> Dict[str, Any]:
    """Build one edge record with real curve type, length, convexity, dihedral angle."""
    try:
        adaptor = BRepAdaptor_Curve(edge)
        curve_type = _CURVE_TYPE_MAP.get(adaptor.GetType(), "other")
        props = GProp_GProps()
        brepgprop.LinearProperties(edge, props)
        length = float(props.Mass())
    except Exception as exc:  # degenerate / unsupported curve
        _log.debug("Edge measurement failed: %s", exc)
        curve_type = "other"
        length = 0.0

    return {
        "idx": idx,
        "curve_type": curve_type,
        "length": length,
        "convexity": float(convexity),
        "dihedral_angle": float(dihedral),
    }


def _edge_tangent(edge: Any) -> "np.ndarray | None":
    """Return the unit tangent of an edge at its mid-parameter, or None."""
    try:
        curve_handle, first, last = BRep_Tool.Curve(edge)
        if curve_handle is None:
            return None
        point = gp_Pnt()
        deriv = gp_Vec()
        curve_handle.D1(0.5 * (first + last), point, deriv)
        t = np.array([deriv.X(), deriv.Y(), deriv.Z()], dtype=float)
        n = float(np.linalg.norm(t))
        if n < 1e-12:
            return None
        return t / n
    except Exception as exc:  # degenerate / curveless edge (e.g. seam)
        _log.debug("Edge tangent unavailable: %s", exc)
        return None


def _edge_orientation_in_face(edge: Any, face: Any) -> "float | None":
    """Return +1 if the edge is FORWARD in the face's wire, -1 if REVERSED.

    This gives the coedge orientation, which is what makes the convexity sign
    well-defined: ``cross(outward_normal, forward_tangent)`` points into the
    face's interior.
    """
    try:
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while explorer.More():
            candidate = explorer.Current()
            if candidate.IsSame(edge):
                return 1.0 if candidate.Orientation() == TopAbs_FORWARD else -1.0
            explorer.Next()
    except Exception as exc:
        _log.debug("Edge orientation lookup failed: %s", exc)
    return None


def _edge_dihedral_convexity(
    edge: Any,
    adj_indices: List[int],
    face_normals: List[np.ndarray],
    face_shapes: List[Any],
) -> Tuple[float, float]:
    """Compute (dihedral_angle, convexity) for an edge from its two adjacent faces.

    The dihedral angle is the unsigned angle between the two faces' outward
    normals (radians, [0, π]).

    Convexity uses the *signed* coedge test, which is correct for both planar and
    curved faces (verified on box/notch/hole/pocket solids). With ``nA``/``nB`` the
    two faces' outward normals and ``tA`` the edge tangent oriented along its
    FORWARD coedge in face A, ``cross(nA, tA)`` points into A's interior; the sign
    of ``s = dot(cross(nA, tA), nB)`` then distinguishes the fold direction:
    ``s < 0`` → **convex**, ``s > 0`` → **concave**. A normal-dot-product test
    cannot sign a 90° edge and a face-centroid test is wrong for cylinders (centroid
    on the axis); this avoids both. Encoding: ``1.0`` convex, ``0.0`` concave,
    ``0.5`` tangent/boundary/unknown.

    Boundary edges (one face), non-manifold edges (>2 faces), tangent joins
    (near-parallel normals), and missing geometry all return the neutral ``0.5``.
    """
    if len(adj_indices) < 2:
        # Boundary edge (1 face) or non-manifold (>2): no well-defined dihedral.
        return 0.0, 0.5

    n1 = face_normals[adj_indices[0]]
    n2 = face_normals[adj_indices[1]]
    dihedral = compute_dihedral_angle(n1, n2)

    norm1 = float(np.linalg.norm(n1))
    norm2 = float(np.linalg.norm(n2))
    if norm1 < 1e-10 or norm2 < 1e-10:
        return dihedral, 0.5

    cos_angle = float(np.dot(n1, n2) / (norm1 * norm2))
    if abs(cos_angle) > 0.999:
        # Faces are tangent / coplanar across the edge — smooth, no convex/concave.
        return dihedral, 0.5

    tangent = _edge_tangent(edge)
    orientation = _edge_orientation_in_face(edge, face_shapes[adj_indices[0]])
    if tangent is None or orientation is None:
        return dihedral, 0.5

    forward_tangent = tangent * orientation
    inward_a = np.cross(n1 / norm1, forward_tangent)  # into face A's interior
    s = float(np.dot(inward_a, n2 / norm2))

    if s < -1e-6:
        return dihedral, 1.0  # convex
    if s > 1e-6:
        return dihedral, 0.0  # concave
    return dihedral, 0.5  # indeterminate
