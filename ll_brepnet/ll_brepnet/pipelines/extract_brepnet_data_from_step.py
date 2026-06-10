"""Extract a BRepNet-style ``.npz`` training record from a STEP file.

The record captures, for one solid:

* the **coedge graph** -- ``coedge_to_next / prev / mate / face / edge`` integer
  incidence arrays (oriented half-edge adjacency), plus a ``coedge_reversed``
  flag;
* per-**face** features (surface-type one-hot + area) and a ``[7, U, V]`` UV-grid
  (xyz + normal + trimming mask);
* per-**edge** features (curve-type one-hot + length + convexity) and a
  ``[6, U]`` U-grid (xyz + tangent).

Geometry is normalised to the unit box ``[-1, 1]^3`` before sampling. The coedge
topology is obtained from ``cadling``'s ``CoedgeExtractor`` (oriented loops via
``BRepTools_WireExplorer`` + mate finding via ``MapShapesAndAncestors``); all
face/edge indexing, features and UV-grids are computed here so the record does
not depend on ``cadling``'s shape-identity registry (which can register nothing
on some pythonocc builds).

This is an independent, MIT-licensed implementation -- see ``ATTRIBUTION.md``.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .entity_mapper import BRepEntityMapper

_log = logging.getLogger(__name__)

# UV-grid resolution (UV-Net default).
NUM_U = 10
NUM_V = 10

# Feature layouts (kept small and explicit; standardisation happens later in the
# dataset using statistics over the training split).
SURFACE_TYPES = ["plane", "cylinder", "cone", "sphere", "torus", "bspline", "other"]
CURVE_TYPES = ["line", "circle", "ellipse", "bspline", "other"]
NUM_FACE_FEATURES = len(SURFACE_TYPES) + 1  # + area
NUM_EDGE_FEATURES = len(CURVE_TYPES) + 2  # + length + convexity

HAS_OCC = False
try:
    from cadling.lib.topology.coedge_extractor import CoedgeExtractor
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepLProp import BRepLProp_SLProps
    from OCC.Core.GeomAbs import (
        GeomAbs_BezierCurve,
        GeomAbs_BezierSurface,
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
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Face, TopoDS_Shape
    from occwl.edge import Edge as OCCWLEdge
    from occwl.face import Face as OCCWLFace
    from occwl.uvgrid import ugrid, uvgrid

    HAS_OCC = True
except ImportError as _exc:  # pragma: no cover - exercised only without pythonocc
    _log.debug("pythonocc/occwl/cadling not available; STEP extraction is inert: %s", _exc)


# ---------------------------------------------------------------------------
# STEP loading + normalisation
# ---------------------------------------------------------------------------


def load_step_shape(step_file: Path) -> TopoDS_Shape:
    """Read a STEP file and return its (compound) ``TopoDS_Shape``."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_file))
    if status != IFSelect_RetDone:
        raise OSError(f"Failed to read STEP file: {step_file}")
    reader.TransferRoots()
    return reader.OneShape()


def scale_shape_to_unit_box(shape: TopoDS_Shape) -> TopoDS_Shape:
    """Center ``shape`` at the origin and uniformly scale it into ``[-1, 1]^3``.

    A single uniform scale (the largest half-extent maps to 1.0) preserves the
    aspect ratio, matching how UV-Net / BRepNet normalise solids before
    sampling geometry.
    """
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    cx, cy, cz = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
    half = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2.0
    if half <= 0.0:
        return shape
    scale = 1.0 / half

    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(-cx, -cy, -cz))
    scale_trsf = gp_Trsf()
    scale_trsf.SetScale(gp_Pnt(0.0, 0.0, 0.0), scale)
    combined = scale_trsf.Multiplied(trsf)
    return BRepBuilderAPI_Transform(shape, combined, True).Shape()


# ---------------------------------------------------------------------------
# Per-face / per-edge geometry features
# ---------------------------------------------------------------------------


def _one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec


def face_surface_onehot(face: TopoDS_Face) -> np.ndarray:
    """One-hot of the face's surface type over :data:`SURFACE_TYPES`."""
    surf = BRepAdaptor_Surface(face)
    t = surf.GetType()
    mapping = {
        GeomAbs_Plane: 0,
        GeomAbs_Cylinder: 1,
        GeomAbs_Cone: 2,
        GeomAbs_Sphere: 3,
        GeomAbs_Torus: 4,
        GeomAbs_BSplineSurface: 5,
        GeomAbs_BezierSurface: 5,
    }
    return _one_hot(mapping.get(t, len(SURFACE_TYPES) - 1), len(SURFACE_TYPES))


def face_area(face: TopoDS_Face) -> float:
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return float(props.Mass())


def edge_curve_onehot(edge: TopoDS_Edge) -> np.ndarray:
    """One-hot of the edge's curve type over :data:`CURVE_TYPES`."""
    try:
        curve = BRepAdaptor_Curve(edge)
        t = curve.GetType()
    except Exception:
        return _one_hot(len(CURVE_TYPES) - 1, len(CURVE_TYPES))
    mapping = {
        GeomAbs_Line: 0,
        GeomAbs_Circle: 1,
        GeomAbs_Ellipse: 2,
        GeomAbs_BSplineCurve: 3,
        GeomAbs_BezierCurve: 3,
    }
    return _one_hot(mapping.get(t, len(CURVE_TYPES) - 1), len(CURVE_TYPES))


def edge_length(edge: TopoDS_Edge) -> float:
    props = GProp_GProps()
    brepgprop.LinearProperties(edge, props)
    return float(props.Mass())


def _face_normal_at(face: TopoDS_Face, pnt: gp_Pnt) -> np.ndarray | None:
    """Outward unit normal of ``face`` at the surface point nearest ``pnt``."""
    surface = BRep_Tool.Surface(face)
    proj = GeomAPI_ProjectPointOnSurf(pnt, surface)
    if proj.NbPoints() < 1:
        return None
    u, v = proj.LowerDistanceParameters()
    props = BRepLProp_SLProps(BRepAdaptor_Surface(face), float(u), float(v), 1, 1e-6)
    if not props.IsNormalDefined():
        return None
    n = props.Normal()
    normal = np.array([n.X(), n.Y(), n.Z()], dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    normal /= norm
    # Account for the face orientation flag (the surface normal is geometric).
    from OCC.Core.TopAbs import TopAbs_REVERSED

    if face.Orientation() == TopAbs_REVERSED:
        normal = -normal
    return normal


def _edge_mid_point_tangent(edge: TopoDS_Edge) -> tuple[gp_Pnt, np.ndarray] | None:
    """Return the 3D midpoint and unit tangent of ``edge``."""
    try:
        curve = BRepAdaptor_Curve(edge)
        u_mid = 0.5 * (curve.FirstParameter() + curve.LastParameter())
        pnt = gp_Pnt()
        vec = gp_Vec()
        curve.D1(u_mid, pnt, vec)
        tangent = np.array([vec.X(), vec.Y(), vec.Z()], dtype=np.float64)
        norm = np.linalg.norm(tangent)
        if norm < 1e-12:
            return None
        return pnt, tangent / norm
    except Exception:
        return None


def edge_convexity(
    edge: TopoDS_Edge,
    face_a: TopoDS_Face,
    face_b: TopoDS_Face,
) -> float:
    """Signed convexity of an edge shared by ``face_a`` and ``face_b``.

    Computes ``s = (n_a x t) . n_b`` at the edge midpoint, where ``n_a`` /
    ``n_b`` are the (orientation-corrected) face normals and ``t`` the edge
    tangent. ``s < 0`` -> convex (1.0), ``s > 0`` -> concave (0.0), and
    near-tangent edges (``|s|`` small) -> smooth (0.5). Returns 0.5 when the
    geometry cannot be evaluated (e.g. a boundary/seam edge).
    """
    mt = _edge_mid_point_tangent(edge)
    if mt is None:
        return 0.5
    pnt, tangent = mt
    n_a = _face_normal_at(face_a, pnt)
    n_b = _face_normal_at(face_b, pnt)
    if n_a is None or n_b is None:
        return 0.5
    s = float(np.dot(np.cross(n_a, tangent), n_b))
    if abs(s) < 1e-4:
        return 0.5
    return 1.0 if s < 0.0 else 0.0


# ---------------------------------------------------------------------------
# UV-grids (point + normal + inside mask for faces; point + tangent for edges)
# ---------------------------------------------------------------------------


def face_uv_grid(face: TopoDS_Face, num_u: int = NUM_U, num_v: int = NUM_V) -> np.ndarray:
    """Return a ``[7, num_u, num_v]`` UV-grid: xyz(3) + normal(3) + mask(1).

    Channels-first so it feeds directly into a 2D CNN surface encoder. Failed
    samples yield a zero grid (the trimming mask stays 0), which the model can
    distinguish from material via that mask channel.
    """
    try:
        wrapped = OCCWLFace(face)
        points = uvgrid(wrapped, num_u=num_u, num_v=num_v, method="point")
        normals = uvgrid(wrapped, num_u=num_u, num_v=num_v, method="normal")
        inside = uvgrid(wrapped, num_u=num_u, num_v=num_v, method="inside")
        if points is None or normals is None or inside is None:
            return np.zeros((7, num_u, num_v), dtype=np.float32)
        grid = np.concatenate([points, normals, inside.astype(np.float32)], axis=2)  # [u, v, 7]
        return np.transpose(grid, (2, 0, 1)).astype(np.float32)
    except Exception as exc:
        _log.debug("face UV-grid extraction failed: %s", exc)
        return np.zeros((7, num_u, num_v), dtype=np.float32)


def edge_u_grid(edge: TopoDS_Edge, num_u: int = NUM_U) -> np.ndarray:
    """Return a ``[6, num_u]`` U-grid: xyz(3) + tangent(3), channels-first."""
    try:
        wrapped = OCCWLEdge(edge)
        points = ugrid(wrapped, num_u=num_u, method="point")
        tangents = ugrid(wrapped, num_u=num_u, method="tangent")
        if points is None or tangents is None:
            return np.zeros((6, num_u), dtype=np.float32)
        grid = np.concatenate([points, tangents], axis=1)  # [u, 6]
        return np.transpose(grid, (1, 0)).astype(np.float32)
    except Exception as exc:
        _log.debug("edge U-grid extraction failed: %s", exc)
        return np.zeros((6, num_u), dtype=np.float32)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class BRepDataExtractor:
    """Extract BRepNet-style arrays from one STEP file.

    Args:
        step_file: Path to the ``.step`` / ``.stp`` file.
        scale_body: Normalise the solid into the unit box before sampling.
        num_u, num_v: UV-grid resolution.
    """

    def __init__(
        self,
        step_file: Path,
        scale_body: bool = True,
        num_u: int = NUM_U,
        num_v: int = NUM_V,
    ):
        self.step_file = Path(step_file)
        self.scale_body = scale_body
        self.num_u = num_u
        self.num_v = num_v

    def extract_arrays(self) -> dict[str, np.ndarray]:
        """Return the full set of named arrays for this solid."""
        if not HAS_OCC:
            raise RuntimeError("pythonocc-core / occwl / cadling are required for STEP extraction")

        shape = load_step_shape(self.step_file)
        if self.scale_body:
            shape = scale_shape_to_unit_box(shape)

        mapper = BRepEntityMapper(shape)
        num_faces = mapper.num_faces
        num_edges = mapper.num_edges
        if num_faces == 0 or num_edges == 0:
            raise ValueError(f"No faces/edges extracted from {self.step_file}")

        coedges = CoedgeExtractor().extract_coedges(shape)
        if not coedges:
            raise ValueError(f"No coedges extracted from {self.step_file}")
        num_coedges = len(coedges)

        # Coedge id -> position so next/prev/mate (which reference coedge ids)
        # become 0-based array indices.
        id_to_pos = {c.id: pos for pos, c in enumerate(coedges)}

        coedge_to_next = np.arange(num_coedges, dtype=np.int64)
        coedge_to_prev = np.arange(num_coedges, dtype=np.int64)
        coedge_to_mate = np.arange(num_coedges, dtype=np.int64)
        coedge_to_face = np.zeros(num_coedges, dtype=np.int64)
        coedge_to_edge = np.zeros(num_coedges, dtype=np.int64)
        coedge_reversed = np.zeros(num_coedges, dtype=np.float32)

        for pos, c in enumerate(coedges):
            if c.next_id is not None:
                coedge_to_next[pos] = id_to_pos.get(c.next_id, pos)
            if c.prev_id is not None:
                coedge_to_prev[pos] = id_to_pos.get(c.prev_id, pos)
            if c.mate_id is not None:
                coedge_to_mate[pos] = id_to_pos.get(c.mate_id, pos)
            fi = mapper.face_index(c.face_id)
            ei = mapper.edge_index(c.edge_id)
            coedge_to_face[pos] = fi if fi is not None else 0
            coedge_to_edge[pos] = ei if ei is not None else 0
            coedge_reversed[pos] = 1.0 if c.orientation == "REVERSED" else 0.0

        # Per-face features + grids (index order from the mapper).
        face_features = np.zeros((num_faces, NUM_FACE_FEATURES), dtype=np.float32)
        face_point_grids = np.zeros((num_faces, 7, self.num_u, self.num_v), dtype=np.float32)
        for i in range(num_faces):
            face = mapper.face_by_index(i)
            face_features[i, : len(SURFACE_TYPES)] = face_surface_onehot(face)
            face_features[i, len(SURFACE_TYPES)] = face_area(face)
            face_point_grids[i] = face_uv_grid(face, self.num_u, self.num_v)

        # Adjacent faces per edge (for convexity), from the coedge incidence.
        edge_faces: dict[int, list[int]] = {}
        for pos in range(num_coedges):
            edge_faces.setdefault(int(coedge_to_edge[pos]), [])
            fi = int(coedge_to_face[pos])
            if fi not in edge_faces[int(coedge_to_edge[pos])]:
                edge_faces[int(coedge_to_edge[pos])].append(fi)

        # Per-edge features + grids.
        edge_features = np.zeros((num_edges, NUM_EDGE_FEATURES), dtype=np.float32)
        edge_point_grids = np.zeros((num_edges, 6, self.num_u), dtype=np.float32)
        for j in range(num_edges):
            edge = mapper.edge_by_index(j)
            edge_features[j, : len(CURVE_TYPES)] = edge_curve_onehot(edge)
            edge_features[j, len(CURVE_TYPES)] = edge_length(edge)
            faces_j = edge_faces.get(j, [])
            if len(faces_j) >= 2:
                conv = edge_convexity(
                    edge, mapper.face_by_index(faces_j[0]), mapper.face_by_index(faces_j[1])
                )
            else:
                conv = 0.5
            edge_features[j, len(CURVE_TYPES) + 1] = conv
            edge_point_grids[j] = edge_u_grid(edge, self.num_u)

        return {
            "coedge_to_next": coedge_to_next,
            "coedge_to_prev": coedge_to_prev,
            "coedge_to_mate": coedge_to_mate,
            "coedge_to_face": coedge_to_face,
            "coedge_to_edge": coedge_to_edge,
            "coedge_reversed": coedge_reversed,
            "face_features": face_features,
            "face_point_grids": face_point_grids,
            "edge_features": edge_features,
            "edge_point_grids": edge_point_grids,
            "num_faces": np.int64(num_faces),
            "num_edges": np.int64(num_edges),
            "num_coedges": np.int64(num_coedges),
        }

    def process(self, output_dir: Path) -> Path:
        """Extract arrays and write ``<file_stem>.npz`` into ``output_dir``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays = self.extract_arrays()
        out_path = output_dir / f"{self.step_file.stem}.npz"
        np.savez_compressed(out_path, **arrays)
        _log.info(
            "Wrote %s (F=%d E=%d C=%d)",
            out_path.name,
            int(arrays["num_faces"]),
            int(arrays["num_edges"]),
            int(arrays["num_coedges"]),
        )
        return out_path


def _iter_step_files(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    files: list[Path] = []
    for pattern in ("*.step", "*.stp", "*.STEP", "*.STP"):
        files.extend(sorted(path.glob(pattern)))
    return files


def _extract_one(task: tuple) -> str | None:
    """Worker: extract one STEP file to ``.npz``; returns the path or ``None``.

    Top-level (picklable) so it can run inside a ``ProcessPoolExecutor``.
    """
    step_file, output_path, scale_body, num_u, num_v, force_regeneration = task
    out_npz = Path(output_path) / f"{Path(step_file).stem}.npz"
    if out_npz.exists() and not force_regeneration:
        return str(out_npz)
    try:
        extractor = BRepDataExtractor(
            Path(step_file), scale_body=scale_body, num_u=num_u, num_v=num_v
        )
        return str(extractor.process(Path(output_path)))
    except Exception as exc:
        _log.error("Failed to extract %s: %s", Path(step_file).name, exc)
        return None


def extract_step_files(
    step_files: list[Path],
    output_path: Path,
    scale_body: bool = True,
    num_u: int = NUM_U,
    num_v: int = NUM_V,
    force_regeneration: bool = True,
    num_workers: int = 1,
) -> list[Path]:
    """Extract ``.npz`` records for an explicit list of STEP files.

    Args:
        step_files: The STEP files to extract.
        output_path: Directory to write ``.npz`` records into.
        scale_body: Normalise each solid into the unit box.
        num_u, num_v: UV-grid resolution.
        force_regeneration: Re-extract even if the ``.npz`` already exists.
        num_workers: Parallel worker processes (``>1`` uses a process pool).

    Returns:
        The list of written ``.npz`` paths.
    """
    # Keep each worker single-threaded for OpenMP safety on macOS; children of
    # the process pool inherit this from the parent environment.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(f), str(output_path), scale_body, num_u, num_v, force_regeneration) for f in step_files
    ]

    written: list[Path] = []
    if num_workers and num_workers > 1 and len(tasks) > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
            for res in pool.map(_extract_one, tasks, chunksize=4):
                if res:
                    written.append(Path(res))
    else:
        for task in tasks:
            res = _extract_one(task)
            if res:
                written.append(Path(res))

    _log.info("Extracted %d/%d STEP files", len(written), len(tasks))
    return written


def extract_brepnet_data_from_step(
    step_path: Path,
    output_path: Path,
    scale_body: bool = True,
    num_u: int = NUM_U,
    num_v: int = NUM_V,
    force_regeneration: bool = True,
    num_workers: int = 1,
) -> list[Path]:
    """Extract ``.npz`` records for every STEP file under ``step_path``.

    Args:
        step_path: A STEP file or a directory of STEP files.
        output_path: Directory to write ``.npz`` records into.
        scale_body: Normalise each solid into the unit box.
        num_u, num_v: UV-grid resolution.
        force_regeneration: Re-extract even if the ``.npz`` already exists.
        num_workers: Parallel worker processes (``>1`` uses a process pool).

    Returns:
        The list of written ``.npz`` paths.
    """
    return extract_step_files(
        _iter_step_files(step_path),
        output_path,
        scale_body=scale_body,
        num_u=num_u,
        num_v=num_v,
        force_regeneration=force_regeneration,
        num_workers=num_workers,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract BRepNet-style .npz records from STEP files."
    )
    parser.add_argument("--step", required=True, help="STEP file or directory of STEP files")
    parser.add_argument("--output", required=True, help="Output directory for .npz records")
    parser.add_argument("--no-scale", action="store_true", help="Do not scale to the unit box")
    parser.add_argument("--num-u", type=int, default=NUM_U)
    parser.add_argument("--num-v", type=int, default=NUM_V)
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel worker processes")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    written = extract_brepnet_data_from_step(
        Path(args.step),
        Path(args.output),
        scale_body=not args.no_scale,
        num_u=args.num_u,
        num_v=args.num_v,
        num_workers=args.num_workers,
    )
    print(f"Wrote {len(written)} .npz record(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
