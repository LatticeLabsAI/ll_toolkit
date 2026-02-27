"""Surface execution module for LatentProposal objects.

This module executes LatentProposal objects by fitting B-spline surfaces to
decoded point grids and sewing them into closed solids.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ll_gen.proposals.latent_proposal import LatentProposal

# Lazy import pythonocc
try:
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
    from OCC.Core.GeomAPI import (
        GeomAPI_PointsToBSplineSurface,
        GeomAPI_PointsToBSpline,
    )
    from OCC.Core.GeomAbs import GeomAbs_C2
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_Sewing,
    )
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepLProp import BRepLProp_CLProps
    from OCC.Core.TopExp import topexp, TopExp_Explorer
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds

    _OCC_AVAILABLE = True
except ImportError:
    _OCC_AVAILABLE = False

# Lazy import cadling modules
try:
    from cadling.generation.reconstruction.surface_fitter import BSplineSurfaceFitter
    _CADLING_SURFACE_FITTER_AVAILABLE = True
except ImportError:
    _CADLING_SURFACE_FITTER_AVAILABLE = False

try:
    from cadling.generation.reconstruction.topology_merger import TopologyMerger
    _CADLING_TOPOLOGY_MERGER_AVAILABLE = True
except ImportError:
    _CADLING_TOPOLOGY_MERGER_AVAILABLE = False

_log = logging.getLogger(__name__)


def execute_latent_proposal(proposal: LatentProposal) -> Any:
    """Execute a LatentProposal by fitting B-spline surfaces and sewing them.

    This function takes a LatentProposal object, fits B-spline surfaces to each
    decoded face grid, handles edge deduplication through topology merging, and
    creates a closed solid by sewing the faces together.

    Args:
        proposal: A LatentProposal object containing decoded point grids for
            faces and edges.

    Returns:
        A TopoDS_Shape representing the sewed solid.

    Raises:
        RuntimeError: If pythonocc is not available or if surface fitting fails.
    """
    if not _OCC_AVAILABLE:
        raise RuntimeError(
            "pythonocc is required for surface execution. "
            "Please install it using: pip install pythonocc-core"
        )

    _log.info("Starting execution of LatentProposal")

    # Step 1: Fit B-spline surfaces for each face
    _log.info("Step 1: Fitting B-spline surfaces for %d faces", len(proposal.face_grids))
    faces = []
    for i, face_grid in enumerate(proposal.face_grids):
        try:
            face = _fit_bspline_surface(face_grid, tolerance=1e-3)
            faces.append(face)
            _log.debug(f"Fitted B-spline surface for face {i}")
        except Exception as e:
            _log.error(f"Failed to fit B-spline surface for face {i}: {e}")
            raise

    # Step 2: Fit B-spline curves for each edge
    _log.info("Step 2: Fitting B-spline curves for %d edges", len(proposal.edge_points))
    edges = []
    for i, edge_points in enumerate(proposal.edge_points):
        try:
            edge = _fit_bspline_curve(edge_points)
            edges.append(edge)
            _log.debug(f"Fitted B-spline curve for edge {i}")
        except Exception as e:
            _log.error(f"Failed to fit B-spline curve for edge {i}: {e}")
            raise

    # Step 3: Mating deduplication (topology merger)
    _log.info("Step 3: Deduplicating mating edges")
    if _CADLING_TOPOLOGY_MERGER_AVAILABLE:
        _log.info("Using cadling TopologyMerger for topology merging")
        merger = TopologyMerger()
        edges = merger.merge_edges(edges)
    else:
        _log.info("Using built-in edge deduplication")
        edges = _deduplicate_edges(edges)

    # Step 4: Surface trimming (edges bound faces)
    _log.info("Step 4: Trimming surfaces with deduplicated edges")
    trimmed_faces = _trim_surfaces_with_edges(faces, edges)

    # Step 5: Shell sewing
    _log.info("Step 5: Sewing trimmed faces into a closed solid")
    sewed_shape = _sew_faces(trimmed_faces)

    _log.info("LatentProposal execution completed successfully")
    return sewed_shape


def _fit_bspline_surface(grid: np.ndarray, tolerance: float = 1e-3) -> Any:
    """Fit a B-spline surface to a point grid.

    Args:
        grid: Point grid of shape [U, V, 3] where each point is [x, y, z].
        tolerance: Tolerance for the B-spline surface fitting.

    Returns:
        A TopoDS_Face with the fitted B-spline surface.

    Raises:
        RuntimeError: If surface fitting fails.
    """
    if _CADLING_SURFACE_FITTER_AVAILABLE:
        _log.debug("Using cadling BSplineSurfaceFitter")
        fitter = BSplineSurfaceFitter()
        return fitter.fit_surface(grid, tolerance=tolerance)

    if not _OCC_AVAILABLE:
        raise RuntimeError("pythonocc is required for B-spline surface fitting")

    # Convert grid to OCC point array
    u_count, v_count = grid.shape[0], grid.shape[1]
    points = TColgp_Array2OfPnt(1, u_count, 1, v_count)

    for i in range(u_count):
        for j in range(v_count):
            pt = gp_Pnt(float(grid[i, j, 0]), float(grid[i, j, 1]), float(grid[i, j, 2]))
            points.SetValue(i + 1, j + 1, pt)

    # Fit B-spline surface
    try:
        fitter = GeomAPI_PointsToBSplineSurface(points, 3, 3, GeomAbs_C2, 1e-3)
        if not fitter.IsDone():
            raise RuntimeError("B-spline surface fitting failed")

        surface = fitter.Surface()

        # Build face from surface
        face = BRepBuilderAPI_MakeFace(surface, tolerance)
        if not face.IsDone():
            raise RuntimeError("Failed to create face from B-spline surface")

        _log.debug("Successfully fitted B-spline surface")
        return face.Face()

    except Exception as e:
        _log.error(f"Error fitting B-spline surface: {e}")
        raise RuntimeError(f"B-spline surface fitting error: {e}") from e


def _fit_bspline_curve(points: np.ndarray) -> Any:
    """Fit a B-spline curve to a set of points.

    Args:
        points: Array of shape [N, 3] containing curve points [x, y, z].

    Returns:
        A TopoDS_Edge with the fitted B-spline curve.

    Raises:
        RuntimeError: If curve fitting fails.
    """
    if not _OCC_AVAILABLE:
        raise RuntimeError("pythonocc is required for B-spline curve fitting")

    # Convert points to OCC array
    n_points = points.shape[0]
    occ_points = TColgp_Array1OfPnt(1, n_points)

    for i in range(n_points):
        pt = gp_Pnt(float(points[i, 0]), float(points[i, 1]), float(points[i, 2]))
        occ_points.SetValue(i + 1, pt)

    # Fit B-spline curve
    try:
        fitter = GeomAPI_PointsToBSpline(occ_points, 2, 6, GeomAbs_C2, 1e-3)
        if not fitter.IsDone():
            raise RuntimeError("B-spline curve fitting failed")

        curve = fitter.Curve()

        # Build edge from curve
        edge_builder = BRepBuilderAPI_MakeEdge(curve)
        if not edge_builder.IsDone():
            raise RuntimeError("Failed to create edge from B-spline curve")

        _log.debug("Successfully fitted B-spline curve")
        return edge_builder.Edge()

    except Exception as e:
        _log.error(f"Error fitting B-spline curve: {e}")
        raise RuntimeError(f"B-spline curve fitting error: {e}") from e


def _deduplicate_edges(edges: list[Any]) -> list[Any]:
    """Deduplicate edges by merging spatially close and similar edges.

    Compares all edge pairs, computing bounding box distance and shape similarity.
    If bbox_dist < 0.08 AND shape_sim < 0.2, merges edges by averaging vertex
    positions.

    Args:
        edges: List of TopoDS_Edge objects to deduplicate.

    Returns:
        List of deduplicated edges (merged edges are averaged).
    """
    if not _OCC_AVAILABLE:
        raise RuntimeError("pythonocc is required for edge deduplication")

    _log.debug(f"Deduplicating {len(edges)} edges")

    # Store edge data: (edge_object, bbox_center, sampled_points)
    edge_data = []
    for i, edge in enumerate(edges):
        try:
            bbox_center = _compute_edge_bbox_center(edge)
            sampled_pts = _sample_edge_points(edge, n_samples=20)
            edge_data.append({
                'edge': edge,
                'bbox_center': bbox_center,
                'sampled_points': sampled_pts,
                'original_index': i,
                'merged': False,
            })
        except Exception as e:
            _log.warning(f"Failed to extract data from edge {i}: {e}")
            edge_data.append({
                'edge': edge,
                'bbox_center': np.array([0.0, 0.0, 0.0]),
                'sampled_points': np.array([]),
                'original_index': i,
                'merged': False,
            })

    # Spatial binning: bin edges by midpoint into grid cells to avoid O(n^2)
    cell_size = 0.08  # Same as bbox_dist threshold
    bins: dict[tuple[int, int, int], list[int]] = {}
    for idx, data in enumerate(edge_data):
        center = data['bbox_center']
        cell = (
            int(np.floor(center[0] / cell_size)),
            int(np.floor(center[1] / cell_size)),
            int(np.floor(center[2] / cell_size)),
        )
        bins.setdefault(cell, []).append(idx)

    # Identify duplicate edge pairs (only compare within same/adjacent cells)
    merged_pairs = []
    visited_pairs: set[tuple[int, int]] = set()
    for cell, indices in bins.items():
        # Gather candidates from this cell and all 26 neighbors
        candidates = set(indices)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    if neighbor in bins:
                        candidates.update(bins[neighbor])

        sorted_candidates = sorted(candidates)
        for ci, i in enumerate(sorted_candidates):
            if edge_data[i]['merged']:
                continue
            for j in sorted_candidates[ci + 1:]:
                if edge_data[j]['merged']:
                    continue
                pair_key = (i, j)
                if pair_key in visited_pairs:
                    continue
                visited_pairs.add(pair_key)

                # Compute bounding box distance
                bbox_dist = np.linalg.norm(
                    edge_data[i]['bbox_center'] - edge_data[j]['bbox_center']
                )

                # Compute shape similarity (Chamfer distance)
                if edge_data[i]['sampled_points'].size > 0 and edge_data[j]['sampled_points'].size > 0:
                    shape_sim = _chamfer_distance(
                        edge_data[i]['sampled_points'],
                        edge_data[j]['sampled_points'],
                    )
                else:
                    shape_sim = float('inf')

                # Check if edges should be merged
                if bbox_dist < 0.08 and shape_sim < 0.2:
                    _log.debug(
                        f"Merging edges {i} and {j} "
                        f"(bbox_dist={bbox_dist:.6f}, shape_sim={shape_sim:.6f})"
                    )
                    merged_pairs.append((i, j))
                    edge_data[j]['merged'] = True

    # Create merged edges by averaging vertex positions
    deduplicated_edges = []
    for i, data in enumerate(edge_data):
        if data['merged']:
            continue

        # Find all edges that should be merged with this one
        edges_to_merge = [data['edge']]
        for pair_i, pair_j in merged_pairs:
            if pair_i == i:
                edges_to_merge.append(edge_data[pair_j]['edge'])
            elif pair_j == i:
                edges_to_merge.append(edge_data[pair_i]['edge'])

        if len(edges_to_merge) > 1:
            # Average the edges
            merged_edge = _average_edges(edges_to_merge)
            deduplicated_edges.append(merged_edge)
        else:
            deduplicated_edges.append(data['edge'])

    _log.debug(f"Deduplication reduced {len(edges)} edges to {len(deduplicated_edges)}")
    return deduplicated_edges


def _compute_edge_bbox_center(edge: Any) -> np.ndarray:
    """Compute the center of an edge's bounding box.

    Args:
        edge: A TopoDS_Edge object.

    Returns:
        A numpy array [x, y, z] of the bounding box center.
    """
    if not _OCC_AVAILABLE:
        raise RuntimeError("pythonocc is required")

    bbox = Bnd_Box()
    brepbndlib.Add(edge, bbox)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    center = np.array([
        (xmin + xmax) / 2.0,
        (ymin + ymax) / 2.0,
        (zmin + zmax) / 2.0,
    ])

    return center


def _sample_edge_points(edge: Any, n_samples: int = 20) -> np.ndarray:
    """Sample points along an edge.

    Args:
        edge: A TopoDS_Edge object.
        n_samples: Number of points to sample along the edge.

    Returns:
        A numpy array of shape [n_samples, 3] with sampled points.
    """
    if not _OCC_AVAILABLE:
        raise RuntimeError("pythonocc is required")

    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

    adaptor = BRepAdaptor_Curve(edge)
    curve = adaptor.Curve().Curve()

    # Get curve bounds
    u_first = adaptor.FirstParameter()
    u_last = adaptor.LastParameter()

    # Sample points
    points = np.zeros((n_samples, 3), dtype=np.float64)
    for i in range(n_samples):
        u = u_first + (u_last - u_first) * i / (n_samples - 1)
        pt = gp_Pnt()
        curve.D0(u, pt)
        points[i] = np.array([pt.X(), pt.Y(), pt.Z()])

    return points


def _chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Compute the Chamfer distance between two point sets.

    The Chamfer distance is the sum of minimum distances from each point in
    pts_a to pts_b and vice versa, averaged.

    Args:
        pts_a: Array of shape [N, 3].
        pts_b: Array of shape [M, 3].

    Returns:
        The Chamfer distance (float).
    """
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return float('inf')

    # Distance from pts_a to pts_b
    dist_a_to_b = np.min(
        np.linalg.norm(pts_a[:, np.newaxis, :] - pts_b[np.newaxis, :, :], axis=2),
        axis=1,
    )
    avg_a_to_b = np.mean(dist_a_to_b)

    # Distance from pts_b to pts_a
    dist_b_to_a = np.min(
        np.linalg.norm(pts_b[:, np.newaxis, :] - pts_a[np.newaxis, :, :], axis=2),
        axis=1,
    )
    avg_b_to_a = np.mean(dist_b_to_a)

    return (avg_a_to_b + avg_b_to_a) / 2.0


def _average_edges(edges: list[Any]) -> Any:
    """Average multiple edges by averaging their sampled points.

    Args:
        edges: List of TopoDS_Edge objects to average.

    Returns:
        A new TopoDS_Edge representing the averaged curve.
    """
    if not _OCC_AVAILABLE:
        raise RuntimeError("pythonocc is required")

    # Sample points from all edges
    all_sampled_points = []
    for edge in edges:
        sampled_pts = _sample_edge_points(edge, n_samples=20)
        all_sampled_points.append(sampled_pts)

    # Average the sampled points
    all_sampled_points = np.array(all_sampled_points)
    averaged_points = np.mean(all_sampled_points, axis=0)

    # Fit a new curve to the averaged points
    return _fit_bspline_curve(averaged_points)


def _trim_surfaces_with_edges(faces: list[Any], edges: list[Any]) -> list[Any]:
    """Trim B-spline surfaces with deduplicated edges.

    For merged edges that bound a face, creates wires from edge curves and uses
    BRepBuilderAPI_MakeFace to trim the B-spline surface.

    Args:
        faces: List of TopoDS_Face objects.
        edges: List of TopoDS_Edge objects.

    Returns:
        List of trimmed TopoDS_Face objects.
    """
    if not _OCC_AVAILABLE:
        raise RuntimeError("pythonocc is required")

    _log.debug(f"Trimming {len(faces)} surfaces with {len(edges)} edges")

    trimmed_faces = []
    for i, face in enumerate(faces):
        try:
            # Find edges that are close to this face (within tolerance)
            face_edges = []
            for edge in edges:
                try:
                    dist_calc = BRepExtrema_DistShapeShape(face, edge)
                    if dist_calc.IsDone() and dist_calc.Value() < 1e-4:
                        face_edges.append(edge)
                except Exception:
                    continue

            if not face_edges:
                # No matching edges; keep original face
                trimmed_faces.append(face)
                _log.debug(f"Face {i}: no matching edges, keeping original")
                continue

            # Build wire from matching edges
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge in face_edges:
                wire_builder.Add(edge)

            if not wire_builder.IsDone():
                _log.warning(f"Face {i}: wire construction failed, keeping original")
                trimmed_faces.append(face)
                continue

            wire = wire_builder.Wire()

            # Extract surface and create trimmed face
            surface = BRep_Tool.Surface(face)
            face_builder = BRepBuilderAPI_MakeFace(surface, wire, True)

            if face_builder.IsDone():
                trimmed_faces.append(face_builder.Face())
                _log.debug(f"Face {i}: successfully trimmed with {len(face_edges)} edges")
            else:
                _log.warning(f"Face {i}: face construction failed, keeping original")
                trimmed_faces.append(face)

        except Exception as e:
            _log.warning(f"Failed to trim face {i}: {e}")
            trimmed_faces.append(face)

    return trimmed_faces


def _sew_faces(faces: list[Any]) -> Any:
    """Sew faces into a closed solid.

    Collects all faces, creates a BRepBuilderAPI_Sewing, adds each face, calls
    Perform(), and extracts the sewed shape.

    Args:
        faces: List of TopoDS_Face objects to sew.

    Returns:
        The sewed TopoDS_Shape (typically a Shell or Solid).

    Raises:
        RuntimeError: If sewing fails.
    """
    if not _OCC_AVAILABLE:
        raise RuntimeError("pythonocc is required")

    _log.debug(f"Sewing {len(faces)} faces")

    try:
        sewer = BRepBuilderAPI_Sewing()

        # Add each face to the sewer
        for i, face in enumerate(faces):
            sewer.Add(face)
            _log.debug(f"Added face {i} to sewer")

        # Perform sewing
        sewer.Perform()

        # Extract the sewed shape
        sewed_shape = sewer.SewedShape()

        # Check for free/degenerated edges
        _check_sewed_shape_quality(sewed_shape)

        _log.info("Successfully sewed faces into a closed solid")
        return sewed_shape

    except Exception as e:
        _log.error(f"Error sewing faces: {e}")
        raise RuntimeError(f"Face sewing error: {e}") from e


def _check_sewed_shape_quality(shape: Any) -> None:
    """Check the quality of a sewed shape for free or degenerated edges.

    Args:
        shape: A TopoDS_Shape (typically a Shell or Solid).

    Logs warnings if free or degenerated edges are detected.
    """
    if not _OCC_AVAILABLE:
        return

    try:
        # Map all edges in the shape
        edge_map = TopTools_IndexedMapOfShape()
        topexp.MapShapes(shape, TopAbs_EDGE, edge_map)

        free_edges = []
        degenerated_edges = []

        # Detect free edges using ShapeAnalysis_FreeBounds
        try:
            fb = ShapeAnalysis_FreeBounds(shape)
            open_wires = fb.GetOpenWires()

            if not open_wires.IsNull():
                # Count edges in open wires (these are free edges)
                explorer = TopExp_Explorer(open_wires, TopAbs_EDGE)
                idx = 0
                while explorer.More():
                    free_edges.append(idx)
                    idx += 1
                    explorer.Next()
        except Exception as e:
            _log.debug(f"Free bounds analysis failed: {e}")

        # Detect degenerated edges
        for i in range(1, edge_map.Extent() + 1):
            edge = edge_map.FindKey(i)
            if BRep_Tool.Degenerated(edge):
                degenerated_edges.append(i - 1)

        if free_edges:
            _log.warning(f"Sewed shape has {len(free_edges)} free edges")
        if degenerated_edges:
            _log.warning(
                f"Sewed shape has {len(degenerated_edges)} degenerated edges"
            )

    except Exception as e:
        _log.debug(f"Could not check sewed shape quality: {e}")
