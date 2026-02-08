"""B-spline surface fitter for diffusion-based CAD generation.

Fits B-spline surfaces from dense point grids produced by BrepGen-style
diffusion models. Each face in the generated BRep is represented as a
32x32x3 point grid that must be fit with a smooth B-spline surface, trimmed
by boundary curves, and validated against the original points.

Classes:
    BSplineSurfaceFitter: Main surface fitting class.

Example:
    fitter = BSplineSurfaceFitter(max_degree=3, tolerance=1e-3)

    # Fit a single surface
    result = fitter.fit_surface(point_grid)  # shape (32, 32, 3)
    print(f"Quality: {result['quality']}")

    # Batch fit and validate
    results = fitter.fit_and_validate(point_grids, edge_curves)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)

# Lazy import of pythonocc
_has_pythonocc = False
try:
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakeEdge,
    )
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.Geom import Geom_BSplineSurface
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TColgp import TColgp_Array2OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Wire
    from OCC.Core.GeomAbs import GeomAbs_C2
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    _has_pythonocc = True
    _log.debug("pythonocc-core available for surface fitting")
except ImportError:
    _log.warning(
        "pythonocc-core not available. BSplineSurfaceFitter will operate "
        "in point-cloud-only mode without surface construction."
    )


class BSplineSurfaceFitter:
    """Fit B-spline surfaces from point grids produced by diffusion models.

    BrepGen and similar diffusion-based BRep generators output faces as
    dense point grids (typically 32x32x3). This fitter converts those grids
    into smooth, parametric B-spline surfaces suitable for STEP export.

    The pipeline for each face:
        1. Convert point grid to TColgp_Array2OfPnt
        2. Fit via GeomAPI_PointsToBSplineSurface
        3. Optionally trim with boundary curves
        4. Compute quality metric (Chamfer distance to original points)

    Attributes:
        max_degree: Maximum B-spline degree (default 3 = cubic).
        tolerance: Fitting tolerance for point approximation.
        continuity: Surface continuity class (C0, C1, C2).
        has_pythonocc: Whether pythonocc-core is available.

    Example:
        fitter = BSplineSurfaceFitter(max_degree=3, tolerance=1e-3)
        result = fitter.fit_surface(point_grid)
        if result['valid']:
            face = result['face']
    """

    def __init__(
        self,
        max_degree: int = 3,
        tolerance: float = 1e-3,
        continuity: str = "C2",
    ):
        """Initialize the B-spline surface fitter.

        Args:
            max_degree: Maximum polynomial degree for the B-spline surface.
                Higher degrees allow more flexibility but risk oscillation.
            tolerance: Maximum allowed deviation between the fitted surface
                and the input point grid.
            continuity: Continuity class for the surface ("C0", "C1", "C2").
                C2 is recommended for manufacturing-quality surfaces.
        """
        self.max_degree = max_degree
        self.tolerance = tolerance
        self.continuity = continuity
        self.has_pythonocc = _has_pythonocc

        # Map continuity string to OCC enum
        self._continuity_map = {
            "C0": 0,
            "C1": 1,
            "C2": 2,
        }

        if not self.has_pythonocc:
            _log.warning(
                "BSplineSurfaceFitter initialized without pythonocc. "
                "Surface fitting disabled; only quality metrics available."
            )

    def fit_surface(self, point_grid: np.ndarray) -> Dict[str, Any]:
        """Fit a B-spline surface from a 2D point grid.

        Takes a regularly-sampled point grid (e.g., 32x32x3 from a diffusion
        model) and fits a B-spline surface through the points using
        GeomAPI_PointsToBSplineSurface.

        Args:
            point_grid: NumPy array of shape (M, N, 3) containing 3D points
                sampled on the target surface. Typical shape: (32, 32, 3).

        Returns:
            Dictionary containing:
                - 'surface': OCC Geom_BSplineSurface (or None).
                - 'face': OCC TopoDS_Face (or None).
                - 'valid': Whether the surface was successfully created.
                - 'quality': Chamfer distance to original points.
                - 'max_deviation': Maximum point-to-surface deviation.
                - 'degree_u': Actual degree in U direction.
                - 'degree_v': Actual degree in V direction.
                - 'errors': List of error messages.
        """
        result: Dict[str, Any] = {
            "surface": None,
            "face": None,
            "valid": False,
            "quality": float("inf"),
            "max_deviation": float("inf"),
            "degree_u": 0,
            "degree_v": 0,
            "errors": [],
        }

        # Validate input
        if point_grid.ndim != 3 or point_grid.shape[2] != 3:
            result["errors"].append(
                f"Expected shape (M, N, 3), got {point_grid.shape}"
            )
            return result

        m, n = point_grid.shape[:2]
        if m < 2 or n < 2:
            result["errors"].append(
                f"Grid too small: ({m}, {n}). Minimum 2x2 required."
            )
            return result

        if not self.has_pythonocc:
            # Compute point-cloud-only quality metrics
            result["quality"] = self._compute_grid_smoothness(point_grid)
            result["errors"].append(
                "pythonocc not available; surface not constructed"
            )
            return result

        # Convert numpy grid to OCC point array
        try:
            occ_points = TColgp_Array2OfPnt(1, m, 1, n)
            for i in range(m):
                for j in range(n):
                    x, y, z = point_grid[i, j]
                    occ_points.SetValue(i + 1, j + 1, gp_Pnt(float(x), float(y), float(z)))
        except Exception as e:
            result["errors"].append(f"Point array conversion failed: {e}")
            return result

        # Fit B-spline surface
        try:
            fitter = GeomAPI_PointsToBSplineSurface(
                occ_points,
                self.max_degree,  # DegMin
                self.max_degree,  # DegMax
                GeomAbs_C2,       # Continuity
                self.tolerance,   # Tol3d
            )

            if not fitter.IsDone():
                result["errors"].append("GeomAPI_PointsToBSplineSurface failed")
                return result

            surface = fitter.Surface()
            result["surface"] = surface
            result["degree_u"] = surface.UDegree()
            result["degree_v"] = surface.VDegree()

        except Exception as e:
            result["errors"].append(f"Surface fitting failed: {e}")
            return result

        # Create face from surface
        try:
            face_builder = BRepBuilderAPI_MakeFace(surface, self.tolerance)
            if face_builder.IsDone():
                face = face_builder.Face()
                result["face"] = face
                result["valid"] = True

                # Validate face
                analyzer = BRepCheck_Analyzer(face)
                if not analyzer.IsValid():
                    result["valid"] = False
                    result["errors"].append(
                        "BRepCheck_Analyzer reports invalid face"
                    )
            else:
                result["errors"].append("BRepBuilderAPI_MakeFace failed")
        except Exception as e:
            result["errors"].append(f"Face construction failed: {e}")

        # Compute quality metrics
        try:
            quality, max_dev = self._compute_surface_quality(
                surface, point_grid
            )
            result["quality"] = quality
            result["max_deviation"] = max_dev
        except Exception as e:
            _log.warning("Quality computation failed: %s", e)

        _log.debug(
            "Surface fit: valid=%s, quality=%.4e, max_dev=%.4e, "
            "degree=(%d, %d)",
            result["valid"],
            result["quality"],
            result["max_deviation"],
            result["degree_u"],
            result["degree_v"],
        )

        return result

    def trim_surface(
        self,
        surface: Any,
        boundary_curves: List[Any],
    ) -> Optional[Any]:
        """Trim a B-spline surface with boundary curves.

        Creates a face from the surface bounded by the given edge curves,
        resulting in a trimmed surface patch.

        Args:
            surface: OCC Geom_BSplineSurface to trim.
            boundary_curves: List of OCC edges defining the face boundary.

        Returns:
            TopoDS_Face if trimming succeeds, None otherwise.
        """
        if not self.has_pythonocc:
            _log.error("Cannot trim surface: pythonocc not available")
            return None

        if not boundary_curves:
            _log.warning("No boundary curves provided; returning untrimmed face")
            try:
                face_builder = BRepBuilderAPI_MakeFace(surface, self.tolerance)
                if face_builder.IsDone():
                    return face_builder.Face()
            except Exception as e:
                _log.error("Untrimmed face construction failed: %s", e)
            return None

        # Build wire from boundary curves
        try:
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge in boundary_curves:
                wire_builder.Add(edge)

            if not wire_builder.IsDone():
                _log.warning("Wire construction from boundary curves failed")
                return None

            wire = wire_builder.Wire()

            # Create trimmed face
            face_builder = BRepBuilderAPI_MakeFace(surface, wire, True)
            if face_builder.IsDone():
                face = face_builder.Face()
                _log.debug("Trimmed face created successfully")
                return face
            else:
                _log.warning("Trimmed face construction failed")
                return None

        except Exception as e:
            _log.error("Surface trimming failed: %s", e)
            return None

    def compute_quality(
        self,
        fitted_surface: Any,
        original_points: np.ndarray,
    ) -> float:
        """Compute Chamfer distance between fitted surface and original points.

        Samples points on the fitted surface and computes the symmetric
        Chamfer distance to the original point grid.

        Args:
            fitted_surface: OCC Geom_BSplineSurface.
            original_points: Original point grid of shape (M, N, 3).

        Returns:
            Chamfer distance (lower is better). Returns inf if computation fails.
        """
        if not self.has_pythonocc or fitted_surface is None:
            return float("inf")

        try:
            quality, _ = self._compute_surface_quality(
                fitted_surface, original_points
            )
            return quality
        except Exception as e:
            _log.error("Quality computation failed: %s", e)
            return float("inf")

    def fit_and_validate(
        self,
        point_grids: List[np.ndarray],
        edge_curves: Optional[List[List[Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Batch fit and validate multiple surfaces.

        Processes a list of point grids (one per face), optionally trims
        each with boundary curves, and returns validated results.

        Args:
            point_grids: List of point grids, each of shape (M, N, 3).
            edge_curves: Optional list of edge curve lists for trimming.
                Each entry corresponds to the boundary curves for one face.
                If None, surfaces are created untrimmed.

        Returns:
            List of result dictionaries (one per face), each containing
            the same keys as fit_surface() output plus 'trimmed' flag.
        """
        results: List[Dict[str, Any]] = []

        for i, grid in enumerate(point_grids):
            _log.debug("Fitting surface %d/%d (grid shape: %s)", i + 1, len(point_grids), grid.shape)

            result = self.fit_surface(grid)
            result["face_index"] = i
            result["trimmed"] = False

            # Apply trimming if boundary curves provided
            if (
                edge_curves is not None
                and i < len(edge_curves)
                and edge_curves[i]
                and result["surface"] is not None
            ):
                trimmed_face = self.trim_surface(
                    result["surface"], edge_curves[i]
                )
                if trimmed_face is not None:
                    result["face"] = trimmed_face
                    result["trimmed"] = True
                    _log.debug("Face %d trimmed successfully", i)
                else:
                    _log.warning("Face %d trimming failed; using untrimmed", i)

            results.append(result)

        # Summary statistics
        n_valid = sum(1 for r in results if r["valid"])
        avg_quality = np.mean(
            [r["quality"] for r in results if r["quality"] < float("inf")]
        ) if any(r["quality"] < float("inf") for r in results) else float("inf")

        _log.info(
            "Surface fitting complete: %d/%d valid, avg quality=%.4e",
            n_valid,
            len(results),
            avg_quality,
        )

        return results

    def _compute_surface_quality(
        self,
        surface: Any,
        original_points: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute quality metrics between fitted surface and original points.

        Evaluates the surface at parameter values corresponding to the grid
        and computes both mean and maximum deviation.

        Args:
            surface: OCC Geom_BSplineSurface.
            original_points: Original grid of shape (M, N, 3).

        Returns:
            Tuple of (Chamfer distance, maximum deviation).
        """
        m, n = original_points.shape[:2]

        # Get surface parameter bounds
        u_min = surface.UKnot(1)
        u_max = surface.UKnot(surface.NbUKnots())
        v_min = surface.VKnot(1)
        v_max = surface.VKnot(surface.NbVKnots())

        deviations = []
        for i in range(m):
            for j in range(n):
                # Map grid indices to parameter space
                u = u_min + (i / max(m - 1, 1)) * (u_max - u_min)
                v = v_min + (j / max(n - 1, 1)) * (v_max - v_min)

                # Evaluate surface at (u, v)
                pt_surface = surface.Value(u, v)
                sx, sy, sz = pt_surface.X(), pt_surface.Y(), pt_surface.Z()

                # Original point
                ox, oy, oz = original_points[i, j]

                # Euclidean distance
                dist = math.sqrt(
                    (sx - ox) ** 2 + (sy - oy) ** 2 + (sz - oz) ** 2
                )
                deviations.append(dist)

        deviations_arr = np.array(deviations)
        chamfer = float(np.mean(deviations_arr))
        max_dev = float(np.max(deviations_arr))

        return chamfer, max_dev

    @staticmethod
    def _compute_grid_smoothness(point_grid: np.ndarray) -> float:
        """Compute smoothness metric for point grid (no pythonocc needed).

        Uses finite differences to estimate surface roughness as a proxy
        for fit quality.

        Args:
            point_grid: Point grid of shape (M, N, 3).

        Returns:
            RMS of second-order finite differences (lower = smoother).
        """
        # Second-order central differences in both directions
        d2u = np.diff(point_grid, n=2, axis=0)
        d2v = np.diff(point_grid, n=2, axis=1)

        rms_u = np.sqrt(np.mean(np.sum(d2u ** 2, axis=-1)))
        rms_v = np.sqrt(np.mean(np.sum(d2v ** 2, axis=-1)))

        return float((rms_u + rms_v) / 2.0)
