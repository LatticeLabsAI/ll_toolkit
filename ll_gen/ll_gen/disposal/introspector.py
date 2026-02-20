"""GeometryReport generation via OCC introspection APIs.

Computes volume, surface area, bounding box, center of mass, inertia
tensor, face/edge/vertex counts, surface and curve type classification,
Euler characteristic, and solid status from a ``TopoDS_Shape``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from ll_gen.proposals.disposal_result import GeometryReport

_log = logging.getLogger(__name__)

_OCC_AVAILABLE = False
try:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GeomAbs import (
        GeomAbs_BezierCurve,
        GeomAbs_BezierSurface,
        GeomAbs_BSplineCurve,
        GeomAbs_BSplineSurface,
        GeomAbs_Circle,
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Ellipse,
        GeomAbs_Hyperbola,
        GeomAbs_Line,
        GeomAbs_OtherCurve,
        GeomAbs_OtherSurface,
        GeomAbs_Parabola,
        GeomAbs_Plane,
        GeomAbs_Sphere,
        GeomAbs_SurfaceOfExtrusion,
        GeomAbs_SurfaceOfRevolution,
        GeomAbs_Torus,
    )
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopAbs import (
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_VERTEX,
    )
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.TopoDS import topods

    _OCC_AVAILABLE = True
except ImportError:
    _log.debug("pythonocc not available; introspector will not function")


# Surface type name mapping
_SURFACE_TYPE_NAMES = {}
if _OCC_AVAILABLE:
    _SURFACE_TYPE_NAMES = {
        GeomAbs_Plane: "Plane",
        GeomAbs_Cylinder: "Cylinder",
        GeomAbs_Cone: "Cone",
        GeomAbs_Sphere: "Sphere",
        GeomAbs_Torus: "Torus",
        GeomAbs_BezierSurface: "BezierSurface",
        GeomAbs_BSplineSurface: "BSplineSurface",
        GeomAbs_SurfaceOfRevolution: "SurfaceOfRevolution",
        GeomAbs_SurfaceOfExtrusion: "SurfaceOfExtrusion",
        GeomAbs_OtherSurface: "OtherSurface",
    }

# Curve type name mapping
_CURVE_TYPE_NAMES = {}
if _OCC_AVAILABLE:
    _CURVE_TYPE_NAMES = {
        GeomAbs_Line: "Line",
        GeomAbs_Circle: "Circle",
        GeomAbs_Ellipse: "Ellipse",
        GeomAbs_Hyperbola: "Hyperbola",
        GeomAbs_Parabola: "Parabola",
        GeomAbs_BezierCurve: "BezierCurve",
        GeomAbs_BSplineCurve: "BSplineCurve",
        GeomAbs_OtherCurve: "OtherCurve",
    }


def introspect(shape: Any) -> GeometryReport:
    """Compute a full GeometryReport for a TopoDS_Shape.

    Uses the following OCC APIs:

    - ``BRepGProp.VolumeProperties`` → volume, center of mass, inertia
    - ``BRepGProp.SurfaceProperties`` → surface area
    - ``BRepBndLib.Add`` → axis-aligned bounding box
    - ``TopExp_Explorer`` → face/edge/vertex/shell/solid counts
    - ``BRepAdaptor_Surface.GetType`` → surface type classification
    - ``BRepAdaptor_Curve.GetType`` → curve type classification

    Args:
        shape: A ``TopoDS_Shape`` to introspect.

    Returns:
        Populated ``GeometryReport``.

    Raises:
        ImportError: If pythonocc is not installed.
    """
    if not _OCC_AVAILABLE:
        raise ImportError(
            "pythonocc-core is required for introspection. "
            "Install with: conda install -c conda-forge pythonocc-core"
        )

    report = GeometryReport()

    # --- Volume properties ---
    try:
        vol_props = GProp_GProps()
        brepgprop.VolumeProperties(shape, vol_props)
        report.volume = vol_props.Mass()

        com = vol_props.CentreOfMass()
        report.center_of_mass = (com.X(), com.Y(), com.Z())

        # Inertia matrix (3x3, stored as flat 9-tuple)
        mat = vol_props.MatrixOfInertia()
        report.inertia_tensor = (
            mat.Value(1, 1), mat.Value(1, 2), mat.Value(1, 3),
            mat.Value(2, 1), mat.Value(2, 2), mat.Value(2, 3),
            mat.Value(3, 1), mat.Value(3, 2), mat.Value(3, 3),
        )
    except Exception as exc:
        _log.debug("Volume properties failed: %s", exc)

    # --- Surface area ---
    try:
        surf_props = GProp_GProps()
        brepgprop.SurfaceProperties(shape, surf_props)
        report.surface_area = surf_props.Mass()
    except Exception as exc:
        _log.debug("Surface properties failed: %s", exc)

    # --- Bounding box ---
    try:
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        if not bbox.IsVoid():
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            report.bounding_box = (xmin, ymin, zmin, xmax, ymax, zmax)
    except Exception as exc:
        _log.debug("Bounding box failed: %s", exc)

    # --- Entity counts ---
    report.face_count = _count_entities(shape, TopAbs_FACE)
    report.edge_count = _count_entities(shape, TopAbs_EDGE)
    report.vertex_count = _count_entities(shape, TopAbs_VERTEX)
    report.shell_count = _count_entities(shape, TopAbs_SHELL)
    report.solid_count = _count_entities(shape, TopAbs_SOLID)

    # --- Euler characteristic ---
    report.euler_characteristic = (
        report.vertex_count - report.edge_count + report.face_count
    )

    # --- Is solid ---
    report.is_solid = report.solid_count > 0

    # --- Surface type classification ---
    report.surface_types = _classify_surfaces(shape)

    # --- Curve type classification ---
    report.curve_types = _classify_curves(shape)

    return report


def _count_entities(shape: Any, topabs_type: Any) -> int:
    """Count unique entities of a given TopAbs type.

    Uses TopTools_IndexedMapOfShape to avoid counting duplicates
    (e.g., an edge shared by two faces should only be counted once).
    """
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopExp import topexp

    entity_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, topabs_type, entity_map)

    # Handle API differences between pythonocc versions
    if hasattr(entity_map, "Size"):
        return entity_map.Size()
    elif hasattr(entity_map, "Extent"):
        return entity_map.Extent()
    else:
        # Fallback to explorer counting
        count = 0
        explorer = TopExp_Explorer(shape, topabs_type)
        while explorer.More():
            count += 1
            explorer.Next()
        return count


def _classify_surfaces(shape: Any) -> Dict[str, int]:
    """Classify all faces by their underlying surface type.

    Returns:
        Dict mapping surface type name to count.
    """
    type_counts: Dict[str, int] = {}
    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    while explorer.More():
        face = topods.Face(explorer.Current())
        try:
            adaptor = BRepAdaptor_Surface(face)
            surface_type = adaptor.GetType()
            type_name = _SURFACE_TYPE_NAMES.get(surface_type, "Unknown")
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        except Exception:
            type_counts["Unknown"] = type_counts.get("Unknown", 0) + 1
        explorer.Next()

    return type_counts


def _classify_curves(shape: Any) -> Dict[str, int]:
    """Classify all edges by their underlying curve type.

    Returns:
        Dict mapping curve type name to count.
    """
    type_counts: Dict[str, int] = {}
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)

    while explorer.More():
        edge = topods.Edge(explorer.Current())
        try:
            adaptor = BRepAdaptor_Curve(edge)
            curve_type = adaptor.GetType()
            type_name = _CURVE_TYPE_NAMES.get(curve_type, "Unknown")
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        except Exception:
            type_counts["Unknown"] = type_counts.get("Unknown", 0) + 1
        explorer.Next()

    return type_counts


def introspect_with_distance(
    shape: Any,
    reference_shape: Any,
) -> Tuple[GeometryReport, float]:
    """Introspect a shape and compute minimum distance to a reference.

    Uses ``BRepExtrema_DistShapeShape`` to find the closest point
    pair between ``shape`` and ``reference_shape``.

    Args:
        shape: Shape to introspect.
        reference_shape: Reference shape for distance computation.

    Returns:
        Tuple of (GeometryReport, minimum_distance).
    """
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

    report = introspect(shape)

    try:
        dist_calc = BRepExtrema_DistShapeShape(shape, reference_shape)
        dist_calc.Perform()

        if dist_calc.IsDone():
            min_dist = dist_calc.Value()
            return report, min_dist
    except Exception as exc:
        _log.warning("Distance computation failed: %s", exc)

    return report, float("inf")
