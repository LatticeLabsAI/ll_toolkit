"""Map OpenCASCADE BRepCheck error codes to neural-interpretable categories.

OpenCASCADE's ``BRepCheck_Analyzer`` reports 37 distinct error codes
across faces, edges, wires, shells, and solids.  Neural generators
(LLMs or RL agents) cannot act on this granularity.

This module collapses the 37 codes into 6 categories that map to
actionable neural corrections:

==============================  ====================================
Category                        Neural action
==============================  ====================================
``INVALID_PARAMS``              Adjust parameter values
``TOPOLOGY_ERROR``              Restructure command sequence
``BOOLEAN_FAILURE``             Simplify booleans, adjust tolerances
``SELF_INTERSECTION``           Separate overlapping geometry
``DEGENERATE_SHAPE``            Increase feature sizes
``TOLERANCE_VIOLATION``         Adjust precision tier
==============================  ====================================

Each mapping also carries a severity level and an LLM-readable
suggestion string.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ll_gen.config import ErrorCategory, ErrorSeverity

_log = logging.getLogger(__name__)

# Lazy import flag for pythonocc
_OCC_AVAILABLE = False
try:
    from OCC.Core.BRepCheck import (
        BRepCheck_Analyzer,
        BRepCheck_BadOrientation,
        BRepCheck_CheckFail,
        BRepCheck_EmptyShell,
        BRepCheck_EmptyWire,
        BRepCheck_EnclosedRegion,
        BRepCheck_FreeEdge,
        BRepCheck_IntersectingWires,
        BRepCheck_InvalidDegeneratedFlag,
        BRepCheck_InvalidImbricationOfShells,
        BRepCheck_InvalidImbricationOfWires,
        BRepCheck_InvalidMultiConnexity,
        BRepCheck_InvalidPointOnCurve,
        BRepCheck_InvalidPointOnCurveOnSurface,
        BRepCheck_InvalidPointOnSurface,
        BRepCheck_InvalidPolygonOnTriangulation,
        BRepCheck_InvalidRange,
        BRepCheck_InvalidSameParameterFlag,
        BRepCheck_InvalidSameRangeFlag,
        BRepCheck_InvalidToleranceValue,
        BRepCheck_InvalidWire,
        BRepCheck_NoError,
        BRepCheck_NoSurface,
        BRepCheck_NotClosed,
        BRepCheck_NotConnected,
        BRepCheck_OrientationOfExternalWire,
        BRepCheck_RedundantEdge,
        BRepCheck_RedundantFace,
        BRepCheck_RedundantWire,
        BRepCheck_SelfIntersectingWire,
        BRepCheck_SubshapeNotInShape,
        BRepCheck_UnorientableShape,
    )
    from OCC.Core.TopAbs import (
        TopAbs_COMPOUND,
        TopAbs_COMPSOLID,
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_VERTEX,
        TopAbs_WIRE,
    )
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import TopoDS_Shape

    _OCC_AVAILABLE = True
except ImportError:
    _log.debug("pythonocc not available; error_mapper will use string-based lookup")


# ---------------------------------------------------------------------------
# Mapped error record
# ---------------------------------------------------------------------------

@dataclass
class MappedError:
    """A single OCC error mapped to a neural-interpretable category.

    Attributes:
        occ_code_name: OCC BRepCheck_Status enum name string.
        occ_code_value: Integer value of the OCC enum.
        category: Neural-interpretable error category.
        severity: How critical the error is.
        description: Human-readable error description.
        suggestion: Actionable suggestion for correction.
        entity_type: TopAbs type of the failing entity.
        entity_index: Enumeration index of the failing entity.
    """

    occ_code_name: str = ""
    occ_code_value: int = -1
    category: ErrorCategory = ErrorCategory.TOPOLOGY_ERROR
    severity: ErrorSeverity = ErrorSeverity.CRITICAL
    description: str = ""
    suggestion: str = ""
    entity_type: str = ""
    entity_index: int = 0


# ---------------------------------------------------------------------------
# The master mapping table
# ---------------------------------------------------------------------------

# Format:  OCC_CODE_NAME -> (ErrorCategory, ErrorSeverity, description, suggestion)
OCC_ERROR_MAP: Dict[str, Tuple[ErrorCategory, ErrorSeverity, str, str]] = {
    # --- INVALID_PARAMS: parameter/coordinate errors ---
    "BRepCheck_InvalidPointOnCurve": (
        ErrorCategory.INVALID_PARAMS,
        ErrorSeverity.CRITICAL,
        "Vertex does not lie on its edge curve within tolerance.",
        "Adjust vertex coordinates or increase edge curve tolerance.",
    ),
    "BRepCheck_InvalidPointOnCurveOnSurface": (
        ErrorCategory.INVALID_PARAMS,
        ErrorSeverity.CRITICAL,
        "Point on 2D curve (pcurve) does not project to correct 3D position on surface.",
        "Recompute pcurve or adjust surface parameterization.",
    ),
    "BRepCheck_InvalidPointOnSurface": (
        ErrorCategory.INVALID_PARAMS,
        ErrorSeverity.CRITICAL,
        "Edge curve does not lie on its face surface within tolerance.",
        "Adjust edge curve or surface definition to restore coincidence.",
    ),
    "BRepCheck_InvalidRange": (
        ErrorCategory.INVALID_PARAMS,
        ErrorSeverity.WARNING,
        "Parameter range of an edge is invalid (e.g. first > last).",
        "Swap or correct the parameter range bounds.",
    ),
    "BRepCheck_InvalidSameParameterFlag": (
        ErrorCategory.TOLERANCE_VIOLATION,
        ErrorSeverity.WARNING,
        "SameParameter flag is set but 3D and 2D representations diverge beyond tolerance.",
        "Recompute SameParameter or increase tolerance.",
    ),
    "BRepCheck_InvalidSameRangeFlag": (
        ErrorCategory.TOLERANCE_VIOLATION,
        ErrorSeverity.WARNING,
        "SameRange flag is set but parameter ranges of 3D and 2D curves differ.",
        "Recompute SameRange or adjust parameter bounds.",
    ),
    "BRepCheck_InvalidDegeneratedFlag": (
        ErrorCategory.DEGENERATE_SHAPE,
        ErrorSeverity.WARNING,
        "Edge is marked degenerate but has non-zero geometric length, or vice versa.",
        "Correct the degenerated flag or increase minimum feature size.",
    ),

    # --- TOPOLOGY_ERROR: structural/connectivity errors ---
    "BRepCheck_NotClosed": (
        ErrorCategory.TOPOLOGY_ERROR,
        ErrorSeverity.CRITICAL,
        "Shell or wire is not closed (watertightness failure).",
        "Ensure all sketch loops are closed before extrusion. Check for gaps between edges.",
    ),
    "BRepCheck_NotConnected": (
        ErrorCategory.TOPOLOGY_ERROR,
        ErrorSeverity.CRITICAL,
        "Wire or shell is disconnected — contains isolated segments.",
        "Ensure continuous edge chains. Remove floating edges or reconnect.",
    ),
    "BRepCheck_BadOrientation": (
        ErrorCategory.TOPOLOGY_ERROR,
        ErrorSeverity.CRITICAL,
        "Face normal orientation is inconsistent with surrounding faces.",
        "Reverse face orientation or restructure the shell topology.",
    ),
    "BRepCheck_SubshapeNotInShape": (
        ErrorCategory.TOPOLOGY_ERROR,
        ErrorSeverity.CRITICAL,
        "A sub-shape referenced by the topology is not actually part of the shape.",
        "Rebuild the topology to include all referenced sub-shapes.",
    ),
    "BRepCheck_OrientationOfExternalWire": (
        ErrorCategory.TOPOLOGY_ERROR,
        ErrorSeverity.WARNING,
        "External wire of a face has incorrect orientation.",
        "Reverse the outer wire orientation.",
    ),
    "BRepCheck_UnorientableShape": (
        ErrorCategory.TOPOLOGY_ERROR,
        ErrorSeverity.CRITICAL,
        "Shape cannot be consistently oriented (Möbius-like topology).",
        "Simplify the geometry to avoid non-orientable configurations.",
    ),
    "BRepCheck_InvalidImbricationOfShells": (
        ErrorCategory.TOPOLOGY_ERROR,
        ErrorSeverity.CRITICAL,
        "Shells are improperly nested or overlapping.",
        "Separate overlapping shells or merge them via boolean union.",
    ),
    "BRepCheck_InvalidImbricationOfWires": (
        ErrorCategory.TOPOLOGY_ERROR,
        ErrorSeverity.WARNING,
        "Wires within a face are improperly nested.",
        "Reorder inner/outer wires or check hole placement.",
    ),

    # --- BOOLEAN_FAILURE: BOPAlgo and boolean operation errors ---
    "BRepCheck_CheckFail": (
        ErrorCategory.BOOLEAN_FAILURE,
        ErrorSeverity.CRITICAL,
        "Generic BRepCheck failure — often caused by failed boolean operation.",
        "Simplify boolean operations, split into smaller steps, or increase fuzzy tolerance.",
    ),

    # --- SELF_INTERSECTION: geometric overlap errors ---
    "BRepCheck_SelfIntersectingWire": (
        ErrorCategory.SELF_INTERSECTION,
        ErrorSeverity.CRITICAL,
        "Wire (edge loop) crosses itself.",
        "Separate overlapping sketch segments. Reduce fillet radius if it causes overlap.",
    ),
    "BRepCheck_IntersectingWires": (
        ErrorCategory.SELF_INTERSECTION,
        ErrorSeverity.CRITICAL,
        "Two wires on the same face intersect each other.",
        "Ensure inner wires (holes) do not overlap with outer boundary or each other.",
    ),
    "BRepCheck_EnclosedRegion": (
        ErrorCategory.SELF_INTERSECTION,
        ErrorSeverity.WARNING,
        "A wire encloses a region that overlaps with another wire's region.",
        "Adjust wire positions to eliminate overlapping enclosed regions.",
    ),

    # --- DEGENERATE_SHAPE: zero-measure or empty entities ---
    "BRepCheck_EmptyWire": (
        ErrorCategory.DEGENERATE_SHAPE,
        ErrorSeverity.CRITICAL,
        "Wire contains no edges.",
        "Add at least one sketch primitive (LINE, ARC, CIRCLE) to the sketch loop.",
    ),
    "BRepCheck_EmptyShell": (
        ErrorCategory.DEGENERATE_SHAPE,
        ErrorSeverity.CRITICAL,
        "Shell contains no faces.",
        "Ensure the extrusion operation produces at least one face.",
    ),
    "BRepCheck_NoSurface": (
        ErrorCategory.DEGENERATE_SHAPE,
        ErrorSeverity.CRITICAL,
        "Face has no underlying geometric surface.",
        "Assign a valid surface definition to the face.",
    ),
    "BRepCheck_RedundantEdge": (
        ErrorCategory.DEGENERATE_SHAPE,
        ErrorSeverity.INFO,
        "Edge appears multiple times in the same wire.",
        "Remove duplicate edges from the wire.",
    ),
    "BRepCheck_RedundantWire": (
        ErrorCategory.DEGENERATE_SHAPE,
        ErrorSeverity.INFO,
        "Duplicate wire in a face.",
        "Remove the redundant wire definition.",
    ),
    "BRepCheck_RedundantFace": (
        ErrorCategory.DEGENERATE_SHAPE,
        ErrorSeverity.INFO,
        "Duplicate face in a shell.",
        "Remove the redundant face definition.",
    ),
    "BRepCheck_InvalidWire": (
        ErrorCategory.DEGENERATE_SHAPE,
        ErrorSeverity.CRITICAL,
        "Wire is structurally invalid (edges don't form a connected chain).",
        "Reconnect edges to form a continuous closed loop.",
    ),
    "BRepCheck_InvalidMultiConnexity": (
        ErrorCategory.DEGENERATE_SHAPE,
        ErrorSeverity.WARNING,
        "Edge is shared by more than 2 faces (non-manifold).",
        "Split the edge or adjust face topology to restore manifoldness.",
    ),
    "BRepCheck_FreeEdge": (
        ErrorCategory.DEGENERATE_SHAPE,
        ErrorSeverity.WARNING,
        "Edge belongs to only one face (free edge on shell boundary).",
        "Close the shell by adding missing faces, or remove the free edge.",
    ),

    # --- TOLERANCE_VIOLATION: precision and consistency errors ---
    "BRepCheck_InvalidToleranceValue": (
        ErrorCategory.TOLERANCE_VIOLATION,
        ErrorSeverity.WARNING,
        "A tolerance value is out of the valid range.",
        "Adjust the precision tier or explicitly set tolerance bounds.",
    ),
    "BRepCheck_InvalidPolygonOnTriangulation": (
        ErrorCategory.TOLERANCE_VIOLATION,
        ErrorSeverity.INFO,
        "Polygon on triangulation is inconsistent with the edge curve.",
        "Recompute the mesh triangulation for the affected edge.",
    ),

    # --- BOPAlgo (boolean) specific status codes ---
    "BOPAlgo_AlertTooFewArguments": (
        ErrorCategory.BOOLEAN_FAILURE,
        ErrorSeverity.CRITICAL,
        "Boolean operation received fewer than 2 arguments.",
        "Ensure both tool and object shapes are provided to the boolean op.",
    ),
    "BOPAlgo_AlertBOPNotAllowed": (
        ErrorCategory.BOOLEAN_FAILURE,
        ErrorSeverity.CRITICAL,
        "Boolean operation type is not allowed for the given shape types.",
        "Check that both shapes are solids (not shells or wires).",
    ),
    "BOPAlgo_AlertSelfInterferingShape": (
        ErrorCategory.SELF_INTERSECTION,
        ErrorSeverity.CRITICAL,
        "One of the boolean operands is self-intersecting.",
        "Fix the self-intersecting shape before attempting boolean operations.",
    ),
    "BOPAlgo_AlertTooSmallEdge": (
        ErrorCategory.DEGENERATE_SHAPE,
        ErrorSeverity.WARNING,
        "Boolean operation produced an edge shorter than tolerance.",
        "Increase minimum feature size or merge nearby vertices.",
    ),
    "BOPAlgo_AlertNullInputShapes": (
        ErrorCategory.BOOLEAN_FAILURE,
        ErrorSeverity.CRITICAL,
        "One or both boolean input shapes are null.",
        "Verify that all shape construction steps succeeded before boolean.",
    ),
}


# ---------------------------------------------------------------------------
# Mapping functions
# ---------------------------------------------------------------------------

def map_single_error(
    occ_code_name: str,
    entity_type: str = "",
    entity_index: int = 0,
) -> MappedError:
    """Map a single OCC error code to a MappedError.

    Args:
        occ_code_name: String name of the BRepCheck_Status or BOPAlgo
            status code (e.g. ``"BRepCheck_NotClosed"``).
        entity_type: TopAbs type name of the entity with the error.
        entity_index: Enumeration index of the failing entity.

    Returns:
        MappedError with category, severity, description, suggestion.
        Falls back to TOPOLOGY_ERROR/CRITICAL for unknown codes.
    """
    if occ_code_name in OCC_ERROR_MAP:
        category, severity, description, suggestion = OCC_ERROR_MAP[occ_code_name]
    else:
        _log.warning("Unknown OCC error code: %s — defaulting to TOPOLOGY_ERROR", occ_code_name)
        category = ErrorCategory.TOPOLOGY_ERROR
        severity = ErrorSeverity.CRITICAL
        description = f"Unknown BRepCheck error: {occ_code_name}"
        suggestion = "Inspect the shape manually and simplify the geometry."

    return MappedError(
        occ_code_name=occ_code_name,
        occ_code_value=-1,
        category=category,
        severity=severity,
        description=description,
        suggestion=suggestion,
        entity_type=entity_type,
        entity_index=entity_index,
    )


def map_brep_errors(
    analyzer: Any,
    shape: Any,
) -> List[MappedError]:
    """Walk a BRepCheck_Analyzer result and map all errors.

    Iterates over every sub-shape in the input shape using
    ``TopExp_Explorer`` for each TopAbs type (SOLID, SHELL, FACE,
    WIRE, EDGE, VERTEX).  For each sub-shape, queries the analyzer
    for its status list and maps each non-NoError status.

    Args:
        analyzer: ``BRepCheck_Analyzer`` instance (already performed).
        shape: ``TopoDS_Shape`` that was analyzed.

    Returns:
        List of ``MappedError`` for all failing sub-shapes.

    Raises:
        ImportError: If pythonocc is not installed.
    """
    if not _OCC_AVAILABLE:
        raise ImportError(
            "pythonocc-core is required for map_brep_errors. "
            "Install with: conda install -c conda-forge pythonocc-core"
        )

    # TopAbs type constants and their string names
    topabs_types = [
        (TopAbs_SOLID, "SOLID"),
        (TopAbs_SHELL, "SHELL"),
        (TopAbs_FACE, "FACE"),
        (TopAbs_WIRE, "WIRE"),
        (TopAbs_EDGE, "EDGE"),
        (TopAbs_VERTEX, "VERTEX"),
    ]

    # Reverse lookup from OCC enum object to string name
    _status_name_cache: Dict[int, str] = {}

    def _get_status_name(status_obj: Any) -> str:
        """Convert an OCC BRepCheck_Status enum to its string name."""
        val = int(status_obj)
        if val in _status_name_cache:
            return _status_name_cache[val]

        # Try to get name from the module's attributes
        import OCC.Core.BRepCheck as brep_mod
        for attr_name in dir(brep_mod):
            if attr_name.startswith("BRepCheck_"):
                try:
                    attr_val = getattr(brep_mod, attr_name)
                    if int(attr_val) == val:
                        _status_name_cache[val] = attr_name
                        return attr_name
                except (TypeError, ValueError):
                    continue

        name = f"BRepCheck_Unknown_{val}"
        _status_name_cache[val] = name
        return name

    errors: List[MappedError] = []

    # Build a lookup of known BRepCheck/BOPAlgo status enums once
    import OCC.Core.BRepCheck as brep_mod

    known_statuses: List[Tuple[str, Any]] = []
    for code_name in OCC_ERROR_MAP:
        enum_obj = getattr(brep_mod, code_name, None)
        if enum_obj is not None:
            known_statuses.append((code_name, enum_obj))

    for topabs_type, type_name in topabs_types:
        explorer = TopExp_Explorer(shape, topabs_type)
        entity_idx = 0

        while explorer.More():
            sub_shape = explorer.Current()
            result = analyzer.Result(sub_shape)

            if result is not None:
                # Check each known error status against the result
                for code_name, status_enum in known_statuses:
                    try:
                        is_present = result.IsStatusOnShape(status_enum)
                        if is_present:
                            mapped = map_single_error(
                                code_name, type_name, entity_idx
                            )
                            mapped.occ_code_value = int(status_enum)
                            errors.append(mapped)
                    except Exception:
                        _log.debug(
                            "Could not check status %s on %s #%d",
                            code_name, type_name, entity_idx,
                            exc_info=True,
                        )
                        continue

            explorer.Next()
            entity_idx += 1

    return errors


def categorize_errors(
    mapped_errors: List[MappedError],
) -> Dict[ErrorCategory, List[MappedError]]:
    """Group mapped errors by their neural-interpretable category.

    Args:
        mapped_errors: List of MappedError instances.

    Returns:
        Dict from ErrorCategory to list of errors in that category,
        sorted by severity (CRITICAL first).
    """
    grouped: Dict[ErrorCategory, List[MappedError]] = {}
    for err in mapped_errors:
        grouped.setdefault(err.category, []).append(err)

    # Sort each group: CRITICAL first, then WARNING, then INFO
    severity_order = {
        ErrorSeverity.CRITICAL: 0,
        ErrorSeverity.WARNING: 1,
        ErrorSeverity.INFO: 2,
    }
    for cat in grouped:
        grouped[cat].sort(key=lambda e: severity_order.get(e.severity, 3))

    return grouped
