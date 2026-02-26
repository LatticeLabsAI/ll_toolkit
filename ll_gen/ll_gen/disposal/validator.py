"""Full BRepCheck validation with 37-code coverage.

Wraps OpenCASCADE's ``BRepCheck_Analyzer`` and adds higher-level
checks (manifoldness, Euler characteristic, watertightness) that
the raw analyzer doesn't explicitly report.

Returns a ``ValidationReport`` containing per-entity findings
mapped through ``ll_gen.feedback.error_mapper``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ll_gen.config import DisposalConfig, ErrorCategory, ErrorSeverity
from ll_gen.disposal._occ_utils import count_entities as _count_entities
from ll_gen.proposals.disposal_result import ValidationFinding

_log = logging.getLogger(__name__)

_OCC_AVAILABLE = False
try:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopAbs import (
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
    _log.debug("pythonocc not available; validator will not function")


@dataclass
class ValidationReport:
    """Result of shape validation.

    Attributes:
        is_valid: Whether the shape passed all checks.
        findings: Per-entity validation findings.
        face_count: Number of topological faces.
        edge_count: Number of topological edges.
        vertex_count: Number of topological vertices.
        shell_count: Number of shells.
        solid_count: Number of solids.
        is_manifold: Whether every edge is shared by exactly 2 faces.
        is_watertight: Whether all shells are closed.
        euler_characteristic: V − E + F.
        primary_category: The most severe error category, or None.
    """

    is_valid: bool = True
    findings: List[ValidationFinding] = field(default_factory=list)
    face_count: int = 0
    edge_count: int = 0
    vertex_count: int = 0
    shell_count: int = 0
    solid_count: int = 0
    is_manifold: bool = True
    is_watertight: bool = True
    euler_characteristic: Optional[int] = None
    primary_category: Optional[ErrorCategory] = None


def validate_shape(
    shape: Any,
    config: Optional[DisposalConfig] = None,
) -> ValidationReport:
    """Run full validation on a TopoDS_Shape.

    Performs four levels of checking:

    1. **BRepCheck_Analyzer** — runs all 37 OCC checks on every
       sub-shape and maps results through ``error_mapper``.

    2. **Manifoldness** — counts face-sharing for each edge.
       Non-manifold if any edge has ≠ 2 adjacent faces.

    3. **Euler characteristic** — V − E + F for the entire shape.
       Expected value: 2 for genus-0 solids.

    4. **Watertightness** — checks all shells for closedness via
       ``BRepCheck_NotClosed``.

    Args:
        shape: A ``TopoDS_Shape`` to validate.
        config: Disposal config for tolerance settings.

    Returns:
        ValidationReport with all findings.

    Raises:
        ImportError: If pythonocc is not installed.
    """
    if not _OCC_AVAILABLE:
        raise ImportError(
            "pythonocc-core is required for validation. "
            "Install with: conda install -c conda-forge pythonocc-core"
        )

    if config is None:
        config = DisposalConfig()

    report = ValidationReport()

    # --- Count topological entities ---
    report.face_count = _count_entities(shape, TopAbs_FACE)
    report.edge_count = _count_entities(shape, TopAbs_EDGE)
    report.vertex_count = _count_entities(shape, TopAbs_VERTEX)
    report.shell_count = _count_entities(shape, TopAbs_SHELL)
    report.solid_count = _count_entities(shape, TopAbs_SOLID)

    # --- Euler characteristic ---
    report.euler_characteristic = (
        report.vertex_count - report.edge_count + report.face_count
    )

    # --- BRepCheck_Analyzer ---
    analyzer = BRepCheck_Analyzer(shape)
    if not analyzer.IsValid():
        from ll_gen.feedback.error_mapper import map_brep_errors

        mapped_errors = map_brep_errors(analyzer, shape)
        for err in mapped_errors:
            report.findings.append(ValidationFinding(
                entity_type=err.entity_type,
                entity_index=err.entity_index,
                error_code=err.occ_code_name,
                error_category=err.category,
                severity=err.severity,
                description=err.description,
                suggestion=err.suggestion,
            ))

    # --- Manifoldness check ---
    if config.check_manifoldness:
        _check_manifoldness(shape, report)

    # --- Watertightness check ---
    if config.check_watertightness:
        _check_watertightness(shape, report)

    # --- Euler check ---
    if config.check_euler and report.euler_characteristic is not None:
        if report.solid_count == 1 and report.euler_characteristic != 2:
            report.findings.append(ValidationFinding(
                entity_type="SHAPE",
                entity_index=0,
                error_code="EulerViolation",
                error_category=ErrorCategory.TOPOLOGY_ERROR,
                severity=ErrorSeverity.WARNING,
                description=(
                    f"Euler characteristic V−E+F = {report.euler_characteristic} "
                    f"(expected 2 for genus-0 solid; "
                    f"V={report.vertex_count}, E={report.edge_count}, "
                    f"F={report.face_count})"
                ),
                suggestion=(
                    "The shape may have handles/holes (non-zero genus) "
                    "or disconnected components."
                ),
            ))

    # --- Self-intersection check ---
    if config.check_self_intersection:
        _check_self_intersection(shape, report)

    # --- Determine overall validity and primary category ---
    critical_findings = [
        f for f in report.findings
        if f.severity == ErrorSeverity.CRITICAL
    ]
    report.is_valid = len(critical_findings) == 0

    if critical_findings:
        # Primary category is the most common critical category
        cat_counts: Dict[ErrorCategory, int] = {}
        for f in critical_findings:
            cat_counts[f.error_category] = cat_counts.get(f.error_category, 0) + 1
        report.primary_category = max(cat_counts, key=cat_counts.get)

    report.is_manifold = not any(
        f.error_code in ("BRepCheck_InvalidMultiConnexity", "BRepCheck_FreeEdge",
                         "NonManifoldEdge")
        for f in report.findings
    )
    report.is_watertight = not any(
        f.error_code in ("BRepCheck_NotClosed", "BRepCheck_NotConnected",
                         "ShellNotClosed")
        for f in report.findings
    )

    return report


# ---------------------------------------------------------------------------
# Compatibility helpers for pythonocc versions
# ---------------------------------------------------------------------------

def _get_extent(obj: Any) -> int:
    """Get extent/size from OCC collection objects.

    Handles API changes between pythonocc versions where
    Extent() was renamed to Size().
    """
    if hasattr(obj, "Size"):
        return obj.Size()
    elif hasattr(obj, "Extent"):
        return obj.Extent()
    else:
        # Fallback: try to iterate
        return len(list(obj))


# ---------------------------------------------------------------------------
# Manifoldness check
# ---------------------------------------------------------------------------

def _check_manifoldness(shape: Any, report: ValidationReport) -> None:
    """Check that every edge is shared by exactly 2 faces.

    Non-manifold edges (shared by 0, 1, or 3+ faces) indicate
    topological defects.
    """
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCC.Core.TopExp import topexp

    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    for i in range(1, _get_extent(edge_face_map) + 1):
        edge = edge_face_map.FindKey(i)
        face_list = edge_face_map.FindFromIndex(i)
        num_faces = _get_extent(face_list)

        if num_faces == 0:
            report.findings.append(ValidationFinding(
                entity_type="EDGE",
                entity_index=i - 1,
                error_code="FreeEdge_Manifold",
                error_category=ErrorCategory.DEGENERATE_SHAPE,
                severity=ErrorSeverity.WARNING,
                description=f"Edge #{i-1} is not adjacent to any face (isolated edge).",
                suggestion="Remove the isolated edge or connect it to faces.",
            ))
        elif num_faces == 1:
            # Could be boundary edge of an open shell — warning, not error
            report.findings.append(ValidationFinding(
                entity_type="EDGE",
                entity_index=i - 1,
                error_code="BoundaryEdge",
                error_category=ErrorCategory.TOPOLOGY_ERROR,
                severity=ErrorSeverity.WARNING,
                description=f"Edge #{i-1} is shared by only 1 face (open boundary).",
                suggestion="Close the shell by adding missing faces.",
            ))
        elif num_faces > 2:
            report.findings.append(ValidationFinding(
                entity_type="EDGE",
                entity_index=i - 1,
                error_code="NonManifoldEdge",
                error_category=ErrorCategory.TOPOLOGY_ERROR,
                severity=ErrorSeverity.CRITICAL,
                description=(
                    f"Edge #{i-1} is shared by {num_faces} faces "
                    f"(non-manifold; expected exactly 2)."
                ),
                suggestion="Split the edge or adjust face topology to restore manifoldness.",
            ))


# ---------------------------------------------------------------------------
# Watertightness check
# ---------------------------------------------------------------------------

def _check_watertightness(shape: Any, report: ValidationReport) -> None:
    """Check that all shells are closed."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Shell

    shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    shell_idx = 0

    while shell_explorer.More():
        shell = shell_explorer.Current()
        sa = ShapeAnalysis_Shell()
        sa.LoadShells(shell)

        if sa.HasFreeEdges():
            report.findings.append(ValidationFinding(
                entity_type="SHELL",
                entity_index=shell_idx,
                error_code="ShellNotClosed",
                error_category=ErrorCategory.TOPOLOGY_ERROR,
                severity=ErrorSeverity.CRITICAL,
                description=f"Shell #{shell_idx} has free edges (not watertight).",
                suggestion="Close the shell by adding missing faces or sealing gaps.",
            ))

        shell_explorer.Next()
        shell_idx += 1


# ---------------------------------------------------------------------------
# Self-intersection check
# ---------------------------------------------------------------------------

def _check_self_intersection(shape: Any, report: ValidationReport) -> None:
    """Check for shape self-intersection using BOPAlgo_CheckerSI.

    This is a more thorough check than BRepCheck's wire-level
    self-intersection — it detects face-face and solid-solid
    intersections.
    """
    try:
        from OCC.Core.BOPAlgo import BOPAlgo_CheckerSI

        checker = BOPAlgo_CheckerSI()
        checker.SetData(shape)
        checker.Perform()

        if checker.HasErrors():
            report.findings.append(ValidationFinding(
                entity_type="SHAPE",
                entity_index=0,
                error_code="BOPAlgo_SelfIntersection",
                error_category=ErrorCategory.SELF_INTERSECTION,
                severity=ErrorSeverity.CRITICAL,
                description="Shape contains self-intersecting geometry.",
                suggestion=(
                    "Separate overlapping regions. Check that fillet radii "
                    "don't cause adjacent faces to overlap."
                ),
            ))
    except (ImportError, AttributeError):
        # BOPAlgo_CheckerSI may not be available in all OCC builds
        _log.debug("BOPAlgo_CheckerSI not available; skipping self-intersection check")
    except Exception as exc:
        _log.warning("Self-intersection check failed: %s", exc)
