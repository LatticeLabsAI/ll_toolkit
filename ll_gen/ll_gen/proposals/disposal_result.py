"""Disposal result — complete outcome of deterministic execution.

Every proposal, regardless of type, produces a ``DisposalResult``
after passing through the disposal engine.  This dataclass captures
the shape (if any), validation status, repair history, geometry
report, export paths, and reward signal for RL training.

Supporting dataclasses
----------------------
GeometryReport
    Introspection data computed via ``BRepGProp``, ``BRepBndLib``,
    ``TopExp_Explorer``, and ``ShapeAnalysis``.
ValidationFinding
    Per-entity error from ``BRepCheck_Analyzer``.
RepairAction
    Record of a single ``ShapeFix_*`` repair step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ll_gen.config import ErrorCategory, ErrorSeverity


@dataclass
class GeometryReport:
    """Introspection data for a ``TopoDS_Shape``.

    All measurements are in the model's native unit system.
    Fields are set to ``None`` when the corresponding OCC query
    fails (e.g. volume is undefined for non-solid shapes).

    Attributes:
        volume: Solid volume from ``BRepGProp.VolumeProperties``.
        surface_area: Total surface area from ``BRepGProp.SurfaceProperties``.
        bounding_box: Axis-aligned bbox as
            ``(x_min, y_min, z_min, x_max, y_max, z_max)``.
        center_of_mass: ``(x, y, z)`` from volume properties.
        inertia_tensor: 3×3 inertia matrix as a flat 9-element tuple
            ``(Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz)``.
        face_count: Number of topological faces.
        edge_count: Number of topological edges.
        vertex_count: Number of topological vertices.
        shell_count: Number of shells.
        solid_count: Number of solids.
        surface_types: Mapping from surface type name (e.g. "Plane",
            "Cylinder", "BSplineSurface") to count.
        curve_types: Mapping from curve type name to count.
        euler_characteristic: V − E + F.
        is_solid: Whether the shape is a closed solid.
        oriented_bounding_box: Optional OBB as 15-element tuple
            (center_x, center_y, center_z, half_x, half_y, half_z,
             axis1_x, axis1_y, axis1_z, axis2_x, axis2_y, axis2_z,
             axis3_x, axis3_y, axis3_z).
    """

    volume: Optional[float] = None
    surface_area: Optional[float] = None
    bounding_box: Optional[Tuple[float, ...]] = None
    center_of_mass: Optional[Tuple[float, float, float]] = None
    inertia_tensor: Optional[Tuple[float, ...]] = None
    face_count: int = 0
    edge_count: int = 0
    vertex_count: int = 0
    shell_count: int = 0
    solid_count: int = 0
    surface_types: Dict[str, int] = field(default_factory=dict)
    curve_types: Dict[str, int] = field(default_factory=dict)
    euler_characteristic: Optional[int] = None
    is_solid: bool = False
    oriented_bounding_box: Optional[Tuple[float, ...]] = None

    @property
    def bbox_dimensions(self) -> Optional[Tuple[float, float, float]]:
        """(width, height, depth) from bounding box."""
        if self.bounding_box is None or len(self.bounding_box) < 6:
            return None
        bb = self.bounding_box
        return (bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])

    @property
    def bbox_diagonal(self) -> Optional[float]:
        """Diagonal length of the bounding box."""
        dims = self.bbox_dimensions
        if dims is None:
            return None
        return (dims[0] ** 2 + dims[1] ** 2 + dims[2] ** 2) ** 0.5

    def matches_dimensions(
        self,
        target_dims: Tuple[float, float, float],
        tolerance_pct: float = 0.10,
    ) -> bool:
        """Check if bounding box dimensions match a target within tolerance.

        Compares sorted dimension tuples so axis order doesn't matter.

        Args:
            target_dims: Expected ``(w, h, d)`` in any order.
            tolerance_pct: Fractional tolerance (0.10 = 10%).

        Returns:
            True if all three sorted dimensions are within tolerance.
        """
        dims = self.bbox_dimensions
        if dims is None:
            return False
        actual = sorted(dims)
        expected = sorted(target_dims)
        for a, e in zip(actual, expected):
            if e == 0:
                if abs(a) > tolerance_pct:
                    return False
            elif abs(a - e) / abs(e) > tolerance_pct:
                return False
        return True


@dataclass
class ValidationFinding:
    """A single validation error or warning on a specific sub-shape.

    Attributes:
        entity_type: TopAbs type name ("FACE", "EDGE", "VERTEX", "WIRE",
            "SHELL", "SOLID").
        entity_index: Index of the entity within its type (0-based
            enumeration order from ``TopExp_Explorer``).
        error_code: Raw OCC ``BRepCheck_Status`` name (e.g.
            ``"BRepCheck_NotClosed"``).
        error_category: Mapped neural-interpretable category.
        severity: How critical this finding is.
        description: Human-readable description of the error.
        suggestion: Actionable suggestion for correction.
    """

    entity_type: str = ""
    entity_index: int = 0
    error_code: str = ""
    error_category: ErrorCategory = ErrorCategory.TOPOLOGY_ERROR
    severity: ErrorSeverity = ErrorSeverity.CRITICAL
    description: str = ""
    suggestion: str = ""


@dataclass
class RepairAction:
    """Record of a single deterministic repair step.

    Attributes:
        tool: Which ShapeFix tool was used ("ShapeFix_Shape",
            "ShapeFix_Wire", "ShapeFix_Face", "ShapeFix_Shell",
            "ShapeFix_Solid", "BOPAlgo_PaveFiller").
        action: Description of what was done.
        status: Outcome ("done", "failed", "partial").
        tolerance_used: Tolerance value used for the repair.
        entities_affected: Number of entities modified.
    """

    tool: str = ""
    action: str = ""
    status: str = "done"
    tolerance_used: Optional[float] = None
    entities_affected: int = 0


@dataclass
class DisposalResult:
    """Complete outcome of deterministic proposal execution.

    This is the primary return type of the ``DisposalEngine.dispose()``
    method.  It carries the constructed shape (or None on failure),
    validation findings, repair history, introspection data, export
    paths, and a scalar reward signal for RL training.

    Attributes:
        shape: The constructed ``TopoDS_Shape``, or None if execution
            failed entirely.  Stored as ``Any`` to avoid importing OCC
            at module level.
        is_valid: Whether the shape passed all validation checks.
        error_category: Primary error category if invalid, else None.
        error_details: Per-entity validation findings.
        geometry_report: Introspection data (volume, bbox, etc.).
        repair_attempted: Whether deterministic repair was tried.
        repair_succeeded: Whether repair produced a valid shape.
        repair_actions: Ordered list of repair steps taken.
        reward_signal: Scalar reward for RL training.
            1.0 = valid, 0.0 = complete failure, with partial credit.
        error_message: LLM-readable error description for code retry.
        suggestion: Actionable correction suggestion.
        step_path: Path to exported STEP file (if valid and exported).
        stl_path: Path to exported STL file (if valid and exported).
        render_paths: Paths to multi-view renders (if generated).
        execution_time_ms: Wall-clock time for disposal in milliseconds.
        proposal_id: ID of the proposal that produced this result.
        proposal_type: Class name of the proposal ("CodeProposal", etc.).
        generation_history: ``GenerationHistory`` from the orchestrator retry
            loop, capturing routing decisions, per-attempt telemetry, and
            total wall-clock time.  ``None`` when the result comes from a
            single ``DisposalEngine.dispose()`` call outside the orchestrator.
    """

    shape: Any = None
    is_valid: bool = False
    error_category: Optional[ErrorCategory] = None
    error_details: List[ValidationFinding] = field(default_factory=list)
    geometry_report: Optional[GeometryReport] = None
    repair_attempted: bool = False
    repair_succeeded: bool = False
    repair_actions: List[RepairAction] = field(default_factory=list)
    reward_signal: float = 0.0
    error_message: Optional[str] = None
    suggestion: Optional[str] = None
    step_path: Optional[Path] = None
    stl_path: Optional[Path] = None
    render_paths: Optional[List[Path]] = None
    execution_time_ms: float = 0.0
    proposal_id: str = ""
    proposal_type: str = ""
    generation_history: Any = None

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def has_shape(self) -> bool:
        """Whether any shape was constructed (even if invalid)."""
        return self.shape is not None

    @property
    def has_geometry_report(self) -> bool:
        """Whether introspection data is available."""
        return self.geometry_report is not None

    @property
    def num_errors(self) -> int:
        """Total number of validation findings."""
        return len(self.error_details)

    @property
    def critical_errors(self) -> List[ValidationFinding]:
        """Validation findings with CRITICAL severity."""
        return [
            f for f in self.error_details
            if f.severity == ErrorSeverity.CRITICAL
        ]

    @property
    def num_critical_errors(self) -> int:
        """Count of CRITICAL severity findings."""
        return len(self.critical_errors)

    @property
    def errors_by_category(self) -> Dict[ErrorCategory, List[ValidationFinding]]:
        """Group validation findings by error category."""
        grouped: Dict[ErrorCategory, List[ValidationFinding]] = {}
        for finding in self.error_details:
            grouped.setdefault(finding.error_category, []).append(finding)
        return grouped

    @property
    def was_repaired(self) -> bool:
        """Whether the shape was modified by repair."""
        return self.repair_attempted and self.repair_succeeded

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (shape excluded).

        Useful for JSON logging and training data recording.
        """
        return {
            "is_valid": self.is_valid,
            "error_category": (
                self.error_category.value if self.error_category else None
            ),
            "error_details": [
                {
                    "entity_type": f.entity_type,
                    "entity_index": f.entity_index,
                    "error_code": f.error_code,
                    "error_category": f.error_category.value,
                    "severity": f.severity.value,
                    "description": f.description,
                    "suggestion": f.suggestion,
                }
                for f in self.error_details
            ],
            "geometry_report": (
                {
                    "volume": self.geometry_report.volume,
                    "surface_area": self.geometry_report.surface_area,
                    "bounding_box": self.geometry_report.bounding_box,
                    "center_of_mass": self.geometry_report.center_of_mass,
                    "face_count": self.geometry_report.face_count,
                    "edge_count": self.geometry_report.edge_count,
                    "vertex_count": self.geometry_report.vertex_count,
                    "surface_types": self.geometry_report.surface_types,
                    "euler_characteristic": (
                        self.geometry_report.euler_characteristic
                    ),
                    "is_solid": self.geometry_report.is_solid,
                }
                if self.geometry_report
                else None
            ),
            "repair_attempted": self.repair_attempted,
            "repair_succeeded": self.repair_succeeded,
            "repair_actions": [
                {
                    "tool": a.tool,
                    "action": a.action,
                    "status": a.status,
                    "tolerance_used": a.tolerance_used,
                    "entities_affected": a.entities_affected,
                }
                for a in self.repair_actions
            ],
            "reward_signal": self.reward_signal,
            "error_message": self.error_message,
            "suggestion": self.suggestion,
            "step_path": str(self.step_path) if self.step_path else None,
            "stl_path": str(self.stl_path) if self.stl_path else None,
            "execution_time_ms": self.execution_time_ms,
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type,
        }

    def summary(self) -> Dict[str, Any]:
        """Compact summary for logging."""
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type,
            "is_valid": self.is_valid,
            "has_shape": self.has_shape,
            "num_errors": self.num_errors,
            "num_critical": self.num_critical_errors,
            "error_category": (
                self.error_category.value if self.error_category else None
            ),
            "repair_attempted": self.repair_attempted,
            "repair_succeeded": self.repair_succeeded,
            "reward_signal": self.reward_signal,
            "execution_time_ms": self.execution_time_ms,
        }
