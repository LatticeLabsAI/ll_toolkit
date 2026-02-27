"""ShapeFix auto-repair with tolerance escalation.

OpenCASCADE provides a family of ``ShapeFix_*`` tools that can
automatically repair common topological and geometric defects:

- ``ShapeFix_Shape`` — top-level repair dispatcher
- ``ShapeFix_Wire`` — fix wire topology (gaps, reorder, close)
- ``ShapeFix_Face`` — fix face issues (wire orientation, bounds)
- ``ShapeFix_Shell`` — fix shell orientation and closedness
- ``ShapeFix_Solid`` — ensure solid completeness

For boolean failures, the repairer also retries the failing
boolean with escalating fuzzy tolerance via ``BOPAlgo_PaveFiller``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ll_gen.config import DisposalConfig, ErrorCategory
from ll_gen.disposal.validator import ValidationReport, validate_shape
from ll_gen.proposals.disposal_result import RepairAction

_log = logging.getLogger(__name__)

_OCC_AVAILABLE = False
try:
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.ShapeFix import (
        ShapeFix_Face,
        ShapeFix_Shape,
        ShapeFix_Shell,
        ShapeFix_Solid,
        ShapeFix_Wire,
    )
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import TopoDS_Shape, topods
    from OCC.Core.ShapeBuild import ShapeBuild_ReShape
    from OCC.Core.ShapeExtend import ShapeExtend_DONE4

    _OCC_AVAILABLE = True
except ImportError:
    _log.debug("pythonocc not available; repairer will not function")


@dataclass
class RepairResult:
    """Outcome of a repair attempt.

    Attributes:
        shape: The repaired shape (or original if repair failed).
        succeeded: Whether repair produced a valid shape.
        actions: Ordered list of repair steps taken.
        validation_after: Validation report after repair.
    """

    shape: Any = None
    succeeded: bool = False
    actions: List[RepairAction] = field(default_factory=list)
    validation_after: Optional[ValidationReport] = None


def repair_shape(
    shape: Any,
    validation_report: ValidationReport,
    config: Optional[DisposalConfig] = None,
) -> RepairResult:
    """Attempt deterministic repair of a shape.

    The repair strategy is ordered from least to most aggressive:

    1. ``ShapeFix_Shape`` — general-purpose fixer that dispatches
       to wire/face/shell fixers internally.
    2. Wire-level repair — fixes gaps, reorders edges, closes wires.
    3. Face-level repair — corrects wire orientation and bounds.
    4. Shell-level repair — fixes orientation and closedness.
    5. Solid-level repair — ensures solid completeness.
    6. Boolean tolerance escalation — for boolean failures, retries
       with increasing fuzzy tolerance.

    After each repair pass, the shape is re-validated.  Repair stops
    as soon as the shape becomes valid or the max passes are exhausted.

    Args:
        shape: TopoDS_Shape to repair.
        validation_report: Pre-computed validation report.
        config: Disposal config with repair settings.

    Returns:
        RepairResult with repaired shape and action log.

    Raises:
        ImportError: If pythonocc is not installed.
    """
    if not _OCC_AVAILABLE:
        raise ImportError(
            "pythonocc-core is required for repair. "
            "Install with: conda install -c conda-forge pythonocc-core"
        )

    if config is None:
        config = DisposalConfig()

    result = RepairResult(shape=shape)

    if validation_report.is_valid:
        result.succeeded = True
        result.validation_after = validation_report
        return result

    current_shape = shape
    error_categories = {f.error_category for f in validation_report.findings}

    for pass_num in range(config.max_repair_passes):
        _log.info("Repair pass %d/%d", pass_num + 1, config.max_repair_passes)

        # --- ShapeFix_Shape (general) ---
        current_shape, action = _apply_shapefix_shape(
            current_shape, config
        )
        result.actions.append(action)

        # --- Wire-level fixes ---
        if (ErrorCategory.TOPOLOGY_ERROR in error_categories
                or ErrorCategory.DEGENERATE_SHAPE in error_categories):
            current_shape, wire_actions = _apply_wire_fixes(
                current_shape, config
            )
            result.actions.extend(wire_actions)

        # --- Face-level fixes ---
        if ErrorCategory.TOPOLOGY_ERROR in error_categories:
            current_shape, face_actions = _apply_face_fixes(
                current_shape, config
            )
            result.actions.extend(face_actions)

        # --- Shell-level fixes ---
        if ErrorCategory.TOPOLOGY_ERROR in error_categories:
            current_shape, shell_actions = _apply_shell_fixes(
                current_shape, config
            )
            result.actions.extend(shell_actions)

        # --- Solid-level fixes ---
        current_shape, solid_action = _apply_solid_fix(current_shape)
        if solid_action is not None:
            result.actions.append(solid_action)

        # --- Re-validate ---
        new_report = validate_shape(current_shape, config)

        if new_report.is_valid:
            _log.info("Repair succeeded on pass %d", pass_num + 1)
            result.shape = current_shape
            result.succeeded = True
            result.validation_after = new_report
            return result

        # Update error categories for next pass
        error_categories = {f.error_category for f in new_report.findings}

    # --- Boolean tolerance escalation (last resort) ---
    if ErrorCategory.BOOLEAN_FAILURE in error_categories:
        for tol in config.fuzzy_tolerance_steps:
            current_shape, fuzzy_action = _apply_fuzzy_boolean_retry(
                current_shape, tol
            )
            result.actions.append(fuzzy_action)

            new_report = validate_shape(current_shape, config)
            if new_report.is_valid:
                _log.info("Fuzzy boolean repair succeeded at tolerance %e", tol)
                result.shape = current_shape
                result.succeeded = True
                result.validation_after = new_report
                return result

    # Repair did not fully succeed — return best effort
    result.shape = current_shape
    result.validation_after = validate_shape(current_shape, config)
    result.succeeded = result.validation_after.is_valid

    return result


# ---------------------------------------------------------------------------
# Individual repair tools
# ---------------------------------------------------------------------------

def _apply_shapefix_shape(
    shape: Any,
    config: DisposalConfig,
) -> tuple:
    """Apply ShapeFix_Shape general repair.

    Returns:
        (repaired_shape, RepairAction)
    """
    try:
        fixer = ShapeFix_Shape(shape)
        fixer.SetPrecision(config.shapefix_precision)
        fixer.SetMaxTolerance(config.shapefix_max_tolerance)
        fixer.SetMinTolerance(config.shapefix_min_tolerance)

        fixer.Perform()
        repaired = fixer.Shape()

        action = RepairAction(
            tool="ShapeFix_Shape",
            action="General shape repair (wire/face/shell delegation)",
            status="done",
            tolerance_used=config.shapefix_precision,
        )

        return repaired, action

    except Exception as exc:
        _log.warning("ShapeFix_Shape failed: %s", exc)
        return shape, RepairAction(
            tool="ShapeFix_Shape",
            action=f"Failed: {exc}",
            status="failed",
        )


def _apply_wire_fixes(
    shape: Any,
    config: DisposalConfig,
) -> tuple:
    """Apply ShapeFix_Wire to each wire in the shape.

    Fixes:
    - Reorders edges to form a connected chain
    - Closes gaps between edges
    - Removes degenerated edges
    - Fixes small edges

    Returns:
        (repaired_shape, list of RepairActions)
    """
    actions: List[RepairAction] = []
    reshaper = ShapeBuild_ReShape()
    wire_explorer = TopExp_Explorer(shape, TopAbs_WIRE)
    wires_fixed = 0

    while wire_explorer.More():
        wire = topods.Wire(wire_explorer.Current())
        try:
            fixer = ShapeFix_Wire()
            fixer.Load(wire)
            fixer.SetPrecision(config.shapefix_precision)

            # Fix reorder
            if fixer.FixReorder():
                wires_fixed += 1

            # Fix small edges
            if fixer.FixSmall(True):
                wires_fixed += 1

            # Fix connected
            if fixer.FixConnected():
                wires_fixed += 1

            # Fix degenerated
            if fixer.FixDegenerated():
                wires_fixed += 1

            # Replace old wire with fixed wire in the parent shape
            fixed_wire = fixer.Wire()
            reshaper.Replace(wire, fixed_wire)

        except Exception as exc:
            _log.debug("Wire fix failed: %s", exc)

        wire_explorer.Next()

    result = reshaper.Apply(shape)

    if wires_fixed > 0:
        actions.append(RepairAction(
            tool="ShapeFix_Wire",
            action=f"Fixed {wires_fixed} wire issues (reorder, small, connected, degenerated)",
            status="done",
            tolerance_used=config.shapefix_precision,
            entities_affected=wires_fixed,
        ))

    return result, actions


def _apply_face_fixes(
    shape: Any,
    config: DisposalConfig,
) -> tuple:
    """Apply ShapeFix_Face to each face in the shape.

    Fixes:
    - Wire orientation (outer vs inner)
    - Missing natural bounds
    - Wire addition/removal

    Returns:
        (repaired_shape, list of RepairActions)
    """
    actions: List[RepairAction] = []
    reshaper = ShapeBuild_ReShape()
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces_fixed = 0

    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        try:
            fixer = ShapeFix_Face(face)
            fixer.SetPrecision(config.shapefix_precision)

            fixer.FixOrientation()
            fixer.FixAddNaturalBound()
            fixer.Perform()

            if fixer.Status(ShapeExtend_DONE4):
                faces_fixed += 1
                fixed_face = fixer.Face()
                reshaper.Replace(face, fixed_face)

        except Exception as exc:
            _log.debug("Face fix failed: %s", exc)

        face_explorer.Next()

    result = reshaper.Apply(shape)

    if faces_fixed > 0:
        actions.append(RepairAction(
            tool="ShapeFix_Face",
            action=f"Fixed {faces_fixed} face issues (orientation, bounds)",
            status="done",
            tolerance_used=config.shapefix_precision,
            entities_affected=faces_fixed,
        ))

    return result, actions


def _apply_shell_fixes(
    shape: Any,
    config: DisposalConfig,
) -> tuple:
    """Apply ShapeFix_Shell to each shell in the shape.

    Fixes:
    - Face orientation consistency
    - Shell closedness

    Returns:
        (repaired_shape, list of RepairActions)
    """
    actions: List[RepairAction] = []
    reshaper = ShapeBuild_ReShape()
    shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    shells_fixed = 0

    while shell_explorer.More():
        shell = topods.Shell(shell_explorer.Current())
        try:
            fixer = ShapeFix_Shell(shell)
            fixer.SetPrecision(config.shapefix_precision)
            fixer.Perform()

            if fixer.FixFaceOrientation(shell):
                shells_fixed += 1
                fixed_shell = fixer.Shell()
                reshaper.Replace(shell, fixed_shell)

        except Exception as exc:
            _log.debug("Shell fix failed: %s", exc)

        shell_explorer.Next()

    result = reshaper.Apply(shape)

    if shells_fixed > 0:
        actions.append(RepairAction(
            tool="ShapeFix_Shell",
            action=f"Fixed {shells_fixed} shell orientation issues",
            status="done",
            tolerance_used=config.shapefix_precision,
            entities_affected=shells_fixed,
        ))

    return result, actions


def _apply_solid_fix(shape: Any) -> tuple:
    """Apply ShapeFix_Solid to ensure solid completeness.

    Returns:
        (repaired_shape, RepairAction or None)
    """
    try:
        fixer = ShapeFix_Solid(shape)
        fixer.Perform()
        repaired = fixer.Shape()

        if fixer.Status(ShapeExtend_DONE4):
            return repaired, RepairAction(
                tool="ShapeFix_Solid",
                action="Ensured solid completeness",
                status="done",
            )
        return repaired, None

    except Exception as exc:
        _log.debug("Solid fix failed: %s", exc)
        return shape, None


def _apply_fuzzy_boolean_retry(
    shape: Any,
    fuzzy_tolerance: float,
) -> tuple:
    """Attempt to re-validate with increased fuzzy tolerance.

    For boolean failures, OpenCASCADE's BOPAlgo can retry with
    relaxed intersection tolerances.

    This applies ShapeFix_Shape with the given tolerance, which
    internally adjusts edge/vertex tolerances to heal small gaps
    that cause boolean failures.

    Returns:
        (repaired_shape, RepairAction)
    """
    try:
        fixer = ShapeFix_Shape(shape)
        fixer.SetPrecision(fuzzy_tolerance)
        fixer.SetMaxTolerance(fuzzy_tolerance * 10)
        fixer.SetMinTolerance(fuzzy_tolerance / 10)
        fixer.Perform()
        repaired = fixer.Shape()

        return repaired, RepairAction(
            tool="ShapeFix_Shape_Fuzzy",
            action=f"Fuzzy tolerance repair at {fuzzy_tolerance:.1e}",
            status="done",
            tolerance_used=fuzzy_tolerance,
        )

    except Exception as exc:
        _log.warning("Fuzzy boolean retry failed at %e: %s", fuzzy_tolerance, exc)
        return shape, RepairAction(
            tool="ShapeFix_Shape_Fuzzy",
            action=f"Failed at tolerance {fuzzy_tolerance:.1e}: {exc}",
            status="failed",
            tolerance_used=fuzzy_tolerance,
        )
