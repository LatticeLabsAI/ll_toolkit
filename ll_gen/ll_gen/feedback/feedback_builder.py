"""Build structured feedback for neural retry.

Three feedback formats serve three consumers:

1. **Code feedback** (``build_code_feedback``) — LLM-readable error
   message injected into the retry prompt.  Includes the failing code,
   the specific error, and a concrete suggestion.

2. **Neural feedback** (``build_neural_feedback``) — Structured dict
   for VAE/diffusion re-generation.  Contains error category, failed
   entity indices, and parameter adjustment hints.

3. **Training feedback** (``build_training_feedback``) — Dict for
   RL reward shaping during training.  Includes per-finding penalty
   breakdown and a composite reward.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ll_gen.config import ErrorCategory, ErrorSeverity
from ll_gen.proposals.code_proposal import CodeProposal
from ll_gen.proposals.disposal_result import (
    DisposalResult,
    GeometryReport,
    ValidationFinding,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM code feedback
# ---------------------------------------------------------------------------

def build_code_feedback(
    result: DisposalResult,
    original_proposal: CodeProposal,
) -> str:
    """Build an LLM-readable error message for code generation retry.

    The returned string is designed to be injected into the LLM's
    system prompt as error context so it can generate a corrected
    script.  It includes:

    - The original code (truncated to 100 lines if needed)
    - The primary error category and description
    - All critical findings with entity type and suggestion
    - Geometry report summary (if available)
    - An explicit instruction to correct the issue

    Args:
        result: DisposalResult from the failed disposal attempt.
        original_proposal: The CodeProposal that was executed.

    Returns:
        Multi-line string suitable for LLM retry prompting.
    """
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("PREVIOUS ATTEMPT FAILED — CORRECTION REQUIRED")
    lines.append("=" * 60)
    lines.append("")

    # Error summary
    if result.error_category:
        lines.append(f"Error Category: {result.error_category.value}")
    if result.error_message:
        lines.append(f"Error: {result.error_message}")
    lines.append("")

    # Critical findings
    critical = [
        f for f in result.error_details
        if f.severity == ErrorSeverity.CRITICAL
    ]
    if critical:
        lines.append(f"Critical issues ({len(critical)}):")
        for i, finding in enumerate(critical, 1):
            lines.append(
                f"  {i}. [{finding.entity_type} #{finding.entity_index}] "
                f"{finding.error_code}: {finding.description}"
            )
            if finding.suggestion:
                lines.append(f"     Suggestion: {finding.suggestion}")
        lines.append("")

    # Warnings
    warnings = [
        f for f in result.error_details
        if f.severity == ErrorSeverity.WARNING
    ]
    if warnings:
        lines.append(f"Warnings ({len(warnings)}):")
        for finding in warnings:
            lines.append(f"  - {finding.error_code}: {finding.description}")
        lines.append("")

    # Repair attempts
    if result.repair_attempted:
        lines.append("Deterministic repair was attempted:")
        for action in result.repair_actions:
            lines.append(
                f"  - {action.tool}: {action.action} → {action.status}"
            )
        if result.repair_succeeded:
            lines.append("  Repair succeeded but re-check is needed.")
        else:
            lines.append("  Repair did NOT fix the issues.")
        lines.append("")

    # Geometry report
    if result.geometry_report:
        gr = result.geometry_report
        lines.append("Geometry analysis of the (possibly invalid) shape:")
        if gr.bounding_box:
            dims = gr.bbox_dimensions
            if dims:
                lines.append(
                    f"  Bounding box: {dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f}"
                )
        if gr.face_count > 0:
            lines.append(
                f"  Faces: {gr.face_count}, "
                f"Edges: {gr.edge_count}, "
                f"Vertices: {gr.vertex_count}"
            )
        if gr.volume is not None:
            lines.append(f"  Volume: {gr.volume:.2f}")
        if gr.surface_types:
            types_str = ", ".join(
                f"{k}: {v}" for k, v in sorted(gr.surface_types.items())
            )
            lines.append(f"  Surface types: {types_str}")
        lines.append("")

    # The failing code
    if original_proposal.code:
        code_lines = original_proposal.code.strip().splitlines()
        if len(code_lines) > 100:
            code_display = "\n".join(code_lines[:100])
            code_display += f"\n... ({len(code_lines) - 100} more lines)"
        else:
            code_display = original_proposal.code.strip()

        lines.append("The failing code:")
        lines.append("```python")
        lines.append(code_display)
        lines.append("```")
        lines.append("")

    # Explicit correction instruction
    if result.suggestion:
        lines.append(f"CORRECTION: {result.suggestion}")
    else:
        # Generate suggestion from error category
        suggestion = _suggest_from_category(result.error_category, critical)
        if suggestion:
            lines.append(f"CORRECTION: {suggestion}")

    lines.append("")
    lang = getattr(original_proposal, "language", None)
    lang_name = lang.value if lang is not None else "CadQuery"
    lines.append(
        f"Please regenerate a corrected {lang_name} script that avoids "
        "these issues. Ensure all sketch loops are closed, fillet "
        "radii do not exceed edge lengths, and boolean operations "
        "operate on valid solid shapes."
    )

    return "\n".join(lines)


def _suggest_from_category(
    category: Optional[ErrorCategory],
    critical_findings: List[ValidationFinding],
) -> str:
    """Generate a correction suggestion from the error category.

    Args:
        category: Primary error category.
        critical_findings: Critical findings for additional context.

    Returns:
        Actionable suggestion string.
    """
    if category is None:
        return ""

    suggestions = {
        ErrorCategory.INVALID_PARAMS: (
            "Check coordinate values and parameter ranges. "
            "Ensure all points lie on their respective curves/surfaces."
        ),
        ErrorCategory.TOPOLOGY_ERROR: (
            "Ensure all sketch loops are properly closed. "
            "Check that boolean operations produce watertight solids."
        ),
        ErrorCategory.BOOLEAN_FAILURE: (
            "Simplify the boolean operation chain. Try splitting "
            "complex booleans into smaller sequential steps. "
            "Consider increasing the fuzzy tolerance."
        ),
        ErrorCategory.SELF_INTERSECTION: (
            "Check for overlapping sketch primitives. "
            "Reduce fillet/chamfer radii that may cause geometry overlap. "
            "Ensure inner wires (holes) don't overlap outer boundaries."
        ),
        ErrorCategory.DEGENERATE_SHAPE: (
            "Increase minimum feature sizes. "
            "Check for zero-length edges or zero-area faces. "
            "Ensure extrusion height is non-zero."
        ),
        ErrorCategory.TOLERANCE_VIOLATION: (
            "Adjust the precision tier to PRECISION (10-bit). "
            "Or relax tolerance values if the geometry allows it."
        ),
    }

    base = suggestions.get(category, "")

    # Add specific context from findings
    if critical_findings:
        codes = set(f.error_code for f in critical_findings)
        if "BRepCheck_NotClosed" in codes:
            base += " The shell is not closed — check for gaps between faces."
        if "BRepCheck_SelfIntersectingWire" in codes:
            base += " A wire crosses itself — reduce fillet radius or separate overlapping segments."

    return base


# ---------------------------------------------------------------------------
# Neural generation feedback
# ---------------------------------------------------------------------------

def build_neural_feedback(result: DisposalResult) -> Dict[str, Any]:
    """Build structured feedback for neural (VAE/diffusion) retry.

    Returns a dict that the neural generator can use to condition its
    next attempt:

    - ``error_category``: Primary category string.
    - ``failed_entity_indices``: Dict mapping entity type to list of
      failing entity indices.
    - ``parameter_hints``: Adjustment hints per error category.
    - ``topology_stats``: V/E/F counts and euler characteristic.
    - ``severity_counts``: Number of findings per severity level.

    Args:
        result: DisposalResult from the failed disposal attempt.

    Returns:
        Structured feedback dict.
    """
    # Collect failed entity indices by type
    failed_entities: Dict[str, List[int]] = {}
    for finding in result.error_details:
        key = finding.entity_type
        failed_entities.setdefault(key, []).append(finding.entity_index)

    # Deduplicate indices
    for key in failed_entities:
        failed_entities[key] = sorted(set(failed_entities[key]))

    # Severity breakdown
    severity_counts = {
        "critical": sum(
            1 for f in result.error_details
            if f.severity == ErrorSeverity.CRITICAL
        ),
        "warning": sum(
            1 for f in result.error_details
            if f.severity == ErrorSeverity.WARNING
        ),
        "info": sum(
            1 for f in result.error_details
            if f.severity == ErrorSeverity.INFO
        ),
    }

    # Parameter adjustment hints based on error categories
    categories = result.errors_by_category
    hints: Dict[str, str] = {}

    if ErrorCategory.INVALID_PARAMS in categories:
        hints["coordinates"] = "perturb_towards_curves"
    if ErrorCategory.TOPOLOGY_ERROR in categories:
        hints["topology"] = "restructure_sketch_loops"
    if ErrorCategory.BOOLEAN_FAILURE in categories:
        hints["booleans"] = "simplify_or_split"
    if ErrorCategory.SELF_INTERSECTION in categories:
        hints["geometry"] = "separate_overlapping_regions"
    if ErrorCategory.DEGENERATE_SHAPE in categories:
        hints["features"] = "increase_minimum_size"
    if ErrorCategory.TOLERANCE_VIOLATION in categories:
        hints["precision"] = "increase_quantization_bits"

    # Topology stats from geometry report
    topology_stats: Dict[str, Any] = {}
    if result.geometry_report:
        gr = result.geometry_report
        topology_stats = {
            "face_count": gr.face_count,
            "edge_count": gr.edge_count,
            "vertex_count": gr.vertex_count,
            "euler_characteristic": gr.euler_characteristic,
            "is_solid": gr.is_solid,
        }

    return {
        "error_category": (
            result.error_category.value if result.error_category else None
        ),
        "failed_entity_indices": failed_entities,
        "parameter_hints": hints,
        "topology_stats": topology_stats,
        "severity_counts": severity_counts,
        "num_repair_actions": len(result.repair_actions),
        "repair_succeeded": result.repair_succeeded,
    }


# ---------------------------------------------------------------------------
# Training feedback (RL reward shaping)
# ---------------------------------------------------------------------------

def build_training_feedback(result: DisposalResult) -> Dict[str, Any]:
    """Build feedback for RL training reward shaping.

    Returns a dict containing per-finding penalty breakdown, tier
    pass/fail status, and the composite reward signal.

    Args:
        result: DisposalResult from disposal.

    Returns:
        Training feedback dict.
    """
    # Per-category penalty breakdown
    category_penalties: Dict[str, float] = {}
    for cat, findings in result.errors_by_category.items():
        critical_count = sum(
            1 for f in findings if f.severity == ErrorSeverity.CRITICAL
        )
        warning_count = sum(
            1 for f in findings if f.severity == ErrorSeverity.WARNING
        )
        category_penalties[cat.value] = (
            -0.1 * critical_count - 0.05 * warning_count
        )

    # Validation tier pass/fail
    tiers = {
        "shape_constructed": result.has_shape,
        "manifold": _check_manifold(result),
        "watertight": _check_watertight(result),
        "euler_valid": _check_euler(result),
        "no_self_intersection": _check_no_self_intersection(result),
        "repairable": result.repair_succeeded if result.repair_attempted else None,
    }

    # Count passing tiers (excluding None)
    tier_results = {k: v for k, v in tiers.items() if v is not None}
    tiers_passed = sum(1 for v in tier_results.values() if v)
    tiers_total = len(tier_results)

    return {
        "reward_signal": result.reward_signal,
        "category_penalties": category_penalties,
        "validation_tiers": tiers,
        "tiers_passed": tiers_passed,
        "tiers_total": tiers_total,
        "is_valid": result.is_valid,
        "has_shape": result.has_shape,
        "repair_attempted": result.repair_attempted,
        "repair_succeeded": result.repair_succeeded,
    }


def _check_manifold(result: DisposalResult) -> bool:
    """Check if the shape has no non-manifold errors."""
    manifold_codes = {
        "BRepCheck_InvalidMultiConnexity",
        "BRepCheck_FreeEdge",
    }
    return not any(
        f.error_code in manifold_codes for f in result.error_details
    )


def _check_watertight(result: DisposalResult) -> bool:
    """Check if the shape has no watertightness errors."""
    watertight_codes = {
        "BRepCheck_NotClosed",
        "BRepCheck_NotConnected",
    }
    return not any(
        f.error_code in watertight_codes for f in result.error_details
    )


def _check_euler(result: DisposalResult) -> bool:
    """Check if Euler characteristic is valid.

    For genus-0 closed solids (no through-holes), the expected Euler
    characteristic is V - E + F = 2.  However, shapes with cylindrical
    through-holes (common in CAD) have non-genus-0 topology where
    χ = 2 - 2g (g = number of handles/holes).  When the geometry report
    indicates cylindrical surfaces are present, we accept χ != 2 as
    valid since the shape likely has intentional through-holes.
    """
    if result.geometry_report and result.geometry_report.euler_characteristic is not None:
        ec = result.geometry_report.euler_characteristic
        if ec == 2:
            return True
        # Shapes with cylindrical through-holes have χ < 2; accept if
        # surface_types indicates cylinders are present (likely holes).
        surface_types = getattr(result.geometry_report, "surface_types", None)
        if surface_types and "Cylinder" in surface_types:
            _log.debug(
                "Euler characteristic %d != 2 but shape has cylindrical "
                "surfaces — accepting non-genus-0 topology",
                ec,
            )
            return True
        _log.debug(
            "Euler characteristic %d != 2 (genus-0 assumption); no "
            "cylindrical surfaces detected to justify higher genus",
            ec,
        )
        return False
    return False  # Missing report means shape failed validation


def _check_no_self_intersection(result: DisposalResult) -> bool:
    """Check if the shape has no self-intersection errors."""
    si_codes = {
        "BRepCheck_SelfIntersectingWire",
        "BRepCheck_IntersectingWires",
        "BOPAlgo_AlertSelfInterferingShape",
    }
    return not any(
        f.error_code in si_codes for f in result.error_details
    )
