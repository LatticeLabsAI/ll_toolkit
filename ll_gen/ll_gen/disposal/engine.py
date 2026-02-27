"""DisposalEngine — the main deterministic execution entry point.

Orchestrates: execute → validate → repair → introspect → export
for any proposal type (Code, CommandSequence, or Latent).

Every proposal produces a ``DisposalResult`` containing the shape
(or None), validation findings, repair history, geometry report,
export paths, and reward signal.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from ll_gen.config import DisposalConfig, ErrorCategory, ExportConfig, FeedbackConfig
from ll_gen.disposal.exporter import export_step, export_stl
from ll_gen.disposal.introspector import introspect
from ll_gen.disposal.repairer import repair_shape
from ll_gen.disposal.validator import validate_shape
from ll_gen.feedback.error_mapper import categorize_errors
from ll_gen.feedback.reward_signal import compute_reward
from ll_gen.proposals.base import BaseProposal
from ll_gen.proposals.code_proposal import CodeProposal
from ll_gen.proposals.command_proposal import CommandSequenceProposal
from ll_gen.proposals.disposal_result import (
    DisposalResult,
    GeometryReport,
    RepairAction,
    ValidationFinding,
)
from ll_gen.proposals.latent_proposal import LatentProposal

_log = logging.getLogger(__name__)


class DisposalEngine:
    """Deterministic disposal engine for the propose/dispose architecture.

    Accepts typed proposals from the neural layer, executes them
    through the appropriate executor, validates the result, attempts
    repair if invalid, computes introspection data, and exports to
    STEP/STL.

    Args:
        disposal_config: Validation and repair settings.
        export_config: STEP/STL export settings.
        feedback_config: Reward signal weights.
        output_dir: Directory for exported files.

    Example::

        engine = DisposalEngine()
        result = engine.dispose(code_proposal)
        if result.is_valid:
            print(f"Success! STEP at {result.step_path}")
        else:
            print(f"Failed: {result.error_message}")
    """

    def __init__(
        self,
        disposal_config: Optional[DisposalConfig] = None,
        export_config: Optional[ExportConfig] = None,
        feedback_config: Optional[FeedbackConfig] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.disposal_config = disposal_config or DisposalConfig()
        self.export_config = export_config or ExportConfig()
        self.feedback_config = feedback_config or FeedbackConfig()
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def dispose(
        self,
        proposal: BaseProposal,
        export: bool = True,
        render: bool = False,
    ) -> DisposalResult:
        """Execute a proposal through the full disposal pipeline.

        Pipeline stages:

        1. **Execute** — Convert proposal to ``TopoDS_Shape`` via the
           appropriate executor (code, command, or surface).
        2. **Validate** — Run ``BRepCheck_Analyzer`` + manifold + Euler
           + watertight checks.
        3. **Repair** (if invalid) — Apply ``ShapeFix_*`` tools with
           tolerance escalation.
        4. **Introspect** — Compute ``GeometryReport`` (volume, bbox,
           face counts, surface types).
        5. **Export** (if valid) — Write STEP and/or STL files.
        6. **Reward** — Compute scalar reward for RL training.

        Args:
            proposal: A ``CodeProposal``, ``CommandSequenceProposal``,
                or ``LatentProposal``.
            export: Whether to export valid shapes to STEP/STL.
            render: Whether to generate multi-view renders.

        Returns:
            ``DisposalResult`` with complete outcome data.
        """
        start_time = time.monotonic()

        result = DisposalResult(
            proposal_id=proposal.proposal_id,
            proposal_type=type(proposal).__name__,
        )

        # ----------------------------------------------------------
        # Stage 1: Execute
        # ----------------------------------------------------------
        shape = None
        try:
            shape = self._execute(proposal)
            result.shape = shape
            _log.info(
                "Execution succeeded for proposal %s (%s)",
                proposal.proposal_id,
                result.proposal_type,
            )
        except Exception as exc:
            result.error_message = str(exc)
            result.suggestion = _suggest_from_execution_error(exc, proposal)
            _log.warning(
                "Execution failed for proposal %s: %s",
                proposal.proposal_id,
                exc,
            )
            # Compute reward for complete failure
            result.reward_signal = compute_reward(result, self.feedback_config)
            result.execution_time_ms = (time.monotonic() - start_time) * 1000
            return result

        # ----------------------------------------------------------
        # Stage 2: Validate
        # ----------------------------------------------------------
        try:
            val_report = validate_shape(shape, self.disposal_config)

            result.is_valid = val_report.is_valid
            result.error_category = val_report.primary_category

            for finding in val_report.findings:
                result.error_details.append(finding)

            _log.info(
                "Validation: valid=%s, findings=%d, category=%s",
                val_report.is_valid,
                len(val_report.findings),
                val_report.primary_category,
            )
        except ImportError:
            _log.warning("pythonocc not available; skipping validation")
            result.is_valid = False
            result.error_message = "validation unavailable"
            val_report = None
        except Exception as exc:
            _log.warning("Validation failed: %s", exc)
            result.is_valid = False
            result.error_message = "validation unavailable"
            val_report = None

        # ----------------------------------------------------------
        # Stage 3: Repair (if invalid)
        # ----------------------------------------------------------
        if (
            not result.is_valid
            and self.disposal_config.enable_auto_repair
            and val_report is not None
        ):
            try:
                repair_result = repair_shape(
                    shape, val_report, self.disposal_config,
                )
                result.repair_attempted = True
                result.repair_succeeded = repair_result.succeeded
                result.repair_actions = repair_result.actions

                if repair_result.succeeded:
                    shape = repair_result.shape
                    result.shape = shape
                    result.is_valid = True
                    result.error_category = None
                    # Re-validate to update findings
                    if repair_result.validation_after is not None:
                        result.error_details = [
                            f for f in repair_result.validation_after.findings
                        ]
                    _log.info("Repair succeeded for proposal %s", proposal.proposal_id)
                else:
                    _log.info(
                        "Repair attempted but did not produce valid shape "
                        "for proposal %s",
                        proposal.proposal_id,
                    )
            except ImportError:
                _log.warning("pythonocc not available; skipping repair")
            except Exception as exc:
                _log.warning("Repair failed: %s", exc)

        # ----------------------------------------------------------
        # Stage 4: Introspect
        # ----------------------------------------------------------
        if shape is not None and self.disposal_config.always_introspect:
            try:
                result.geometry_report = introspect(shape)
            except ImportError:
                _log.debug("pythonocc not available; skipping introspection")
            except Exception as exc:
                _log.warning("Introspection failed: %s", exc)

        # ----------------------------------------------------------
        # Stage 5: Build error message and suggestion
        # ----------------------------------------------------------
        if not result.is_valid:
            result.error_message = self._build_error_message(result)
            result.suggestion = self._build_suggestion(result)

        # ----------------------------------------------------------
        # Stage 6: Export (if valid and requested)
        # ----------------------------------------------------------
        if result.is_valid and export and shape is not None:
            try:
                step_path = (
                    self.output_dir / f"{proposal.proposal_id}.step"
                )
                result.step_path = export_step(
                    shape, step_path, self.export_config.step_schema,
                )
            except Exception as exc:
                _log.warning("STEP export failed: %s", exc)

            try:
                stl_path = (
                    self.output_dir / f"{proposal.proposal_id}.stl"
                )
                result.stl_path = export_stl(
                    shape,
                    stl_path,
                    self.export_config.stl_linear_deflection,
                    self.export_config.stl_angular_deflection,
                    self.export_config.stl_ascii,
                )
            except Exception as exc:
                _log.warning("STL export failed: %s", exc)

        # ----------------------------------------------------------
        # Stage 6b: Render (if requested)
        # ----------------------------------------------------------
        if render and shape is not None:
            try:
                from ll_gen.disposal.exporter import render_views

                render_dir = self.output_dir / "renders"
                result.render_paths = render_views(
                    shape,
                    render_dir,
                    self.export_config.render_views,
                    self.export_config.render_resolution,
                    prefix=proposal.proposal_id,
                )
            except Exception as exc:
                _log.warning("Rendering failed: %s", exc)

        # ----------------------------------------------------------
        # Stage 7: Compute reward signal
        # ----------------------------------------------------------
        result.reward_signal = compute_reward(result, self.feedback_config)

        result.execution_time_ms = (time.monotonic() - start_time) * 1000
        _log.info(
            "Disposal complete: valid=%s, reward=%.3f, time=%.0fms",
            result.is_valid,
            result.reward_signal,
            result.execution_time_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Executor routing
    # ------------------------------------------------------------------

    def _execute(self, proposal: BaseProposal) -> Any:
        """Route to the appropriate executor based on proposal type.

        Args:
            proposal: Typed proposal.

        Returns:
            TopoDS_Shape.

        Raises:
            TypeError: For unknown proposal types.
            RuntimeError: If execution fails.
        """
        if isinstance(proposal, CodeProposal):
            from ll_gen.disposal.code_executor import execute_code_proposal

            return execute_code_proposal(
                proposal,
                timeout=60 if self.disposal_config.enable_auto_repair else 30,
            )

        elif isinstance(proposal, CommandSequenceProposal):
            from ll_gen.disposal.command_executor import execute_command_proposal

            return execute_command_proposal(proposal)

        elif isinstance(proposal, LatentProposal):
            from ll_gen.disposal.surface_executor import execute_latent_proposal

            return execute_latent_proposal(proposal)

        else:
            raise TypeError(
                f"Unknown proposal type: {type(proposal).__name__}. "
                f"Expected CodeProposal, CommandSequenceProposal, "
                f"or LatentProposal."
            )

    # ------------------------------------------------------------------
    # Error message construction
    # ------------------------------------------------------------------

    def _build_error_message(self, result: DisposalResult) -> str:
        """Build a human-readable error summary."""
        parts = []

        if result.error_category:
            parts.append(f"Primary error: {result.error_category.value}")

        critical = result.critical_errors
        if critical:
            parts.append(f"{len(critical)} critical issue(s):")
            for f in critical[:5]:
                parts.append(f"  - {f.error_code}: {f.description}")
            if len(critical) > 5:
                parts.append(f"  ... and {len(critical) - 5} more")

        if result.repair_attempted:
            if result.repair_succeeded:
                parts.append("Deterministic repair succeeded.")
            else:
                parts.append(
                    f"Deterministic repair attempted ({len(result.repair_actions)} actions) "
                    f"but did not fully resolve the issues."
                )

        return "\n".join(parts) if parts else "Unknown error"

    def _build_suggestion(self, result: DisposalResult) -> str:
        """Build an actionable suggestion based on the error pattern."""
        if not result.error_details:
            return "Inspect the shape manually."

        suggestions = set()
        for finding in result.error_details:
            if finding.suggestion:
                suggestions.add(finding.suggestion)

        if suggestions:
            return " ".join(sorted(suggestions)[:3])

        # Fallback based on category
        category_suggestions = {
            ErrorCategory.INVALID_PARAMS: "Adjust parameter values.",
            ErrorCategory.TOPOLOGY_ERROR: "Ensure sketch loops are closed.",
            ErrorCategory.BOOLEAN_FAILURE: "Simplify boolean operations.",
            ErrorCategory.SELF_INTERSECTION: "Reduce fillet radii.",
            ErrorCategory.DEGENERATE_SHAPE: "Increase feature sizes.",
            ErrorCategory.TOLERANCE_VIOLATION: "Increase precision.",
        }
        if result.error_category:
            return category_suggestions.get(
                result.error_category, "Simplify the geometry."
            )

        return "Simplify the geometry."


def _suggest_from_execution_error(
    exc: Exception,
    proposal: BaseProposal,
) -> str:
    """Generate a suggestion from an execution exception."""
    msg = str(exc).lower()

    if "timeout" in msg:
        return (
            "The script took too long to execute. Simplify the "
            "geometry or reduce the number of boolean operations."
        )
    if "syntax" in msg:
        return "Fix the syntax error in the generated code."
    if "import" in msg:
        return (
            "The generated code uses an unavailable module. "
            "Restrict to cadquery, math, and numpy."
        )
    if "cadquery" in msg or "workplane" in msg:
        return (
            "The CadQuery script failed. Check that all API calls "
            "are valid and that the workplane chain is unbroken."
        )
    if "boolean" in msg or "fuse" in msg or "cut" in msg:
        return (
            "A boolean operation failed. Ensure both operands are "
            "valid solids and try splitting into smaller steps."
        )

    return f"Execution error: {str(exc)[:200]}"
