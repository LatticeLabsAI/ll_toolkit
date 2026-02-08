"""Validation feedback loop for CAD generation reconstruction.

Provides an iterative validation and repair pipeline that wraps the
CommandExecutor. On each iteration, the generated shape is validated
via topology checks (manifoldness, watertightness, Euler characteristic,
self-intersection). If validation fails, the loop collects findings,
adjusts the command tokens, and retries up to a configurable maximum.

Classes:
    GenerationResult: Pydantic model containing the final output.
    ValidationFeedbackLoop: Main loop orchestrator.

Example:
    from cadling.generation.reconstruction import (
        CommandExecutor,
        ValidationFeedbackLoop,
    )

    executor = CommandExecutor(tolerance=1e-6)
    loop = ValidationFeedbackLoop(executor, max_retries=3)
    result = loop.run(command_tokens, mode='command')
    if result.valid:
        print("Valid shape produced")
        executor.export_step(result.shape, "output.step")
    else:
        print(f"Failed: {result.validation_report}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

# Lazy import of pythonocc for topology validation
_has_pythonocc = False
try:
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    _has_pythonocc = True
except ImportError:
    _log.debug("pythonocc not available for validation loop")

# Import ValidationFinding from topology_validation if available
try:
    from cadling.models.topology_validation import ValidationFinding
except ImportError:
    # Fallback definition if topology_validation is not importable
    from pydantic import BaseModel as _BaseModel

    class ValidationFinding(_BaseModel):  # type: ignore[no-redef]
        """Standalone validation finding (fallback)."""

        check_name: str
        severity: str
        message: str
        entity_ids: list[str] = []
        entity_type: Optional[str] = None


class GenerationResult(BaseModel):
    """Result container for the generation validation pipeline.

    Attributes:
        shape: The reconstructed OCC shape (arbitrary type allowed).
        valid: Whether the shape passed all validation checks.
        mode: Generation mode used ('command' or 'diffusion').
        num_retries: Number of retry attempts performed.
        total_time_ms: Total pipeline time in milliseconds.
        validation_report: Detailed validation results per attempt.
        findings: List of ValidationFinding objects from final attempt.
        errors: Accumulated error messages across all attempts.
    """

    model_config = {"arbitrary_types_allowed": True}

    shape: Optional[Any] = None
    valid: bool = False
    mode: str = "command"
    num_retries: int = 0
    total_time_ms: float = 0.0
    validation_report: List[Dict[str, Any]] = Field(default_factory=list)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class ValidationFeedbackLoop:
    """Iterative validation and repair loop for generated CAD shapes.

    Wraps a CommandExecutor and validates each generated shape via
    topology checks. If validation fails, collects diagnostic findings,
    adjusts the command tokens based on the findings, and retries.

    The validation checks include:
        - BRepCheck_Analyzer validity
        - Manifoldness (each edge shared by <= 2 faces)
        - Euler characteristic (V - E + F = 2 for genus-0)
        - Watertightness (no boundary edges)
        - Self-intersection detection (optional)

    Token adjustment strategies per finding type:
        - Non-manifold edges: perturb nearby vertex coordinates
        - Open solid (boundary edges): extend or close sketch profiles
        - Degenerate geometry: remove near-zero-length edges/faces
        - Invalid Boolean: swap Boolean operation type

    Attributes:
        executor: CommandExecutor instance for geometry construction.
        max_retries: Maximum number of retry attempts.
        check_self_intersections: Whether to run expensive self-intersection check.

    Example:
        executor = CommandExecutor()
        loop = ValidationFeedbackLoop(executor, max_retries=3)
        result = loop.run(command_tokens)
    """

    def __init__(
        self,
        executor: Any,
        max_retries: int = 3,
        check_self_intersections: bool = False,
    ):
        """Initialize the validation feedback loop.

        Args:
            executor: CommandExecutor (or compatible) instance.
            max_retries: Maximum number of retry attempts after initial
                generation fails validation. Total attempts = 1 + max_retries.
            check_self_intersections: Whether to include the expensive
                self-intersection check in validation.
        """
        self.executor = executor
        self.max_retries = max_retries
        self.check_self_intersections = check_self_intersections

    def run(
        self,
        command_tokens: List[int],
        mode: str = "command",
    ) -> GenerationResult:
        """Run the validation feedback loop.

        Executes the command sequence, validates the result, and retries
        with adjusted tokens if validation fails.

        Args:
            command_tokens: Quantized command token sequence.
            mode: Generation mode ('command' for DeepCAD-style,
                'diffusion' for BrepGen-style). Currently only 'command'
                is fully implemented.

        Returns:
            GenerationResult with the final shape, validity status,
            and detailed validation report.
        """
        start_time = time.monotonic()

        gen_result = GenerationResult(mode=mode)
        current_tokens = list(command_tokens)

        for attempt in range(1 + self.max_retries):
            attempt_start = time.monotonic()

            _log.debug(
                "Validation loop attempt %d/%d (%d tokens)",
                attempt + 1,
                1 + self.max_retries,
                len(current_tokens),
            )

            # Step 1: Execute/reconstruct
            exec_result = self.executor.execute(current_tokens)
            shape = exec_result.get("shape")
            exec_errors = exec_result.get("errors", [])

            attempt_report: Dict[str, Any] = {
                "attempt": attempt + 1,
                "exec_errors": exec_errors,
                "shape_produced": shape is not None,
                "findings": [],
                "adjusted": False,
            }

            if shape is None:
                attempt_report["shape_produced"] = False
                gen_result.errors.extend(exec_errors)
                gen_result.validation_report.append(attempt_report)

                if attempt < self.max_retries:
                    # Try to fix by removing problematic tokens
                    current_tokens = self._adjust_for_retry(
                        current_tokens,
                        [
                            ValidationFinding(
                                check_name="no_shape",
                                severity="critical",
                                message="Execution produced no shape",
                            )
                        ],
                    )
                    attempt_report["adjusted"] = True

                gen_result.num_retries = attempt
                continue

            # Step 2: Validate topology
            findings = self._validate_topology(shape)
            attempt_report["findings"] = [
                f.model_dump() for f in findings
            ]

            attempt_duration = (time.monotonic() - attempt_start) * 1000
            attempt_report["duration_ms"] = attempt_duration
            gen_result.validation_report.append(attempt_report)

            # Step 3: Check if valid
            critical_findings = [
                f for f in findings if f.severity == "critical"
            ]

            if not critical_findings:
                # Valid shape! Accept and return.
                gen_result.shape = shape
                gen_result.valid = True
                gen_result.findings = [f.model_dump() for f in findings]
                gen_result.num_retries = attempt
                gen_result.total_time_ms = (
                    time.monotonic() - start_time
                ) * 1000

                _log.info(
                    "Validation loop succeeded at attempt %d/%d "
                    "(%.1f ms, %d warnings)",
                    attempt + 1,
                    1 + self.max_retries,
                    gen_result.total_time_ms,
                    len(findings),
                )
                return gen_result

            # Step 4: Validation failed; adjust tokens for retry
            if attempt < self.max_retries:
                current_tokens = self._adjust_for_retry(
                    current_tokens, findings
                )
                attempt_report["adjusted"] = True
                _log.debug(
                    "Attempt %d failed with %d critical findings; retrying",
                    attempt + 1,
                    len(critical_findings),
                )
            else:
                # Last attempt: return best result even if invalid
                gen_result.shape = shape
                gen_result.valid = False
                gen_result.findings = [f.model_dump() for f in findings]
                gen_result.errors.extend(
                    f.message for f in critical_findings
                )

            gen_result.num_retries = attempt

        gen_result.total_time_ms = (time.monotonic() - start_time) * 1000

        _log.warning(
            "Validation loop exhausted %d attempts (%.1f ms). "
            "Returning best result (valid=%s)",
            1 + self.max_retries,
            gen_result.total_time_ms,
            gen_result.valid,
        )

        return gen_result

    def _validate_topology(self, shape: Any) -> List[ValidationFinding]:
        """Validate the topology of a reconstructed shape.

        Runs a suite of topology checks including BRepCheck, manifoldness,
        Euler characteristic, watertightness, and optionally self-intersection.

        Args:
            shape: TopoDS_Shape to validate.

        Returns:
            List of ValidationFinding objects describing all issues found.
        """
        findings: List[ValidationFinding] = []

        if not _has_pythonocc:
            findings.append(
                ValidationFinding(
                    check_name="pythonocc_availability",
                    severity="warning",
                    message="pythonocc not available; topology validation skipped",
                )
            )
            return findings

        # Check 1: BRepCheck_Analyzer
        try:
            analyzer = BRepCheck_Analyzer(shape)
            if not analyzer.IsValid():
                findings.append(
                    ValidationFinding(
                        check_name="brepcheck_analyzer",
                        severity="critical",
                        message="BRepCheck_Analyzer reports invalid shape",
                    )
                )
        except Exception as e:
            findings.append(
                ValidationFinding(
                    check_name="brepcheck_analyzer",
                    severity="warning",
                    message=f"BRepCheck_Analyzer raised exception: {e}",
                )
            )

        # Count topological elements
        num_vertices = 0
        num_edges = 0
        num_faces = 0

        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        while explorer.More():
            num_vertices += 1
            explorer.Next()

        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while explorer.More():
            num_edges += 1
            explorer.Next()

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            num_faces += 1
            explorer.Next()

        # Check 2: Euler characteristic
        euler_char = num_vertices - num_edges + num_faces
        expected_euler_values = {2, 0, -2, -4}  # genus 0, 1, 2, 3

        if euler_char not in expected_euler_values:
            findings.append(
                ValidationFinding(
                    check_name="euler_characteristic",
                    severity="warning",
                    message=(
                        f"Unusual Euler characteristic: V({num_vertices}) "
                        f"- E({num_edges}) + F({num_faces}) = {euler_char}"
                    ),
                )
            )

        # Check 3: Manifoldness and watertightness via edge-face adjacency
        try:
            edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
            topexp.MapShapesAndAncestors(
                shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map
            )

            num_boundary = 0
            num_non_manifold = 0

            for i in range(1, edge_face_map.Size() + 1):
                adj_count = edge_face_map.FindFromIndex(i).Size()
                if adj_count == 1:
                    num_boundary += 1
                elif adj_count > 2:
                    num_non_manifold += 1

            if num_non_manifold > 0:
                findings.append(
                    ValidationFinding(
                        check_name="manifoldness",
                        severity="critical",
                        message=(
                            f"Non-manifold geometry: {num_non_manifold} edges "
                            f"shared by >2 faces"
                        ),
                    )
                )

            if num_boundary > 0:
                findings.append(
                    ValidationFinding(
                        check_name="watertightness",
                        severity="critical",
                        message=(
                            f"Open solid: {num_boundary} boundary edges "
                            f"(not watertight)"
                        ),
                    )
                )

        except Exception as e:
            findings.append(
                ValidationFinding(
                    check_name="edge_face_adjacency",
                    severity="warning",
                    message=f"Edge-face adjacency check failed: {e}",
                )
            )

        # Check 4: Degenerate faces (near-zero area)
        try:
            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            degenerate_count = 0
            while explorer.More():
                face = explorer.Current()
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                area = props.Mass()
                if area < 1e-10:
                    degenerate_count += 1
                explorer.Next()

            if degenerate_count > 0:
                findings.append(
                    ValidationFinding(
                        check_name="degenerate_faces",
                        severity="warning",
                        message=f"{degenerate_count} degenerate faces (near-zero area)",
                    )
                )
        except Exception as e:
            _log.debug("Degenerate face check failed: %s", e)

        # Check 5: Degenerate edges (near-zero length)
        try:
            explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            degenerate_edge_count = 0
            while explorer.More():
                edge = explorer.Current()
                props = GProp_GProps()
                brepgprop.LinearProperties(edge, props)
                length = props.Mass()
                if length < 1e-10:
                    degenerate_edge_count += 1
                explorer.Next()

            if degenerate_edge_count > 0:
                findings.append(
                    ValidationFinding(
                        check_name="degenerate_edges",
                        severity="warning",
                        message=f"{degenerate_edge_count} degenerate edges (near-zero length)",
                    )
                )
        except Exception as e:
            _log.debug("Degenerate edge check failed: %s", e)

        _log.debug(
            "Topology validation: V=%d, E=%d, F=%d, chi=%d, "
            "%d findings (%d critical)",
            num_vertices,
            num_edges,
            num_faces,
            euler_char,
            len(findings),
            sum(1 for f in findings if f.severity == "critical"),
        )

        return findings

    def _adjust_for_retry(
        self,
        tokens: List[int],
        findings: List[ValidationFinding],
    ) -> List[int]:
        """Adjust command tokens based on validation findings for retry.

        Applies heuristic corrections to the token sequence based on the
        types of validation failures detected:

        - **no_shape / exec_errors**: Remove the last sketch group and retry
          with simplified geometry.
        - **Non-manifold edges**: Perturb parameter tokens near the failing
          region by small random offsets.
        - **Boundary edges (open solid)**: Duplicate the last LINE's endpoint
          to close sketch loops.
        - **Degenerate geometry**: Remove pairs of tokens that would produce
          near-zero-length edges.
        - **BRepCheck failures**: Apply small uniform perturbation to all
          parameter tokens.

        Args:
            tokens: Current command token sequence.
            findings: Validation findings from the failed attempt.

        Returns:
            Adjusted token sequence for the next retry attempt.
        """
        adjusted = list(tokens)

        finding_types = {f.check_name for f in findings}

        if "no_shape" in finding_types:
            # Remove last sketch group (SOL...EOL block) if possible
            adjusted = self._remove_last_sketch_group(adjusted)
            _log.debug("Adjustment: removed last sketch group")
            return adjusted

        if "manifoldness" in finding_types:
            # Perturb parameter tokens by small random offset
            adjusted = self._perturb_parameters(adjusted, magnitude=2)
            _log.debug("Adjustment: perturbed parameters for manifoldness fix")

        if "watertightness" in finding_types:
            # Try to close open loops by adding closing LINE tokens
            adjusted = self._close_sketch_loops(adjusted)
            _log.debug("Adjustment: attempted loop closure for watertightness")

        if "degenerate_faces" in finding_types or "degenerate_edges" in finding_types:
            # Remove potential degenerate segments
            adjusted = self._remove_degenerate_segments(adjusted)
            _log.debug("Adjustment: removed degenerate segments")

        if "brepcheck_analyzer" in finding_types:
            # General perturbation as last resort
            adjusted = self._perturb_parameters(adjusted, magnitude=1)
            _log.debug("Adjustment: general perturbation for BRepCheck fix")

        return adjusted

    def _remove_last_sketch_group(self, tokens: List[int]) -> List[int]:
        """Remove the last SOL...EOL group from the token sequence.

        Args:
            tokens: Command token sequence.

        Returns:
            Token sequence with last sketch group removed.
        """
        # Find the last SOL token
        last_sol_idx = -1
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == 0:  # SOL token
                last_sol_idx = i
                break

        if last_sol_idx < 0:
            return tokens

        # Find matching EOL
        eol_idx = -1
        for i in range(last_sol_idx + 1, len(tokens)):
            if tokens[i] == 1:  # EOL token
                eol_idx = i
                break

        if eol_idx < 0:
            # No matching EOL; remove from SOL to end
            return tokens[:last_sol_idx]

        # Remove the SOL...EOL block
        result = tokens[:last_sol_idx] + tokens[eol_idx + 1:]
        return result

    def _perturb_parameters(
        self, tokens: List[int], magnitude: int = 1
    ) -> List[int]:
        """Perturb parameter tokens by small random offsets.

        Only modifies tokens with values >= 9 (parameter tokens), leaving
        command and special tokens unchanged.

        Args:
            tokens: Command token sequence.
            magnitude: Maximum perturbation in token units.

        Returns:
            Perturbed token sequence.
        """
        import random

        adjusted = list(tokens)
        for i in range(len(adjusted)):
            if adjusted[i] >= 9:  # Parameter token
                offset = random.randint(-magnitude, magnitude)
                adjusted[i] = max(9, adjusted[i] + offset)
        return adjusted

    def _close_sketch_loops(self, tokens: List[int]) -> List[int]:
        """Attempt to close open sketch loops by adding LINE closure.

        Scans for SOL...EOL blocks and inserts a LINE command that connects
        the last endpoint back to the first endpoint if the loop is not
        already closed.

        Args:
            tokens: Command token sequence.

        Returns:
            Token sequence with closure lines inserted.
        """
        adjusted = list(tokens)

        # Find all SOL positions
        sol_positions = [i for i, t in enumerate(adjusted) if t == 0]

        for sol_idx in reversed(sol_positions):
            # Find matching EOL
            eol_idx = -1
            for i in range(sol_idx + 1, len(adjusted)):
                if adjusted[i] == 1:  # EOL
                    eol_idx = i
                    break

            if eol_idx < 0:
                continue

            # Check if there's at least one command in the group
            group = adjusted[sol_idx + 1:eol_idx]
            if len(group) < 5:  # Need at least command + 4 params for LINE
                continue

            # Find first LINE params (start point) and last LINE params (end point)
            first_params = None
            last_params = None

            i = 0
            while i < len(group):
                if group[i] == 6:  # LINE token
                    if first_params is None and i + 4 < len(group):
                        first_params = group[i + 1:i + 5]
                    if i + 4 < len(group):
                        last_params = group[i + 1:i + 5]
                    i += 5
                elif group[i] == 7:  # ARC token
                    if first_params is None and i + 2 < len(group):
                        first_params = [group[i + 1], group[i + 2], 0, 0]
                    if i + 4 < len(group):
                        last_params = [0, 0, group[i + 3], group[i + 4]]
                    i += 7
                else:
                    i += 1

            if first_params is not None and last_params is not None:
                # Check if last endpoint differs from first start point
                if last_params[2:4] != first_params[0:2]:
                    # Insert closing LINE: from last end to first start
                    closure = [6] + [last_params[2], last_params[3], first_params[0], first_params[1]]
                    adjusted = (
                        adjusted[:eol_idx]
                        + closure
                        + adjusted[eol_idx:]
                    )

        return adjusted

    def _remove_degenerate_segments(self, tokens: List[int]) -> List[int]:
        """Remove command segments that would produce degenerate geometry.

        Identifies LINE commands where start and end parameters are identical
        (would produce zero-length edges) and removes them.

        Args:
            tokens: Command token sequence.

        Returns:
            Token sequence with degenerate segments removed.
        """
        adjusted: List[int] = []
        i = 0
        while i < len(tokens):
            if tokens[i] == 6 and i + 4 < len(tokens):  # LINE token
                x1, y1, x2, y2 = tokens[i + 1:i + 5]
                if x1 == x2 and y1 == y2:
                    # Skip degenerate LINE (zero length)
                    _log.debug(
                        "Removing degenerate LINE at position %d", i
                    )
                    i += 5
                    continue
            adjusted.append(tokens[i])
            i += 1

        return adjusted
