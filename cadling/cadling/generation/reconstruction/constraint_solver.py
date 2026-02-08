"""Constraint solver for sketch constraint satisfaction.

Detects and solves geometric constraints in 2D sketch profiles, ensuring
that near-coincident points are snapped together, near-tangent curves are
smoothed, and dimensional constraints (parallel, perpendicular, equal
length/radius) are satisfied.

Uses Newton's method to iteratively adjust primitive parameters until all
constraints are satisfied within tolerance.

Classes:
    ConstraintType: Enumeration of supported constraint types.
    SketchConstraint: Individual constraint between primitives.
    SketchPrimitive: Base representation of a sketch primitive.
    ConstraintSolver: Main solver class.

Example:
    solver = ConstraintSolver(tolerance=1e-4, max_iterations=100)
    constraints = solver.detect_near_constraints(primitives)
    adjusted, converged = solver.solve(primitives, constraints)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)


class ConstraintType(str, Enum):
    """Types of geometric constraints for sketch profiles.

    Each constraint type maps to a set of residual equations that the
    solver minimizes to zero.
    """

    COINCIDENT = "coincident"
    TANGENT = "tangent"
    PERPENDICULAR = "perpendicular"
    PARALLEL = "parallel"
    CONCENTRIC = "concentric"
    EQUAL_LENGTH = "equal_length"
    EQUAL_RADIUS = "equal_radius"
    DISTANCE = "distance"
    ANGLE = "angle"


@dataclass
class SketchConstraint:
    """Individual constraint between two sketch primitives.

    Attributes:
        type: Constraint type from ConstraintType enum.
        source_idx: Index of the source primitive in the primitives list.
        target_idx: Index of the target primitive in the primitives list.
        value: Constraint parameter value. Interpretation depends on type:
            - DISTANCE: target distance between endpoints.
            - ANGLE: target angle in radians.
            - Others: not used (set to 0.0).
    """

    type: ConstraintType
    source_idx: int
    target_idx: int
    value: float = 0.0


@dataclass
class SketchPrimitive:
    """Representation of a 2D sketch primitive for constraint solving.

    All primitives are represented as parameter vectors that the solver
    can adjust. The interpretation depends on the primitive_type.

    Attributes:
        primitive_type: One of "line", "arc", "circle".
        params: Parameter vector:
            - line: [x1, y1, x2, y2]
            - arc: [x1, y1, x2, y2, cx, cy]
            - circle: [cx, cy, radius]
    """

    primitive_type: str
    params: List[float] = field(default_factory=list)

    @property
    def start_point(self) -> Tuple[float, float]:
        """Get the start point of the primitive."""
        if self.primitive_type in ("line", "arc"):
            return (self.params[0], self.params[1])
        elif self.primitive_type == "circle":
            # Circle has no start; use rightmost point
            cx, cy, r = self.params[:3]
            return (cx + r, cy)
        return (0.0, 0.0)

    @property
    def end_point(self) -> Tuple[float, float]:
        """Get the end point of the primitive."""
        if self.primitive_type in ("line", "arc"):
            return (self.params[2], self.params[3])
        elif self.primitive_type == "circle":
            # Circle loops back; same as start
            cx, cy, r = self.params[:3]
            return (cx + r, cy)
        return (0.0, 0.0)

    @property
    def center(self) -> Optional[Tuple[float, float]]:
        """Get center for arc or circle primitives."""
        if self.primitive_type == "arc" and len(self.params) >= 6:
            return (self.params[4], self.params[5])
        elif self.primitive_type == "circle" and len(self.params) >= 2:
            return (self.params[0], self.params[1])
        return None

    @property
    def length(self) -> float:
        """Compute primitive length."""
        if self.primitive_type == "line":
            dx = self.params[2] - self.params[0]
            dy = self.params[3] - self.params[1]
            return math.sqrt(dx * dx + dy * dy)
        elif self.primitive_type == "circle":
            return 2.0 * math.pi * abs(self.params[2])
        elif self.primitive_type == "arc":
            # Approximate arc length from center and endpoints
            if len(self.params) >= 6:
                cx, cy = self.params[4], self.params[5]
                r = math.sqrt(
                    (self.params[0] - cx) ** 2 + (self.params[1] - cy) ** 2
                )
                a1 = math.atan2(self.params[1] - cy, self.params[0] - cx)
                a2 = math.atan2(self.params[3] - cy, self.params[2] - cx)
                angle = abs(a2 - a1)
                if angle > math.pi:
                    angle = 2 * math.pi - angle
                return r * angle
        return 0.0

    @property
    def radius(self) -> Optional[float]:
        """Get radius for arc or circle."""
        if self.primitive_type == "circle" and len(self.params) >= 3:
            return abs(self.params[2])
        elif self.primitive_type == "arc" and len(self.params) >= 6:
            cx, cy = self.params[4], self.params[5]
            return math.sqrt(
                (self.params[0] - cx) ** 2 + (self.params[1] - cy) ** 2
            )
        return None

    @property
    def direction(self) -> Optional[Tuple[float, float]]:
        """Get direction vector for lines; tangent at start for arcs."""
        if self.primitive_type == "line":
            dx = self.params[2] - self.params[0]
            dy = self.params[3] - self.params[1]
            mag = math.sqrt(dx * dx + dy * dy)
            if mag > 1e-12:
                return (dx / mag, dy / mag)
        elif self.primitive_type == "arc" and len(self.params) >= 6:
            # Tangent at start: perpendicular to radius
            cx, cy = self.params[4], self.params[5]
            rx = self.params[0] - cx
            ry = self.params[1] - cy
            mag = math.sqrt(rx * rx + ry * ry)
            if mag > 1e-12:
                return (-ry / mag, rx / mag)
        return None


class ConstraintSolver:
    """Solver for 2D sketch geometric constraints.

    Detects near-constraints from approximate geometry produced by the
    generation model, builds a constraint residual system, and uses
    Newton's method to iteratively adjust primitive parameters until all
    constraints are satisfied within tolerance.

    The solver supports:
        - COINCIDENT: snap endpoints together
        - TANGENT: align tangent directions at shared endpoints
        - PERPENDICULAR: enforce 90-degree angle between directions
        - PARALLEL: enforce zero angle between directions
        - CONCENTRIC: snap circle/arc centers together
        - EQUAL_LENGTH: equalize line/arc lengths
        - EQUAL_RADIUS: equalize circle/arc radii
        - DISTANCE: enforce specific distance between endpoints
        - ANGLE: enforce specific angle between directions

    Attributes:
        tolerance: Convergence tolerance for constraint residuals.
        max_iterations: Maximum Newton iterations.
        detection_threshold: Distance threshold for detecting near-constraints.
        angle_threshold: Angle threshold (radians) for tangent/perpendicular/parallel.
        damping: Damping factor for Newton updates to improve stability.

    Example:
        solver = ConstraintSolver(tolerance=1e-4, max_iterations=100)
        constraints = solver.detect_near_constraints(primitives)
        adjusted, converged = solver.solve(primitives, constraints)
    """

    def __init__(
        self,
        tolerance: float = 1e-4,
        max_iterations: int = 100,
        detection_threshold: float = 0.05,
        angle_threshold: float = 0.1,
        damping: float = 0.5,
    ):
        """Initialize the constraint solver.

        Args:
            tolerance: Convergence tolerance for residual norm.
            max_iterations: Maximum number of Newton iterations.
            detection_threshold: Distance below which endpoints are
                considered near-coincident.
            angle_threshold: Angle (radians) below which directions are
                considered near-parallel or near-perpendicular.
            damping: Damping factor (0, 1] for Newton step updates.
                Lower values improve stability at the cost of convergence speed.
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.detection_threshold = detection_threshold
        self.angle_threshold = angle_threshold
        self.damping = damping

    def detect_near_constraints(
        self, primitives: List[SketchPrimitive]
    ) -> List[SketchConstraint]:
        """Detect near-constraints from approximate primitive geometry.

        Scans all pairs of primitives for geometric relationships that are
        close to exact constraints (within detection thresholds), and returns
        the inferred constraint list.

        Args:
            primitives: List of sketch primitives to analyze.

        Returns:
            List of detected SketchConstraint objects.
        """
        constraints: List[SketchConstraint] = []
        n = len(primitives)

        for i in range(n):
            for j in range(i + 1, n):
                pi = primitives[i]
                pj = primitives[j]

                # Check endpoint coincidence (end of i -> start of j)
                constraints.extend(
                    self._detect_coincident(pi, pj, i, j)
                )

                # Check concentricity for circles/arcs
                constraints.extend(
                    self._detect_concentric(pi, pj, i, j)
                )

                # Check parallel/perpendicular for lines
                constraints.extend(
                    self._detect_directional(pi, pj, i, j)
                )

                # Check equal length
                constraints.extend(
                    self._detect_equal_length(pi, pj, i, j)
                )

                # Check equal radius
                constraints.extend(
                    self._detect_equal_radius(pi, pj, i, j)
                )

                # Check tangency for line-arc or arc-arc pairs
                constraints.extend(
                    self._detect_tangent(pi, pj, i, j)
                )

        _log.debug(
            "Detected %d near-constraints from %d primitives",
            len(constraints),
            n,
        )
        return constraints

    def _detect_coincident(
        self,
        pi: SketchPrimitive,
        pj: SketchPrimitive,
        i: int,
        j: int,
    ) -> List[SketchConstraint]:
        """Detect near-coincident endpoint pairs."""
        constraints: List[SketchConstraint] = []

        # Check end of pi -> start of pj
        ei = pi.end_point
        sj = pj.start_point
        dist = math.sqrt((ei[0] - sj[0]) ** 2 + (ei[1] - sj[1]) ** 2)
        if dist < self.detection_threshold and dist > 0:
            constraints.append(
                SketchConstraint(
                    type=ConstraintType.COINCIDENT,
                    source_idx=i,
                    target_idx=j,
                    value=0.0,
                )
            )

        # Check end of pj -> start of pi
        ej = pj.end_point
        si = pi.start_point
        dist = math.sqrt((ej[0] - si[0]) ** 2 + (ej[1] - si[1]) ** 2)
        if dist < self.detection_threshold and dist > 0:
            constraints.append(
                SketchConstraint(
                    type=ConstraintType.COINCIDENT,
                    source_idx=j,
                    target_idx=i,
                    value=0.0,
                )
            )

        return constraints

    def _detect_concentric(
        self,
        pi: SketchPrimitive,
        pj: SketchPrimitive,
        i: int,
        j: int,
    ) -> List[SketchConstraint]:
        """Detect near-concentric circle/arc pairs."""
        ci = pi.center
        cj = pj.center
        if ci is None or cj is None:
            return []

        dist = math.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
        if dist < self.detection_threshold and dist > 0:
            return [
                SketchConstraint(
                    type=ConstraintType.CONCENTRIC,
                    source_idx=i,
                    target_idx=j,
                    value=0.0,
                )
            ]
        return []

    def _detect_directional(
        self,
        pi: SketchPrimitive,
        pj: SketchPrimitive,
        i: int,
        j: int,
    ) -> List[SketchConstraint]:
        """Detect near-parallel or near-perpendicular line pairs."""
        di = pi.direction
        dj = pj.direction
        if di is None or dj is None:
            return []

        # Dot product gives cosine of angle
        dot = di[0] * dj[0] + di[1] * dj[1]
        cross = di[0] * dj[1] - di[1] * dj[0]

        constraints: List[SketchConstraint] = []

        # Near-parallel: |dot| close to 1
        if abs(abs(dot) - 1.0) < self.angle_threshold:
            constraints.append(
                SketchConstraint(
                    type=ConstraintType.PARALLEL,
                    source_idx=i,
                    target_idx=j,
                    value=0.0,
                )
            )

        # Near-perpendicular: |dot| close to 0
        if abs(dot) < self.angle_threshold:
            constraints.append(
                SketchConstraint(
                    type=ConstraintType.PERPENDICULAR,
                    source_idx=i,
                    target_idx=j,
                    value=0.0,
                )
            )

        return constraints

    def _detect_equal_length(
        self,
        pi: SketchPrimitive,
        pj: SketchPrimitive,
        i: int,
        j: int,
    ) -> List[SketchConstraint]:
        """Detect near-equal length primitives."""
        li = pi.length
        lj = pj.length
        if li < 1e-12 or lj < 1e-12:
            return []

        relative_diff = abs(li - lj) / max(li, lj)
        if relative_diff < self.detection_threshold:
            return [
                SketchConstraint(
                    type=ConstraintType.EQUAL_LENGTH,
                    source_idx=i,
                    target_idx=j,
                    value=0.0,
                )
            ]
        return []

    def _detect_equal_radius(
        self,
        pi: SketchPrimitive,
        pj: SketchPrimitive,
        i: int,
        j: int,
    ) -> List[SketchConstraint]:
        """Detect near-equal radius circle/arc pairs."""
        ri = pi.radius
        rj = pj.radius
        if ri is None or rj is None:
            return []
        if ri < 1e-12 or rj < 1e-12:
            return []

        relative_diff = abs(ri - rj) / max(ri, rj)
        if relative_diff < self.detection_threshold:
            return [
                SketchConstraint(
                    type=ConstraintType.EQUAL_RADIUS,
                    source_idx=i,
                    target_idx=j,
                    value=0.0,
                )
            ]
        return []

    def _detect_tangent(
        self,
        pi: SketchPrimitive,
        pj: SketchPrimitive,
        i: int,
        j: int,
    ) -> List[SketchConstraint]:
        """Detect near-tangent pairs at shared endpoints."""
        # Only check if they share an endpoint (near-coincident end->start)
        ei = pi.end_point
        sj = pj.start_point
        dist = math.sqrt((ei[0] - sj[0]) ** 2 + (ei[1] - sj[1]) ** 2)
        if dist > self.detection_threshold:
            return []

        # Check tangent direction alignment at the shared point
        di = pi.direction
        dj = pj.direction
        if di is None or dj is None:
            return []

        # For arcs, we need tangent at the end point, not start
        if pi.primitive_type == "arc" and pi.center is not None:
            cx, cy = pi.center
            rx = pi.params[2] - cx  # end point relative to center
            ry = pi.params[3] - cy
            mag = math.sqrt(rx * rx + ry * ry)
            if mag > 1e-12:
                di = (-ry / mag, rx / mag)

        dot = di[0] * dj[0] + di[1] * dj[1]
        if abs(abs(dot) - 1.0) < self.angle_threshold:
            return [
                SketchConstraint(
                    type=ConstraintType.TANGENT,
                    source_idx=i,
                    target_idx=j,
                    value=0.0,
                )
            ]
        return []

    def build_constraint_system(
        self,
        primitives: List[SketchPrimitive],
        constraints: List[SketchConstraint],
    ) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        """Build the constraint residual system.

        Assembles the parameter vector from all primitives, and constructs
        a residual function that maps parameters to constraint violations.

        Args:
            primitives: List of sketch primitives.
            constraints: List of constraints to satisfy.

        Returns:
            Tuple of (initial parameter vector, residual function).
        """
        # Build parameter vector: concatenate all primitive params
        param_offsets: List[int] = []
        offset = 0
        for p in primitives:
            param_offsets.append(offset)
            offset += len(p.params)

        x0 = np.zeros(offset, dtype=np.float64)
        for i, p in enumerate(primitives):
            start = param_offsets[i]
            x0[start:start + len(p.params)] = p.params

        def residual_fn(x: np.ndarray) -> np.ndarray:
            """Compute constraint residual vector."""
            residuals: List[float] = []

            for c in constraints:
                si = param_offsets[c.source_idx]
                ti = param_offsets[c.target_idx]
                ps = primitives[c.source_idx]
                pt = primitives[c.target_idx]
                n_ps = len(ps.params)
                n_pt = len(pt.params)
                xs = x[si:si + n_ps]
                xt = x[ti:ti + n_pt]

                if c.type == ConstraintType.COINCIDENT:
                    # end_point(source) == start_point(target)
                    if ps.primitive_type in ("line", "arc"):
                        ex, ey = xs[2], xs[3]
                    else:
                        ex = xs[0] + abs(xs[2]) if len(xs) >= 3 else xs[0]
                        ey = xs[1]

                    if pt.primitive_type in ("line", "arc"):
                        sx, sy = xt[0], xt[1]
                    else:
                        sx = xt[0] + abs(xt[2]) if len(xt) >= 3 else xt[0]
                        sy = xt[1]

                    residuals.append(ex - sx)
                    residuals.append(ey - sy)

                elif c.type == ConstraintType.CONCENTRIC:
                    # centers must match
                    if ps.primitive_type == "circle":
                        cx1, cy1 = xs[0], xs[1]
                    elif ps.primitive_type == "arc" and len(xs) >= 6:
                        cx1, cy1 = xs[4], xs[5]
                    else:
                        continue

                    if pt.primitive_type == "circle":
                        cx2, cy2 = xt[0], xt[1]
                    elif pt.primitive_type == "arc" and len(xt) >= 6:
                        cx2, cy2 = xt[4], xt[5]
                    else:
                        continue

                    residuals.append(cx1 - cx2)
                    residuals.append(cy1 - cy2)

                elif c.type == ConstraintType.PARALLEL:
                    # cross product of directions = 0
                    d1 = self._direction_from_params(ps.primitive_type, xs)
                    d2 = self._direction_from_params(pt.primitive_type, xt)
                    if d1 is not None and d2 is not None:
                        cross = d1[0] * d2[1] - d1[1] * d2[0]
                        residuals.append(cross)

                elif c.type == ConstraintType.PERPENDICULAR:
                    # dot product of directions = 0
                    d1 = self._direction_from_params(ps.primitive_type, xs)
                    d2 = self._direction_from_params(pt.primitive_type, xt)
                    if d1 is not None and d2 is not None:
                        dot = d1[0] * d2[0] + d1[1] * d2[1]
                        residuals.append(dot)

                elif c.type == ConstraintType.TANGENT:
                    # At shared endpoint: direction cross product = 0
                    d1 = self._end_tangent_from_params(ps.primitive_type, xs)
                    d2 = self._direction_from_params(pt.primitive_type, xt)
                    if d1 is not None and d2 is not None:
                        cross = d1[0] * d2[1] - d1[1] * d2[0]
                        residuals.append(cross)

                elif c.type == ConstraintType.EQUAL_LENGTH:
                    l1 = self._length_from_params(ps.primitive_type, xs)
                    l2 = self._length_from_params(pt.primitive_type, xt)
                    residuals.append(l1 - l2)

                elif c.type == ConstraintType.EQUAL_RADIUS:
                    r1 = self._radius_from_params(ps.primitive_type, xs)
                    r2 = self._radius_from_params(pt.primitive_type, xt)
                    if r1 is not None and r2 is not None:
                        residuals.append(r1 - r2)

                elif c.type == ConstraintType.DISTANCE:
                    if ps.primitive_type in ("line", "arc"):
                        ex, ey = xs[2], xs[3]
                    else:
                        ex, ey = xs[0], xs[1]

                    if pt.primitive_type in ("line", "arc"):
                        sx, sy = xt[0], xt[1]
                    else:
                        sx, sy = xt[0], xt[1]

                    dist = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
                    residuals.append(dist - c.value)

                elif c.type == ConstraintType.ANGLE:
                    d1 = self._direction_from_params(ps.primitive_type, xs)
                    d2 = self._direction_from_params(pt.primitive_type, xt)
                    if d1 is not None and d2 is not None:
                        dot = d1[0] * d2[0] + d1[1] * d2[1]
                        dot = max(-1.0, min(1.0, dot))
                        angle = math.acos(dot)
                        residuals.append(angle - c.value)

            return np.array(residuals, dtype=np.float64)

        return x0, residual_fn

    def solve(
        self,
        primitives: List[SketchPrimitive],
        constraints: List[SketchConstraint],
    ) -> Tuple[List[SketchPrimitive], bool]:
        """Solve constraint system using damped Newton's method.

        Iteratively adjusts primitive parameters to minimize constraint
        residuals. Uses numerical Jacobian computation and damped updates
        for stability.

        Args:
            primitives: List of sketch primitives to adjust.
            constraints: List of constraints to satisfy.

        Returns:
            Tuple of (adjusted primitives list, whether solver converged).
        """
        if not constraints:
            _log.debug("No constraints to solve")
            return primitives, True

        # Build initial parameter vector and residual function
        x, residual_fn = self.build_constraint_system(primitives, constraints)

        converged = False
        for iteration in range(self.max_iterations):
            r = residual_fn(x)
            residual_norm = np.linalg.norm(r)

            if residual_norm < self.tolerance:
                converged = True
                _log.debug(
                    "Constraint solver converged at iteration %d "
                    "(residual=%.2e)",
                    iteration,
                    residual_norm,
                )
                break

            # Compute numerical Jacobian
            jacobian = self._numerical_jacobian(residual_fn, x, r)

            # Solve J * dx = -r using least squares (overdetermined system)
            try:
                dx, _, _, _ = np.linalg.lstsq(jacobian, -r, rcond=None)
            except np.linalg.LinAlgError:
                _log.warning(
                    "Singular Jacobian at iteration %d; terminating", iteration
                )
                break

            # Damped update
            x = x + self.damping * dx

            if iteration % 20 == 0:
                _log.debug(
                    "Iteration %d: residual=%.4e, |dx|=%.4e",
                    iteration,
                    residual_norm,
                    np.linalg.norm(dx),
                )

        if not converged:
            final_residual = np.linalg.norm(residual_fn(x))
            _log.warning(
                "Constraint solver did not converge after %d iterations "
                "(residual=%.4e)",
                self.max_iterations,
                final_residual,
            )

        # Reconstruct primitives from adjusted parameter vector
        adjusted = self._reconstruct_primitives(primitives, x)
        return adjusted, converged

    def _numerical_jacobian(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        r0: np.ndarray,
        eps: float = 1e-7,
    ) -> np.ndarray:
        """Compute numerical Jacobian via forward finite differences.

        Args:
            residual_fn: Residual function mapping params to residuals.
            x: Current parameter vector.
            r0: Current residual vector (avoids recomputation).
            eps: Finite difference step size.

        Returns:
            Jacobian matrix of shape (num_residuals, num_params).
        """
        n_params = len(x)
        n_residuals = len(r0)
        jacobian = np.zeros((n_residuals, n_params), dtype=np.float64)

        for k in range(n_params):
            x_plus = x.copy()
            x_plus[k] += eps
            r_plus = residual_fn(x_plus)
            jacobian[:, k] = (r_plus - r0) / eps

        return jacobian

    def _reconstruct_primitives(
        self,
        original: List[SketchPrimitive],
        x: np.ndarray,
    ) -> List[SketchPrimitive]:
        """Reconstruct SketchPrimitive list from adjusted parameter vector.

        Args:
            original: Original primitives (used for types and structure).
            x: Adjusted parameter vector.

        Returns:
            New list of SketchPrimitive objects with updated params.
        """
        adjusted: List[SketchPrimitive] = []
        offset = 0
        for p in original:
            n = len(p.params)
            new_params = x[offset:offset + n].tolist()
            adjusted.append(
                SketchPrimitive(
                    primitive_type=p.primitive_type,
                    params=new_params,
                )
            )
            offset += n
        return adjusted

    @staticmethod
    def _direction_from_params(
        ptype: str, params: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Compute direction vector from primitive parameters.

        Args:
            ptype: Primitive type string.
            params: Parameter array.

        Returns:
            Unit direction (dx, dy) or None.
        """
        if ptype == "line" and len(params) >= 4:
            dx = params[2] - params[0]
            dy = params[3] - params[1]
            mag = math.sqrt(dx * dx + dy * dy)
            if mag > 1e-12:
                return (dx / mag, dy / mag)
        elif ptype == "arc" and len(params) >= 6:
            # Tangent at start: perpendicular to center->start
            cx, cy = params[4], params[5]
            rx = params[0] - cx
            ry = params[1] - cy
            mag = math.sqrt(rx * rx + ry * ry)
            if mag > 1e-12:
                return (-ry / mag, rx / mag)
        return None

    @staticmethod
    def _end_tangent_from_params(
        ptype: str, params: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Compute tangent direction at end point from primitive parameters.

        Args:
            ptype: Primitive type string.
            params: Parameter array.

        Returns:
            Unit tangent (dx, dy) at end point, or None.
        """
        if ptype == "line" and len(params) >= 4:
            dx = params[2] - params[0]
            dy = params[3] - params[1]
            mag = math.sqrt(dx * dx + dy * dy)
            if mag > 1e-12:
                return (dx / mag, dy / mag)
        elif ptype == "arc" and len(params) >= 6:
            cx, cy = params[4], params[5]
            rx = params[2] - cx
            ry = params[3] - cy
            mag = math.sqrt(rx * rx + ry * ry)
            if mag > 1e-12:
                return (-ry / mag, rx / mag)
        return None

    @staticmethod
    def _length_from_params(ptype: str, params: np.ndarray) -> float:
        """Compute length from primitive parameters.

        Args:
            ptype: Primitive type string.
            params: Parameter array.

        Returns:
            Length of the primitive.
        """
        if ptype == "line" and len(params) >= 4:
            dx = params[2] - params[0]
            dy = params[3] - params[1]
            return math.sqrt(dx * dx + dy * dy)
        elif ptype == "circle" and len(params) >= 3:
            return 2.0 * math.pi * abs(params[2])
        elif ptype == "arc" and len(params) >= 6:
            cx, cy = params[4], params[5]
            r = math.sqrt((params[0] - cx) ** 2 + (params[1] - cy) ** 2)
            a1 = math.atan2(params[1] - cy, params[0] - cx)
            a2 = math.atan2(params[3] - cy, params[2] - cx)
            angle = abs(a2 - a1)
            if angle > math.pi:
                angle = 2 * math.pi - angle
            return r * angle
        return 0.0

    @staticmethod
    def _radius_from_params(
        ptype: str, params: np.ndarray
    ) -> Optional[float]:
        """Compute radius from primitive parameters.

        Args:
            ptype: Primitive type string.
            params: Parameter array.

        Returns:
            Radius or None if not applicable.
        """
        if ptype == "circle" and len(params) >= 3:
            return abs(params[2])
        elif ptype == "arc" and len(params) >= 6:
            cx, cy = params[4], params[5]
            return math.sqrt((params[0] - cx) ** 2 + (params[1] - cy) ** 2)
        return None
