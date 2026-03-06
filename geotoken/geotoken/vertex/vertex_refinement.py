"""Coarse-to-fine vertex refinement for generated meshes.

After neural networks predict vertex positions at quantization-grid
resolution, this module refines them to sub-grid accuracy using:

1. **Chamfer gradient descent**: Move vertices toward the nearest
   points on a reference surface (if available).
2. **Face quality optimization**: Improve triangle aspect ratios and
   area uniformity.
3. **Constraint application**: Enforce positional constraints such as
   fixed vertices, symmetry planes, or planarity.
4. **Scipy L-BFGS-B optimization**: General-purpose vertex position
   optimization with configurable objectives.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy import sparse as sp_sparse

    _has_scipy_sparse = True
except ImportError:  # pragma: no cover
    _has_scipy_sparse = False

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class RefinementResult:
    """Result of iterative vertex refinement."""

    refined_vertices: np.ndarray  # (N, 3)
    iterations: int
    converged: bool
    initial_error: float
    final_error: float
    error_history: List[float] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Result of scipy-based vertex optimization."""

    optimized_vertices: np.ndarray  # (N, 3)
    success: bool
    message: str
    iterations: int
    final_error: float


# ---------------------------------------------------------------------------
# CoarseToFineRefiner
# ---------------------------------------------------------------------------


class CoarseToFineRefiner:
    """Refine coarse vertex predictions to sub-quantization accuracy.

    Takes vertex positions that were predicted at discrete grid resolution
    (e.g., 256 levels over [-1, 1]) and optimizes them toward a continuous
    target by iteratively computing position updates.

    The refinement objective can combine:

    - **Target fitting**: Minimize Chamfer distance to a reference point
      cloud (e.g., ground-truth surface samples).
    - **Face quality**: Penalize elongated or degenerate triangles.
    - **Smoothness**: Penalize large differences between adjacent
      vertex positions (Laplacian smoothing).

    Args:
        max_iterations: Maximum number of gradient-descent steps.
        learning_rate: Step size for position updates.
        convergence_threshold: Stop when error change is below this.
        face_quality_weight: Weight for face quality term (0.0 to disable).
        smoothness_weight: Weight for Laplacian smoothing term (0.0 to disable).

    Example::

        refiner = CoarseToFineRefiner(max_iterations=20, learning_rate=0.05)
        result = refiner.refine(
            coarse_vertices=predicted_verts,
            target_points=ground_truth_samples,
            face_indices=predicted_faces,
        )
        print(f"Converged: {result.converged}, Error: {result.final_error:.6f}")
    """

    def __init__(
        self,
        max_iterations: int = 10,
        learning_rate: float = 0.01,
        convergence_threshold: float = 1e-6,
        face_quality_weight: float = 0.0,
        smoothness_weight: float = 0.0,
    ) -> None:
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.face_quality_weight = face_quality_weight
        self.smoothness_weight = smoothness_weight

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def refine(
        self,
        coarse_vertices: np.ndarray,
        target_points: Optional[np.ndarray] = None,
        face_indices: Optional[np.ndarray] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        output_dtype: Optional[np.dtype] = None,
    ) -> RefinementResult:
        """Iteratively refine vertex positions.

        At each iteration:
        1. Compute gradient of the fitting objective.
        2. Optionally add face-quality and smoothness gradients.
        3. Update positions by ``learning_rate * gradient``.
        4. Apply hard constraints (fixed vertices, etc.).
        5. Check convergence.

        .. note::

            Refinement is performed internally in float64 for numerical
            stability.  The returned vertices are cast to ``output_dtype``
            (default ``np.float32``), which may reduce precision.  Pass
            ``output_dtype=np.float64`` to preserve full precision.

        Args:
            coarse_vertices: Initial predictions ``(N, 3)`` float.
            target_points: Reference surface point cloud ``(M, 3)``.
                If ``None``, only face quality and smoothness are used.
            face_indices: Triangle indices ``(F, 3)`` int.  Required
                for face quality and smoothness objectives.
            constraints: List of constraint dicts.  Supported types:

                - ``{"type": "fixed", "vertex_index": int, "position": [x, y, z]}``
                - ``{"type": "planar", "vertex_indices": [int, ...], "normal": [nx, ny, nz], "d": float}``
                - ``{"type": "symmetric", "pairs": [(i, j), ...], "axis": int}``

            output_dtype: NumPy dtype for the returned vertex array.
                Defaults to ``np.float32``.  Use ``np.float64`` to avoid
                precision loss from the internal float64 computation.

        Returns:
            ``RefinementResult`` with refined vertices and convergence info.
        """
        if output_dtype is None:
            output_dtype = np.float32
        vertices = coarse_vertices.astype(np.float64).copy()
        error_history: List[float] = []

        # Pre-build KDTree for target if available
        target_tree = None
        if target_points is not None:
            try:
                from scipy.spatial import KDTree
                target_tree = KDTree(target_points.astype(np.float64))
            except ImportError:
                _log.warning("scipy not available, using brute-force nearest neighbor")

        # Pre-build adjacency for smoothness
        adjacency = None
        if self.smoothness_weight > 0 and face_indices is not None:
            adjacency = self._build_adjacency(vertices, face_indices)

        # Compute initial error
        initial_error = self._compute_error(
            vertices, target_points, target_tree, face_indices
        )
        error_history.append(initial_error)

        for iteration in range(self.max_iterations):
            # 1. Compute total gradient
            gradient = np.zeros_like(vertices)

            # Chamfer gradient (toward target surface)
            if target_points is not None:
                chamfer_grad = self._compute_chamfer_gradient(
                    vertices, target_points, target_tree
                )
                gradient += chamfer_grad

            # Face quality gradient (improve triangle shapes)
            if self.face_quality_weight > 0 and face_indices is not None:
                quality_grad = self._compute_face_quality_gradient(
                    vertices, face_indices
                )
                gradient += self.face_quality_weight * quality_grad

            # Smoothness gradient (Laplacian smoothing)
            if self.smoothness_weight > 0 and adjacency is not None:
                smooth_grad = self._compute_smoothness_gradient(
                    vertices, adjacency
                )
                gradient += self.smoothness_weight * smooth_grad

            # 2. Update positions
            vertices += self.learning_rate * gradient

            # 3. Apply constraints
            if constraints:
                vertices = self._apply_constraints(vertices, constraints)

            # 4. Check convergence
            error = self._compute_error(
                vertices, target_points, target_tree, face_indices
            )
            error_history.append(error)

            error_change = abs(error_history[-2] - error)
            if error_change < self.convergence_threshold:
                _log.debug(
                    "Converged at iteration %d (error change %.2e)",
                    iteration + 1, error_change,
                )
                return RefinementResult(
                    refined_vertices=vertices.astype(output_dtype),
                    iterations=iteration + 1,
                    converged=True,
                    initial_error=initial_error,
                    final_error=error,
                    error_history=error_history,
                )

        _log.debug(
            "Reached max iterations (%d), final error %.6f",
            self.max_iterations, error_history[-1],
        )
        return RefinementResult(
            refined_vertices=vertices.astype(output_dtype),
            iterations=self.max_iterations,
            converged=False,
            initial_error=initial_error,
            final_error=error_history[-1],
            error_history=error_history,
        )

    # ------------------------------------------------------------------
    # Gradient computations
    # ------------------------------------------------------------------

    def _compute_chamfer_gradient(
        self,
        vertices: np.ndarray,
        target_points: np.ndarray,
        target_tree: Any = None,
    ) -> np.ndarray:
        """Gradient of one-directional Chamfer distance.

        For each vertex, compute the direction toward its nearest
        point on the target surface.  This is the negative gradient
        of ``sum ||v_i - nn(v_i)||^2``.

        Args:
            vertices: Current positions ``(N, 3)``.
            target_points: Reference surface ``(M, 3)``.
            target_tree: Pre-built KDTree (optional, for speed).

        Returns:
            Gradient array ``(N, 3)`` pointing toward target.
        """
        if target_tree is not None:
            _, nearest_idx = target_tree.query(vertices)
            nearest_points = target_points[nearest_idx]
        else:
            # Brute-force fallback
            # Compute distances: (N, M)
            diffs = vertices[:, np.newaxis, :] - target_points[np.newaxis, :, :]
            dists_sq = (diffs ** 2).sum(axis=2)
            nearest_idx = dists_sq.argmin(axis=1)
            nearest_points = target_points[nearest_idx]

        # Gradient: direction from vertex to nearest target point
        gradient = nearest_points - vertices
        return gradient

    def _compute_face_quality_gradient(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> np.ndarray:
        """Gradient encouraging equilateral triangles.

        Penalizes faces whose edge-length ratios deviate from 1.0.
        Each vertex receives a gradient contribution from each of
        its incident faces pushing it toward a more regular triangle.

        Args:
            vertices: ``(N, 3)`` positions.
            faces: ``(F, 3)`` triangle indices.

        Returns:
            Gradient ``(N, 3)``.
        """
        gradient = np.zeros_like(vertices)

        if faces.size == 0:
            return gradient

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Edge vectors
        e01 = v1 - v0
        e12 = v2 - v1
        e20 = v0 - v2

        # Edge lengths
        l01 = np.linalg.norm(e01, axis=1, keepdims=True) + 1e-10
        l12 = np.linalg.norm(e12, axis=1, keepdims=True) + 1e-10
        l20 = np.linalg.norm(e20, axis=1, keepdims=True) + 1e-10

        # Target: all edges equal to mean edge length
        mean_l = (l01 + l12 + l20) / 3.0

        # Gradient: push edges toward mean length
        # For edge (v0, v1): if too long, pull together; if too short, push apart
        scale_01 = (mean_l - l01) / l01
        scale_12 = (mean_l - l12) / l12
        scale_20 = (mean_l - l20) / l20

        # Accumulate gradients per vertex
        grad_v0 = -scale_01 * (e01 / l01) + scale_20 * (e20 / l20)
        grad_v1 = scale_01 * (e01 / l01) - scale_12 * (e12 / l12)
        grad_v2 = scale_12 * (e12 / l12) - scale_20 * (e20 / l20)

        np.add.at(gradient, faces[:, 0], grad_v0)
        np.add.at(gradient, faces[:, 1], grad_v1)
        np.add.at(gradient, faces[:, 2], grad_v2)

        return gradient

    def _compute_smoothness_gradient(
        self,
        vertices: np.ndarray,
        adjacency: Any,
    ) -> np.ndarray:
        """Laplacian smoothing gradient.

        Each vertex moves toward the centroid of its neighbors.

        When *adjacency* is a sparse Laplacian matrix ``L``
        (built by ``_build_adjacency`` with scipy), the gradient is
        computed as a single sparse matrix–dense matrix multiply:
        ``gradient = L @ vertices``.

        Falls back to a Python loop when adjacency is a dict.

        Args:
            vertices: ``(N, 3)`` positions.
            adjacency: Sparse Laplacian ``(N, N)`` or dict fallback.

        Returns:
            Gradient ``(N, 3)``.
        """
        # Fast path: sparse Laplacian matrix
        if _has_scipy_sparse and sp_sparse.issparse(adjacency):
            return adjacency @ vertices

        # Fallback: Python dict iteration
        gradient = np.zeros_like(vertices)
        for i, neighbors in adjacency.items():
            if not neighbors:
                continue
            neighbor_center = vertices[neighbors].mean(axis=0)
            gradient[i] = neighbor_center - vertices[i]
        return gradient

    # ------------------------------------------------------------------
    # Error computation
    # ------------------------------------------------------------------

    def _compute_error(
        self,
        vertices: np.ndarray,
        target_points: Optional[np.ndarray],
        target_tree: Any,
        face_indices: Optional[np.ndarray],
    ) -> float:
        """Compute total error for current vertex positions."""
        error = 0.0

        if target_points is not None:
            if target_tree is not None:
                dists, _ = target_tree.query(vertices)
                error += float(np.mean(dists ** 2))
            else:
                diffs = vertices[:, np.newaxis, :] - target_points[np.newaxis, :, :]
                dists_sq = (diffs ** 2).sum(axis=2)
                error += float(np.mean(dists_sq.min(axis=1)))

        if self.face_quality_weight > 0 and face_indices is not None and face_indices.size > 0:
            v0 = vertices[face_indices[:, 0]]
            v1 = vertices[face_indices[:, 1]]
            v2 = vertices[face_indices[:, 2]]

            l01 = np.linalg.norm(v1 - v0, axis=1)
            l12 = np.linalg.norm(v2 - v1, axis=1)
            l20 = np.linalg.norm(v0 - v2, axis=1)

            mean_l = (l01 + l12 + l20) / 3.0
            variance = ((l01 - mean_l) ** 2 + (l12 - mean_l) ** 2 + (l20 - mean_l) ** 2) / 3.0
            error += self.face_quality_weight * float(np.mean(variance))

        return error

    # ------------------------------------------------------------------
    # Constraint application
    # ------------------------------------------------------------------

    def _apply_constraints(
        self,
        vertices: np.ndarray,
        constraints: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Apply hard positional constraints to vertices.

        Supported constraint types:

        - ``"fixed"``: Pin a vertex to a specific position.
        - ``"planar"``: Project vertices onto a plane.
        - ``"symmetric"``: Enforce mirror symmetry between vertex pairs.

        Args:
            vertices: ``(N, 3)`` mutable vertex array.
            constraints: List of constraint specification dicts.

        Returns:
            Constrained vertex array ``(N, 3)``.
        """
        for constraint in constraints:
            ctype = constraint.get("type", "")

            if ctype == "fixed":
                idx = constraint["vertex_index"]
                pos = constraint["position"]
                if 0 <= idx < len(vertices):
                    vertices[idx] = pos

            elif ctype == "planar":
                indices = constraint["vertex_indices"]
                normal = np.array(constraint["normal"], dtype=np.float64)
                normal = normal / (np.linalg.norm(normal) + 1e-10)
                d = constraint.get("d", 0.0)

                for idx in indices:
                    if 0 <= idx < len(vertices):
                        dist = np.dot(vertices[idx], normal) - d
                        vertices[idx] -= dist * normal

            elif ctype == "symmetric":
                pairs = constraint["pairs"]
                axis = constraint.get("axis", 0)  # 0=x, 1=y, 2=z

                for i, j in pairs:
                    if 0 <= i < len(vertices) and 0 <= j < len(vertices):
                        avg = (vertices[i] + vertices[j]) / 2.0
                        vertices[i] = avg.copy()
                        vertices[j] = avg.copy()
                        # Mirror on the specified axis
                        vertices[i][axis] = abs(avg[axis])
                        vertices[j][axis] = -abs(avg[axis])

            else:
                _log.warning("Unknown constraint type '%s', skipping", ctype)

        return vertices

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _build_adjacency(
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> Any:
        """Build vertex adjacency as a sparse Laplacian matrix.

        When scipy is available, returns a CSR sparse matrix ``L`` where
        ``L @ vertices`` computes the Laplacian (neighbor-centroid minus
        vertex position) for every vertex in one operation.

        Falls back to a Python dict adjacency list when scipy is not
        installed.

        Args:
            vertices: ``(N, 3)`` positions (used for vertex count).
            faces: ``(F, 3)`` triangle indices.

        Returns:
            Sparse CSR matrix ``(N, N)`` or dict adjacency fallback.
        """
        n_verts = len(vertices)

        if _has_scipy_sparse:
            # Extract directed edges from triangles (vectorised)
            i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
            rows = np.concatenate([i0, i1, i2, i1, i2, i0])
            cols = np.concatenate([i1, i2, i0, i0, i1, i2])

            # Build binary adjacency (duplicates summed, but we normalise)
            data = np.ones(len(rows), dtype=np.float64)
            adj = sp_sparse.csr_matrix(
                (data, (rows, cols)), shape=(n_verts, n_verts)
            )
            # Remove duplicate edges (clamp to 1)
            adj.data[:] = 1.0

            # Row-normalised adjacency: each row sums to 1
            degree = np.array(adj.sum(axis=1)).ravel()
            degree[degree == 0] = 1.0  # avoid division by zero
            inv_degree = 1.0 / degree
            # D^{-1} A  — row-normalised adjacency
            norm_adj = sp_sparse.diags(inv_degree) @ adj
            # Laplacian: L = D^{-1}A - I  (so L @ v = centroid - v)
            laplacian = norm_adj - sp_sparse.eye(n_verts, format="csr")
            return laplacian

        # Fallback: Python dict adjacency
        adjacency_sets: Dict[int, set] = {
            i: set() for i in range(n_verts)
        }
        for fi in range(len(faces)):
            v0, v1, v2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            for a, b in [(v0, v1), (v1, v2), (v2, v0)]:
                adjacency_sets[a].add(b)
                adjacency_sets[b].add(a)
        adjacency: Dict[int, List[int]] = {
            k: list(v) for k, v in adjacency_sets.items()
        }
        return adjacency


# ---------------------------------------------------------------------------
# VertexPositionOptimizer
# ---------------------------------------------------------------------------


class VertexPositionOptimizer:
    """General-purpose vertex optimization using scipy.

    Wraps ``scipy.optimize.minimize`` for vertex position refinement
    with configurable objective functions and coordinate bounds.

    This is more flexible than ``CoarseToFineRefiner`` but requires
    a user-defined objective function.

    Args:
        method: Scipy optimization method (default ``"L-BFGS-B"``).
        max_iterations: Maximum optimizer iterations.
        tolerance: Convergence tolerance for the optimizer.

    Example::

        optimizer = VertexPositionOptimizer()

        def chamfer_obj(verts):
            diffs = verts[:, None, :] - target[None, :, :]
            return np.mean(np.min(np.sum(diffs**2, axis=2), axis=1))

        result = optimizer.optimize(coarse_verts, chamfer_obj)
    """

    def __init__(
        self,
        method: str = "L-BFGS-B",
        max_iterations: int = 100,
        tolerance: float = 1e-8,
    ) -> None:
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(
        self,
        vertices: np.ndarray,
        objective_fn: Callable[[np.ndarray], float],
        gradient_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> OptimizationResult:
        """Optimize vertex positions to minimize an objective.

        Args:
            vertices: Initial positions ``(N, 3)`` float.
            objective_fn: Callable that takes ``(N, 3)`` array and
                returns a scalar error.
            gradient_fn: Optional callable that takes ``(N, 3)`` and
                returns ``(N, 3)`` gradient.  If ``None``, scipy uses
                finite differences.
            bounds: Optional ``(min_coord, max_coord)`` tuple applied
                to all coordinate values.

        Returns:
            ``OptimizationResult`` with optimized vertices.
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            _log.error("scipy not available; cannot run VertexPositionOptimizer")
            return OptimizationResult(
                optimized_vertices=vertices.copy(),
                success=False,
                message="scipy not available",
                iterations=0,
                final_error=float("inf"),
            )

        x0 = vertices.astype(np.float64).flatten()

        def flat_objective(x: np.ndarray) -> float:
            return objective_fn(x.reshape(-1, 3))

        flat_grad = None
        if gradient_fn is not None:
            def flat_grad(x: np.ndarray) -> np.ndarray:
                return gradient_fn(x.reshape(-1, 3)).flatten()

        # Build bounds list if specified
        scipy_bounds = None
        if bounds is not None:
            scipy_bounds = [(bounds[0], bounds[1])] * len(x0)

        result = minimize(
            flat_objective,
            x0,
            jac=flat_grad,
            method=self.method,
            bounds=scipy_bounds,
            options={
                "maxiter": self.max_iterations,
                "ftol": self.tolerance,
            },
        )

        optimized = result.x.reshape(-1, 3).astype(np.float32)

        return OptimizationResult(
            optimized_vertices=optimized,
            success=bool(result.success),
            message=str(result.message),
            iterations=int(result.nit) if hasattr(result, "nit") else 0,
            final_error=float(result.fun),
        )
