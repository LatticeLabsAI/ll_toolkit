"""Vertex validation for predicted mesh geometry.

Provides comprehensive checks on predicted vertex positions and mesh
topology to ensure geometric feasibility before reconstruction.

Checks include:
- Coordinate bounds (vertices within expected ranges)
- Collision / near-duplicate detection (KDTree-based, O(n log n))
- Degenerate face detection (zero-area triangles)
- Edge-manifold property (each edge shared by ≤ 2 faces)
- Consistent face winding
- Watertight / closed mesh
- Euler characteristic (V - E + F = 2 for genus-0)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class BoundsCheckResult:
    """Result of bounds checking."""

    all_in_bounds: bool
    out_of_bounds_indices: np.ndarray  # indices of violating vertices
    max_violation: float  # how far the worst vertex is from bounds
    num_violations: int


@dataclass
class CollisionCheckResult:
    """Result of near-duplicate / collision checking."""

    collision_free: bool
    collision_pairs: list[Tuple[int, int]]  # pairs of vertices that are too close
    min_distance: float  # smallest pairwise distance found
    num_collisions: int


@dataclass
class DegeneracyCheckResult:
    """Result of degenerate face detection."""

    has_degenerate: bool
    degenerate_face_indices: list[int]
    zero_area_count: int
    min_area: float


@dataclass
class ManifoldCheckResult:
    """Result of edge-manifold checking."""

    is_manifold: bool
    non_manifold_edges: list[Tuple[int, int]]  # edge vertex pairs
    boundary_edges: list[Tuple[int, int]]  # edges with only 1 face
    num_non_manifold: int
    num_boundary: int


@dataclass
class WindingCheckResult:
    """Result of face winding consistency check."""

    consistent: bool
    num_inconsistent: int
    inconsistent_face_indices: list[int]


@dataclass
class EulerCheckResult:
    """Result of Euler characteristic check."""

    valid: bool
    V: int  # vertices
    E: int  # edges
    F: int  # faces
    euler: int  # V - E + F
    expected_euler: int  # typically 2 for genus-0


@dataclass
class VertexValidationReport:
    """Complete validation report combining all checks."""

    valid: bool
    bounds: Optional[BoundsCheckResult] = None
    collisions: Optional[CollisionCheckResult] = None
    degeneracy: Optional[DegeneracyCheckResult] = None
    manifold: Optional[ManifoldCheckResult] = None
    winding: Optional[WindingCheckResult] = None
    euler: Optional[EulerCheckResult] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# VertexValidator
# ---------------------------------------------------------------------------


class VertexValidator:
    """Validate predicted vertex positions for geometric feasibility.

    Runs configurable checks on vertex arrays and optional face arrays
    to detect common issues in generated meshes before passing them
    to reconstruction.

    Args:
        coord_bounds: (min, max) coordinate range. Vertices outside
            this range are flagged.
        collision_tolerance: Minimum allowed distance between distinct
            vertices. Pairs closer than this are flagged.
        area_tolerance: Minimum allowed face area. Faces with smaller
            area are considered degenerate.
        manifold_check: Whether to run edge-manifold validation.

    Example::

        validator = VertexValidator(coord_bounds=(-1.0, 1.0))
        report = validator.validate(vertices, faces)
        if not report.valid:
            for err in report.errors:
                print(f"Error: {err}")
    """

    def __init__(
        self,
        coord_bounds: Tuple[float, float] = (-100.0, 100.0),
        collision_tolerance: float = 1e-4,
        area_tolerance: float = 1e-10,
        manifold_check: bool = True,
    ) -> None:
        self.coord_min, self.coord_max = coord_bounds
        self.collision_tol = collision_tolerance
        self.area_tol = area_tolerance
        self.do_manifold = manifold_check

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def validate(
        self,
        vertices: np.ndarray,
        faces: Optional[np.ndarray] = None,
    ) -> VertexValidationReport:
        """Run all configured validation checks.

        Args:
            vertices: Vertex positions ``(N, 3)`` float.
            faces: Triangle indices ``(F, 3)`` int, optional.

        Returns:
            ``VertexValidationReport`` with per-check results and
            aggregated errors / warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # --- Shape validation ---
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            errors.append(
                f"Vertices must be (N, 3), got shape {vertices.shape}"
            )
            return VertexValidationReport(
                valid=False, errors=errors, warnings=warnings
            )

        # --- Bounds ---
        bounds = self.check_bounds(vertices)
        if not bounds.all_in_bounds:
            errors.append(
                f"{bounds.num_violations} vertices out of bounds "
                f"[{self.coord_min}, {self.coord_max}], "
                f"max violation {bounds.max_violation:.6f}"
            )

        # --- Collisions ---
        collisions = self.check_collisions(vertices)
        if not collisions.collision_free:
            warnings.append(
                f"{collisions.num_collisions} near-duplicate vertex pairs "
                f"(min distance {collisions.min_distance:.2e})"
            )

        # --- Face-dependent checks ---
        degeneracy = None
        manifold = None
        winding = None
        euler = None

        if faces is not None:
            if faces.ndim != 2 or faces.shape[1] != 3:
                errors.append(
                    f"Faces must be (F, 3), got shape {faces.shape}"
                )
            else:
                # Index validity
                if faces.size > 0:
                    if faces.min() < 0 or faces.max() >= len(vertices):
                        errors.append(
                            f"Face indices out of range [0, {len(vertices)}): "
                            f"min={faces.min()}, max={faces.max()}"
                        )
                    else:
                        # Degeneracy
                        degeneracy = self.check_degeneracy(vertices, faces)
                        if degeneracy.has_degenerate:
                            warnings.append(
                                f"{degeneracy.zero_area_count} degenerate faces "
                                f"(area < {self.area_tol})"
                            )

                        # Manifold
                        if self.do_manifold:
                            manifold = self.check_manifold(faces)
                            if not manifold.is_manifold:
                                errors.append(
                                    f"{manifold.num_non_manifold} non-manifold edges"
                                )
                            if manifold.num_boundary > 0:
                                warnings.append(
                                    f"{manifold.num_boundary} boundary edges "
                                    "(mesh not closed)"
                                )

                        # Winding
                        winding = self.check_face_winding(vertices, faces)
                        if not winding.consistent:
                            warnings.append(
                                f"{winding.num_inconsistent} faces with "
                                "inconsistent winding"
                            )

                        # Euler
                        euler = self.check_euler(faces)

        valid = len(errors) == 0

        report = VertexValidationReport(
            valid=valid,
            bounds=bounds,
            collisions=collisions,
            degeneracy=degeneracy,
            manifold=manifold,
            winding=winding,
            euler=euler,
            errors=errors,
            warnings=warnings,
        )

        _log.debug(
            "Validation %s: %d errors, %d warnings",
            "PASSED" if valid else "FAILED",
            len(errors),
            len(warnings),
        )
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_bounds(self, vertices: np.ndarray) -> BoundsCheckResult:
        """Check all vertices are within ``[coord_min, coord_max]``.

        Args:
            vertices: ``(N, 3)`` vertex positions.

        Returns:
            ``BoundsCheckResult`` with violation details.
        """
        below = vertices < self.coord_min
        above = vertices > self.coord_max
        violations = below | above

        out_of_bounds_mask = violations.any(axis=1)
        out_of_bounds_indices = np.where(out_of_bounds_mask)[0]

        max_violation = 0.0
        if out_of_bounds_indices.size > 0:
            below_amount = np.maximum(0.0, self.coord_min - vertices)
            above_amount = np.maximum(0.0, vertices - self.coord_max)
            max_violation = float(
                np.maximum(below_amount, above_amount).max()
            )

        return BoundsCheckResult(
            all_in_bounds=out_of_bounds_indices.size == 0,
            out_of_bounds_indices=out_of_bounds_indices,
            max_violation=max_violation,
            num_violations=int(out_of_bounds_indices.size),
        )

    def check_collisions(self, vertices: np.ndarray) -> CollisionCheckResult:
        """Check for vertex pairs closer than ``collision_tolerance``.

        Uses ``scipy.spatial.KDTree`` for ``O(n log n)`` performance
        instead of ``O(n²)`` brute-force pairwise comparison.

        Args:
            vertices: ``(N, 3)`` vertex positions.

        Returns:
            ``CollisionCheckResult`` with collision pair details.
        """
        if len(vertices) < 2:
            return CollisionCheckResult(
                collision_free=True,
                collision_pairs=[],
                min_distance=float("inf"),
                num_collisions=0,
            )

        try:
            from scipy.spatial import KDTree

            tree = KDTree(vertices)
            pairs_set: set[Tuple[int, int]] = set()

            # Query all pairs within tolerance
            close_pairs = tree.query_pairs(r=self.collision_tol)
            for i, j in close_pairs:
                pairs_set.add((min(i, j), max(i, j)))

            # Get actual minimum distance
            dists, _ = tree.query(vertices, k=2)
            min_distance = float(dists[:, 1].min()) if len(vertices) > 1 else float("inf")

        except ImportError:
            _log.debug("scipy not available, using brute-force collision check")
            # Fallback: brute-force for small meshes
            pairs_set = set()
            min_distance = float("inf")

            if len(vertices) <= 5000:
                diffs = vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :]
                dists_sq = (diffs ** 2).sum(axis=2)
                np.fill_diagonal(dists_sq, np.inf)
                min_distance = float(np.sqrt(dists_sq.min()))

                close_i, close_j = np.where(
                    dists_sq < self.collision_tol ** 2
                )
                for i, j in zip(close_i, close_j):
                    if i < j:
                        pairs_set.add((int(i), int(j)))
            else:
                _log.warning(
                    "Skipping collision check for %d vertices (scipy unavailable)",
                    len(vertices),
                )

        collision_pairs = sorted(pairs_set)

        return CollisionCheckResult(
            collision_free=len(collision_pairs) == 0,
            collision_pairs=collision_pairs,
            min_distance=min_distance,
            num_collisions=len(collision_pairs),
        )

    def check_degeneracy(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> DegeneracyCheckResult:
        """Check for degenerate faces (zero or near-zero area).

        A face is degenerate if its three vertices are collinear or
        coincident, producing a triangle with area below ``area_tolerance``.

        Args:
            vertices: ``(N, 3)`` vertex positions.
            faces: ``(F, 3)`` triangle indices.

        Returns:
            ``DegeneracyCheckResult`` with degenerate face indices.
        """
        if faces.size == 0:
            return DegeneracyCheckResult(
                has_degenerate=False,
                degenerate_face_indices=[],
                zero_area_count=0,
                min_area=float("inf"),
            )

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Area = 0.5 * ||(v1 - v0) × (v2 - v0)||
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        degenerate_mask = areas < self.area_tol
        degenerate_indices = np.where(degenerate_mask)[0].tolist()

        return DegeneracyCheckResult(
            has_degenerate=len(degenerate_indices) > 0,
            degenerate_face_indices=degenerate_indices,
            zero_area_count=len(degenerate_indices),
            min_area=float(areas.min()) if areas.size > 0 else float("inf"),
        )

    def check_manifold(self, faces: np.ndarray) -> ManifoldCheckResult:
        """Check edge-manifold property of the mesh.

        A mesh is edge-manifold if every edge is shared by exactly
        1 face (boundary) or 2 faces (interior). An edge shared by
        3+ faces indicates non-manifold geometry.

        Args:
            faces: ``(F, 3)`` triangle indices.

        Returns:
            ``ManifoldCheckResult`` with non-manifold and boundary edges.
        """
        if faces.size == 0:
            return ManifoldCheckResult(
                is_manifold=True,
                non_manifold_edges=[],
                boundary_edges=[],
                num_non_manifold=0,
                num_boundary=0,
            )

        edge_counts: dict[Tuple[int, int], int] = {}

        for face_idx in range(len(faces)):
            v0, v1, v2 = int(faces[face_idx, 0]), int(faces[face_idx, 1]), int(faces[face_idx, 2])
            for edge in [(v0, v1), (v1, v2), (v2, v0)]:
                key = (min(edge), max(edge))
                edge_counts[key] = edge_counts.get(key, 0) + 1

        non_manifold = [e for e, c in edge_counts.items() if c > 2]
        boundary = [e for e, c in edge_counts.items() if c == 1]

        return ManifoldCheckResult(
            is_manifold=len(non_manifold) == 0,
            non_manifold_edges=non_manifold,
            boundary_edges=boundary,
            num_non_manifold=len(non_manifold),
            num_boundary=len(boundary),
        )

    def check_face_winding(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> WindingCheckResult:
        """Check face winding consistency using normal orientation.

        For a closed mesh, consistent winding means adjacent faces have
        normals pointing in compatible directions (both outward). We
        check by verifying that normals of adjacent faces don't point
        in opposite directions through the shared edge.

        Args:
            vertices: ``(N, 3)`` vertex positions.
            faces: ``(F, 3)`` triangle indices.

        Returns:
            ``WindingCheckResult`` with inconsistency count.
        """
        if faces.size == 0:
            return WindingCheckResult(
                consistent=True, num_inconsistent=0,
                inconsistent_face_indices=[],
            )

        # Compute face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
        magnitudes = np.maximum(magnitudes, 1e-10)
        normals = normals / magnitudes

        # Build edge → face mapping for adjacency
        edge_to_faces: dict[Tuple[int, int], list[int]] = {}
        for fi in range(len(faces)):
            v_a, v_b, v_c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            for edge in [(v_a, v_b), (v_b, v_c), (v_c, v_a)]:
                key = (min(edge), max(edge))
                if key not in edge_to_faces:
                    edge_to_faces[key] = []
                edge_to_faces[key].append(fi)

        # Check adjacent face normal consistency
        inconsistent_faces: set[int] = set()
        for _edge, face_list in edge_to_faces.items():
            if len(face_list) == 2:
                fi, fj = face_list[0], face_list[1]
                dot = np.dot(normals[fi], normals[fj])
                # If normals point in nearly opposite directions through
                # a shared edge, winding is likely inconsistent
                if dot < -0.9:
                    inconsistent_faces.add(fi)
                    inconsistent_faces.add(fj)

        return WindingCheckResult(
            consistent=len(inconsistent_faces) == 0,
            num_inconsistent=len(inconsistent_faces),
            inconsistent_face_indices=sorted(inconsistent_faces),
        )

    def check_euler(self, faces: np.ndarray) -> EulerCheckResult:
        """Verify Euler characteristic ``V - E + F``.

        For a closed genus-0 solid (sphere-like), Euler = 2.
        Deviations indicate holes, handles, or topological issues.

        Args:
            faces: ``(F, 3)`` triangle indices.

        Returns:
            ``EulerCheckResult`` with computed values.
        """
        if faces.size == 0:
            return EulerCheckResult(
                valid=True, V=0, E=0, F=0, euler=0, expected_euler=2
            )

        # Count unique vertices
        V = int(np.unique(faces).size)
        F = len(faces)

        # Count unique edges
        edges: set[Tuple[int, int]] = set()
        for fi in range(F):
            v0, v1, v2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            for edge in [(v0, v1), (v1, v2), (v2, v0)]:
                edges.add((min(edge), max(edge)))
        E = len(edges)

        euler = V - E + F
        expected = 2  # genus-0

        return EulerCheckResult(
            valid=(euler == expected),
            V=V, E=E, F=F,
            euler=euler,
            expected_euler=expected,
        )


# ---------------------------------------------------------------------------
# TopologyValidator
# ---------------------------------------------------------------------------


class TopologyValidator:
    """Higher-level topology validation combining multiple checks.

    Wraps ``VertexValidator`` checks with topology-specific logic for
    validating predicted mesh structures before reconstruction.

    Example::

        topo = TopologyValidator()
        report = topo.validate_mesh(vertices, faces)
        print(f"Watertight: {report['watertight']}")
        print(f"Euler: {report['euler'].euler}")
    """

    def __init__(
        self,
        area_tolerance: float = 1e-10,
    ) -> None:
        self.area_tol = area_tolerance

    def validate_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> dict[str, Any]:
        """Run comprehensive topology validation on a mesh.

        Args:
            vertices: ``(N, 3)`` vertex positions.
            faces: ``(F, 3)`` triangle indices.

        Returns:
            Dict with validation results for manifold, watertight,
            euler, winding, and degeneracy checks.
        """
        validator = VertexValidator(
            area_tolerance=self.area_tol,
            manifold_check=True,
        )

        manifold = validator.check_manifold(faces)
        winding = validator.check_face_winding(vertices, faces)
        euler = validator.check_euler(faces)
        degeneracy = validator.check_degeneracy(vertices, faces)

        # Watertight = manifold + no boundary edges
        watertight = manifold.is_manifold and manifold.num_boundary == 0

        return {
            "manifold": manifold,
            "watertight": watertight,
            "euler": euler,
            "winding": winding,
            "degeneracy": degeneracy,
            "valid": (
                manifold.is_manifold
                and watertight
                and winding.consistent
                and not degeneracy.has_degenerate
            ),
        }
