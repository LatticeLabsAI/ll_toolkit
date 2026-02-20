"""Latent proposal — decoded point grids from diffusion models.

A LatentProposal carries the output of a BrepGen-style structured
diffusion model: per-face point grids (32×32×3), per-edge point
arrays (N×3), and bounding boxes for both.

The disposal engine's ``surface_executor`` fits B-spline surfaces
to the point grids, performs mating deduplication on edges across
face boundaries, trims surfaces with edge curves, and sews everything
into a closed solid via ``BRepBuilderAPI_Sewing``.
"""
from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from ll_gen.proposals.base import BaseProposal


@dataclass
class LatentProposal(BaseProposal):
    """A proposal containing decoded B-rep component geometry.

    Attributes:
        face_grids: Per-face point grids.  Each entry is an ndarray of
            shape ``(U, V, 3)`` — typically ``(32, 32, 3)`` — sampling
            the face surface in parameter space.
        edge_points: Per-edge point arrays.  Each entry is an ndarray
            of shape ``(N, 3)`` sampling the edge curve.
        face_bboxes: Face bounding boxes, shape ``(F, 6)`` where each
            row is ``(x_min, y_min, z_min, x_max, y_max, z_max)``.
        edge_bboxes: Edge bounding boxes, shape ``(E, 6)``.
        vertex_positions: Vertex positions, shape ``(V, 3)``.  Vertices
            are the endpoints/intersections of edges.
        face_edge_adjacency: Mapping from face index to list of edge
            indices that bound that face.
        stage_latents: Raw latent tensors from each diffusion stage,
            keyed by stage name (``"face_positions"``,
            ``"face_geometry"``, ``"edge_positions"``,
            ``"edge_vertex_geometry"``).  Preserved for debugging and
            ablation studies.
    """

    face_grids: List[np.ndarray] = field(default_factory=list)
    edge_points: List[np.ndarray] = field(default_factory=list)
    face_bboxes: Optional[np.ndarray] = None
    edge_bboxes: Optional[np.ndarray] = None
    vertex_positions: Optional[np.ndarray] = None
    face_edge_adjacency: Optional[Dict[int, List[int]]] = None
    stage_latents: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_faces(self) -> int:
        """Number of face grids in this proposal."""
        return len(self.face_grids)

    @property
    def num_edges(self) -> int:
        """Number of edge point arrays."""
        return len(self.edge_points)

    @property
    def num_vertices(self) -> int:
        """Number of vertex positions."""
        if self.vertex_positions is None:
            return 0
        return self.vertex_positions.shape[0]

    @property
    def face_grid_resolution(self) -> Optional[tuple]:
        """(U, V) resolution of face grids, or None if empty."""
        if not self.face_grids:
            return None
        g = self.face_grids[0]
        return (g.shape[0], g.shape[1])

    @property
    def total_points(self) -> int:
        """Total number of 3D sample points across all faces and edges."""
        face_pts = sum(g.shape[0] * g.shape[1] for g in self.face_grids)
        edge_pts = sum(e.shape[0] for e in self.edge_points)
        return face_pts + edge_pts

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_shapes(self) -> List[str]:
        """Check internal consistency of array shapes.

        Returns:
            List of error messages.  Empty list means all checks pass.
        """
        errors: List[str] = []

        for i, g in enumerate(self.face_grids):
            if g.ndim != 3:
                errors.append(
                    f"face_grids[{i}] has ndim={g.ndim}, expected 3"
                )
            elif g.shape[2] != 3:
                errors.append(
                    f"face_grids[{i}] last dim={g.shape[2]}, expected 3"
                )

        for i, e in enumerate(self.edge_points):
            if e.ndim != 2:
                errors.append(
                    f"edge_points[{i}] has ndim={e.ndim}, expected 2"
                )
            elif e.shape[1] != 3:
                errors.append(
                    f"edge_points[{i}] last dim={e.shape[1]}, expected 3"
                )

        if self.face_bboxes is not None:
            if self.face_bboxes.ndim != 2 or self.face_bboxes.shape[1] != 6:
                errors.append(
                    f"face_bboxes shape={self.face_bboxes.shape}, "
                    f"expected (F, 6)"
                )
            elif self.face_bboxes.shape[0] != self.num_faces:
                errors.append(
                    f"face_bboxes has {self.face_bboxes.shape[0]} rows "
                    f"but {self.num_faces} face grids"
                )

        if self.edge_bboxes is not None:
            if self.edge_bboxes.ndim != 2 or self.edge_bboxes.shape[1] != 6:
                errors.append(
                    f"edge_bboxes shape={self.edge_bboxes.shape}, "
                    f"expected (E, 6)"
                )
            elif self.edge_bboxes.shape[0] != self.num_edges:
                errors.append(
                    f"edge_bboxes has {self.edge_bboxes.shape[0]} rows "
                    f"but {self.num_edges} edge point arrays"
                )

        if self.vertex_positions is not None:
            if (
                self.vertex_positions.ndim != 2
                or self.vertex_positions.shape[1] != 3
            ):
                errors.append(
                    f"vertex_positions shape={self.vertex_positions.shape}, "
                    f"expected (V, 3)"
                )

        return errors

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def compute_bounding_box(self) -> Optional[np.ndarray]:
        """Compute overall axis-aligned bounding box from all points.

        Returns:
            Array of shape ``(6,)`` as
            ``[x_min, y_min, z_min, x_max, y_max, z_max]``,
            or None if no geometry data is present.
        """
        all_points: List[np.ndarray] = []

        for g in self.face_grids:
            all_points.append(g.reshape(-1, 3))
        for e in self.edge_points:
            all_points.append(e)
        if self.vertex_positions is not None:
            all_points.append(self.vertex_positions)

        if not all_points:
            return None

        pts = np.concatenate(all_points, axis=0)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        return np.concatenate([mins, maxs])

    def compute_face_areas_approximate(self) -> List[float]:
        """Approximate face areas from point grid spacing.

        Uses the cross-product method on the UV grid to estimate
        surface area per face.

        Returns:
            List of approximate areas for each face.
        """
        areas: List[float] = []
        for g in self.face_grids:
            U, V, _ = g.shape
            area = 0.0
            for u in range(U - 1):
                for v in range(V - 1):
                    # Two triangles per grid cell
                    p00 = g[u, v]
                    p10 = g[u + 1, v]
                    p01 = g[u, v + 1]
                    p11 = g[u + 1, v + 1]
                    # Triangle 1: p00, p10, p01
                    area += 0.5 * np.linalg.norm(
                        np.cross(p10 - p00, p01 - p00)
                    )
                    # Triangle 2: p10, p11, p01
                    area += 0.5 * np.linalg.norm(
                        np.cross(p11 - p10, p01 - p10)
                    )
            areas.append(float(area))
        return areas

    # ------------------------------------------------------------------
    # Retry support
    # ------------------------------------------------------------------

    def with_error_context(self, error: Dict[str, Any]) -> "LatentProposal":
        """Create a retry proposal with error context.

        Preserves stage_latents so the diffusion model can re-denoise
        from an intermediate stage, but clears decoded geometry.
        """
        new = copy.copy(self)
        new.proposal_id = uuid.uuid4().hex[:16]
        new.attempt = self.next_attempt()
        new.error_context = error
        new.timestamp = datetime.now(timezone.utc).isoformat()
        new.face_grids = []
        new.edge_points = []
        new.face_bboxes = None
        new.edge_bboxes = None
        new.vertex_positions = None
        new.face_edge_adjacency = None
        new.confidence = 0.0
        # Keep stage_latents for partial re-generation
        return new

    def summary(self) -> Dict[str, Any]:
        """Extended summary with latent-specific fields."""
        base = super().summary()
        base.update({
            "num_faces": self.num_faces,
            "num_edges": self.num_edges,
            "num_vertices": self.num_vertices,
            "face_grid_resolution": self.face_grid_resolution,
            "total_points": self.total_points,
            "has_stage_latents": self.stage_latents is not None,
        })
        return base
