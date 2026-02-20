"""Coedge extraction from OCC B-Rep topology.

This module provides extraction of ordered coedge sequences from B-Rep models
using OCC's BRepTools_WireExplorer for proper edge ordering within face loops.

Each topological edge in a B-Rep solid has two oriented coedges (one per adjacent
face). This is the fundamental structure required by BRepNet and similar GNN
architectures.

Example:
    from cadling.lib.topology.coedge_extractor import CoedgeExtractor

    extractor = CoedgeExtractor()
    coedges = extractor.extract_coedges(shape)

    for coedge in coedges:
        print(f"Face: {coedge.face_id}, Edge: {coedge.edge_id}")
        print(f"Mate: {coedge.mate_id}, Orientation: {coedge.orientation}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge

from cadling.lib.topology.face_identity import ShapeIdentityRegistry

_log = logging.getLogger(__name__)

# Availability flag
HAS_OCC = False

try:
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Wire
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE, TopAbs_FORWARD, TopAbs_REVERSED
    from OCC.Core.BRepTools import BRepTools_WireExplorer
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopoDS import topods

    HAS_OCC = True
except ImportError:
    _log.debug("pythonocc-core not available. CoedgeExtractor will return empty results.")


@dataclass
class Coedge:
    """Represents a coedge (half-edge) in B-Rep topology.

    Each topological edge appears twice in a B-Rep solid - once for each
    adjacent face. The coedge captures this oriented edge-in-face relationship.

    Attributes:
        id: Unique coedge identifier
        face_id: ID of the face this coedge belongs to
        edge_id: ID of the underlying edge
        face_index: Sequential index of the face
        edge_index: Sequential index of the edge
        orientation: Edge orientation in this face (FORWARD or REVERSED)
        position_in_loop: Position in the face's wire loop (0-based)
        loop_size: Total edges in this wire loop
        next_id: ID of next coedge in same face loop (cyclic)
        prev_id: ID of previous coedge in same face loop (cyclic)
        mate_id: ID of mate coedge (same edge, adjacent face)
        features: Optional feature vector for this coedge
    """

    id: int
    face_id: str
    edge_id: str
    face_index: int = -1
    edge_index: int = -1
    orientation: str = "FORWARD"
    position_in_loop: int = 0
    loop_size: int = 0
    next_id: Optional[int] = None
    prev_id: Optional[int] = None
    mate_id: Optional[int] = None
    features: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "face_id": self.face_id,
            "edge_id": self.edge_id,
            "face_index": self.face_index,
            "edge_index": self.edge_index,
            "orientation": self.orientation,
            "position_in_loop": self.position_in_loop,
            "loop_size": self.loop_size,
            "next_id": self.next_id,
            "prev_id": self.prev_id,
            "mate_id": self.mate_id,
        }


class CoedgeExtractor:
    """Extract ordered coedge sequences from OCC B-Rep.

    Uses BRepTools_WireExplorer for proper edge ordering within face loops,
    and MapShapesAndAncestors for finding mate coedges.

    Attributes:
        _registry: ShapeIdentityRegistry for stable IDs
    """

    def __init__(self, registry: Optional[ShapeIdentityRegistry] = None):
        """Initialize coedge extractor.

        Args:
            registry: Optional ShapeIdentityRegistry. If None, a new one is created.
        """
        self._registry = registry or ShapeIdentityRegistry()

    @property
    def registry(self) -> ShapeIdentityRegistry:
        """Get the shape identity registry."""
        return self._registry

    def extract_coedges(self, shape: "TopoDS_Shape") -> List[Coedge]:
        """Extract all coedges with proper ordering and mate pointers.

        Args:
            shape: TopoDS_Shape to extract coedges from

        Returns:
            List of Coedge objects with next/prev/mate pointers set
        """
        if not HAS_OCC:
            _log.warning("OCC not available, returning empty coedge list")
            return []

        try:
            # Register all shapes first
            self._registry.register_all(shape)

            # Build edge -> faces mapping for mate finding
            edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
            topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces)

            # Extract coedges face by face
            coedges: List[Coedge] = []
            coedge_key_to_idx: Dict[Tuple[str, str], int] = {}

            face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while face_explorer.More():
                face = topods.Face(face_explorer.Current())
                face_coedges = self._extract_face_coedges(face, len(coedges))

                # Index coedges by (face_id, edge_id)
                for coedge in face_coedges:
                    coedge_key_to_idx[(coedge.face_id, coedge.edge_id)] = coedge.id
                    coedges.append(coedge)

                face_explorer.Next()

            # Build next/prev pointers (within same face loop)
            self._build_loop_pointers(coedges)

            # Build mate pointers (same edge, different face)
            self._build_mate_pointers(coedges, edge_to_faces, coedge_key_to_idx)

            _log.info(f"Extracted {len(coedges)} coedges from {self._registry.num_faces} faces")
            return coedges

        except Exception as e:
            _log.error(f"Failed to extract coedges: {e}")
            return []

    def _extract_face_coedges(
        self,
        face: "TopoDS_Face",
        start_id: int,
    ) -> List[Coedge]:
        """Extract ordered coedges from a single face using wire explorer.

        Args:
            face: TopoDS_Face to extract coedges from
            start_id: Starting ID for coedges

        Returns:
            List of Coedge objects for this face
        """
        if not HAS_OCC:
            return []

        coedges: List[Coedge] = []
        face_id = self._registry.get_id(face)
        face_index = self._registry.get_face_index(face_id)

        try:
            # Iterate over wires (outer + inner loops)
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)

            while wire_explorer.More():
                wire = topods.Wire(wire_explorer.Current())

                # Get edges in order using BRepTools_WireExplorer
                loop_edges: List[Tuple[str, str, int]] = []  # (edge_id, orientation, edge_index)

                edge_explorer = BRepTools_WireExplorer(wire, face)
                while edge_explorer.More():
                    edge = edge_explorer.Current()
                    edge_id = self._registry.get_id(edge)
                    edge_index = self._registry.get_edge_index(edge_id)

                    # Get orientation
                    orientation = edge.Orientation()
                    orient_str = "FORWARD" if orientation == TopAbs_FORWARD else "REVERSED"

                    loop_edges.append((edge_id, orient_str, edge_index))
                    edge_explorer.Next()

                # Create coedges for this loop
                loop_size = len(loop_edges)
                for pos, (edge_id, orientation, edge_index) in enumerate(loop_edges):
                    coedge = Coedge(
                        id=start_id + len(coedges),
                        face_id=face_id,
                        edge_id=edge_id,
                        face_index=face_index,
                        edge_index=edge_index,
                        orientation=orientation,
                        position_in_loop=pos,
                        loop_size=loop_size,
                    )
                    coedges.append(coedge)

                wire_explorer.Next()

        except Exception as e:
            _log.debug(f"Failed to extract coedges from face {face_id}: {e}")

        return coedges

    def _build_loop_pointers(self, coedges: List[Coedge]) -> None:
        """Build next/prev pointers for coedges within the same face loop.

        Args:
            coedges: List of coedges to update in-place
        """
        # Group coedges by face
        face_coedges: Dict[str, List[Coedge]] = {}
        for coedge in coedges:
            if coedge.face_id not in face_coedges:
                face_coedges[coedge.face_id] = []
            face_coedges[coedge.face_id].append(coedge)

        # Set next/prev within each face
        for face_id, face_ces in face_coedges.items():
            # Sort by position in loop
            sorted_ces = sorted(face_ces, key=lambda c: c.position_in_loop)

            for i, coedge in enumerate(sorted_ces):
                loop_size = coedge.loop_size
                if loop_size == 0:
                    loop_size = len(sorted_ces)

                # Cyclic next/prev
                next_pos = (coedge.position_in_loop + 1) % loop_size
                prev_pos = (coedge.position_in_loop - 1) % loop_size

                # Find coedges at those positions
                for other in sorted_ces:
                    if other.position_in_loop == next_pos:
                        coedge.next_id = other.id
                    if other.position_in_loop == prev_pos:
                        coedge.prev_id = other.id

    def _build_mate_pointers(
        self,
        coedges: List[Coedge],
        edge_to_faces: "TopTools_IndexedDataMapOfShapeListOfShape",
        coedge_key_to_idx: Dict[Tuple[str, str], int],
    ) -> None:
        """Build mate pointers connecting coedges across adjacent faces.

        For each edge shared by two faces, the coedges on each face are mates.

        Args:
            coedges: List of coedges to update in-place
            edge_to_faces: OCC map of edges to adjacent faces
            coedge_key_to_idx: Map from (face_id, edge_id) to coedge index
        """
        if not HAS_OCC:
            return

        try:
            # Build edge_id -> list of coedge indices
            edge_to_coedges: Dict[str, List[int]] = {}
            for coedge in coedges:
                if coedge.edge_id not in edge_to_coedges:
                    edge_to_coedges[coedge.edge_id] = []
                edge_to_coedges[coedge.edge_id].append(coedge.id)

            # For each edge with exactly 2 coedges, set them as mates
            for edge_id, coedge_ids in edge_to_coedges.items():
                if len(coedge_ids) == 2:
                    # These are mates
                    c1_id, c2_id = coedge_ids
                    coedges[c1_id].mate_id = c2_id
                    coedges[c2_id].mate_id = c1_id
                elif len(coedge_ids) == 1:
                    # Boundary edge (only one face) - mate is self
                    coedges[coedge_ids[0]].mate_id = coedge_ids[0]
                else:
                    # Non-manifold edge (>2 faces) - set first two as mates, log warning
                    _log.debug(f"Non-manifold edge {edge_id} with {len(coedge_ids)} faces")
                    if len(coedge_ids) >= 2:
                        coedges[coedge_ids[0]].mate_id = coedge_ids[1]
                        coedges[coedge_ids[1]].mate_id = coedge_ids[0]

        except Exception as e:
            _log.warning(f"Failed to build mate pointers: {e}")

    def extract_coedge_data(
        self,
        shape: "TopoDS_Shape",
        feature_dim: int = 12,
    ) -> Dict[str, Any]:
        """Extract coedge data in tensor-ready format for BRepNet.

        Args:
            shape: TopoDS_Shape to extract from
            feature_dim: Dimension of coedge feature vectors

        Returns:
            Dictionary with:
                - features: np.ndarray [num_coedges, feature_dim]
                - next_indices: np.ndarray [num_coedges] (int64)
                - prev_indices: np.ndarray [num_coedges] (int64)
                - mate_indices: np.ndarray [num_coedges] (int64)
                - face_indices: np.ndarray [num_coedges] (int64)
                - coedges: List[Coedge] original coedge objects
        """
        coedges = self.extract_coedges(shape)
        num_coedges = len(coedges)

        if num_coedges == 0:
            return {
                "features": np.zeros((0, feature_dim), dtype=np.float32),
                "next_indices": np.zeros(0, dtype=np.int64),
                "prev_indices": np.zeros(0, dtype=np.int64),
                "mate_indices": np.zeros(0, dtype=np.int64),
                "face_indices": np.zeros(0, dtype=np.int64),
                "coedges": [],
            }

        # Build index arrays
        next_indices = np.zeros(num_coedges, dtype=np.int64)
        prev_indices = np.zeros(num_coedges, dtype=np.int64)
        mate_indices = np.zeros(num_coedges, dtype=np.int64)
        face_indices = np.zeros(num_coedges, dtype=np.int64)

        for coedge in coedges:
            i = coedge.id
            next_indices[i] = coedge.next_id if coedge.next_id is not None else i
            prev_indices[i] = coedge.prev_id if coedge.prev_id is not None else i
            mate_indices[i] = coedge.mate_id if coedge.mate_id is not None else i
            face_indices[i] = coedge.face_index if coedge.face_index >= 0 else 0

        # Extract features for each coedge
        features = self._extract_coedge_features(coedges, shape, feature_dim)

        return {
            "features": features,
            "next_indices": next_indices,
            "prev_indices": prev_indices,
            "mate_indices": mate_indices,
            "face_indices": face_indices,
            "coedges": coedges,
        }

    def _extract_coedge_features(
        self,
        coedges: List[Coedge],
        shape: "TopoDS_Shape",
        feature_dim: int,
    ) -> np.ndarray:
        """Extract feature vectors for coedges.

        Coedge features (12 dims):
          - edge curve type one-hot (6): LINE, CIRCLE, ELLIPSE, B_SPLINE, PARABOLA, OTHER
          - edge length (1)
          - tangent vector at midpoint (3)
          - curvature at midpoint (1)
          - convexity flag (1): 1=convex, 0=concave, 0.5=tangent

        Args:
            coedges: List of Coedge objects
            shape: Original TopoDS_Shape for accessing geometry
            feature_dim: Feature dimension

        Returns:
            Feature matrix [num_coedges, feature_dim]
        """
        from cadling.lib.occ_wrapper import OCCEdge, HAS_OCC as OCC_AVAILABLE

        num_coedges = len(coedges)
        features = np.zeros((num_coedges, feature_dim), dtype=np.float32)

        if not OCC_AVAILABLE:
            return features

        # Curve type mapping
        curve_types = ["LINE", "CIRCLE", "ELLIPSE", "BSPLINE", "PARABOLA", "OTHER"]

        for coedge in coedges:
            i = coedge.id

            # Get the edge
            edge = self._registry.get_edge(coedge.edge_id)
            if edge is None:
                continue

            # Wrap in OCCEdge for feature extraction
            occ_edge = OCCEdge(edge)
            edge_features = occ_edge.extract_features()

            # Curve type one-hot (dims 0-5)
            curve_type = edge_features.get("curve_type", "OTHER").upper()
            for ct_idx, ct_name in enumerate(curve_types):
                if ct_name in curve_type:
                    features[i, ct_idx] = 1.0
                    break
            else:
                features[i, 5] = 1.0  # OTHER

            # Edge length (dim 6)
            features[i, 6] = edge_features.get("length", 1.0)

            # Tangent vector (dims 7-9)
            tangent = edge_features.get("tangent", [0.0, 0.0, 1.0])
            features[i, 7:10] = tangent[:3]

            # Curvature (dim 10)
            features[i, 10] = edge_features.get("curvature", 0.0)

            # Convexity (dim 11) - computed from mate normals
            features[i, 11] = self._compute_convexity(coedge, coedges, shape)

        return features

    def _compute_convexity(
        self,
        coedge: Coedge,
        all_coedges: List[Coedge],
        shape: "TopoDS_Shape",
    ) -> float:
        """Compute convexity of a coedge from adjacent face normals.

        Args:
            coedge: The coedge to compute convexity for
            all_coedges: All coedges for mate lookup
            shape: Original shape for geometry access

        Returns:
            Convexity value: 1.0=convex, 0.0=concave, 0.5=tangent/unknown
        """
        from cadling.lib.occ_wrapper import OCCFace, HAS_OCC as OCC_AVAILABLE

        if not OCC_AVAILABLE:
            return 0.5

        # Get mate coedge
        if coedge.mate_id is None or coedge.mate_id == coedge.id:
            return 0.5  # Boundary edge

        mate = all_coedges[coedge.mate_id]

        # Get face normals
        face1 = self._registry.get_face(coedge.face_id)
        face2 = self._registry.get_face(mate.face_id)

        if face1 is None or face2 is None:
            return 0.5

        try:
            occ_face1 = OCCFace(face1)
            occ_face2 = OCCFace(face2)

            n1 = np.array(occ_face1.normal_at())
            n2 = np.array(occ_face2.normal_at())

            # Compute cross product and dot with something to determine convexity
            # Simplified: use normal dot product
            dot = np.dot(n1, n2)

            if abs(dot) > 0.99:
                return 0.5  # Nearly coplanar
            elif dot > 0:
                return 1.0  # Convex (normals pointing similar directions)
            else:
                return 0.0  # Concave (normals pointing away)

        except Exception:
            return 0.5

    def to_brep_net_format(
        self,
        shape: "TopoDS_Shape",
    ) -> Dict[str, np.ndarray]:
        """Export coedge data in BRepNet-compatible format.

        Args:
            shape: TopoDS_Shape to extract from

        Returns:
            Dictionary with tensors ready for BRepNetEncoder
        """
        data = self.extract_coedge_data(shape, feature_dim=12)

        return {
            "coedge_features": data["features"],
            "coedge_next": data["next_indices"],
            "coedge_prev": data["prev_indices"],
            "coedge_mate": data["mate_indices"],
            "coedge_to_face": data["face_indices"],
            "num_coedges": len(data["coedges"]),
            "num_faces": self._registry.num_faces,
        }
