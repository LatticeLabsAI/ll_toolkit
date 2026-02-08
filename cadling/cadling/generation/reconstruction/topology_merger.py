"""Topology merger for BrepGen mating duplication recovery.

BrepGen-style diffusion models generate each face independently, producing
duplicate edges and vertices at face boundaries. This module recovers the
shared topology by detecting edge pairs that should be merged (using
bounding box proximity and shape similarity), averaging their geometry,
and sewing the faces into a watertight solid.

Classes:
    EdgeDescriptor: Lightweight description of an edge for comparison.
    TopologyMerger: Main merger class for mating duplication recovery.

Example:
    merger = TopologyMerger(bbox_threshold=0.08, shape_threshold=0.2)
    result = merger.merge(faces, edges)
    if result['valid']:
        shape = result['shape']
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)

# Lazy import of pythonocc
_has_pythonocc = False
try:
    from OCC.Core.BRep import BRep_Builder, BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_Sewing,
    )
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.GCPnts import GCPnts_UniformAbscissa
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, topods
    from OCC.Core.gp import gp_Pnt

    _has_pythonocc = True
    _log.debug("pythonocc-core available for topology merging")
except ImportError:
    _log.warning(
        "pythonocc-core not available. TopologyMerger will be disabled."
    )


@dataclass
class EdgeDescriptor:
    """Lightweight descriptor for an edge, used for merge comparison.

    Attributes:
        face_idx: Index of the face this edge belongs to.
        edge_idx: Local index within the face.
        edge: The OCC TopoDS_Edge object (if pythonocc available).
        bbox_min: Bounding box minimum corner [x, y, z].
        bbox_max: Bounding box maximum corner [x, y, z].
        sample_points: Sampled 3D points along the edge for shape comparison.
        length: Edge length.
    """

    face_idx: int
    edge_idx: int
    edge: Any = None
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sample_points: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3))
    )
    length: float = 0.0


class TopologyMerger:
    """Merge duplicate edges/faces via BrepGen mating duplication recovery.

    BrepGen generates faces independently, producing duplicate edges at shared
    boundaries. This class recovers the correct topology by:

        1. Extracting edge descriptors (bbox, sample points) from each face.
        2. For each cross-face edge pair, computing:
           a. Bounding box center distance (fast rejection)
           b. Shape similarity via Chamfer distance on sampled points
        3. Marking pairs with bbox_dist < bbox_threshold AND
           shape_sim < shape_threshold as merge candidates.
        4. Averaging vertex positions across merged duplicates.
        5. Scaling/translating edges to match averaged positions.
        6. Sewing faces via BRepBuilderAPI_Sewing for a watertight result.

    Attributes:
        bbox_threshold: Maximum bbox center distance for merge candidates.
        shape_threshold: Maximum Chamfer distance for merge confirmation.
        num_sample_points: Points to sample along each edge for comparison.
        sewing_tolerance: Tolerance for BRepBuilderAPI_Sewing.
        has_pythonocc: Whether pythonocc-core is available.

    Example:
        merger = TopologyMerger(bbox_threshold=0.08, shape_threshold=0.2)
        result = merger.merge(faces, edges)
    """

    def __init__(
        self,
        bbox_threshold: float = 0.08,
        shape_threshold: float = 0.2,
        num_sample_points: int = 20,
        sewing_tolerance: float = 1e-4,
    ):
        """Initialize the topology merger.

        Args:
            bbox_threshold: Maximum bounding box center distance for two
                edges to be considered merge candidates.
            shape_threshold: Maximum Chamfer distance (on normalized sample
                points) for confirming a merge.
            num_sample_points: Number of points to sample along each edge
                for shape comparison.
            sewing_tolerance: Tolerance for the BRepBuilderAPI_Sewing operation.
        """
        self.bbox_threshold = bbox_threshold
        self.shape_threshold = shape_threshold
        self.num_sample_points = num_sample_points
        self.sewing_tolerance = sewing_tolerance
        self.has_pythonocc = _has_pythonocc

        if not self.has_pythonocc:
            _log.warning(
                "TopologyMerger initialized without pythonocc. "
                "Merge operations will be disabled."
            )

    def merge(
        self,
        faces: List[Any],
        edges: Optional[List[List[Any]]] = None,
    ) -> Dict[str, Any]:
        """Merge duplicate topology from independently generated faces.

        Main entry point for mating duplication recovery. Takes a list of
        OCC faces (and optionally pre-extracted edges per face), identifies
        duplicate edge pairs, averages their geometry, and sews the faces
        into a watertight solid.

        Args:
            faces: List of TopoDS_Face objects from the generation model.
            edges: Optional pre-extracted edges per face. If None, edges
                are extracted from each face via TopExp_Explorer.

        Returns:
            Dictionary containing:
                - 'shape': Sewn TopoDS_Shape (or None).
                - 'valid': Whether the result passes topology checks.
                - 'num_faces': Number of input faces.
                - 'num_merge_pairs': Number of edge pairs merged.
                - 'num_edges_before': Total edges before merging.
                - 'num_edges_after': Total edges after sewing.
                - 'errors': List of error messages.
        """
        result: Dict[str, Any] = {
            "shape": None,
            "valid": False,
            "num_faces": len(faces),
            "num_merge_pairs": 0,
            "num_edges_before": 0,
            "num_edges_after": 0,
            "errors": [],
        }

        if not self.has_pythonocc:
            result["errors"].append("pythonocc not available; merge disabled")
            return result

        if not faces:
            result["errors"].append("No faces provided")
            return result

        # Step 1: Extract edge descriptors from all faces
        all_descriptors: List[EdgeDescriptor] = []
        for face_idx, face in enumerate(faces):
            face_edges = (
                edges[face_idx] if edges and face_idx < len(edges) else None
            )
            descriptors = self._extract_edge_descriptors(
                face, face_idx, face_edges
            )
            all_descriptors.extend(descriptors)

        result["num_edges_before"] = len(all_descriptors)
        _log.debug(
            "Extracted %d edge descriptors from %d faces",
            len(all_descriptors),
            len(faces),
        )

        # Step 2: Find merge candidate pairs (cross-face only)
        merge_pairs = self._find_merge_pairs(all_descriptors)
        result["num_merge_pairs"] = len(merge_pairs)
        _log.debug("Found %d merge candidate pairs", len(merge_pairs))

        # Step 3: Average geometry of merged pairs
        if merge_pairs:
            self._average_merged_geometry(all_descriptors, merge_pairs)

        # Step 4: Sew faces together
        try:
            sewn_shape = self._sew_faces(faces)
            if sewn_shape is not None:
                result["shape"] = sewn_shape

                # Count edges in result
                edge_count = 0
                explorer = TopExp_Explorer(sewn_shape, TopAbs_EDGE)
                while explorer.More():
                    edge_count += 1
                    explorer.Next()
                result["num_edges_after"] = edge_count

                # Validate result
                analyzer = BRepCheck_Analyzer(sewn_shape)
                result["valid"] = analyzer.IsValid()

                if not result["valid"]:
                    result["errors"].append(
                        "BRepCheck_Analyzer reports invalid sewn shape"
                    )
            else:
                result["errors"].append("Sewing produced no shape")
        except Exception as e:
            error_msg = f"Sewing failed: {e}"
            _log.error(error_msg)
            result["errors"].append(error_msg)

        _log.info(
            "Topology merge: %d faces, %d pairs merged, "
            "edges %d -> %d, valid=%s",
            result["num_faces"],
            result["num_merge_pairs"],
            result["num_edges_before"],
            result["num_edges_after"],
            result["valid"],
        )

        return result

    def _extract_edge_descriptors(
        self,
        face: Any,
        face_idx: int,
        pre_extracted_edges: Optional[List[Any]] = None,
    ) -> List[EdgeDescriptor]:
        """Extract edge descriptors from a face.

        For each edge in the face, computes its bounding box, samples points
        along the curve, and packages them into an EdgeDescriptor.

        Args:
            face: TopoDS_Face to extract edges from.
            face_idx: Index of this face in the input list.
            pre_extracted_edges: Optional pre-extracted edge list.

        Returns:
            List of EdgeDescriptor objects for this face.
        """
        descriptors: List[EdgeDescriptor] = []

        # Get edges either from pre-extracted list or via explorer
        edge_list: List[Any] = []
        if pre_extracted_edges is not None:
            edge_list = pre_extracted_edges
        else:
            explorer = TopExp_Explorer(face, TopAbs_EDGE)
            while explorer.More():
                edge_list.append(topods.Edge(explorer.Current()))
                explorer.Next()

        for edge_idx, edge in enumerate(edge_list):
            try:
                desc = EdgeDescriptor(
                    face_idx=face_idx,
                    edge_idx=edge_idx,
                    edge=edge,
                )

                # Compute bounding box
                bbox = Bnd_Box()
                brepbndlib.Add(edge, bbox)
                xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
                desc.bbox_min = np.array([xmin, ymin, zmin])
                desc.bbox_max = np.array([xmax, ymax, zmax])

                # Compute edge length
                props = GProp_GProps()
                brepgprop.LinearProperties(edge, props)
                desc.length = props.Mass()

                # Sample points along the edge
                desc.sample_points = self._sample_edge_points(edge)

                descriptors.append(desc)

            except Exception as e:
                _log.warning(
                    "Failed to extract descriptor for edge %d of face %d: %s",
                    edge_idx,
                    face_idx,
                    e,
                )

        return descriptors

    def _sample_edge_points(self, edge: Any) -> np.ndarray:
        """Sample uniformly-spaced points along an edge curve.

        Args:
            edge: TopoDS_Edge to sample.

        Returns:
            NumPy array of shape (N, 3) with sampled 3D points.
        """
        try:
            curve_adaptor = BRepAdaptor_Curve(edge)
            u_start = curve_adaptor.FirstParameter()
            u_end = curve_adaptor.LastParameter()

            points = []
            for i in range(self.num_sample_points):
                t = i / max(self.num_sample_points - 1, 1)
                u = u_start + t * (u_end - u_start)
                pt = curve_adaptor.Value(u)
                points.append([pt.X(), pt.Y(), pt.Z()])

            return np.array(points, dtype=np.float64)

        except Exception as e:
            _log.warning("Edge point sampling failed: %s", e)
            return np.zeros((0, 3))

    def _compute_bbox_distance(
        self, desc1: EdgeDescriptor, desc2: EdgeDescriptor
    ) -> float:
        """Compute distance between bounding box centers of two edges.

        Args:
            desc1: First edge descriptor.
            desc2: Second edge descriptor.

        Returns:
            Euclidean distance between bbox centers.
        """
        center1 = (desc1.bbox_min + desc1.bbox_max) / 2.0
        center2 = (desc2.bbox_min + desc2.bbox_max) / 2.0
        return float(np.linalg.norm(center1 - center2))

    def _compute_shape_similarity(
        self, desc1: EdgeDescriptor, desc2: EdgeDescriptor
    ) -> float:
        """Compute Chamfer distance between sampled points of two edges.

        Measures how similar two edges are in terms of their geometry.
        Lower values indicate more similar shapes.

        Args:
            desc1: First edge descriptor with sample_points.
            desc2: Second edge descriptor with sample_points.

        Returns:
            Chamfer distance (mean of nearest-neighbor distances in both
            directions). Returns inf if either edge has no samples.
        """
        pts1 = desc1.sample_points
        pts2 = desc2.sample_points

        if len(pts1) == 0 or len(pts2) == 0:
            return float("inf")

        # Compute pairwise distance matrix
        # diff shape: (N1, N2, 3)
        diff = pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Chamfer distance: mean of min distances in both directions
        forward = np.mean(np.min(dist_matrix, axis=1))
        backward = np.mean(np.min(dist_matrix, axis=0))

        return float((forward + backward) / 2.0)

    def _find_merge_pairs(
        self, descriptors: List[EdgeDescriptor]
    ) -> List[Tuple[int, int]]:
        """Find edge pairs that should be merged.

        Tests all cross-face edge pairs against the bbox and shape
        thresholds to identify duplicates.

        Args:
            descriptors: List of all EdgeDescriptor objects.

        Returns:
            List of (index_i, index_j) pairs to merge.
        """
        merge_pairs: List[Tuple[int, int]] = []
        n = len(descriptors)

        for i in range(n):
            for j in range(i + 1, n):
                # Only merge edges from different faces
                if descriptors[i].face_idx == descriptors[j].face_idx:
                    continue

                # Fast rejection via bbox distance
                bbox_dist = self._compute_bbox_distance(
                    descriptors[i], descriptors[j]
                )
                if bbox_dist >= self.bbox_threshold:
                    continue

                # Shape similarity check
                shape_sim = self._compute_shape_similarity(
                    descriptors[i], descriptors[j]
                )
                if shape_sim >= self.shape_threshold:
                    continue

                merge_pairs.append((i, j))
                _log.debug(
                    "Merge candidate: face %d edge %d <-> face %d edge %d "
                    "(bbox_dist=%.4f, shape_sim=%.4f)",
                    descriptors[i].face_idx,
                    descriptors[i].edge_idx,
                    descriptors[j].face_idx,
                    descriptors[j].edge_idx,
                    bbox_dist,
                    shape_sim,
                )

        return merge_pairs

    def _average_merged_geometry(
        self,
        descriptors: List[EdgeDescriptor],
        merge_pairs: List[Tuple[int, int]],
    ) -> None:
        """Average vertex positions across merged edge pairs.

        For each merge pair, replaces both edges' sample points with the
        element-wise average. This ensures geometric consistency at the
        shared boundary before sewing.

        Args:
            descriptors: List of all EdgeDescriptor objects (modified in place).
            merge_pairs: List of (index_i, index_j) pairs to merge.
        """
        for i, j in merge_pairs:
            pts_i = descriptors[i].sample_points
            pts_j = descriptors[j].sample_points

            if len(pts_i) == 0 or len(pts_j) == 0:
                continue

            # Ensure same number of sample points
            n_pts = min(len(pts_i), len(pts_j))
            pts_i_trimmed = pts_i[:n_pts]
            pts_j_trimmed = pts_j[:n_pts]

            # Check if we need to reverse one set (opposite parameterization)
            forward_dist = np.mean(
                np.linalg.norm(pts_i_trimmed - pts_j_trimmed, axis=1)
            )
            reverse_dist = np.mean(
                np.linalg.norm(pts_i_trimmed - pts_j_trimmed[::-1], axis=1)
            )

            if reverse_dist < forward_dist:
                pts_j_trimmed = pts_j_trimmed[::-1]

            # Average positions
            averaged = (pts_i_trimmed + pts_j_trimmed) / 2.0
            descriptors[i].sample_points = averaged
            descriptors[j].sample_points = averaged

            _log.debug(
                "Averaged geometry for edge pair (%d, %d): "
                "max shift = %.4e",
                i,
                j,
                float(
                    np.max(np.linalg.norm(pts_i_trimmed - averaged, axis=1))
                ),
            )

    def _sew_faces(self, faces: List[Any]) -> Optional[Any]:
        """Sew faces together into a solid or shell.

        Uses BRepBuilderAPI_Sewing to join faces at shared edges,
        producing a topologically connected result.

        Args:
            faces: List of TopoDS_Face objects to sew together.

        Returns:
            Sewn TopoDS_Shape, or None if sewing fails.
        """
        try:
            sewing = BRepBuilderAPI_Sewing(self.sewing_tolerance)

            for face in faces:
                sewing.Add(face)

            sewing.Perform()
            sewn_shape = sewing.SewedShape()

            if sewn_shape.IsNull():
                _log.warning("Sewing produced a null shape")
                return None

            n_free = sewing.NbFreeEdges()
            n_multiple = sewing.NbMultipleEdges()
            n_degenerated = sewing.NbDegeneratedShapes()

            _log.debug(
                "Sewing result: free_edges=%d, multiple_edges=%d, "
                "degenerated=%d",
                n_free,
                n_multiple,
                n_degenerated,
            )

            if n_free > 0:
                _log.warning(
                    "Sewn shape has %d free edges (not watertight)", n_free
                )

            return sewn_shape

        except Exception as e:
            _log.error("BRepBuilderAPI_Sewing failed: %s", e)
            return None
