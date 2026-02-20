"""Graph-based CAD reconstruction from decoded features.

This module provides direct reconstruction of OCC primitives from decoded
graph features, enabling generative CAD workflows without LLM intermediaries.

The reconstructor converts:
- Node features → OCC surface primitives (planes, cylinders, spheres, etc.)
- Edge features → OCC curves (lines, circles, splines)
- Adjacency → Topology assembly instructions

Example:
    from cadling.generation.reconstruction.graph_reconstructor import GraphReconstructor

    reconstructor = GraphReconstructor()

    # Reconstruct from decoded graph
    primitives = reconstructor.reconstruct(
        node_features=node_features,  # [N, 24]
        edge_index=edge_index,        # [2, M]
    )

    # Assemble into solid
    from cadling.generation.reconstruction.topology_assembler import TopologyAssembler
    assembler = TopologyAssembler()
    shape = assembler.assemble(primitives, adjacency)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

_log = logging.getLogger(__name__)

# Availability flags
HAS_OCC = False

try:
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeVertex,
    )
    from OCC.Core.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakeSphere,
        BRepPrimAPI_MakeCone,
        BRepPrimAPI_MakeTorus,
    )
    from OCC.Core.gp import (
        gp_Pnt, gp_Vec, gp_Dir, gp_Ax1, gp_Ax2, gp_Ax3,
        gp_Pln, gp_Circ, gp_Lin,
    )
    from OCC.Core.Geom import (
        Geom_Plane,
        Geom_CylindricalSurface,
        Geom_ConicalSurface,
        Geom_SphericalSurface,
        Geom_ToroidalSurface,
    )
    from OCC.Core.GC import GC_MakeLine, GC_MakeCircle

    HAS_OCC = True
except ImportError:
    _log.debug("pythonocc-core not available. GraphReconstructor will have limited functionality.")


# Surface type indices (matching occ_wrapper.py)
SURFACE_TYPE_MAP = {
    0: "PLANE",
    1: "CYLINDER",
    2: "CONE",
    3: "SPHERE",
    4: "TORUS",
    5: "BEZIER",
    6: "BSPLINE",
    7: "REVOLUTION",
    8: "EXTRUSION",
    9: "OTHER",
}

CURVE_TYPE_MAP = {
    0: "LINE",
    1: "CIRCLE",
    2: "ELLIPSE",
    3: "HYPERBOLA",
    4: "PARABOLA",
    5: "BEZIER",
    6: "BSPLINE",
    7: "OTHER",
}


@dataclass
class ReconstructedPrimitive:
    """Represents a reconstructed primitive (face, edge, or vertex).

    Attributes:
        primitive_type: Type of primitive (FACE, EDGE, VERTEX)
        surface_type: Surface type for faces (PLANE, CYLINDER, etc.)
        curve_type: Curve type for edges (LINE, CIRCLE, etc.)
        occ_shape: Reconstructed OCC shape (if successful)
        parameters: Parameters used for reconstruction
        confidence: Confidence score for reconstruction
        error: Error message if reconstruction failed
    """

    primitive_type: str  # FACE, EDGE, VERTEX
    surface_type: Optional[str] = None
    curve_type: Optional[str] = None
    occ_shape: Optional[Any] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether reconstruction succeeded."""
        return self.occ_shape is not None and self.error is None


@dataclass
class ReconstructionResult:
    """Result of graph reconstruction.

    Attributes:
        faces: List of reconstructed faces
        edges: List of reconstructed edges
        vertices: List of reconstructed vertices
        adjacency: Face adjacency dictionary
        errors: List of error messages
        success_rate: Proportion of successfully reconstructed primitives
    """

    faces: List[ReconstructedPrimitive] = field(default_factory=list)
    edges: List[ReconstructedPrimitive] = field(default_factory=list)
    vertices: List[ReconstructedPrimitive] = field(default_factory=list)
    adjacency: Dict[int, List[int]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate proportion of successful reconstructions."""
        total = len(self.faces) + len(self.edges) + len(self.vertices)
        if total == 0:
            return 0.0

        successful = sum(1 for f in self.faces if f.success)
        successful += sum(1 for e in self.edges if e.success)
        successful += sum(1 for v in self.vertices if v.success)

        return successful / total

    @property
    def num_successful_faces(self) -> int:
        """Number of successfully reconstructed faces."""
        return sum(1 for f in self.faces if f.success)

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary format."""
        return {
            "num_faces": len(self.faces),
            "num_edges": len(self.edges),
            "num_vertices": len(self.vertices),
            "success_rate": self.success_rate,
            "errors": self.errors,
            "face_types": [f.surface_type for f in self.faces],
            "edge_types": [e.curve_type for e in self.edges],
        }


class GraphReconstructor:
    """Reconstruct OCC primitives from decoded graph features.

    Converts node features (surface types, curvatures, centroids, dimensions)
    into OCC surface primitives. This enables direct CAD reconstruction from
    learned representations without LLM generation.

    Attributes:
        tolerance: Geometric tolerance for reconstruction
        default_scale: Default scale factor for dimensions
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        default_scale: float = 1.0,
    ):
        """Initialize graph reconstructor.

        Args:
            tolerance: Geometric tolerance
            default_scale: Default scale factor
        """
        self.tolerance = tolerance
        self.default_scale = default_scale

        if not HAS_OCC:
            _log.warning("OCC not available. GraphReconstructor will return empty results.")

    def reconstruct(
        self,
        node_features: np.ndarray,
        edge_index: Optional[np.ndarray] = None,
        edge_features: Optional[np.ndarray] = None,
    ) -> ReconstructionResult:
        """Reconstruct primitives from decoded graph features.

        Args:
            node_features: Face features [N, feature_dim] (usually 24 dims)
            edge_index: Adjacency in COO format [2, M]
            edge_features: Optional edge features [M, edge_dim]

        Returns:
            ReconstructionResult with reconstructed primitives
        """
        result = ReconstructionResult()

        if not HAS_OCC:
            result.errors.append("OCC not available for reconstruction")
            return result

        num_faces = node_features.shape[0]
        _log.info(f"Reconstructing {num_faces} faces from graph features")

        # Reconstruct faces
        for i in range(num_faces):
            face_primitive = self._reconstruct_face(i, node_features[i])
            result.faces.append(face_primitive)

        # Build adjacency from edge_index
        if edge_index is not None:
            result.adjacency = self._build_adjacency(edge_index)

        # Reconstruct edges if features provided
        if edge_features is not None and edge_index is not None:
            num_edges = edge_features.shape[0]
            for i in range(num_edges):
                edge_primitive = self._reconstruct_edge(i, edge_features[i], edge_index, i)
                result.edges.append(edge_primitive)

        _log.info(
            f"Reconstruction complete: {result.num_successful_faces}/{num_faces} faces, "
            f"success_rate={result.success_rate:.1%}"
        )

        return result

    def _reconstruct_face(
        self,
        index: int,
        features: np.ndarray,
    ) -> ReconstructedPrimitive:
        """Reconstruct a single face from its features.

        Args:
            index: Face index
            features: Feature vector (typically 24 dims)

        Returns:
            ReconstructedPrimitive for this face
        """
        # Decode surface type from one-hot (first 10 dims)
        surface_type_idx = int(np.argmax(features[:10]))
        surface_type = SURFACE_TYPE_MAP.get(surface_type_idx, "OTHER")

        # Extract geometric parameters
        feature_dim = len(features)
        params: Dict[str, Any] = {
            "surface_type": surface_type,
        }

        # Area (dim 10)
        if feature_dim > 10:
            params["area"] = float(features[10])

        # Curvature (dims 11-12)
        if feature_dim > 12:
            params["gaussian_curvature"] = float(features[11])
            params["mean_curvature"] = float(features[12])

        # Normal (dims 13-15)
        if feature_dim > 15:
            params["normal"] = features[13:16].tolist()

        # Centroid (dims 16-18)
        if feature_dim > 18:
            params["centroid"] = features[16:19].tolist()

        # Bbox dimensions (dims 19-21)
        if feature_dim > 21:
            params["bbox_dimensions"] = features[19:22].tolist()

        # Attempt reconstruction based on surface type
        try:
            occ_face = self._create_surface(surface_type, params)

            return ReconstructedPrimitive(
                primitive_type="FACE",
                surface_type=surface_type,
                occ_shape=occ_face,
                parameters=params,
                confidence=1.0 if occ_face is not None else 0.0,
            )

        except Exception as e:
            _log.debug(f"Failed to reconstruct face {index}: {e}")
            return ReconstructedPrimitive(
                primitive_type="FACE",
                surface_type=surface_type,
                parameters=params,
                error=str(e),
            )

    def _create_surface(
        self,
        surface_type: str,
        params: Dict[str, Any],
    ) -> Optional[Any]:
        """Create OCC surface from type and parameters.

        Args:
            surface_type: Surface type name
            params: Geometric parameters

        Returns:
            OCC TopoDS_Face or None
        """
        if not HAS_OCC:
            return None

        centroid = params.get("centroid", [0.0, 0.0, 0.0])
        normal = params.get("normal", [0.0, 0.0, 1.0])
        dimensions = params.get("bbox_dimensions", [1.0, 1.0, 1.0])
        mean_curv = params.get("mean_curvature", 0.0)
        area = params.get("area", 1.0)

        # Create origin point
        origin = gp_Pnt(centroid[0], centroid[1], centroid[2])

        # Create normal direction
        norm_mag = np.linalg.norm(normal)
        if norm_mag > 1e-10:
            normal_dir = gp_Dir(normal[0] / norm_mag, normal[1] / norm_mag, normal[2] / norm_mag)
        else:
            normal_dir = gp_Dir(0, 0, 1)

        try:
            if surface_type == "PLANE":
                return self._create_planar_face(origin, normal_dir, dimensions, area)

            elif surface_type == "CYLINDER":
                return self._create_cylindrical_face(origin, normal_dir, dimensions, mean_curv, area)

            elif surface_type == "SPHERE":
                return self._create_spherical_face(origin, dimensions, mean_curv, area)

            elif surface_type == "CONE":
                return self._create_conical_face(origin, normal_dir, dimensions, mean_curv, area)

            elif surface_type == "TORUS":
                return self._create_toroidal_face(origin, normal_dir, dimensions, mean_curv)

            else:
                # Default to planar for unsupported types
                _log.debug(f"Unsupported surface type {surface_type}, using planar approximation")
                return self._create_planar_face(origin, normal_dir, dimensions, area)

        except Exception as e:
            _log.debug(f"Failed to create {surface_type} surface: {e}")
            return None

    def _create_planar_face(
        self,
        origin: "gp_Pnt",
        normal: "gp_Dir",
        dimensions: List[float],
        area: float,
    ) -> Optional[Any]:
        """Create a planar face."""
        if not HAS_OCC:
            return None

        # Create plane
        plane = gp_Pln(origin, normal)

        # Estimate size from area or dimensions
        if area > 0:
            size = np.sqrt(area)
        else:
            size = max(dimensions[0], dimensions[1], 1.0)

        # Create bounded face
        face_builder = BRepBuilderAPI_MakeFace(
            plane,
            -size / 2, size / 2,  # U bounds
            -size / 2, size / 2,  # V bounds
        )

        if face_builder.IsDone():
            return face_builder.Face()

        return None

    def _create_cylindrical_face(
        self,
        origin: "gp_Pnt",
        axis: "gp_Dir",
        dimensions: List[float],
        mean_curvature: float,
        area: float,
    ) -> Optional[Any]:
        """Create a cylindrical face."""
        if not HAS_OCC:
            return None

        # Estimate radius from curvature or dimensions
        if abs(mean_curvature) > 1e-10:
            radius = abs(1.0 / mean_curvature)
        else:
            radius = min(dimensions[0], dimensions[1]) / 2.0
            if radius <= 0:
                radius = 1.0

        # Estimate height from area and radius
        if area > 0 and radius > 0:
            height = area / (2 * np.pi * radius)
        else:
            height = max(dimensions[2], 1.0)

        # Create cylinder
        ax2 = gp_Ax2(origin, axis)

        try:
            cylinder = BRepPrimAPI_MakeCylinder(ax2, radius, height)
            if cylinder.IsDone():
                return cylinder.Shape()
        except Exception as e:
            _log.debug(f"Cylinder creation failed: {e}")

        return None

    def _create_spherical_face(
        self,
        origin: "gp_Pnt",
        dimensions: List[float],
        mean_curvature: float,
        area: float,
    ) -> Optional[Any]:
        """Create a spherical face."""
        if not HAS_OCC:
            return None

        # Estimate radius from curvature or area
        if abs(mean_curvature) > 1e-10:
            radius = abs(1.0 / mean_curvature)
        elif area > 0:
            radius = np.sqrt(area / (4 * np.pi))
        else:
            radius = max(dimensions) / 2.0
            if radius <= 0:
                radius = 1.0

        try:
            sphere = BRepPrimAPI_MakeSphere(origin, radius)
            if sphere.IsDone():
                return sphere.Shape()
        except Exception as e:
            _log.debug(f"Sphere creation failed: {e}")

        return None

    def _create_conical_face(
        self,
        origin: "gp_Pnt",
        axis: "gp_Dir",
        dimensions: List[float],
        mean_curvature: float,
        area: float,
    ) -> Optional[Any]:
        """Create a conical face."""
        if not HAS_OCC:
            return None

        # Estimate parameters
        base_radius = max(dimensions[0], dimensions[1]) / 2.0
        if base_radius <= 0:
            base_radius = 1.0

        height = max(dimensions[2], base_radius)

        # Semi-angle from curvature (approximate)
        if abs(mean_curvature) > 1e-10:
            top_radius = base_radius * 0.5  # Approximate
        else:
            top_radius = base_radius * 0.1

        ax2 = gp_Ax2(origin, axis)

        try:
            cone = BRepPrimAPI_MakeCone(ax2, base_radius, top_radius, height)
            if cone.IsDone():
                return cone.Shape()
        except Exception as e:
            _log.debug(f"Cone creation failed: {e}")

        return None

    def _create_toroidal_face(
        self,
        origin: "gp_Pnt",
        axis: "gp_Dir",
        dimensions: List[float],
        mean_curvature: float,
    ) -> Optional[Any]:
        """Create a toroidal face."""
        if not HAS_OCC:
            return None

        # Estimate radii
        major_radius = max(dimensions[0], dimensions[1]) / 2.0
        if major_radius <= 0:
            major_radius = 2.0

        minor_radius = major_radius / 4.0  # Default ratio

        # Adjust from curvature if available
        if abs(mean_curvature) > 1e-10:
            minor_radius = abs(1.0 / mean_curvature)
            if minor_radius > major_radius:
                minor_radius = major_radius / 2.0

        ax2 = gp_Ax2(origin, axis)

        try:
            torus = BRepPrimAPI_MakeTorus(ax2, major_radius, minor_radius)
            if torus.IsDone():
                return torus.Shape()
        except Exception as e:
            _log.debug(f"Torus creation failed: {e}")

        return None

    def _reconstruct_edge(
        self,
        index: int,
        features: np.ndarray,
        edge_index: np.ndarray,
        edge_idx_in_list: int,
    ) -> ReconstructedPrimitive:
        """Reconstruct a single edge from its features.

        Args:
            index: Edge index
            features: Feature vector
            edge_index: Full edge index array
            edge_idx_in_list: Position in edge list

        Returns:
            ReconstructedPrimitive for this edge
        """
        # Decode curve type from one-hot (first 6 dims typically)
        feature_dim = len(features)

        if feature_dim >= 6:
            curve_type_idx = int(np.argmax(features[:6]))
        else:
            curve_type_idx = 7  # OTHER

        curve_type = CURVE_TYPE_MAP.get(curve_type_idx, "OTHER")

        params: Dict[str, Any] = {
            "curve_type": curve_type,
        }

        # Length (dim 6)
        if feature_dim > 6:
            params["length"] = float(features[6])

        # Tangent (dims 7-9)
        if feature_dim > 9:
            params["tangent"] = features[7:10].tolist()

        # Curvature (dim 10)
        if feature_dim > 10:
            params["curvature"] = float(features[10])

        # Convexity (dim 11)
        if feature_dim > 11:
            params["convexity"] = float(features[11])

        # For now, return without OCC shape (edge reconstruction requires more context)
        return ReconstructedPrimitive(
            primitive_type="EDGE",
            curve_type=curve_type,
            parameters=params,
            confidence=0.5,  # Lower confidence for edges
        )

    def _build_adjacency(self, edge_index: np.ndarray) -> Dict[int, List[int]]:
        """Build adjacency dictionary from edge index."""
        adjacency: Dict[int, List[int]] = {}

        if edge_index.shape[0] != 2:
            return adjacency

        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i])
            dst = int(edge_index[1, i])

            if src not in adjacency:
                adjacency[src] = []
            if dst not in adjacency[src]:
                adjacency[src].append(dst)

        return adjacency
