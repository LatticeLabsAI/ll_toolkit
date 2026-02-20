"""OCC Wrapper for AI/ML-friendly pythonocc-core access.

This module provides a unified wrapper around pythonocc-core and OCCWL that enables
AI-native CAD processing. Each method tries OCCWL first for enhanced functionality,
then falls back to pythonocc for compatibility.

The wrapper provides:
- OCCShape: Wraps TopoDS_Shape with face/edge iteration, adjacency graphs, feature extraction
- OCCFace: Wraps TopoDS_Face with surface type, UV sampling, curvature computation
- OCCEdge: Wraps TopoDS_Edge with curve type, length, tangent computation

Example:
    from cadling.lib.occ_wrapper import OCCShape, HAS_OCCWL

    shape = OCCShape(topods_shape)
    for face in shape.faces():
        print(f"Surface type: {face.surface_type()}")
        print(f"Area: {face.area()}")

    # Get face adjacency for GNN
    adjacency = shape.face_adjacency_graph()

    # Extract all features (cached)
    features = shape.extract_all_features()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

_log = logging.getLogger(__name__)

# Availability flags
HAS_OCC = False
HAS_OCCWL = False

# Lazy import OCC modules
try:
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex, TopoDS_Wire
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE
    from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepTools import breptools, BRepTools_WireExplorer
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
        GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
        GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, GeomAbs_OtherSurface,
        GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola,
        GeomAbs_Parabola, GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_OtherCurve
    )
    from OCC.Core.GeomLProp import GeomLProp_SLProps
    from OCC.Core.BRepLProp import BRepLProp_CLProps
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopoDS import topods
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir

    HAS_OCC = True
    _log.debug("pythonocc-core loaded successfully")

    # Patch OCC.Extend.DataExchange for OCCWL compatibility with pythonocc 7.8.x
    # The function list_of_shapes_to_compound was removed in 7.8.x but OCCWL needs it
    def _patch_occwl_compatibility():
        """Add missing list_of_shapes_to_compound to OCC.Extend.DataExchange."""
        try:
            from OCC.Extend import DataExchange
            if not hasattr(DataExchange, 'list_of_shapes_to_compound'):
                from OCC.Core.TopoDS import TopoDS_Compound
                from OCC.Core.BRep import BRep_Builder

                def list_of_shapes_to_compound(shapes):
                    """Convert a list of shapes to a compound."""
                    compound = TopoDS_Compound()
                    builder = BRep_Builder()
                    builder.MakeCompound(compound)
                    for shape in shapes:
                        builder.Add(compound, shape)
                    return compound, True

                DataExchange.list_of_shapes_to_compound = list_of_shapes_to_compound
                _log.debug("Patched OCC.Extend.DataExchange with list_of_shapes_to_compound")
        except Exception as e:
            _log.debug(f"Could not patch DataExchange: {e}")

    _patch_occwl_compatibility()

except ImportError as e:
    _log.warning(f"pythonocc-core not available: {e}. OCC wrapper will return empty results.")

# Try to import OCCWL for enhanced functionality
try:
    from occwl.solid import Solid
    from occwl.face import Face as OCCWLFace
    from occwl.edge import Edge as OCCWLEdge
    from occwl.graph import face_adjacency
    from occwl.uvgrid import uvgrid, ugrid

    HAS_OCCWL = True
    _log.debug("OCCWL loaded successfully")
except ImportError as e:
    _log.debug(f"OCCWL not available: {e}. Install via: pip install git+https://github.com/AutodeskAILab/occwl.git")


# Surface type mapping
SURFACE_TYPE_MAP = {
    "PLANE": 0,
    "CYLINDER": 1,
    "CONE": 2,
    "SPHERE": 3,
    "TORUS": 4,
    "BEZIER": 5,
    "BSPLINE": 6,
    "REVOLUTION": 7,
    "EXTRUSION": 8,
    "OTHER": 9,
}

# Curve type mapping
CURVE_TYPE_MAP = {
    "LINE": 0,
    "CIRCLE": 1,
    "ELLIPSE": 2,
    "HYPERBOLA": 3,
    "PARABOLA": 4,
    "BEZIER": 5,
    "BSPLINE": 6,
    "OTHER": 7,
}


class OCCEdge:
    """Wrapper for TopoDS_Edge with AI-friendly feature extraction.

    Provides unified access to edge properties via OCCWL (if available)
    or pythonocc fallbacks.

    Attributes:
        _occ_edge: The underlying TopoDS_Edge object
        _occwl_edge: OCCWL Edge wrapper (if available)
    """

    def __init__(self, occ_edge: "TopoDS_Edge"):
        """Initialize OCCEdge wrapper.

        Args:
            occ_edge: OpenCASCADE TopoDS_Edge object
        """
        self._occ_edge = occ_edge
        self._occwl_edge = None

        if HAS_OCCWL:
            try:
                self._occwl_edge = OCCWLEdge(occ_edge)
            except Exception as e:
                _log.debug(f"Failed to create OCCWL Edge: {e}")

    @property
    def occ_edge(self) -> "TopoDS_Edge":
        """Get underlying OCC edge."""
        return self._occ_edge

    def curve_type(self) -> str:
        """Get the curve type of this edge.

        Returns:
            String name: LINE, CIRCLE, ELLIPSE, HYPERBOLA, PARABOLA, BEZIER, BSPLINE, or OTHER
        """
        if self._occwl_edge is not None:
            try:
                return self._occwl_edge.curve_type().upper()
            except Exception:
                pass

        return self._curve_type_pythonocc()

    def _curve_type_pythonocc(self) -> str:
        """Get curve type using pythonocc."""
        if not HAS_OCC:
            return "OTHER"

        try:
            adaptor = BRepAdaptor_Curve(self._occ_edge)
            curve_type = adaptor.GetType()

            type_map = {
                GeomAbs_Line: "LINE",
                GeomAbs_Circle: "CIRCLE",
                GeomAbs_Ellipse: "ELLIPSE",
                GeomAbs_Hyperbola: "HYPERBOLA",
                GeomAbs_Parabola: "PARABOLA",
                GeomAbs_BezierCurve: "BEZIER",
                GeomAbs_BSplineCurve: "BSPLINE",
                GeomAbs_OtherCurve: "OTHER",
            }

            return type_map.get(curve_type, "OTHER")

        except Exception as e:
            _log.debug(f"Failed to get curve type: {e}")
            return "OTHER"

    def length(self) -> float:
        """Compute edge length.

        Returns:
            Edge length in model units
        """
        if self._occwl_edge is not None:
            try:
                return self._occwl_edge.length()
            except Exception:
                pass

        return self._length_pythonocc()

    def _length_pythonocc(self) -> float:
        """Compute length using pythonocc."""
        if not HAS_OCC:
            return 0.0

        try:
            props = GProp_GProps()
            brepgprop.LinearProperties(self._occ_edge, props)
            return props.Mass()
        except Exception as e:
            _log.debug(f"Failed to compute edge length: {e}")
            return 0.0

    def tangent_at(self, u: float = 0.5) -> List[float]:
        """Compute tangent vector at parameter u.

        Args:
            u: Parameter value in [0, 1] range (0.5 = midpoint)

        Returns:
            Normalized tangent vector [x, y, z]
        """
        if not HAS_OCC:
            return [0.0, 0.0, 1.0]

        try:
            adaptor = BRepAdaptor_Curve(self._occ_edge)
            u_start = adaptor.FirstParameter()
            u_end = adaptor.LastParameter()
            u_param = u_start + u * (u_end - u_start)

            point = gp_Pnt()
            tangent = gp_Vec()
            adaptor.D1(u_param, point, tangent)

            mag = tangent.Magnitude()
            if mag > 1e-10:
                tangent.Normalize()
                return [tangent.X(), tangent.Y(), tangent.Z()]

        except Exception as e:
            _log.debug(f"Failed to compute tangent: {e}")

        return [0.0, 0.0, 1.0]

    def curvature_at(self, u: float = 0.5) -> float:
        """Compute curvature at parameter u.

        Args:
            u: Parameter value in [0, 1] range (0.5 = midpoint)

        Returns:
            Curvature value (0 for lines, 1/r for circles)
        """
        if not HAS_OCC:
            return 0.0

        try:
            adaptor = BRepAdaptor_Curve(self._occ_edge)
            u_start = adaptor.FirstParameter()
            u_end = adaptor.LastParameter()
            u_param = u_start + u * (u_end - u_start)

            props = BRepLProp_CLProps(adaptor, u_param, 2, 1e-6)

            if props.IsTangentDefined():
                curvature = props.Curvature()
                if curvature >= 0 and curvature < 1e10:
                    return curvature

        except Exception as e:
            _log.debug(f"Failed to compute curvature: {e}")

        return 0.0

    def endpoints(self) -> Tuple[List[float], List[float]]:
        """Get edge start and end points.

        Returns:
            Tuple of (start_point, end_point) as [x, y, z] lists
        """
        if not HAS_OCC:
            return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        try:
            adaptor = BRepAdaptor_Curve(self._occ_edge)
            p_start = adaptor.Value(adaptor.FirstParameter())
            p_end = adaptor.Value(adaptor.LastParameter())

            return (
                [p_start.X(), p_start.Y(), p_start.Z()],
                [p_end.X(), p_end.Y(), p_end.Z()]
            )

        except Exception as e:
            _log.debug(f"Failed to get endpoints: {e}")
            return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def extract_features(self) -> Dict[str, Any]:
        """Extract all features for ML use.

        Returns:
            Dictionary with curve_type, length, tangent, curvature, endpoints
        """
        return {
            "curve_type": self.curve_type(),
            "curve_type_idx": CURVE_TYPE_MAP.get(self.curve_type(), 7),
            "length": self.length(),
            "tangent": self.tangent_at(0.5),
            "curvature": self.curvature_at(0.5),
            "start_point": self.endpoints()[0],
            "end_point": self.endpoints()[1],
        }


class OCCFace:
    """Wrapper for TopoDS_Face with AI-friendly feature extraction.

    Provides unified access to face properties via OCCWL (if available)
    or pythonocc fallbacks.

    Attributes:
        _occ_face: The underlying TopoDS_Face object
        _occwl_face: OCCWL Face wrapper (if available)
    """

    def __init__(self, occ_face: "TopoDS_Face"):
        """Initialize OCCFace wrapper.

        Args:
            occ_face: OpenCASCADE TopoDS_Face object
        """
        self._occ_face = occ_face
        self._occwl_face = None

        if HAS_OCCWL:
            try:
                self._occwl_face = OCCWLFace(occ_face)
            except Exception as e:
                _log.debug(f"Failed to create OCCWL Face: {e}")

    @property
    def occ_face(self) -> "TopoDS_Face":
        """Get underlying OCC face."""
        return self._occ_face

    def surface_type(self) -> str:
        """Get the surface type of this face.

        Returns:
            String name: PLANE, CYLINDER, CONE, SPHERE, TORUS, BEZIER, BSPLINE, REVOLUTION, EXTRUSION, or OTHER
        """
        if self._occwl_face is not None:
            try:
                return self._occwl_face.surface_type().upper()
            except Exception:
                pass

        return self._surface_type_pythonocc()

    def _surface_type_pythonocc(self) -> str:
        """Get surface type using pythonocc."""
        if not HAS_OCC:
            return "OTHER"

        try:
            adaptor = BRepAdaptor_Surface(self._occ_face)
            surface_type = adaptor.GetType()

            type_map = {
                GeomAbs_Plane: "PLANE",
                GeomAbs_Cylinder: "CYLINDER",
                GeomAbs_Cone: "CONE",
                GeomAbs_Sphere: "SPHERE",
                GeomAbs_Torus: "TORUS",
                GeomAbs_BezierSurface: "BEZIER",
                GeomAbs_BSplineSurface: "BSPLINE",
                GeomAbs_SurfaceOfRevolution: "REVOLUTION",
                GeomAbs_SurfaceOfExtrusion: "EXTRUSION",
                GeomAbs_OtherSurface: "OTHER",
            }

            return type_map.get(surface_type, "OTHER")

        except Exception as e:
            _log.debug(f"Failed to get surface type: {e}")
            return "OTHER"

    def area(self) -> float:
        """Compute surface area.

        Returns:
            Surface area in model units squared
        """
        if self._occwl_face is not None:
            try:
                return self._occwl_face.area()
            except Exception:
                pass

        return self._area_pythonocc()

    def _area_pythonocc(self) -> float:
        """Compute area using pythonocc."""
        if not HAS_OCC:
            return 0.0

        try:
            props = GProp_GProps()
            brepgprop.SurfaceProperties(self._occ_face, props)
            return props.Mass()
        except Exception as e:
            _log.debug(f"Failed to compute area: {e}")
            return 0.0

    def normal_at(self, u: float = 0.5, v: float = 0.5) -> List[float]:
        """Compute normal vector at parameter (u, v).

        Args:
            u: U parameter in [0, 1] range
            v: V parameter in [0, 1] range

        Returns:
            Unit normal vector [x, y, z]
        """
        if self._occwl_face is not None:
            try:
                # OCCWL uses normalized UV
                normal = self._occwl_face.normal(np.array([u, v]))
                return normal.tolist()
            except Exception:
                pass

        return self._normal_pythonocc(u, v)

    def _normal_pythonocc(self, u: float = 0.5, v: float = 0.5) -> List[float]:
        """Compute normal using pythonocc."""
        if not HAS_OCC:
            return [0.0, 0.0, 1.0]

        try:
            # Get UV bounds
            u_min, u_max, v_min, v_max = breptools.UVBounds(self._occ_face)
            u_param = u_min + u * (u_max - u_min)
            v_param = v_min + v * (v_max - v_min)

            surface = BRep_Tool.Surface(self._occ_face)
            props = GeomLProp_SLProps(surface, u_param, v_param, 1, 1e-9)

            normal_dir = props.Normal()
            return [normal_dir.X(), normal_dir.Y(), normal_dir.Z()]

        except Exception as e:
            _log.debug(f"Failed to compute normal: {e}")
            return [0.0, 0.0, 1.0]

    def centroid(self) -> List[float]:
        """Compute face centroid (center of mass).

        Returns:
            Centroid coordinates [x, y, z]
        """
        if not HAS_OCC:
            return [0.0, 0.0, 0.0]

        try:
            props = GProp_GProps()
            brepgprop.SurfaceProperties(self._occ_face, props)
            center = props.CentreOfMass()
            return [center.X(), center.Y(), center.Z()]
        except Exception as e:
            _log.debug(f"Failed to compute centroid: {e}")
            return [0.0, 0.0, 0.0]

    def curvature_at(self, u: float = 0.5, v: float = 0.5) -> Tuple[float, float]:
        """Compute Gaussian and mean curvature at parameter (u, v).

        Args:
            u: U parameter in [0, 1] range
            v: V parameter in [0, 1] range

        Returns:
            Tuple of (gaussian_curvature, mean_curvature)
        """
        if not HAS_OCC:
            return (0.0, 0.0)

        try:
            u_min, u_max, v_min, v_max = breptools.UVBounds(self._occ_face)
            u_param = u_min + u * (u_max - u_min)
            v_param = v_min + v * (v_max - v_min)

            surface = BRep_Tool.Surface(self._occ_face)
            props = GeomLProp_SLProps(surface, u_param, v_param, 2, 1e-9)

            if not props.IsCurvatureDefined():
                return (0.0, 0.0)

            return (props.GaussianCurvature(), props.MeanCurvature())

        except Exception as e:
            _log.debug(f"Failed to compute curvature: {e}")
            return (0.0, 0.0)

    def bbox(self) -> Tuple[List[float], List[float]]:
        """Compute bounding box.

        Returns:
            Tuple of (min_point, max_point) as [x, y, z] lists
        """
        if not HAS_OCC:
            return ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        try:
            bbox = Bnd_Box()
            brepbndlib.Add(self._occ_face, bbox)
            x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()
            return (
                [x_min, y_min, z_min],
                [x_max, y_max, z_max]
            )
        except Exception as e:
            _log.debug(f"Failed to compute bbox: {e}")
            return ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    def uv_grid(self, num_u: int = 10, num_v: int = 10) -> Optional[np.ndarray]:
        """Sample UV grid from face.

        Args:
            num_u: Number of samples in U direction
            num_v: Number of samples in V direction

        Returns:
            Array of shape [num_u, num_v, 7] with (x, y, z, nx, ny, nz, mask)
            or None if extraction fails
        """
        if HAS_OCCWL and self._occwl_face is not None:
            try:
                from cadling.lib.geometry.uv_grid_extractor import FaceUVGridExtractor

                return FaceUVGridExtractor.extract_uv_grid(
                    self._occ_face, num_u, num_v
                )
            except Exception:
                pass

        # Fallback: manual UV sampling
        return self._uv_grid_pythonocc(num_u, num_v)

    def _uv_grid_pythonocc(self, num_u: int, num_v: int) -> Optional[np.ndarray]:
        """Sample UV grid using pythonocc."""
        if not HAS_OCC:
            return None

        try:
            grid = np.zeros((num_u, num_v, 7), dtype=np.float32)

            u_min, u_max, v_min, v_max = breptools.UVBounds(self._occ_face)
            surface = BRep_Tool.Surface(self._occ_face)

            for i in range(num_u):
                for j in range(num_v):
                    u = i / (num_u - 1) if num_u > 1 else 0.5
                    v = j / (num_v - 1) if num_v > 1 else 0.5

                    u_param = u_min + u * (u_max - u_min)
                    v_param = v_min + v * (v_max - v_min)

                    # Get point
                    point = surface.Value(u_param, v_param)
                    grid[i, j, 0] = point.X()
                    grid[i, j, 1] = point.Y()
                    grid[i, j, 2] = point.Z()

                    # Get normal
                    try:
                        props = GeomLProp_SLProps(surface, u_param, v_param, 1, 1e-9)
                        normal = props.Normal()
                        grid[i, j, 3] = normal.X()
                        grid[i, j, 4] = normal.Y()
                        grid[i, j, 5] = normal.Z()
                    except Exception:
                        grid[i, j, 3:6] = [0.0, 0.0, 1.0]

                    # Mask (assume inside for now)
                    grid[i, j, 6] = 1.0

            return grid

        except Exception as e:
            _log.debug(f"Failed to sample UV grid: {e}")
            return None

    def edges(self) -> Iterator["OCCEdge"]:
        """Iterate over edges in this face.

        Yields:
            OCCEdge wrappers for each edge
        """
        if not HAS_OCC:
            return

        try:
            explorer = TopExp_Explorer(self._occ_face, TopAbs_EDGE)
            while explorer.More():
                edge = topods.Edge(explorer.Current())
                yield OCCEdge(edge)
                explorer.Next()
        except Exception as e:
            _log.debug(f"Failed to iterate edges: {e}")

    def extract_features(self) -> Dict[str, Any]:
        """Extract all features for ML use.

        Returns:
            Dictionary with surface_type, area, normal, centroid, curvature, bbox
        """
        bbox_min, bbox_max = self.bbox()
        gauss_curv, mean_curv = self.curvature_at()

        return {
            "surface_type": self.surface_type(),
            "surface_type_idx": SURFACE_TYPE_MAP.get(self.surface_type(), 9),
            "area": self.area(),
            "normal": self.normal_at(),
            "centroid": self.centroid(),
            "gaussian_curvature": gauss_curv,
            "mean_curvature": mean_curv,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_dimensions": [
                bbox_max[i] - bbox_min[i] for i in range(3)
            ],
        }


class OCCShape:
    """Wrapper for TopoDS_Shape with AI-friendly feature extraction.

    Provides unified access to shape topology and geometry via OCCWL (if available)
    or pythonocc fallbacks.

    Attributes:
        _occ_shape: The underlying TopoDS_Shape object
        _occwl_solid: OCCWL Solid wrapper (if available and shape is a solid)
        _face_registry: Cached face ID -> face mapping
        _edge_registry: Cached edge ID -> edge mapping
    """

    def __init__(self, occ_shape: "TopoDS_Shape"):
        """Initialize OCCShape wrapper.

        Args:
            occ_shape: OpenCASCADE TopoDS_Shape object
        """
        self._occ_shape = occ_shape
        self._occwl_solid = None
        self._face_registry: Optional[Dict[str, "TopoDS_Face"]] = None
        self._edge_registry: Optional[Dict[str, "TopoDS_Edge"]] = None

        if HAS_OCCWL:
            try:
                self._occwl_solid = Solid(occ_shape)
            except Exception as e:
                _log.debug(f"Failed to create OCCWL Solid: {e}")

    @property
    def occ_shape(self) -> "TopoDS_Shape":
        """Get underlying OCC shape."""
        return self._occ_shape

    def get_shape_id(self, shape) -> str:
        """Get stable ID for a shape using HashCode with fallback.

        Args:
            shape: Any TopoDS_Shape object

        Returns:
            String representation of the shape's hash code
        """
        try:
            return str(shape.HashCode(2**31 - 1))
        except AttributeError:
            return str(hash(shape))

    def faces(self) -> Iterator[OCCFace]:
        """Iterate over all faces in the shape.

        Yields:
            OCCFace wrappers for each face
        """
        if not HAS_OCC:
            return

        try:
            explorer = TopExp_Explorer(self._occ_shape, TopAbs_FACE)
            while explorer.More():
                face = topods.Face(explorer.Current())
                yield OCCFace(face)
                explorer.Next()
        except Exception as e:
            _log.debug(f"Failed to iterate faces: {e}")

    def face_list(self) -> List[OCCFace]:
        """Get list of all faces.

        Returns:
            List of OCCFace wrappers
        """
        return list(self.faces())

    def edges(self) -> Iterator[OCCEdge]:
        """Iterate over all edges in the shape.

        Yields:
            OCCEdge wrappers for each edge
        """
        if not HAS_OCC:
            return

        try:
            explorer = TopExp_Explorer(self._occ_shape, TopAbs_EDGE)
            while explorer.More():
                edge = topods.Edge(explorer.Current())
                yield OCCEdge(edge)
                explorer.Next()
        except Exception as e:
            _log.debug(f"Failed to iterate edges: {e}")

    def edge_list(self) -> List[OCCEdge]:
        """Get list of all edges.

        Returns:
            List of OCCEdge wrappers
        """
        return list(self.edges())

    def num_faces(self) -> int:
        """Count number of faces."""
        return len(self.face_list())

    def num_edges(self) -> int:
        """Count number of edges."""
        return len(self.edge_list())

    def num_vertices(self) -> int:
        """Count number of vertices."""
        if not HAS_OCC:
            return 0

        count = 0
        try:
            explorer = TopExp_Explorer(self._occ_shape, TopAbs_VERTEX)
            while explorer.More():
                count += 1
                explorer.Next()
        except Exception as e:
            _log.debug(f"Failed to count vertices: {e}")

        return count

    def face_adjacency_graph(self) -> Dict[int, List[int]]:
        """Build face adjacency graph.

        Returns:
            Dictionary mapping face index to list of adjacent face indices
        """
        if HAS_OCCWL and self._occwl_solid is not None:
            try:
                return face_adjacency(self._occwl_solid)
            except Exception:
                pass

        return self._face_adjacency_pythonocc()

    def _face_adjacency_pythonocc(self) -> Dict[int, List[int]]:
        """Build face adjacency using pythonocc."""
        if not HAS_OCC:
            return {}

        try:
            # Build edge -> faces mapping
            edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
            topexp.MapShapesAndAncestors(
                self._occ_shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces
            )

            # Get all faces with stable indices
            faces = self.face_list()
            face_to_idx = {
                self.get_shape_id(f.occ_face): i for i, f in enumerate(faces)
            }

            # Build adjacency
            adjacency: Dict[int, List[int]] = {i: [] for i in range(len(faces))}

            for i in range(1, edge_to_faces.Extent() + 1):
                face_list_shape = edge_to_faces.FindFromIndex(i)
                face_indices = []

                for j in range(1, face_list_shape.Extent() + 1):
                    face = topods.Face(face_list_shape.Value(j))
                    face_id = self.get_shape_id(face)
                    if face_id in face_to_idx:
                        face_indices.append(face_to_idx[face_id])

                # Add adjacency for pairs
                for fi in face_indices:
                    for fj in face_indices:
                        if fi != fj and fj not in adjacency[fi]:
                            adjacency[fi].append(fj)

            return adjacency

        except Exception as e:
            _log.debug(f"Failed to build adjacency: {e}")
            return {}

    def register_all_faces(self) -> Dict[str, "TopoDS_Face"]:
        """Register all faces with stable IDs.

        Returns:
            Dictionary mapping face_id -> TopoDS_Face
        """
        if self._face_registry is not None:
            return self._face_registry

        self._face_registry = {}

        for face in self.faces():
            face_id = self.get_shape_id(face.occ_face)
            self._face_registry[face_id] = face.occ_face

        return self._face_registry

    def register_all_edges(self) -> Dict[str, "TopoDS_Edge"]:
        """Register all edges with stable IDs.

        Returns:
            Dictionary mapping edge_id -> TopoDS_Edge
        """
        if self._edge_registry is not None:
            return self._edge_registry

        self._edge_registry = {}

        for edge in self.edges():
            edge_id = self.get_shape_id(edge.occ_edge)
            self._edge_registry[edge_id] = edge.occ_edge

        return self._edge_registry

    def bbox(self) -> Tuple[List[float], List[float]]:
        """Compute overall bounding box.

        Returns:
            Tuple of (min_point, max_point) as [x, y, z] lists
        """
        if not HAS_OCC:
            return ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        try:
            bbox = Bnd_Box()
            brepbndlib.Add(self._occ_shape, bbox)
            x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()
            return (
                [x_min, y_min, z_min],
                [x_max, y_max, z_max]
            )
        except Exception as e:
            _log.debug(f"Failed to compute bbox: {e}")
            return ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    def volume(self) -> float:
        """Compute solid volume (if applicable).

        Returns:
            Volume in model units cubed
        """
        if not HAS_OCC:
            return 0.0

        try:
            props = GProp_GProps()
            brepgprop.VolumeProperties(self._occ_shape, props)
            return props.Mass()
        except Exception as e:
            _log.debug(f"Failed to compute volume: {e}")
            return 0.0

    def surface_area(self) -> float:
        """Compute total surface area.

        Returns:
            Total surface area in model units squared
        """
        if not HAS_OCC:
            return 0.0

        try:
            props = GProp_GProps()
            brepgprop.SurfaceProperties(self._occ_shape, props)
            return props.Mass()
        except Exception as e:
            _log.debug(f"Failed to compute surface area: {e}")
            return 0.0

    def extract_all_features(self) -> Dict[str, Any]:
        """Extract comprehensive features for ML/AI use.

        Returns:
            Dictionary with shape-level and per-face/edge features
        """
        bbox_min, bbox_max = self.bbox()

        # Shape-level features
        features = {
            "num_faces": self.num_faces(),
            "num_edges": self.num_edges(),
            "num_vertices": self.num_vertices(),
            "volume": self.volume(),
            "surface_area": self.surface_area(),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_dimensions": [
                bbox_max[i] - bbox_min[i] for i in range(3)
            ],
            "faces": [],
            "edges": [],
            "adjacency": self.face_adjacency_graph(),
        }

        # Per-face features
        for face in self.faces():
            features["faces"].append(face.extract_features())

        # Per-edge features
        for edge in self.edges():
            features["edges"].append(edge.extract_features())

        return features

    def extract_face_feature_matrix(self) -> np.ndarray:
        """Extract face features as numpy matrix for GNN input.

        Returns:
            Array of shape [num_faces, 24] with face features
        """
        faces = self.face_list()
        num_faces = len(faces)

        if num_faces == 0:
            return np.zeros((0, 24), dtype=np.float32)

        features = np.zeros((num_faces, 24), dtype=np.float32)

        for i, face in enumerate(faces):
            f = face.extract_features()

            # Surface type one-hot (10 dims)
            type_idx = f["surface_type_idx"]
            features[i, type_idx] = 1.0

            # Area (1 dim)
            features[i, 10] = f["area"]

            # Curvature (2 dims)
            features[i, 11] = f["gaussian_curvature"]
            features[i, 12] = f["mean_curvature"]

            # Normal (3 dims)
            features[i, 13:16] = f["normal"]

            # Centroid (3 dims)
            features[i, 16:19] = f["centroid"]

            # Bbox dimensions (3 dims)
            features[i, 19:22] = f["bbox_dimensions"]

            # Reserved (2 dims)
            features[i, 22] = 0.0
            features[i, 23] = 0.0

        return features

    def extract_edge_index(self) -> np.ndarray:
        """Extract edge index for GNN in COO format.

        Returns:
            Array of shape [2, num_edges] with source and target node indices
        """
        adjacency = self.face_adjacency_graph()

        edges = []
        for src, targets in adjacency.items():
            for tgt in targets:
                edges.append([src, tgt])

        if not edges:
            return np.zeros((2, 0), dtype=np.int64)

        return np.array(edges, dtype=np.int64).T
