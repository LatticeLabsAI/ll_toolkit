"""Enhanced Feature Extraction for CAD AI Models

Extracts industry-standard enhanced node and edge features following BRepNet, UV-Net,
and AAGNet specifications:
- Node features: 48 dimensions (up from 24)
- Edge features: 16 dimensions (up from 8)

Includes UV-grid statistics, extended geometric properties, and topological attributes.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

try:
    from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Edge
    from OCC.Core.BRepTools import BRepTools
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
    from OCC.Core.GeomAbs import (
        GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse,
        GeomAbs_Hyperbola, GeomAbs_Parabola, GeomAbs_BezierCurve,
        GeomAbs_BSplineCurve, GeomAbs_OffsetCurve, GeomAbs_OtherCurve
    )
    from OCC.Core.GeomLProp import GeomLProp_CLProps
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import BRepGProp
    HAS_OCC = True
except ImportError:
    HAS_OCC = False
    logging.warning("OpenCASCADE (pythonocc-core) not available.")

from cadling.lib.geometry.face_geometry import FaceGeometryExtractor
from cadling.lib.graph.features import compute_dihedral_angle
from cadling.lib.graph.brep_graph import SURFACE_TYPES


logger = logging.getLogger(__name__)


# Curve type encoding (5 common types + OTHER)
CURVE_TYPES = {
    "LINE": 0,
    "CIRCLE": 1,
    "ELLIPSE": 2,
    "BSPLINE": 3,
    "BEZIER": 4,
    "OTHER": 5,
}
NUM_CURVE_TYPES = len(CURVE_TYPES)


def extract_enhanced_node_features(
    occ_face: TopoDS_Face,
    surface_type: str = "UNKNOWN",
    uv_grid: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Extract 48-dimensional enhanced node features for a face.

    Features (48 total):
    - Surface type one-hot (10) - PLANE, CYLINDRICAL_SURFACE, etc.
    - Area (1) - surface area
    - Centroid (3) - x, y, z coordinates
    - Normal vector (3) - nx, ny, nz
    - Gaussian curvature (1)
    - Mean curvature (1)
    - Bounding box dimensions (3) - dx, dy, dz
    - UV parametric range (4) - u_min, u_max, v_min, v_max
    - Trimming complexity (1) - number of outer wire edges
    - Orientation (1) - FORWARD=1, REVERSED=-1
    - Is planar flag (1) - binary
    - Area ratio (1) - area / bounding_box_area
    - UV-grid statistics (18):
        - Mean point coords (3)
        - Std point coords (3)
        - Mean normal coords (3)
        - Std normal coords (3)
        - Trimming ratio (1) - percentage inside trimmed region
        - UV coverage (1) - (u_max-u_min)*(v_max-v_min) normalized
        - Normal variation (2) - std of nx, ny (surface roughness)
        - Curvature estimate from normals (2)

    Args:
        occ_face: OpenCASCADE TopoDS_Face object
        surface_type: Surface type string (for one-hot encoding)
        uv_grid: Optional UV-grid array [10, 10, 7] from UV-Net extraction

    Returns:
        np.ndarray of shape [48] containing all enhanced features
    """
    if not HAS_OCC:
        logger.warning("OpenCASCADE not available - returning zero features")
        return np.zeros(48, dtype=np.float32)

    features = []

    try:
        # Initialize geometry extractor
        extractor = FaceGeometryExtractor()

        # 1. Surface type one-hot (10)
        surface_type_upper = surface_type.upper()
        surface_idx = SURFACE_TYPES.get(surface_type_upper, SURFACE_TYPES["UNKNOWN"])
        surface_one_hot = np.zeros(len(SURFACE_TYPES), dtype=np.float32)
        surface_one_hot[surface_idx] = 1.0
        features.append(surface_one_hot)

        # 2. Area (1)
        area = extractor.compute_surface_area(occ_face)
        if area is None or not np.isfinite(area):
            area = 0.0
        features.append(np.array([area], dtype=np.float32))

        # 3. Centroid (3)
        centroid = extractor.compute_centroid(occ_face)
        if centroid is None:
            centroid = [0.0, 0.0, 0.0]
        features.append(np.array(centroid, dtype=np.float32))

        # 4. Normal vector (3)
        normal = extractor.compute_normal_at_center(occ_face)
        if normal is None:
            normal = [0.0, 0.0, 1.0]
        features.append(np.array(normal, dtype=np.float32))

        # 5. Gaussian curvature (1)
        # 6. Mean curvature (1)
        curvature = extractor.compute_curvature_at_center(occ_face)
        if curvature is None:
            gaussian_k, mean_h = 0.0, 0.0
        else:
            gaussian_k, mean_h = curvature
            if not np.isfinite(gaussian_k):
                gaussian_k = 0.0
            if not np.isfinite(mean_h):
                mean_h = 0.0
        features.append(np.array([gaussian_k], dtype=np.float32))
        features.append(np.array([mean_h], dtype=np.float32))

        # 7. Bounding box dimensions (3)
        bbox_dims = extractor.compute_bbox_dimensions(occ_face)
        if bbox_dims is None:
            bbox_dims = [1.0, 1.0, 1.0]
        features.append(np.array(bbox_dims, dtype=np.float32))

        # 8. UV parametric range (4)
        try:
            u_min, u_max, v_min, v_max = BRepTools.breptools_UVBounds(occ_face)
            uv_range = [u_min, u_max, v_min, v_max]
        except:
            uv_range = [0.0, 1.0, 0.0, 1.0]
        features.append(np.array(uv_range, dtype=np.float32))

        # 9. Trimming complexity (1) - count edges in outer wire
        try:
            from OCC.Extend.TopologyUtils import TopologyExplorer
            topo = TopologyExplorer(occ_face)
            num_edges = len(list(topo.edges()))
            trimming_complexity = float(num_edges)
        except:
            trimming_complexity = 0.0
        features.append(np.array([trimming_complexity], dtype=np.float32))

        # 10. Orientation (1)
        orientation = occ_face.Orientation()
        if orientation == TopAbs_FORWARD:
            orientation_value = 1.0
        elif orientation == TopAbs_REVERSED:
            orientation_value = -1.0
        else:
            orientation_value = 0.0
        features.append(np.array([orientation_value], dtype=np.float32))

        # 11. Is planar flag (1)
        is_planar = 1.0 if surface_type_upper == "PLANE" else 0.0
        features.append(np.array([is_planar], dtype=np.float32))

        # 12. Area ratio (1) - area / bounding_box_area
        bbox_area = bbox_dims[0] * bbox_dims[1] if bbox_dims[0] > 0 and bbox_dims[1] > 0 else 1.0
        if bbox_area > 0 and area > 0:
            area_ratio = min(area / bbox_area, 1.0)  # Clamp to [0, 1]
        else:
            area_ratio = 0.0
        features.append(np.array([area_ratio], dtype=np.float32))

        # 13. UV-grid statistics (18)
        if uv_grid is not None and uv_grid.shape == (10, 10, 7):
            # Extract UV-grid channels
            points = uv_grid[:, :, 0:3]  # [10, 10, 3]
            normals = uv_grid[:, :, 3:6]  # [10, 10, 3]
            trimming_mask = uv_grid[:, :, 6]  # [10, 10]

            # Mean point coords (3)
            mean_points = np.mean(points, axis=(0, 1))  # [3]

            # Std point coords (3)
            std_points = np.std(points, axis=(0, 1))  # [3]

            # Mean normal coords (3)
            mean_normals = np.mean(normals, axis=(0, 1))  # [3]

            # Std normal coords (3)
            std_normals = np.std(normals, axis=(0, 1))  # [3]

            # Trimming ratio (1)
            trimming_ratio = np.mean(trimming_mask)  # Percentage inside

            # UV coverage (1)
            u_range = uv_range[1] - uv_range[0]
            v_range = uv_range[3] - uv_range[2]
            uv_coverage = u_range * v_range

            # Normal variation (2) - std of nx, ny (surface roughness)
            normal_var_x = std_normals[0]
            normal_var_y = std_normals[1]

            # Curvature estimate from normals (2)
            # Estimate curvature by looking at normal direction changes
            # Compute gradient magnitude along u and v directions
            try:
                du_normals = np.diff(normals, axis=0)  # [9, 10, 3]
                dv_normals = np.diff(normals, axis=1)  # [10, 9, 3]
                curvature_u = np.mean(np.linalg.norm(du_normals, axis=2))
                curvature_v = np.mean(np.linalg.norm(dv_normals, axis=2))
            except:
                curvature_u, curvature_v = 0.0, 0.0

            uv_stats = np.concatenate([
                mean_points,  # 3
                std_points,   # 3
                mean_normals, # 3
                std_normals,  # 3
                [trimming_ratio],  # 1
                [uv_coverage],     # 1
                [normal_var_x, normal_var_y],  # 2
                [curvature_u, curvature_v]     # 2
            ])
        else:
            # No UV-grid available - use zeros
            uv_stats = np.zeros(18, dtype=np.float32)

        features.append(uv_stats.astype(np.float32))

        # Concatenate all features
        feature_vector = np.concatenate(features)

        # Verify dimension
        if feature_vector.shape[0] != 48:
            logger.warning(f"Feature vector has wrong dimension: {feature_vector.shape[0]}, expected 48")
            # Pad or truncate to 48
            if feature_vector.shape[0] < 48:
                feature_vector = np.pad(feature_vector, (0, 48 - feature_vector.shape[0]))
            else:
                feature_vector = feature_vector[:48]

        return feature_vector

    except Exception as e:
        logger.warning(f"Failed to extract enhanced node features: {e}")
        return np.zeros(48, dtype=np.float32)


def extract_enhanced_edge_features(
    occ_edge: TopoDS_Edge,
    curve_type: str = "OTHER",
    uv_grid: Optional[np.ndarray] = None,
    adjacent_faces: Optional[List[TopoDS_Face]] = None
) -> np.ndarray:
    """
    Extract 16-dimensional enhanced edge features for an edge.

    Features (16 total):
    - Curve type one-hot (6) - LINE, CIRCLE, ELLIPSE, BSPLINE, BEZIER, OTHER
    - Edge length (1)
    - Dihedral angle (1) - angle between adjacent face normals
    - Convexity (1) - single value: -1=concave, 0=tangent, 1=convex
    - Edge midpoint (3) - x, y, z
    - Tangent vector at midpoint (3) - tx, ty, tz (normalized)
    - Edge curvature (1) - curvature at midpoint
    - Is degenerate flag (1) - binary

    Args:
        occ_edge: OpenCASCADE TopoDS_Edge object
        curve_type: Curve type string (for one-hot encoding)
        uv_grid: Optional UV-grid array [10, 6] from UV-Net extraction
        adjacent_faces: Optional list of adjacent TopoDS_Face objects (for dihedral angle)

    Returns:
        np.ndarray of shape [16] containing all enhanced features
    """
    if not HAS_OCC:
        logger.warning("OpenCASCADE not available - returning zero features")
        return np.zeros(16, dtype=np.float32)

    features = []

    try:
        # 1. Curve type one-hot (6)
        curve_type_upper = curve_type.upper()
        curve_idx = CURVE_TYPES.get(curve_type_upper, CURVE_TYPES["OTHER"])
        curve_one_hot = np.zeros(NUM_CURVE_TYPES, dtype=np.float32)
        curve_one_hot[curve_idx] = 1.0
        features.append(curve_one_hot)

        # 2. Edge length (1)
        try:
            props = GProp_GProps()
            BRepGProp.brepgprop_LinearProperties(occ_edge, props)
            edge_length = props.Mass()
            if not np.isfinite(edge_length):
                edge_length = 0.0
        except:
            edge_length = 0.0
        features.append(np.array([edge_length], dtype=np.float32))

        # Get curve and parameter range for subsequent features
        try:
            curve, u_min, u_max = BRep_Tool.Curve(occ_edge)
            u_mid = (u_min + u_max) / 2.0
            has_curve = curve is not None
        except:
            has_curve = False
            u_mid = 0.0

        # 3. Dihedral angle (1)
        if adjacent_faces is not None and len(adjacent_faces) == 2:
            try:
                extractor = FaceGeometryExtractor()
                normal1 = extractor.compute_normal_at_center(adjacent_faces[0])
                normal2 = extractor.compute_normal_at_center(adjacent_faces[1])
                if normal1 is not None and normal2 is not None:
                    dihedral = compute_dihedral_angle(
                        np.array(normal1),
                        np.array(normal2)
                    )
                else:
                    dihedral = np.pi / 2.0  # Default to 90 degrees
            except:
                dihedral = np.pi / 2.0
        else:
            dihedral = np.pi / 2.0  # Default to 90 degrees
        features.append(np.array([dihedral], dtype=np.float32))

        # 4. Convexity (1) - derived from dihedral angle
        # convex: dihedral > π/2, concave: dihedral < π/2, tangent: dihedral ≈ π/2
        if dihedral > (np.pi / 2.0 + 0.1):
            convexity = 1.0  # Convex
        elif dihedral < (np.pi / 2.0 - 0.1):
            convexity = -1.0  # Concave
        else:
            convexity = 0.0  # Tangent
        features.append(np.array([convexity], dtype=np.float32))

        # 5. Edge midpoint (3)
        # 6. Tangent vector at midpoint (3)
        # 7. Edge curvature (1)
        if has_curve:
            try:
                # Get point and derivatives at midpoint
                props = GeomLProp_CLProps(curve, u_mid, 2, 1e-9)  # Order 2 for curvature

                # Midpoint
                point = props.Value()
                midpoint = [point.X(), point.Y(), point.Z()]

                # Tangent vector
                tangent = props.Tangent()
                tangent_vec = [tangent.X(), tangent.Y(), tangent.Z()]
                # Normalize
                tangent_norm = np.linalg.norm(tangent_vec)
                if tangent_norm > 1e-10:
                    tangent_vec = (np.array(tangent_vec) / tangent_norm).tolist()

                # Curvature
                if props.IsTangentDefined():
                    edge_curvature = props.Curvature()
                    if not np.isfinite(edge_curvature):
                        edge_curvature = 0.0
                else:
                    edge_curvature = 0.0

            except:
                midpoint = [0.0, 0.0, 0.0]
                tangent_vec = [1.0, 0.0, 0.0]
                edge_curvature = 0.0
        else:
            midpoint = [0.0, 0.0, 0.0]
            tangent_vec = [1.0, 0.0, 0.0]
            edge_curvature = 0.0

        features.append(np.array(midpoint, dtype=np.float32))
        features.append(np.array(tangent_vec, dtype=np.float32))
        features.append(np.array([edge_curvature], dtype=np.float32))

        # 8. Is degenerate flag (1)
        try:
            is_degenerate = 1.0 if BRep_Tool.Degenerated(occ_edge) else 0.0
        except:
            is_degenerate = 0.0
        features.append(np.array([is_degenerate], dtype=np.float32))

        # Concatenate all features
        feature_vector = np.concatenate(features)

        # Verify dimension
        if feature_vector.shape[0] != 16:
            logger.warning(f"Edge feature vector has wrong dimension: {feature_vector.shape[0]}, expected 16")
            # Pad or truncate to 16
            if feature_vector.shape[0] < 16:
                feature_vector = np.pad(feature_vector, (0, 16 - feature_vector.shape[0]))
            else:
                feature_vector = feature_vector[:16]

        return feature_vector

    except Exception as e:
        logger.warning(f"Failed to extract enhanced edge features: {e}")
        return np.zeros(16, dtype=np.float32)


def get_curve_type_from_edge(occ_edge: TopoDS_Edge) -> str:
    """
    Determine curve type from OpenCASCADE edge.

    Args:
        occ_edge: TopoDS_Edge object

    Returns:
        Curve type string: "LINE", "CIRCLE", "ELLIPSE", "BSPLINE", "BEZIER", or "OTHER"
    """
    if not HAS_OCC:
        return "OTHER"

    try:
        curve, u_min, u_max = BRep_Tool.Curve(occ_edge)
        if curve is None:
            return "OTHER"

        # Get curve type using GeomAdaptor
        from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
        adaptor = GeomAdaptor_Curve(curve)
        curve_type = adaptor.GetType()

        if curve_type == GeomAbs_Line:
            return "LINE"
        elif curve_type == GeomAbs_Circle:
            return "CIRCLE"
        elif curve_type == GeomAbs_Ellipse:
            return "ELLIPSE"
        elif curve_type == GeomAbs_BSplineCurve:
            return "BSPLINE"
        elif curve_type == GeomAbs_BezierCurve:
            return "BEZIER"
        else:
            return "OTHER"

    except Exception as e:
        logger.debug(f"Failed to determine curve type: {e}")
        return "OTHER"


def get_surface_type_from_face(occ_face: TopoDS_Face) -> str:
    """
    Determine surface type from OpenCASCADE face.

    Args:
        occ_face: TopoDS_Face object

    Returns:
        Surface type string matching SURFACE_TYPES keys
    """
    if not HAS_OCC:
        return "UNKNOWN"

    try:
        from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
        from OCC.Core.GeomAbs import (
            GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
            GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
            GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion
        )

        surface = BRep_Tool.Surface(occ_face)
        if surface is None:
            return "UNKNOWN"

        adaptor = GeomAdaptor_Surface(surface)
        surface_type = adaptor.GetType()

        if surface_type == GeomAbs_Plane:
            return "PLANE"
        elif surface_type == GeomAbs_Cylinder:
            return "CYLINDRICAL_SURFACE"
        elif surface_type == GeomAbs_Cone:
            return "CONICAL_SURFACE"
        elif surface_type == GeomAbs_Sphere:
            return "SPHERICAL_SURFACE"
        elif surface_type == GeomAbs_Torus:
            return "TOROIDAL_SURFACE"
        elif surface_type == GeomAbs_BSplineSurface:
            return "B_SPLINE_SURFACE"
        elif surface_type == GeomAbs_SurfaceOfRevolution:
            return "SURFACE_OF_REVOLUTION"
        elif surface_type == GeomAbs_SurfaceOfExtrusion:
            return "SURFACE_OF_LINEAR_EXTRUSION"
        else:
            return "UNKNOWN"

    except Exception as e:
        logger.debug(f"Failed to determine surface type: {e}")
        return "UNKNOWN"
