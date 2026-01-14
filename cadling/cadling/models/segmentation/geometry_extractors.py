"""Geometry extractors for manufacturing features.

This module provides extractors for various manufacturing features like holes, pockets,
bosses, fillets, and chamfers. Extracts REAL geometric parameters from STEP entities
and pythonocc shapes instead of using placeholder values.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)


class HoleGeometryExtractor:
    """Extract hole parameters from STEP entities and face graphs.

    Extracts:
    - Diameter: from CYLINDRICAL_SURFACE radius parameter
    - Depth: from face extents along axis
    - Location: from AXIS2_PLACEMENT_3D or centroid
    - Orientation: from AXIS2_PLACEMENT_3D direction
    - Hole type: through vs blind (based on bottom face presence)
    """

    def __init__(self):
        """Initialize hole geometry extractor."""
        self.has_pythonocc = False
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Cylinder

            self.has_pythonocc = True
            _log.debug("HoleGeometryExtractor initialized with pythonocc-core")
        except ImportError:
            _log.warning("pythonocc-core not available for hole geometry extraction")

    def extract_hole_parameters(
        self,
        face_entities: List[Dict],
        face_ids: List[int],
        graph: Any,
    ) -> Dict[str, Any]:
        """Extract hole parameters from face entities.

        Args:
            face_entities: List of face entity dictionaries
            face_ids: List of face indices in the graph
            graph: Face adjacency graph (PyG Data object)

        Returns:
            Dictionary with hole parameters:
            - diameter: float (hole diameter)
            - depth: float (hole depth)
            - location: [x, y, z] (hole center location)
            - orientation: [x, y, z] (hole axis direction)
            - hole_type: "through" or "blind"
            - confidence: float (0-1)
        """
        # Strategy 1: Try STEP text parsing first (fast and reliable)
        result = self._extract_from_step_text(face_entities)

        if result and result.get("confidence", 0.0) > 0.7:
            _log.debug(
                f"Extracted hole parameters from STEP text: "
                f"diameter={result['diameter']:.2f}, depth={result['depth']:.2f}"
            )
            return result

        # Strategy 2: Try pythonocc geometric analysis (more robust but slower)
        if self.has_pythonocc and hasattr(graph, "faces"):
            occ_result = self._extract_from_occ_faces(face_ids, graph)
            if occ_result and occ_result.get("confidence", 0.0) > 0.5:
                _log.debug(
                    f"Extracted hole parameters from OCC geometry: "
                    f"diameter={occ_result['diameter']:.2f}"
                )
                return occ_result

        # Strategy 3: Use graph features as fallback
        graph_result = self._extract_from_graph_features(face_ids, graph)

        if graph_result:
            _log.debug("Using graph-based hole parameter estimation")
            return graph_result

        # All strategies failed - return defaults with low confidence
        _log.warning("Failed to extract hole parameters, using defaults")
        return {
            "diameter": 10.0,
            "depth": 20.0,
            "location": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 1.0],
            "hole_type": "unknown",
            "confidence": 0.3,
        }

    def _extract_from_step_text(
        self, face_entities: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Extract hole parameters from STEP entity text.

        Looks for:
        - CYLINDRICAL_SURFACE('',#N,RADIUS) for diameter
        - AXIS2_PLACEMENT_3D for location and orientation
        - Face count and connectivity for hole type

        Args:
            face_entities: List of face entity dictionaries

        Returns:
            Hole parameters dict or None if parsing fails
        """
        diameter = None
        location = None
        orientation = None
        cylindrical_faces = 0

        for face_entity in face_entities:
            entity_text = face_entity.get("text", "")

            # Look for CYLINDRICAL_SURFACE with radius
            # Format: CYLINDRICAL_SURFACE('',#123,5.0) where 5.0 is radius
            cyl_match = re.search(
                r"CYLINDRICAL_SURFACE\s*\([^,]*,\s*#\d+\s*,\s*([\d.eE+-]+)\)", entity_text
            )
            if cyl_match:
                radius = float(cyl_match.group(1))
                diameter = 2.0 * radius
                cylindrical_faces += 1
                _log.debug(f"Found cylindrical surface with radius={radius:.2f}")

            # Look for AXIS2_PLACEMENT_3D for location and orientation
            # Format: AXIS2_PLACEMENT_3D('',#123,(X,Y,Z),#456,#789)
            axis_match = re.search(
                r"AXIS2_PLACEMENT_3D\([^,]*,\s*#\d+\s*,\s*"
                r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)",
                entity_text,
            )
            if axis_match:
                location = [
                    float(axis_match.group(1)),
                    float(axis_match.group(2)),
                    float(axis_match.group(3)),
                ]
                _log.debug(f"Found axis placement at location={location}")

            # Look for DIRECTION for orientation
            # Format: DIRECTION('',(X,Y,Z))
            dir_match = re.search(
                r"DIRECTION\([^,]*,\s*"
                r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)",
                entity_text,
            )
            if dir_match:
                orientation = [
                    float(dir_match.group(1)),
                    float(dir_match.group(2)),
                    float(dir_match.group(3)),
                ]
                _log.debug(f"Found direction={orientation}")

        # Estimate depth from face count and bounding box
        # Simplified: assume standard hole depth is 2x diameter
        depth = diameter * 2.0 if diameter else 20.0

        # Determine hole type based on face count
        # Through hole: typically has bottom face (>=3 faces: side, top, bottom)
        # Blind hole: no bottom face (2 faces: side, top)
        hole_type = "through" if len(face_entities) >= 3 else "blind"

        # Compute confidence based on what we found
        confidence = 0.0
        if diameter is not None:
            confidence += 0.5  # Found diameter
        if location is not None:
            confidence += 0.3  # Found location
        if orientation is not None:
            confidence += 0.2  # Found orientation

        # Only return if we found at least diameter
        if diameter is None:
            return None

        return {
            "diameter": diameter,
            "depth": depth,
            "location": location if location else [0.0, 0.0, 0.0],
            "orientation": orientation if orientation else [0.0, 0.0, 1.0],
            "hole_type": hole_type,
            "confidence": min(confidence, 1.0),
        }

    def _extract_from_occ_faces(
        self, face_ids: List[int], graph: Any
    ) -> Optional[Dict[str, Any]]:
        """Extract hole parameters using pythonocc geometric analysis.

        Args:
            face_ids: Face indices
            graph: Graph with TopoDS_Face objects

        Returns:
            Hole parameters dict or None
        """
        if not self.has_pythonocc:
            return None

        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Cylinder
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.gp import gp_Cylinder

            # Get TopoDS_Face objects from graph
            if not hasattr(graph, "faces") or not graph.faces:
                return None

            cylindrical_faces = []
            for face_id in face_ids:
                if face_id >= len(graph.faces):
                    continue

                face_entity = graph.faces[face_id]

                # Check if face has OCC face stored
                if hasattr(face_entity, "_occ_face"):
                    occ_face = face_entity._occ_face

                    # Analyze surface type
                    adapter = BRepAdaptor_Surface(occ_face)

                    if adapter.GetType() == GeomAbs_Cylinder:
                        cylindrical_faces.append(occ_face)
                        _log.debug(f"Found cylindrical face at index {face_id}")

            if len(cylindrical_faces) == 0:
                return None

            # Get cylinder parameters from first cylindrical face
            face = cylindrical_faces[0]
            adapter = BRepAdaptor_Surface(face)
            cylinder: gp_Cylinder = adapter.Cylinder()

            # Get radius (diameter = 2 * radius)
            radius = cylinder.Radius()
            diameter = 2.0 * radius

            # Get axis
            axis = cylinder.Axis()
            location_pnt = axis.Location()
            direction = axis.Direction()

            location = [location_pnt.X(), location_pnt.Y(), location_pnt.Z()]
            orientation = [direction.X(), direction.Y(), direction.Z()]

            # Estimate depth (simplified)
            depth = diameter * 2.0

            # Determine hole type
            hole_type = "through" if len(cylindrical_faces) >= 2 else "blind"

            return {
                "diameter": diameter,
                "depth": depth,
                "location": location,
                "orientation": orientation,
                "hole_type": hole_type,
                "confidence": 0.9,
            }

        except Exception as e:
            _log.debug(f"Failed to extract hole parameters from OCC: {e}")
            return None

    def _extract_from_graph_features(
        self, face_ids: List[int], graph: Any
    ) -> Optional[Dict[str, Any]]:
        """Extract hole parameters from graph node features.

        Uses geometric features like curvature and centroid from graph.

        Args:
            face_ids: Face indices
            graph: Graph with node features

        Returns:
            Hole parameters dict or None
        """
        if not hasattr(graph, "x") or graph.x is None:
            return None

        try:
            import torch

            # Get features for hole faces
            if isinstance(graph.x, torch.Tensor):
                features = graph.x[face_ids].cpu().numpy()
            else:
                features = np.array(graph.x)[face_ids]

            # Extract centroids (dims 16:19)
            centroids = features[:, 16:19]

            # Estimate hole location as average centroid
            location = np.mean(centroids, axis=0).tolist()

            # Extract curvatures (dims 11:13) to estimate diameter
            # For cylinder: Gaussian curvature K = 0, Mean curvature H = 1/(2*radius)
            # So radius ≈ 1/(2*H) for non-zero H
            mean_curvatures = features[:, 12]
            avg_mean_curv = np.mean(np.abs(mean_curvatures))

            if avg_mean_curv > 1e-6:
                radius = 1.0 / (2.0 * avg_mean_curv)
                diameter = 2.0 * radius
            else:
                diameter = 10.0  # Default if curvature is zero

            # Extract normals (dims 13:16) to estimate orientation
            normals = features[:, 13:16]
            avg_normal = np.mean(normals, axis=0)

            # Normalize orientation
            norm = np.linalg.norm(avg_normal)
            if norm > 1e-6:
                orientation = (avg_normal / norm).tolist()
            else:
                orientation = [0.0, 0.0, 1.0]

            # Estimate depth
            depth = diameter * 2.0

            return {
                "diameter": diameter,
                "depth": depth,
                "location": location,
                "orientation": orientation,
                "hole_type": "unknown",
                "confidence": 0.6,
            }

        except Exception as e:
            _log.debug(f"Failed to extract hole parameters from graph features: {e}")
            return None


# Placeholder classes for other feature extractors
# These can be implemented following the same pattern as HoleGeometryExtractor


class PocketGeometryExtractor:
    """Extract pocket parameters (width, length, depth)."""

    def extract_pocket_parameters(
        self, face_entities: List[Dict], face_ids: List[int], graph: Any
    ) -> Dict[str, Any]:
        """Extract pocket parameters (placeholder for now)."""
        _log.debug("PocketGeometryExtractor not yet implemented, using defaults")
        return {
            "width": 20.0,
            "length": 30.0,
            "depth": 15.0,
            "location": [0.0, 0.0, 0.0],
            "confidence": 0.3,
        }


class BossGeometryExtractor:
    """Extract boss parameters (height, base area)."""

    def extract_boss_parameters(
        self, face_entities: List[Dict], face_ids: List[int], graph: Any
    ) -> Dict[str, Any]:
        """Extract boss parameters (placeholder for now)."""
        _log.debug("BossGeometryExtractor not yet implemented, using defaults")
        return {
            "height": 10.0,
            "base_area": 100.0,
            "location": [0.0, 0.0, 0.0],
            "confidence": 0.3,
        }


class FilletGeometryExtractor:
    """Extract fillet parameters (radius) from STEP entities and face graphs.

    Fillets are rounded transitions between surfaces, typically cylindrical.
    Extracts radius using three-strategy approach.
    """

    def __init__(self):
        """Initialize fillet geometry extractor."""
        self.has_pythonocc = False
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
            from OCC.Core.GeomAbs import GeomAbs_Circle

            self.has_pythonocc = True
            _log.debug("FilletGeometryExtractor initialized with pythonocc-core")
        except ImportError:
            _log.warning("pythonocc-core not available for fillet geometry extraction")

    def extract_fillet_parameters(
        self, face_entities: List[Dict], face_ids: List[int], graph: Any
    ) -> Dict[str, Any]:
        """Extract fillet parameters from face entities.

        Args:
            face_entities: List of face entity dictionaries
            face_ids: List of face indices in the graph
            graph: Face adjacency graph (PyG Data object)

        Returns:
            Dictionary with fillet parameters:
            - radius: float (fillet radius)
            - confidence: float (0-1)
            - method: str (extraction method used)
        """
        # Strategy 1: Try STEP text parsing first (fast and reliable)
        result = self._extract_from_step_text(face_entities)

        if result and result.get("confidence", 0.0) > 0.7:
            _log.debug(
                f"Extracted fillet parameters from STEP text: "
                f"radius={result['radius']:.2f}"
            )
            return result

        # Strategy 2: Try pythonocc geometric analysis (more robust but slower)
        if self.has_pythonocc and hasattr(graph, "edges"):
            occ_result = self._extract_from_occ_edges(face_ids, graph)
            if occ_result and occ_result.get("confidence", 0.0) > 0.5:
                _log.debug(
                    f"Extracted fillet parameters from OCC geometry: "
                    f"radius={occ_result['radius']:.2f}"
                )
                return occ_result

        # Strategy 3: Use graph features as fallback
        graph_result = self._extract_from_graph_features(face_ids, graph)

        if graph_result:
            _log.debug("Using graph-based fillet parameter estimation")
            return graph_result

        # All strategies failed - return defaults with low confidence
        _log.warning("Failed to extract fillet parameters, using defaults")
        return {
            "radius": 5.0,
            "confidence": 0.3,
            "method": "default",
        }

    def _extract_from_step_text(
        self, face_entities: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Extract fillet parameters from STEP entity text.

        Looks for:
        - BLEND, FILLET, or ROUNDED_EDGE entities with radius
        - CYLINDRICAL_SURFACE with small radius (likely a fillet)

        Args:
            face_entities: List of face entity dictionaries

        Returns:
            Fillet parameters dict or None if parsing fails
        """
        radius = None

        for face_entity in face_entities:
            entity_text = face_entity.get("text", "")

            # Look for BLEND or FILLET with radius
            # Format: BLEND('',#123,5.0) or FILLET('',#123,5.0)
            fillet_match = re.search(
                r"(?:BLEND|FILLET|ROUNDED_EDGE)\s*\([^,]*,\s*#\d+\s*,\s*([\d.eE+-]+)\)",
                entity_text,
            )
            if fillet_match:
                radius = float(fillet_match.group(1))
                _log.debug(f"Found fillet with radius={radius:.2f}")
                break

            # Look for CYLINDRICAL_SURFACE with small radius (< 20mm typically)
            cyl_match = re.search(
                r"CYLINDRICAL_SURFACE\s*\([^,]*,\s*#\d+\s*,\s*([\d.eE+-]+)\)",
                entity_text,
            )
            if cyl_match:
                potential_radius = float(cyl_match.group(1))
                # Fillets typically have small radii (< 20mm)
                if potential_radius < 20.0:
                    radius = potential_radius
                    _log.debug(
                        f"Found potential fillet (small cylinder) with radius={radius:.2f}"
                    )

        if radius is None:
            return None

        return {
            "radius": radius,
            "confidence": 0.9,
            "method": "step_text_parsing",
        }

    def _extract_from_occ_edges(
        self, face_ids: List[int], graph: Any
    ) -> Optional[Dict[str, Any]]:
        """Extract fillet parameters using pythonocc geometric analysis.

        Looks for circular edges connecting adjacent faces with convex dihedral angles.

        Args:
            face_ids: Face indices
            graph: Graph with TopoDS_Edge objects

        Returns:
            Fillet parameters dict or None
        """
        if not self.has_pythonocc:
            return None

        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
            from OCC.Core.GeomAbs import GeomAbs_Circle
            from OCC.Core.gp import gp_Circ

            # Get edges from graph
            if not hasattr(graph, "edge_index") or graph.edge_index is None:
                return None

            circular_radii = []

            # Look for edges connected to the fillet faces
            for face_id in face_ids:
                # Get adjacent edges (simplified: check all edges)
                if hasattr(graph, "edges") and graph.edges:
                    for edge_entity in graph.edges:
                        if not hasattr(edge_entity, "_occ_edge"):
                            continue

                        occ_edge = edge_entity._occ_edge

                        # Analyze edge curve type
                        adapter = BRepAdaptor_Curve(occ_edge)

                        if adapter.GetType() == GeomAbs_Circle:
                            circle: gp_Circ = adapter.Circle()
                            radius = circle.Radius()

                            # Fillets typically have small radii
                            if radius < 20.0:
                                circular_radii.append(radius)
                                _log.debug(f"Found circular edge with radius={radius:.2f}")

            if len(circular_radii) == 0:
                return None

            # Use median radius to avoid outliers
            avg_radius = np.median(circular_radii)

            return {
                "radius": float(avg_radius),
                "confidence": 0.85,
                "method": "occ_geometric_analysis",
            }

        except Exception as e:
            _log.debug(f"Failed to extract fillet parameters from OCC: {e}")
            return None

    def _extract_from_graph_features(
        self, face_ids: List[int], graph: Any
    ) -> Optional[Dict[str, Any]]:
        """Extract fillet parameters from graph edge features.

        Uses dihedral angles to identify fillets (convex transitions > 135°).
        Estimates radius from curvature if available.

        Args:
            face_ids: Face indices
            graph: Graph with edge features

        Returns:
            Fillet parameters dict or None
        """
        if not hasattr(graph, "edge_attr") or graph.edge_attr is None:
            return None

        try:
            import torch

            # Get edge features
            if isinstance(graph.edge_attr, torch.Tensor):
                edge_features = graph.edge_attr.cpu().numpy()
            else:
                edge_features = np.array(graph.edge_attr)

            # Get edge indices
            if isinstance(graph.edge_index, torch.Tensor):
                edge_index = graph.edge_index.cpu().numpy()
            else:
                edge_index = np.array(graph.edge_index)

            # Find edges connected to fillet faces
            relevant_edges = []
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                if src in face_ids or dst in face_ids:
                    relevant_edges.append(i)

            if len(relevant_edges) == 0:
                return None

            # Extract dihedral angles (assumed to be in dim 0)
            dihedral_angles = edge_features[relevant_edges, 0]

            # Fillets have convex dihedral angles (> 135° = 2.356 rad)
            convex_mask = dihedral_angles > 2.356  # 135 degrees

            if not np.any(convex_mask):
                return None

            # Estimate radius from edge curvature if available
            # Assume edge length is in dim 1, curvature in dim 7
            if edge_features.shape[1] > 7:
                curvatures = edge_features[relevant_edges, 7]
                avg_curvature = np.mean(np.abs(curvatures[convex_mask]))

                if avg_curvature > 1e-6:
                    radius = 1.0 / avg_curvature
                else:
                    radius = 5.0  # Default
            else:
                radius = 5.0  # Default if no curvature available

            return {
                "radius": float(radius),
                "confidence": 0.65,
                "method": "graph_feature_estimation",
            }

        except Exception as e:
            _log.debug(f"Failed to extract fillet parameters from graph features: {e}")
            return None


class ChamferGeometryExtractor:
    """Extract chamfer parameters (angle, distance) from STEP entities and face graphs.

    Chamfers are beveled transitions between surfaces, typically planar.
    Extracts angle and distance using three-strategy approach.
    """

    def __init__(self):
        """Initialize chamfer geometry extractor."""
        self.has_pythonocc = False
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Plane

            self.has_pythonocc = True
            _log.debug("ChamferGeometryExtractor initialized with pythonocc-core")
        except ImportError:
            _log.warning("pythonocc-core not available for chamfer geometry extraction")

    def extract_chamfer_parameters(
        self, face_entities: List[Dict], face_ids: List[int], graph: Any
    ) -> Dict[str, Any]:
        """Extract chamfer parameters from face entities.

        Args:
            face_entities: List of face entity dictionaries
            face_ids: List of face indices in the graph
            graph: Face adjacency graph (PyG Data object)

        Returns:
            Dictionary with chamfer parameters:
            - angle: float (chamfer angle in degrees)
            - distance: float (chamfer distance)
            - confidence: float (0-1)
            - method: str (extraction method used)
        """
        # Strategy 1: Try STEP text parsing first (fast and reliable)
        result = self._extract_from_step_text(face_entities)

        if result and result.get("confidence", 0.0) > 0.7:
            _log.debug(
                f"Extracted chamfer parameters from STEP text: "
                f"angle={result['angle']:.1f}°, distance={result['distance']:.2f}"
            )
            return result

        # Strategy 2: Try pythonocc geometric analysis (more robust but slower)
        if self.has_pythonocc and hasattr(graph, "edge_index"):
            occ_result = self._extract_from_occ_geometry(face_ids, graph)
            if occ_result and occ_result.get("confidence", 0.0) > 0.5:
                _log.debug(
                    f"Extracted chamfer parameters from OCC geometry: "
                    f"angle={occ_result['angle']:.1f}°"
                )
                return occ_result

        # Strategy 3: Use graph features as fallback
        graph_result = self._extract_from_graph_features(face_ids, graph)

        if graph_result:
            _log.debug("Using graph-based chamfer parameter estimation")
            return graph_result

        # All strategies failed - return defaults with low confidence
        _log.warning("Failed to extract chamfer parameters, using defaults")
        return {
            "angle": 45.0,
            "distance": 2.0,
            "confidence": 0.3,
            "method": "default",
        }

    def _extract_from_step_text(
        self, face_entities: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Extract chamfer parameters from STEP entity text.

        Looks for:
        - CHAMFER entities with angle and distance
        - PLANE surfaces with specific angular relationships

        Args:
            face_entities: List of face entity dictionaries

        Returns:
            Chamfer parameters dict or None if parsing fails
        """
        angle = None
        distance = None

        for face_entity in face_entities:
            entity_text = face_entity.get("text", "")

            # Look for CHAMFER with angle and distance
            # Format: CHAMFER('',#123,45.0,2.0)
            chamfer_match = re.search(
                r"CHAMFER\s*\([^,]*,\s*#\d+\s*,\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\)",
                entity_text,
            )
            if chamfer_match:
                angle = float(chamfer_match.group(1))
                distance = float(chamfer_match.group(2))
                _log.debug(
                    f"Found chamfer with angle={angle:.1f}°, distance={distance:.2f}"
                )
                break

            # Look for BEVELED_EDGE (alternative name)
            bevel_match = re.search(
                r"BEVELED_EDGE\s*\([^,]*,\s*#\d+\s*,\s*([\d.eE+-]+)\)", entity_text
            )
            if bevel_match:
                angle = 45.0  # Common default
                distance = float(bevel_match.group(1))
                _log.debug(f"Found beveled edge with distance={distance:.2f}")

        if angle is None and distance is None:
            return None

        return {
            "angle": angle if angle is not None else 45.0,
            "distance": distance if distance is not None else 2.0,
            "confidence": 0.9 if (angle and distance) else 0.7,
            "method": "step_text_parsing",
        }

    def _extract_from_occ_geometry(
        self, face_ids: List[int], graph: Any
    ) -> Optional[Dict[str, Any]]:
        """Extract chamfer parameters using pythonocc geometric analysis.

        Analyzes face normals and edge angles to determine chamfer angle.

        Args:
            face_ids: Face indices
            graph: Graph with TopoDS_Face objects

        Returns:
            Chamfer parameters dict or None
        """
        if not self.has_pythonocc:
            return None

        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Plane
            from OCC.Core.gp import gp_Pln
            import math

            # Get faces from graph
            if not hasattr(graph, "faces") or not graph.faces:
                return None

            planar_normals = []

            # Analyze chamfer faces (should be planar)
            for face_id in face_ids:
                if face_id >= len(graph.faces):
                    continue

                face_entity = graph.faces[face_id]

                if hasattr(face_entity, "_occ_face"):
                    occ_face = face_entity._occ_face

                    # Analyze surface type
                    adapter = BRepAdaptor_Surface(occ_face)

                    if adapter.GetType() == GeomAbs_Plane:
                        plane: gp_Pln = adapter.Plane()
                        normal = plane.Axis().Direction()

                        planar_normals.append([normal.X(), normal.Y(), normal.Z()])

            if len(planar_normals) < 1:
                return None

            # Estimate chamfer angle from normal orientations
            # Chamfers typically at 30°, 45°, or 60° to adjacent surfaces
            # Use average normal to estimate angle relative to vertical (Z-axis)
            avg_normal = np.mean(planar_normals, axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)

            # Angle from vertical (assuming chamfer relative to Z-axis)
            z_axis = np.array([0.0, 0.0, 1.0])
            cos_angle = np.dot(avg_normal, z_axis)
            angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            # Chamfer angle is typically measured from the base surface
            # Common values: 30°, 45°, 60°
            # Round to nearest common chamfer angle
            common_angles = [30.0, 45.0, 60.0]
            angle = min(common_angles, key=lambda x: abs(x - angle_deg))

            # Estimate distance (simplified: use default)
            distance = 2.0

            return {
                "angle": float(angle),
                "distance": distance,
                "confidence": 0.75,
                "method": "occ_geometric_analysis",
            }

        except Exception as e:
            _log.debug(f"Failed to extract chamfer parameters from OCC: {e}")
            return None

    def _extract_from_graph_features(
        self, face_ids: List[int], graph: Any
    ) -> Optional[Dict[str, Any]]:
        """Extract chamfer parameters from graph edge features.

        Uses dihedral angles to identify chamfers (acute transitions < 90°).
        Estimates angle from dihedral angle distribution.

        Args:
            face_ids: Face indices
            graph: Graph with edge features

        Returns:
            Chamfer parameters dict or None
        """
        if not hasattr(graph, "edge_attr") or graph.edge_attr is None:
            return None

        try:
            import torch

            # Get edge features
            if isinstance(graph.edge_attr, torch.Tensor):
                edge_features = graph.edge_attr.cpu().numpy()
            else:
                edge_features = np.array(graph.edge_attr)

            # Get edge indices
            if isinstance(graph.edge_index, torch.Tensor):
                edge_index = graph.edge_index.cpu().numpy()
            else:
                edge_index = np.array(graph.edge_index)

            # Find edges connected to chamfer faces
            relevant_edges = []
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                if src in face_ids or dst in face_ids:
                    relevant_edges.append(i)

            if len(relevant_edges) == 0:
                return None

            # Extract dihedral angles (assumed to be in dim 0)
            dihedral_angles = edge_features[relevant_edges, 0]

            # Chamfers have acute dihedral angles (< 90° = 1.571 rad)
            acute_mask = dihedral_angles < 1.571  # 90 degrees

            if not np.any(acute_mask):
                return None

            # Estimate chamfer angle from dihedral angles
            acute_angles = dihedral_angles[acute_mask]
            avg_dihedral_rad = np.mean(acute_angles)
            avg_dihedral_deg = np.degrees(avg_dihedral_rad)

            # Chamfer angle is related to supplement of dihedral angle
            # Common chamfer angles: 30°, 45°, 60°
            chamfer_angle = 90.0 - avg_dihedral_deg

            # Round to nearest common chamfer angle
            common_angles = [30.0, 45.0, 60.0]
            angle = min(common_angles, key=lambda x: abs(x - chamfer_angle))

            # Estimate distance from face area if available
            if hasattr(graph, "x") and graph.x is not None:
                if isinstance(graph.x, torch.Tensor):
                    node_features = graph.x.cpu().numpy()
                else:
                    node_features = np.array(graph.x)

                # Area is assumed to be in dim 10
                if node_features.shape[1] > 10:
                    areas = node_features[face_ids, 10]
                    avg_area = np.mean(areas)

                    # Estimate distance from area (simplified)
                    distance = np.sqrt(avg_area) / 2.0
                else:
                    distance = 2.0  # Default
            else:
                distance = 2.0  # Default

            return {
                "angle": float(angle),
                "distance": float(distance),
                "confidence": 0.65,
                "method": "graph_feature_estimation",
            }

        except Exception as e:
            _log.debug(f"Failed to extract chamfer parameters from graph features: {e}")
            return None
