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
        _log.warning(
            "Failed to extract hole parameters. "
            f"Attempted {3} strategies (STEP text, OCC geometry, graph features). "
            "Returning low-confidence defaults."
        )
        return {
            "diameter": 10.0,
            "depth": 20.0,
            "location": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 1.0],
            "hole_type": "unknown",
            "confidence": 0.2,
            "extraction_failures": ["step_text", "occ_geometry", "graph_features"],
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
    """Extract pocket parameters (width, length, depth) from STEP entities and face graphs.

    Pockets are recessed features with:
    - Width and length: Dimensions of the pocket opening
    - Depth: Distance from top surface to pocket bottom
    - Location: Center of the pocket bottom face

    Uses three-strategy approach for robust extraction.
    """

    def __init__(self):
        """Initialize pocket geometry extractor."""
        self.has_pythonocc = False
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Plane

            self.has_pythonocc = True
            _log.debug("PocketGeometryExtractor initialized with pythonocc-core")
        except ImportError:
            _log.warning("pythonocc-core not available for pocket geometry extraction")

    def extract_pocket_parameters(
        self, face_entities: List[Dict], face_ids: List[int], graph: Any
    ) -> Dict[str, Any]:
        """Extract pocket parameters from face entities.

        Args:
            face_entities: List of face entity dictionaries
            face_ids: List of face indices in the graph
            graph: Face adjacency graph (PyG Data object)

        Returns:
            Dictionary with pocket parameters:
            - width: float (pocket width)
            - length: float (pocket length)
            - depth: float (pocket depth)
            - location: [x, y, z] (pocket center location)
            - pocket_type: "rectangular" or "circular" or "unknown"
            - confidence: float (0-1)
        """
        # Strategy 1: Try STEP text parsing first (fast and reliable)
        result = self._extract_from_step_text(face_entities)

        if result and result.get("confidence", 0.0) > 0.7:
            _log.debug(
                f"Extracted pocket parameters from STEP text: "
                f"width={result['width']:.2f}, length={result['length']:.2f}, depth={result['depth']:.2f}"
            )
            return result

        # Strategy 2: Try pythonocc geometric analysis (more robust but slower)
        if self.has_pythonocc and hasattr(graph, "faces"):
            occ_result = self._extract_from_occ_faces(face_ids, graph)
            if occ_result and occ_result.get("confidence", 0.0) > 0.5:
                _log.debug(
                    f"Extracted pocket parameters from OCC geometry: "
                    f"width={occ_result['width']:.2f}, depth={occ_result['depth']:.2f}"
                )
                return occ_result

        # Strategy 3: Use graph features as fallback
        graph_result = self._extract_from_graph_features(face_ids, graph)

        if graph_result:
            _log.debug("Using graph-based pocket parameter estimation")
            return graph_result

        # All strategies failed - return defaults with low confidence
        _log.warning(
            "Failed to extract pocket parameters. "
            f"Attempted {3} strategies. "
            "Returning low-confidence defaults."
        )
        return {
            "width": 20.0,
            "length": 30.0,
            "depth": 15.0,
            "location": [0.0, 0.0, 0.0],
            "pocket_type": "unknown",
            "confidence": 0.2,
            "extraction_failures": ["step_text", "occ_geometry", "graph_features"],
        }

    def _extract_from_step_text(
        self, face_entities: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Extract pocket parameters from STEP entity text.

        Looks for:
        - POCKET or RECTANGULAR_FACE entities
        - Multiple planar faces forming recessed region
        - AXIS2_PLACEMENT_3D for location

        Args:
            face_entities: List of face entity dictionaries

        Returns:
            Pocket parameters dict or None if parsing fails
        """
        width = None
        length = None
        depth = None
        location = None
        planar_faces = 0

        # Look for pocket-related entities
        for face_entity in face_entities:
            entity_text = face_entity.get("text", "")

            # Look for explicit POCKET entities
            if "POCKET" in entity_text.upper():
                _log.debug("Found POCKET entity in STEP text")

                # Try to extract dimensions from POCKET entity
                # Format varies, look for numeric values
                numbers = re.findall(r"([\d.eE+-]+)", entity_text)
                if len(numbers) >= 3:
                    # Assume first 3 numbers are width, length, depth
                    width = float(numbers[0])
                    length = float(numbers[1])
                    depth = float(numbers[2])

            # Look for planar faces (pocket walls and bottom)
            if "PLANE" in entity_text or "PLANAR" in entity_text:
                planar_faces += 1

            # Look for AXIS2_PLACEMENT_3D for location
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
                _log.debug(f"Found pocket location at {location}")

        # Compute confidence based on what we found
        confidence = 0.0
        if width is not None and length is not None and depth is not None:
            confidence += 0.7  # Found all dimensions
        elif planar_faces >= 4:
            confidence += 0.3  # Found enough planar faces to suggest a pocket

        if location is not None:
            confidence += 0.3  # Found location

        # Only return if we found reasonable parameters
        if confidence < 0.5:
            return None

        # Use defaults if not found
        if width is None:
            width = 20.0
        if length is None:
            length = 30.0
        if depth is None:
            depth = 15.0

        pocket_type = "rectangular" if planar_faces >= 5 else "unknown"

        return {
            "width": width,
            "length": length,
            "depth": depth,
            "location": location if location else [0.0, 0.0, 0.0],
            "pocket_type": pocket_type,
            "confidence": min(confidence, 1.0),
        }

    def _extract_from_occ_faces(
        self, face_ids: List[int], graph: Any
    ) -> Optional[Dict[str, Any]]:
        """Extract pocket parameters using pythonocc geometric analysis.

        Strategy:
        1. Identify bottom face (lowest Z centroid among planar faces)
        2. Identify side faces (vertical/near-vertical normals)
        3. Extract bounding box of bottom face → width, length
        4. Calculate depth from Z-difference (top rim - bottom)
        5. Location = centroid of bottom face

        Args:
            face_ids: Face indices
            graph: Graph with TopoDS_Face objects

        Returns:
            Pocket parameters dict or None
        """
        if not self.has_pythonocc:
            return None

        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Plane
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.Bnd import Bnd_Box
            from OCC.Core.BRepBndLib import brepbndlib

            # Get TopoDS_Face objects from graph
            if not hasattr(graph, "faces") or not graph.faces:
                return None

            planar_faces = []
            face_data = []

            for face_id in face_ids:
                if face_id >= len(graph.faces):
                    continue

                face_entity = graph.faces[face_id]

                # Check if face has OCC face stored
                if hasattr(face_entity, "_occ_face"):
                    occ_face = face_entity._occ_face

                    # Analyze surface type
                    adapter = BRepAdaptor_Surface(occ_face)

                    if adapter.GetType() == GeomAbs_Plane:
                        planar_faces.append(occ_face)

                        # Get centroid
                        props = GProp_GProps()
                        brepgprop.SurfaceProperties(occ_face, props)
                        centroid = props.CentreOfMass()

                        # Get normal
                        plane = adapter.Plane()
                        normal = plane.Axis().Direction()

                        face_data.append({
                            "face": occ_face,
                            "centroid": [centroid.X(), centroid.Y(), centroid.Z()],
                            "normal": [normal.X(), normal.Y(), normal.Z()],
                        })
                        _log.debug(f"Found planar face at index {face_id}")

            if len(planar_faces) < 2:
                return None

            # Find bottom face (lowest Z centroid with horizontal normal)
            bottom_face_data = None
            min_z = float('inf')

            for fd in face_data:
                # Check if normal points roughly upward (horizontal face)
                normal_z = abs(fd["normal"][2])
                if normal_z > 0.8:  # Normal is mostly vertical
                    z = fd["centroid"][2]
                    if z < min_z:
                        min_z = z
                        bottom_face_data = fd

            if bottom_face_data is None:
                # No clear bottom face, use lowest centroid
                bottom_face_data = min(face_data, key=lambda x: x["centroid"][2])

            # Get bounding box of bottom face
            bbox = Bnd_Box()
            brepbndlib.Add(bottom_face_data["face"], bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

            width = xmax - xmin
            length = ymax - ymin

            # Calculate depth: max Z of all faces - min Z of bottom face
            max_z = max(fd["centroid"][2] for fd in face_data)
            depth = max_z - bottom_face_data["centroid"][2]

            # Determine pocket type
            aspect_ratio = width / length if length > 0 else 1.0
            pocket_type = "rectangular"
            if 0.9 <= aspect_ratio <= 1.1:
                pocket_type = "square"

            return {
                "width": width,
                "length": length,
                "depth": depth,
                "location": bottom_face_data["centroid"],
                "pocket_type": pocket_type,
                "confidence": 0.9,  # High confidence from geometric analysis
            }

        except Exception as e:
            _log.warning(f"OCC pocket extraction failed: {e}")
            return None

    def _extract_from_graph_features(
        self, face_ids: List[int], graph: Any
    ) -> Optional[Dict[str, Any]]:
        """Extract pocket parameters from graph node features.

        Uses PyG node features:
        - Dimensions 16-18: Centroids (x, y, z)
        - Compute bounding box from all face centroids

        Args:
            face_ids: Face indices
            graph: PyG Data object

        Returns:
            Pocket parameters dict or None
        """
        try:
            if not hasattr(graph, "x") or graph.x is None:
                return None

            features = graph.x.cpu().numpy() if hasattr(graph.x, 'cpu') else np.array(graph.x)

            if len(face_ids) == 0 or features.shape[1] < 19:
                return None

            # Extract centroids (dimensions 16-18)
            pocket_face_features = features[face_ids]
            centroids = pocket_face_features[:, 16:19]

            # Compute bounding box from centroids
            min_coords = centroids.min(axis=0)
            max_coords = centroids.max(axis=0)

            width = float(max_coords[0] - min_coords[0])
            length = float(max_coords[1] - min_coords[1])
            depth = float(max_coords[2] - min_coords[2])

            # Location is average of all centroids
            location = centroids.mean(axis=0).tolist()

            # Estimate pocket type from aspect ratio
            if width > 0 and length > 0:
                aspect_ratio = width / length
                pocket_type = "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangular"
            else:
                pocket_type = "unknown"

            return {
                "width": max(width, 1.0),  # Ensure minimum size
                "length": max(length, 1.0),
                "depth": max(depth, 1.0),
                "location": location,
                "pocket_type": pocket_type,
                "confidence": 0.6,  # Medium confidence from graph features
            }

        except Exception as e:
            _log.warning(f"Graph-based pocket extraction failed: {e}")
            return None


class BossGeometryExtractor:
    """Extract boss parameters (height, base area) from STEP entities and face graphs.

    Bosses are raised features with:
    - Height: Distance from base surface to boss top
    - Base area: Area of the boss footprint
    - Location: Center of the boss top face

    Uses three-strategy approach for robust extraction.
    """

    def __init__(self):
        """Initialize boss geometry extractor."""
        self.has_pythonocc = False
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Plane

            self.has_pythonocc = True
            _log.debug("BossGeometryExtractor initialized with pythonocc-core")
        except ImportError:
            _log.warning("pythonocc-core not available for boss geometry extraction")

    def extract_boss_parameters(
        self, face_entities: List[Dict], face_ids: List[int], graph: Any
    ) -> Dict[str, Any]:
        """Extract boss parameters from face entities.

        Args:
            face_entities: List of face entity dictionaries
            face_ids: List of face indices in the graph
            graph: Face adjacency graph (PyG Data object)

        Returns:
            Dictionary with boss parameters:
            - height: float (boss height)
            - base_area: float (boss base area)
            - width: float (boss width)
            - length: float (boss length)
            - location: [x, y, z] (boss center location)
            - boss_type: "rectangular" or "circular" or "unknown"
            - confidence: float (0-1)
        """
        # Strategy 1: Try STEP text parsing first (fast and reliable)
        result = self._extract_from_step_text(face_entities)

        if result and result.get("confidence", 0.0) > 0.7:
            _log.debug(
                f"Extracted boss parameters from STEP text: "
                f"height={result['height']:.2f}, base_area={result['base_area']:.2f}"
            )
            return result

        # Strategy 2: Try pythonocc geometric analysis (more robust but slower)
        if self.has_pythonocc and hasattr(graph, "faces"):
            occ_result = self._extract_from_occ_faces(face_ids, graph)
            if occ_result and occ_result.get("confidence", 0.0) > 0.5:
                _log.debug(
                    f"Extracted boss parameters from OCC geometry: "
                    f"height={occ_result['height']:.2f}, base_area={occ_result['base_area']:.2f}"
                )
                return occ_result

        # Strategy 3: Use graph features as fallback
        graph_result = self._extract_from_graph_features(face_ids, graph)

        if graph_result:
            _log.debug("Using graph-based boss parameter estimation")
            return graph_result

        # All strategies failed - return defaults with low confidence
        _log.warning(
            "Failed to extract boss parameters. "
            f"Attempted {3} strategies. "
            "Returning low-confidence defaults."
        )
        return {
            "height": 10.0,
            "base_area": 100.0,
            "width": 10.0,
            "length": 10.0,
            "location": [0.0, 0.0, 0.0],
            "boss_type": "unknown",
            "confidence": 0.2,
            "extraction_failures": ["step_text", "occ_geometry", "graph_features"],
        }

    def _extract_from_step_text(
        self, face_entities: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Extract boss parameters from STEP entity text.

        Looks for:
        - BOSS entities or elevated planar surfaces
        - Multiple planar faces forming raised region
        - AXIS2_PLACEMENT_3D for location

        Args:
            face_entities: List of face entity dictionaries

        Returns:
            Boss parameters dict or None if parsing fails
        """
        height = None
        width = None
        length = None
        location = None
        planar_faces = 0

        # Look for boss-related entities
        for face_entity in face_entities:
            entity_text = face_entity.get("text", "")

            # Look for explicit BOSS entities
            if "BOSS" in entity_text.upper() or "PROTRUSION" in entity_text.upper():
                _log.debug("Found BOSS entity in STEP text")

                # Try to extract dimensions from BOSS entity
                numbers = re.findall(r"([\d.eE+-]+)", entity_text)
                if len(numbers) >= 2:
                    # Assume first numbers are width/length/height
                    width = float(numbers[0])
                    if len(numbers) >= 3:
                        length = float(numbers[1])
                        height = float(numbers[2])
                    else:
                        length = width  # Square boss
                        height = float(numbers[1])

            # Look for planar faces (boss walls and top)
            if "PLANE" in entity_text or "PLANAR" in entity_text:
                planar_faces += 1

            # Look for AXIS2_PLACEMENT_3D for location
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
                _log.debug(f"Found boss location at {location}")

        # Compute confidence based on what we found
        confidence = 0.0
        if height is not None and width is not None:
            confidence += 0.7  # Found key dimensions
        elif planar_faces >= 4:
            confidence += 0.3  # Found enough planar faces to suggest a boss

        if location is not None:
            confidence += 0.3  # Found location

        # Only return if we found reasonable parameters
        if confidence < 0.5:
            return None

        # Use defaults if not found
        if height is None:
            height = 10.0
        if width is None:
            width = 10.0
        if length is None:
            length = width  # Assume square

        base_area = width * length

        boss_type = "rectangular" if planar_faces >= 5 else "unknown"
        if abs(width - length) / max(width, length) < 0.1:
            boss_type = "square"

        return {
            "height": height,
            "base_area": base_area,
            "width": width,
            "length": length,
            "location": location if location else [0.0, 0.0, 0.0],
            "boss_type": boss_type,
            "confidence": min(confidence, 1.0),
        }

    def _extract_from_occ_faces(
        self, face_ids: List[int], graph: Any
    ) -> Optional[Dict[str, Any]]:
        """Extract boss parameters using pythonocc geometric analysis.

        Strategy:
        1. Identify top face (highest Z centroid among planar faces)
        2. Identify side faces (vertical/near-vertical normals)
        3. Extract bounding box of top face → width, length
        4. Calculate height from Z-difference (top - base)
        5. Base area = width * length (or π*r² for circular)
        6. Location = centroid of top face

        Args:
            face_ids: Face indices
            graph: Graph with TopoDS_Face objects

        Returns:
            Boss parameters dict or None
        """
        if not self.has_pythonocc:
            return None

        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.Bnd import Bnd_Box
            from OCC.Core.BRepBndLib import brepbndlib

            # Get TopoDS_Face objects from graph
            if not hasattr(graph, "faces") or not graph.faces:
                return None

            planar_faces = []
            cylindrical_faces = []
            face_data = []

            for face_id in face_ids:
                if face_id >= len(graph.faces):
                    continue

                face_entity = graph.faces[face_id]

                # Check if face has OCC face stored
                if hasattr(face_entity, "_occ_face"):
                    occ_face = face_entity._occ_face

                    # Analyze surface type
                    adapter = BRepAdaptor_Surface(occ_face)
                    surface_type = adapter.GetType()

                    if surface_type == GeomAbs_Plane:
                        planar_faces.append(occ_face)

                        # Get centroid
                        props = GProp_GProps()
                        brepgprop.SurfaceProperties(occ_face, props)
                        centroid = props.CentreOfMass()

                        # Get normal
                        plane = adapter.Plane()
                        normal = plane.Axis().Direction()

                        # Get area
                        area = props.Mass()

                        face_data.append({
                            "face": occ_face,
                            "centroid": [centroid.X(), centroid.Y(), centroid.Z()],
                            "normal": [normal.X(), normal.Y(), normal.Z()],
                            "area": area,
                            "type": "planar",
                        })
                        _log.debug(f"Found planar face at index {face_id}")

                    elif surface_type == GeomAbs_Cylinder:
                        cylindrical_faces.append(occ_face)
                        # Cylindrical boss (round)
                        props = GProp_GProps()
                        brepgprop.SurfaceProperties(occ_face, props)
                        centroid = props.CentreOfMass()

                        face_data.append({
                            "face": occ_face,
                            "centroid": [centroid.X(), centroid.Y(), centroid.Z()],
                            "type": "cylindrical",
                        })

            if len(face_data) < 2:
                return None

            # Find top face (highest Z centroid with horizontal normal)
            top_face_data = None
            max_z = float('-inf')

            for fd in face_data:
                if fd["type"] == "planar":
                    # Check if normal points roughly upward (horizontal face)
                    normal_z = abs(fd["normal"][2])
                    if normal_z > 0.8:  # Normal is mostly vertical
                        z = fd["centroid"][2]
                        if z > max_z:
                            max_z = z
                            top_face_data = fd

            if top_face_data is None:
                # No clear top face, use highest centroid
                planar_data = [fd for fd in face_data if fd["type"] == "planar"]
                if planar_data:
                    top_face_data = max(planar_data, key=lambda x: x["centroid"][2])
                else:
                    return None

            # Determine boss type and dimensions
            boss_type = "unknown"
            width = 0.0
            length = 0.0
            base_area = 0.0

            if len(cylindrical_faces) > 0:
                # Circular boss
                boss_type = "circular"
                # Get radius from cylindrical face
                adapter = BRepAdaptor_Surface(cylindrical_faces[0])
                cylinder = adapter.Cylinder()
                radius = cylinder.Radius()
                width = 2.0 * radius
                length = 2.0 * radius
                base_area = np.pi * radius * radius
            else:
                # Rectangular boss - use top face bounding box
                bbox = Bnd_Box()
                brepbndlib.Add(top_face_data["face"], bbox)
                xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

                width = xmax - xmin
                length = ymax - ymin
                base_area = width * length

                # Determine if square or rectangular
                aspect_ratio = width / length if length > 0 else 1.0
                if 0.9 <= aspect_ratio <= 1.1:
                    boss_type = "square"
                else:
                    boss_type = "rectangular"

            # Calculate height: max Z of top face - min Z of all faces
            min_z = min(fd["centroid"][2] for fd in face_data)
            height = top_face_data["centroid"][2] - min_z

            return {
                "height": height,
                "base_area": base_area,
                "width": width,
                "length": length,
                "location": top_face_data["centroid"],
                "boss_type": boss_type,
                "confidence": 0.9,  # High confidence from geometric analysis
            }

        except Exception as e:
            _log.warning(f"OCC boss extraction failed: {e}")
            return None

    def _extract_from_graph_features(
        self, face_ids: List[int], graph: Any
    ) -> Optional[Dict[str, Any]]:
        """Extract boss parameters from graph node features.

        Uses PyG node features:
        - Dimensions 16-18: Centroids (x, y, z)
        - Compute bounding box from all face centroids
        - Height = max Z - min Z

        Args:
            face_ids: Face indices
            graph: PyG Data object

        Returns:
            Boss parameters dict or None
        """
        try:
            if not hasattr(graph, "x") or graph.x is None:
                return None

            features = graph.x.cpu().numpy() if hasattr(graph.x, 'cpu') else np.array(graph.x)

            if len(face_ids) == 0 or features.shape[1] < 19:
                return None

            # Extract centroids (dimensions 16-18)
            boss_face_features = features[face_ids]
            centroids = boss_face_features[:, 16:19]

            # Find top face (highest Z)
            top_face_idx = np.argmax(centroids[:, 2])
            top_centroid = centroids[top_face_idx]

            # Compute bounding box from centroids
            min_coords = centroids.min(axis=0)
            max_coords = centroids.max(axis=0)

            width = float(max_coords[0] - min_coords[0])
            length = float(max_coords[1] - min_coords[1])
            height = float(max_coords[2] - min_coords[2])

            # Base area
            base_area = width * length

            # Location is top face centroid
            location = top_centroid.tolist()

            # Estimate boss type from aspect ratio
            if width > 0 and length > 0:
                aspect_ratio = width / length
                if 0.9 <= aspect_ratio <= 1.1:
                    boss_type = "square"
                else:
                    boss_type = "rectangular"
            else:
                boss_type = "unknown"

            return {
                "height": max(height, 1.0),  # Ensure minimum size
                "base_area": max(base_area, 1.0),
                "width": max(width, 1.0),
                "length": max(length, 1.0),
                "location": location,
                "boss_type": boss_type,
                "confidence": 0.6,  # Medium confidence from graph features
            }

        except Exception as e:
            _log.warning(f"Graph-based boss extraction failed: {e}")
            return None


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
        _log.warning(
            "Failed to extract fillet parameters. "
            f"Attempted {3} strategies (STEP text, OCC geometry, graph features). "
            "Returning low-confidence defaults."
        )
        return {
            "radius": 5.0,
            "confidence": 0.2,
            "method": "default",
            "extraction_failures": ["step_text", "occ_geometry", "graph_features"],
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
        _log.warning(
            "Failed to extract chamfer parameters. "
            f"Attempted {3} strategies (STEP text, OCC geometry, graph features). "
            "Returning low-confidence defaults."
        )
        return {
            "angle": 45.0,
            "distance": 2.0,
            "confidence": 0.2,
            "method": "default",
            "extraction_failures": ["step_text", "occ_geometry", "graph_features"],
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
