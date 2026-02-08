"""
STEP Feature Extractor - Extracts geometric and topological features from STEP entities.

Implements feature extraction from scratch without external dependencies.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)


class STEPFeatureExtractor:
    """Extracts features from parsed STEP entities for ML/AI processing."""

    # Entity type categories for feature extraction
    POINT_ENTITIES = {
        "CARTESIAN_POINT",
        "VERTEX_POINT",
        "POINT_ON_CURVE",
        "POINT_ON_SURFACE",
    }

    CURVE_ENTITIES = {
        "LINE",
        "CIRCLE",
        "ELLIPSE",
        "PARABOLA",
        "HYPERBOLA",
        "B_SPLINE_CURVE",
        "BEZIER_CURVE",
        "POLYLINE",
        "TRIMMED_CURVE",
    }

    SURFACE_ENTITIES = {
        "PLANE",
        "CYLINDRICAL_SURFACE",
        "CONICAL_SURFACE",
        "SPHERICAL_SURFACE",
        "TOROIDAL_SURFACE",
        "B_SPLINE_SURFACE",
        "SURFACE_OF_REVOLUTION",
        "SURFACE_OF_LINEAR_EXTRUSION",
    }

    TOPOLOGY_ENTITIES = {
        "VERTEX_POINT",
        "EDGE_CURVE",
        "ORIENTED_EDGE",
        "EDGE_LOOP",
        "FACE_BOUND",
        "FACE_SURFACE",
        "CLOSED_SHELL",
        "OPEN_SHELL",
        "MANIFOLD_SOLID_BREP",
    }

    SHAPE_ENTITIES = {
        "ADVANCED_BREP_SHAPE_REPRESENTATION",
        "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
        "GEOMETRICALLY_BOUNDED_SURFACE_SHAPE_REPRESENTATION",
        "GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION",
    }

    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_cache: Dict[int, Dict[str, Any]] = {}

    def extract_features(
        self, entities: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Extract features from all entities.

        Args:
            entities: Dictionary mapping entity_id to parsed entity data

        Returns:
            Dictionary mapping entity_id to extracted features
        """
        features = {}

        for entity_id, entity_data in entities.items():
            entity_type = entity_data.get("type", "")
            features[entity_id] = self.extract_entity_features(
                entity_id, entity_type, entity_data, entities
            )

        return features

    def extract_entity_features(
        self,
        entity_id: int,
        entity_type: str,
        entity_data: Dict[str, Any],
        all_entities: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Extract features from a single entity.

        Args:
            entity_id: The entity ID
            entity_type: The entity type name
            entity_data: The parsed entity data
            all_entities: All entities for reference resolution

        Returns:
            Dictionary of extracted features
        """
        # Check cache first
        if entity_id in self.feature_cache:
            return self.feature_cache[entity_id]

        features = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "category": self._categorize_entity(entity_type),
        }

        # Extract type-specific features
        if entity_type in self.POINT_ENTITIES:
            features.update(self._extract_point_features(entity_data, all_entities))
        elif entity_type in self.CURVE_ENTITIES:
            features.update(self._extract_curve_features(entity_data, all_entities))
        elif entity_type in self.SURFACE_ENTITIES:
            features.update(self._extract_surface_features(entity_data, all_entities))
        elif entity_type in self.TOPOLOGY_ENTITIES:
            features.update(
                self._extract_topology_features(entity_data, all_entities)
            )

        # Extract generic numeric features
        features.update(self._extract_numeric_features(entity_data))

        # Extract reference features
        features.update(self._extract_reference_features(entity_data))

        # Cache the features
        self.feature_cache[entity_id] = features

        return features

    def _categorize_entity(self, entity_type: str) -> str:
        """Categorize entity type."""
        if entity_type in self.POINT_ENTITIES:
            return "point"
        elif entity_type in self.CURVE_ENTITIES:
            return "curve"
        elif entity_type in self.SURFACE_ENTITIES:
            return "surface"
        elif entity_type in self.TOPOLOGY_ENTITIES:
            return "topology"
        elif entity_type in self.SHAPE_ENTITIES:
            return "shape"
        else:
            return "other"

    def _extract_point_features(
        self, entity_data: Dict[str, Any], all_entities: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract features from point entities."""
        features = {}
        params = entity_data.get("params", [])

        # CARTESIAN_POINT has coordinates in first parameter (list of floats)
        if entity_data.get("type") == "CARTESIAN_POINT" and params:
            coords = self._extract_coordinates(params[0])
            if coords:
                features["coordinates"] = coords
                features["dimension"] = len(coords)
                features["x"] = coords[0] if len(coords) > 0 else 0.0
                features["y"] = coords[1] if len(coords) > 1 else 0.0
                features["z"] = coords[2] if len(coords) > 2 else 0.0
                features["distance_from_origin"] = math.sqrt(
                    sum(c**2 for c in coords)
                )

        # VERTEX_POINT references a CARTESIAN_POINT
        elif entity_data.get("type") == "VERTEX_POINT" and params:
            ref_id = self._extract_reference(params[0])
            if ref_id and ref_id in all_entities:
                ref_entity = all_entities[ref_id]
                ref_features = self._extract_point_features(ref_entity, all_entities)
                features.update(ref_features)
                features["point_reference"] = ref_id

        return features

    def _extract_curve_features(
        self, entity_data: Dict[str, Any], all_entities: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract features from curve entities."""
        features = {}
        entity_type = entity_data.get("type", "")
        params = entity_data.get("params", [])

        if entity_type == "LINE" and len(params) >= 2:
            # LINE(point, direction)
            point_ref = self._extract_reference(params[0])
            direction_ref = self._extract_reference(params[1])

            if point_ref and point_ref in all_entities:
                point_features = self._extract_point_features(
                    all_entities[point_ref], all_entities
                )
                features["start_point"] = point_features.get("coordinates")

            if direction_ref and direction_ref in all_entities:
                dir_entity = all_entities[direction_ref]
                if dir_entity.get("type") == "VECTOR":
                    dir_params = dir_entity.get("params", [])
                    if len(dir_params) >= 2:
                        direction = self._extract_coordinates(dir_params[0])
                        magnitude = self._extract_float(dir_params[1])
                        features["direction"] = direction
                        features["magnitude"] = magnitude

            features["curve_type"] = "line"
            features["is_linear"] = True

        elif entity_type == "CIRCLE" and len(params) >= 2:
            # CIRCLE(axis_placement, radius)
            axis_ref = self._extract_reference(params[0])
            radius = self._extract_float(params[1])

            features["radius"] = radius
            features["curve_type"] = "circle"
            features["is_closed"] = True
            features["is_linear"] = False

            if axis_ref and axis_ref in all_entities:
                axis_features = self._extract_axis_placement_features(
                    all_entities[axis_ref], all_entities
                )
                features.update(axis_features)

        elif entity_type == "ELLIPSE" and len(params) >= 3:
            # ELLIPSE(axis_placement, semi_axis1, semi_axis2)
            axis_ref = self._extract_reference(params[0])
            semi_axis1 = self._extract_float(params[1])
            semi_axis2 = self._extract_float(params[2])

            features["semi_axis1"] = semi_axis1
            features["semi_axis2"] = semi_axis2
            features["curve_type"] = "ellipse"
            features["is_closed"] = True
            features["is_linear"] = False
            features["eccentricity"] = self._compute_eccentricity(semi_axis1, semi_axis2)

        elif entity_type == "B_SPLINE_CURVE":
            features["curve_type"] = "b_spline"
            features["is_linear"] = False
            # Extract degree, control points, etc.
            if params:
                if len(params) > 0:
                    features["degree"] = self._extract_int(params[0])
                if len(params) > 1:
                    control_points = self._extract_list_of_references(params[1])
                    features["num_control_points"] = len(control_points)

        return features

    def _extract_surface_features(
        self, entity_data: Dict[str, Any], all_entities: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract features from surface entities."""
        features = {}
        entity_type = entity_data.get("type", "")
        params = entity_data.get("params", [])

        if entity_type == "PLANE" and params:
            # PLANE(axis_placement)
            axis_ref = self._extract_reference(params[0])
            features["surface_type"] = "plane"
            features["is_planar"] = True

            if axis_ref and axis_ref in all_entities:
                axis_features = self._extract_axis_placement_features(
                    all_entities[axis_ref], all_entities
                )
                features.update(axis_features)

        elif entity_type == "CYLINDRICAL_SURFACE" and len(params) >= 2:
            # CYLINDRICAL_SURFACE(axis_placement, radius)
            axis_ref = self._extract_reference(params[0])
            radius = self._extract_float(params[1])

            features["surface_type"] = "cylindrical"
            features["is_planar"] = False
            features["radius"] = radius

        elif entity_type == "SPHERICAL_SURFACE" and len(params) >= 2:
            # SPHERICAL_SURFACE(axis_placement, radius)
            axis_ref = self._extract_reference(params[0])
            radius = self._extract_float(params[1])

            features["surface_type"] = "spherical"
            features["is_planar"] = False
            features["radius"] = radius
            features["is_closed"] = True

        elif entity_type == "CONICAL_SURFACE" and len(params) >= 3:
            # CONICAL_SURFACE(axis_placement, radius, semi_angle)
            axis_ref = self._extract_reference(params[0])
            radius = self._extract_float(params[1])
            semi_angle = self._extract_float(params[2])

            features["surface_type"] = "conical"
            features["is_planar"] = False
            features["radius"] = radius
            features["semi_angle"] = semi_angle

        elif entity_type == "TOROIDAL_SURFACE" and len(params) >= 3:
            # TOROIDAL_SURFACE(axis_placement, major_radius, minor_radius)
            axis_ref = self._extract_reference(params[0])
            major_radius = self._extract_float(params[1])
            minor_radius = self._extract_float(params[2])

            features["surface_type"] = "toroidal"
            features["is_planar"] = False
            features["major_radius"] = major_radius
            features["minor_radius"] = minor_radius

        elif entity_type == "B_SPLINE_SURFACE":
            features["surface_type"] = "b_spline"
            features["is_planar"] = False
            # Extract degrees, control points, etc.

        return features

    def _extract_topology_features(
        self, entity_data: Dict[str, Any], all_entities: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract features from topology entities."""
        features = {}
        entity_type = entity_data.get("type", "")
        params = entity_data.get("params", [])

        if entity_type == "EDGE_CURVE" and len(params) >= 3:
            # EDGE_CURVE(start_vertex, end_vertex, curve, same_sense)
            start_ref = self._extract_reference(params[0])
            end_ref = self._extract_reference(params[1])
            curve_ref = self._extract_reference(params[2])

            features["topology_type"] = "edge"
            features["start_vertex_ref"] = start_ref
            features["end_vertex_ref"] = end_ref
            features["curve_ref"] = curve_ref
            features["connectivity_degree"] = 2

        elif entity_type == "FACE_SURFACE" and len(params) >= 2:
            # FACE_SURFACE(bounds, surface, same_sense)
            bounds_refs = self._extract_list_of_references(params[0])
            surface_ref = self._extract_reference(params[1])

            features["topology_type"] = "face"
            features["num_bounds"] = len(bounds_refs)
            features["surface_ref"] = surface_ref
            features["bound_refs"] = bounds_refs

        elif entity_type == "CLOSED_SHELL" or entity_type == "OPEN_SHELL":
            # SHELL(list_of_faces)
            face_refs = self._extract_list_of_references(params[0]) if params else []

            features["topology_type"] = "shell"
            features["is_closed"] = entity_type == "CLOSED_SHELL"
            features["num_faces"] = len(face_refs)
            features["face_refs"] = face_refs

        elif entity_type == "MANIFOLD_SOLID_BREP" and params:
            # MANIFOLD_SOLID_BREP(name, shell)
            shell_ref = self._extract_reference(params[1]) if len(params) > 1 else None

            features["topology_type"] = "solid"
            features["is_manifold"] = True
            features["shell_ref"] = shell_ref

        elif entity_type == "ORIENTED_EDGE" and len(params) >= 4:
            # ORIENTED_EDGE(name, start, end, edge, orientation)
            edge_ref = self._extract_reference(params[3])
            orientation = self._extract_boolean(params[4]) if len(params) > 4 else True

            features["topology_type"] = "oriented_edge"
            features["edge_ref"] = edge_ref
            features["orientation"] = orientation

        elif entity_type == "EDGE_LOOP" and params:
            # EDGE_LOOP(name, list_of_oriented_edges)
            edge_refs = self._extract_list_of_references(params[1]) if len(params) > 1 else []

            features["topology_type"] = "loop"
            features["is_closed"] = True
            features["num_edges"] = len(edge_refs)
            features["edge_refs"] = edge_refs

        return features

    def _extract_axis_placement_features(
        self, entity_data: Dict[str, Any], all_entities: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract features from AXIS2_PLACEMENT entities."""
        features = {}
        entity_type = entity_data.get("type", "")
        params = entity_data.get("params", [])

        if entity_type == "AXIS2_PLACEMENT_3D" and len(params) >= 1:
            # AXIS2_PLACEMENT_3D(name, location, axis, ref_direction)
            location_ref = self._extract_reference(params[1]) if len(params) > 1 else None

            if location_ref and location_ref in all_entities:
                location_features = self._extract_point_features(
                    all_entities[location_ref], all_entities
                )
                features["center"] = location_features.get("coordinates")

            if len(params) > 2:
                axis_ref = self._extract_reference(params[2])
                if axis_ref and axis_ref in all_entities:
                    # Extract axis direction - DIRECTION entities have coordinates like CARTESIAN_POINT
                    direction_entity = all_entities[axis_ref]
                    direction_features = self._extract_point_features(direction_entity, all_entities)
                    features["axis_direction"] = direction_features.get("coordinates")

            # Extract reference direction if available
            if len(params) > 3:
                ref_dir_ref = self._extract_reference(params[3])
                if ref_dir_ref and ref_dir_ref in all_entities:
                    ref_features = self._extract_point_features(all_entities[ref_dir_ref], all_entities)
                    features["ref_direction"] = ref_features.get("coordinates")

        return features

    def _extract_numeric_features(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all numeric values from entity parameters."""
        features = {}
        params = entity_data.get("params", [])

        # Count numeric parameters
        numeric_values = []
        for param in params:
            if isinstance(param, (int, float)):
                numeric_values.append(float(param))
            elif isinstance(param, str):
                # Try to extract numeric value from string
                num = self._extract_float(param)
                if num is not None:
                    numeric_values.append(num)

        if numeric_values:
            features["numeric_params"] = numeric_values
            features["num_numeric_params"] = len(numeric_values)
            features["numeric_sum"] = sum(numeric_values)
            features["numeric_mean"] = sum(numeric_values) / len(numeric_values)
            features["numeric_min"] = min(numeric_values)
            features["numeric_max"] = max(numeric_values)

        return features

    def _extract_reference_features(
        self, entity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract reference count and connectivity features."""
        features = {}
        params = entity_data.get("params", [])

        # Count references to other entities
        ref_count = 0
        refs = []
        for param in params:
            if isinstance(param, str) and param.startswith("#"):
                ref_id = self._extract_reference(param)
                if ref_id:
                    refs.append(ref_id)
                    ref_count += 1
            elif isinstance(param, list):
                # Handle lists of references
                for item in param:
                    if isinstance(item, str) and item.startswith("#"):
                        ref_id = self._extract_reference(item)
                        if ref_id:
                            refs.append(ref_id)
                            ref_count += 1

        features["num_references"] = ref_count
        if refs:
            features["reference_ids"] = refs

        return features

    def _extract_coordinates(self, param: Any) -> Optional[List[float]]:
        """Extract coordinate list from parameter."""
        if isinstance(param, list):
            coords = []
            for item in param:
                if isinstance(item, (int, float)):
                    coords.append(float(item))
                elif isinstance(item, str):
                    num = self._extract_float(item)
                    if num is not None:
                        coords.append(num)
            return coords if coords else None
        elif isinstance(param, str):
            # Try to parse as list: "(1.0, 2.0, 3.0)"
            match = re.search(r"\(([\d\.,\s\-+eE]+)\)", param)
            if match:
                coord_str = match.group(1)
                coords = []
                for num_str in coord_str.split(","):
                    try:
                        coords.append(float(num_str.strip()))
                    except ValueError as e:
                        _log.debug(f"Failed to parse coordinate '{num_str}': {e}")
                        continue
                return coords if coords else None
        return None

    def _extract_float(self, param: Any) -> Optional[float]:
        """Extract float value from parameter."""
        if isinstance(param, (int, float)):
            return float(param)
        elif isinstance(param, str):
            # Remove any leading/trailing whitespace and quotes
            param = param.strip().strip("'\"")
            try:
                return float(param)
            except ValueError:
                # Try to extract number from string
                match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", param)
                if match:
                    try:
                        return float(match.group(0))
                    except ValueError as e:
                        _log.debug(f"Failed to parse float from regex match '{match.group(0)}': {e}")
        return None

    def _extract_int(self, param: Any) -> Optional[int]:
        """Extract integer value from parameter."""
        if isinstance(param, int):
            return param
        elif isinstance(param, str):
            param = param.strip().strip("'\"")
            try:
                return int(param)
            except ValueError:
                match = re.search(r"[-+]?\d+", param)
                if match:
                    try:
                        return int(match.group(0))
                    except ValueError as e:
                        _log.debug(f"Failed to parse int from regex match '{match.group(0)}': {e}")
        return None

    def _extract_reference(self, param: Any) -> Optional[int]:
        """Extract entity reference ID from parameter."""
        if isinstance(param, int):
            return param
        elif isinstance(param, str):
            match = re.search(r"#(\d+)", param)
            if match:
                return int(match.group(1))
        return None

    def _extract_list_of_references(self, param: Any) -> List[int]:
        """Extract list of entity reference IDs."""
        refs = []
        if isinstance(param, list):
            for item in param:
                ref = self._extract_reference(item)
                if ref:
                    refs.append(ref)
        elif isinstance(param, str):
            # Parse list format: "(#1, #2, #3)"
            matches = re.findall(r"#(\d+)", param)
            refs = [int(m) for m in matches]
        return refs

    def _extract_boolean(self, param: Any) -> bool:
        """Extract boolean value from parameter."""
        if isinstance(param, bool):
            return param
        elif isinstance(param, str):
            param_lower = param.strip().strip("'\"").upper()
            if param_lower in (".T.", "TRUE", "T"):
                return True
            elif param_lower in (".F.", "FALSE", "F"):
                return False
        return True  # Default to True

    def _compute_eccentricity(self, semi_major: float, semi_minor: float) -> float:
        """Compute eccentricity of an ellipse."""
        if semi_major == 0:
            return 0.0
        return math.sqrt(1 - (semi_minor / semi_major) ** 2)

    def compute_global_features(
        self, all_features: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute global features across all entities.

        Args:
            all_features: Dictionary of all entity features

        Returns:
            Dictionary of global features
        """
        global_features = {
            "total_entities": len(all_features),
        }

        # Count entities by category
        category_counts = Counter()
        entity_type_counts = Counter()

        for features in all_features.values():
            category = features.get("category", "other")
            entity_type = features.get("entity_type", "UNKNOWN")
            category_counts[category] += 1
            entity_type_counts[entity_type] += 1

        global_features["category_distribution"] = dict(category_counts)
        global_features["entity_type_distribution"] = dict(entity_type_counts)
        global_features["num_point_entities"] = category_counts.get("point", 0)
        global_features["num_curve_entities"] = category_counts.get("curve", 0)
        global_features["num_surface_entities"] = category_counts.get("surface", 0)
        global_features["num_topology_entities"] = category_counts.get("topology", 0)

        # Compute bounding box if we have point entities
        all_coords = []
        for features in all_features.values():
            if "coordinates" in features:
                coords = features["coordinates"]
                if coords and len(coords) >= 3:
                    all_coords.append(coords)

        if all_coords:
            x_coords = [c[0] for c in all_coords]
            y_coords = [c[1] for c in all_coords]
            z_coords = [c[2] for c in all_coords]

            global_features["bounding_box"] = {
                "x_min": min(x_coords),
                "x_max": max(x_coords),
                "y_min": min(y_coords),
                "y_max": max(y_coords),
                "z_min": min(z_coords),
                "z_max": max(z_coords),
            }

            global_features["bounding_box_volume"] = (
                (max(x_coords) - min(x_coords))
                * (max(y_coords) - min(y_coords))
                * (max(z_coords) - min(z_coords))
            )

        # Compute connectivity statistics
        ref_counts = [
            features.get("num_references", 0) for features in all_features.values()
        ]
        if ref_counts:
            global_features["avg_references_per_entity"] = sum(ref_counts) / len(
                ref_counts
            )
            global_features["max_references"] = max(ref_counts)
            global_features["total_references"] = sum(ref_counts)

        return global_features

    def create_feature_vector(
        self, features: Dict[str, Any], vector_size: int = 128
    ) -> List[float]:
        """
        Create a fixed-size feature vector for ML models.

        Args:
            features: Dictionary of features
            vector_size: Size of output vector

        Returns:
            Fixed-size feature vector
        """
        vector = [0.0] * vector_size
        idx = 0

        # Encode category (one-hot)
        categories = ["point", "curve", "surface", "topology", "shape", "other"]
        category = features.get("category", "other")
        if category in categories and idx < vector_size:
            vector[idx + categories.index(category)] = 1.0
        idx += len(categories)

        # Add numeric features
        if "coordinates" in features and idx < vector_size - 3:
            coords = features["coordinates"]
            for i, c in enumerate(coords[:3]):
                if idx + i < vector_size:
                    vector[idx + i] = c
            idx += 3

        if "radius" in features and idx < vector_size:
            vector[idx] = features["radius"]
            idx += 1

        if "distance_from_origin" in features and idx < vector_size:
            vector[idx] = features["distance_from_origin"]
            idx += 1

        if "num_references" in features and idx < vector_size:
            vector[idx] = features["num_references"]
            idx += 1

        # Boolean features
        if "is_closed" in features and idx < vector_size:
            vector[idx] = 1.0 if features["is_closed"] else 0.0
            idx += 1

        if "is_planar" in features and idx < vector_size:
            vector[idx] = 1.0 if features["is_planar"] else 0.0
            idx += 1

        if "is_linear" in features and idx < vector_size:
            vector[idx] = 1.0 if features["is_linear"] else 0.0
            idx += 1

        # Fill remaining with normalized numeric params
        if "numeric_params" in features:
            numeric_params = features["numeric_params"]
            for i, val in enumerate(numeric_params):
                if idx + i < vector_size:
                    vector[idx + i] = val

        return vector
