"""Pattern detection model for CAD features.

Detects repeating patterns in CAD geometry:
- Linear patterns (arrays along one axis)
- Circular patterns (rotational arrays)
- Mirror patterns (symmetric features across planes)
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem
    from cadling.core import CADlingDocument

from cadling.models.base_model import EnrichmentModel

_log = logging.getLogger(__name__)


class PatternDetectionModel(EnrichmentModel):
    """Enrichment model for detecting geometric patterns.
    
    Analyzes CAD items to detect:
    - Linear patterns: Repeated features along straight lines
    - Circular patterns: Features arranged in circular/rotational arrays
    - Mirror patterns: Symmetric features across reflection planes
    
    Attributes:
        position_tolerance: Distance threshold for grouping features (default 0.01)
        angle_tolerance: Angle threshold in degrees for circular patterns (default 1.0)
        min_pattern_count: Minimum instances to consider a pattern (default 3)
        has_numpy: Whether numpy is available
    """

    def __init__(
        self,
        position_tolerance: float = 0.01,
        angle_tolerance: float = 1.0,
        min_pattern_count: int = 3
    ):
        """Initialize pattern detection model.
        
        Args:
            position_tolerance: Maximum distance between pattern instances
            angle_tolerance: Maximum angle deviation for circular patterns (degrees)
            min_pattern_count: Minimum number of instances to form a pattern
        """
        super().__init__()
        
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance
        self.min_pattern_count = min_pattern_count
        
        # Check numpy availability
        try:
            import numpy as np
            self.has_numpy = True
        except ImportError:
            _log.warning("numpy not available, pattern detection disabled")
            self.has_numpy = False

    def __call__(self, doc: "CADlingDocument", item_batch: List["CADItem"]) -> None:
        """Detect patterns in CAD items.
        
        Args:
            doc: CADling document containing items
            item_batch: Batch of items to process
        """
        if not self.has_numpy:
            _log.debug("Pattern detection skipped: numpy not available")
            return
        
        try:
            # Detect patterns across all items in document
            pattern_result = self._detect_patterns(doc, item_batch)
            
            if pattern_result:
                # Store at document level
                if not hasattr(doc, 'properties'):
                    doc.properties = {}
                
                doc.properties["pattern_detection"] = pattern_result
                
                _log.debug(
                    f"Detected {len(pattern_result.get('linear_patterns', []))} linear, "
                    f"{len(pattern_result.get('circular_patterns', []))} circular, "
                    f"{len(pattern_result.get('mirror_patterns', []))} mirror patterns"
                )
                
        except Exception as e:
            _log.error(f"Pattern detection failed: {e}", exc_info=True)

    def _detect_patterns(
        self,
        doc: "CADlingDocument",
        item_batch: List["CADItem"]
    ) -> Optional[Dict[str, Any]]:
        """Detect all pattern types in item batch.
        
        Args:
            doc: CADling document
            item_batch: Items to analyze
            
        Returns:
            Dictionary with detected patterns or None
        """
        # Extract features with positions from items
        features = self._extract_features_from_items(item_batch)
        
        if len(features) < self.min_pattern_count:
            _log.debug(f"Not enough features ({len(features)}) for pattern detection")
            return {
                "linear_patterns": [],
                "circular_patterns": [],
                "mirror_patterns": [],
                "num_features_analyzed": len(features),
                "total_patterns_found": 0,
            }
        
        results = {}
        
        # Detect linear patterns
        linear_patterns = self.detect_linear_patterns(features)
        if linear_patterns:
            results["linear_patterns"] = linear_patterns
        
        # Detect circular patterns
        circular_patterns = self.detect_circular_patterns(features)
        if circular_patterns:
            results["circular_patterns"] = circular_patterns
        
        # Detect mirror patterns
        mirror_patterns = self.detect_mirror_patterns(features)
        if mirror_patterns:
            results["mirror_patterns"] = mirror_patterns
        
        # Summary statistics
        results["num_features_analyzed"] = len(features)
        results["total_patterns_found"] = (
            len(results.get("linear_patterns", [])) +
            len(results.get("circular_patterns", [])) +
            len(results.get("mirror_patterns", []))
        )
        
        # Always return results even if no patterns found
        if "linear_patterns" not in results:
            results["linear_patterns"] = []
        if "circular_patterns" not in results:
            results["circular_patterns"] = []
        if "mirror_patterns" not in results:
            results["mirror_patterns"] = []
        return results

    def _extract_features_from_items(
        self,
        item_batch: List["CADItem"]
    ) -> List[Dict[str, Any]]:
        """Extract feature positions and properties from items.
        
        Args:
            item_batch: Items to extract features from
            
        Returns:
            List of feature dictionaries with position, type, etc.
        """
        features = []
        
        for item in item_batch:
            # Check for geometry analysis properties
            if "geometry_analysis" in item.properties:
                geom = item.properties["geometry_analysis"]

                # Handle both 'centroid' and 'center_of_mass' keys for compatibility
                position = geom.get("centroid") or geom.get("center_of_mass", [0, 0, 0])
                # Normalize center_of_mass dict to list if needed
                if isinstance(position, dict):
                    position = [position.get("x", 0), position.get("y", 0), position.get("z", 0)]

                # Extract feature type from label or feature_recognition
                feature_type = "unknown"
                if hasattr(item, 'label') and hasattr(item.label, 'text'):
                    feature_type = item.label.text
                elif "feature_recognition" in item.properties:
                    feat_rec = item.properties["feature_recognition"]
                    if isinstance(feat_rec, dict) and "feature_type" in feat_rec:
                        feature_type = feat_rec["feature_type"]

                feature = {
                    "item": item,
                    "position": position,
                    "type": feature_type,
                    "volume": geom.get("volume", 0),
                    "bbox": geom.get("bounding_box", {}),
                }

                # Add surface analysis if available
                if "surface_analysis" in item.properties:
                    surf = item.properties["surface_analysis"]
                    # Handle both 'surface_type_distribution' and 'surface_type' keys
                    feature["surface_type"] = surf.get("surface_type_distribution") or surf.get("surface_type", {})
                
                features.append(feature)
            
            # Also check for feature recognition results
            elif "feature_recognition" in item.properties:
                feat_rec = item.properties["feature_recognition"]
                
                # Extract holes, pockets, etc.
                for feat_type in ["holes", "pockets", "bosses"]:
                    if feat_type in feat_rec:
                        for feat in feat_rec[feat_type]:
                            feature = {
                                "item": item,
                                "position": feat.get("location", [0, 0, 0]),
                                "type": feat_type[:-1],  # Remove plural
                                "diameter": feat.get("diameter", 0),
                                "depth": feat.get("depth", 0),
                            }
                            features.append(feature)
        
        return features

    def detect_linear_patterns(
        self,
        features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect linear patterns in features.
        
        A linear pattern is a set of features arranged along a straight line
        with equal spacing.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            List of detected linear pattern dictionaries
        """
        if len(features) < self.min_pattern_count:
            return []
        
        patterns = []
        positions = np.array([f["position"] for f in features])
        
        # Group features by type
        feature_groups = self._group_features_by_type(features)
        
        for feat_type, group_features in feature_groups.items():
            if len(group_features) < self.min_pattern_count:
                continue
            
            group_positions = np.array([f["position"] for f in group_features])
            
            # Try to find linear patterns in this group
            pattern_groups = self._find_linear_patterns_in_group(
                group_features,
                group_positions
            )
            
            for pattern_indices in pattern_groups:
                if len(pattern_indices) >= self.min_pattern_count:
                    pattern_positions = group_positions[pattern_indices]
                    
                    # Compute direction vector
                    direction = self._compute_pattern_direction(pattern_positions)
                    
                    # Compute spacing
                    spacings = self._compute_spacings(pattern_positions, direction)
                    avg_spacing = float(np.mean(spacings))
                    
                    pattern = {
                        "type": "linear",
                        "feature_type": feat_type,
                        "count": len(pattern_indices),
                        "direction": direction.tolist(),
                        "spacing": avg_spacing,
                        "spacing_std": float(np.std(spacings)),
                        "positions": pattern_positions.tolist(),
                        "feature_indices": pattern_indices,
                    }
                    
                    patterns.append(pattern)
        
        return patterns

    def _find_linear_patterns_in_group(
        self,
        features: List[Dict[str, Any]],
        positions: np.ndarray
    ) -> List[List[int]]:
        """Find linear patterns within a feature group.
        
        Args:
            features: Features in the group
            positions: Position array
            
        Returns:
            List of index lists, each representing a linear pattern
        """
        if len(positions) < self.min_pattern_count:
            return []
        
        patterns = []
        used = set()
        
        # Try each pair as potential pattern seed
        for i in range(len(positions)):
            if i in used:
                continue
            
            for j in range(i + 1, len(positions)):
                if j in used:
                    continue
                
                # Compute direction from i to j
                direction = positions[j] - positions[i]
                dist = np.linalg.norm(direction)
                
                if dist < 1e-9:
                    continue
                
                direction = direction / dist
                
                # Find all features aligned with this direction
                aligned = [i, j]
                
                for k in range(len(positions)):
                    if k in aligned or k in used:
                        continue
                    
                    # Check if k is aligned with i-j direction
                    vec = positions[k] - positions[i]
                    proj = np.dot(vec, direction)
                    
                    # Distance from k to line i-j
                    rejection = vec - proj * direction
                    perp_dist = np.linalg.norm(rejection)
                    
                    if perp_dist < self.position_tolerance:
                        # Check if spacing is consistent
                        if self._is_consistent_spacing(positions, aligned, k, direction):
                            aligned.append(k)
                
                # If we found enough aligned features, it's a pattern
                if len(aligned) >= self.min_pattern_count:
                    # Sort by position along direction
                    aligned = sorted(aligned, key=lambda idx: np.dot(positions[idx], direction))
                    patterns.append(aligned)
                    used.update(aligned)
                    break
        
        return patterns

    def _is_consistent_spacing(
        self,
        positions: np.ndarray,
        aligned: List[int],
        candidate: int,
        direction: np.ndarray
    ) -> bool:
        """Check if candidate maintains consistent spacing.
        
        Args:
            positions: All positions
            aligned: Indices of already aligned features
            candidate: Index of candidate feature
            direction: Pattern direction vector
            
        Returns:
            True if spacing is consistent
        """
        # Project all positions onto direction
        projections = [np.dot(positions[idx], direction) for idx in aligned]
        candidate_proj = np.dot(positions[candidate], direction)
        
        # Compute spacings
        all_projs = sorted(projections + [candidate_proj])
        spacings = np.diff(all_projs)
        
        if len(spacings) < 2:
            return True
        
        # Check if spacings are similar
        mean_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)
        
        return std_spacing < self.position_tolerance

    def _compute_pattern_direction(self, positions: np.ndarray) -> np.ndarray:
        """Compute direction vector of linear pattern using PCA.
        
        Args:
            positions: Array of positions [N, 3]
            
        Returns:
            Normalized direction vector [3]
        """
        # Center positions
        centered = positions - np.mean(positions, axis=0)
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Eigenvector with largest eigenvalue is principal direction
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        direction = eigenvectors[:, -1]  # Largest eigenvalue
        
        return direction / np.linalg.norm(direction)

    def _compute_spacings(
        self,
        positions: np.ndarray,
        direction: np.ndarray
    ) -> np.ndarray:
        """Compute spacings between consecutive features along direction.
        
        Args:
            positions: Feature positions
            direction: Pattern direction
            
        Returns:
            Array of spacings
        """
        # Project onto direction
        projections = np.dot(positions, direction)
        
        # Sort by projection
        sorted_projs = np.sort(projections)
        
        # Compute differences
        spacings = np.diff(sorted_projs)
        
        return spacings

    def detect_circular_patterns(
        self,
        features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect circular (rotational) patterns.
        
        A circular pattern is a set of features arranged in a circle
        at equal angular intervals.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            List of detected circular pattern dictionaries
        """
        if len(features) < self.min_pattern_count:
            return []
        
        patterns = []
        
        # Group by type
        feature_groups = self._group_features_by_type(features)
        
        for feat_type, group_features in feature_groups.items():
            if len(group_features) < self.min_pattern_count:
                continue
            
            group_positions = np.array([f["position"] for f in group_features])
            
            # Try to find circular patterns
            circular_groups = self._find_circular_patterns_in_group(
                group_features,
                group_positions
            )
            
            for pattern_indices, center, axis, radius in circular_groups:
                if len(pattern_indices) >= self.min_pattern_count:
                    pattern_positions = group_positions[pattern_indices]
                    
                    # Compute angular spacing
                    angles = self._compute_angles_from_center(
                        pattern_positions,
                        center,
                        axis
                    )
                    
                    angular_spacing = float(np.mean(np.diff(np.sort(angles))))
                    
                    pattern = {
                        "type": "circular",
                        "feature_type": feat_type,
                        "count": len(pattern_indices),
                        "center": center.tolist(),
                        "axis": axis.tolist(),
                        "radius": float(radius),
                        "angular_spacing": angular_spacing,
                        "positions": pattern_positions.tolist(),
                        "feature_indices": pattern_indices,
                    }
                    
                    patterns.append(pattern)
        
        return patterns

    def _find_circular_patterns_in_group(
        self,
        features: List[Dict[str, Any]],
        positions: np.ndarray
    ) -> List[Tuple[List[int], np.ndarray, np.ndarray, float]]:
        """Find circular patterns within feature group.
        
        Args:
            features: Features in group
            positions: Position array
            
        Returns:
            List of (indices, center, axis, radius) tuples
        """
        patterns = []
        
        if len(positions) < self.min_pattern_count:
            return patterns
        
        # Try to fit circles to subsets of points
        # Simple heuristic: assume patterns are in XY, XZ, or YZ planes
        
        for axis_idx in range(3):  # Try each axis
            axis = np.zeros(3)
            axis[axis_idx] = 1.0
            
            # Project positions onto plane perpendicular to axis
            plane_positions = self._project_onto_plane(positions, axis)
            
            # Try to find circular arrangement
            circle_result = self._fit_circle_2d(plane_positions)
            
            if circle_result is not None:
                center_2d, radius, inlier_indices = circle_result
                
                if len(inlier_indices) >= self.min_pattern_count:
                    # Convert 2D center back to 3D
                    center = self._unproject_from_plane(center_2d, axis, positions)
                    
                    patterns.append((inlier_indices, center, axis, radius))
        
        return patterns

    def _project_onto_plane(
        self,
        positions: np.ndarray,
        normal: np.ndarray
    ) -> np.ndarray:
        """Project 3D positions onto plane perpendicular to normal.
        
        Args:
            positions: 3D positions [N, 3]
            normal: Plane normal [3]
            
        Returns:
            2D positions [N, 2]
        """
        # Create orthonormal basis for plane
        # Find two vectors perpendicular to normal
        if abs(normal[0]) < 0.9:
            u = np.cross(normal, [1, 0, 0])
        else:
            u = np.cross(normal, [0, 1, 0])
        
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        # Project onto u-v plane
        projected = np.column_stack([
            np.dot(positions, u),
            np.dot(positions, v)
        ])
        
        return projected

    def _fit_circle_2d(
        self,
        positions: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float, List[int]]]:
        """Fit circle to 2D points using least squares.
        
        Args:
            positions: 2D positions [N, 2]
            
        Returns:
            (center, radius, inlier_indices) or None
        """
        if len(positions) < 3:
            return None
        
        # Algebraic circle fit
        # (x - cx)^2 + (y - cy)^2 = r^2
        # x^2 + y^2 - 2*cx*x - 2*cy*y + (cx^2 + cy^2 - r^2) = 0
        
        A = np.column_stack([
            2 * positions[:, 0],
            2 * positions[:, 1],
            np.ones(len(positions))
        ])
        
        b = positions[:, 0]**2 + positions[:, 1]**2
        
        try:
            # Solve least squares
            params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            center = np.array([params[0], params[1]])
            radius = np.sqrt(params[2] + center[0]**2 + center[1]**2)
            
            # Find inliers
            distances = np.sqrt(np.sum((positions - center)**2, axis=1))
            errors = np.abs(distances - radius)
            
            inliers = np.where(errors < self.position_tolerance)[0].tolist()
            
            if len(inliers) >= self.min_pattern_count:
                return center, radius, inliers
            
        except np.linalg.LinAlgError as e:
            _log.debug(f"Circle fitting failed (singular matrix): {e}")
        
        return None

    def _unproject_from_plane(
        self,
        position_2d: np.ndarray,
        normal: np.ndarray,
        reference_positions: np.ndarray
    ) -> np.ndarray:
        """Convert 2D plane position back to 3D.
        
        Args:
            position_2d: 2D position [2]
            normal: Plane normal [3]
            reference_positions: Original 3D positions for determining plane offset
            
        Returns:
            3D position [3]
        """
        # Compute plane basis
        if abs(normal[0]) < 0.9:
            u = np.cross(normal, [1, 0, 0])
        else:
            u = np.cross(normal, [0, 1, 0])
        
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        # Compute plane offset using mean of reference positions
        mean_pos = np.mean(reference_positions, axis=0)
        offset = np.dot(mean_pos, normal)
        
        # Reconstruct 3D position
        position_3d = position_2d[0] * u + position_2d[1] * v + offset * normal
        
        return position_3d

    def _compute_angles_from_center(
        self,
        positions: np.ndarray,
        center: np.ndarray,
        axis: np.ndarray
    ) -> np.ndarray:
        """Compute angles of positions around center.
        
        Args:
            positions: Feature positions [N, 3]
            center: Circle center [3]
            axis: Rotation axis [3]
            
        Returns:
            Array of angles in radians [N]
        """
        # Vectors from center to positions
        vectors = positions - center
        
        # Project onto plane perpendicular to axis
        projected = vectors - np.outer(np.dot(vectors, axis), axis)
        
        # Compute angles using atan2
        # Need reference direction in plane
        if abs(axis[0]) < 0.9:
            ref = np.cross(axis, [1, 0, 0])
        else:
            ref = np.cross(axis, [0, 1, 0])
        
        ref = ref / np.linalg.norm(ref)
        perp = np.cross(axis, ref)
        
        x = np.dot(projected, ref)
        y = np.dot(projected, perp)
        
        angles = np.arctan2(y, x)
        
        return angles

    def detect_mirror_patterns(
        self,
        features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect mirror (reflection) patterns.
        
        A mirror pattern is a pair of features that are symmetric
        across a reflection plane.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            List of detected mirror pattern dictionaries
        """
        if len(features) < 2:
            return []
        
        patterns = []
        
        # Group by type
        feature_groups = self._group_features_by_type(features)
        
        for feat_type, group_features in feature_groups.items():
            if len(group_features) < 2:
                continue
            
            group_positions = np.array([f["position"] for f in group_features])
            
            # Try standard planes (XY, XZ, YZ) and origin-centered planes
            mirror_planes = self._generate_candidate_mirror_planes(group_positions)
            
            for plane_normal, plane_point in mirror_planes:
                mirrored_pairs = self._find_mirror_pairs(
                    group_features,
                    group_positions,
                    plane_normal,
                    plane_point
                )
                
                if len(mirrored_pairs) >= 1:  # At least one pair
                    pattern = {
                        "type": "mirror",
                        "feature_type": feat_type,
                        "pair_count": len(mirrored_pairs),
                        "plane_normal": plane_normal.tolist(),
                        "plane_point": plane_point.tolist(),
                        "pairs": [
                            {
                                "indices": [idx1, idx2],
                                "positions": [
                                    group_positions[idx1].tolist(),
                                    group_positions[idx2].tolist()
                                ]
                            }
                            for idx1, idx2 in mirrored_pairs
                        ],
                    }
                    
                    patterns.append(pattern)
        
        return patterns

    def _generate_candidate_mirror_planes(
        self,
        positions: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate candidate mirror planes to test.
        
        Args:
            positions: Feature positions
            
        Returns:
            List of (normal, point) tuples defining planes
        """
        planes = []
        
        # Standard axis-aligned planes through origin
        for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            planes.append((np.array(axis), np.array([0, 0, 0])))
        
        # Planes through centroid
        centroid = np.mean(positions, axis=0)
        for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            planes.append((np.array(axis), centroid))
        
        return planes

    def _find_mirror_pairs(
        self,
        features: List[Dict[str, Any]],
        positions: np.ndarray,
        plane_normal: np.ndarray,
        plane_point: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Find pairs of features that are mirror symmetric.
        
        Args:
            features: Feature list
            positions: Position array
            plane_normal: Mirror plane normal
            plane_point: Point on mirror plane
            
        Returns:
            List of (index1, index2) pairs
        """
        pairs = []
        matched = set()
        
        for i in range(len(positions)):
            if i in matched:
                continue
            
            # Reflect position i across plane
            reflected = self._reflect_point(positions[i], plane_normal, plane_point)
            
            # Find closest unmatched feature to reflection
            for j in range(i + 1, len(positions)):
                if j in matched:
                    continue
                
                distance = np.linalg.norm(positions[j] - reflected)
                
                if distance < self.position_tolerance:
                    pairs.append((i, j))
                    matched.add(i)
                    matched.add(j)
                    break
        
        return pairs

    def _reflect_point(
        self,
        point: np.ndarray,
        plane_normal: np.ndarray,
        plane_point: np.ndarray
    ) -> np.ndarray:
        """Reflect point across plane.
        
        Args:
            point: Point to reflect [3]
            plane_normal: Plane normal [3]
            plane_point: Point on plane [3]
            
        Returns:
            Reflected point [3]
        """
        # Distance from point to plane
        distance = np.dot(point - plane_point, plane_normal)
        
        # Reflect
        reflected = point - 2 * distance * plane_normal
        
        return reflected

    def _group_features_by_type(
        self,
        features: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group features by type.
        
        Args:
            features: List of features
            
        Returns:
            Dictionary mapping type -> feature list
        """
        groups = {}
        
        for feature in features:
            feat_type = feature.get("type", "unknown")
            
            if feat_type not in groups:
                groups[feat_type] = []
            
            groups[feat_type].append(feature)
        
        return groups

    def extract_pattern_parameters(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key parameters from detected pattern.
        
        Args:
            pattern: Pattern dictionary
            
        Returns:
            Dictionary of pattern parameters
        """
        params = {
            "type": pattern["type"],
            "count": pattern["count"],
        }
        
        if pattern["type"] == "linear":
            params["spacing"] = pattern["spacing"]
            params["direction"] = pattern["direction"]
            
        elif pattern["type"] == "circular":
            params["radius"] = pattern["radius"]
            params["angular_spacing"] = pattern["angular_spacing"]
            params["axis"] = pattern["axis"]
            
        elif pattern["type"] == "mirror":
            params["pair_count"] = pattern["pair_count"]
            params["plane_normal"] = pattern["plane_normal"]
            params["plane_point"] = pattern["plane_point"]

        return params
