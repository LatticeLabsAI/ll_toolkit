"""Interference check model for CAD assemblies.

This module provides interference and collision detection capabilities for
analyzing spatial relationships between parts in CAD assemblies, including:
- Interference detection (part collisions/overlaps)
- Clearance computation (minimum distances between parts)
- Containment detection (one part inside another)
- Spatial validation and quality checking

Classes:
    InterferenceCheckModel: Main model for interference checking
    Interference: Represents a collision/interference between parts
    Clearance: Represents clearance distance between parts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from cadling.models.base_model import EnrichmentModel

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument

_log = logging.getLogger(__name__)


@dataclass
class Interference:
    """Represents an interference/collision between two parts.

    Attributes:
        interference_id: Unique identifier
        part1_id: ID of first part
        part2_id: ID of second part
        interference_type: Type (collision, overlap, penetration)
        volume: Volume of interference region (mm³)
        location: Center of interference region [x, y, z]
        severity: Severity score [0, 1] (0=touch, 1=major overlap)
        confidence: Confidence score [0, 1]
    """
    interference_id: str
    part1_id: str
    part2_id: str
    interference_type: str = "collision"
    volume: Optional[float] = None
    location: Optional[List[float]] = None
    severity: float = 0.5
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "interference_id": self.interference_id,
            "part1_id": self.part1_id,
            "part2_id": self.part2_id,
            "interference_type": self.interference_type,
            "severity": float(self.severity),
            "confidence": float(self.confidence),
        }
        if self.volume is not None:
            result["volume"] = float(self.volume)
        if self.location:
            result["location"] = [float(x) for x in self.location]
        return result


@dataclass
class Clearance:
    """Represents clearance distance between two parts.

    Attributes:
        clearance_id: Unique identifier
        part1_id: ID of first part
        part2_id: ID of second part
        distance: Minimum distance between parts (mm)
        point1: Closest point on part1 [x, y, z]
        point2: Closest point on part2 [x, y, z]
        direction: Direction vector from point1 to point2
        is_sufficient: Whether clearance meets minimum requirement
    """
    clearance_id: str
    part1_id: str
    part2_id: str
    distance: float
    point1: Optional[List[float]] = None
    point2: Optional[List[float]] = None
    direction: Optional[List[float]] = None
    is_sufficient: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "clearance_id": self.clearance_id,
            "part1_id": self.part1_id,
            "part2_id": self.part2_id,
            "distance": float(self.distance),
            "is_sufficient": self.is_sufficient,
        }
        if self.point1:
            result["point1"] = [float(x) for x in self.point1]
        if self.point2:
            result["point2"] = [float(x) for x in self.point2]
        if self.direction:
            result["direction"] = [float(x) for x in self.direction]
        return result


class InterferenceCheckModel(EnrichmentModel):
    """Interference check enrichment model for assemblies.

    Analyzes spatial relationships between parts to detect:
    - Interferences/collisions (parts overlapping)
    - Clearances (minimum distances between parts)
    - Containment (one part inside another)

    This model works with pythonocc-core shapes for STEP/IGES assemblies.

    Attributes:
        has_pythonocc: Whether pythonocc-core is available
        min_clearance: Minimum acceptable clearance (mm)
        check_containment: Whether to check for containment
        tolerance: Numerical tolerance for distance computation (mm)

    Example:
        model = InterferenceCheckModel(min_clearance=0.5)
        result = converter.convert(
            "assembly.step",
            pipeline_options=PipelineOptions(
                enrichment_models=[model]
            )
        )
        if "interference_check" in result.document.properties:
            check = result.document.properties["interference_check"]
            print(f"Interferences: {check['num_interferences']}")
            print(f"Insufficient clearances: {check['num_insufficient_clearances']}")
    """

    def __init__(
        self,
        min_clearance: float = 0.1,
        check_containment: bool = True,
        tolerance: float = 1e-6,
    ):
        """Initialize interference check model.

        Args:
            min_clearance: Minimum acceptable clearance (mm)
            check_containment: Whether to check for part containment
            tolerance: Numerical tolerance for computations (mm)
        """
        super().__init__()

        self.min_clearance = min_clearance
        self.check_containment = check_containment
        self.tolerance = tolerance

        # Check for pythonocc-core availability
        self.has_pythonocc = False
        try:
            from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
            from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common

            self.has_pythonocc = True
            _log.debug("pythonocc-core available for interference checking")
        except ImportError:
            _log.warning("pythonocc-core not available. Interference checking disabled.")

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Check for interferences and compute clearances in assembly.

        Args:
            doc: CADlingDocument being analyzed
            item_batch: List of CADItem objects (parts) to analyze
        """
        if not self.has_pythonocc:
            _log.warning(
                "Interference check requires pythonocc-core. "
                "Install with: conda install pythonocc-core -c conda-forge"
            )
            doc.properties["interference_check"] = {
                "status": "unavailable",
                "reason": "pythonocc-core not installed",
                "interferences": [],
                "clearances": [],
                "containment_issues": [],
                "num_interferences": 0,
                "num_clearances_checked": 0,
                "num_insufficient_clearances": 0,
                "num_containment_issues": 0,
                "has_interferences": False,
                "has_insufficient_clearances": False,
            }
            return

        try:
            # Check if assembly analysis already ran
            assembly_analysis = doc.properties.get("assembly_analysis", {})
            assembly_graph = assembly_analysis.get("graph", {})

            if not assembly_graph or not assembly_graph.get("nodes"):
                _log.debug("No assembly graph found. Run AssemblyAnalysisModel first.")
                return

            # Detect interferences
            interferences = self.check_interferences(doc, item_batch)

            # Compute clearances
            clearances = self._compute_all_clearances(doc, item_batch)

            # Check containment
            containment_issues = []
            if self.check_containment:
                containment_issues = self._check_all_containment(doc, item_batch)

            # Count insufficient clearances
            insufficient_clearances = [
                c for c in clearances if not c.is_sufficient
            ]

            # Store results
            doc.properties["interference_check"] = {
                "status": "success",
                "interferences": [inter.to_dict() for inter in interferences],
                "clearances": [clear.to_dict() for clear in clearances],
                "containment_issues": containment_issues,
                "num_interferences": len(interferences),
                "num_clearances_checked": len(clearances),
                "num_insufficient_clearances": len(insufficient_clearances),
                "num_containment_issues": len(containment_issues),
                "has_interferences": len(interferences) > 0,
                "has_insufficient_clearances": len(insufficient_clearances) > 0,
            }

            _log.info(
                f"Interference check completed: {len(interferences)} interferences, "
                f"{len(insufficient_clearances)} insufficient clearances"
            )

        except Exception as e:
            _log.error(f"Interference check failed: {e}", exc_info=True)

    def check_interferences(
        self, doc: CADlingDocument, item_batch: list[CADItem]
    ) -> List[Interference]:
        """Detect colliding/interfering parts in assembly.

        Uses boolean intersection operations to find overlapping geometry.

        Args:
            doc: CADlingDocument containing assembly
            item_batch: List of items to analyze

        Returns:
            List of Interference objects for detected collisions
        """
        interferences = []

        if not self.has_pythonocc:
            return interferences

        try:
            from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop

            # Get solid parts
            solid_items = [
                item for item in item_batch
                if hasattr(item, "item_type") and item.item_type in ["brep_solid", "step_entity"]
            ]

            if len(solid_items) < 2:
                return interferences

            _log.debug(f"Checking interferences among {len(solid_items)} parts")

            # Pre-extract bounding boxes for all parts (for O(1) lookup)
            part_bboxes = {}
            for i, item in enumerate(solid_items):
                part_id = f"part_{i}"
                bbox = self._get_item_bbox(item)
                if bbox:
                    part_bboxes[part_id] = bbox

            interference_id = 0
            bbox_filtered_count = 0

            # Check all pairs of parts
            for i, item1 in enumerate(solid_items):
                part1_id = f"part_{i}"
                shape1 = self._get_occ_shape(item1)

                if shape1 is None:
                    continue

                bbox1 = part_bboxes.get(part1_id)

                for j, item2 in enumerate(solid_items[i + 1:], start=i + 1):
                    part2_id = f"part_{j}"

                    # Bounding box pre-filter: skip expensive boolean op if bboxes don't overlap
                    bbox2 = part_bboxes.get(part2_id)
                    if bbox1 and bbox2 and not self._bboxes_overlap(bbox1, bbox2):
                        bbox_filtered_count += 1
                        continue  # Skip - no possible intersection

                    shape2 = self._get_occ_shape(item2)

                    if shape2 is None:
                        continue

                    # Compute boolean intersection (expensive operation)
                    common_op = BRepAlgoAPI_Common(shape1, shape2)
                    common_op.Build()

                    if not common_op.IsDone():
                        continue

                    common_shape = common_op.Shape()

                    # Compute volume of intersection
                    props = GProp_GProps()
                    brepgprop.VolumeProperties(common_shape, props)
                    intersection_volume = props.Mass()

                    # If volume > tolerance, there's an interference
                    if intersection_volume > self.tolerance:
                        # Get center of mass of intersection
                        center = props.CentreOfMass()

                        # Compute severity based on volume relative to part volumes
                        severity = self._compute_severity(
                            intersection_volume, item1, item2
                        )

                        interference = Interference(
                            interference_id=f"interference_{interference_id}",
                            part1_id=part1_id,
                            part2_id=part2_id,
                            interference_type="collision",
                            volume=intersection_volume,
                            location=[center.X(), center.Y(), center.Z()],
                            severity=severity,
                            confidence=0.9,  # High confidence for boolean ops
                        )

                        interferences.append(interference)
                        interference_id += 1

                        _log.warning(
                            f"Interference detected: {part1_id} <-> {part2_id}, "
                            f"volume={intersection_volume:.3f} mm³"
                        )

            if bbox_filtered_count > 0:
                _log.debug(
                    f"Skipped {bbox_filtered_count} part pairs via bounding box pre-filtering"
                )

        except Exception as e:
            _log.warning(f"Error checking interferences: {e}")

        return interferences

    def compute_clearances(
        self, part1: CADItem, part2: CADItem
    ) -> Optional[Clearance]:
        """Compute minimum clearance between two parts.

        Args:
            part1: First part item
            part2: Second part item

        Returns:
            Clearance object with minimum distance, or None if computation fails
        """
        if not self.has_pythonocc:
            return None

        try:
            from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

            shape1 = self._get_occ_shape(part1)
            shape2 = self._get_occ_shape(part2)

            if shape1 is None or shape2 is None:
                return None

            # Compute minimum distance
            dist_calc = BRepExtrema_DistShapeShape(shape1, shape2)
            dist_calc.Perform()

            if not dist_calc.IsDone():
                return None

            min_distance = dist_calc.Value()

            # Get closest points
            if dist_calc.NbSolution() > 0:
                pt1 = dist_calc.PointOnShape1(1)
                pt2 = dist_calc.PointOnShape2(1)

                point1 = [pt1.X(), pt1.Y(), pt1.Z()]
                point2 = [pt2.X(), pt2.Y(), pt2.Z()]

                # Compute direction vector
                direction = np.array(point2) - np.array(point1)
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = (direction / norm).tolist()
                else:
                    direction = [0, 0, 0]

                clearance = Clearance(
                    clearance_id="",  # Will be set by caller
                    part1_id="",      # Will be set by caller
                    part2_id="",      # Will be set by caller
                    distance=min_distance,
                    point1=point1,
                    point2=point2,
                    direction=direction,
                    is_sufficient=(min_distance >= self.min_clearance),
                )

                return clearance

        except Exception as e:
            _log.debug(f"Error computing clearance: {e}")

        return None

    def detect_containment(self, part1: CADItem, part2: CADItem) -> bool:
        """Check if part1 contains part2.

        Uses bounding box and centroid checks as heuristic.

        Args:
            part1: Outer part item
            part2: Inner part item

        Returns:
            True if part2 appears to be contained within part1
        """
        if not self.has_pythonocc:
            return False

        try:
            # Use geometry properties if available
            if "geometry_analysis" not in getattr(part1, "properties", {}):
                return False
            if "geometry_analysis" not in getattr(part2, "properties", {}):
                return False

            geom1 = part1.properties["geometry_analysis"]
            geom2 = part2.properties["geometry_analysis"]

            bbox1 = geom1.get("bounding_box", {})
            bbox2 = geom2.get("bounding_box", {})

            # Check if bbox2 is entirely within bbox1
            is_contained = (
                bbox1.get("min_x", 0) <= bbox2.get("min_x", 0) and
                bbox1.get("max_x", 0) >= bbox2.get("max_x", 0) and
                bbox1.get("min_y", 0) <= bbox2.get("min_y", 0) and
                bbox1.get("max_y", 0) >= bbox2.get("max_y", 0) and
                bbox1.get("min_z", 0) <= bbox2.get("min_z", 0) and
                bbox1.get("max_z", 0) >= bbox2.get("max_z", 0)
            )

            if is_contained:
                _log.debug(
                    f"Containment detected: part2 appears contained within part1"
                )

            return is_contained

        except Exception as e:
            _log.debug(f"Error detecting containment: {e}")

        return False

    def _compute_all_clearances(
        self, doc: CADlingDocument, item_batch: list[CADItem]
    ) -> List[Clearance]:
        """Compute clearances for all part pairs.

        Args:
            doc: CADlingDocument
            item_batch: List of items

        Returns:
            List of Clearance objects
        """
        clearances = []

        solid_items = [
            item for item in item_batch
            if hasattr(item, "item_type") and item.item_type in ["brep_solid", "step_entity"]
        ]

        if len(solid_items) < 2:
            return clearances

        # Pre-extract bounding boxes for all parts
        part_bboxes = {}
        for i, item in enumerate(solid_items):
            part_id = f"part_{i}"
            bbox = self._get_item_bbox(item)
            if bbox:
                part_bboxes[part_id] = bbox

        clearance_id = 0
        bbox_filtered_count = 0

        # Threshold: skip clearance computation if bbox distance > this value
        # (parts are clearly far apart, clearance will be >> min_clearance)
        max_distance_threshold = max(self.min_clearance * 100, 100.0)  # mm

        # Check all pairs
        for i, item1 in enumerate(solid_items):
            part1_id = f"part_{i}"
            bbox1 = part_bboxes.get(part1_id)

            for j, item2 in enumerate(solid_items[i + 1:], start=i + 1):
                part2_id = f"part_{j}"
                bbox2 = part_bboxes.get(part2_id)

                # Bounding box pre-filter: skip if parts are clearly far apart
                if bbox1 and bbox2:
                    bbox_dist = self._bbox_min_distance(bbox1, bbox2)
                    if bbox_dist > max_distance_threshold:
                        bbox_filtered_count += 1
                        continue  # Skip - parts are far apart

                clearance = self.compute_clearances(item1, item2)

                if clearance:
                    clearance.clearance_id = f"clearance_{clearance_id}"
                    clearance.part1_id = part1_id
                    clearance.part2_id = part2_id

                    clearances.append(clearance)
                    clearance_id += 1

                    if not clearance.is_sufficient:
                        _log.warning(
                            f"Insufficient clearance: {part1_id} <-> {part2_id}, "
                            f"distance={clearance.distance:.3f} mm "
                            f"(min={self.min_clearance} mm)"
                        )

        if bbox_filtered_count > 0:
            _log.debug(
                f"Skipped {bbox_filtered_count} part pairs in clearance computation "
                f"(bbox distance > {max_distance_threshold:.1f} mm)"
            )

        return clearances

    def _check_all_containment(
        self, doc: CADlingDocument, item_batch: list[CADItem]
    ) -> List[Dict[str, str]]:
        """Check for containment issues in all part pairs.

        Args:
            doc: CADlingDocument
            item_batch: List of items

        Returns:
            List of containment issue dicts
        """
        containment_issues = []

        solid_items = [
            item for item in item_batch
            if hasattr(item, "item_type") and item.item_type in ["brep_solid", "step_entity"]
        ]

        if len(solid_items) < 2:
            return containment_issues

        # Check all pairs
        for i, item1 in enumerate(solid_items):
            part1_id = f"part_{i}"
            for j, item2 in enumerate(solid_items[i + 1:], start=i + 1):
                part2_id = f"part_{j}"

                # Check both directions
                if self.detect_containment(item1, item2):
                    containment_issues.append({
                        "outer_part": part1_id,
                        "inner_part": part2_id,
                        "type": "full_containment"
                    })
                elif self.detect_containment(item2, item1):
                    containment_issues.append({
                        "outer_part": part2_id,
                        "inner_part": part1_id,
                        "type": "full_containment"
                    })

        return containment_issues

    def _get_occ_shape(self, item: CADItem) -> Optional[any]:
        """Get OCC shape from item.

        Args:
            item: CADItem containing shape

        Returns:
            OCC TopoDS_Shape if available, None otherwise
        """
        if not self.has_pythonocc:
            return None

        # Try to get shape from _occ_shape attribute
        if hasattr(item, "_occ_shape"):
            return item._occ_shape

        # Try to get from properties
        if "occ_shape" in getattr(item, "properties", {}):
            return item.properties["occ_shape"]

        return None

    def _compute_severity(
        self, intersection_volume: float, item1: CADItem, item2: CADItem
    ) -> float:
        """Compute severity of interference based on volumes.

        Args:
            intersection_volume: Volume of intersection
            item1, item2: Part items

        Returns:
            Severity score [0, 1]
        """
        # Get part volumes if available
        vol1 = None
        vol2 = None

        if "geometry_analysis" in getattr(item1, "properties", {}):
            vol1 = item1.properties["geometry_analysis"].get("volume")

        if "geometry_analysis" in getattr(item2, "properties", {}):
            vol2 = item2.properties["geometry_analysis"].get("volume")

        if vol1 is None or vol2 is None or vol1 <= 0 or vol2 <= 0:
            # Default severity if volumes unknown
            return 0.5

        # Severity = intersection volume / min part volume
        min_volume = min(vol1, vol2)
        severity = min(intersection_volume / min_volume, 1.0)

        return severity

    def _get_item_bbox(self, item: CADItem) -> Optional[Dict[str, float]]:
        """Get bounding box from item properties.

        Args:
            item: CADItem to get bounding box from

        Returns:
            Dict with min_x, max_x, min_y, max_y, min_z, max_z or None
        """
        if "geometry_analysis" in getattr(item, "properties", {}):
            geom = item.properties["geometry_analysis"]
            bbox = geom.get("bounding_box")
            if bbox and all(k in bbox for k in ["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]):
                return bbox
        return None

    def _bboxes_overlap(self, bbox1: Dict[str, float], bbox2: Dict[str, float]) -> bool:
        """Check if two bounding boxes overlap.

        Fast O(1) check to filter out part pairs that cannot possibly intersect.

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box

        Returns:
            True if bounding boxes overlap, False otherwise
        """
        # Check for separation along each axis
        # If separated on any axis, boxes don't overlap
        if bbox1["max_x"] < bbox2["min_x"] or bbox2["max_x"] < bbox1["min_x"]:
            return False
        if bbox1["max_y"] < bbox2["min_y"] or bbox2["max_y"] < bbox1["min_y"]:
            return False
        if bbox1["max_z"] < bbox2["min_z"] or bbox2["max_z"] < bbox1["min_z"]:
            return False
        return True

    def _bbox_min_distance(self, bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
        """Compute minimum distance between two bounding boxes.

        Fast O(1) lower bound for actual part distance. Used to skip
        expensive clearance computations for parts that are clearly far apart.

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box

        Returns:
            Minimum distance between bounding boxes (0 if overlapping)
        """
        # Compute gap on each axis (0 if overlapping on that axis)
        gap_x = max(0, bbox1["min_x"] - bbox2["max_x"], bbox2["min_x"] - bbox1["max_x"])
        gap_y = max(0, bbox1["min_y"] - bbox2["max_y"], bbox2["min_y"] - bbox1["max_y"])
        gap_z = max(0, bbox1["min_z"] - bbox2["max_z"], bbox2["min_z"] - bbox1["max_z"])

        # Euclidean distance between boxes
        return np.sqrt(gap_x**2 + gap_y**2 + gap_z**2)
