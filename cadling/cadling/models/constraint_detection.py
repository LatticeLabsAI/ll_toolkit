"""Constraint detection model for CAD assemblies.

This module provides constraint and mate detection capabilities for analyzing
assembly relationships between parts, including:
- Concentric mate detection (shafts, bearings, cylindrical fits)
- Planar contact detection (flat mating surfaces)
- Fastener detection (bolts, screws, nuts connecting parts)
- Constraint type classification and validation

Classes:
    ConstraintDetectionModel: Main model for constraint detection
    Mate: Represents a mating relationship between parts
    Fastener: Represents a fastener (bolt, screw, nut) in assembly
    ConstraintType: Enum of constraint types
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from cadling.models.base_model import EnrichmentModel

# Try to import scipy for spatial indexing (optional but recommended)
try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument

_log = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of assembly constraints."""
    CONCENTRIC = "concentric"           # Cylindrical surfaces aligned
    PLANAR = "planar"                   # Flat surfaces in contact
    COINCIDENT = "coincident"           # Points or edges aligned
    PARALLEL = "parallel"               # Surfaces parallel but offset
    PERPENDICULAR = "perpendicular"     # Surfaces at 90 degrees
    TANGENT = "tangent"                 # Surfaces tangent
    DISTANCE = "distance"               # Fixed distance constraint
    ANGLE = "angle"                     # Fixed angle constraint


class FastenerType(Enum):
    """Types of fasteners in assemblies."""
    BOLT = "bolt"
    SCREW = "screw"
    NUT = "nut"
    WASHER = "washer"
    RIVET = "rivet"
    PIN = "pin"
    UNKNOWN = "unknown"


@dataclass
class Mate:
    """Represents a mating relationship/constraint between parts.

    Attributes:
        mate_id: Unique identifier
        part1_id: ID of first part
        part2_id: ID of second part
        constraint_type: Type of constraint
        parameters: Constraint-specific parameters (radius for concentric, etc.)
        location: 3D location of mate [x, y, z]
        direction: Direction vector for aligned mates
        confidence: Confidence score [0, 1]
    """
    mate_id: str
    part1_id: str
    part2_id: str
    constraint_type: ConstraintType
    parameters: Dict[str, Any] = field(default_factory=dict)
    location: Optional[List[float]] = None
    direction: Optional[List[float]] = None
    confidence: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "mate_id": self.mate_id,
            "part1_id": self.part1_id,
            "part2_id": self.part2_id,
            "constraint_type": self.constraint_type.value,
            "parameters": self.parameters,
            "confidence": float(self.confidence),
        }
        if self.location:
            result["location"] = [float(x) for x in self.location]
        if self.direction:
            result["direction"] = [float(x) for x in self.direction]
        return result


@dataclass
class Fastener:
    """Represents a fastener connecting parts.

    Attributes:
        fastener_id: Unique identifier
        fastener_type: Type of fastener
        connected_parts: List of part IDs connected by this fastener
        location: 3D location [x, y, z]
        axis: Axis direction for bolts/screws
        diameter: Nominal diameter (mm)
        length: Nominal length (mm)
        thread_pitch: Thread pitch for threaded fasteners (mm)
        confidence: Confidence score [0, 1]
    """
    fastener_id: str
    fastener_type: FastenerType
    connected_parts: List[str] = field(default_factory=list)
    location: Optional[List[float]] = None
    axis: Optional[List[float]] = None
    diameter: Optional[float] = None
    length: Optional[float] = None
    thread_pitch: Optional[float] = None
    confidence: float = 0.6

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "fastener_id": self.fastener_id,
            "fastener_type": self.fastener_type.value,
            "connected_parts": self.connected_parts,
            "confidence": float(self.confidence),
        }
        if self.location:
            result["location"] = [float(x) for x in self.location]
        if self.axis:
            result["axis"] = [float(x) for x in self.axis]
        if self.diameter is not None:
            result["diameter"] = float(self.diameter)
        if self.length is not None:
            result["length"] = float(self.length)
        if self.thread_pitch is not None:
            result["thread_pitch"] = float(self.thread_pitch)
        return result


class ConstraintDetectionModel(EnrichmentModel):
    """Constraint detection enrichment model for assemblies.

    Analyzes assembly relationships to detect:
    - Mating constraints between parts (concentric, planar, tangent, etc.)
    - Fasteners connecting parts (bolts, screws, nuts)
    - Constraint parameters (radii, angles, distances)

    This model works with pythonocc-core shapes for STEP/IGES assemblies.

    Attributes:
        has_pythonocc: Whether pythonocc-core is available
        concentric_tolerance: Tolerance for concentric mate detection (mm)
        planar_tolerance: Tolerance for planar contact detection (mm)
        min_contact_area: Minimum area for contact detection (mm²)

    Example:
        model = ConstraintDetectionModel()
        result = converter.convert(
            "assembly.step",
            pipeline_options=PipelineOptions(
                enrichment_models=[model]
            )
        )
        if "constraint_detection" in result.document.properties:
            constraints = result.document.properties["constraint_detection"]
            print(f"Mates: {len(constraints['mates'])}")
            print(f"Fasteners: {len(constraints['fasteners'])}")
    """

    def __init__(
        self,
        concentric_tolerance: float = 0.1,
        planar_tolerance: float = 0.01,
        min_contact_area: float = 1.0,
    ):
        """Initialize constraint detection model.

        Args:
            concentric_tolerance: Tolerance for concentric detection (mm)
            planar_tolerance: Tolerance for planar contact (mm)
            min_contact_area: Minimum contact area (mm²)
        """
        super().__init__()

        self.concentric_tolerance = concentric_tolerance
        self.planar_tolerance = planar_tolerance
        self.min_contact_area = min_contact_area

        # Check for pythonocc-core availability
        self.has_pythonocc = False
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
            from OCC.Core.GeomAbs import GeomAbs_SurfaceType, GeomAbs_CurveType

            self.has_pythonocc = True
            _log.debug("pythonocc-core available for constraint detection")
        except ImportError:
            _log.warning("pythonocc-core not available. Constraint detection disabled.")

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Detect constraints and fasteners in assembly.

        Args:
            doc: CADlingDocument being analyzed
            item_batch: List of CADItem objects (parts) to analyze
        """
        if not self.has_pythonocc:
            _log.debug("Constraint detection skipped: pythonocc not available")
            return

        try:
            # Check if assembly analysis already ran
            assembly_analysis = doc.properties.get("assembly_analysis", {})
            assembly_graph = assembly_analysis.get("graph", {})

            if not assembly_graph or not assembly_graph.get("nodes"):
                _log.debug("No assembly graph found. Run AssemblyAnalysisModel first.")
                return

            # Detect mates
            concentric_mates = self.detect_concentric_mates(doc, item_batch)
            planar_mates = self.detect_planar_contacts(doc, item_batch)

            all_mates = concentric_mates + planar_mates

            # Detect fasteners
            fasteners = self.detect_fasteners(doc, item_batch)

            # Store results
            doc.properties["constraint_detection"] = {
                "mates": [mate.to_dict() for mate in all_mates],
                "fasteners": [fastener.to_dict() for fastener in fasteners],
                "num_mates": len(all_mates),
                "num_concentric_mates": len(concentric_mates),
                "num_planar_mates": len(planar_mates),
                "num_fasteners": len(fasteners),
            }

            _log.info(
                f"Constraint detection completed: {len(all_mates)} mates, "
                f"{len(fasteners)} fasteners"
            )

        except Exception as e:
            _log.error(f"Constraint detection failed: {e}", exc_info=True)

    def detect_concentric_mates(
        self, doc: CADlingDocument, item_batch: list[CADItem]
    ) -> List[Mate]:
        """Detect concentric mates between cylindrical surfaces.

        Finds pairs of cylindrical surfaces (holes, shafts, bearings) that are
        aligned concentrically.

        Args:
            doc: CADlingDocument containing assembly
            item_batch: List of items to analyze

        Returns:
            List of Mate objects for concentric relationships
        """
        mates = []

        if not self.has_pythonocc:
            return mates

        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Cylinder

            # Get solid parts from assembly
            solid_items = [
                item for item in item_batch
                if hasattr(item, "item_type") and item.item_type in ["brep_solid", "step_entity"]
            ]

            if len(solid_items) < 2:
                return mates

            _log.debug(f"Detecting concentric mates among {len(solid_items)} parts")

            # Extract cylindrical faces from each part
            part_cylinders = {}  # part_id -> list of (face, radius, axis, center)

            for i, item in enumerate(solid_items):
                part_id = f"part_{i}"
                cylinders = self._extract_cylindrical_faces(item)
                if cylinders:
                    part_cylinders[part_id] = cylinders

            # Pre-normalize all cylinder axes (avoid repeated normalization in loop)
            for part_id, cylinders in part_cylinders.items():
                for cyl in cylinders:
                    axis = np.array(cyl["axis"])
                    norm = np.linalg.norm(axis)
                    if norm > 0:
                        cyl["axis_normalized"] = axis / norm
                    else:
                        cyl["axis_normalized"] = axis

            # Build flat list of all cylinders with part references for spatial indexing
            all_cylinders = []
            cylinder_part_map = []  # Maps cylinder index to (part_id, local_idx)
            for part_id, cylinders in part_cylinders.items():
                for local_idx, cyl in enumerate(cylinders):
                    all_cylinders.append(cyl)
                    cylinder_part_map.append((part_id, local_idx))

            mate_id = 0

            # Use KDTree for O(n log n) spatial queries if scipy available
            if HAS_SCIPY and len(all_cylinders) > 10:
                # Build KDTree from cylinder centers
                centers = np.array([cyl["center"] for cyl in all_cylinders])
                tree = KDTree(centers)

                # Query nearby cylinders (within 2x max radius as search radius)
                max_radius = max(cyl["radius"] for cyl in all_cylinders)
                search_radius = max(max_radius * 2, self.concentric_tolerance * 10)

                # Find pairs within search radius
                pairs = tree.query_pairs(r=search_radius, output_type='ndarray')

                for i, j in pairs:
                    part1_id, _ = cylinder_part_map[i]
                    part2_id, _ = cylinder_part_map[j]

                    # Skip cylinders from same part
                    if part1_id == part2_id:
                        continue

                    cyl1 = all_cylinders[i]
                    cyl2 = all_cylinders[j]

                    # Check if cylinders are concentric
                    if self._are_concentric_fast(cyl1, cyl2):
                        mate = Mate(
                            mate_id=f"concentric_{mate_id}",
                            part1_id=part1_id,
                            part2_id=part2_id,
                            constraint_type=ConstraintType.CONCENTRIC,
                            parameters={
                                "radius1": cyl1["radius"],
                                "radius2": cyl2["radius"],
                                "radial_clearance": abs(cyl1["radius"] - cyl2["radius"]),
                            },
                            location=cyl1["center"],
                            direction=cyl1["axis"],
                            confidence=0.85,
                        )
                        mates.append(mate)
                        mate_id += 1

                        _log.debug(
                            f"Found concentric mate: {part1_id} <-> {part2_id}, "
                            f"r1={cyl1['radius']:.2f}, r2={cyl2['radius']:.2f}"
                        )
            else:
                # Fallback: O(n²) comparison for small datasets or no scipy
                part_ids = list(part_cylinders.keys())

                for i, part1_id in enumerate(part_ids):
                    for part2_id in part_ids[i + 1:]:
                        cylinders1 = part_cylinders[part1_id]
                        cylinders2 = part_cylinders[part2_id]

                        # Check all pairs of cylinders
                        for cyl1 in cylinders1:
                            for cyl2 in cylinders2:
                                # Check if cylinders are concentric
                                if self._are_concentric_fast(cyl1, cyl2):
                                    mate = Mate(
                                        mate_id=f"concentric_{mate_id}",
                                        part1_id=part1_id,
                                        part2_id=part2_id,
                                        constraint_type=ConstraintType.CONCENTRIC,
                                        parameters={
                                            "radius1": cyl1["radius"],
                                            "radius2": cyl2["radius"],
                                            "radial_clearance": abs(cyl1["radius"] - cyl2["radius"]),
                                        },
                                        location=cyl1["center"],
                                        direction=cyl1["axis"],
                                        confidence=0.85,
                                    )
                                    mates.append(mate)
                                    mate_id += 1

                                    _log.debug(
                                        f"Found concentric mate: {part1_id} <-> {part2_id}, "
                                        f"r1={cyl1['radius']:.2f}, r2={cyl2['radius']:.2f}"
                                    )

        except Exception as e:
            _log.warning(f"Error detecting concentric mates: {e}")

        return mates

    def detect_planar_contacts(
        self, doc: CADlingDocument, item_batch: list[CADItem]
    ) -> List[Mate]:
        """Detect planar contact surfaces between parts.

        Finds pairs of planar faces that are in contact or near-contact.

        Args:
            doc: CADlingDocument containing assembly
            item_batch: List of items to analyze

        Returns:
            List of Mate objects for planar contacts
        """
        mates = []

        if not self.has_pythonocc:
            return mates

        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Plane

            # Get solid parts
            solid_items = [
                item for item in item_batch
                if hasattr(item, "item_type") and item.item_type in ["brep_solid", "step_entity"]
            ]

            if len(solid_items) < 2:
                return mates

            _log.debug(f"Detecting planar contacts among {len(solid_items)} parts")

            # Extract planar faces from each part
            part_planes = {}  # part_id -> list of (face, normal, point, area)

            for i, item in enumerate(solid_items):
                part_id = f"part_{i}"
                planes = self._extract_planar_faces(item)
                if planes:
                    part_planes[part_id] = planes

            # Pre-normalize all plane normals (avoid repeated normalization in loop)
            for part_id, planes in part_planes.items():
                for plane in planes:
                    normal = np.array(plane["normal"])
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        plane["normal_normalized"] = normal / norm
                    else:
                        plane["normal_normalized"] = normal

            # Build flat list of all planes with part references for spatial indexing
            all_planes = []
            plane_part_map = []  # Maps plane index to (part_id, local_idx)
            for part_id, planes in part_planes.items():
                for local_idx, plane in enumerate(planes):
                    all_planes.append(plane)
                    plane_part_map.append((part_id, local_idx))

            mate_id = 0

            # Use KDTree for O(n log n) spatial queries if scipy available
            if HAS_SCIPY and len(all_planes) > 10:
                # Build KDTree from plane points
                points = np.array([plane["point"] for plane in all_planes])
                tree = KDTree(points)

                # Search radius based on typical plane proximity
                search_radius = self.planar_tolerance * 100  # Generous search radius

                # Find pairs within search radius
                pairs = tree.query_pairs(r=search_radius, output_type='ndarray')

                for i, j in pairs:
                    part1_id, _ = plane_part_map[i]
                    part2_id, _ = plane_part_map[j]

                    # Skip planes from same part
                    if part1_id == part2_id:
                        continue

                    plane1 = all_planes[i]
                    plane2 = all_planes[j]

                    # Check if planes are in contact
                    if self._are_planar_contact_fast(plane1, plane2):
                        mate = Mate(
                            mate_id=f"planar_{mate_id}",
                            part1_id=part1_id,
                            part2_id=part2_id,
                            constraint_type=ConstraintType.PLANAR,
                            parameters={
                                "area1": plane1["area"],
                                "area2": plane2["area"],
                                "overlap_area": min(plane1["area"], plane2["area"]),
                            },
                            location=plane1["point"],
                            direction=plane1["normal"],
                            confidence=0.80,
                        )
                        mates.append(mate)
                        mate_id += 1

                        _log.debug(
                            f"Found planar contact: {part1_id} <-> {part2_id}, "
                            f"area1={plane1['area']:.2f}, area2={plane2['area']:.2f}"
                        )
            else:
                # Fallback: O(n²) comparison for small datasets or no scipy
                part_ids = list(part_planes.keys())

                for i, part1_id in enumerate(part_ids):
                    for part2_id in part_ids[i + 1:]:
                        planes1 = part_planes[part1_id]
                        planes2 = part_planes[part2_id]

                        # Check all pairs of planes
                        for plane1 in planes1:
                            for plane2 in planes2:
                                # Check if planes are in contact
                                if self._are_planar_contact_fast(plane1, plane2):
                                    mate = Mate(
                                        mate_id=f"planar_{mate_id}",
                                        part1_id=part1_id,
                                        part2_id=part2_id,
                                        constraint_type=ConstraintType.PLANAR,
                                        parameters={
                                            "area1": plane1["area"],
                                            "area2": plane2["area"],
                                            "overlap_area": min(plane1["area"], plane2["area"]),
                                        },
                                        location=plane1["point"],
                                        direction=plane1["normal"],
                                        confidence=0.80,
                                    )
                                    mates.append(mate)
                                    mate_id += 1

                                    _log.debug(
                                        f"Found planar contact: {part1_id} <-> {part2_id}, "
                                        f"area1={plane1['area']:.2f}, area2={plane2['area']:.2f}"
                                    )

        except Exception as e:
            _log.warning(f"Error detecting planar contacts: {e}")

        return mates

    def detect_fasteners(
        self, doc: CADlingDocument, item_batch: list[CADItem]
    ) -> List[Fastener]:
        """Detect fasteners (bolts, screws, nuts) in assembly.

        Uses geometric heuristics to identify small cylindrical parts that
        likely represent fasteners connecting larger parts.

        Args:
            doc: CADlingDocument containing assembly
            item_batch: List of items to analyze

        Returns:
            List of Fastener objects
        """
        fasteners = []

        if not self.has_pythonocc:
            return fasteners

        try:
            _log.debug(f"Detecting fasteners among {len(item_batch)} items")

            # Strategy: Look for small cylindrical parts with specific geometry
            for i, item in enumerate(item_batch):
                if not hasattr(item, "item_type"):
                    continue

                # Check geometry properties if available
                if "geometry_analysis" in getattr(item, "properties", {}):
                    geom = item.properties["geometry_analysis"]
                    volume = geom.get("volume", 0)
                    bbox = geom.get("bounding_box", {})

                    # Heuristic: Small volume and elongated shape
                    if volume < 1000:  # mm³ threshold for small parts
                        dims = [
                            bbox.get("max_x", 0) - bbox.get("min_x", 0),
                            bbox.get("max_y", 0) - bbox.get("min_y", 0),
                            bbox.get("max_z", 0) - bbox.get("min_z", 0),
                        ]
                        dims.sort()

                        # Check for elongated cylindrical shape
                        if dims[2] > 3 * dims[1]:  # Length > 3 * diameter
                            fastener_type = self._classify_fastener(item, dims)

                            fastener = Fastener(
                                fastener_id=f"fastener_{i}",
                                fastener_type=fastener_type,
                                diameter=dims[1],  # Approximate diameter
                                length=dims[2],     # Approximate length
                                location=[
                                    (bbox.get("min_x", 0) + bbox.get("max_x", 0)) / 2,
                                    (bbox.get("min_y", 0) + bbox.get("max_y", 0)) / 2,
                                    (bbox.get("min_z", 0) + bbox.get("max_z", 0)) / 2,
                                ],
                                axis=[0, 0, 1],  # Assume Z-aligned (simplification)
                                confidence=0.6,  # Lower confidence for heuristic approach
                            )

                            # Find connected parts (parts within proximity)
                            fastener.connected_parts = self._find_connected_parts(
                                item, item_batch
                            )

                            fasteners.append(fastener)

                            _log.debug(
                                f"Detected {fastener_type.value}: "
                                f"d={dims[1]:.1f}mm, L={dims[2]:.1f}mm"
                            )

        except Exception as e:
            _log.warning(f"Error detecting fasteners: {e}")

        return fasteners

    def _extract_cylindrical_faces(self, item: CADItem) -> List[Dict[str, Any]]:
        """Extract cylindrical faces from a part.

        Returns:
            List of dicts with keys: face, radius, axis, center
        """
        cylinders = []

        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Cylinder
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.gp import gp_Pnt, gp_Dir

            shape = getattr(item, "_occ_shape", None)
            if shape is None:
                return cylinders

            # Explore faces
            explorer = TopExp_Explorer(shape, TopAbs_FACE)

            while explorer.More():
                face = explorer.Current()
                explorer.Next()

                try:
                    adaptor = BRepAdaptor_Surface(face)

                    if adaptor.GetType() == GeomAbs_Cylinder:
                        # Get cylinder properties
                        cylinder = adaptor.Cylinder()
                        radius = cylinder.Radius()
                        axis_gp = cylinder.Axis()
                        location = axis_gp.Location()
                        direction = axis_gp.Direction()

                        cylinders.append({
                            "face": face,
                            "radius": radius,
                            "axis": [direction.X(), direction.Y(), direction.Z()],
                            "center": [location.X(), location.Y(), location.Z()],
                        })

                except (AttributeError, RuntimeError, TypeError) as e:
                    _log.debug("Skipping non-cylindrical or invalid face: %s", e)
                    continue

        except Exception as e:
            _log.debug(f"Error extracting cylindrical faces: {e}")

        return cylinders

    def _extract_planar_faces(self, item: CADItem) -> List[Dict[str, Any]]:
        """Extract planar faces from a part.

        Returns:
            List of dicts with keys: face, normal, point, area
        """
        planes = []

        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Plane
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop

            shape = getattr(item, "_occ_shape", None)
            if shape is None:
                return planes

            # Explore faces
            explorer = TopExp_Explorer(shape, TopAbs_FACE)

            while explorer.More():
                face = explorer.Current()
                explorer.Next()

                try:
                    adaptor = BRepAdaptor_Surface(face)

                    if adaptor.GetType() == GeomAbs_Plane:
                        # Get plane properties
                        plane_surf = adaptor.Plane()
                        location = plane_surf.Location()
                        direction = plane_surf.Axis().Direction()

                        # Compute face area
                        props = GProp_GProps()
                        brepgprop.SurfaceProperties(face, props)
                        area = props.Mass()

                        planes.append({
                            "face": face,
                            "normal": [direction.X(), direction.Y(), direction.Z()],
                            "point": [location.X(), location.Y(), location.Z()],
                            "area": area,
                        })

                except (AttributeError, RuntimeError, TypeError) as e:
                    _log.debug("Skipping non-planar or invalid face: %s", e)
                    continue

        except Exception as e:
            _log.debug(f"Error extracting planar faces: {e}")

        return planes

    def _are_concentric(self, cyl1: Dict, cyl2: Dict) -> bool:
        """Check if two cylinders are concentric.

        Args:
            cyl1, cyl2: Cylinder dicts with radius, axis, center

        Returns:
            True if cylinders are concentric within tolerance
        """
        # Check if axes are parallel
        axis1 = np.array(cyl1["axis"])
        axis2 = np.array(cyl2["axis"])

        # Normalize axes
        axis1 = axis1 / np.linalg.norm(axis1)
        axis2 = axis2 / np.linalg.norm(axis2)

        # Check parallelism (dot product close to ±1)
        dot_product = abs(np.dot(axis1, axis2))
        if dot_product < 0.95:  # ~18 degrees tolerance
            return False

        # Check if centers are aligned radially
        center1 = np.array(cyl1["center"])
        center2 = np.array(cyl2["center"])

        # Vector from center1 to center2
        center_vec = center2 - center1

        # Use first axis for projection (they're parallel)
        # Project center_vec onto axis to get axial component
        axial_component = np.dot(center_vec, axis1) * axis1

        # Subtract axial to get radial component
        radial_component = center_vec - axial_component

        # Radial distance
        radial_dist = np.linalg.norm(radial_component)

        # Check radial alignment
        if radial_dist > self.concentric_tolerance:
            return False

        return True

    def _are_concentric_fast(self, cyl1: Dict, cyl2: Dict) -> bool:
        """Check if two cylinders are concentric (optimized version).

        Uses pre-normalized axes stored in 'axis_normalized' key to avoid
        repeated normalization. Falls back to regular _are_concentric if
        pre-normalized axes not available.

        Args:
            cyl1, cyl2: Cylinder dicts with radius, axis, center, axis_normalized

        Returns:
            True if cylinders are concentric within tolerance
        """
        # Use pre-normalized axes if available
        axis1 = cyl1.get("axis_normalized")
        axis2 = cyl2.get("axis_normalized")

        if axis1 is None or axis2 is None:
            # Fall back to standard method
            return self._are_concentric(cyl1, cyl2)

        # Check parallelism (dot product close to ±1)
        dot_product = abs(np.dot(axis1, axis2))
        if dot_product < 0.95:  # ~18 degrees tolerance
            return False

        # Check if centers are aligned radially
        center1 = np.array(cyl1["center"])
        center2 = np.array(cyl2["center"])

        # Vector from center1 to center2
        center_vec = center2 - center1

        # Use first axis for projection (they're parallel)
        # Project center_vec onto axis to get axial component
        axial_component = np.dot(center_vec, axis1) * axis1

        # Subtract axial to get radial component
        radial_component = center_vec - axial_component

        # Radial distance
        radial_dist = np.linalg.norm(radial_component)

        # Check radial alignment
        if radial_dist > self.concentric_tolerance:
            return False

        return True

    def _are_planar_contact(self, plane1: Dict, plane2: Dict) -> bool:
        """Check if two planes are in contact.

        Args:
            plane1, plane2: Plane dicts with normal, point, area

        Returns:
            True if planes are in contact within tolerance
        """
        # Check if normals are anti-parallel (facing each other)
        normal1 = np.array(plane1["normal"])
        normal2 = np.array(plane2["normal"])

        # Normalize
        normal1 = normal1 / np.linalg.norm(normal1)
        normal2 = normal2 / np.linalg.norm(normal2)

        # Check anti-parallelism (dot product close to -1)
        dot_product = np.dot(normal1, normal2)
        if dot_product > -0.95:  # Not facing each other
            return False

        # Check if planes are co-planar (points lie on each other's plane)
        point1 = np.array(plane1["point"])
        point2 = np.array(plane2["point"])

        # Distance from point2 to plane1
        dist = abs(np.dot(point2 - point1, normal1))

        if dist > self.planar_tolerance:
            return False

        # Check minimum area overlap
        min_area = min(plane1["area"], plane2["area"])
        if min_area < self.min_contact_area:
            return False

        return True

    def _are_planar_contact_fast(self, plane1: Dict, plane2: Dict) -> bool:
        """Check if two planes are in contact (optimized version).

        Uses pre-normalized normals stored in 'normal_normalized' key to avoid
        repeated normalization. Falls back to regular _are_planar_contact if
        pre-normalized normals not available.

        Args:
            plane1, plane2: Plane dicts with normal, point, area, normal_normalized

        Returns:
            True if planes are in contact within tolerance
        """
        # Use pre-normalized normals if available
        normal1 = plane1.get("normal_normalized")
        normal2 = plane2.get("normal_normalized")

        if normal1 is None or normal2 is None:
            # Fall back to standard method
            return self._are_planar_contact(plane1, plane2)

        # Check anti-parallelism (dot product close to -1)
        dot_product = np.dot(normal1, normal2)
        if dot_product > -0.95:  # Not facing each other
            return False

        # Check if planes are co-planar (points lie on each other's plane)
        point1 = np.array(plane1["point"])
        point2 = np.array(plane2["point"])

        # Distance from point2 to plane1
        dist = abs(np.dot(point2 - point1, normal1))

        if dist > self.planar_tolerance:
            return False

        # Check minimum area overlap
        min_area = min(plane1["area"], plane2["area"])
        if min_area < self.min_contact_area:
            return False

        return True

    def _classify_fastener(self, item: CADItem, dims: List[float]) -> FastenerType:
        """Classify fastener type based on geometry.

        Args:
            item: CAD item
            dims: Bounding box dimensions [width, width, length]

        Returns:
            FastenerType enum value
        """
        # Simple heuristic based on aspect ratio and size
        aspect_ratio = dims[2] / dims[1] if dims[1] > 0 else 0

        # Very elongated -> bolt/screw
        if aspect_ratio > 5:
            if dims[1] < 3:  # Small diameter
                return FastenerType.SCREW
            else:
                return FastenerType.BOLT

        # Moderate aspect ratio with small diameter -> pin
        elif aspect_ratio > 2 and dims[1] < 5:
            return FastenerType.PIN

        # Short and wide -> washer or nut
        elif aspect_ratio < 1.5:
            if dims[2] < 5:  # Thin
                return FastenerType.WASHER
            else:
                return FastenerType.NUT

        return FastenerType.UNKNOWN

    def _find_connected_parts(
        self, fastener_item: CADItem, all_items: list[CADItem]
    ) -> List[str]:
        """Find parts connected by a fastener.

        Args:
            fastener_item: Fastener item
            all_items: All items in assembly

        Returns:
            List of part IDs connected by this fastener
        """
        connected = []

        # Simple proximity check
        if "geometry_analysis" not in getattr(fastener_item, "properties", {}):
            return connected

        fastener_geom = fastener_item.properties["geometry_analysis"]
        fastener_centroid = fastener_geom.get("centroid", [0, 0, 0])

        for i, item in enumerate(all_items):
            if item is fastener_item:
                continue

            if "geometry_analysis" in getattr(item, "properties", {}):
                item_geom = item.properties["geometry_analysis"]
                item_centroid = item_geom.get("centroid", [0, 0, 0])

                # Compute distance
                dist = np.linalg.norm(
                    np.array(fastener_centroid) - np.array(item_centroid)
                )

                # If within reasonable distance, consider connected
                if dist < 50:  # mm threshold (arbitrary)
                    connected.append(f"part_{i}")

        return connected
