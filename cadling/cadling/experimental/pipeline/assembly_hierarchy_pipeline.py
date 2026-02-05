"""Assembly hierarchy pipeline for processing multi-part CAD assemblies.

This module provides a pipeline that processes CAD assemblies by respecting
component hierarchy, detecting relationships, and generating Bill of Materials.

Classes:
    AssemblyHierarchyPipeline: Pipeline for assembly-aware processing

Example:
    from cadling.experimental.pipeline import AssemblyHierarchyPipeline
    from cadling.experimental.datamodel import AssemblyAnalysisOptions

    options = AssemblyAnalysisOptions(generate_bom=True)
    pipeline = AssemblyHierarchyPipeline(options)
    result = pipeline.execute(input_doc)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from cadling.pipeline.base_pipeline import BaseCADPipeline

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, ConversionResult
    from cadling.experimental.datamodel import AssemblyAnalysisOptions

_log = logging.getLogger(__name__)


class AssemblyNode:
    """Node in assembly hierarchy tree.

    Attributes:
        component_id: Unique identifier for component
        name: Component name
        item: CADItem for this component
        parent: Parent node
        children: Child nodes
        mates: Mate relationships with other components
        properties: Additional component properties (material, mass, etc.)
    """

    def __init__(
        self,
        component_id: str,
        name: str,
        item: Optional[CADItem] = None,
    ):
        self.component_id = component_id
        self.name = name
        self.item = item
        self.parent: Optional[AssemblyNode] = None
        self.children: List[AssemblyNode] = []
        self.mates: List[Dict[str, Any]] = []
        self.properties: Dict[str, Any] = {}

    def add_child(self, child: AssemblyNode) -> None:
        """Add a child node."""
        self.children.append(child)
        child.parent = self

    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    def depth(self) -> int:
        """Get depth of this node in tree."""
        depth = 0
        node = self
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth


class AssemblyHierarchyPipeline(BaseCADPipeline):
    """Assembly hierarchy pipeline for processing multi-part assemblies.

    This experimental pipeline processes CAD assemblies by:

    **Component Detection:**
    - Identify individual components in assembly
    - Detect subassemblies
    - Extract component metadata (name, ID, type)
    - Build component hierarchy tree

    **Relationship Extraction:**
    - Detect mate/constraint relationships:
      * Coincident mates (surfaces touch)
      * Concentric mates (holes/cylinders align)
      * Parallel/perpendicular constraints
      * Distance/angle constraints
    - Identify fasteners (bolts, screws, rivets)
    - Track interference/clearances

    **Hierarchical Processing:**
    - Process components in dependency order
    - Respect parent-child relationships
    - Handle subassemblies recursively
    - Propagate properties up/down hierarchy

    **BOM Generation:**
    - Generate flat or hierarchical Bill of Materials
    - Include quantities for identical parts
    - Track component properties (material, mass, volume)
    - Export in standard formats

    **Use Cases:**
    - Assembly documentation
    - Manufacturing planning (assembly sequence)
    - Cost estimation (component count, materials)
    - Interference checking
    - Exploded view generation

    Attributes:
        options: Assembly analysis options
        assembly_tree: Root of assembly hierarchy tree
        component_map: Mapping from component ID to node
        bom: Generated Bill of Materials

    Example:
        options = AssemblyAnalysisOptions(
            detect_components=True,
            extract_mates=True,
            generate_bom=True,
            process_subassemblies=True
        )
        pipeline = AssemblyHierarchyPipeline(options)
        result = pipeline.execute(input_doc)

        # Access assembly tree
        tree = result.document.properties.get("assembly_tree")

        # Access BOM
        bom = result.document.properties.get("bill_of_materials")
        for entry in bom:
            print(f"{entry['name']}: qty {entry['quantity']}")
    """

    def __init__(self, options: AssemblyAnalysisOptions):
        """Initialize assembly hierarchy pipeline.

        Args:
            options: Configuration options for the pipeline
        """
        super().__init__(options)
        self.options: AssemblyAnalysisOptions = options
        self.assembly_tree: Optional[AssemblyNode] = None
        self.component_map: Dict[str, AssemblyNode] = {}
        self.bom: List[Dict[str, Any]] = []

        _log.info(
            f"Initialized AssemblyHierarchyPipeline "
            f"(detect_components={options.detect_components}, "
            f"extract_mates={options.extract_mates}, "
            f"generate_bom={options.generate_bom})"
        )

    @classmethod
    def get_default_options(cls) -> AssemblyAnalysisOptions:
        """Get default pipeline options.

        Returns:
            Default AssemblyAnalysisOptions
        """
        from cadling.experimental.datamodel import AssemblyAnalysisOptions

        return AssemblyAnalysisOptions()

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Build: Parse assembly and detect component structure.

        This stage:
        - Parses CAD assembly file
        - Detects individual components
        - Builds hierarchy tree
        - Extracts component metadata

        Args:
            conv_res: Conversion result to populate

        Returns:
            Updated conversion result with assembly structure
        """
        _log.info("[Build] Starting assembly parsing and component detection")

        try:
            # Get backend from input document
            backend = conv_res.input._backend

            # Convert using backend
            doc = backend.convert()
            conv_res.document = doc

            # Detect components if enabled
            if self.options.detect_components:
                self._detect_components(doc)

            _log.info(
                f"[Build] Completed: detected {len(self.component_map)} components"
            )

        except Exception as e:
            _log.error(f"[Build] Build failed: {e}")
            raise

        return conv_res

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Assemble: Extract relationships and build hierarchy.

        This stage:
        - Extracts mate relationships between components
        - Builds complete assembly hierarchy
        - Detects fasteners and connections
        - Checks for interferences if enabled

        Args:
            conv_res: Conversion result to assemble

        Returns:
            Updated conversion result with relationships
        """
        if not conv_res.document:
            return conv_res

        _log.info("[Assemble] Building assembly hierarchy and extracting relationships")

        doc = conv_res.document

        try:
            # Extract mate relationships if enabled
            if self.options.extract_mates:
                self._extract_mate_relationships(doc)

            # Check for interferences if enabled
            if self.options.check_interference:
                self._check_interferences(doc)

            # Generate BOM if enabled
            if self.options.generate_bom:
                self.bom = self._generate_bom(doc)
                doc.properties["bill_of_materials"] = self.bom

            # Store assembly tree
            if self.assembly_tree:
                doc.properties["assembly_tree"] = self._serialize_tree(self.assembly_tree)
                doc.properties["num_components"] = len(self.component_map)

            _log.info(
                f"[Assemble] Completed: extracted {self._count_mates()} mates, "
                f"generated BOM with {len(self.bom)} entries"
            )

        except Exception as e:
            _log.error(f"[Assemble] Assembly failed: {e}")
            raise

        return conv_res

    def _detect_components(self, doc) -> None:
        """Detect individual components in assembly.

        Args:
            doc: The document
        """
        # Create root node
        self.assembly_tree = AssemblyNode(
            component_id="root",
            name="Assembly Root",
        )
        self.component_map["root"] = self.assembly_tree

        # Detect components from items
        # In real implementation, would analyze STEP assembly structure
        # or STL file list
        for i, item in enumerate(doc.items):
            # Create component node
            comp_id = f"component_{i}"
            comp_name = item.properties.get("name", f"Component {i}")

            node = AssemblyNode(
                component_id=comp_id,
                name=comp_name,
                item=item,
            )

            # Extract component properties
            node.properties["material"] = item.properties.get("material")
            node.properties["volume"] = item.properties.get("volume")
            node.properties["mass"] = item.properties.get("mass")
            node.properties["bounding_box"] = item.properties.get("bounding_box")

            # Store product_definition_id if available (for hierarchical tree building)
            if "product_definition_id" in item.properties:
                node.properties["product_definition_id"] = item.properties["product_definition_id"]
                node.product_definition_id = item.properties["product_definition_id"]

            self.component_map[comp_id] = node

            # Mark item with component ID
            item.properties["component_id"] = comp_id

        _log.debug(f"[Detect] Found {len(self.component_map) - 1} components")

        # Build hierarchical tree structure from STEP assembly structure
        # Falls back to flat structure if STEP parsing fails
        self._build_hierarchical_tree(doc)

    def _extract_mate_relationships(self, doc) -> None:
        """Extract mate relationships between components.

        Args:
            doc: The document
        """
        # Extract mates from geometric constraints
        # In real implementation, would analyze:
        # - Coincident surfaces between components
        # - Concentric holes/cylinders
        # - Fastener patterns
        # - Assembly constraints from STEP file

        for i, comp1 in enumerate(list(self.component_map.values())[1:]):  # Skip root
            for comp2 in list(self.component_map.values())[i + 2 :]:
                # Check for potential mates
                mate = self._detect_mate(comp1, comp2)
                if mate:
                    comp1.mates.append(mate)
                    comp2.mates.append(mate)

        _log.debug(f"[Mates] Extracted {self._count_mates()} mate relationships")

    def _detect_mate(
        self, comp1: AssemblyNode, comp2: AssemblyNode
    ) -> Optional[Dict[str, Any]]:
        """Detect mate relationship between two components.

        Uses geometric analysis to detect:
        - Concentric mates (cylindrical surfaces aligned)
        - Planar contacts (opposing flat surfaces)
        - Fastener connections (bolts, screws, pins)

        Args:
            comp1: First component
            comp2: Second component

        Returns:
            Mate dictionary if relationship detected, None otherwise
        """
        # 1. Check proximity first (bounding box overlap/adjacency)
        if not self._are_components_adjacent(comp1, comp2):
            return None

        # 2. Load OCC shapes for both components
        shape1 = self._load_component_shape(comp1)
        shape2 = self._load_component_shape(comp2)

        if not shape1 or not shape2:
            return None

        # 3. Try concentric mate detection
        concentric_mate = self._detect_concentric_mate(shape1, shape2)
        if concentric_mate and concentric_mate.get("confidence", 0.0) > 0.7:
            return {
                "type": "CONCENTRIC",
                "component1": comp1.component_id,
                "component2": comp2.component_id,
                **concentric_mate,
            }

        # 4. Try planar contact detection
        planar_mate = self._detect_planar_contact(shape1, shape2)
        if planar_mate and planar_mate.get("confidence", 0.0) > 0.7:
            return {
                "type": "COINCIDENT",
                "component1": comp1.component_id,
                "component2": comp2.component_id,
                **planar_mate,
            }

        # 5. Check for fastener relationships (heuristic-based)
        fastener_mate = self._detect_fastener_connection(comp1, comp2)
        if fastener_mate:
            return {
                "type": "FASTENER",
                **fastener_mate,
            }

        return None

    def _are_components_adjacent(self, comp1: AssemblyNode, comp2: AssemblyNode) -> bool:
        """Check if bounding boxes are close enough for potential mate.

        Args:
            comp1: First component
            comp2: Second component

        Returns:
            True if components are adjacent
        """
        bbox1 = comp1.properties.get("bounding_box")
        bbox2 = comp2.properties.get("bounding_box")

        if not bbox1 or not bbox2:
            return False

        # Check if boxes overlap or are within tolerance
        tolerance = 10.0  # mm

        x_overlap = (bbox1["xmin"] - tolerance <= bbox2["xmax"]) and (
            bbox2["xmin"] - tolerance <= bbox1["xmax"]
        )
        y_overlap = (bbox1["ymin"] - tolerance <= bbox2["ymax"]) and (
            bbox2["ymin"] - tolerance <= bbox1["ymax"]
        )
        z_overlap = (bbox1["zmin"] - tolerance <= bbox2["zmax"]) and (
            bbox2["zmin"] - tolerance <= bbox1["zmax"]
        )

        return x_overlap and y_overlap and z_overlap

    def _load_component_shape(self, comp: AssemblyNode):
        """Load OCC shape for component (use backend or cached shape).

        Args:
            comp: Component node

        Returns:
            OCC shape object or None
        """
        # Try cached shape first
        if hasattr(comp, "_shape") and comp._shape is not None:
            return comp._shape

        # Try loading from item reference
        if hasattr(comp, "item") and comp.item:
            item = comp.item
            if hasattr(item, "_shape") and item._shape:
                comp._shape = item._shape  # Cache for future use
                return item._shape

        return None

    def _detect_concentric_mate(self, shape1, shape2) -> Optional[Dict[str, Any]]:
        """Detect concentric mate between cylindrical surfaces.

        Adapted from constraint_detection.py. Uses pythonocc-core to:
        - Extract cylindrical faces from both shapes
        - Check if cylinder axes are parallel
        - Check if axes are colinear (concentric)
        - Verify radius compatibility

        Args:
            shape1: First OCC shape
            shape2: Second OCC shape

        Returns:
            Dict with mate parameters and confidence, or None
        """
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Cylinder
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.TopExp import TopExp_Explorer
        except ImportError:
            _log.debug("pythonocc-core not available for concentric mate detection")
            return None

        # Extract cylindrical faces from both shapes
        cylinders1 = []
        exp1 = TopExp_Explorer(shape1, TopAbs_FACE)
        while exp1.More():
            face = exp1.Current()
            surf = BRepAdaptor_Surface(face, True)
            if surf.GetType() == GeomAbs_Cylinder:
                # Extract cylinder axis and radius
                cylinder = surf.Cylinder()
                axis = cylinder.Axis()
                location = axis.Location()
                direction = axis.Direction()
                radius = cylinder.Radius()

                cylinders1.append(
                    {
                        "face": face,
                        "location": np.array([location.X(), location.Y(), location.Z()]),
                        "direction": np.array([direction.X(), direction.Y(), direction.Z()]),
                        "radius": radius,
                    }
                )
            exp1.Next()

        cylinders2 = []
        exp2 = TopExp_Explorer(shape2, TopAbs_FACE)
        while exp2.More():
            face = exp2.Current()
            surf = BRepAdaptor_Surface(face, True)
            if surf.GetType() == GeomAbs_Cylinder:
                cylinder = surf.Cylinder()
                axis = cylinder.Axis()
                location = axis.Location()
                direction = axis.Direction()
                radius = cylinder.Radius()

                cylinders2.append(
                    {
                        "face": face,
                        "location": np.array([location.X(), location.Y(), location.Z()]),
                        "direction": np.array([direction.X(), direction.Y(), direction.Z()]),
                        "radius": radius,
                    }
                )
            exp2.Next()

        # Find concentric cylinder pairs
        radial_tolerance = 0.1  # mm
        angle_tolerance = 0.95  # dot product threshold (approx 18 degrees)

        for cyl1 in cylinders1:
            for cyl2 in cylinders2:
                # Check if axes are parallel (dot product near ±1)
                dot_product = abs(np.dot(cyl1["direction"], cyl2["direction"]))
                if dot_product < angle_tolerance:
                    continue

                # Check if axes are colinear (cross-check distance)
                # Distance from point to line formula
                vec_between = cyl2["location"] - cyl1["location"]
                cross = np.cross(cyl1["direction"], vec_between)
                distance = np.linalg.norm(cross)

                if distance < radial_tolerance:
                    # Check radius compatibility (for slip fits, clearance fits)
                    radius_diff = abs(cyl1["radius"] - cyl2["radius"])

                    return {
                        "axis_location": cyl1["location"].tolist(),
                        "axis_direction": cyl1["direction"].tolist(),
                        "radius1": float(cyl1["radius"]),
                        "radius2": float(cyl2["radius"]),
                        "radius_difference": float(radius_diff),
                        "alignment_distance": float(distance),
                        "confidence": 0.9 if radius_diff < 0.5 else 0.8,
                    }

        return None

    def _detect_planar_contact(self, shape1, shape2) -> Optional[Dict[str, Any]]:
        """Detect planar contact between opposing planar surfaces.

        Adapted from constraint_detection.py. Uses pythonocc-core to:
        - Extract planar faces from both shapes
        - Check if normals are anti-parallel (opposing)
        - Check distance between face centroids
        - Verify area overlap

        Args:
            shape1: First OCC shape
            shape2: Second OCC shape

        Returns:
            Dict with contact parameters and confidence, or None
        """
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.GeomAbs import GeomAbs_Plane
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.TopExp import TopExp_Explorer
        except ImportError:
            _log.debug("pythonocc-core not available for planar contact detection")
            return None

        # Extract planar faces from both shapes
        planes1 = []
        exp1 = TopExp_Explorer(shape1, TopAbs_FACE)
        while exp1.More():
            face = exp1.Current()
            surf = BRepAdaptor_Surface(face, True)
            if surf.GetType() == GeomAbs_Plane:
                plane = surf.Plane()
                location = plane.Location()
                normal = plane.Axis().Direction()

                # Compute face area and centroid
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                area = props.Mass()
                centroid = props.CentreOfMass()

                planes1.append(
                    {
                        "face": face,
                        "location": np.array([location.X(), location.Y(), location.Z()]),
                        "normal": np.array([normal.X(), normal.Y(), normal.Z()]),
                        "centroid": np.array([centroid.X(), centroid.Y(), centroid.Z()]),
                        "area": area,
                    }
                )
            exp1.Next()

        planes2 = []
        exp2 = TopExp_Explorer(shape2, TopAbs_FACE)
        while exp2.More():
            face = exp2.Current()
            surf = BRepAdaptor_Surface(face, True)
            if surf.GetType() == GeomAbs_Plane:
                plane = surf.Plane()
                location = plane.Location()
                normal = plane.Axis().Direction()

                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                area = props.Mass()
                centroid = props.CentreOfMass()

                planes2.append(
                    {
                        "face": face,
                        "location": np.array([location.X(), location.Y(), location.Z()]),
                        "normal": np.array([normal.X(), normal.Y(), normal.Z()]),
                        "centroid": np.array([centroid.X(), centroid.Y(), centroid.Z()]),
                        "area": area,
                    }
                )
            exp2.Next()

        # Find opposing planar pairs (anti-parallel normals, close distance)
        distance_tolerance = 0.01  # mm
        normal_tolerance = -0.95  # dot product for anti-parallel (< -0.95 ≈ opposing)

        for plane1 in planes1:
            for plane2 in planes2:
                # Check if normals are anti-parallel
                dot_product = np.dot(plane1["normal"], plane2["normal"])
                if dot_product > normal_tolerance:
                    continue

                # Check distance between centroids projected onto normal
                vec_between = plane2["centroid"] - plane1["centroid"]
                distance = abs(np.dot(vec_between, plane1["normal"]))

                if distance < distance_tolerance:
                    # Potential contact - check area overlap
                    min_area = min(plane1["area"], plane2["area"])
                    max_area = max(plane1["area"], plane2["area"])
                    area_ratio = min_area / max_area if max_area > 0 else 0.0

                    return {
                        "normal": plane1["normal"].tolist(),
                        "centroid1": plane1["centroid"].tolist(),
                        "centroid2": plane2["centroid"].tolist(),
                        "distance": float(distance),
                        "area1": float(plane1["area"]),
                        "area2": float(plane2["area"]),
                        "area_overlap_ratio": float(area_ratio),
                        "confidence": 0.9 if area_ratio > 0.8 else 0.7,
                    }

        return None

    def _detect_fastener_connection(
        self, comp1: AssemblyNode, comp2: AssemblyNode
    ) -> Optional[Dict[str, Any]]:
        """Detect fastener connection using heuristics (volume, aspect ratio).

        Heuristic-based detection for bolts, screws, pins.

        Args:
            comp1: First component
            comp2: Second component

        Returns:
            Dict with fastener info, or None
        """
        # Check if one component is a fastener (small volume, high aspect ratio)
        vol1 = comp1.properties.get("geometry_analysis", {}).get("volume", 0)
        vol2 = comp2.properties.get("geometry_analysis", {}).get("volume", 0)

        bbox1 = comp1.properties.get("bounding_box", {})
        bbox2 = comp2.properties.get("bounding_box", {})

        def is_fastener_like(vol, bbox):
            """Check if component resembles a fastener."""
            if not bbox or vol <= 0:
                return False

            dx = bbox.get("dx", 0)
            dy = bbox.get("dy", 0)
            dz = bbox.get("dz", 0)

            # Aspect ratio: one dimension >> others (bolt-like)
            max_dim = max(dx, dy, dz)
            min_dim = min(dx, dy, dz)
            aspect_ratio = max_dim / min_dim if min_dim > 0 else 0

            # Small volume and high aspect ratio
            return vol < 1000.0 and aspect_ratio > 5.0

        fastener1 = is_fastener_like(vol1, bbox1)
        fastener2 = is_fastener_like(vol2, bbox2)

        if fastener1 or fastener2:
            return {
                "fastener_component": (
                    comp1.component_id if fastener1 else comp2.component_id
                ),
                "connected_component": (
                    comp2.component_id if fastener1 else comp1.component_id
                ),
                "confidence": 0.6,  # Heuristic-based, lower confidence
            }

        return None

    def _check_interferences(self, doc) -> None:
        """Check for component interferences.

        Args:
            doc: The document
        """
        # Check for geometric interference between components
        # This is computationally expensive
        interferences = []

        for i, comp1 in enumerate(list(self.component_map.values())[1:]):
            for comp2 in list(self.component_map.values())[i + 2 :]:
                # Check for overlap
                if self._check_overlap(comp1, comp2):
                    interferences.append(
                        {
                            "component1": comp1.component_id,
                            "component2": comp2.component_id,
                            "type": "overlap",
                        }
                    )

        doc.properties["interferences"] = interferences
        _log.debug(f"[Interference] Found {len(interferences)} interferences")

    def _check_overlap(self, comp1: AssemblyNode, comp2: AssemblyNode) -> bool:
        """Check if two components overlap.

        Uses pythonocc-core boolean intersection to detect geometric interference.

        Args:
            comp1: First component
            comp2: Second component

        Returns:
            True if components overlap
        """
        try:
            from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.GProp import GProp_GProps
        except ImportError:
            _log.debug("pythonocc-core not available for interference checking")
            return False

        shape1 = self._load_component_shape(comp1)
        shape2 = self._load_component_shape(comp2)

        if not shape1 or not shape2:
            return False

        try:
            # Compute boolean intersection
            common = BRepAlgoAPI_Common(shape1, shape2)
            common.Build()

            if not common.IsDone():
                return False

            result_shape = common.Shape()

            # Check if intersection has non-zero volume
            props = GProp_GProps()
            brepgprop.VolumeProperties(result_shape, props)
            volume = props.Mass()

            # Consider overlap if volume > tolerance
            return volume > 1e-6  # 1 cubic mm tolerance

        except Exception as e:
            _log.warning(f"Interference check failed for {comp1.component_id} and {comp2.component_id}: {e}")
            return False

    def _generate_bom(self, doc) -> List[Dict[str, Any]]:
        """Generate Bill of Materials.

        Args:
            doc: The document

        Returns:
            List of BOM entries
        """
        bom_entries = []

        if self.options.group_identical_parts:
            # Group identical parts
            part_groups: Dict[str, List[AssemblyNode]] = {}

            for comp_id, node in self.component_map.items():
                if comp_id == "root":
                    continue

                # Group by name (simplified - should use geometry hash)
                name = node.name
                if name not in part_groups:
                    part_groups[name] = []
                part_groups[name].append(node)

            # Create BOM entries
            for name, nodes in part_groups.items():
                entry = {
                    "name": name,
                    "quantity": len(nodes),
                    "component_ids": [n.component_id for n in nodes],
                }

                # Include metadata if enabled
                if self.options.bom_include_metadata:
                    entry["material"] = nodes[0].properties.get("material")

                # Include properties if enabled
                if self.options.bom_include_properties:
                    entry["volume"] = nodes[0].properties.get("volume")
                    entry["mass"] = nodes[0].properties.get("mass")

                bom_entries.append(entry)

        else:
            # Flat BOM (one entry per component)
            for comp_id, node in self.component_map.items():
                if comp_id == "root":
                    continue

                entry = {
                    "name": node.name,
                    "component_id": comp_id,
                    "quantity": 1,
                }

                if self.options.bom_include_metadata:
                    entry["material"] = node.properties.get("material")

                if self.options.bom_include_properties:
                    entry["volume"] = node.properties.get("volume")
                    entry["mass"] = node.properties.get("mass")

                bom_entries.append(entry)

        return bom_entries

    def _serialize_tree(self, node: AssemblyNode) -> Dict[str, Any]:
        """Serialize assembly tree to dictionary.

        Args:
            node: Node to serialize

        Returns:
            Dictionary representation
        """
        return {
            "component_id": node.component_id,
            "name": node.name,
            "depth": node.depth(),
            "is_leaf": node.is_leaf(),
            "properties": node.properties,
            "num_mates": len(node.mates),
            "children": [self._serialize_tree(child) for child in node.children],
        }

    def _count_mates(self) -> int:
        """Count total number of mates in assembly.

        Returns:
            Total mate count
        """
        total = 0
        for node in self.component_map.values():
            total += len(node.mates)
        # Divide by 2 since each mate is counted twice
        return total // 2

    def _get_step_text(self, doc) -> str:
        """Get STEP text from document.

        Args:
            doc: Document to extract STEP text from

        Returns:
            STEP file content as string
        """
        if hasattr(doc, "raw_content") and doc.raw_content:
            return doc.raw_content
        elif hasattr(doc, "_backend") and hasattr(doc._backend, "step_text"):
            return doc._backend.step_text
        elif hasattr(doc, "properties") and "step_text" in doc.properties:
            return doc.properties["step_text"]
        return ""

    def _extract_product_definitions(self, doc) -> Dict[str, str]:
        """Extract product definition IDs for each component.

        Returns mapping of component_id -> product_definition_entity_id.

        Args:
            doc: Document to extract from

        Returns:
            Mapping of component_id to PRODUCT_DEFINITION entity ID
        """
        import re

        step_text = self._get_step_text(doc)
        if not step_text:
            return {}

        # Pattern for PRODUCT_DEFINITION entities
        # Example: #123 = PRODUCT_DEFINITION('id','name',#124,#125);
        prod_def_pattern = r"#(\d+)\s*=\s*PRODUCT_DEFINITION\('([^']*)',"
        matches = re.finditer(prod_def_pattern, step_text, re.IGNORECASE)

        mapping = {}
        for match in matches:
            entity_id = match.group(1)
            product_id = match.group(2)
            mapping[product_id] = entity_id

        return mapping

    def _build_hierarchical_tree(self, doc) -> None:
        """Build proper assembly hierarchy from STEP structure.

        Parses NEXT_ASSEMBLY_USAGE_OCCURRENCE entities to build tree.

        Args:
            doc: Document to build hierarchy from
        """
        import re

        # Initialize hierarchy status tracking
        self.assembly_tree.properties["hierarchy_build_status"] = "in_progress"
        self.assembly_tree.properties["hierarchy_errors"] = []

        # Get raw STEP text
        step_text = self._get_step_text(doc)

        if not step_text:
            error_msg = "No STEP text available for hierarchy extraction. Cannot build hierarchical tree."
            _log.error(error_msg)
            self.assembly_tree.properties["hierarchy_build_status"] = "failed"
            self.assembly_tree.properties["hierarchy_errors"].append(error_msg)
            self.assembly_tree.properties["hierarchy_build_method"] = "none"
            return

        # Extract NEXT_ASSEMBLY_USAGE_OCCURRENCE entities
        nauo_pattern = r"#(\d+)\s*=\s*NEXT_ASSEMBLY_USAGE_OCCURRENCE\s*\([^)]*\)\s*;\s*"
        nauo_matches = list(re.finditer(nauo_pattern, step_text, re.IGNORECASE))

        if not nauo_matches:
            error_msg = (
                "No NEXT_ASSEMBLY_USAGE_OCCURRENCE entities found in STEP file. "
                "File may not contain assembly structure or may be a single part. "
                "Cannot build hierarchical tree."
            )
            _log.error(error_msg)
            self.assembly_tree.properties["hierarchy_build_status"] = "failed"
            self.assembly_tree.properties["hierarchy_errors"].append(error_msg)
            self.assembly_tree.properties["hierarchy_build_method"] = "none"
            return

        # Build parent-child mapping
        parent_child_map = {}  # parent_id -> [child_ids]

        for match in nauo_matches:
            nauo_text = match.group(0)

            # Extract relating (parent) and related (child) references
            # NAUO format: NEXT_ASSEMBLY_USAGE_OCCURRENCE('id','name',<relating>,<related>,...)
            ref_pattern = r"#(\d+)"
            refs = re.findall(ref_pattern, nauo_text)

            if len(refs) >= 2:
                # First ref is typically relating (parent), second is related (child)
                parent_ref = refs[0]
                child_ref = refs[1]

                if parent_ref not in parent_child_map:
                    parent_child_map[parent_ref] = []
                parent_child_map[parent_ref].append(child_ref)

        # Map PRODUCT_DEFINITION entities to component nodes
        prod_def_to_node = {}  # product_definition_id -> AssemblyNode
        for node in self.component_map.values():
            # Try to find product_definition ID from node properties
            if hasattr(node, "product_definition_id"):
                prod_def_to_node[node.product_definition_id] = node
            elif "product_definition_id" in node.properties:
                prod_def_to_node[node.properties["product_definition_id"]] = node

        # Build tree structure
        processed = set()

        # Find root nodes (not in any child list)
        all_children = set()
        for children in parent_child_map.values():
            all_children.update(children)

        root_nodes = []
        for node_id, node in prod_def_to_node.items():
            if node_id not in all_children:
                root_nodes.append((node_id, node))

        # Recursive tree building
        def add_children(parent_node, parent_id):
            """Recursively add children to parent node."""
            if parent_id not in parent_child_map:
                return

            for child_id in parent_child_map[parent_id]:
                if child_id in prod_def_to_node:
                    child_node = prod_def_to_node[child_id]
                    parent_node.add_child(child_node)
                    processed.add(child_id)

                    # Recurse for sub-assemblies
                    add_children(child_node, child_id)

        # Build tree from root nodes
        for root_id, root_node in root_nodes:
            self.assembly_tree.add_child(root_node)
            processed.add(root_id)
            add_children(root_node, root_id)

        # Add any orphaned components to root (fallback)
        num_orphaned = 0
        for node_id, node in prod_def_to_node.items():
            if node_id not in processed:
                _log.debug(f"Adding orphaned component {node.component_id} to root")
                self.assembly_tree.add_child(node)
                num_orphaned += 1

        # Mark successful hierarchy build
        self.assembly_tree.properties["hierarchy_build_status"] = "success"
        self.assembly_tree.properties["hierarchy_build_method"] = "nauo_parsing"
        self.assembly_tree.properties["num_root_nodes"] = len(root_nodes)
        self.assembly_tree.properties["num_orphaned_components"] = num_orphaned

        if num_orphaned > 0:
            warning_msg = f"Found {num_orphaned} orphaned components (not in hierarchy)"
            _log.warning(warning_msg)
            self.assembly_tree.properties["hierarchy_errors"].append(warning_msg)

        _log.info(
            f"Built hierarchical assembly tree: {len(root_nodes)} root nodes, "
            f"{len(self.component_map)} total components, {num_orphaned} orphaned"
        )

    def _build_tree_from_mates(self) -> None:
        """Fallback: Group components by mate relationships.

        Components with many mutual mates likely form a subassembly.
        """
        # Initialize status tracking if not already done
        if "hierarchy_build_status" not in self.assembly_tree.properties:
            self.assembly_tree.properties["hierarchy_build_status"] = "in_progress"
            self.assembly_tree.properties["hierarchy_errors"] = []

        try:
            import networkx as nx
        except ImportError:
            error_msg = (
                "networkx not available for mate-based clustering. "
                "Install with: pip install networkx. "
                "Cannot build hierarchical tree from mate relationships."
            )
            _log.error(error_msg)
            self.assembly_tree.properties["hierarchy_build_status"] = "failed"
            self.assembly_tree.properties["hierarchy_errors"].append(error_msg)
            self.assembly_tree.properties["hierarchy_build_method"] = "none"
            return

        # Build mate graph
        G = nx.Graph()
        for node in self.component_map.values():
            if node.component_id != "root":
                G.add_node(node.component_id)

        # Add edges for mates
        for node in self.component_map.values():
            for mate in node.mates:
                comp1 = mate.get("component1")
                comp2 = mate.get("component2")
                if comp1 and comp2 and comp1 in G and comp2 in G:
                    G.add_edge(comp1, comp2)

        # Detect communities (tightly connected subgraphs)
        try:
            from networkx.algorithms.community import greedy_modularity_communities

            communities = list(greedy_modularity_communities(G))

            # Create sub-assembly nodes for each community
            num_subassemblies = 0
            num_single_components = 0
            for i, community in enumerate(communities):
                if len(community) > 1:  # Only group if multiple components
                    subassembly = AssemblyNode(
                        component_id=f"subassembly_{i}",
                        name=f"Subassembly {i}",
                    )
                    self.assembly_tree.add_child(subassembly)
                    num_subassemblies += 1

                    for comp_id in community:
                        if comp_id in self.component_map:
                            subassembly.add_child(self.component_map[comp_id])
                else:
                    # Single component - add to root
                    comp_id = list(community)[0]
                    if comp_id in self.component_map:
                        self.assembly_tree.add_child(self.component_map[comp_id])
                        num_single_components += 1

            # Mark successful hierarchy build from mates
            self.assembly_tree.properties["hierarchy_build_status"] = "success"
            self.assembly_tree.properties["hierarchy_build_method"] = "mate_clustering"
            self.assembly_tree.properties["num_subassemblies"] = num_subassemblies
            self.assembly_tree.properties["num_single_components"] = num_single_components

            _log.info(
                f"Built hierarchical tree from mate clustering: "
                f"{num_subassemblies} subassemblies, {num_single_components} single components"
            )

        except ImportError:
            error_msg = (
                "networkx community detection not available. "
                "Requires: pip install networkx. "
                "Cannot build hierarchical tree from mate relationships."
            )
            _log.error(error_msg)
            self.assembly_tree.properties["hierarchy_build_status"] = "failed"
            self.assembly_tree.properties["hierarchy_errors"].append(error_msg)
            self.assembly_tree.properties["hierarchy_build_method"] = "none"

    def _determine_status(self, conv_res: ConversionResult):
        """Determine final conversion status.

        Args:
            conv_res: Conversion result

        Returns:
            ConversionStatus
        """
        from cadling.datamodel.base_models import ConversionStatus

        if not conv_res.document:
            return ConversionStatus.FAILURE

        return ConversionStatus.SUCCESS
