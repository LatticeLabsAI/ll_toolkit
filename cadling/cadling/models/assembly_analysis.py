"""Assembly analysis enrichment model.

This module provides enrichment models for analyzing multi-part CAD assemblies,
including assembly graph construction, mating surface detection, BOM generation,
and subassembly identification.

Classes:
    AssemblyAnalysisModel: Analyze assembly structure and relationships
    AssemblyGraph: Data structure for assembly relationships
    Contact: Represents contact between two parts
    BillOfMaterials: Assembly BOM data structure
    Subassembly: Logical subassembly grouping
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

from cadling.models.base_model import EnrichmentModel

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument
    from cadling.datamodel.stl import AssemblyItem

_log = logging.getLogger(__name__)


@dataclass
class Contact:
    """Represents a contact/mating surface between two parts.

    Attributes:
        part1_id: ID of first part
        part2_id: ID of second part
        contact_type: Type of contact (planar, cylindrical, spherical)
        contact_area: Area of contact surface
        contact_center: Center point of contact [x, y, z]
        contact_normal: Normal vector of contact surface
        distance: Distance between parts (0 for touching)
        confidence: Confidence score [0, 1]
    """
    part1_id: str
    part2_id: str
    contact_type: str
    contact_area: float = 0.0
    contact_center: Optional[List[float]] = None
    contact_normal: Optional[List[float]] = None
    distance: float = 0.0
    confidence: float = 0.5


@dataclass
class Subassembly:
    """Represents a logical subassembly within an assembly.

    Attributes:
        subassembly_id: Unique identifier
        name: Subassembly name
        part_ids: List of part IDs in subassembly
        parent_id: Parent assembly/subassembly ID
        transform: Transformation matrix for subassembly
        metadata: Additional metadata
    """
    subassembly_id: str
    name: str
    part_ids: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    transform: Optional[List[List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BillOfMaterials:
    """Bill of Materials for an assembly.

    Attributes:
        assembly_id: ID of assembly
        items: List of BOM items
        total_parts: Total number of parts
        unique_parts: Number of unique parts
        hierarchy_depth: Maximum depth of assembly hierarchy
    """
    assembly_id: str
    items: List[Dict[str, Any]] = field(default_factory=list)
    total_parts: int = 0
    unique_parts: int = 0
    hierarchy_depth: int = 0

    def add_item(
        self,
        part_id: str,
        part_name: str,
        quantity: int = 1,
        level: int = 0,
        **metadata
    ):
        """Add an item to the BOM.

        Args:
            part_id: Unique part identifier
            part_name: Part name/description
            quantity: Quantity in assembly
            level: Hierarchy level (0 = top level)
            **metadata: Additional part metadata
        """
        self.items.append({
            "part_id": part_id,
            "part_name": part_name,
            "quantity": quantity,
            "level": level,
            **metadata
        })
        self.total_parts += quantity

        # Update hierarchy depth
        if level + 1 > self.hierarchy_depth:
            self.hierarchy_depth = level + 1


@dataclass
class AssemblyGraph:
    """Data structure for assembly relationships.

    Attributes:
        nodes: Dictionary of node_id -> node_data
        edges: List of (source_id, target_id, edge_data) tuples
        root_id: ID of root assembly node
    """
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Tuple[str, str, Dict[str, Any]]] = field(default_factory=list)
    root_id: Optional[str] = None

    def add_node(self, node_id: str, node_type: str = "part", **attributes):
        """Add a node to the graph.

        Args:
            node_id: Unique node identifier
            node_type: Type of node (part, assembly, subassembly)
            **attributes: Additional node attributes
        """
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            **attributes
        }

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "contains",
        **attributes
    ):
        """Add an edge to the graph.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of edge (contains, mates_with, adjacent_to)
            **attributes: Additional edge attributes
        """
        self.edges.append((source_id, target_id, {
            "type": edge_type,
            **attributes
        }))

    def get_children(self, node_id: str) -> List[str]:
        """Get child node IDs for a given node.

        Args:
            node_id: Parent node ID

        Returns:
            List of child node IDs
        """
        children = []
        for source, target, edge_data in self.edges:
            if source == node_id and edge_data.get("type") == "contains":
                children.append(target)
        return children

    def get_depth(
        self,
        node_id: str,
        current_depth: int = 0,
        _visited: set[str] | None = None,
    ) -> int:
        """Get maximum depth of hierarchy from a node.

        Args:
            node_id: Node to measure depth from
            current_depth: Current depth (for recursion)
            _visited: Internal cycle guard — do not pass manually

        Returns:
            Maximum depth
        """
        if _visited is None:
            _visited = set()
        if node_id in _visited:
            _log.warning("Cycle detected at node %s — stopping traversal", node_id)
            return current_depth
        _visited.add(node_id)

        children = self.get_children(node_id)
        if not children:
            return current_depth

        max_child_depth = current_depth
        for child_id in children:
            child_depth = self.get_depth(child_id, current_depth + 1, _visited)
            if child_depth > max_child_depth:
                max_child_depth = child_depth

        return max_child_depth


class AssemblyAnalysisModel(EnrichmentModel):
    """Enrichment model for analyzing multi-part CAD assemblies.

    Analyzes assembly structure including:
    - Assembly hierarchy graphs
    - Mating surface detection
    - Bill of Materials generation
    - Subassembly identification

    Results are stored in doc.properties["assembly_analysis"].

    Example:
        model = AssemblyAnalysisModel()
        model(doc, doc.items)

        # Access results
        result = doc.properties.get("assembly_analysis", {})
        assembly_graph = result.get("assembly_graph")
        bom = result.get("bom")
    """

    #: Maximum number of part pairs to evaluate for contact detection.
    #: Prevents runaway O(n²) pythonocc calls on large assemblies.
    MAX_CONTACT_PAIRS: int = 50_000

    def __init__(
        self,
        detect_contacts: bool = True,
        compute_bom: bool = True,
        identify_subassemblies: bool = True,
        contact_tolerance: float = 0.01,
        max_contact_pairs: int | None = None,
    ):
        """Initialize assembly analysis model.

        Args:
            detect_contacts: Whether to detect mating surfaces
            compute_bom: Whether to compute BOM
            identify_subassemblies: Whether to detect subassemblies
            contact_tolerance: Distance tolerance for contact detection (mm)
            max_contact_pairs: Cap on part-pair evaluations (default MAX_CONTACT_PAIRS)
        """
        super().__init__()
        self.detect_contacts = detect_contacts
        self.compute_bom = compute_bom
        self.identify_subassemblies = identify_subassemblies
        self.contact_tolerance = contact_tolerance
        self.max_contact_pairs = max_contact_pairs if max_contact_pairs is not None else self.MAX_CONTACT_PAIRS

        # Check for pythonocc availability
        try:
            from OCC.Core.BRepBndLib import brepbndlib
            self.has_pythonocc = True
        except ImportError:
            self.has_pythonocc = False
            _log.warning(
                "pythonocc-core not available. Assembly analysis will use "
                "simplified algorithms."
            )


    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Analyze assembly structure and add results to document.

        Args:
            doc: CADlingDocument to analyze
            item_batch: Items to process (unused, analyzes entire document)
        """
        if not self.has_pythonocc:
            _log.warning(
                "Skipping assembly analysis - missing required dependencies"
            )
            return

        try:
            # Build assembly graph
            assembly_graph = self.build_assembly_graph(doc)

            # Initialize results
            results = {
                "assembly_graph": self._serialize_graph(assembly_graph),
                "num_parts": len([n for n in assembly_graph.nodes.values()
                                 if n["type"] == "part"]),
                "num_assemblies": len([n for n in assembly_graph.nodes.values()
                                      if n["type"] == "assembly"]),
                "hierarchy_depth": assembly_graph.get_depth(assembly_graph.root_id)
                                  if assembly_graph.root_id else 0,
            }

            # Detect mating surfaces
            if self.detect_contacts and self.has_pythonocc:
                contacts = self._detect_all_contacts(doc, assembly_graph)
                results["contacts"] = [self._serialize_contact(c) for c in contacts]
                results["num_contacts"] = len(contacts)

            # Compute BOM
            if self.compute_bom:
                bom = self.compute_assembly_bom(doc, assembly_graph)
                results["bom"] = self._serialize_bom(bom)

            # Detect subassemblies
            if self.identify_subassemblies:
                subassemblies = self.detect_subassemblies(assembly_graph)
                results["subassemblies"] = [
                    self._serialize_subassembly(s) for s in subassemblies
                ]
                results["num_subassemblies"] = len(subassemblies)

            # Store results in document properties
            doc.properties["assembly_analysis"] = results

            # Stamp provenance on all items analyzed in the assembly
            for item in item_batch:
                item.add_provenance(
                    component_type="enrichment_model",
                    component_name=self.__class__.__name__,
                )

            _log.info(
                f"Assembly analysis complete: {results['num_parts']} parts, "
                f"{results['hierarchy_depth']} levels"
            )

        except Exception as e:
            _log.error(f"Assembly analysis failed: {e}", exc_info=True)

    def build_assembly_graph(self, doc: CADlingDocument) -> AssemblyGraph:
        """Build assembly relationship graph from document.

        Args:
            doc: CADlingDocument containing assembly items

        Returns:
            AssemblyGraph representing assembly hierarchy
        """
        graph = AssemblyGraph()

        # Find assembly items
        assembly_items = [
            item for item in doc.items
            if hasattr(item, 'item_type') and item.item_type == "assembly"
        ]

        # If no explicit assembly items, treat all items as parts in single assembly
        if not assembly_items:
            # Create virtual root assembly
            root_id = "root_assembly"
            graph.add_node(root_id, node_type="assembly", name="Root Assembly")
            graph.root_id = root_id

            # Add all items as parts
            for i, item in enumerate(doc.items):
                part_id = getattr(item, 'item_id', f"part_{i}")
                part_name = getattr(item, 'name', f"Part {i}")

                graph.add_node(part_id, node_type="part", name=part_name, item=item)
                graph.add_edge(root_id, part_id, edge_type="contains")
        else:
            # Process assembly hierarchy
            for asm_item in assembly_items:
                asm_id = getattr(asm_item, 'item_id', 'assembly_0')
                asm_name = getattr(asm_item, 'name', 'Assembly')

                graph.add_node(asm_id, node_type="assembly", name=asm_name, item=asm_item)

                # Set first assembly as root if not set
                if graph.root_id is None:
                    graph.root_id = asm_id

                # Add components
                if hasattr(asm_item, 'components'):
                    for comp_id in asm_item.components:
                        # Find component item
                        comp_item = None
                        for item in doc.items:
                            if getattr(item, 'item_id', None) == comp_id:
                                comp_item = item
                                break

                        if comp_item:
                            comp_type = "part"
                            if hasattr(comp_item, 'item_type') and comp_item.item_type == "assembly":
                                comp_type = "assembly"

                            comp_name = getattr(comp_item, 'name', comp_id)
                            graph.add_node(comp_id, node_type=comp_type, name=comp_name, item=comp_item)
                            graph.add_edge(asm_id, comp_id, edge_type="contains")

        return graph

    def detect_mating_surfaces(
        self,
        part1: CADItem,
        part2: CADItem
    ) -> list[Contact]:
        """Find contacting/mating surfaces between two parts.

        Uses BRepExtrema for precise minimum distance computation, then
        iterates face pairs to identify actual mating surfaces within
        the contact tolerance.

        Args:
            part1: First part item
            part2: Second part item

        Returns:
            List of Contact objects representing mating surfaces
        """
        contacts = []

        if not self.has_pythonocc:
            return contacts

        try:
            from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.GeomAbs import (
                GeomAbs_Plane,
                GeomAbs_Cylinder,
                GeomAbs_Cone,
                GeomAbs_Sphere,
                GeomAbs_Torus,
            )
            from OCC.Core.TopAbs import TopAbs_REVERSED
            from OCC.Core.TopoDS import topods
            from OCC.Core.Bnd import Bnd_Box
            from OCC.Core.BRepBndLib import brepbndlib

            # Get OCC shapes if available
            shape1 = getattr(part1, '_occ_shape', None)
            shape2 = getattr(part2, '_occ_shape', None)

            if shape1 is None or shape2 is None:
                return contacts

            # Precise minimum distance between shapes
            dist_calc = BRepExtrema_DistShapeShape(shape1, shape2)
            dist_calc.Perform()

            if not dist_calc.IsDone():
                _log.debug("BRepExtrema computation failed for mating detection")
                return contacts

            min_distance = dist_calc.Value()

            # If shapes are farther apart than tolerance, no mating
            if min_distance > self.contact_tolerance:
                return contacts

            part1_id = getattr(part1, 'item_id', 'unknown')
            part2_id = getattr(part2, 'item_id', 'unknown')

            # Collect faces from each shape
            faces1 = []
            explorer = TopExp_Explorer(shape1, TopAbs_FACE)
            while explorer.More():
                faces1.append(topods.Face(explorer.Current()))
                explorer.Next()

            faces2 = []
            explorer = TopExp_Explorer(shape2, TopAbs_FACE)
            while explorer.More():
                faces2.append(topods.Face(explorer.Current()))
                explorer.Next()

            # Pre-compute per-face AABBs for fast rejection
            def _face_aabb(face):
                box = Bnd_Box()
                brepbndlib.Add(face, box)
                return box

            aabbs1 = [_face_aabb(f) for f in faces1]
            aabbs2 = [_face_aabb(f) for f in faces2]
            tol = self.contact_tolerance

            # Find mating face pairs using AABB filter + precise distance
            surface_type_names = {
                GeomAbs_Plane: "planar",
                GeomAbs_Cylinder: "cylindrical",
                GeomAbs_Cone: "conical",
                GeomAbs_Sphere: "spherical",
                GeomAbs_Torus: "toroidal",
            }

            for i, face1 in enumerate(faces1):
                box1 = aabbs1[i]
                for j, face2 in enumerate(faces2):
                    # Fast AABB rejection — skip if bounding boxes are
                    # farther apart than contact_tolerance
                    if box1.Distance(aabbs2[j]) > tol:
                        continue

                    face_dist = BRepExtrema_DistShapeShape(face1, face2)
                    face_dist.Perform()

                    if not face_dist.IsDone():
                        continue

                    if face_dist.Value() > self.contact_tolerance:
                        continue

                    # These faces are in contact — classify the contact
                    adaptor1 = BRepAdaptor_Surface(face1)
                    adaptor2 = BRepAdaptor_Surface(face2)
                    stype1 = adaptor1.GetType()
                    stype2 = adaptor2.GetType()

                    # Determine contact type from surface geometry
                    if stype1 == stype2 and stype1 in surface_type_names:
                        contact_type = surface_type_names[stype1]
                    else:
                        name1 = surface_type_names.get(stype1, "freeform")
                        name2 = surface_type_names.get(stype2, "freeform")
                        contact_type = f"{name1}-{name2}"

                    # Compute contact area (smaller face area as proxy)
                    props1 = GProp_GProps()
                    props2 = GProp_GProps()
                    brepgprop.SurfaceProperties(face1, props1)
                    brepgprop.SurfaceProperties(face2, props2)
                    contact_area = min(props1.Mass(), props2.Mass())

                    # Contact center from closest points
                    if face_dist.NbSolution() > 0:
                        pt1 = face_dist.PointOnShape1(1)
                        pt2 = face_dist.PointOnShape2(1)
                        center = [
                            (pt1.X() + pt2.X()) / 2,
                            (pt1.Y() + pt2.Y()) / 2,
                            (pt1.Z() + pt2.Z()) / 2,
                        ]
                    else:
                        center = None

                    # Contact normal from planar faces
                    contact_normal = None
                    if stype1 == GeomAbs_Plane:
                        pln = adaptor1.Plane()
                        ax = pln.Axis().Direction()
                        # Flip if face orientation is reversed
                        sign = -1.0 if face1.Orientation() == TopAbs_REVERSED else 1.0
                        contact_normal = [
                            sign * ax.X(),
                            sign * ax.Y(),
                            sign * ax.Z(),
                        ]

                    # Confidence based on contact quality
                    dist_val = face_dist.Value()
                    if dist_val < 1e-6:
                        confidence = 0.95  # Touching
                    elif dist_val < self.contact_tolerance * 0.5:
                        confidence = 0.85
                    else:
                        confidence = 0.7

                    contact = Contact(
                        part1_id=part1_id,
                        part2_id=part2_id,
                        contact_type=contact_type,
                        contact_area=float(contact_area),
                        contact_center=center,
                        contact_normal=contact_normal,
                        distance=float(dist_val),
                        confidence=confidence,
                    )
                    contacts.append(contact)

        except Exception as e:
            _log.debug(f"Error detecting mating surfaces: {e}")

        return contacts

    def compute_assembly_bom(
        self,
        doc: CADlingDocument,
        assembly_graph: Optional[AssemblyGraph] = None
    ) -> BillOfMaterials:
        """Generate Bill of Materials from assembly.

        Args:
            doc: CADlingDocument to analyze
            assembly_graph: Optional pre-built assembly graph

        Returns:
            BillOfMaterials object
        """
        if assembly_graph is None:
            assembly_graph = self.build_assembly_graph(doc)

        bom = BillOfMaterials(
            assembly_id=assembly_graph.root_id or "root"
        )

        # Count parts by ID (for quantity tracking)
        part_counts: Dict[str, int] = {}
        part_data: Dict[str, Dict[str, Any]] = {}

        # Traverse graph and collect parts
        def traverse(node_id: str, level: int = 0):
            node = assembly_graph.nodes.get(node_id)
            if not node:
                return

            if node["type"] == "part":
                # Track part occurrences
                if node_id in part_counts:
                    part_counts[node_id] += 1
                else:
                    part_counts[node_id] = 1
                    part_data[node_id] = {
                        "name": node.get("name", node_id),
                        "level": level
                    }

            # Recurse to children
            children = assembly_graph.get_children(node_id)
            for child_id in children:
                traverse(child_id, level + 1)

        # Start traversal from root
        if assembly_graph.root_id:
            traverse(assembly_graph.root_id)

        # Add items to BOM
        for part_id, quantity in part_counts.items():
            data = part_data[part_id]
            bom.add_item(
                part_id=part_id,
                part_name=data["name"],
                quantity=quantity,
                level=data["level"]
            )

        bom.unique_parts = len(part_counts)

        return bom

    def detect_subassemblies(
        self,
        assembly_graph: AssemblyGraph
    ) -> list[Subassembly]:
        """Identify logical subassemblies within assembly.

        Uses clustering/grouping heuristics to identify subassemblies:
        - Groups of parts with high interconnectivity
        - Nodes with multiple children
        - Named assembly nodes

        Args:
            assembly_graph: Assembly graph to analyze

        Returns:
            List of Subassembly objects
        """
        subassemblies = []

        # Strategy 1: Nodes explicitly marked as assembly type
        for node_id, node_data in assembly_graph.nodes.items():
            if node_data["type"] == "assembly":
                children = assembly_graph.get_children(node_id)
                if len(children) > 0:
                    subasm = Subassembly(
                        subassembly_id=node_id,
                        name=node_data.get("name", node_id),
                        part_ids=children,
                        parent_id=assembly_graph.root_id
                    )
                    subassemblies.append(subasm)

        # Strategy 2: Groups of parts with common naming patterns
        # (e.g., "Wheel_Assembly_1", "Wheel_Assembly_2")
        part_nodes = [
            (nid, n) for nid, n in assembly_graph.nodes.items()
            if n["type"] == "part"
        ]

        # Individual hardware parts that should NOT form subassemblies
        # even when multiple instances share a prefix (e.g., Bolt_M6_1,
        # Bolt_M6_2 are individual fasteners, not a subassembly).
        _hardware_patterns = {
            "bolt", "screw", "nut", "washer", "pin", "rivet",
            "nail", "stud", "dowel", "clip", "clamp", "spacer",
            "bushing", "bearing", "gasket", "oring", "o_ring",
            "spring", "key", "cotter", "fastener", "insert",
        }

        # Group by name prefix — only when the trailing segment is a
        # numeric instance counter (e.g. "_1", "_02").
        prefix_groups: Dict[str, List[str]] = {}
        for node_id, node_data in part_nodes:
            name = node_data.get("name", "")
            parts = name.split("_")
            if len(parts) > 1 and parts[-1].isdigit():
                prefix = "_".join(parts[:-1])
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(node_id)

        # Create subassemblies for groups with multiple parts,
        # excluding groups whose prefix indicates individual hardware.
        for prefix, part_ids in prefix_groups.items():
            if len(part_ids) >= 2:
                prefix_lower = prefix.lower()
                # Skip if any token in the prefix is a hardware keyword
                prefix_tokens = set(prefix_lower.split("_"))
                if prefix_tokens & _hardware_patterns:
                    continue
                subasm = Subassembly(
                    subassembly_id=f"subasm_{prefix}",
                    name=f"{prefix} Group",
                    part_ids=part_ids,
                    metadata={"detection_method": "naming_pattern"}
                )
                subassemblies.append(subasm)

        return subassemblies

    def _detect_all_contacts(
        self,
        doc: CADlingDocument,
        assembly_graph: AssemblyGraph
    ) -> list[Contact]:
        """Detect all contacts between parts in assembly.

        Args:
            doc: CADlingDocument
            assembly_graph: Assembly graph

        Returns:
            List of Contact objects
        """
        all_contacts = []

        # Get all part nodes
        part_nodes = [
            (nid, n) for nid, n in assembly_graph.nodes.items()
            if n["type"] == "part" and "item" in n
        ]

        n_parts = len(part_nodes)
        total_pairs = n_parts * (n_parts - 1) // 2

        if total_pairs > self.max_contact_pairs:
            _log.warning(
                "Assembly has %d parts (%d pairs) — exceeds max_contact_pairs=%d. "
                "Skipping contact detection. Increase max_contact_pairs or reduce assembly size.",
                n_parts, total_pairs, self.max_contact_pairs,
            )
            return all_contacts

        # Pre-compute AABBs for broad-phase filtering to avoid expensive
        # pythonocc calls on distant part pairs.
        aabbs: dict[str, tuple[float, ...]] = {}
        if self.has_pythonocc:
            try:
                from OCC.Core.BRepBndLib import brepbndlib
                from OCC.Core.Bnd import Bnd_Box

                for nid, node in part_nodes:
                    shape = getattr(node.get("item"), "_occ_shape", None)
                    if shape is not None:
                        bbox = Bnd_Box()
                        brepbndlib.Add(shape, bbox)
                        aabbs[nid] = bbox.Get()  # (xmin,ymin,zmin,xmax,ymax,zmax)
            except Exception:
                _log.debug("AABB pre-computation failed; falling back to full pairwise check")

        for i, (id1, node1) in enumerate(part_nodes):
            for id2, node2 in part_nodes[i + 1:]:
                # Broad-phase: skip pairs whose AABBs don't overlap (with tolerance)
                if id1 in aabbs and id2 in aabbs:
                    a = aabbs[id1]
                    b = aabbs[id2]
                    tol = self.contact_tolerance
                    if (a[0] > b[3] + tol or b[0] > a[3] + tol or
                            a[1] > b[4] + tol or b[1] > a[4] + tol or
                            a[2] > b[5] + tol or b[2] > a[5] + tol):
                        continue

                contacts = self.detect_mating_surfaces(node1["item"], node2["item"])
                all_contacts.extend(contacts)

        return all_contacts

    def _serialize_graph(self, graph: AssemblyGraph) -> Dict[str, Any]:
        """Serialize assembly graph to dictionary.

        Args:
            graph: AssemblyGraph to serialize

        Returns:
            Dictionary representation
        """
        # Remove item references (not serializable)
        nodes = {}
        for node_id, node_data in graph.nodes.items():
            nodes[node_id] = {k: v for k, v in node_data.items() if k != "item"}

        return {
            "nodes": nodes,
            "edges": [
                {
                    "source": src,
                    "target": tgt,
                    "data": data
                }
                for src, tgt, data in graph.edges
            ],
            "root_id": graph.root_id
        }

    def _serialize_contact(self, contact: Contact) -> Dict[str, Any]:
        """Serialize Contact to dictionary."""
        return {
            "part1_id": contact.part1_id,
            "part2_id": contact.part2_id,
            "contact_type": contact.contact_type,
            "contact_area": contact.contact_area,
            "contact_center": contact.contact_center,
            "contact_normal": contact.contact_normal,
            "distance": contact.distance,
            "confidence": contact.confidence
        }

    def _serialize_bom(self, bom: BillOfMaterials) -> Dict[str, Any]:
        """Serialize BillOfMaterials to dictionary."""
        return {
            "assembly_id": bom.assembly_id,
            "items": bom.items,
            "total_parts": bom.total_parts,
            "unique_parts": bom.unique_parts,
            "hierarchy_depth": bom.hierarchy_depth
        }

    def _serialize_subassembly(self, subasm: Subassembly) -> Dict[str, Any]:
        """Serialize Subassembly to dictionary."""
        return {
            "subassembly_id": subasm.subassembly_id,
            "name": subasm.name,
            "part_ids": subasm.part_ids,
            "parent_id": subasm.parent_id,
            "transform": subasm.transform,
            "metadata": subasm.metadata
        }
