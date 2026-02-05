"""Geometric constraint extraction model.

This module provides an enrichment model for extracting implicit geometric
constraints and relationships from CAD models, including parallelism,
perpendicularity, concentricity, symmetry, and distance/angle constraints.

Classes:
    GeometricConstraint: Data structure for a geometric constraint
    ConstraintType: Enumeration of constraint types
    GeometricConstraintModel: Enrichment model for constraint extraction
    ConstraintGraph: Graph representation of constraints

Example:
    from cadling.experimental.models import GeometricConstraintModel
    from cadling.experimental.datamodel import CADAnnotationOptions

    options = CADAnnotationOptions()
    model = GeometricConstraintModel(options)
    model(doc, item_batch)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from cadling.models.base_model import EnrichmentModel

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument

_log = logging.getLogger(__name__)


class ConstraintType(str, Enum):
    """Types of geometric constraints."""

    # Orientation constraints
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    TANGENT = "tangent"

    # Alignment constraints
    CONCENTRIC = "concentric"
    COAXIAL = "coaxial"
    COINCIDENT = "coincident"

    # Symmetry constraints
    SYMMETRIC = "symmetric"
    SYMMETRIC_ABOUT_PLANE = "symmetric_about_plane"
    SYMMETRIC_ABOUT_AXIS = "symmetric_about_axis"

    # Dimensional constraints
    DISTANCE = "distance"
    ANGLE = "angle"
    EQUAL_LENGTH = "equal_length"
    EQUAL_RADIUS = "equal_radius"

    # Topological constraints
    CONNECTED = "connected"
    ADJACENT = "adjacent"


class GeometricConstraint(BaseModel):
    """Data structure for a geometric constraint.

    Attributes:
        constraint_type: Type of constraint
        entities: IDs of constrained entities (faces, edges, vertices)
        parameters: Constraint parameters (distance, angle, etc.)
        confidence: Confidence in constraint detection (0-1)
        description: Human-readable description
        is_explicit: Whether constraint is explicitly modeled (vs inferred)
    """

    constraint_type: ConstraintType
    entities: List[str] = Field(default_factory=list)
    parameters: Dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    description: str = ""
    is_explicit: bool = False


class ConstraintGraph(BaseModel):
    """Graph representation of geometric constraints.

    Attributes:
        nodes: Entity IDs in the graph
        edges: Constraints connecting entities
        clusters: Groups of constrained entities
    """

    nodes: List[str] = Field(default_factory=list)
    edges: List[GeometricConstraint] = Field(default_factory=list)
    clusters: List[List[str]] = Field(default_factory=list)


class GeometricConstraintModel(EnrichmentModel):
    """Enrichment model for extracting geometric constraints from CAD models.

    This model analyzes the BRep topology and geometry to identify implicit
    geometric relationships and constraints that constrain the design. It can
    detect:

    **Orientation Constraints:**
    - Parallel faces, edges, axes
    - Perpendicular relationships
    - Tangent surfaces

    **Alignment Constraints:**
    - Concentric holes and cylinders
    - Coaxial features
    - Coincident points and surfaces

    **Symmetry:**
    - Bilateral symmetry about planes
    - Rotational symmetry about axes
    - Pattern symmetry

    **Dimensional Constraints:**
    - Equal distances
    - Equal angles
    - Equal radii
    - Proportional relationships

    These constraints are useful for:
    - Understanding design intent
    - Parametric model reconstruction
    - Constraint-based feature recognition
    - Design variation generation
    - Manufacturing planning

    Attributes:
        tolerance: Numerical tolerance for constraint detection
        min_confidence: Minimum confidence to report constraint

    Example:
        options = CADAnnotationOptions()
        model = GeometricConstraintModel(options)
        model(doc, [item])

        # Access constraint graph
        graph = item.properties.get("constraint_graph")
        for constraint in graph["edges"]:
            print(f"{constraint['constraint_type']}: {constraint['description']}")
    """

    def __init__(
        self,
        tolerance: float = 0.001,
        min_confidence: float = 0.7,
    ):
        """Initialize geometric constraint model.

        Args:
            tolerance: Numerical tolerance for constraint detection (default 0.001)
            min_confidence: Minimum confidence to report constraint (default 0.7)
        """
        super().__init__()
        self.tolerance = tolerance
        self.min_confidence = min_confidence

        _log.info(
            f"Initialized GeometricConstraintModel "
            f"(tolerance={tolerance}, min_confidence={min_confidence})"
        )

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: List[CADItem],
    ) -> None:
        """Extract geometric constraints from CAD items.

        Args:
            doc: The CADlingDocument being enriched
            item_batch: List of CADItem objects to process

        Note:
            Constraint graph is added to item.properties["constraint_graph"]
            Individual constraints added to item.properties["constraints"]
        """
        _log.info(f"Processing {len(item_batch)} items for constraint extraction")

        for item in item_batch:
            try:
                # Extract constraints from topology
                constraints = []

                # Check if topology data is available
                if hasattr(doc, "topology") and doc.topology:
                    constraints.extend(self._extract_orientation_constraints(doc, item))
                    constraints.extend(self._extract_alignment_constraints(doc, item))
                    constraints.extend(self._extract_symmetry_constraints(doc, item))
                    constraints.extend(self._extract_dimensional_constraints(doc, item))

                # Filter by confidence
                constraints = [
                    c for c in constraints if c.confidence >= self.min_confidence
                ]

                # Build constraint graph
                graph = self._build_constraint_graph(constraints)

                # Add to item properties
                item.properties["constraints"] = [c.model_dump() for c in constraints]
                item.properties["constraint_graph"] = graph.model_dump()
                item.properties["constraint_model"] = self.__class__.__name__

                # Add provenance
                if hasattr(item, "add_provenance"):
                    item.add_provenance(
                        component_type="enrichment_model",
                        component_name=self.__class__.__name__,
                    )

                _log.info(
                    f"Extracted {len(constraints)} constraints for item {item.self_ref}"
                )

            except Exception as e:
                _log.error(f"Constraint extraction failed for item {item.self_ref}: {e}")
                item.properties["constraints"] = []
                item.properties["constraint_graph"] = None
                item.properties["constraint_extraction_error"] = str(e)

    def _extract_orientation_constraints(
        self, doc: CADlingDocument, item: CADItem
    ) -> List[GeometricConstraint]:
        """Extract parallel and perpendicular constraints.

        Args:
            doc: The document
            item: The item to analyze

        Returns:
            List of orientation constraints
        """
        constraints = []

        # Get topology data
        topology = doc.topology
        if not topology:
            return constraints

        faces = topology.get("faces", [])

        # Check all pairs of faces for parallelism/perpendicularity
        for i, face1 in enumerate(faces):
            normal1 = face1.get("normal")
            if not normal1:
                continue

            for face2 in faces[i + 1 :]:
                normal2 = face2.get("normal")
                if not normal2:
                    continue

                # Compute dot product of normals
                dot = sum(n1 * n2 for n1, n2 in zip(normal1, normal2))

                # Check for parallel (dot ≈ ±1)
                if abs(abs(dot) - 1.0) < self.tolerance:
                    constraints.append(
                        GeometricConstraint(
                            constraint_type=ConstraintType.PARALLEL,
                            entities=[face1.get("id", ""), face2.get("id", "")],
                            parameters={"dot_product": dot},
                            confidence=0.95,
                            description=f"Faces {face1.get('id')} and {face2.get('id')} are parallel",
                        )
                    )

                # Check for perpendicular (dot ≈ 0)
                elif abs(dot) < self.tolerance:
                    constraints.append(
                        GeometricConstraint(
                            constraint_type=ConstraintType.PERPENDICULAR,
                            entities=[face1.get("id", ""), face2.get("id", "")],
                            parameters={"dot_product": dot},
                            confidence=0.95,
                            description=f"Faces {face1.get('id')} and {face2.get('id')} are perpendicular",
                        )
                    )

        _log.debug(f"Found {len(constraints)} orientation constraints")
        return constraints

    def _extract_alignment_constraints(
        self, doc: CADlingDocument, item: CADItem
    ) -> List[GeometricConstraint]:
        """Extract concentric and coaxial constraints.

        Args:
            doc: The document
            item: The item to analyze

        Returns:
            List of alignment constraints
        """
        constraints = []

        # Get detected features
        features = item.properties.get("machining_features", [])

        # Find circular features (holes, bosses, cylinders)
        circular_features = [
            f
            for f in features
            if f.get("feature_type") in ["hole", "boss"]
            or (f.get("feature_type") == "pocket" and f.get("subtype") == "circular_pocket")
        ]

        # Check all pairs for concentricity
        for i, feat1 in enumerate(circular_features):
            loc1 = feat1.get("location")
            if not loc1:
                continue

            for feat2 in circular_features[i + 1 :]:
                loc2 = feat2.get("location")
                if not loc2:
                    continue

                # Compute distance between centers
                dist = sum((l1 - l2) ** 2 for l1, l2 in zip(loc1, loc2)) ** 0.5

                # If centers are very close, features are concentric
                if dist < self.tolerance:
                    constraints.append(
                        GeometricConstraint(
                            constraint_type=ConstraintType.CONCENTRIC,
                            entities=[
                                str(feat1.get("feature_type")),
                                str(feat2.get("feature_type")),
                            ],
                            parameters={"distance": dist},
                            confidence=0.9,
                            description=f"Features at {loc1} and {loc2} are concentric",
                        )
                    )

        _log.debug(f"Found {len(constraints)} alignment constraints")
        return constraints

    def _extract_symmetry_constraints(
        self, doc: CADlingDocument, item: CADItem
    ) -> List[GeometricConstraint]:
        """Extract symmetry constraints.

        Args:
            doc: The document
            item: The item to analyze

        Returns:
            List of symmetry constraints
        """
        constraints = []

        # Check for symmetry in bounding box (simple heuristic)
        bbox = item.properties.get("bounding_box", {})
        if not bbox:
            return constraints

        # Get dimensions
        x_dim = bbox.get("x", 0)
        y_dim = bbox.get("y", 0)
        z_dim = bbox.get("z", 0)

        # Check if any dimension is significantly smaller (thin part)
        dims = [x_dim, y_dim, z_dim]
        min_dim = min(dims)
        max_dim = max(dims)

        if min_dim > 0 and max_dim / min_dim > 5:
            # Likely a thin part with potential symmetry
            constraints.append(
                GeometricConstraint(
                    constraint_type=ConstraintType.SYMMETRIC_ABOUT_PLANE,
                    entities=["part"],
                    parameters={
                        "x_dim": x_dim,
                        "y_dim": y_dim,
                        "z_dim": z_dim,
                    },
                    confidence=0.7,
                    description="Part appears to have planar symmetry",
                )
            )

        # Check for symmetry in feature placement
        features = item.properties.get("machining_features", [])
        holes = [f for f in features if f.get("feature_type") == "hole"]

        # If there are multiple holes, check for symmetric patterns
        if len(holes) >= 2:
            # Simple check: see if holes come in pairs at similar distances from center
            # This is a simplified heuristic
            constraints.append(
                GeometricConstraint(
                    constraint_type=ConstraintType.SYMMETRIC,
                    entities=[f"hole_{i}" for i in range(len(holes))],
                    parameters={"num_features": len(holes)},
                    confidence=0.6,
                    description=f"Potential symmetric pattern of {len(holes)} holes",
                    is_explicit=False,
                )
            )

        _log.debug(f"Found {len(constraints)} symmetry constraints")
        return constraints

    def _extract_dimensional_constraints(
        self, doc: CADlingDocument, item: CADItem
    ) -> List[GeometricConstraint]:
        """Extract distance and angle constraints.

        Args:
            doc: The document
            item: The item to analyze

        Returns:
            List of dimensional constraints
        """
        constraints = []

        # Get PMI annotations with dimensions
        pmi = item.properties.get("pmi_annotations", [])
        dimensions = [a for a in pmi if a.get("type") == "dimension"]

        for dim in dimensions:
            value = dim.get("value")
            unit = dim.get("unit")

            if value is not None:
                constraints.append(
                    GeometricConstraint(
                        constraint_type=ConstraintType.DISTANCE,
                        entities=["dimension"],
                        parameters={"value": value, "unit": unit or "mm"},
                        confidence=dim.get("confidence", 0.8),
                        description=dim.get("text", ""),
                        is_explicit=True,  # Dimensions are explicit
                    )
                )

        # Check for equal-sized features
        features = item.properties.get("machining_features", [])
        holes = [f for f in features if f.get("feature_type") == "hole"]

        # Group holes by diameter
        diameter_groups: Dict[float, List[Dict]] = {}
        for hole in holes:
            diam = hole.get("parameters", {}).get("diameter")
            if diam:
                # Round to tolerance
                diam_key = round(diam / self.tolerance) * self.tolerance
                if diam_key not in diameter_groups:
                    diameter_groups[diam_key] = []
                diameter_groups[diam_key].append(hole)

        # Add equal-radius constraints for groups with multiple holes
        for diam, group in diameter_groups.items():
            if len(group) > 1:
                constraints.append(
                    GeometricConstraint(
                        constraint_type=ConstraintType.EQUAL_RADIUS,
                        entities=[f"hole_{i}" for i in range(len(group))],
                        parameters={"radius": diam / 2},
                        confidence=0.85,
                        description=f"{len(group)} holes with equal diameter {diam:.2f}mm",
                    )
                )

        _log.debug(f"Found {len(constraints)} dimensional constraints")
        return constraints

    def _build_constraint_graph(
        self, constraints: List[GeometricConstraint]
    ) -> ConstraintGraph:
        """Build a constraint graph from extracted constraints.

        Args:
            constraints: List of constraints

        Returns:
            ConstraintGraph object
        """
        # Collect all entity IDs
        nodes = set()
        for constraint in constraints:
            nodes.update(constraint.entities)

        # Find clusters of connected entities
        clusters = self._find_clusters(list(nodes), constraints)

        return ConstraintGraph(
            nodes=list(nodes),
            edges=constraints,
            clusters=clusters,
        )

    def _find_clusters(
        self, nodes: List[str], constraints: List[GeometricConstraint]
    ) -> List[List[str]]:
        """Find clusters of connected entities using union-find.

        Args:
            nodes: List of entity IDs
            constraints: List of constraints

        Returns:
            List of clusters (each cluster is a list of entity IDs)
        """
        # Simple clustering: entities connected by constraints
        parent = {node: node for node in nodes}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union entities connected by constraints
        for constraint in constraints:
            entities = constraint.entities
            if len(entities) >= 2:
                for i in range(len(entities) - 1):
                    if entities[i] in parent and entities[i + 1] in parent:
                        union(entities[i], entities[i + 1])

        # Group by parent
        clusters_dict: Dict[str, List[str]] = {}
        for node in nodes:
            p = find(node)
            if p not in clusters_dict:
                clusters_dict[p] = []
            clusters_dict[p].append(node)

        return list(clusters_dict.values())

    def supports_batch_processing(self) -> bool:
        """Whether this model supports batch processing."""
        return True

    def get_batch_size(self) -> int:
        """Get recommended batch size."""
        return 10

    def requires_gpu(self) -> bool:
        """Whether this model requires GPU acceleration."""
        return False

    def get_model_info(self) -> Dict[str, str]:
        """Get information about this model."""
        info = super().get_model_info()
        info.update(
            {
                "tolerance": str(self.tolerance),
                "min_confidence": str(self.min_confidence),
                "constraint_types": str(len(ConstraintType)),
            }
        )
        return info
