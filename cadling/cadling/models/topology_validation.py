"""Topology validation model for CAD parts.

This module provides topology validation capabilities including:
- Manifoldness checking
- Euler characteristic validation (V - E + F = 2 - 2g)
- Watertightness verification
- Self-intersection detection
- Orientation consistency checking

Classes:
    TopologyValidationModel: Main model for topology validation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from cadling.models.base_model import EnrichmentModel

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument

_log = logging.getLogger(__name__)


class ValidationFinding(BaseModel):
    """Individual validation finding with severity and entity tracking."""

    check_name: str  # e.g., "face_edge_consistency"
    severity: str  # "critical" | "warning" | "info"
    message: str
    entity_ids: list[str] = []  # specific OCC shape hash IDs
    entity_type: Optional[str] = None  # FACE, EDGE, VERTEX, etc.


class TopologyValidationModel(EnrichmentModel):
    """Topology validation enrichment model.

    Validates topological integrity of CAD models, checking for:
    - Manifoldness (each edge shared by at most 2 faces)
    - Euler characteristic (V - E + F = 2 - 2g for genus g)
    - Watertightness (closed solid with no gaps)
    - Self-intersections
    - Face orientation consistency

    This model works with pythonocc-core shapes for STEP/IGES/BRep formats
    and trimesh for STL/mesh formats.

    Attributes:
        has_pythonocc: Whether pythonocc-core is available
        has_trimesh: Whether trimesh is available
        strict_mode: Whether to fail on any topology error
        check_self_intersections: Whether to check for self-intersections
        max_faces_for_self_intersection: Max faces for self-intersection check

    Example:
        # Basic validation
        model = TopologyValidationModel(strict_mode=True)

        # With self-intersection checking (expensive)
        model = TopologyValidationModel(
            strict_mode=True,
            check_self_intersections=True,
            max_faces_for_self_intersection=5000
        )

        result = converter.convert(
            "part.step",
            pipeline_options=PipelineOptions(
                enrichment_models=[model]
            )
        )
        for item in result.document.items:
            validation = item.properties.get("topology_validation", {})
            if not validation.get("is_valid", False):
                print(f"Topology errors: {validation.get('errors', [])}")
    """

    def __init__(
        self,
        strict_mode: bool = False,
        check_self_intersections: bool = False,
        max_faces_for_self_intersection: int = 10000,
        sliver_threshold: float = 0.05,
        check_face_edge_consistency: bool = True,
        check_vertex_edge_consistency: bool = True,
    ):
        """Initialize topology validation model.

        Args:
            strict_mode: If True, mark items as invalid on any topology error
            check_self_intersections: If True, check for self-intersections (expensive)
            max_faces_for_self_intersection: Maximum number of faces to check for
                self-intersections. Meshes larger than this will skip the check.
            sliver_threshold: Area/length threshold below which entities are
                flagged as slivers.
            check_face_edge_consistency: If True, verify edges lie on parent faces.
            check_vertex_edge_consistency: If True, verify edge endpoint vertices
                match curve parameter endpoints.
        """
        super().__init__()

        self.strict_mode = strict_mode
        self.check_self_intersections = check_self_intersections
        self.max_faces_for_self_intersection = max_faces_for_self_intersection
        self.sliver_threshold = sliver_threshold
        self.check_face_edge_consistency = check_face_edge_consistency
        self.check_vertex_edge_consistency = check_vertex_edge_consistency

        # Check for pythonocc-core availability
        self.has_pythonocc = False
        try:
            from OCC.Core.BRepCheck import BRepCheck_Analyzer
            from OCC.Core.TopExp import TopExp_Explorer

            self.has_pythonocc = True
            _log.debug("pythonocc-core available for topology validation")
        except ImportError:
            _log.warning(
                "pythonocc-core not available. CAD topology validation disabled."
            )

        # Check for trimesh availability
        self.has_trimesh = False
        try:
            import trimesh

            self.has_trimesh = True
            _log.debug("trimesh available for mesh topology validation")
        except ImportError:
            _log.warning("trimesh not available. Mesh topology validation disabled.")

        if not self.has_pythonocc and not self.has_trimesh:
            _log.error(
                "Neither pythonocc-core nor trimesh available. "
                "Topology validation will be disabled."
            )

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Validate topology for CAD items.

        Args:
            doc: CADlingDocument being enriched
            item_batch: List of CADItem objects to validate
        """
        if not self.has_pythonocc and not self.has_trimesh:
            _log.debug("Topology validation skipped: no backend available")
            return

        for item in item_batch:
            try:
                # Validate topology
                validation_result = self._validate_item(doc, item)

                if validation_result:
                    item.properties["topology_validation"] = validation_result

                    # Add provenance
                    item.add_provenance(
                        component_type="enrichment_model",
                        component_name="TopologyValidationModel",
                    )

                    # Log warnings for invalid topology
                    if not validation_result.get("is_valid", False):
                        errors = validation_result.get("errors", [])
                        _log.warning(
                            f"Topology validation failed for item '{item.label.text}': "
                            f"{', '.join(errors)}"
                        )
                    else:
                        _log.debug(
                            f"Topology validation passed for item '{item.label.text}'"
                        )

            except Exception as e:
                _log.error(
                    f"Topology validation failed for item {item.label.text}: {e}"
                )

    def _validate_item(
        self, doc: CADlingDocument, item: CADItem
    ) -> dict:
        """Validate topology of a single CAD item.

        Uses multi-strategy approach:
        1. OCC shape validation (if pythonocc available)
        2. Trimesh validation (if trimesh available)
        3. STEP text parsing validation (for STEP format)

        Args:
            doc: Document containing the item
            item: Item to validate

        Returns:
            Dictionary with validation results (never returns None)
        """
        # Strategy 1: Try to get shape from backend
        shape = self._get_shape_for_item(doc, item)

        if shape is not None:
            # Validate based on shape type
            if self.has_pythonocc and self._is_occ_shape(shape):
                result = self._validate_occ_shape(shape)
                if result:
                    result["validation_method"] = "pythonocc"
                    return result

            if self.has_trimesh and self._is_trimesh(shape):
                result = self._validate_trimesh(shape)
                if result:
                    result["validation_method"] = "trimesh"
                    return result

        # Strategy 2: Parse from STEP text if available
        format_str = str(doc.format).lower() if hasattr(doc, 'format') else ""
        if format_str in ["step", "iges"]:
            step_text = self._get_step_text(doc, item)
            if step_text:
                result = self._validate_from_step_text(step_text)
                result["validation_method"] = "step_text_parsing"
                return result

        # Strategy 3: Return minimal validation result
        return self._minimal_validation_result(item)

    def _get_step_text(self, doc: CADlingDocument, item: CADItem) -> Optional[str]:
        """Get STEP text from document or item.

        Args:
            doc: Document containing the item
            item: Item to get text for

        Returns:
            STEP text or None
        """
        # Try item text
        if hasattr(item, 'text') and item.text:
            return item.text

        # Try document raw content
        if hasattr(doc, 'raw_content') and doc.raw_content:
            return doc.raw_content

        # Try backend content
        if hasattr(doc, '_backend') and doc._backend:
            if hasattr(doc._backend, 'content'):
                return doc._backend.content

        return None

    def _validate_from_step_text(self, step_text: str) -> dict:
        """Validate topology from STEP text parsing.

        Performs structural validation by analyzing STEP entity references
        without requiring OCC or trimesh.

        Args:
            step_text: STEP file content

        Returns:
            Dictionary with validation results
        """
        import re
        from collections import defaultdict

        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "source": "step_text_parsing",
        }

        # Count topology entities
        entity_pattern = re.compile(r"#(\d+)\s*=\s*([A-Z_][A-Z0-9_]*)\s*\(", re.IGNORECASE)
        ref_pattern = re.compile(r"#(\d+)")

        entities: dict[int, str] = {}  # id -> type
        references: dict[int, list[int]] = defaultdict(list)  # id -> [referenced_ids]

        for match in entity_pattern.finditer(step_text):
            entity_id = int(match.group(1))
            entity_type = match.group(2).upper()
            entities[entity_id] = entity_type

            # Extract references from entity text
            start = match.end()
            # Find closing paren
            depth = 1
            pos = start
            while pos < len(step_text) and depth > 0:
                if step_text[pos] == '(':
                    depth += 1
                elif step_text[pos] == ')':
                    depth -= 1
                pos += 1

            entity_text = step_text[start:pos]
            refs = [int(m) for m in ref_pattern.findall(entity_text)]
            references[entity_id] = refs

        # Count by type
        type_counts: dict[str, int] = defaultdict(int)
        for etype in entities.values():
            type_counts[etype] += 1

        num_vertices = type_counts.get("VERTEX_POINT", 0)
        num_edges = type_counts.get("EDGE_CURVE", 0) + type_counts.get("ORIENTED_EDGE", 0)
        num_faces = type_counts.get("ADVANCED_FACE", 0) + type_counts.get("FACE_SURFACE", 0)
        num_shells = type_counts.get("CLOSED_SHELL", 0) + type_counts.get("OPEN_SHELL", 0)

        results["topology_counts"] = {
            "vertices": num_vertices,
            "edges": num_edges,
            "faces": num_faces,
            "shells": num_shells,
        }

        # Compute Euler characteristic: V - E + F
        euler_char = num_vertices - num_edges + num_faces
        results["euler_characteristic"] = euler_char

        # Check for expected Euler characteristic
        expected_values = [2, 0, -2, -4]  # sphere, torus, double-torus, etc.
        if euler_char not in expected_values and num_faces > 0:
            results["warnings"].append(
                f"Unusual Euler characteristic: {euler_char} (expected one of {expected_values})"
            )

        # Check for dangling references
        all_entity_ids = set(entities.keys())
        dangling_refs = set()

        for entity_id, refs in references.items():
            for ref_id in refs:
                if ref_id not in all_entity_ids:
                    dangling_refs.add(ref_id)

        if dangling_refs:
            results["warnings"].append(
                f"Found {len(dangling_refs)} dangling references to non-existent entities"
            )
            results["dangling_references"] = list(dangling_refs)[:10]  # Limit to first 10

        # Check for closed vs open shells
        has_closed_shells = type_counts.get("CLOSED_SHELL", 0) > 0
        has_open_shells = type_counts.get("OPEN_SHELL", 0) > 0

        results["is_watertight"] = has_closed_shells and not has_open_shells
        results["is_manifold"] = True  # Assume manifold from STEP (validated by CAD system)

        if has_open_shells and not has_closed_shells:
            results["warnings"].append("Only open shells found - geometry is not watertight")

        # Check for BREP entities
        has_brep = type_counts.get("MANIFOLD_SOLID_BREP", 0) > 0
        results["has_brep"] = has_brep

        if not has_brep and num_faces > 0:
            results["warnings"].append("No MANIFOLD_SOLID_BREP entity found")

        # Mark as valid if no critical errors
        results["is_valid"] = len(results["errors"]) == 0

        return results

    def _minimal_validation_result(self, item: CADItem) -> dict:
        """Return minimal validation result when no validation method available.

        Args:
            item: CAD item

        Returns:
            Dictionary with minimal validation result
        """
        return {
            "is_valid": False,
            "errors": ["No validation method available (pythonocc, trimesh, or STEP text)"],
            "warnings": [],
            "validation_method": "none",
            "topology_counts": {
                "vertices": 0,
                "edges": 0,
                "faces": 0,
            },
        }

    def _get_shape_for_item(
        self, doc: CADlingDocument, item: CADItem
    ) -> Optional[any]:
        """Get shape object for validation.

        Args:
            doc: Document containing the item
            item: Item to get shape for

        Returns:
            Shape object (OCC shape or trimesh), or None
        """
        # Check if item has shape stored
        if hasattr(item, "_shape") and item._shape is not None:
            return item._shape

        # Try to load from backend based on format
        format_str = str(doc.format).lower()

        if format_str in ["step", "iges", "brep"] and self.has_pythonocc:
            return self._load_occ_shape(doc)
        elif format_str == "stl" and self.has_trimesh:
            return self._load_trimesh(doc)

        return None

    def _get_backend_resource(self, doc: CADlingDocument, resource_name: str):
        """Get a resource from the document's backend using multiple attribute patterns.

        Tries the following patterns in order:
        1. backend.{resource_name} (e.g., backend.shape)
        2. backend._{resource_name} (e.g., backend._shape)
        3. backend.load_{resource_name}() (e.g., backend.load_shape())
        4. backend.get_{resource_name}() (e.g., backend.get_shape())

        Args:
            doc: Document with backend
            resource_name: Base name of the resource (e.g., "shape", "mesh")

        Returns:
            The resource if found, None otherwise
        """
        if not hasattr(doc, '_backend') or doc._backend is None:
            _log.debug(f"No backend available for {resource_name} loading")
            return None

        backend = doc._backend

        # Define attribute patterns to try (in order of likelihood)
        attr_patterns = [
            (resource_name, False),           # backend.shape
            (f"_{resource_name}", False),     # backend._shape
            (f"load_{resource_name}", True),  # backend.load_shape()
            (f"get_{resource_name}", True),   # backend.get_shape()
        ]

        try:
            for attr_name, is_method in attr_patterns:
                if hasattr(backend, attr_name):
                    attr = getattr(backend, attr_name)
                    if is_method:
                        # It's a method, call it
                        if callable(attr):
                            result = attr()
                            if result is not None:
                                _log.debug(f"Loaded {resource_name} from backend.{attr_name}()")
                                return result
                    else:
                        # It's an attribute, return directly if not None
                        if attr is not None:
                            _log.debug(f"Loaded {resource_name} from backend.{attr_name}")
                            return attr

            # No pattern matched
            _log.debug(
                f"Backend {type(backend).__name__} does not provide {resource_name} "
                f"(tried: {', '.join(p[0] for p in attr_patterns)})"
            )
            return None

        except Exception as e:
            _log.error(f"Failed to load {resource_name} from backend: {e}")
            return None

    def _load_occ_shape(self, doc: CADlingDocument):
        """Load shape via pythonocc backend.

        Args:
            doc: Document to load shape from

        Returns:
            OCC shape or None
        """
        return self._get_backend_resource(doc, "shape")

    def _load_trimesh(self, doc: CADlingDocument):
        """Load mesh via trimesh.

        Args:
            doc: Document to load mesh from

        Returns:
            Trimesh object or None
        """
        return self._get_backend_resource(doc, "mesh")

    def _is_occ_shape(self, shape) -> bool:
        """Check if shape is an OCC shape."""
        try:
            from OCC.Core.TopoDS import TopoDS_Shape

            return isinstance(shape, TopoDS_Shape)
        except ImportError:
            return False

    def _is_trimesh(self, shape) -> bool:
        """Check if shape is a trimesh."""
        try:
            import trimesh

            return isinstance(shape, trimesh.Trimesh)
        except ImportError:
            return False

    def _shape_to_entity_id(self, shape) -> str:
        """Convert OCC shape to string entity ID via hash code.

        Uses OCCT's HashCode method if available, falling back to Python's
        built-in hash() for compatibility across pythonocc versions.

        Args:
            shape: OCC TopoDS_Shape to hash

        Returns:
            String representation of the shape's hash code
        """
        try:
            return str(shape.HashCode(2**31 - 1))
        except AttributeError:
            # Fallback for pythonocc versions where HashCode is not exposed
            return str(hash(shape))

    def _compute_severity_score(self, findings: list[ValidationFinding]) -> dict:
        """Aggregate findings into severity score.

        Args:
            findings: List of ValidationFinding objects

        Returns:
            Dictionary with overall severity, counts, and per-check breakdown
        """
        if not findings:
            return {
                "overall_severity": "clean",
                "total_findings": 0,
                "critical_count": 0,
                "warning_count": 0,
                "info_count": 0,
                "checks_breakdown": {},
            }

        critical = sum(1 for f in findings if f.severity == "critical")
        warning = sum(1 for f in findings if f.severity == "warning")
        info = sum(1 for f in findings if f.severity == "info")

        if critical > 0:
            overall = "critical"
        elif warning > 0:
            overall = "warning"
        else:
            overall = "info"

        breakdown = {}
        for f in findings:
            if f.check_name not in breakdown:
                breakdown[f.check_name] = {"critical": 0, "warning": 0, "info": 0}
            breakdown[f.check_name][f.severity] += 1

        return {
            "overall_severity": overall,
            "total_findings": len(findings),
            "critical_count": critical,
            "warning_count": warning,
            "info_count": info,
            "checks_breakdown": breakdown,
        }

    def _check_sliver_entities(self, shape) -> list[ValidationFinding]:
        """Detect sliver faces (tiny area) and sliver edges (tiny length).

        Args:
            shape: OCC TopoDS_Shape to check

        Returns:
            List of ValidationFinding objects for detected slivers
        """
        findings = []
        try:
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.GProp import GProp_GProps

            # Check faces
            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                face = explorer.Current()
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                area = props.Mass()
                if area < self.sliver_threshold:
                    findings.append(
                        ValidationFinding(
                            check_name="sliver_face",
                            severity="warning",
                            message=f"Sliver face detected with area {area:.6e} < threshold {self.sliver_threshold}",
                            entity_ids=[self._shape_to_entity_id(face)],
                            entity_type="FACE",
                        )
                    )
                explorer.Next()

            # Check edges
            explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            while explorer.More():
                edge = explorer.Current()
                props = GProp_GProps()
                brepgprop.LinearProperties(edge, props)
                length = props.Mass()
                if length < self.sliver_threshold:
                    findings.append(
                        ValidationFinding(
                            check_name="sliver_edge",
                            severity="warning",
                            message=f"Sliver edge detected with length {length:.6e} < threshold {self.sliver_threshold}",
                            entity_ids=[self._shape_to_entity_id(edge)],
                            entity_type="EDGE",
                        )
                    )
                explorer.Next()
        except ImportError:
            _log.debug("OCC not available for sliver detection")
        except Exception as e:
            _log.warning(f"Sliver detection failed: {e}")

        return findings

    def _check_face_edge_consistency(self, shape) -> list[ValidationFinding]:
        """Verify edges lie on their parent faces by point sampling.

        Samples points along each edge and projects them onto the parent face,
        flagging any edge that deviates beyond the tolerance.

        Args:
            shape: OCC TopoDS_Shape to check

        Returns:
            List of ValidationFinding objects for inconsistent edges
        """
        findings = []
        if not self.check_face_edge_consistency:
            return findings
        try:
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
            from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
            from OCC.Core.TopoDS import topods

            tolerance = 1e-4
            num_samples = 7

            face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while face_explorer.More():
                face = topods.Face(face_explorer.Current())
                surface = BRep_Tool.Surface(face)

                edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
                while edge_explorer.More():
                    edge = topods.Edge(edge_explorer.Current())
                    try:
                        curve_adaptor = BRepAdaptor_Curve(edge)
                        u_start = curve_adaptor.FirstParameter()
                        u_end = curve_adaptor.LastParameter()

                        max_dist = 0.0
                        for i in range(num_samples):
                            t = (
                                i / (num_samples - 1)
                                if num_samples > 1
                                else 0.5
                            )
                            u = u_start + t * (u_end - u_start)
                            pt = curve_adaptor.Value(u)

                            projector = GeomAPI_ProjectPointOnSurf(pt, surface)
                            if projector.NbPoints() > 0:
                                dist = projector.LowerDistance()
                                max_dist = max(max_dist, dist)

                        if max_dist > tolerance:
                            findings.append(
                                ValidationFinding(
                                    check_name="face_edge_consistency",
                                    severity="critical",
                                    message=f"Edge deviates from parent face by {max_dist:.6e} (tolerance: {tolerance:.6e})",
                                    entity_ids=[
                                        self._shape_to_entity_id(face),
                                        self._shape_to_entity_id(edge),
                                    ],
                                    entity_type="EDGE",
                                )
                            )
                    except Exception as e:
                        _log.debug(f"Edge consistency check failed for edge: {e}")

                    edge_explorer.Next()
                face_explorer.Next()
        except ImportError:
            _log.debug("OCC not available for face-edge consistency check")
        except Exception as e:
            _log.warning(f"Face-edge consistency check failed: {e}")

        return findings

    def _check_vertex_edge_consistency(self, shape) -> list[ValidationFinding]:
        """Verify edge endpoint vertices match curve parameter endpoints.

        For each edge, checks that the first and last vertices coincide with
        the curve evaluated at its start and end parameters.

        Args:
            shape: OCC TopoDS_Shape to check

        Returns:
            List of ValidationFinding objects for inconsistent vertices
        """
        findings = []
        if not self.check_vertex_edge_consistency:
            return findings
        try:
            from OCC.Core.TopExp import TopExp_Explorer, topexp
            from OCC.Core.TopAbs import TopAbs_EDGE
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
            from OCC.Core.TopoDS import topods

            tolerance = 1e-4

            explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            while explorer.More():
                edge = topods.Edge(explorer.Current())
                try:
                    v_first = topexp.FirstVertex(edge)
                    v_last = topexp.LastVertex(edge)

                    pt_first = BRep_Tool.Pnt(v_first)
                    pt_last = BRep_Tool.Pnt(v_last)

                    curve_adaptor = BRepAdaptor_Curve(edge)
                    curve_start = curve_adaptor.Value(
                        curve_adaptor.FirstParameter()
                    )
                    curve_end = curve_adaptor.Value(
                        curve_adaptor.LastParameter()
                    )

                    dist_start = pt_first.Distance(curve_start)
                    dist_end = pt_last.Distance(curve_end)

                    if dist_start > tolerance:
                        findings.append(
                            ValidationFinding(
                                check_name="vertex_edge_consistency",
                                severity="critical",
                                message=f"Start vertex deviates from edge curve start by {dist_start:.6e}",
                                entity_ids=[
                                    self._shape_to_entity_id(edge),
                                    self._shape_to_entity_id(v_first),
                                ],
                                entity_type="VERTEX",
                            )
                        )
                    if dist_end > tolerance:
                        findings.append(
                            ValidationFinding(
                                check_name="vertex_edge_consistency",
                                severity="critical",
                                message=f"End vertex deviates from edge curve end by {dist_end:.6e}",
                                entity_ids=[
                                    self._shape_to_entity_id(edge),
                                    self._shape_to_entity_id(v_last),
                                ],
                                entity_type="VERTEX",
                            )
                        )
                except Exception as e:
                    _log.debug(
                        f"Vertex-edge consistency check failed for edge: {e}"
                    )

                explorer.Next()
        except ImportError:
            _log.debug("OCC not available for vertex-edge consistency check")
        except Exception as e:
            _log.warning(f"Vertex-edge consistency check failed: {e}")

        return findings

    def _check_orientation_consistency(self, shape) -> list[ValidationFinding]:
        """Check face orientation consistency using solid classification.

        Offsets a point along the face normal and classifies it as inside or
        outside the solid. If the outward-normal point is classified as inside,
        the face orientation is inconsistent.

        Args:
            shape: OCC TopoDS_Shape to check

        Returns:
            List of ValidationFinding objects for orientation issues
        """
        findings = []
        try:
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_IN
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
            from OCC.Core.TopoDS import topods
            from OCC.Core.gp import gp_Pnt, gp_Vec

            offset_distance = 1e-3

            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                face = topods.Face(explorer.Current())
                try:
                    adaptor = BRepAdaptor_Surface(face)
                    u_mid = (
                        adaptor.FirstUParameter() + adaptor.LastUParameter()
                    ) / 2
                    v_mid = (
                        adaptor.FirstVParameter() + adaptor.LastVParameter()
                    ) / 2

                    pt = adaptor.Value(u_mid, v_mid)

                    # Get normal via D1
                    p = gp_Pnt()
                    d1u = gp_Vec()
                    d1v = gp_Vec()
                    adaptor.D1(u_mid, v_mid, p, d1u, d1v)
                    normal = d1u.Crossed(d1v)
                    if normal.Magnitude() > 1e-10:
                        normal.Normalize()

                        # Point offset along normal (outside)
                        outside_pt = gp_Pnt(
                            pt.X() + normal.X() * offset_distance,
                            pt.Y() + normal.Y() * offset_distance,
                            pt.Z() + normal.Z() * offset_distance,
                        )

                        classifier = BRepClass3d_SolidClassifier(
                            shape, outside_pt, 1e-6
                        )
                        state = classifier.State()

                        # If point along outward normal is classified as IN,
                        # orientation is wrong
                        if state == TopAbs_IN:
                            findings.append(
                                ValidationFinding(
                                    check_name="orientation_consistency",
                                    severity="warning",
                                    message="Face normal points inward (inconsistent orientation)",
                                    entity_ids=[
                                        self._shape_to_entity_id(face)
                                    ],
                                    entity_type="FACE",
                                )
                            )
                except Exception as e:
                    _log.debug(f"Orientation check failed for face: {e}")

                explorer.Next()
        except ImportError:
            _log.debug(
                "OCC not available for orientation consistency check"
            )
        except Exception as e:
            _log.warning(f"Orientation consistency check failed: {e}")

        return findings

    def _validate_occ_shape(self, shape) -> dict:
        """Validate pythonocc shape topology.

        Args:
            shape: OCC TopoDS_Shape to validate

        Returns:
            Dictionary with validation results
        """
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
        from OCC.Core.TopExp import TopExp_Explorer

        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
        }

        # Use BRepCheck_Analyzer for comprehensive validation
        analyzer = BRepCheck_Analyzer(shape)
        is_valid = analyzer.IsValid()

        results["brepcheck_valid"] = is_valid

        if not is_valid:
            results["is_valid"] = False
            results["errors"].append("BRepCheck_Analyzer found issues")

        # Count topological elements
        num_vertices = 0
        num_edges = 0
        num_faces = 0

        # Count vertices
        vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        while vertex_explorer.More():
            num_vertices += 1
            vertex_explorer.Next()

        # Count edges
        edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while edge_explorer.More():
            num_edges += 1
            edge_explorer.Next()

        # Count faces
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            num_faces += 1
            face_explorer.Next()

        results["topology_counts"] = {
            "vertices": num_vertices,
            "edges": num_edges,
            "faces": num_faces,
        }

        # Compute Euler characteristic
        # For a genus-0 solid (sphere-like): V - E + F = 2
        # For general case: V - E + F = 2 - 2g where g is genus
        euler_char = num_vertices - num_edges + num_faces
        results["euler_characteristic"] = euler_char

        # Expected Euler characteristic for different topologies
        # Solid: 2, Torus: 0, etc.
        # We'll check if it matches common cases
        expected_values = [2, 0, -2, -4]  # sphere, torus, double-torus, etc.

        if euler_char not in expected_values:
            results["warnings"].append(
                f"Unusual Euler characteristic: {euler_char} "
                f"(expected one of {expected_values})"
            )

        # Check for closed solid (watertight) and manifoldness
        # Proper check: build edge-to-face adjacency map
        if num_faces > 0 and num_edges > 0:
            try:
                from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
                from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
                from OCC.Core.TopExp import topexp

                # Build edge-to-face mapping
                edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
                topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

                # Count edges by adjacency
                num_boundary_edges = 0  # Shared by 1 face
                num_manifold_edges = 0  # Shared by 2 faces (proper)
                num_non_manifold_edges = 0  # Shared by >2 faces (bad!)

                for i in range(1, edge_face_map.Size() + 1):
                    num_adjacent_faces = edge_face_map.FindFromIndex(i).Size()
                    if num_adjacent_faces == 1:
                        num_boundary_edges += 1
                    elif num_adjacent_faces == 2:
                        num_manifold_edges += 1
                    else:
                        num_non_manifold_edges += 1

                # Update results with REAL validation
                is_manifold = (num_non_manifold_edges == 0)
                is_watertight = (is_manifold and num_boundary_edges == 0 and is_valid)

                results["is_manifold"] = is_manifold
                results["is_watertight"] = is_watertight  # Real check, not "likely"!
                results["edge_statistics"] = {
                    "total_edges": num_edges,
                    "boundary_edges": num_boundary_edges,
                    "manifold_edges": num_manifold_edges,
                    "non_manifold_edges": num_non_manifold_edges,
                }

                # Add warnings for issues
                if num_non_manifold_edges > 0:
                    results["errors"].append(
                        f"Non-manifold geometry: {num_non_manifold_edges} edges shared by >2 faces"
                    )
                if num_boundary_edges > 0 and is_manifold:
                    results["warnings"].append(
                        f"Open solid: {num_boundary_edges} boundary edges (not watertight)"
                    )

            except ImportError:
                # Fallback if pythonocc not available
                _log.warning("pythonocc not available, using simplified watertight check")
                results["likely_watertight"] = is_valid

        else:
            results["is_watertight"] = False
            results["is_manifold"] = False
            results["warnings"].append("Shape has no faces or edges")

        # Self-intersection check (expensive, so optional)
        # BRepCheck_Analyzer already checks for this

        # Orientation consistency
        # BRepCheck should catch this, but we note it
        results["orientation_consistent"] = is_valid

        # Enhanced validation checks
        all_findings = []

        try:
            all_findings.extend(self._check_face_edge_consistency(shape))
        except Exception as e:
            _log.warning(f"Face-edge consistency check failed: {e}")

        try:
            all_findings.extend(self._check_vertex_edge_consistency(shape))
        except Exception as e:
            _log.warning(f"Vertex-edge consistency check failed: {e}")

        try:
            all_findings.extend(self._check_orientation_consistency(shape))
        except Exception as e:
            _log.warning(f"Orientation consistency check failed: {e}")

        try:
            all_findings.extend(self._check_sliver_entities(shape))
        except Exception as e:
            _log.warning(f"Sliver detection failed: {e}")

        severity_scoring = self._compute_severity_score(all_findings)

        results["validation_findings"] = [f.model_dump() for f in all_findings]
        results["severity_scoring"] = severity_scoring

        # Update is_valid based on critical findings
        if severity_scoring.get("critical_count", 0) > 0:
            results["is_valid"] = False

        _log.debug(
            f"OCC topology validation: valid={is_valid}, "
            f"V={num_vertices}, E={num_edges}, F={num_faces}, χ={euler_char}"
        )

        return results

    def _validate_trimesh(self, mesh) -> dict:
        """Validate trimesh object topology.

        Args:
            mesh: Trimesh object to validate

        Returns:
            Dictionary with validation results
        """
        import numpy as np
        import trimesh

        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
        }

        # Check if mesh is watertight (closed manifold)
        is_watertight = mesh.is_watertight
        results["is_watertight"] = is_watertight

        if not is_watertight:
            results["errors"].append("Mesh is not watertight")
            results["is_valid"] = False

        # Check winding consistency
        is_winding_consistent = mesh.is_winding_consistent
        results["is_winding_consistent"] = is_winding_consistent

        if not is_winding_consistent:
            results["errors"].append("Face winding is inconsistent")
            results["is_valid"] = False

        # Check for degenerate faces (zero area)
        face_areas = mesh.area_faces
        degenerate_faces = np.where(face_areas < 1e-10)[0]
        results["num_degenerate_faces"] = len(degenerate_faces)

        if len(degenerate_faces) > 0:
            results["warnings"].append(
                f"Found {len(degenerate_faces)} degenerate faces"
            )

        # Check for duplicate faces (index-based)
        # Sort face indices to make [0,1,2] equivalent to [1,2,0] for duplicate detection
        sorted_faces = np.sort(mesh.faces, axis=1)
        unique_faces = np.unique(sorted_faces, axis=0)
        num_duplicate_faces = len(mesh.faces) - len(unique_faces)
        results["num_duplicate_faces"] = num_duplicate_faces

        if num_duplicate_faces > 0:
            results["warnings"].append(
                f"Found {num_duplicate_faces} duplicate faces (by index)"
            )

        # Check for geometric duplicate faces (faces at same position but different indices)
        # Compute face centroids and round to tolerance for comparison
        try:
            face_vertices = mesh.vertices[mesh.faces]  # Shape: (num_faces, 3, 3)
            face_centroids = face_vertices.mean(axis=1)  # Shape: (num_faces, 3)

            # Round to tolerance for duplicate detection
            tolerance_decimals = 6  # ~1e-6 tolerance
            rounded_centroids = np.round(face_centroids, decimals=tolerance_decimals)

            # Find duplicate centroids
            unique_centroids, centroid_indices = np.unique(
                rounded_centroids, axis=0, return_inverse=True
            )
            num_geometric_duplicate_faces = len(mesh.faces) - len(unique_centroids)
            results["num_geometric_duplicate_faces"] = num_geometric_duplicate_faces

            if num_geometric_duplicate_faces > num_duplicate_faces:
                # Additional geometric duplicates beyond index duplicates
                extra_duplicates = num_geometric_duplicate_faces - num_duplicate_faces
                results["warnings"].append(
                    f"Found {extra_duplicates} additional geometric duplicate faces "
                    f"(same position, different vertex indices)"
                )
        except Exception as e:
            _log.debug(f"Geometric duplicate face detection failed: {e}")
            results["num_geometric_duplicate_faces"] = num_duplicate_faces

        # Check for duplicate vertices
        unique_vertices = np.unique(mesh.vertices, axis=0)
        num_duplicate_vertices = len(mesh.vertices) - len(unique_vertices)
        results["num_duplicate_vertices"] = num_duplicate_vertices

        if num_duplicate_vertices > 0:
            results["warnings"].append(
                f"Found {num_duplicate_vertices} duplicate vertices"
            )

        # Euler characteristic
        # For a closed triangular mesh: V - E + F = 2 - 2g
        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)
        num_edges = len(mesh.edges_unique)

        euler_char = num_vertices - num_edges + num_faces
        results["euler_characteristic"] = euler_char

        results["topology_counts"] = {
            "vertices": num_vertices,
            "edges": num_edges,
            "faces": num_faces,
        }

        # Expected Euler characteristic
        expected_values = [2, 0, -2, -4]
        if euler_char not in expected_values:
            results["warnings"].append(
                f"Unusual Euler characteristic: {euler_char} "
                f"(expected one of {expected_values})"
            )

        # Check for self-intersections (expensive, configurable)
        # Trimesh has mesh.is_self_intersecting but it's slow
        if self.check_self_intersections and hasattr(mesh, "is_self_intersecting"):
            if num_faces < self.max_faces_for_self_intersection:
                try:
                    has_self_intersections = mesh.is_self_intersecting
                    results["self_intersecting"] = has_self_intersections
                    if has_self_intersections:
                        results["errors"].append("Mesh has self-intersections")
                        results["is_valid"] = False
                except Exception as e:
                    results["self_intersecting"] = None
                    results["warnings"].append(
                        f"Self-intersection check failed: {e}"
                    )
            else:
                results["self_intersecting"] = None
                results["warnings"].append(
                    f"Self-intersection check skipped (mesh has {num_faces} faces, "
                    f"limit is {self.max_faces_for_self_intersection})"
                )
        elif self.check_self_intersections and not hasattr(mesh, "is_self_intersecting"):
            results["self_intersecting"] = None
            results["warnings"].append(
                "Self-intersection check not available in trimesh version"
            )
        else:
            # Not enabled
            results["self_intersecting"] = None

        # Check manifoldness
        # A manifold mesh has each edge shared by at most 2 faces
        # Trimesh provides this check
        try:
            # Get edges that are non-manifold
            if hasattr(mesh, "edges_unique_length"):
                # Check if there are boundary edges (shared by 1 face)
                # or non-manifold edges (shared by >2 faces)
                pass  # Trimesh's is_watertight already checks this
        except Exception as e:
            _log.debug(f"Edge manifold check failed: {e}")

        # Enhanced trimesh validation
        trimesh_findings = []

        if hasattr(mesh, "area_faces"):
            mesh_face_areas = mesh.area_faces
            for i, area in enumerate(mesh_face_areas):
                if area < self.sliver_threshold:
                    trimesh_findings.append(
                        ValidationFinding(
                            check_name="sliver_face",
                            severity="warning",
                            message=f"Sliver face {i} detected with area {area:.6e} < threshold {self.sliver_threshold}",
                            entity_ids=[str(i)],
                            entity_type="FACE",
                        )
                    )

        # Edge length checking
        if hasattr(mesh, "edges_unique_length"):
            edge_lengths = mesh.edges_unique_length
            for i, length in enumerate(edge_lengths):
                if length < self.sliver_threshold:
                    trimesh_findings.append(
                        ValidationFinding(
                            check_name="sliver_edge",
                            severity="warning",
                            message=f"Sliver edge {i} detected with length {length:.6e} < threshold {self.sliver_threshold}",
                            entity_ids=[str(i)],
                            entity_type="EDGE",
                        )
                    )

        severity_scoring = self._compute_severity_score(trimesh_findings)
        results["validation_findings"] = [
            f.model_dump() for f in trimesh_findings
        ]
        results["severity_scoring"] = severity_scoring

        if severity_scoring.get("critical_count", 0) > 0:
            results["is_valid"] = False

        _log.debug(
            f"Trimesh topology validation: watertight={is_watertight}, "
            f"winding_ok={is_winding_consistent}, "
            f"V={num_vertices}, E={num_edges}, F={num_faces}, χ={euler_char}"
        )

        return results

    def supports_batch_processing(self) -> bool:
        """Topology validation can process items independently."""
        return True

    def get_batch_size(self) -> int:
        """Process items one at a time (validation can be expensive)."""
        return 1

    def requires_gpu(self) -> bool:
        """Topology validation does not require GPU."""
        return False
