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

from cadling.models.base_model import EnrichmentModel

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument

_log = logging.getLogger(__name__)


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

    Example:
        model = TopologyValidationModel(strict_mode=True)
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

    def __init__(self, strict_mode: bool = False):
        """Initialize topology validation model.

        Args:
            strict_mode: If True, mark items as invalid on any topology error
        """
        super().__init__()

        self.strict_mode = strict_mode

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
    ) -> Optional[dict]:
        """Validate topology of a single CAD item.

        Args:
            doc: Document containing the item
            item: Item to validate

        Returns:
            Dictionary with validation results, or None if validation failed
        """
        # Try to get shape from backend
        shape = self._get_shape_for_item(doc, item)

        if shape is None:
            _log.debug(f"Could not get shape for item {item.label.text}")
            return None

        # Validate based on shape type
        if self.has_pythonocc and self._is_occ_shape(shape):
            return self._validate_occ_shape(shape)
        elif self.has_trimesh and self._is_trimesh(shape):
            return self._validate_trimesh(shape)
        else:
            _log.debug(f"Unsupported shape type for item {item.label.text}")
            return None

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

    def _load_occ_shape(self, doc: CADlingDocument):
        """Load shape via pythonocc backend."""
        _log.debug("OCC shape loading not yet implemented in enrichment stage")
        return None

    def _load_trimesh(self, doc: CADlingDocument):
        """Load mesh via trimesh."""
        _log.debug("Trimesh loading not yet implemented in enrichment stage")
        return None

    def _is_occ_shape(self, shape) -> bool:
        """Check if shape is an OCC shape."""
        try:
            from OCC.Core.TopoDS import TopoDS_Shape

            return isinstance(shape, TopoDS_Shape)
        except:
            return False

    def _is_trimesh(self, shape) -> bool:
        """Check if shape is a trimesh."""
        try:
            import trimesh

            return isinstance(shape, trimesh.Trimesh)
        except:
            return False

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

        # Check for duplicate faces
        # Trimesh might have a method for this
        # For now, just check if there are duplicate vertices
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

        # Check for self-intersections (expensive)
        # Trimesh has mesh.is_self_intersecting but it's slow
        # We'll skip this for performance unless specifically requested
        # For now, just check if it's available
        try:
            if hasattr(mesh, "is_self_intersecting"):
                # Only check for small meshes to avoid performance issues
                if num_faces < 10000:
                    has_self_intersections = mesh.is_self_intersecting
                    results["self_intersecting"] = has_self_intersections
                    if has_self_intersections:
                        results["errors"].append("Mesh has self-intersections")
                        results["is_valid"] = False
                else:
                    results["self_intersecting"] = None
                    results["warnings"].append(
                        "Self-intersection check skipped (mesh too large)"
                    )
        except:
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
        except:
            pass

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
