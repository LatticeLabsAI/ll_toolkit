"""Geometry analysis model for CAD parts.

This module provides geometry analysis capabilities for CAD parts, including:
- Bounding box computation
- Volume and surface area calculation
- Center of mass and inertia tensor computation
- Basic geometric properties extraction

Classes:
    GeometryAnalysisModel: Main model for geometry analysis
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from cadling.models.base_model import EnrichmentModel

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument

_log = logging.getLogger(__name__)


class GeometryAnalysisModel(EnrichmentModel):
    """Geometry analysis enrichment model.

    Analyzes 3D geometry and computes geometric properties such as:
    - Bounding box (3D bounds)
    - Volume
    - Surface area
    - Center of mass
    - Moment of inertia tensor

    This model works with pythonocc-core shapes for STEP/IGES/BRep formats
    and trimesh for STL/mesh formats.

    Attributes:
        has_pythonocc: Whether pythonocc-core is available
        has_trimesh: Whether trimesh is available
        density: Material density for mass/inertia (default: 1.0)

    Example:
        model = GeometryAnalysisModel(density=7.85)  # Steel density g/cm³
        result = converter.convert(
            "part.step",
            pipeline_options=PipelineOptions(
                enrichment_models=[model]
            )
        )
        for item in result.document.items:
            if "geometry_analysis" in item.properties:
                print(f"Volume: {item.properties['geometry_analysis']['volume']}")
    """

    def __init__(self, density: float = 1.0):
        """Initialize geometry analysis model.

        Args:
            density: Material density for mass/inertia calculations (g/cm³)
        """
        super().__init__()

        self.density = density

        # Check for pythonocc-core availability
        self.has_pythonocc = False
        try:
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.GProp import GProp_GProps

            self.has_pythonocc = True
            _log.debug("pythonocc-core available for geometry analysis")
        except ImportError:
            _log.warning("pythonocc-core not available. CAD geometry analysis disabled.")

        # Check for trimesh availability
        self.has_trimesh = False
        try:
            import trimesh

            self.has_trimesh = True
            _log.debug("trimesh available for mesh analysis")
        except ImportError:
            _log.warning("trimesh not available. Mesh analysis disabled.")

        if not self.has_pythonocc and not self.has_trimesh:
            _log.error(
                "Neither pythonocc-core nor trimesh available. "
                "Geometry analysis will be disabled."
            )

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Analyze geometry for CAD items.

        Args:
            doc: CADlingDocument being enriched
            item_batch: List of CADItem objects to analyze
        """
        if not self.has_pythonocc and not self.has_trimesh:
            _log.debug("Geometry analysis skipped: no backend available")
            return

        for item in item_batch:
            try:
                # Determine item format and analyze accordingly
                analysis_result = self._analyze_item(doc, item)

                if analysis_result:
                    item.properties["geometry_analysis"] = analysis_result

                    # Add provenance
                    item.add_provenance(
                        component_type="enrichment_model",
                        component_name="GeometryAnalysisModel",
                    )

                    _log.debug(
                        f"Analyzed geometry for item '{item.label.text}': "
                        f"volume={analysis_result.get('volume', 'N/A')}"
                    )

            except Exception as e:
                _log.error(f"Geometry analysis failed for item {item.label.text}: {e}")

    def _analyze_item(
        self, doc: CADlingDocument, item: CADItem
    ) -> dict:
        """Analyze a single CAD item.

        Uses multi-strategy approach:
        1. OCC shape analysis (if pythonocc available)
        2. Trimesh analysis (if trimesh available)
        3. STEP text parsing (for STEP format)
        4. Item properties aggregation (last resort)

        Args:
            doc: Document containing the item
            item: Item to analyze

        Returns:
            Dictionary with analysis results (never returns None)
        """
        # Strategy 1: Try to get shape from backend
        shape = self._get_shape_for_item(doc, item)

        if shape is not None:
            # Analyze based on shape type
            if self.has_pythonocc and self._is_occ_shape(shape):
                result = self._analyze_occ_shape(shape)
                if result:
                    result["analysis_method"] = "pythonocc"
                    return result

            if self.has_trimesh and self._is_trimesh(shape):
                result = self._analyze_trimesh(shape)
                if result:
                    result["analysis_method"] = "trimesh"
                    return result

        # Strategy 2: Parse from STEP text if available
        format_str = str(doc.format).lower() if hasattr(doc, 'format') else ""
        if format_str in ["step", "iges"]:
            step_text = self._get_step_text(doc, item)
            if step_text:
                result = self._analyze_from_step_text(step_text)
                if result:
                    result["analysis_method"] = "step_text_parsing"
                    return result

        # Strategy 3: Aggregate from item properties
        result = self._analyze_from_item_properties(item)
        result["analysis_method"] = "item_properties"
        return result

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

    def _analyze_from_step_text(self, step_text: str) -> Optional[dict]:
        """Analyze geometry from STEP text parsing.

        Extracts geometric properties directly from STEP entity text
        without requiring OCC or trimesh.

        Args:
            step_text: STEP file content

        Returns:
            Dictionary with analysis results or None
        """
        import re

        results = {
            "source": "step_text_parsing",
        }

        # Extract CARTESIAN_POINT coordinates for bounding box
        point_pattern = re.compile(
            r"CARTESIAN_POINT\s*\([^)]*,\s*\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)\s*\)",
            re.IGNORECASE
        )

        points = []
        for match in point_pattern.finditer(step_text):
            try:
                x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                points.append((x, y, z))
            except ValueError:
                continue

        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            zs = [p[2] for p in points]

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            zmin, zmax = min(zs), max(zs)

            results["bounding_box"] = {
                "xmin": xmin, "xmax": xmax,
                "ymin": ymin, "ymax": ymax,
                "zmin": zmin, "zmax": zmax,
                "dx": xmax - xmin,
                "dy": ymax - ymin,
                "dz": zmax - zmin,
            }

            # Bounding box volume
            bbox_vol = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
            results["bounding_box_volume"] = bbox_vol

            # Center of mass estimate (centroid of points)
            results["center_of_mass"] = {
                "x": sum(xs) / len(xs),
                "y": sum(ys) / len(ys),
                "z": sum(zs) / len(zs),
            }

        # Extract radius from CYLINDRICAL_SURFACE or CIRCLE
        radius_pattern = re.compile(
            r"(?:CYLINDRICAL_SURFACE|SPHERICAL_SURFACE|CIRCLE)\s*\([^,]*,[^,]*,\s*([-\d.eE+]+)\s*\)",
            re.IGNORECASE
        )

        radii = []
        for match in radius_pattern.finditer(step_text):
            try:
                radii.append(float(match.group(1)))
            except ValueError:
                continue

        if radii:
            results["detected_radii"] = radii
            results["max_radius"] = max(radii)
            results["min_radius"] = min(radii)

        # Count topology entities for volume estimation
        shell_count = len(re.findall(r"(?:CLOSED_SHELL|OPEN_SHELL)", step_text, re.IGNORECASE))
        face_count = len(re.findall(r"ADVANCED_FACE", step_text, re.IGNORECASE))
        edge_count = len(re.findall(r"EDGE_CURVE", step_text, re.IGNORECASE))

        results["topology_counts"] = {
            "shells": shell_count,
            "faces": face_count,
            "edges": edge_count,
            "points": len(points),
        }

        # Estimate volume from bounding box with fill factor
        if "bounding_box_volume" in results and results["bounding_box_volume"] > 0:
            # Use empirical fill factor based on entity counts
            fill_factor = 0.6 if shell_count > 0 else 0.4
            results["volume"] = results["bounding_box_volume"] * fill_factor
            results["volume_estimation_method"] = "bounding_box_fill_factor"

        # Surface area estimation (rough)
        if "bounding_box" in results:
            bbox = results["bounding_box"]
            # Surface area of bounding box
            bbox_sa = 2 * (bbox["dx"] * bbox["dy"] + bbox["dy"] * bbox["dz"] + bbox["dx"] * bbox["dz"])
            # Apply factor for typical solid
            results["surface_area"] = bbox_sa * 0.8
            results["surface_area_estimation_method"] = "bounding_box_factor"

        # Compactness
        if "volume" in results and "bounding_box_volume" in results and results["bounding_box_volume"] > 0:
            results["compactness"] = results["volume"] / results["bounding_box_volume"]

        # Surface to volume ratio
        if "volume" in results and "surface_area" in results and results["volume"] > 0:
            results["surface_to_volume_ratio"] = results["surface_area"] / results["volume"]

        return results if len(results) > 1 else None

    def _analyze_from_item_properties(self, item: CADItem) -> dict:
        """Aggregate geometry analysis from item properties.

        Uses pre-computed properties stored on the item as a fallback
        when shape analysis is not available.

        Args:
            item: CAD item with properties

        Returns:
            Dictionary with analysis results
        """
        results = {
            "source": "item_properties",
        }

        props = getattr(item, 'properties', {})

        # Copy relevant geometry properties
        if "bounding_box" in props:
            results["bounding_box"] = props["bounding_box"]

        if "volume" in props:
            results["volume"] = props["volume"]

        if "surface_area" in props:
            results["surface_area"] = props["surface_area"]

        if "center_of_mass" in props:
            results["center_of_mass"] = props["center_of_mass"]

        # Compute derived properties if base data available
        bbox = results.get("bounding_box", {})
        if bbox:
            dx = bbox.get("dx", bbox.get("xmax", 0) - bbox.get("xmin", 0))
            dy = bbox.get("dy", bbox.get("ymax", 0) - bbox.get("ymin", 0))
            dz = bbox.get("dz", bbox.get("zmax", 0) - bbox.get("zmin", 0))

            if "bounding_box_volume" not in results:
                results["bounding_box_volume"] = dx * dy * dz

        if "volume" in results and "bounding_box_volume" in results:
            if results["bounding_box_volume"] > 0:
                results["compactness"] = results["volume"] / results["bounding_box_volume"]

        if "volume" in results and "surface_area" in results:
            if results["volume"] > 0:
                results["surface_to_volume_ratio"] = results["surface_area"] / results["volume"]

        return results

    def _get_shape_for_item(
        self, doc: CADlingDocument, item: CADItem
    ) -> Optional[any]:
        """Get shape object for analysis.

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
            # Load via pythonocc backend
            return self._load_occ_shape(doc)
        elif format_str == "stl" and self.has_trimesh:
            # Load via trimesh
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

    def _analyze_occ_shape(self, shape) -> dict:
        """Analyze pythonocc shape.

        Args:
            shape: OCC TopoDS_Shape to analyze

        Returns:
            Dictionary with analysis results
        """
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.GProp import GProp_GProps

        results = {}

        # Compute volume properties
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)

        results["volume"] = props.Mass()
        results["mass"] = props.Mass() * self.density

        # Center of mass
        cog = props.CentreOfMass()
        results["center_of_mass"] = {
            "x": cog.X(),
            "y": cog.Y(),
            "z": cog.Z(),
        }

        # Moment of inertia (matrix)
        inertia_matrix = props.MatrixOfInertia()
        results["inertia_tensor"] = {
            "Ixx": inertia_matrix.Value(1, 1),
            "Iyy": inertia_matrix.Value(2, 2),
            "Izz": inertia_matrix.Value(3, 3),
            "Ixy": inertia_matrix.Value(1, 2),
            "Ixz": inertia_matrix.Value(1, 3),
            "Iyz": inertia_matrix.Value(2, 3),
        }

        # Surface area
        surface_props = GProp_GProps()
        brepgprop.SurfaceProperties(shape, surface_props)
        results["surface_area"] = surface_props.Mass()

        # Bounding box
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        results["bounding_box"] = {
            "xmin": xmin,
            "ymin": ymin,
            "zmin": zmin,
            "xmax": xmax,
            "ymax": ymax,
            "zmax": zmax,
            "dx": xmax - xmin,
            "dy": ymax - ymin,
            "dz": zmax - zmin,
        }

        # Additional derived properties
        results["bounding_box_volume"] = (
            (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        )

        # Compactness (ratio of volume to bounding box volume)
        if results["bounding_box_volume"] > 0:
            results["compactness"] = results["volume"] / results["bounding_box_volume"]
        else:
            results["compactness"] = 0.0

        # Surface-to-volume ratio
        if results["volume"] > 0:
            results["surface_to_volume_ratio"] = (
                results["surface_area"] / results["volume"]
            )
        else:
            results["surface_to_volume_ratio"] = float("inf")

        _log.debug(
            f"OCC shape analysis complete: V={results['volume']:.2f}, "
            f"SA={results['surface_area']:.2f}"
        )

        return results

    def _analyze_trimesh(self, mesh) -> dict:
        """Analyze trimesh object.

        Args:
            mesh: Trimesh object to analyze

        Returns:
            Dictionary with analysis results
        """
        import trimesh

        results = {}

        # Volume
        results["volume"] = abs(mesh.volume)
        results["mass"] = abs(mesh.volume) * self.density

        # Center of mass
        if mesh.is_watertight:
            com = mesh.center_mass
        else:
            com = mesh.centroid

        results["center_of_mass"] = {
            "x": float(com[0]),
            "y": float(com[1]),
            "z": float(com[2]),
        }

        # Moment of inertia
        if mesh.is_watertight:
            inertia = mesh.moment_inertia
            results["inertia_tensor"] = {
                "Ixx": float(inertia[0, 0]),
                "Iyy": float(inertia[1, 1]),
                "Izz": float(inertia[2, 2]),
                "Ixy": float(inertia[0, 1]),
                "Ixz": float(inertia[0, 2]),
                "Iyz": float(inertia[1, 2]),
            }
        else:
            results["inertia_tensor"] = None
            _log.debug("Inertia tensor not computed: mesh not watertight")

        # Surface area
        results["surface_area"] = mesh.area

        # Bounding box
        bounds = mesh.bounds
        results["bounding_box"] = {
            "xmin": float(bounds[0, 0]),
            "ymin": float(bounds[0, 1]),
            "zmin": float(bounds[0, 2]),
            "xmax": float(bounds[1, 0]),
            "ymax": float(bounds[1, 1]),
            "zmax": float(bounds[1, 2]),
            "dx": float(bounds[1, 0] - bounds[0, 0]),
            "dy": float(bounds[1, 1] - bounds[0, 1]),
            "dz": float(bounds[1, 2] - bounds[0, 2]),
        }

        # Bounding box volume
        extents = mesh.extents
        results["bounding_box_volume"] = float(np.prod(extents))

        # Compactness
        if results["bounding_box_volume"] > 0:
            results["compactness"] = results["volume"] / results["bounding_box_volume"]
        else:
            results["compactness"] = 0.0

        # Surface-to-volume ratio
        if results["volume"] > 0:
            results["surface_to_volume_ratio"] = (
                results["surface_area"] / results["volume"]
            )
        else:
            results["surface_to_volume_ratio"] = float("inf")

        # Mesh-specific properties
        results["num_vertices"] = len(mesh.vertices)
        results["num_faces"] = len(mesh.faces)
        results["is_watertight"] = mesh.is_watertight
        results["is_winding_consistent"] = mesh.is_winding_consistent

        _log.debug(
            f"Trimesh analysis complete: V={results['volume']:.2f}, "
            f"SA={results['surface_area']:.2f}, watertight={mesh.is_watertight}"
        )

        return results

    def supports_batch_processing(self) -> bool:
        """Geometry analysis can process items independently."""
        return True

    def get_batch_size(self) -> int:
        """Process items one at a time (geometry analysis can be expensive)."""
        return 1

    def requires_gpu(self) -> bool:
        """Geometry analysis does not require GPU."""
        return False
