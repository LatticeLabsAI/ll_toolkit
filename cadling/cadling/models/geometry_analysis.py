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
    ) -> Optional[dict]:
        """Analyze a single CAD item.

        Args:
            doc: Document containing the item
            item: Item to analyze

        Returns:
            Dictionary with analysis results, or None if analysis failed
        """
        # Try to get shape from backend
        shape = self._get_shape_for_item(doc, item)

        if shape is None:
            _log.debug(f"Could not get shape for item {item.label.text}")
            return None

        # Analyze based on shape type
        if self.has_pythonocc and self._is_occ_shape(shape):
            return self._analyze_occ_shape(shape)
        elif self.has_trimesh and self._is_trimesh(shape):
            return self._analyze_trimesh(shape)
        else:
            _log.debug(f"Unsupported shape type for item {item.label.text}")
            return None

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
