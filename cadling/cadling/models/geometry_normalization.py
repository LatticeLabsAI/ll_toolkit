"""Geometry normalization model for CAD parts.

This module provides geometry normalization capabilities including centering,
scaling, and principal axis alignment for consistent coordinate systems.

Classes:
    GeometryNormalizationModel: Main model for geometry normalization
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Dict, Any, List
import numpy as np

from cadling.models.base_model import EnrichmentModel
from cadling.core import compute_centroid, compute_bounding_box

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument

_log = logging.getLogger(__name__)


class GeometryNormalizationModel(EnrichmentModel):
    """Geometry normalization enrichment model.

    Normalizes 3D geometry for consistent coordinate systems and scales by:
    - Centering: Translating centroid to origin
    - Scaling: Scaling max bounding box dimension to 1.0
    - PCA Alignment: Aligning principal axes with coordinate axes (optional)

    This model works with both pythonocc-core shapes and trimesh objects.

    Attributes:
        center: Whether to center geometry (translate centroid to origin)
        scale_to_unit: Whether to scale to unit bounding box
        align_principal_axes: Whether to align principal axes (PCA)

    Example:
        model = GeometryNormalizationModel(
            center=True,
            scale_to_unit=True,
            align_principal_axes=False
        )
        result = converter.convert(
            "part.step",
            pipeline_options=PipelineOptions(
                enrichment_models=[model]
            )
        )
        for item in result.document.items:
            if "geometry_normalization" in item.properties:
                norm = item.properties["geometry_normalization"]
                print(f"Scale factor: {norm['scaling']['scale_factor']}")
    """

    def __init__(
        self,
        center: bool = True,
        scale_to_unit: bool = True,
        align_principal_axes: bool = False,
    ):
        """Initialize geometry normalization model.

        Args:
            center: Whether to center geometry (default: True)
            scale_to_unit: Whether to scale to unit box (default: True)
            align_principal_axes: Whether to align principal axes (default: False)
        """
        super().__init__()

        self.center = center
        self.scale_to_unit = scale_to_unit
        self.align_principal_axes = align_principal_axes

        # Check for pythonocc-core availability
        self.has_pythonocc = False
        try:
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.GProp import GProp_GProps

            self.has_pythonocc = True
            _log.debug("pythonocc-core available for geometry normalization")
        except ImportError:
            _log.warning("pythonocc-core not available. OCC normalization disabled.")

        # Check for trimesh availability
        self.has_trimesh = False
        try:
            import trimesh

            self.has_trimesh = True
            _log.debug("trimesh available for mesh normalization")
        except ImportError:
            _log.warning("trimesh not available. Mesh normalization disabled.")

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Compute normalization transformations for CAD items.

        Note: This model computes normalization parameters but does not
        apply them to the geometry. The parameters are stored in item
        properties for later use.

        Args:
            doc: CADlingDocument being enriched
            item_batch: List of CADItem objects to analyze
        """
        if not self.has_pythonocc and not self.has_trimesh:
            _log.debug("Geometry normalization skipped: no backend available")
            return

        for item in item_batch:
            try:
                # Compute normalization parameters
                norm_params = self._compute_normalization_for_item(doc, item)

                if norm_params:
                    item.properties["geometry_normalization"] = norm_params

                    # Add provenance
                    item.add_provenance(
                        component_type="enrichment_model",
                        component_name="GeometryNormalizationModel",
                    )

                    _log.debug(
                        f"Computed normalization for item '{item.label.text}': "
                        f"scale={norm_params.get('scaling', {}).get('scale_factor', 1.0):.4f}"
                    )

            except Exception as e:
                _log.error(f"Geometry normalization failed for item {item.label.text}: {e}")

    def _compute_normalization_for_item(
        self, doc: CADlingDocument, item: CADItem
    ) -> Optional[Dict[str, Any]]:
        """Compute normalization parameters for a single item.

        Args:
            doc: Document containing the item
            item: Item to analyze

        Returns:
            Dictionary with normalization parameters, or None if computation failed
        """
        # Extract points from geometry
        points = self._extract_points_from_item(doc, item)

        if points is None or len(points) == 0:
            _log.debug(f"Could not extract points for item {item.label.text}")
            return None

        # Compute normalization transformations
        return self._compute_normalization(points)

    def _extract_points_from_item(
        self, doc: CADlingDocument, item: CADItem
    ) -> Optional[np.ndarray]:
        """Extract point cloud from item geometry.

        Args:
            doc: Document containing the item
            item: Item to extract points from

        Returns:
            Numpy array of shape [N, 3] with points, or None
        """
        # Try to get geometry from item
        shape = self._get_shape_for_item(doc, item)

        if shape is None:
            return None

        # Extract points based on shape type
        if self.has_pythonocc and self._is_occ_shape(shape):
            return self._extract_points_from_occ_shape(shape)
        elif self.has_trimesh and self._is_trimesh(shape):
            return self._extract_points_from_trimesh(shape)

        return None

    def _get_shape_for_item(
        self, doc: CADlingDocument, item: CADItem
    ) -> Optional[any]:
        """Get shape object from item.

        Args:
            doc: Document containing the item
            item: Item to get shape for

        Returns:
            Shape object (OCC shape or trimesh), or None
        """
        # Check for _shape attribute
        if hasattr(item, "_shape") and item._shape is not None:
            return item._shape

        # Check for _occ_shape attribute
        if hasattr(item, "_occ_shape") and item._occ_shape is not None:
            return item._occ_shape

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

    def _extract_points_from_occ_shape(self, shape) -> Optional[np.ndarray]:
        """Extract vertex points from OCC shape.

        Args:
            shape: OCC TopoDS_Shape

        Returns:
            Numpy array of shape [N, 3] with vertex coordinates
        """
        try:
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_VERTEX
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopoDS import topods

            points = []

            # Iterate through all vertices
            explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
            while explorer.More():
                vertex = topods.Vertex(explorer.Current())
                pnt = BRep_Tool.Pnt(vertex)
                points.append([pnt.X(), pnt.Y(), pnt.Z()])
                explorer.Next()

            if len(points) == 0:
                return None

            return np.array(points, dtype=np.float64)

        except Exception as e:
            _log.debug(f"Failed to extract points from OCC shape: {e}")
            return None

    def _extract_points_from_trimesh(self, mesh) -> Optional[np.ndarray]:
        """Extract vertex points from trimesh.

        Args:
            mesh: Trimesh object

        Returns:
            Numpy array of shape [N, 3] with vertex coordinates
        """
        try:
            # Trimesh stores vertices directly
            return np.array(mesh.vertices, dtype=np.float64)
        except Exception as e:
            _log.debug(f"Failed to extract points from trimesh: {e}")
            return None

    def _compute_normalization(self, points: np.ndarray) -> Dict[str, Any]:
        """Compute normalization transformations.

        Operations:
        1. Centering: Translate centroid to origin
        2. Scaling: Scale max bounding box dimension to 1.0
        3. PCA Alignment: Align principal axes with coordinate axes

        Args:
            points: Numpy array of shape [N, 3]

        Returns:
            Dictionary with normalization parameters:
            - translation: {'centroid': [x,y,z], 'offset': [-x,-y,-z]}
            - scaling: {'scale_factor': float, 'original_max_dimension': float}
            - alignment: {'rotation_matrix': 3x3, 'eigenvalues': [e1,e2,e3]}
              (only if align_principal_axes=True)
        """
        results = {}

        # 1. Centering
        if self.center:
            centroid = np.mean(points, axis=0)
            results["translation"] = {
                "centroid": centroid.tolist(),
                "offset": (-centroid).tolist(),
            }

        # 2. Scaling
        if self.scale_to_unit:
            # Compute bounding box
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            dimensions = max_coords - min_coords

            # Max dimension
            max_dimension = np.max(dimensions)

            if max_dimension > 1e-10:
                scale_factor = 1.0 / max_dimension
            else:
                scale_factor = 1.0

            results["scaling"] = {
                "scale_factor": float(scale_factor),
                "original_max_dimension": float(max_dimension),
                "dimensions": dimensions.tolist(),
            }

        # 3. PCA Alignment (optional)
        if self.align_principal_axes:
            try:
                # Center points for PCA
                centered_points = points - np.mean(points, axis=0)

                # Compute covariance matrix
                covariance = np.cov(centered_points.T)

                # Eigenvalue decomposition
                eigenvalues, eigenvectors = np.linalg.eig(covariance)

                # Sort by eigenvalue (descending)
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

                # Ensure right-handed coordinate system
                if np.linalg.det(eigenvectors) < 0:
                    eigenvectors[:, 2] *= -1

                results["alignment"] = {
                    "rotation_matrix": eigenvectors.tolist(),
                    "eigenvalues": eigenvalues.tolist(),
                }

            except Exception as e:
                _log.warning(f"Failed to compute PCA alignment: {e}")
                results["alignment"] = None

        return results

    def supports_batch_processing(self) -> bool:
        """Geometry normalization can process items independently."""
        return True

    def get_batch_size(self) -> int:
        """Process items one at a time."""
        return 1

    def requires_gpu(self) -> bool:
        """Geometry normalization does not require GPU."""
        return False
