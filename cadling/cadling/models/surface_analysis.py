"""Surface analysis model for CAD parts.

This module provides surface type classification and curvature analysis for
B-Rep faces in CAD models.

Classes:
    SurfaceAnalysisModel: Main model for surface analysis
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Dict, Any
import math

import numpy as np

from cadling.models.base_model import EnrichmentModel
from cadling.lib.geometry.face_geometry import FaceGeometryExtractor

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument
    from cadling.datamodel.brep import BRepFaceItem

_log = logging.getLogger(__name__)


# Mapping from OCC GeomAbs_SurfaceType enum to surface type strings
# See OCC.Core.GeomAbs for enum values
GEOMABS_SURFACE_TYPE_MAP = {
    0: "PLANE",                           # GeomAbs_Plane
    1: "CYLINDRICAL_SURFACE",            # GeomAbs_Cylinder
    2: "CONICAL_SURFACE",                # GeomAbs_Cone
    3: "SPHERICAL_SURFACE",              # GeomAbs_Sphere
    4: "TOROIDAL_SURFACE",               # GeomAbs_Torus
    5: "SURFACE_OF_REVOLUTION",          # GeomAbs_Revolution
    6: "SURFACE_OF_LINEAR_EXTRUSION",    # GeomAbs_Extrusion
    7: "B_SPLINE_SURFACE",               # GeomAbs_BSplineSurface
    8: "B_SPLINE_SURFACE",               # GeomAbs_BezierSurface (map to B_SPLINE)
    9: "UNKNOWN",                         # GeomAbs_OtherSurface
}


class SurfaceAnalysisModel(EnrichmentModel):
    """Surface analysis enrichment model.

    Analyzes surface types and curvature properties for B-Rep faces, including:
    - Surface type classification (plane, cylinder, sphere, etc.)
    - Gaussian curvature (K = k1 * k2)
    - Mean curvature (H = (k1 + k2) / 2)
    - Principal curvatures (k1, k2)
    - Surface smoothness and planarity flags

    This model works with pythonocc-core shapes for STEP/IGES/BRep formats.

    Attributes:
        has_pythonocc: Whether pythonocc-core is available
        face_extractor: FaceGeometryExtractor instance for curvature computation

    Example:
        model = SurfaceAnalysisModel()
        result = converter.convert(
            "part.step",
            pipeline_options=PipelineOptions(
                enrichment_models=[model]
            )
        )
        for item in result.document.items:
            if "surface_analysis" in item.properties:
                analysis = item.properties["surface_analysis"]
                print(f"Surface type: {analysis['surface_type']}")
                print(f"Gaussian curvature: {analysis['gaussian_curvature']}")
    """

    def __init__(self):
        """Initialize surface analysis model."""
        super().__init__()

        # Check for pythonocc-core availability
        self.has_pythonocc = False
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_SurfaceType

            self.has_pythonocc = True
            _log.debug("pythonocc-core available for surface analysis")
        except ImportError:
            _log.warning("pythonocc-core not available. Surface analysis disabled.")

        # Initialize face geometry extractor
        self.face_extractor = FaceGeometryExtractor()

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Analyze surface types and curvature for CAD items.

        Args:
            doc: CADlingDocument being enriched
            item_batch: List of CADItem objects to analyze
        """
        if not self.has_pythonocc:
            _log.debug("Surface analysis skipped: pythonocc not available")
            return

        for item in item_batch:
            try:
                # Only analyze BRep face items
                if not self._is_brep_face(item):
                    continue

                # Analyze face
                analysis_result = self._analyze_face(item)

                if analysis_result:
                    item.properties["surface_analysis"] = analysis_result

                    # Add provenance
                    item.add_provenance(
                        component_type="enrichment_model",
                        component_name="SurfaceAnalysisModel",
                    )

                    _log.debug(
                        f"Analyzed surface for face '{item.label.text}': "
                        f"type={analysis_result.get('surface_type', 'UNKNOWN')}, "
                        f"K={analysis_result.get('gaussian_curvature', 0.0):.6f}"
                    )

            except Exception as e:
                _log.error(f"Surface analysis failed for item {item.label.text}: {e}")

    def _is_brep_face(self, item: CADItem) -> bool:
        """Check if item is a BRep face.

        Args:
            item: CADItem to check

        Returns:
            True if item is a BRep face item
        """
        return item.item_type == "brep_face"

    def _get_occ_face(self, item: CADItem) -> Optional[any]:
        """Get OCC face from item.

        Args:
            item: BRep face item

        Returns:
            TopoDS_Face object or None
        """
        # Check for _shape attribute (should be set by backend)
        if hasattr(item, "_shape") and item._shape is not None:
            return item._shape

        # Check for _occ_shape attribute (alternative name)
        if hasattr(item, "_occ_shape") and item._occ_shape is not None:
            return item._occ_shape

        _log.debug(f"Could not get OCC face for item {item.label.text}")
        return None

    def _analyze_face(self, face_item: CADItem) -> Optional[Dict[str, Any]]:
        """Analyze a single BRep face using OCC.

        Args:
            face_item: BRepFaceItem to analyze

        Returns:
            Dictionary with analysis results:
            - surface_type: str (PLANE, CYLINDRICAL_SURFACE, etc.)
            - surface_type_confidence: float [0, 1]
            - gaussian_curvature: float (K = k1 * k2)
            - mean_curvature: float (H = (k1 + k2) / 2)
            - principal_curvatures: [k1, k2]
            - is_planar: bool
            - is_smooth: bool
        """
        if not self.has_pythonocc:
            _log.warning(
                "Surface analysis requires pythonocc-core. "
                "Install with: conda install pythonocc-core -c conda-forge"
            )
            return {
                "status": "unavailable",
                "reason": "pythonocc-core not installed",
                "surface_type": "UNKNOWN",
                "surface_type_confidence": 0.0,
                "gaussian_curvature": 0.0,
                "mean_curvature": 0.0,
                "principal_curvatures": [0.0, 0.0],
                "is_planar": False,
                "is_smooth": False,
            }

        # Get OCC face
        occ_face = self._get_occ_face(face_item)
        if occ_face is None:
            return {
                "status": "error",
                "reason": "Could not retrieve OCC face from item",
                "surface_type": "UNKNOWN",
                "surface_type_confidence": 0.0,
                "gaussian_curvature": 0.0,
                "mean_curvature": 0.0,
                "principal_curvatures": [0.0, 0.0],
                "is_planar": False,
                "is_smooth": False,
            }

        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import (
                GeomAbs_Plane,
                GeomAbs_Cylinder,
                GeomAbs_Cone,
                GeomAbs_Sphere,
                GeomAbs_Torus,
            )

            results = {
                "status": "success"
            }

            # 1. Classify surface type using BRepAdaptor
            surface_adaptor = BRepAdaptor_Surface(occ_face)
            geom_type = surface_adaptor.GetType()

            # Map GeomAbs_SurfaceType to surface type string
            surface_type = GEOMABS_SURFACE_TYPE_MAP.get(int(geom_type), "UNKNOWN")
            results["surface_type"] = surface_type

            # Surface type confidence (1.0 for classified types, 0.5 for UNKNOWN)
            results["surface_type_confidence"] = 1.0 if surface_type != "UNKNOWN" else 0.5

            # 2. Compute curvature using FaceGeometryExtractor
            curvature = self.face_extractor.compute_curvature_at_center(occ_face)

            if curvature:
                gaussian_curvature, mean_curvature = curvature
                results["gaussian_curvature"] = float(gaussian_curvature)
                results["mean_curvature"] = float(mean_curvature)

                # 3. Compute principal curvatures from Gaussian and Mean
                # Principal curvatures: k1, k2 where:
                #   K = k1 * k2 (Gaussian)
                #   H = (k1 + k2) / 2 (Mean)
                # Solving: k1, k2 = H ± sqrt(H^2 - K)
                try:
                    discriminant = mean_curvature**2 - gaussian_curvature
                    if discriminant >= 0:
                        sqrt_disc = math.sqrt(discriminant)
                        k1 = mean_curvature + sqrt_disc
                        k2 = mean_curvature - sqrt_disc
                        results["principal_curvatures"] = [float(k1), float(k2)]
                    else:
                        # Complex principal curvatures (shouldn't happen for real surfaces)
                        _log.debug(
                            f"Complex principal curvatures: H={mean_curvature}, K={gaussian_curvature}"
                        )
                        results["principal_curvatures"] = [float(mean_curvature), float(mean_curvature)]
                except Exception as e:
                    _log.debug(f"Failed to compute principal curvatures: {e}")
                    results["principal_curvatures"] = [0.0, 0.0]

            else:
                # No curvature computed
                results["gaussian_curvature"] = 0.0
                results["mean_curvature"] = 0.0
                results["principal_curvatures"] = [0.0, 0.0]

            # 4. Planarity check
            # A surface is planar if it's classified as PLANE by OCC
            # Note: We prioritize OCC classification over curvature because curvature
            # computation can fail on some parametric surfaces
            is_planar = (geom_type == GeomAbs_Plane)
            results["is_planar"] = is_planar

            # 5. Smoothness check
            # A surface is smooth if curvatures are not too extreme
            # Consider smooth if |K| < 1.0 and |H| < 1.0 (adjustable thresholds)
            is_smooth = (
                abs(results["gaussian_curvature"]) < 1.0
                and abs(results["mean_curvature"]) < 1.0
            )
            results["is_smooth"] = is_smooth

            return results

        except Exception as e:
            _log.warning(f"Failed to analyze face: {e}")
            return {
                "status": "error",
                "reason": str(e),
                "surface_type": "UNKNOWN",
                "surface_type_confidence": 0.0,
                "gaussian_curvature": 0.0,
                "mean_curvature": 0.0,
                "principal_curvatures": [0.0, 0.0],
                "is_planar": False,
                "is_smooth": False,
            }

    def supports_batch_processing(self) -> bool:
        """Surface analysis can process items independently."""
        return True

    def get_batch_size(self) -> int:
        """Process items one at a time (geometry analysis can be expensive)."""
        return 1

    def requires_gpu(self) -> bool:
        """Surface analysis does not require GPU."""
        return False
