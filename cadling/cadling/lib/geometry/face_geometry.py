"""Face geometry extraction utilities for B-Rep models.

This module provides tools to extract geometric properties from OpenCASCADE faces,
including curvature, normals, centroids, and bounding boxes.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)

try:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepTools import breptools
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.GeomLProp import GeomLProp_SLProps
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.gp import gp_Dir

    HAS_OCC = True
except ImportError:
    HAS_OCC = False


class FaceGeometryExtractor:
    """Extract geometric features from pythonocc TopoDS_Face objects.

    This class uses OpenCASCADE's geometric property classes to compute
    real geometric features instead of placeholder values.

    Features extracted:
    - Gaussian curvature (principal curvatures product)
    - Mean curvature (average of principal curvatures)
    - Normal vector at face center
    - Centroid (center of mass)
    - Bounding box dimensions

    Example:
        extractor = FaceGeometryExtractor()
        features = extractor.extract_features(topods_face)
        gaussian_curv = features["gaussian_curvature"]
        mean_curv = features["mean_curvature"]
        normal = features["normal"]
    """

    def __init__(self):
        """Initialize face geometry extractor."""
        self.has_pythonocc = HAS_OCC
        if HAS_OCC:
            _log.debug("FaceGeometryExtractor initialized with pythonocc-core")
        else:
            _log.warning(
                "pythonocc-core not available. "
                "FaceGeometryExtractor will return None for all features."
            )

    def extract_features(self, face) -> Optional[dict]:
        """Extract all geometric features from a TopoDS_Face.

        Args:
            face: OCC.Core.TopoDS.TopoDS_Face object

        Returns:
            Dictionary with geometric features, or None if extraction fails

        Features:
            - gaussian_curvature: float (K = k1 * k2)
            - mean_curvature: float (H = (k1 + k2) / 2)
            - normal: [x, y, z] unit normal vector
            - centroid: [x, y, z] center of mass
            - bbox_dimensions: [dx, dy, dz] bounding box size
            - surface_area: float
        """
        if not self.has_pythonocc:
            _log.debug("pythonocc not available, cannot extract features")
            return None

        try:
            # Compute curvature and normal at center
            curvature = self.compute_curvature_at_center(face)
            normal = self.compute_normal_at_center(face)
            centroid = self.compute_centroid(face)
            bbox_dims = self.compute_bbox_dimensions(face)
            area = self.compute_surface_area(face)

            return {
                "gaussian_curvature": curvature[0] if curvature else 0.0,
                "mean_curvature": curvature[1] if curvature else 0.0,
                "normal": normal if normal else [0.0, 0.0, 1.0],
                "centroid": centroid if centroid else [0.0, 0.0, 0.0],
                "bbox_dimensions": bbox_dims if bbox_dims else [1.0, 1.0, 1.0],
                "surface_area": area if area else 0.0,
            }

        except Exception as e:
            _log.warning(f"Failed to extract face features: {e}")
            return None

    def compute_curvature_at_center(self, face) -> Optional[Tuple[float, float]]:
        """Compute Gaussian and mean curvature at face center.

        Args:
            face: TopoDS_Face

        Returns:
            (gaussian_curvature, mean_curvature) or None if computation fails
        """
        if not self.has_pythonocc:
            return None

        try:
            # Get UV bounds
            u_min, u_max, v_min, v_max = breptools.UVBounds(face)

            # Compute center UV
            u_center = (u_min + u_max) / 2.0
            v_center = (v_min + v_max) / 2.0

            # Get surface
            surface = BRep_Tool.Surface(face)

            # Create surface property analyzer at center point
            # Order 2 is required for curvature computation
            props = GeomLProp_SLProps(surface, u_center, v_center, 2, 1e-9)

            # Check if curvature is defined
            if not props.IsCurvatureDefined():
                _log.debug("Curvature not defined at face center")
                return (0.0, 0.0)

            # Get Gaussian and mean curvature
            gaussian_curvature = props.GaussianCurvature()
            mean_curvature = props.MeanCurvature()

            return (gaussian_curvature, mean_curvature)

        except Exception as e:
            _log.debug(f"Failed to compute curvature: {e}")
            return None

    def compute_normal_at_center(self, face) -> Optional[list[float]]:
        """Compute unit normal vector at face center.

        Args:
            face: TopoDS_Face

        Returns:
            [x, y, z] unit normal vector, or None if computation fails
        """
        if not self.has_pythonocc:
            return None

        try:
            # Get UV bounds
            u_min, u_max, v_min, v_max = breptools.UVBounds(face)

            # Compute center UV
            u_center = (u_min + u_max) / 2.0
            v_center = (v_min + v_max) / 2.0

            # Get surface
            surface = BRep_Tool.Surface(face)

            # Create surface property analyzer at center point
            # Order 1 is sufficient for normal computation
            props = GeomLProp_SLProps(surface, u_center, v_center, 1, 1e-9)

            # Get normal direction
            normal_dir: gp_Dir = props.Normal()

            # Convert to list
            return [normal_dir.X(), normal_dir.Y(), normal_dir.Z()]

        except Exception as e:
            _log.debug(f"Failed to compute normal: {e}")
            return None

    def compute_centroid(self, face) -> Optional[list[float]]:
        """Compute centroid (center of mass) of face.

        Args:
            face: TopoDS_Face

        Returns:
            [x, y, z] centroid coordinates, or None if computation fails
        """
        if not self.has_pythonocc:
            return None

        try:
            # Compute surface properties
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)

            # Get center of mass
            center = props.CentreOfMass()

            return [center.X(), center.Y(), center.Z()]

        except Exception as e:
            _log.debug(f"Failed to compute centroid: {e}")
            return None

    def compute_bbox_dimensions(self, face) -> Optional[list[float]]:
        """Compute bounding box dimensions of face.

        Args:
            face: TopoDS_Face

        Returns:
            [dx, dy, dz] bounding box dimensions, or None if computation fails
        """
        if not self.has_pythonocc:
            return None

        try:
            # Create bounding box
            bbox = Bnd_Box()
            brepbndlib.Add(face, bbox)

            # Get bounds
            x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()

            # Compute dimensions
            dx = x_max - x_min
            dy = y_max - y_min
            dz = z_max - z_min

            return [dx, dy, dz]

        except Exception as e:
            _log.debug(f"Failed to compute bbox dimensions: {e}")
            return None

    def compute_surface_area(self, face) -> Optional[float]:
        """Compute surface area of face.

        Args:
            face: TopoDS_Face

        Returns:
            Surface area, or None if computation fails
        """
        if not self.has_pythonocc:
            return None

        try:
            # Compute surface properties
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)

            # Get mass (which is area for surfaces)
            area = props.Mass()

            return area

        except Exception as e:
            _log.debug(f"Failed to compute surface area: {e}")
            return None
