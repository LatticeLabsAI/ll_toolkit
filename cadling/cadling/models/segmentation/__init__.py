"""CAD segmentation models.

This module provides ML models for semantic segmentation of CAD data:
- MeshSegmentationModel: Graph neural network for mesh face segmentation
- BRepSegmentationModel: Hybrid GAT+Transformer for B-Rep face segmentation
- ManufacturingFeatureRecognizer: Two-stage feature recognition with parameter extraction

All models inherit from EnrichmentModel and integrate with the cadling pipeline.
"""

from __future__ import annotations

from .mesh_segmentation import MeshSegmentationModel
from .brep_segmentation import BRepSegmentationModel
from .feature_recognition import ManufacturingFeatureRecognizer
from .vision_text_association import VisionTextAssociationModel

__all__ = [
    "MeshSegmentationModel",
    "BRepSegmentationModel",
    "ManufacturingFeatureRecognizer",
    "VisionTextAssociationModel",
]
