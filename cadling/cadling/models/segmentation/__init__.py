"""CAD segmentation models.

This module provides ML models for semantic segmentation of CAD data:
- MeshSegmentationModel: Graph neural network for mesh face segmentation
- BRepSegmentationModel: Hybrid GAT+Transformer for B-Rep face segmentation
- ManufacturingFeatureRecognizer: Two-stage feature recognition with parameter extraction
- SketchGeometryExtractor: Converts 2D primitives to tokenizer-ready command sequences

All models inherit from EnrichmentModel and integrate with the cadling pipeline.
"""

from __future__ import annotations

import logging as _log

# SketchGeometryExtractor has no heavy dependencies (no torch) — always import
from .sketch_geometry_extractor import SketchGeometryExtractor

__all__ = [
    "SketchGeometryExtractor",
]

# The remaining models require torch and related ML libraries.
# Import them lazily so that the package is usable without torch installed.
try:
    from .brep_segmentation import BRepSegmentationModel
    from .feature_recognition import ManufacturingFeatureRecognizer
    from .mesh_segmentation import MeshSegmentationModel
    from .vision_text_association import VisionTextAssociationModel

    __all__ += [
        "MeshSegmentationModel",
        "BRepSegmentationModel",
        "ManufacturingFeatureRecognizer",
        "VisionTextAssociationModel",
    ]
except ImportError:
    _log.getLogger(__name__).debug(
        "torch not available — ML segmentation models will not be importable. "
        "SketchGeometryExtractor is still available."
    )
