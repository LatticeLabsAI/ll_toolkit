"""Data models for CAD documents.

This package provides data structures for representing CAD documents,
including base models, format-specific models, and configuration options.

Core Models:
    - CADlingDocument: Central data structure
    - CADItem: Base class for all CAD items
    - InputFormat: Enum of supported formats
    - ConversionResult: Conversion result wrapper

2D Geometry Models:
    - Sketch2DItem: CADItem for 2D sketch/drawing content
    - SketchProfile: Collection of 2D primitives
    - Primitive2D and subclasses: Line2D, Arc2D, Circle2D, etc.

Configuration:
    - PipelineOptions: Pipeline configuration
    - BackendOptions: Backend configuration
"""

from cadling.datamodel.backend_options import (
    BackendOptions,
    BRepBackendOptions,
    DXFBackendOptions,
    PDFBackendOptions,
    STEPBackendOptions,
    STLBackendOptions,
)
from cadling.datamodel.base_models import (
    BoundingBox3D,
    CADDocumentOrigin,
    CADInputDocument,
    CADItem,
    CADItemLabel,
    CADlingDocument,
    ConversionResult,
    ConversionStatus,
    ErrorItem,
    InputFormat,
    ProcessingStep,
    ProvenanceItem,
    TopologyGraph,
)
from cadling.datamodel.geometry_2d import (
    Arc2D,
    BoundingBox2D,
    Circle2D,
    DimensionAnnotation,
    DimensionType,
    Ellipse2D,
    Line2D,
    Polyline2D,
    Primitive2D,
    PrimitiveType,
    Sketch2DItem,
    SketchProfile,
    Spline2D,
)
from cadling.datamodel.pipeline_options import (
    CADVlmPipelineOptions,
    HybridPipelineOptions,
    PipelineOptions,
    VlmOptions,
)

__all__ = [
    # Base models
    "CADlingDocument",
    "CADItem",
    "CADItemLabel",
    "CADDocumentOrigin",
    "CADInputDocument",
    "ConversionResult",
    "ConversionStatus",
    "InputFormat",
    "BoundingBox3D",
    "TopologyGraph",
    "ProvenanceItem",
    "ProcessingStep",
    "ErrorItem",
    # 2D geometry models
    "PrimitiveType",
    "DimensionType",
    "Primitive2D",
    "Line2D",
    "Arc2D",
    "Circle2D",
    "Polyline2D",
    "Ellipse2D",
    "Spline2D",
    "DimensionAnnotation",
    "BoundingBox2D",
    "SketchProfile",
    "Sketch2DItem",
    # Pipeline options
    "PipelineOptions",
    "CADVlmPipelineOptions",
    "HybridPipelineOptions",
    "VlmOptions",
    # Backend options
    "BackendOptions",
    "STEPBackendOptions",
    "STLBackendOptions",
    "BRepBackendOptions",
    "DXFBackendOptions",
    "PDFBackendOptions",
]
