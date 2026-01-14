"""Data models for CAD documents.

This package provides data structures for representing CAD documents,
including base models, format-specific models, and configuration options.

Core Models:
    - CADlingDocument: Central data structure
    - CADItem: Base class for all CAD items
    - InputFormat: Enum of supported formats
    - ConversionResult: Conversion result wrapper

Configuration:
    - PipelineOptions: Pipeline configuration
    - BackendOptions: Backend configuration
"""

from cadling.datamodel.backend_options import (
    BackendOptions,
    BRepBackendOptions,
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
]
