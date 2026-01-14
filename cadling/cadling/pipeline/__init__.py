"""CAD conversion pipelines.

This package provides pipelines for orchestrating CAD file conversion.
Pipelines follow the Build → Assemble → Enrich pattern.

Available Pipelines:
    - BaseCADPipeline: Abstract base class
    - SimpleCADPipeline: For text-based parsing
    - STEPPipeline: For STEP file conversion
    - STLPipeline: For STL file conversion
    - VisionPipeline: For optical CAD recognition with VLMs
    - HybridPipeline: Combines text parsing + vision analysis
    - CADVlmPipeline: For vision-based annotation extraction
    - EnrichmentModel: Base class for enrichment models
"""

from cadling.pipeline.base_pipeline import BaseCADPipeline, EnrichmentModel
from cadling.pipeline.simple_pipeline import SimpleCADPipeline
from cadling.pipeline.step_pipeline import STEPPipeline
from cadling.pipeline.stl_pipeline import STLPipeline
from cadling.pipeline.vision_pipeline import VisionPipeline
from cadling.pipeline.hybrid_pipeline import HybridPipeline
from cadling.pipeline.vlm_pipeline import CADVlmPipeline, CADVlmPipelineOptions

__all__ = [
    "BaseCADPipeline",
    "SimpleCADPipeline",
    "STEPPipeline",
    "STLPipeline",
    "VisionPipeline",
    "HybridPipeline",
    "CADVlmPipeline",
    "CADVlmPipelineOptions",
    "EnrichmentModel",
]
