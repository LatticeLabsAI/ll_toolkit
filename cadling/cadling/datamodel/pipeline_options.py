"""Pipeline configuration options.

This module provides configuration options for CAD conversion pipelines,
adapted from docling's pipeline options but extended for CAD-specific features.

Classes:
    PipelineOptions: Base pipeline options.
    CADVlmPipelineOptions: Options for vision-language model pipeline.
    HybridPipelineOptions: Options for hybrid text+vision pipeline.

Example:
    options = PipelineOptions(
        enrichment_models=[
            CADPartClassifier(artifacts_path),
            CADSimilarityEmbedder(artifacts_path)
        ]
    )
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from cadling.models.base_model import EnrichmentModel


class PipelineOptions(BaseModel):
    """Base configuration options for CAD pipelines.

    Attributes:
        enrichment_models: List of enrichment models to apply.
        enrichment_batch_size: Batch size for enrichment processing.
        device: Device for model inference ("cpu", "cuda", "mps").
        do_topology_analysis: Whether to build topology graphs.
        max_items: Maximum number of items to process (None for unlimited).
    """

    enrichment_models: Optional[List[EnrichmentModel]] = Field(
        default_factory=list
    )
    enrichment_batch_size: int = 32
    device: str = "cpu"
    do_topology_analysis: bool = True
    max_items: Optional[int] = None

    model_config = {"arbitrary_types_allowed": True}


class VlmOptions(BaseModel):
    """Vision-language model configuration options.

    Attributes:
        model_name: Name of VLM model (e.g., "gpt-4-vision", "claude-3-opus").
        api_key: API key for commercial models.
        api_base_url: Base URL for API (optional).
        max_tokens: Maximum tokens for response.
        temperature: Sampling temperature.
    """

    model_name: str
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.0


class CADVlmPipelineOptions(PipelineOptions):
    """Options for vision-language model pipeline.

    This pipeline renders CAD views and applies VLM for optical recognition
    of dimensions, tolerances, and annotations.

    Attributes:
        vlm_options: VLM configuration.
        views_to_render: List of view names to render.
        render_resolution: Resolution for rendered images.
        do_ocr: Whether to apply OCR for text extraction.
    """

    vlm_options: Optional[VlmOptions] = None
    views_to_render: List[str] = Field(
        default_factory=lambda: ["front", "top", "isometric"]
    )
    render_resolution: int = 1024
    do_ocr: bool = True


class HybridPipelineOptions(PipelineOptions):
    """Options for hybrid text+vision pipeline.

    This pipeline combines text parsing with vision analysis for comprehensive
    CAD understanding.

    Attributes:
        enable_text_parsing: Whether to parse text representation.
        enable_vision: Whether to do vision analysis.
        vlm_options: VLM configuration.
        views_to_render: List of view names to render.
        render_resolution: Resolution for rendered images.
        fusion_strategy: How to fuse text and vision ("merge", "prioritize_text").
    """

    enable_text_parsing: bool = True
    enable_vision: bool = True
    vlm_options: Optional[VlmOptions] = None
    views_to_render: List[str] = Field(
        default_factory=lambda: ["front", "top", "isometric"]
    )
    render_resolution: int = 1024
    fusion_strategy: str = "merge"  # "merge" or "prioritize_text"
