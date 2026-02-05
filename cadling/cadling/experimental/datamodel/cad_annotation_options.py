"""CAD annotation extraction options.

This module provides configuration options for extracting Product Manufacturing
Information (PMI) and annotations from CAD files using vision-language models.

Classes:
    CADAnnotationOptions: Options for PMI and annotation extraction.

Example:
    options = CADAnnotationOptions(
        annotation_types=["dimension", "tolerance", "gdt"],
        vlm_model="gpt-4-vision",
        min_confidence=0.8
    )
"""

from __future__ import annotations

from typing import ClassVar, List

from pydantic import Field

from cadling.datamodel.pipeline_options import PipelineOptions


class CADAnnotationOptions(PipelineOptions):
    """Configuration options for CAD annotation extraction.

    This options class configures the extraction of Product Manufacturing
    Information (PMI) including dimensions, tolerances, GD&T symbols,
    surface finish, and material callouts from CAD files using VLM analysis.

    Attributes:
        kind: Discriminator for option type.
        annotation_types: Types of annotations to extract.
        min_confidence: Minimum confidence threshold for extracted annotations.
        vlm_model: Vision-language model to use for extraction.
        views_to_process: List of view names to process for annotations.
        enable_cross_view_validation: Whether to validate annotations across views.
        extraction_resolution: Resolution for rendering views (higher = better accuracy).
        include_geometric_context: Whether to inject geometric features into prompts.
    """

    kind: ClassVar[str] = "cadling_experimental_annotation"

    annotation_types: List[str] = Field(
        default_factory=lambda: ["dimension", "tolerance", "gdt", "surface_finish"],
        description="Types of annotations to extract (dimension, tolerance, gdt, surface_finish, material, welding)"
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for extracted annotations"
    )
    vlm_model: str = Field(
        default="gpt-4-vision",
        description="VLM model name (gpt-4-vision, claude-3-opus, etc.)"
    )
    views_to_process: List[str] = Field(
        default_factory=lambda: ["front", "top", "right"],
        description="List of view names to process for annotation extraction"
    )
    enable_cross_view_validation: bool = Field(
        default=True,
        description="Whether to validate annotations across multiple views"
    )
    extraction_resolution: int = Field(
        default=2048,
        ge=512,
        le=4096,
        description="Resolution for rendering views (higher = better accuracy but slower)"
    )
    include_geometric_context: bool = Field(
        default=True,
        description="Whether to inject geometric features into VLM prompts"
    )
