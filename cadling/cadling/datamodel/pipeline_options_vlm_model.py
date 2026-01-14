"""Vision-Language Model pipeline options.

This module provides configuration options specific to VLM-based pipelines
for optical CAD recognition and annotation extraction.

Classes:
    VLMModelOptions: Configuration for VLM model usage
    VLMPromptTemplate: Templates for VLM prompts
    VLMResponseFormat: Expected response format
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from cadling.datamodel.vlm_model_specs import VLMProvider


class VLMResponseFormat(str, Enum):
    """Expected response format from VLM."""

    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"


class VLMPromptTemplate(BaseModel):
    """Template for VLM prompts.

    Attributes:
        template: Prompt template string (use {placeholders})
        placeholders: Expected placeholder names
        examples: Few-shot examples
        system_message: System message/role description
    """

    template: str
    placeholders: List[str] = Field(default_factory=list)
    examples: List[Dict[str, str]] = Field(default_factory=list)
    system_message: Optional[str] = None


class VLMModelOptions(BaseModel):
    """Configuration options for VLM model usage in pipelines.

    Attributes:
        provider: VLM provider
        model_id: Model identifier
        api_key: API key (for API models)
        api_endpoint: Custom API endpoint
        max_tokens: Maximum output tokens
        temperature: Sampling temperature (0.0-1.0)
        top_p: Nucleus sampling threshold
        response_format: Expected response format
        prompt_template: Prompt template for CAD annotation extraction
        enable_caching: Enable response caching
        retry_attempts: Number of retry attempts on failure
        timeout_seconds: Request timeout in seconds
        batch_size: Batch size for batch processing
    """

    provider: VLMProvider
    model_id: str
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None

    # Generation parameters
    max_tokens: int = 2048
    temperature: float = 0.0  # Low temperature for deterministic extraction
    top_p: float = 1.0

    # Response handling
    response_format: VLMResponseFormat = VLMResponseFormat.JSON
    prompt_template: Optional[VLMPromptTemplate] = None

    # Performance options
    enable_caching: bool = True
    retry_attempts: int = 3
    timeout_seconds: int = 60
    batch_size: int = 1

    model_config = {"use_enum_values": True}

    @classmethod
    def from_model_id(cls, model_id: str, api_key: Optional[str] = None) -> "VLMModelOptions":
        """Create options from model ID.

        Args:
            model_id: Model identifier
            api_key: Optional API key

        Returns:
            VLMModelOptions instance
        """
        from cadling.datamodel.vlm_model_specs import VLMModelRegistry

        spec = VLMModelRegistry.get(model_id)
        if spec is None:
            raise ValueError(f"Unknown model ID: {model_id}")

        return cls(
            provider=spec.provider,
            model_id=model_id,
            api_key=api_key,
            api_endpoint=spec.api_endpoint,
            max_tokens=spec.max_tokens,
        )


# Default prompt templates

CAD_ANNOTATION_EXTRACTION_TEMPLATE = VLMPromptTemplate(
    template="""You are analyzing a technical CAD drawing image. Extract all visible dimensions, tolerances, notes, and labels.

For each annotation found, provide:
1. Type: "dimension", "tolerance", "note", or "label"
2. Value: The text content
3. Position: Approximate location (x, y coordinates normalized 0-1)
4. Unit: For dimensions (mm, inch, etc.)

Return as a JSON array of annotations.""",
    system_message="You are a technical drawing analysis assistant specialized in extracting manufacturing information from CAD drawings.",
)

DIMENSION_EXTRACTION_TEMPLATE = VLMPromptTemplate(
    template="""Analyze this technical drawing and extract all dimensional information.

Focus on:
- Linear dimensions (length, width, height)
- Radial dimensions (radii, diameters)
- Angular dimensions
- Associated tolerances

Return as structured JSON with dimension type, value, tolerance, and unit.""",
    system_message="You are a precision measurement extraction assistant for technical drawings.",
)

GDT_EXTRACTION_TEMPLATE = VLMPromptTemplate(
    template="""Extract all GD&T (Geometric Dimensioning and Tolerancing) symbols and callouts from this drawing.

For each GD&T feature control frame:
- Symbol type (flatness, straightness, perpendicularity, etc.)
- Tolerance value
- Datum references
- Material condition modifiers (MMC, LMC, RFS)

Return as structured JSON.""",
    system_message="You are a GD&T (Geometric Dimensioning and Tolerancing) expert analyzing technical drawings.",
)


def get_default_vlm_options(model_id: str = "gpt-4-vision-preview") -> VLMModelOptions:
    """Get default VLM options for CAD annotation extraction.

    Args:
        model_id: Model identifier

    Returns:
        VLMModelOptions with CAD-optimized settings
    """
    return VLMModelOptions.from_model_id(
        model_id=model_id,
        api_key=None,  # Will be loaded from environment
    )
