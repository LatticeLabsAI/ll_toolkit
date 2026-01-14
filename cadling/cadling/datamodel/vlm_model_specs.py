"""Vision-Language Model specifications and configurations.

This module provides specifications for various VLM models that can be used
for optical CAD recognition and annotation extraction.

Classes:
    VLMProvider: Enum of VLM providers
    VLMModelSpec: Specification for a VLM model
    VLMModelRegistry: Registry of available VLM models
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class VLMProvider(str, Enum):
    """Vision-Language Model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class VLMCapability(str, Enum):
    """VLM capabilities."""

    IMAGE_UNDERSTANDING = "image_understanding"
    OCR = "ocr"
    OBJECT_DETECTION = "object_detection"
    SPATIAL_REASONING = "spatial_reasoning"
    TECHNICAL_DRAWING = "technical_drawing"
    DIMENSION_EXTRACTION = "dimension_extraction"


class VLMModelSpec(BaseModel):
    """Specification for a Vision-Language Model.

    Attributes:
        model_id: Unique model identifier
        provider: Model provider
        model_name: Model name/version
        capabilities: List of supported capabilities
        max_image_size: Maximum image dimension (pixels)
        max_tokens: Maximum output tokens
        supports_batch: Whether model supports batch processing
        cost_per_1k_tokens: Approximate cost per 1K tokens
        api_endpoint: API endpoint URL (for API models)
        local_path: Local model path (for local models)
    """

    model_id: str
    provider: VLMProvider
    model_name: str
    capabilities: List[VLMCapability] = Field(default_factory=list)
    max_image_size: int = 1024
    max_tokens: int = 4096
    supports_batch: bool = False
    cost_per_1k_tokens: Optional[float] = None
    api_endpoint: Optional[str] = None
    local_path: Optional[str] = None
    description: Optional[str] = None


class VLMModelRegistry:
    """Registry of available VLM models.

    Provides a central place to register and retrieve VLM model specifications.
    """

    _models: Dict[str, VLMModelSpec] = {}

    @classmethod
    def register(cls, spec: VLMModelSpec):
        """Register a VLM model spec.

        Args:
            spec: VLMModelSpec to register
        """
        cls._models[spec.model_id] = spec

    @classmethod
    def get(cls, model_id: str) -> Optional[VLMModelSpec]:
        """Get a VLM model spec by ID.

        Args:
            model_id: Model identifier

        Returns:
            VLMModelSpec if found, None otherwise
        """
        return cls._models.get(model_id)

    @classmethod
    def list_models(
        cls,
        provider: Optional[VLMProvider] = None,
        capability: Optional[VLMCapability] = None,
    ) -> List[VLMModelSpec]:
        """List registered models.

        Args:
            provider: Filter by provider
            capability: Filter by capability

        Returns:
            List of matching VLMModelSpec
        """
        models = list(cls._models.values())

        if provider:
            models = [m for m in models if m.provider == provider]

        if capability:
            models = [m for m in models if capability in m.capabilities]

        return models


# Register default VLM models

# OpenAI GPT-4 Vision
VLMModelRegistry.register(
    VLMModelSpec(
        model_id="gpt-4-vision-preview",
        provider=VLMProvider.OPENAI,
        model_name="gpt-4-vision-preview",
        capabilities=[
            VLMCapability.IMAGE_UNDERSTANDING,
            VLMCapability.OCR,
            VLMCapability.OBJECT_DETECTION,
            VLMCapability.SPATIAL_REASONING,
            VLMCapability.TECHNICAL_DRAWING,
            VLMCapability.DIMENSION_EXTRACTION,
        ],
        max_image_size=2048,
        max_tokens=4096,
        supports_batch=False,
        cost_per_1k_tokens=0.01,
        api_endpoint="https://api.openai.com/v1/chat/completions",
        description="OpenAI GPT-4 Vision (most capable for technical drawings)",
    )
)

VLMModelRegistry.register(
    VLMModelSpec(
        model_id="gpt-4o",
        provider=VLMProvider.OPENAI,
        model_name="gpt-4o",
        capabilities=[
            VLMCapability.IMAGE_UNDERSTANDING,
            VLMCapability.OCR,
            VLMCapability.OBJECT_DETECTION,
            VLMCapability.SPATIAL_REASONING,
            VLMCapability.TECHNICAL_DRAWING,
            VLMCapability.DIMENSION_EXTRACTION,
        ],
        max_image_size=2048,
        max_tokens=4096,
        supports_batch=True,
        cost_per_1k_tokens=0.005,
        api_endpoint="https://api.openai.com/v1/chat/completions",
        description="OpenAI GPT-4o (optimized, faster and cheaper)",
    )
)

# Anthropic Claude
VLMModelRegistry.register(
    VLMModelSpec(
        model_id="claude-3-opus",
        provider=VLMProvider.ANTHROPIC,
        model_name="claude-3-opus-20240229",
        capabilities=[
            VLMCapability.IMAGE_UNDERSTANDING,
            VLMCapability.OCR,
            VLMCapability.OBJECT_DETECTION,
            VLMCapability.SPATIAL_REASONING,
            VLMCapability.TECHNICAL_DRAWING,
            VLMCapability.DIMENSION_EXTRACTION,
        ],
        max_image_size=1568,
        max_tokens=4096,
        supports_batch=False,
        cost_per_1k_tokens=0.015,
        api_endpoint="https://api.anthropic.com/v1/messages",
        description="Claude 3 Opus (excellent for complex technical analysis)",
    )
)

VLMModelRegistry.register(
    VLMModelSpec(
        model_id="claude-3-sonnet",
        provider=VLMProvider.ANTHROPIC,
        model_name="claude-3-sonnet-20240229",
        capabilities=[
            VLMCapability.IMAGE_UNDERSTANDING,
            VLMCapability.OCR,
            VLMCapability.SPATIAL_REASONING,
            VLMCapability.TECHNICAL_DRAWING,
        ],
        max_image_size=1568,
        max_tokens=4096,
        supports_batch=False,
        cost_per_1k_tokens=0.003,
        api_endpoint="https://api.anthropic.com/v1/messages",
        description="Claude 3 Sonnet (balanced performance and cost)",
    )
)

# Google Gemini
VLMModelRegistry.register(
    VLMModelSpec(
        model_id="gemini-pro-vision",
        provider=VLMProvider.GOOGLE,
        model_name="gemini-pro-vision",
        capabilities=[
            VLMCapability.IMAGE_UNDERSTANDING,
            VLMCapability.OCR,
            VLMCapability.OBJECT_DETECTION,
        ],
        max_image_size=2048,
        max_tokens=2048,
        supports_batch=False,
        cost_per_1k_tokens=0.0025,
        api_endpoint="https://generativelanguage.googleapis.com/v1beta/models",
        description="Google Gemini Pro Vision (cost-effective)",
    )
)

# Local/HuggingFace models
VLMModelRegistry.register(
    VLMModelSpec(
        model_id="llava-1.5-7b",
        provider=VLMProvider.HUGGINGFACE,
        model_name="llava-hf/llava-1.5-7b-hf",
        capabilities=[
            VLMCapability.IMAGE_UNDERSTANDING,
            VLMCapability.OCR,
        ],
        max_image_size=336,
        max_tokens=2048,
        supports_batch=True,
        cost_per_1k_tokens=0.0,  # Free (local)
        local_path="llava-hf/llava-1.5-7b-hf",
        description="LLaVA 1.5 7B (free, runs locally)",
    )
)

VLMModelRegistry.register(
    VLMModelSpec(
        model_id="llava-1.5-13b",
        provider=VLMProvider.HUGGINGFACE,
        model_name="llava-hf/llava-1.5-13b-hf",
        capabilities=[
            VLMCapability.IMAGE_UNDERSTANDING,
            VLMCapability.OCR,
            VLMCapability.SPATIAL_REASONING,
        ],
        max_image_size=336,
        max_tokens=2048,
        supports_batch=True,
        cost_per_1k_tokens=0.0,  # Free (local)
        local_path="llava-hf/llava-1.5-13b-hf",
        description="LLaVA 1.5 13B (better quality, runs locally)",
    )
)

VLMModelRegistry.register(
    VLMModelSpec(
        model_id="qwen-vl-chat",
        provider=VLMProvider.HUGGINGFACE,
        model_name="Qwen/Qwen-VL-Chat",
        capabilities=[
            VLMCapability.IMAGE_UNDERSTANDING,
            VLMCapability.OCR,
            VLMCapability.SPATIAL_REASONING,
        ],
        max_image_size=448,
        max_tokens=2048,
        supports_batch=True,
        cost_per_1k_tokens=0.0,  # Free (local)
        local_path="Qwen/Qwen-VL-Chat",
        description="Qwen-VL (good for technical content, runs locally)",
    )
)


def get_recommended_model_for_cad() -> str:
    """Get recommended model ID for CAD annotation extraction.

    Returns:
        Recommended model ID
    """
    # GPT-4 Vision is currently the most capable for technical drawings
    return "gpt-4-vision-preview"


def get_cost_effective_model_for_cad() -> str:
    """Get cost-effective model ID for CAD annotation extraction.

    Returns:
        Cost-effective model ID
    """
    # GPT-4o provides good balance of quality and cost
    return "gpt-4o"


def get_local_model_for_cad() -> str:
    """Get local (free) model ID for CAD annotation extraction.

    Returns:
        Local model ID
    """
    # LLaVA 1.5 13B is the best free option
    return "llava-1.5-13b"
