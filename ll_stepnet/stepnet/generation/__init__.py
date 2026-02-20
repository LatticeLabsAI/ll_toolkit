"""Generation module for CAD shape generation.

This module provides:
- Error types for generation failures
- Fallback strategies for error recovery
- Beam search decoding utilities
- Generation configuration
"""
from __future__ import annotations

from stepnet.generation.errors import (
    CADGenerationError,
    ModelSamplingError,
    TokenDecodingError,
    ReconstructionError,
    InvalidLatentError,
    DependencyMissingError,
    ValidationError,
)
from stepnet.generation.fallbacks import (
    FallbackStrategy,
    FallbackConfig,
    FallbackResult,
    FallbackHandler,
)
from stepnet.generation.config import (
    DecodingStrategy,
    GenerationConfig,
)
from stepnet.generation.beam_search import (
    GenerationOutput,
    BeamSearchDecoder,
    generate_with_model,
)

__all__ = [
    # Errors
    "CADGenerationError",
    "ModelSamplingError",
    "TokenDecodingError",
    "ReconstructionError",
    "InvalidLatentError",
    "DependencyMissingError",
    "ValidationError",
    # Fallbacks
    "FallbackStrategy",
    "FallbackConfig",
    "FallbackResult",
    "FallbackHandler",
    # Generation config
    "DecodingStrategy",
    "GenerationConfig",
    # Beam search
    "GenerationOutput",
    "BeamSearchDecoder",
    "generate_with_model",
]
