"""Fallback strategies for CAD generation error recovery.

This module provides configurable fallback strategies that can be applied
when errors occur during CAD generation. Each error type can have a different
fallback strategy assigned.

Strategies:
    SKIP       - Skip the failed sample and continue
    RETRY      - Retry generation with a different random seed
    SUBSTITUTE - Replace with a cached valid sample
    PARTIAL    - Return partial results (what was generated so far)
    RAISE      - Re-raise the exception (no fallback)
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TypeVar

from stepnet.generation.errors import (
    CADGenerationError,
    DependencyMissingError,
    InvalidLatentError,
    ModelSamplingError,
    ReconstructionError,
    TokenDecodingError,
    ValidationError,
)

_log = logging.getLogger(__name__)

T = TypeVar("T")


class FallbackStrategy(Enum):
    """Available fallback strategies for error recovery."""

    SKIP = auto()
    """Skip the failed sample and continue with next."""

    RETRY = auto()
    """Retry generation with a different random seed."""

    SUBSTITUTE = auto()
    """Replace with a valid sample from cache."""

    PARTIAL = auto()
    """Return partial results (what was successfully generated)."""

    RAISE = auto()
    """Re-raise the exception (no fallback)."""


@dataclass
class FallbackConfig:
    """Configuration for fallback strategies per error type.

    Attributes:
        on_sampling_error: Strategy for ModelSamplingError.
        on_decoding_error: Strategy for TokenDecodingError.
        on_reconstruction_error: Strategy for ReconstructionError.
        on_invalid_latent_error: Strategy for InvalidLatentError.
        on_validation_error: Strategy for ValidationError.
        on_dependency_error: Strategy for DependencyMissingError (default RAISE).
        max_retries: Maximum retry attempts before falling back to next strategy.
        cache_valid_samples: Whether to cache successful samples for substitution.
        cache_size: Maximum number of samples to cache.
        on_fallback_exhausted: Strategy when all fallbacks fail.
    """

    on_sampling_error: FallbackStrategy = FallbackStrategy.RETRY
    on_decoding_error: FallbackStrategy = FallbackStrategy.SKIP
    on_reconstruction_error: FallbackStrategy = FallbackStrategy.PARTIAL
    on_invalid_latent_error: FallbackStrategy = FallbackStrategy.RETRY
    on_validation_error: FallbackStrategy = FallbackStrategy.PARTIAL
    on_dependency_error: FallbackStrategy = FallbackStrategy.RAISE
    max_retries: int = 3
    cache_valid_samples: bool = True
    cache_size: int = 100
    on_fallback_exhausted: FallbackStrategy = FallbackStrategy.SKIP

    def get_strategy(self, error: CADGenerationError) -> FallbackStrategy:
        """Get the fallback strategy for a specific error type."""
        if isinstance(error, ModelSamplingError):
            return self.on_sampling_error
        elif isinstance(error, TokenDecodingError):
            return self.on_decoding_error
        elif isinstance(error, ReconstructionError):
            return self.on_reconstruction_error
        elif isinstance(error, InvalidLatentError):
            return self.on_invalid_latent_error
        elif isinstance(error, ValidationError):
            return self.on_validation_error
        elif isinstance(error, DependencyMissingError):
            return self.on_dependency_error
        else:
            # Default for unknown error types
            return FallbackStrategy.SKIP


@dataclass
class FallbackResult:
    """Result of applying a fallback strategy.

    Attributes:
        success: Whether the fallback produced a usable result.
        result: The fallback result (may be partial or substituted).
        strategy_used: Which strategy was applied.
        original_error: The error that triggered the fallback.
        retry_count: How many retries were attempted.
        was_substituted: Whether a cached sample was used.
    """

    success: bool
    result: Optional[Any] = None
    strategy_used: FallbackStrategy = FallbackStrategy.SKIP
    original_error: Optional[CADGenerationError] = None
    retry_count: int = 0
    was_substituted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "strategy_used": self.strategy_used.name,
            "retry_count": self.retry_count,
            "was_substituted": self.was_substituted,
            "error_type": type(self.original_error).__name__ if self.original_error else None,
            "error_message": str(self.original_error) if self.original_error else None,
        }


class FallbackHandler:
    """Handle errors with configurable fallback strategies.

    This class manages error recovery during CAD generation, applying
    the appropriate fallback strategy based on the error type and
    configuration.

    Example:
        >>> config = FallbackConfig(max_retries=3)
        >>> handler = FallbackHandler(config)
        >>> try:
        ...     result = generate_cad(latent)
        ... except CADGenerationError as e:
        ...     fallback = handler.handle(e, retry_fn=lambda: generate_cad(new_latent))
        ...     if fallback.success:
        ...         result = fallback.result
    """

    def __init__(
        self,
        config: Optional[FallbackConfig] = None,
        on_error: Optional[Callable[[CADGenerationError], None]] = None,
    ) -> None:
        """Initialize the fallback handler.

        Args:
            config: Fallback configuration. Uses defaults if not provided.
            on_error: Optional callback invoked when an error is handled.
        """
        self.config = config or FallbackConfig()
        self.on_error = on_error
        self._valid_cache: deque = deque(maxlen=self.config.cache_size)
        self._retry_counts: Dict[str, int] = {}

    def cache_valid_sample(self, sample: Any, sample_id: Optional[str] = None) -> None:
        """Cache a successfully generated sample for potential substitution.

        Args:
            sample: The valid sample to cache.
            sample_id: Optional identifier for the sample.
        """
        if self.config.cache_valid_samples:
            self._valid_cache.append({"sample": sample, "id": sample_id})
            _log.debug(f"Cached valid sample (cache size: {len(self._valid_cache)})")

    def get_cached_sample(self) -> Optional[Any]:
        """Get a cached valid sample for substitution.

        Returns:
            A cached sample, or None if cache is empty.
        """
        if self._valid_cache:
            return self._valid_cache[-1]["sample"]
        return None

    def handle(
        self,
        error: CADGenerationError,
        retry_fn: Optional[Callable[[], T]] = None,
        partial_result: Optional[Any] = None,
        context_id: Optional[str] = None,
    ) -> FallbackResult:
        """Apply the appropriate fallback strategy for an error.

        Args:
            error: The error to handle.
            retry_fn: Function to call for RETRY strategy.
            partial_result: Partial result to return for PARTIAL strategy.
            context_id: Identifier for tracking retries across calls.

        Returns:
            FallbackResult with the outcome of the fallback attempt.

        Raises:
            CADGenerationError: If strategy is RAISE or no fallback succeeds.
        """
        # Invoke error callback if configured
        if self.on_error:
            try:
                self.on_error(error)
            except Exception as callback_error:
                _log.warning(f"Error callback failed: {callback_error}")

        strategy = self.config.get_strategy(error)
        _log.info(f"Handling {type(error).__name__} with strategy {strategy.name}")

        # Track retries per context
        retry_key = context_id or str(id(error))
        current_retries = self._retry_counts.get(retry_key, 0)

        if strategy == FallbackStrategy.RAISE:
            raise error

        if strategy == FallbackStrategy.SKIP:
            return FallbackResult(
                success=False,
                strategy_used=strategy,
                original_error=error,
            )

        if strategy == FallbackStrategy.RETRY:
            if retry_fn is None:
                _log.warning("RETRY strategy but no retry_fn provided, falling back to SKIP")
                return FallbackResult(
                    success=False,
                    strategy_used=FallbackStrategy.SKIP,
                    original_error=error,
                )

            if current_retries >= self.config.max_retries:
                _log.warning(f"Max retries ({self.config.max_retries}) exceeded")
                return self._apply_exhausted_fallback(error, partial_result)

            self._retry_counts[retry_key] = current_retries + 1

            try:
                result = retry_fn()
                # Clear retry count on success
                self._retry_counts.pop(retry_key, None)
                return FallbackResult(
                    success=True,
                    result=result,
                    strategy_used=strategy,
                    original_error=error,
                    retry_count=current_retries + 1,
                )
            except CADGenerationError as retry_error:
                _log.debug(f"Retry {current_retries + 1} failed: {retry_error}")
                return self.handle(
                    retry_error,
                    retry_fn=retry_fn,
                    partial_result=partial_result,
                    context_id=retry_key,
                )

        if strategy == FallbackStrategy.SUBSTITUTE:
            cached = self.get_cached_sample()
            if cached is not None:
                return FallbackResult(
                    success=True,
                    result=cached,
                    strategy_used=strategy,
                    original_error=error,
                    was_substituted=True,
                )
            else:
                _log.warning("SUBSTITUTE strategy but cache is empty, falling back to SKIP")
                return FallbackResult(
                    success=False,
                    strategy_used=FallbackStrategy.SKIP,
                    original_error=error,
                )

        if strategy == FallbackStrategy.PARTIAL:
            return FallbackResult(
                success=partial_result is not None,
                result=partial_result,
                strategy_used=strategy,
                original_error=error,
            )

        # Should not reach here
        return FallbackResult(
            success=False,
            strategy_used=FallbackStrategy.SKIP,
            original_error=error,
        )

    def _apply_exhausted_fallback(
        self,
        error: CADGenerationError,
        partial_result: Optional[Any],
    ) -> FallbackResult:
        """Apply fallback when max retries are exhausted."""
        exhausted_strategy = self.config.on_fallback_exhausted

        if exhausted_strategy == FallbackStrategy.RAISE:
            raise error

        if exhausted_strategy == FallbackStrategy.SUBSTITUTE:
            cached = self.get_cached_sample()
            if cached is not None:
                return FallbackResult(
                    success=True,
                    result=cached,
                    strategy_used=exhausted_strategy,
                    original_error=error,
                    was_substituted=True,
                )

        if exhausted_strategy == FallbackStrategy.PARTIAL and partial_result is not None:
            return FallbackResult(
                success=True,
                result=partial_result,
                strategy_used=exhausted_strategy,
                original_error=error,
            )

        return FallbackResult(
            success=False,
            strategy_used=FallbackStrategy.SKIP,
            original_error=error,
        )

    def reset_retry_counts(self) -> None:
        """Reset all retry counters."""
        self._retry_counts.clear()

    def clear_cache(self) -> None:
        """Clear the valid sample cache."""
        self._valid_cache.clear()
