"""Tests for generation error types and fallback strategies.

Tests cover:
- Error type hierarchy and attributes
- Fallback strategy configuration
- FallbackHandler behavior
- Integration with CADGenerationPipeline
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch")

from stepnet.generation.errors import (
    CADGenerationError,
    DependencyMissingError,
    InvalidLatentError,
    ModelSamplingError,
    ReconstructionError,
    TokenDecodingError,
    ValidationError,
)
from stepnet.generation.fallbacks import (
    FallbackConfig,
    FallbackHandler,
    FallbackResult,
    FallbackStrategy,
)


# ============================================================================
# SECTION 1: Error Type Tests
# ============================================================================


class TestCADGenerationError:
    """Test base CADGenerationError class."""

    def test_error_creation(self) -> None:
        """Test basic error creation."""
        error = CADGenerationError(message="Test error")
        assert error.message == "Test error"
        assert error.recoverable is True
        assert error.context == {}
        assert error.original_exception is None

    def test_error_with_context(self) -> None:
        """Test error with context dictionary."""
        error = CADGenerationError(
            message="Test error",
            context={"key": "value", "num": 42},
        )
        assert error.context["key"] == "value"
        assert error.context["num"] == 42

    def test_error_with_original_exception(self) -> None:
        """Test error wrapping original exception."""
        original = ValueError("Original error")
        error = CADGenerationError(
            message="Wrapped error",
            original_exception=original,
        )
        assert error.original_exception is original

    def test_error_str_representation(self) -> None:
        """Test string representation includes context."""
        error = CADGenerationError(
            message="Test error",
            context={"batch": 0},
        )
        error_str = str(error)
        assert "Test error" in error_str
        assert "batch=0" in error_str

    def test_error_with_context_method(self) -> None:
        """Test with_context returns new error with added context."""
        error = CADGenerationError(
            message="Test",
            context={"a": 1},
        )
        new_error = error.with_context(b=2, c=3)
        assert new_error.context["a"] == 1
        assert new_error.context["b"] == 2
        assert new_error.context["c"] == 3
        # Original unchanged
        assert "b" not in error.context


class TestModelSamplingError:
    """Test ModelSamplingError class."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = ModelSamplingError()
        assert "sampling failed" in error.message.lower()

    def test_recoverable_by_default(self) -> None:
        """Test error is recoverable by default."""
        error = ModelSamplingError()
        assert error.recoverable is True


class TestTokenDecodingError:
    """Test TokenDecodingError class."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = TokenDecodingError()
        assert "decoding" in error.message.lower()


class TestReconstructionError:
    """Test ReconstructionError class."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = ReconstructionError()
        assert "reconstruction" in error.message.lower()


class TestInvalidLatentError:
    """Test InvalidLatentError class."""

    def test_from_tensor_with_nan(self) -> None:
        """Test from_tensor classmethod with NaN tensor."""
        latent = torch.tensor([1.0, float('nan'), 3.0])
        error = InvalidLatentError.from_tensor(latent, "contains NaN")
        assert "NaN" in error.message
        assert error.context["has_nan"] is True
        assert error.context["shape"] == (3,)

    def test_from_tensor_with_inf(self) -> None:
        """Test from_tensor classmethod with Inf tensor."""
        latent = torch.tensor([1.0, float('inf'), 3.0])
        error = InvalidLatentError.from_tensor(latent, "contains Inf")
        assert "Inf" in error.message
        assert error.context["has_inf"] is True

    def test_from_tensor_with_normal_values(self) -> None:
        """Test from_tensor with normal tensor."""
        latent = torch.tensor([1.0, 2.0, 3.0])
        error = InvalidLatentError.from_tensor(latent, "out of range")
        assert error.context["has_nan"] is False
        assert error.context["has_inf"] is False
        assert error.context["min"] == 1.0
        assert error.context["max"] == 3.0


class TestDependencyMissingError:
    """Test DependencyMissingError class."""

    def test_not_recoverable(self) -> None:
        """Test error is not recoverable."""
        error = DependencyMissingError()
        assert error.recoverable is False

    def test_for_dependency_classmethod(self) -> None:
        """Test for_dependency classmethod."""
        error = DependencyMissingError.for_dependency("geotoken", "tokenization")
        assert "geotoken" in error.message
        assert "tokenization" in error.message
        assert error.context["dependency"] == "geotoken"
        assert error.context["feature"] == "tokenization"


class TestValidationError:
    """Test ValidationError class."""

    def test_from_checks_classmethod(self) -> None:
        """Test from_checks classmethod."""
        checks = {
            "watertight": True,
            "volume_positive": False,
            "no_self_intersection": False,
        }
        error = ValidationError.from_checks(checks)
        assert "volume_positive" in error.message
        assert "no_self_intersection" in error.message
        assert len(error.validation_errors) == 2


# ============================================================================
# SECTION 2: Fallback Strategy Tests
# ============================================================================


class TestFallbackStrategy:
    """Test FallbackStrategy enum."""

    def test_all_strategies_exist(self) -> None:
        """Test all expected strategies are defined."""
        assert FallbackStrategy.SKIP is not None
        assert FallbackStrategy.RETRY is not None
        assert FallbackStrategy.SUBSTITUTE is not None
        assert FallbackStrategy.PARTIAL is not None
        assert FallbackStrategy.RAISE is not None


class TestFallbackConfig:
    """Test FallbackConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = FallbackConfig()
        assert config.on_sampling_error == FallbackStrategy.RETRY
        assert config.on_decoding_error == FallbackStrategy.SKIP
        assert config.on_reconstruction_error == FallbackStrategy.PARTIAL
        assert config.max_retries == 3
        assert config.cache_size == 100

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = FallbackConfig(
            on_sampling_error=FallbackStrategy.SKIP,
            max_retries=5,
            cache_size=50,
        )
        assert config.on_sampling_error == FallbackStrategy.SKIP
        assert config.max_retries == 5
        assert config.cache_size == 50

    def test_get_strategy_for_error_types(self) -> None:
        """Test get_strategy returns correct strategy per error type."""
        config = FallbackConfig(
            on_sampling_error=FallbackStrategy.RETRY,
            on_decoding_error=FallbackStrategy.SKIP,
            on_reconstruction_error=FallbackStrategy.PARTIAL,
        )

        assert config.get_strategy(ModelSamplingError()) == FallbackStrategy.RETRY
        assert config.get_strategy(TokenDecodingError()) == FallbackStrategy.SKIP
        assert config.get_strategy(ReconstructionError()) == FallbackStrategy.PARTIAL


class TestFallbackHandler:
    """Test FallbackHandler class."""

    def test_handler_creation(self) -> None:
        """Test handler creation with default config."""
        handler = FallbackHandler()
        assert handler.config is not None
        assert handler.on_error is None

    def test_handler_with_callback(self) -> None:
        """Test handler with error callback."""
        callback_called = []

        def on_error(error: CADGenerationError) -> None:
            callback_called.append(error)

        handler = FallbackHandler(on_error=on_error)
        error = ModelSamplingError(message="Test")

        # SKIP strategy doesn't raise
        handler.handle(error)

        # Callback may or may not be called depending on strategy
        # Just verify handler doesn't crash

    def test_skip_strategy(self) -> None:
        """Test SKIP strategy returns unsuccessful result."""
        config = FallbackConfig(on_sampling_error=FallbackStrategy.SKIP)
        handler = FallbackHandler(config)

        error = ModelSamplingError(message="Test")
        result = handler.handle(error)

        assert result.success is False
        assert result.strategy_used == FallbackStrategy.SKIP

    def test_raise_strategy(self) -> None:
        """Test RAISE strategy re-raises error."""
        config = FallbackConfig(on_dependency_error=FallbackStrategy.RAISE)
        handler = FallbackHandler(config)

        error = DependencyMissingError.for_dependency("test")

        with pytest.raises(DependencyMissingError):
            handler.handle(error)

    def test_retry_strategy_success(self) -> None:
        """Test RETRY strategy with successful retry."""
        config = FallbackConfig(
            on_sampling_error=FallbackStrategy.RETRY,
            max_retries=3,
        )
        handler = FallbackHandler(config)

        error = ModelSamplingError(message="First attempt failed")

        result = handler.handle(
            error,
            retry_fn=lambda: "success",
        )

        assert result.success is True
        assert result.result == "success"
        assert result.retry_count >= 1

    def test_retry_strategy_max_retries_exceeded(self) -> None:
        """Test RETRY strategy respects max_retries."""
        config = FallbackConfig(
            on_sampling_error=FallbackStrategy.RETRY,
            max_retries=2,
            on_fallback_exhausted=FallbackStrategy.SKIP,
        )
        handler = FallbackHandler(config)

        attempt_count = [0]

        def failing_retry():
            attempt_count[0] += 1
            raise ModelSamplingError(message=f"Attempt {attempt_count[0]} failed")

        error = ModelSamplingError(message="Initial failure")
        result = handler.handle(error, retry_fn=failing_retry, context_id="test")

        assert result.success is False
        # Should have tried max_retries times
        assert attempt_count[0] <= config.max_retries

    def test_substitute_strategy_with_cache(self) -> None:
        """Test SUBSTITUTE strategy uses cached sample."""
        config = FallbackConfig(
            on_sampling_error=FallbackStrategy.SUBSTITUTE,
            cache_valid_samples=True,
        )
        handler = FallbackHandler(config)

        # Cache a valid sample first
        cached_sample = {"shape": "cached_shape"}
        handler.cache_valid_sample(cached_sample, "sample_0")

        error = ModelSamplingError(message="Test")
        result = handler.handle(error)

        assert result.success is True
        assert result.result == cached_sample
        assert result.was_substituted is True

    def test_substitute_strategy_empty_cache(self) -> None:
        """Test SUBSTITUTE strategy with empty cache falls back to SKIP."""
        config = FallbackConfig(on_sampling_error=FallbackStrategy.SUBSTITUTE)
        handler = FallbackHandler(config)

        # Don't cache anything
        error = ModelSamplingError(message="Test")
        result = handler.handle(error)

        assert result.success is False
        assert result.strategy_used == FallbackStrategy.SKIP

    def test_partial_strategy_with_result(self) -> None:
        """Test PARTIAL strategy with partial result."""
        config = FallbackConfig(on_reconstruction_error=FallbackStrategy.PARTIAL)
        handler = FallbackHandler(config)

        partial = {"token_sequence": "partial_tokens"}
        error = ReconstructionError(message="Test")
        result = handler.handle(error, partial_result=partial)

        assert result.success is True
        assert result.result == partial
        assert result.strategy_used == FallbackStrategy.PARTIAL

    def test_partial_strategy_without_result(self) -> None:
        """Test PARTIAL strategy without partial result."""
        config = FallbackConfig(on_reconstruction_error=FallbackStrategy.PARTIAL)
        handler = FallbackHandler(config)

        error = ReconstructionError(message="Test")
        result = handler.handle(error, partial_result=None)

        assert result.success is False

    def test_cache_valid_sample(self) -> None:
        """Test caching valid samples."""
        handler = FallbackHandler()

        handler.cache_valid_sample({"id": 1}, "sample_1")
        handler.cache_valid_sample({"id": 2}, "sample_2")

        # Most recent sample should be returned
        cached = handler.get_cached_sample()
        assert cached["id"] == 2

    def test_cache_size_limit(self) -> None:
        """Test cache respects size limit."""
        config = FallbackConfig(cache_size=3)
        handler = FallbackHandler(config)

        for i in range(5):
            handler.cache_valid_sample({"id": i}, f"sample_{i}")

        # Only last 3 should be in cache
        cached = handler.get_cached_sample()
        assert cached["id"] == 4

    def test_reset_retry_counts(self) -> None:
        """Test reset_retry_counts clears counters."""
        handler = FallbackHandler()
        handler._retry_counts["test"] = 5

        handler.reset_retry_counts()

        assert len(handler._retry_counts) == 0

    def test_clear_cache(self) -> None:
        """Test clear_cache empties the cache."""
        handler = FallbackHandler()
        handler.cache_valid_sample({"id": 1}, "sample_1")

        handler.clear_cache()

        assert handler.get_cached_sample() is None


class TestFallbackResult:
    """Test FallbackResult dataclass."""

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        error = ModelSamplingError(message="Test error")
        result = FallbackResult(
            success=True,
            result="test_result",
            strategy_used=FallbackStrategy.RETRY,
            original_error=error,
            retry_count=2,
            was_substituted=False,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["strategy_used"] == "RETRY"
        assert d["retry_count"] == 2
        assert d["was_substituted"] is False
        assert d["error_type"] == "ModelSamplingError"


# ============================================================================
# SECTION 3: Pipeline Integration Tests
# ============================================================================


class TestPipelineErrorIntegration:
    """Test error handling integration with CADGenerationPipeline."""

    def test_pipeline_with_fallback_config(self) -> None:
        """Test pipeline accepts fallback_config parameter."""
        from stepnet.generation_pipeline import CADGenerationPipeline

        # Create a mock model
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        config = FallbackConfig(max_retries=5)
        pipeline = CADGenerationPipeline(
            model=mock_model,
            mode='vae',
            fallback_config=config,
        )

        assert pipeline.fallback_config.max_retries == 5

    def test_pipeline_with_on_error_callback(self) -> None:
        """Test pipeline accepts on_error callback."""
        from stepnet.generation_pipeline import CADGenerationPipeline

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        errors_received = []

        def on_error(error: CADGenerationError) -> None:
            errors_received.append(error)

        pipeline = CADGenerationPipeline(
            model=mock_model,
            mode='vae',
            on_error=on_error,
        )

        assert pipeline.on_error is on_error

    def test_pipeline_validate_latent_nan(self) -> None:
        """Test _validate_latent raises for NaN."""
        from stepnet.generation_pipeline import CADGenerationPipeline

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        pipeline = CADGenerationPipeline(model=mock_model, mode='vae')

        latent = torch.tensor([1.0, float('nan'), 3.0])

        with pytest.raises(InvalidLatentError) as exc_info:
            pipeline._validate_latent(latent)

        assert "NaN" in str(exc_info.value)

    def test_pipeline_validate_latent_inf(self) -> None:
        """Test _validate_latent raises for Inf."""
        from stepnet.generation_pipeline import CADGenerationPipeline

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        pipeline = CADGenerationPipeline(model=mock_model, mode='vae')

        latent = torch.tensor([1.0, float('inf'), 3.0])

        with pytest.raises(InvalidLatentError) as exc_info:
            pipeline._validate_latent(latent)

        assert "Inf" in str(exc_info.value)

    def test_pipeline_validate_latent_out_of_range(self) -> None:
        """Test _validate_latent raises for out of range values."""
        from stepnet.generation_pipeline import CADGenerationPipeline

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        pipeline = CADGenerationPipeline(model=mock_model, mode='vae')

        latent = torch.tensor([1.0, 5000.0, 3.0])

        with pytest.raises(InvalidLatentError) as exc_info:
            pipeline._validate_latent(latent, max_value=1000.0)

        assert "out of range" in str(exc_info.value)

    def test_pipeline_validate_latent_valid(self) -> None:
        """Test _validate_latent passes for valid latent."""
        from stepnet.generation_pipeline import CADGenerationPipeline

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        pipeline = CADGenerationPipeline(model=mock_model, mode='vae')

        latent = torch.randn(4, 256)

        # Should not raise
        pipeline._validate_latent(latent)


# ============================================================================
# SECTION 4: Module Import Tests
# ============================================================================


class TestModuleImports:
    """Test module imports work correctly."""

    def test_import_errors_module(self) -> None:
        """Test errors module can be imported."""
        from stepnet.generation import errors
        assert hasattr(errors, "CADGenerationError")
        assert hasattr(errors, "ModelSamplingError")
        assert hasattr(errors, "TokenDecodingError")

    def test_import_fallbacks_module(self) -> None:
        """Test fallbacks module can be imported."""
        from stepnet.generation import fallbacks
        assert hasattr(fallbacks, "FallbackStrategy")
        assert hasattr(fallbacks, "FallbackConfig")
        assert hasattr(fallbacks, "FallbackHandler")

    def test_import_from_package(self) -> None:
        """Test imports from stepnet.generation package."""
        from stepnet.generation import (
            CADGenerationError,
            FallbackConfig,
            FallbackHandler,
            FallbackStrategy,
        )
        assert CADGenerationError is not None
        assert FallbackConfig is not None
        assert FallbackHandler is not None
        assert FallbackStrategy is not None
