"""Error types for CAD generation pipeline.

This module defines a hierarchy of exceptions for different failure modes
in the CAD generation pipeline, enabling structured error handling and
fallback strategies.

Exception Hierarchy:
    CADGenerationError (base)
    ├── ModelSamplingError      - VAE/Diffusion sampling fails
    ├── TokenDecodingError      - Token → TokenSequence fails
    ├── ReconstructionError     - Geometry reconstruction fails
    ├── InvalidLatentError      - NaN, Inf, out of bounds latents
    ├── DependencyMissingError  - geotoken/cadling not installed
    └── ValidationError         - Generated geometry invalid
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CADGenerationError(Exception):
    """Base exception for CAD generation errors.

    Attributes:
        message: Human-readable error description.
        recoverable: Whether the error can be handled with a fallback strategy.
        context: Additional context about the error (latent values, sample id, etc.).
        original_exception: The underlying exception that caused this error.
    """

    message: str
    recoverable: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    original_exception: Optional[Exception] = None

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def __reduce__(self):
        return (self.__class__, (self.message, self.recoverable, self.context, self.original_exception))

    def __str__(self) -> str:
        parts = [self.message]
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"[context: {context_str}]")
        if self.original_exception:
            parts.append(f"[caused by: {type(self.original_exception).__name__}: {self.original_exception}]")
        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"recoverable={self.recoverable}, "
            f"context={self.context!r})"
        )

    def with_context(self, **kwargs: Any) -> "CADGenerationError":
        """Return a copy with additional context."""
        new_context = {**self.context, **kwargs}
        return self.__class__(
            message=self.message,
            recoverable=self.recoverable,
            context=new_context,
            original_exception=self.original_exception,
        )


@dataclass
class ModelSamplingError(CADGenerationError):
    """Error during model sampling (VAE, Diffusion, GAN).

    Raised when:
    - VAE reparameterization produces invalid values
    - Diffusion reverse process fails
    - GAN generator produces degenerate outputs
    """

    message: str = "Model sampling failed"
    recoverable: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    original_exception: Optional[Exception] = None


@dataclass
class TokenDecodingError(CADGenerationError):
    """Error decoding tokens to TokenSequence.

    Raised when:
    - Token IDs are out of vocabulary range
    - Sequence structure is invalid (missing SOL/EOS)
    - Parameter dequantization fails
    """

    message: str = "Token decoding failed"
    recoverable: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    original_exception: Optional[Exception] = None


@dataclass
class ReconstructionError(CADGenerationError):
    """Error reconstructing geometry from command sequence.

    Raised when:
    - Sketch loop construction fails
    - Extrusion produces invalid solid
    - Boolean operations fail
    - pythonocc/cadling execution errors
    """

    message: str = "Geometry reconstruction failed"
    recoverable: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    original_exception: Optional[Exception] = None


@dataclass
class InvalidLatentError(CADGenerationError):
    """Latent vector is invalid (NaN, Inf, out of bounds).

    Raised when:
    - Latent contains NaN values
    - Latent contains Inf values
    - Latent values exceed expected range
    """

    message: str = "Invalid latent vector"
    recoverable: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    original_exception: Optional[Exception] = None

    @classmethod
    def from_tensor(cls, latent: Any, reason: str) -> "InvalidLatentError":
        """Create from a tensor with diagnostic information."""
        import torch

        context = {"reason": reason}
        if isinstance(latent, torch.Tensor):
            context.update({
                "shape": tuple(latent.shape),
                "dtype": str(latent.dtype),
                "has_nan": bool(torch.isnan(latent).any()),
                "has_inf": bool(torch.isinf(latent).any()),
                "min": float(latent.min()) if latent.numel() > 0 else None,
                "max": float(latent.max()) if latent.numel() > 0 else None,
            })
        return cls(
            message=f"Invalid latent vector: {reason}",
            context=context,
        )


@dataclass
class DependencyMissingError(CADGenerationError):
    """Required dependency not installed.

    Raised when:
    - geotoken is required but not installed
    - cadling is required but not installed
    - pythonocc is required but not installed
    """

    message: str = "Required dependency not installed"
    recoverable: bool = False  # Cannot recover from missing dependencies
    context: Dict[str, Any] = field(default_factory=dict)
    original_exception: Optional[Exception] = None
    dependency_name: str = ""

    @classmethod
    def for_dependency(cls, name: str, feature: str = "") -> "DependencyMissingError":
        """Create for a specific missing dependency."""
        msg = f"Required dependency '{name}' not installed"
        if feature:
            msg += f" (needed for {feature})"
        return cls(
            message=msg,
            context={"dependency": name, "feature": feature},
            dependency_name=name,
        )


@dataclass
class ValidationError(CADGenerationError):
    """Generated geometry failed validation.

    Raised when:
    - Shape has zero volume
    - Shape is not watertight
    - Shape has self-intersections
    - Shape fails BRep validity checks
    """

    message: str = "Generated geometry failed validation"
    recoverable: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    original_exception: Optional[Exception] = None
    validation_errors: list = field(default_factory=list)

    @classmethod
    def from_checks(
        cls,
        checks: Dict[str, bool],
        details: Optional[Dict[str, Any]] = None,
    ) -> "ValidationError":
        """Create from a dict of validation check results."""
        failed = [name for name, passed in checks.items() if not passed]
        context = {
            "failed_checks": failed,
            "all_checks": checks,
        }
        if details:
            context.update(details)
        return cls(
            message=f"Validation failed: {', '.join(failed)}",
            context=context,
            validation_errors=failed,
        )
