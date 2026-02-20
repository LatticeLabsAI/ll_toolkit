"""Abstract base class for neural generators.

Defines the interface that all neural generators (VAE, diffusion, VQ-VAE)
must implement. Provides common utilities for device management, metadata
building, and temperature adjustment based on error context.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ll_gen.conditioning.embeddings import ConditioningEmbeddings
from ll_gen.config import ErrorCategory
from ll_gen.proposals.base import BaseProposal

_log = logging.getLogger(__name__)


class BaseNeuralGenerator(ABC):
    """Abstract base class for all neural generators.

    Subclasses must implement:
    - generate(): Single proposal generation
    - generate_candidates(): Batch proposal generation

    Attributes:
        device: Target device ("cpu" or "cuda").
        checkpoint_path: Path to model checkpoint (optional).
        _model: Lazy-initialized neural network model.
    """

    def __init__(
        self,
        device: str = "cpu",
        checkpoint_path: Path | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            device: Target device ("cpu" or "cuda"). If "cuda" is not
                available, falls back to "cpu".
            checkpoint_path: Optional path to load model checkpoint from.
        """
        self.device: str = self._resolve_device(device)
        self.checkpoint_path: Path | None = (
            Path(checkpoint_path) if checkpoint_path else None
        )
        self._model: Any | None = None

    @abstractmethod
    def generate(
        self,
        prompt: str,
        conditioning: ConditioningEmbeddings | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> BaseProposal:
        """Generate a single proposal from a prompt.

        Args:
            prompt: User prompt describing the shape/object to generate.
            conditioning: Optional conditioning embeddings (text/image/multimodal).
            error_context: Optional error feedback from a prior failed attempt.
                Contains keys like "error_category", "previous_latent_vector",
                "failure_description", etc.

        Returns:
            A typed proposal (CommandSequenceProposal or LatentProposal).
        """
        pass

    @abstractmethod
    def generate_candidates(
        self,
        prompt: str,
        num_candidates: int = 3,
        conditioning: ConditioningEmbeddings | None = None,
    ) -> list[BaseProposal]:
        """Generate multiple candidate proposals from a prompt.

        Args:
            prompt: User prompt describing the shape/object to generate.
            num_candidates: Number of proposals to generate.
            conditioning: Optional conditioning embeddings.

        Returns:
            List of proposals, typically ordered by confidence descending.
        """
        pass

    def _resolve_device(self, device: str) -> str:
        """Resolve and validate the target device.

        Falls back to CPU if CUDA is unavailable.

        Args:
            device: Requested device ("cpu", "cuda", or "cuda:0").

        Returns:
            Validated device string.
        """
        if device.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    _log.info(f"Using device: {device}")
                    return device
            except ImportError:
                _log.warning("torch not available; using CPU")

        _log.info("CUDA not available; falling back to CPU")
        return "cpu"

    def _build_metadata(self, model_name: str, **kwargs: Any) -> dict[str, Any]:
        """Build standard metadata dict for generated proposals.

        Args:
            model_name: Name of the neural model (e.g., "STEPVAE", "StructuredDiffusion").
            **kwargs: Additional metadata key-value pairs.

        Returns:
            Metadata dict with model name, device, timestamp, and extra fields.
        """
        metadata: dict[str, Any] = {
            "model_name": model_name,
            "device": self.device,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        metadata.update(kwargs)
        return metadata

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint into self._model.

        Assumes self._model is already initialized. Loads the state dict
        and moves the model to the target device.

        Args:
            path: Path to checkpoint file.

        Raises:
            RuntimeError: If self._model is None or checkpoint cannot be loaded.
            ImportError: If torch is not available.
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required to load checkpoints; install with "
                "'conda install pytorch::pytorch -c conda-forge'"
            ) from None

        if self._model is None:
            raise RuntimeError(
                "self._model is None; subclass must initialize model before "
                "calling load_checkpoint()"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        _log.info(f"Loading checkpoint from {path}")
        state_dict = torch.load(path, map_location=self.device)
        self._model.load_state_dict(state_dict)
        self._model = self._model.to(self.device)
        _log.info("Checkpoint loaded successfully")

    def _adjust_temperature_for_error(
        self,
        base_temperature: float,
        error_context: dict[str, Any] | None,
    ) -> float:
        """Adjust sampling temperature based on error category.

        Lower temperature → more deterministic (repeat previously-working patterns).
        Higher temperature → more exploration (try different approaches).

        Args:
            base_temperature: Base temperature value.
            error_context: Error dict with optional "error_category" key.

        Returns:
            Adjusted temperature value.
        """
        if error_context is None:
            return base_temperature

        error_category_str = error_context.get("error_category")
        if error_category_str is None:
            return base_temperature

        try:
            error_category = ErrorCategory(error_category_str)
        except (ValueError, KeyError):
            _log.warning(f"Unknown error category: {error_category_str}")
            return base_temperature

        # Map error category to temperature adjustment
        if error_category == ErrorCategory.TOPOLOGY_ERROR:
            adjusted = base_temperature * 0.7
            _log.debug(
                f"Topology error detected; reducing temperature from "
                f"{base_temperature:.2f} to {adjusted:.2f}"
            )
            return adjusted
        elif error_category == ErrorCategory.DEGENERATE_SHAPE:
            adjusted = base_temperature * 1.3
            _log.debug(
                f"Degenerate shape detected; increasing temperature from "
                f"{base_temperature:.2f} to {adjusted:.2f}"
            )
            return adjusted
        elif error_category == ErrorCategory.SELF_INTERSECTION:
            adjusted = base_temperature * 0.8
            _log.debug(
                f"Self-intersection detected; reducing temperature from "
                f"{base_temperature:.2f} to {adjusted:.2f}"
            )
            return adjusted
        elif error_category == ErrorCategory.BOOLEAN_FAILURE:
            adjusted = base_temperature * 0.9
            _log.debug(
                f"Boolean failure detected; reducing temperature from "
                f"{base_temperature:.2f} to {adjusted:.2f}"
            )
            return adjusted
        else:
            # INVALID_PARAMS, TOLERANCE_VIOLATION
            return base_temperature
