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

import numpy as np

from ll_gen.conditioning.embeddings import ConditioningEmbeddings
from ll_gen.config import ErrorCategory
from ll_gen.proposals.base import BaseProposal

_log = logging.getLogger(__name__)

# Token ID constants (must match geotoken vocabulary)
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SOL_TOKEN_ID = 6
LINE_TOKEN_ID = 7
ARC_TOKEN_ID = 8
CIRCLE_TOKEN_ID = 9
EXTRUDE_TOKEN_ID = 10
EOS_CMD_TOKEN_ID = 11
PARAM_OFFSET = 12

# Parameter masks: for each command type, which of the 16 parameter slots are active
PARAMETER_MASKS = {
    0: [True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False],  # SOL: 2 params (x, y)
    1: [True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False],  # LINE: 4 params (x1, y1, x2, y2)
    2: [True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False],  # ARC: 6 params (x1, y1, x2, y2, cx, cy)
    3: [True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False],  # CIRCLE: 3 params (cx, cy, r)
    4: [True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False],  # EXTRUDE: 8 params (depth, operation_type, etc.)
    5: [False] * 16,  # EOS: no parameters
}

# Command type to token ID mapping
CMD_TOKEN_MAP = {
    0: SOL_TOKEN_ID,
    1: LINE_TOKEN_ID,
    2: ARC_TOKEN_ID,
    3: CIRCLE_TOKEN_ID,
    4: EXTRUDE_TOKEN_ID,
    5: EOS_CMD_TOKEN_ID,
}


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
        try:
            state_dict = torch.load(
                path, map_location=self.device, weights_only=True
            )
        except TypeError:
            # PyTorch < 2.0 does not support weights_only
            state_dict = torch.load(path, map_location=self.device)
        self._model.load_state_dict(state_dict)
        self._model = self._model.to(self.device)
        _log.info("Checkpoint loaded successfully")

    def _logits_to_token_ids(
        self,
        command_logits: Any,
        param_logits: Any,
        max_seq_len: int = 60,
    ) -> list[int]:
        """Convert raw logits to token IDs.

        Args:
            command_logits: Tensor of shape (batch, seq_len, num_commands).
            param_logits: List or dict of parameter logit tensors.
            max_seq_len: Maximum sequence length to decode.

        Returns:
            List of token IDs following geotoken vocabulary.
        """
        if command_logits is None or param_logits is None:
            return []

        try:
            import torch
        except ImportError:
            _log.warning("torch not available; cannot process logits")
            return []

        token_ids: list[int] = [BOS_TOKEN_ID]

        # Convert numpy arrays to tensors if needed
        if isinstance(command_logits, np.ndarray):
            command_logits = torch.from_numpy(command_logits)

        # Handle batch dimension
        if command_logits.shape[0] > 0:
            command_logits = command_logits[0]  # Take first sample

        seq_len = min(command_logits.shape[0], max_seq_len)

        for pos in range(seq_len):
            # Predict command type
            cmd_logits = command_logits[pos]  # Shape: (num_commands,)
            cmd_type = int(torch.argmax(cmd_logits).item())

            if cmd_type not in CMD_TOKEN_MAP:
                break

            cmd_token = CMD_TOKEN_MAP[cmd_type]
            token_ids.append(cmd_token)

            # Stop at EOS
            if cmd_token == EOS_CMD_TOKEN_ID or cmd_token == EOS_TOKEN_ID:
                break

            # Extract parameters for this command
            if isinstance(param_logits, dict):
                for param_idx in range(16):
                    if param_idx not in param_logits:
                        continue
                    if not PARAMETER_MASKS[cmd_type][param_idx]:
                        continue

                    param_tensor = param_logits[param_idx]
                    if isinstance(param_tensor, np.ndarray):
                        param_tensor = torch.from_numpy(param_tensor)
                    if param_tensor.shape[0] > 0:
                        param_tensor = param_tensor[0]
                    if len(param_tensor.shape) > 1:
                        param_tensor = param_tensor[pos]

                    param_val = int(torch.argmax(param_tensor).item())
                    token_ids.append(PARAM_OFFSET + param_val)

            elif isinstance(param_logits, (list, tuple)):
                for param_idx, param_tensor in enumerate(param_logits):
                    if param_idx >= 16:
                        break
                    if not PARAMETER_MASKS[cmd_type][param_idx]:
                        continue

                    if isinstance(param_tensor, np.ndarray):
                        param_tensor = torch.from_numpy(param_tensor)
                    if param_tensor.shape[0] > 0:
                        param_tensor = param_tensor[0]
                    if len(param_tensor.shape) > 1:
                        param_tensor = param_tensor[pos]

                    param_val = int(torch.argmax(param_tensor).item())
                    token_ids.append(PARAM_OFFSET + param_val)

        if token_ids[-1] != EOS_TOKEN_ID:
            token_ids.append(EOS_TOKEN_ID)

        return token_ids

    def _compute_confidence(
        self,
        command_logits: Any | None,
        param_logits: Any | None,
    ) -> float:
        """Compute confidence from prediction entropy.

        Args:
            command_logits: Command logits tensor.
            param_logits: Parameter logits (dict or list).

        Returns:
            Confidence score in [0, 1].
        """
        if command_logits is None:
            return 0.5

        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            _log.warning("torch not available; returning default confidence")
            return 0.5

        # Convert numpy to tensor if needed
        if isinstance(command_logits, np.ndarray):
            command_logits = torch.from_numpy(command_logits)

        if command_logits.shape[0] > 0:
            command_logits = command_logits[0]

        # Compute entropy of command predictions
        probs = functional.softmax(command_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        mean_entropy = entropy.mean().item()

        # Normalize entropy to [0, 1]
        max_entropy = float(np.log(command_logits.shape[-1]))
        normalized_entropy = min(mean_entropy / max_entropy, 1.0)

        confidence = 1.0 - normalized_entropy
        return float(confidence)

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
