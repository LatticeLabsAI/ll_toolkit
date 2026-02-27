"""Conditioning embeddings — unified representation for text, image, and multimodal inputs.

This module provides the ``ConditioningEmbeddings`` dataclass, which
encapsulates embeddings from various sources and provides methods for
conversion, validation, and metadata management.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


@dataclass
class ConditioningEmbeddings:
    """Unified conditioning embeddings from text, image, or multimodal sources.

    Attributes:
        token_embeddings: Optional (seq_len, embed_dim) array of per-token embeddings.
            None if pooled embedding only, or if source doesn't produce token-level
            representations.
        pooled_embedding: Optional (embed_dim,) single-vector summary of the input.
            Can be None if only token_embeddings is provided.
        source_type: Type of source — "text", "image", or "multimodal".
        source_model: Model identifier (e.g., "bert-base-uncased", "dino_vits16",
            "hash_fallback" for deterministic fallback).
        embed_dim: Embedding dimension. Must be consistent with array shapes.
        metadata: Additional metadata such as sequence length, image size, region
            coordinates, or language tags.
    """

    token_embeddings: np.ndarray | None = None
    pooled_embedding: np.ndarray | None = None
    source_type: str = "text"
    source_model: str = "unknown"
    embed_dim: int = 768
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tensor(self, device: str = "cpu") -> Any | None:
        """Convert pooled embedding to torch tensor.

        Args:
            device: Target device ("cpu" or "cuda").

        Returns:
            torch.Tensor of shape (embed_dim,) or None if pooled_embedding is None.
        """
        if self.pooled_embedding is None:
            return None

        try:
            import torch

            tensor = torch.from_numpy(self.pooled_embedding).float()
            if device.startswith("cuda") and torch.cuda.is_available():
                tensor = tensor.to(device)
            elif device == "cpu":
                tensor = tensor.to("cpu")
            return tensor
        except ImportError:
            _log.warning("torch not available; cannot convert to tensor")
            return None

    def to_token_tensor(self, device: str = "cpu") -> Any | None:
        """Convert token embeddings to torch tensor.

        Args:
            device: Target device ("cpu" or "cuda").

        Returns:
            torch.Tensor of shape (seq_len, embed_dim) or None if token_embeddings
            is None.
        """
        if self.token_embeddings is None:
            return None

        try:
            import torch

            tensor = torch.from_numpy(self.token_embeddings).float()
            if device.startswith("cuda") and torch.cuda.is_available():
                tensor = tensor.to(device)
            elif device == "cpu":
                tensor = tensor.to("cpu")
            return tensor
        except ImportError:
            _log.warning("torch not available; cannot convert to tensor")
            return None

    @classmethod
    def from_tensor(
        cls,
        tensor: Any,
        source_type: str,
        source_model: str,
        token_tensor: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConditioningEmbeddings:
        """Create ConditioningEmbeddings from torch tensor(s).

        Args:
            tensor: torch.Tensor of shape (embed_dim,) for pooled embedding.
            source_type: Type of source ("text", "image", or "multimodal").
            source_model: Model identifier.
            token_tensor: Optional torch.Tensor of shape (seq_len, embed_dim).
            metadata: Optional metadata dictionary.

        Returns:
            ConditioningEmbeddings instance.
        """
        try:
            import torch

            pooled = None
            if tensor is not None:
                if isinstance(tensor, torch.Tensor):
                    pooled = tensor.detach().cpu().numpy()
                else:
                    pooled = np.array(tensor)

            token_emb = None
            if token_tensor is not None:
                if isinstance(token_tensor, torch.Tensor):
                    token_emb = token_tensor.detach().cpu().numpy()
                else:
                    token_emb = np.array(token_tensor)

            embed_dim = pooled.shape[0] if pooled is not None else (
                token_emb.shape[1] if token_emb is not None else 768
            )

            return cls(
                token_embeddings=token_emb,
                pooled_embedding=pooled,
                source_type=source_type,
                source_model=source_model,
                embed_dim=embed_dim,
                metadata=metadata or {},
            )
        except ImportError:
            _log.warning("torch not available; cannot import from tensor")
            return cls(
                source_type=source_type,
                source_model=source_model,
                metadata=metadata or {},
            )

    def validate(self) -> bool:
        """Validate consistency of embeddings and metadata.

        Returns:
            True if embeddings are valid and consistent, False otherwise.
        """
        # Check pooled embedding
        if self.pooled_embedding is not None:
            if self.pooled_embedding.ndim != 1:
                _log.warning(
                    f"Validation failed: pooled_embedding has wrong shape: "
                    f"{self.pooled_embedding.shape}, expected 1D array"
                )
                return False
            if self.pooled_embedding.shape[0] != self.embed_dim:
                _log.warning(
                    f"Validation failed: pooled_embedding shape "
                    f"{self.pooled_embedding.shape[0]} does not match embed_dim "
                    f"{self.embed_dim}"
                )
                return False

        # Check token embeddings
        if self.token_embeddings is not None:
            if self.token_embeddings.ndim != 2:
                _log.warning(
                    f"Validation failed: token_embeddings has wrong shape: "
                    f"{self.token_embeddings.shape}, expected 2D array"
                )
                return False
            if self.token_embeddings.shape[1] != self.embed_dim:
                _log.warning(
                    f"Validation failed: token_embeddings shape[1] "
                    f"{self.token_embeddings.shape[1]} does not match embed_dim "
                    f"{self.embed_dim}"
                )
                return False

        # At least one should be present
        if self.pooled_embedding is None and self.token_embeddings is None:
            _log.warning("Neither pooled nor token embeddings are present")
            return False

        return True

    def summary(self) -> dict[str, Any]:
        """Compact summary of the embeddings.

        Returns:
            Dictionary with keys:
                - source_type
                - source_model
                - embed_dim
                - has_pooled_embedding
                - has_token_embeddings
                - token_seq_len (if available)
                - metadata
        """
        return {
            "source_type": self.source_type,
            "source_model": self.source_model,
            "embed_dim": self.embed_dim,
            "has_pooled_embedding": self.pooled_embedding is not None,
            "has_token_embeddings": self.token_embeddings is not None,
            "token_seq_len": (
                self.token_embeddings.shape[0]
                if self.token_embeddings is not None
                else None
            ),
            "metadata": self.metadata,
        }
