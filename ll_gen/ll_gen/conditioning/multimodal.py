"""Multimodal conditioning — fusion of text and image embeddings.

Combines text and image embeddings using configurable fusion strategies
(concatenation, averaging, text-only, image-only).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ll_gen.conditioning.embeddings import ConditioningEmbeddings
from ll_gen.conditioning.image_encoder import ImageConditioningEncoder
from ll_gen.conditioning.text_encoder import TextConditioningEncoder

_log = logging.getLogger(__name__)


class MultiModalConditioner:
    """Fuses text and image embeddings for multimodal conditioning.

    Attributes:
        text_encoder: TextConditioningEncoder instance.
        image_encoder: ImageConditioningEncoder instance.
        conditioning_dim: Embedding dimension.
        fusion_method: Strategy for combining embeddings.
        device: Torch device.
    """

    def __init__(
        self,
        text_model: str = "bert-base-uncased",
        image_model: str = "dino_vits16",
        conditioning_dim: int = 768,
        fusion_method: str = "concat",
        device: str = "cpu",
    ) -> None:
        """Initialize MultiModalConditioner.

        Args:
            text_model: Text model identifier.
            image_model: Image model identifier.
            conditioning_dim: Base embedding dimension.
            fusion_method: Fusion strategy ("concat", "average", "text_only",
                "image_only").
            device: Torch device ("cpu" or "cuda:*").
        """
        self.text_encoder = TextConditioningEncoder(
            model_name=text_model,
            conditioning_dim=conditioning_dim,
            device=device,
        )
        self.image_encoder = ImageConditioningEncoder(
            model_name=image_model,
            conditioning_dim=conditioning_dim,
            device=device,
        )
        self.conditioning_dim = conditioning_dim
        self.fusion_method = fusion_method
        self.device = device

        if fusion_method not in ("concat", "average", "text_only", "image_only"):
            raise ValueError(
                f"fusion_method must be one of ('concat', 'average', "
                f"'text_only', 'image_only'), got {fusion_method}"
            )

    def encode(
        self,
        prompt: str,
        image_path: str | Path | None = None,
    ) -> ConditioningEmbeddings:
        """Encode text and optional image into fused embeddings.

        Args:
            prompt: Text prompt.
            image_path: Optional path to image file.

        Returns:
            ConditioningEmbeddings with fused embeddings.
        """
        # Encode text
        text_emb = self.text_encoder.encode(prompt)

        # If no image, return text embedding
        if image_path is None:
            return text_emb

        # Encode image
        image_emb = self.image_encoder.encode(image_path)

        # Fuse based on method
        if self.fusion_method == "text_only":
            result_pooled = text_emb.pooled_embedding
            result_tokens = text_emb.token_embeddings
        elif self.fusion_method == "image_only":
            result_pooled = image_emb.pooled_embedding
            result_tokens = image_emb.token_embeddings
        elif self.fusion_method == "average":
            # Element-wise mean of pooled embeddings
            if (
                text_emb.pooled_embedding is not None
                and image_emb.pooled_embedding is not None
            ):
                result_pooled = (
                    text_emb.pooled_embedding + image_emb.pooled_embedding
                ) / 2.0
            elif text_emb.pooled_embedding is not None:
                result_pooled = text_emb.pooled_embedding
            else:
                result_pooled = image_emb.pooled_embedding

            # Stack token embeddings if both available
            if (
                text_emb.token_embeddings is not None
                and image_emb.token_embeddings is not None
            ):
                # Concatenate along sequence dimension, then average
                text_tokens = text_emb.token_embeddings
                image_tokens = image_emb.token_embeddings
                # Pad to same sequence length
                max_len = max(text_tokens.shape[0], image_tokens.shape[0])
                if text_tokens.shape[0] < max_len:
                    text_tokens = np.pad(
                        text_tokens,
                        ((0, max_len - text_tokens.shape[0]), (0, 0)),
                        mode="constant",
                    )
                if image_tokens.shape[0] < max_len:
                    image_tokens = np.pad(
                        image_tokens,
                        ((0, max_len - image_tokens.shape[0]), (0, 0)),
                        mode="constant",
                    )
                result_tokens = (text_tokens + image_tokens) / 2.0
            elif text_emb.token_embeddings is not None:
                result_tokens = text_emb.token_embeddings
            else:
                result_tokens = image_emb.token_embeddings
        else:  # concat
            # Concatenate pooled embeddings
            if (
                text_emb.pooled_embedding is not None
                and image_emb.pooled_embedding is not None
            ):
                result_pooled = np.concatenate(
                    [text_emb.pooled_embedding, image_emb.pooled_embedding],
                    axis=0,
                ).astype(np.float32)
            elif text_emb.pooled_embedding is not None:
                result_pooled = text_emb.pooled_embedding
            else:
                result_pooled = image_emb.pooled_embedding

            # Concatenate token embeddings if both available
            if (
                text_emb.token_embeddings is not None
                and image_emb.token_embeddings is not None
            ):
                result_tokens = np.concatenate(
                    [text_emb.token_embeddings, image_emb.token_embeddings],
                    axis=0,
                ).astype(np.float32)
            elif text_emb.token_embeddings is not None:
                result_tokens = text_emb.token_embeddings
            else:
                result_tokens = image_emb.token_embeddings

        # Compute final embedding dimension
        final_embed_dim = (
            result_pooled.shape[0] if result_pooled is not None else self.conditioning_dim
        )

        return ConditioningEmbeddings(
            token_embeddings=result_tokens,
            pooled_embedding=result_pooled,
            source_type="multimodal",
            source_model=f"{self.text_encoder.model_name}+{self.image_encoder.model_name}",
            embed_dim=final_embed_dim,
            metadata={
                "prompt": prompt,
                "image_path": str(image_path) if image_path else None,
                "fusion_method": self.fusion_method,
                "text_source": text_emb.source_model,
                "image_source": image_emb.source_model,
            },
        )

    def encode_batch(
        self,
        prompts: list[str],
        image_paths: list[str | Path | None] | None = None,
    ) -> list[ConditioningEmbeddings]:
        """Encode multiple text-image pairs.

        Args:
            prompts: List of text prompts.
            image_paths: Optional list of image paths (aligned with prompts).
                If None, encodes text-only.

        Returns:
            List of ConditioningEmbeddings.
        """
        if image_paths is None:
            image_paths = [None] * len(prompts)
        elif len(image_paths) != len(prompts):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) does not match "
                f"number of image_paths ({len(image_paths)})"
            )

        results = []
        for prompt, image_path in zip(prompts, image_paths):
            try:
                result = self.encode(prompt, image_path)
                results.append(result)
            except Exception as e:
                _log.error(f"Error encoding pair (prompt={prompt[:50]}, image={image_path}): {e}")
                raise

        return results
