"""
Cross-attention conditioning for text- and image-guided CAD generation.

Following Text2CAD, this module provides conditioners that inject semantic
information from frozen pretrained encoders (BERT/CLIP for text,
DINOv2/CLIP for images) into the CAD decoder via cross-attention.

All heavy dependencies (transformers, torchvision) are lazily imported
to avoid import-time failures when they are not installed.

Classes:
    - TextConditioner: frozen text encoder + adaptive layer + cross-attn.
    - ImageConditioner: frozen vision encoder + projection + cross-attn.
    - MultiModalConditioner: fuses text and image conditioning.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

_log = logging.getLogger(__name__)


def _try_import_transformers():
    """Lazily import the transformers library.

    Returns:
        The transformers module, or None if unavailable.
    """
    try:
        import transformers
        return transformers
    except ImportError:
        _log.warning(
            "The 'transformers' library is not installed. "
            "Text/image conditioners will not be functional."
        )
        return None


class AdaptiveLayer(nn.Module):
    """Single transformer block with cross-attention for conditioning injection.

    Contains a self-attention sub-layer, a cross-attention sub-layer
    (attending to conditioning embeddings), and a feed-forward sub-layer.

    Args:
        hidden_dim: Model hidden dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Cross-attention (query=hidden, key/value=conditioning)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with self-attention, cross-attention, and FFN.

        Args:
            hidden_states: [B, S, D] decoder hidden states.
            conditioning: [B, C, D] conditioning embeddings.
            attention_mask: Optional mask for the conditioning sequence.

        Returns:
            Updated hidden states [B, S, D].
        """
        # Self-attention
        residual = hidden_states
        h, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
        hidden_states = self.norm1(residual + h)

        # Cross-attention
        residual = hidden_states
        h, _ = self.cross_attn(
            hidden_states,
            conditioning,
            conditioning,
            key_padding_mask=attention_mask,
        )
        hidden_states = self.norm2(residual + h)

        # Feed-forward
        residual = hidden_states
        hidden_states = self.norm3(residual + self.ffn(hidden_states))

        return hidden_states


class TextConditioner(nn.Module):
    """Condition CAD generation on natural-language descriptions.

    Wraps a frozen BERT or CLIP text encoder, projects its hidden states
    to the decoder dimension, and applies AdaptiveLayer blocks for
    cross-attention injection.

    Following Text2CAD, the first ``skip_cross_attention_blocks`` decoder
    blocks skip cross-attention to allow initial CAD structure formation
    before conditioning kicks in.  When ``block_index`` is passed to
    :meth:`forward`, the cross-attention layers are only applied when
    ``block_index >= skip_cross_attention_blocks``.

    Args:
        encoder_name: Hugging Face model identifier (e.g.
            "bert-base-uncased").
        conditioning_dim: Dimension of the conditioning embeddings
            (must match the decoder hidden dim).
        freeze_encoder: Whether to freeze the pretrained encoder weights.
        num_adaptive_layers: Number of AdaptiveLayer blocks.
        skip_cross_attention_blocks: Number of initial decoder blocks
            that skip cross-attention (Text2CAD default = 2).
    """

    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        conditioning_dim: int = 1024,
        freeze_encoder: bool = True,
        num_adaptive_layers: int = 1,
        skip_cross_attention_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.conditioning_dim = conditioning_dim
        self.freeze_encoder = freeze_encoder
        self.skip_cross_attention_blocks = skip_cross_attention_blocks

        # Lazy-load the pretrained encoder
        self._encoder: Optional[nn.Module] = None
        self._tokenizer = None
        self._encoder_hidden_dim: Optional[int] = None

        # Projection from encoder hidden dim to conditioning dim (built on first use)
        self._projection: Optional[nn.Linear] = None

        # Adaptive cross-attention layers
        self.adaptive_layers = nn.ModuleList(
            [
                AdaptiveLayer(
                    hidden_dim=conditioning_dim, num_heads=8, dropout=0.1
                )
                for _ in range(num_adaptive_layers)
            ]
        )

        _log.info(
            "TextConditioner created: encoder=%s, cond_dim=%d, frozen=%s, "
            "adaptive_layers=%d, skip_blocks=%d",
            encoder_name, conditioning_dim, freeze_encoder,
            num_adaptive_layers, skip_cross_attention_blocks,
        )

    def _load_encoder(self) -> None:
        """Lazily load and configure the pretrained text encoder."""
        transformers = _try_import_transformers()
        if transformers is None:
            raise RuntimeError(
                "Cannot load text encoder: 'transformers' is not installed."
            )

        self._encoder = transformers.AutoModel.from_pretrained(self.encoder_name)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.encoder_name
        )

        # Determine encoder hidden dim
        if hasattr(self._encoder.config, "hidden_size"):
            self._encoder_hidden_dim = self._encoder.config.hidden_size
        else:
            self._encoder_hidden_dim = 768  # fallback

        # Build projection
        self._projection = nn.Linear(
            self._encoder_hidden_dim, self.conditioning_dim
        )
        # Move to same device as adaptive layers
        device = next(self.adaptive_layers.parameters()).device
        self._projection = self._projection.to(device)
        self._encoder = self._encoder.to(device)

        if self.freeze_encoder:
            for param in self._encoder.parameters():
                param.requires_grad = False
            self._encoder.eval()

        _log.info(
            "Text encoder loaded: %s (hidden_dim=%d, frozen=%s)",
            self.encoder_name, self._encoder_hidden_dim, self.freeze_encoder,
        )

    @property
    def encoder(self) -> nn.Module:
        """Access the pretrained encoder, loading it on first use."""
        if self._encoder is None:
            self._load_encoder()
        return self._encoder

    @property
    def projection(self) -> nn.Linear:
        """Access the projection layer, loading encoder if needed."""
        if self._projection is None:
            self._load_encoder()
        return self._projection

    @property
    def tokenizer(self):
        """Access the tokenizer, loading encoder if needed."""
        if self._tokenizer is None:
            self._load_encoder()
        return self._tokenizer

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode tokenised text into conditioning embeddings.

        Args:
            input_ids: [B, L] token ids from the text tokenizer.
            attention_mask: [B, L] padding mask.

        Returns:
            Conditioning embeddings [B, L, conditioning_dim].
        """
        encoder = self.encoder
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )
        else:
            outputs = encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

        hidden = outputs.last_hidden_state  # [B, L, encoder_hidden_dim]
        return self.projection(hidden)  # [B, L, conditioning_dim]

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        block_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Condition decoder hidden states on text.

        Following Text2CAD, early decoder blocks can skip cross-attention
        to let the initial CAD structure form before conditioning kicks in.
        When ``block_index`` is provided and is less than
        ``self.skip_cross_attention_blocks``, the hidden states are returned
        unchanged (no cross-attention applied).

        Args:
            hidden_states: [B, S, D] decoder hidden states.
            text_input_ids: [B, L] text token ids.
            text_attention_mask: [B, L] text padding mask.
            block_index: Optional zero-based decoder block index.  When
                provided and < ``skip_cross_attention_blocks``, the
                cross-attention layers are bypassed entirely.

        Returns:
            Conditioned hidden states [B, S, D].
        """
        # Text2CAD skip: first N blocks get no cross-attention
        if (
            block_index is not None
            and block_index < self.skip_cross_attention_blocks
        ):
            _log.debug(
                "Skipping cross-attention at block %d (< %d)",
                block_index,
                self.skip_cross_attention_blocks,
            )
            return hidden_states

        cond = self.encode_text(text_input_ids, text_attention_mask)

        # Invert mask for key_padding_mask (True = ignore)
        key_mask = None
        if text_attention_mask is not None:
            key_mask = text_attention_mask == 0

        for layer in self.adaptive_layers:
            hidden_states = layer(hidden_states, cond, attention_mask=key_mask)

        return hidden_states


class ImageConditioner(nn.Module):
    """Condition CAD generation on rendered images.

    Wraps a frozen DINOv2 or CLIP vision encoder, projects patch
    embeddings to the decoder dimension, and applies AdaptiveLayer
    blocks for cross-attention injection.

    Args:
        encoder_name: Hugging Face model identifier (e.g.
            "facebook/dinov2-base").
        conditioning_dim: Dimension of the conditioning embeddings.
        freeze_encoder: Whether to freeze the vision encoder weights.
        num_adaptive_layers: Number of AdaptiveLayer blocks.
        skip_cross_attention_blocks: Number of initial decoder blocks
            that skip cross-attention (Text2CAD default = 2).
    """

    def __init__(
        self,
        encoder_name: str = "facebook/dinov2-base",
        conditioning_dim: int = 1024,
        freeze_encoder: bool = True,
        num_adaptive_layers: int = 1,
        skip_cross_attention_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.conditioning_dim = conditioning_dim
        self.freeze_encoder = freeze_encoder
        self.skip_cross_attention_blocks = skip_cross_attention_blocks

        self._encoder: Optional[nn.Module] = None
        self._processor = None
        self._encoder_hidden_dim: Optional[int] = None
        self._projection: Optional[nn.Linear] = None

        self.adaptive_layers = nn.ModuleList(
            [
                AdaptiveLayer(
                    hidden_dim=conditioning_dim, num_heads=8, dropout=0.1
                )
                for _ in range(num_adaptive_layers)
            ]
        )

        _log.info(
            "ImageConditioner created: encoder=%s, cond_dim=%d, frozen=%s",
            encoder_name, conditioning_dim, freeze_encoder,
        )

    def _load_encoder(self) -> None:
        """Lazily load and configure the pretrained vision encoder."""
        transformers = _try_import_transformers()
        if transformers is None:
            raise RuntimeError(
                "Cannot load image encoder: 'transformers' is not installed."
            )

        self._encoder = transformers.AutoModel.from_pretrained(self.encoder_name)
        self._processor = transformers.AutoImageProcessor.from_pretrained(
            self.encoder_name
        )

        if hasattr(self._encoder.config, "hidden_size"):
            self._encoder_hidden_dim = self._encoder.config.hidden_size
        else:
            self._encoder_hidden_dim = 768

        self._projection = nn.Linear(
            self._encoder_hidden_dim, self.conditioning_dim
        )
        device = next(self.adaptive_layers.parameters()).device
        self._projection = self._projection.to(device)
        self._encoder = self._encoder.to(device)

        if self.freeze_encoder:
            for param in self._encoder.parameters():
                param.requires_grad = False
            self._encoder.eval()

        _log.info(
            "Image encoder loaded: %s (hidden_dim=%d, frozen=%s)",
            self.encoder_name, self._encoder_hidden_dim, self.freeze_encoder,
        )

    @property
    def encoder(self) -> nn.Module:
        """Access the pretrained vision encoder, loading on first use."""
        if self._encoder is None:
            self._load_encoder()
        return self._encoder

    @property
    def projection(self) -> nn.Linear:
        """Access the projection layer, loading encoder if needed."""
        if self._projection is None:
            self._load_encoder()
        return self._projection

    @property
    def processor(self):
        """Access the image processor, loading encoder if needed."""
        if self._processor is None:
            self._load_encoder()
        return self._processor

    def encode_image(
        self, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """Encode pixel values into conditioning embeddings.

        Args:
            pixel_values: [B, C, H, W] preprocessed image tensors.

        Returns:
            Conditioning embeddings [B, N, conditioning_dim] where N is
            the number of patch tokens.
        """
        encoder = self.encoder
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = encoder(pixel_values=pixel_values)
        else:
            outputs = encoder(pixel_values=pixel_values)

        # Use last_hidden_state (includes CLS + patch tokens)
        hidden = outputs.last_hidden_state  # [B, N, encoder_hidden_dim]
        return self.projection(hidden)  # [B, N, conditioning_dim]

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_values: torch.Tensor,
        block_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Condition decoder hidden states on image features.

        Early decoder blocks skip cross-attention (same logic as
        :class:`TextConditioner`).

        Args:
            hidden_states: [B, S, D] decoder hidden states.
            pixel_values: [B, C, H, W] preprocessed image tensors.
            block_index: Optional zero-based decoder block index.

        Returns:
            Conditioned hidden states [B, S, D].
        """
        if (
            block_index is not None
            and block_index < self.skip_cross_attention_blocks
        ):
            return hidden_states

        cond = self.encode_image(pixel_values)

        for layer in self.adaptive_layers:
            hidden_states = layer(hidden_states, cond)

        return hidden_states


class MultiModalConditioner(nn.Module):
    """Fuses text and image conditioning for CAD generation.

    Combines TextConditioner and ImageConditioner by concatenating
    their conditioning embeddings along the sequence dimension before
    passing through shared AdaptiveLayer blocks.

    Args:
        text_encoder_name: Hugging Face text model identifier.
        image_encoder_name: Hugging Face vision model identifier.
        conditioning_dim: Shared conditioning dimension.
        freeze_encoders: Whether to freeze both pretrained encoders.
        num_adaptive_layers: Number of shared AdaptiveLayer blocks.
    """

    def __init__(
        self,
        text_encoder_name: str = "bert-base-uncased",
        image_encoder_name: str = "facebook/dinov2-base",
        conditioning_dim: int = 1024,
        freeze_encoders: bool = True,
        num_adaptive_layers: int = 1,
        skip_cross_attention_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.conditioning_dim = conditioning_dim
        self.skip_cross_attention_blocks = skip_cross_attention_blocks

        self.text_conditioner = TextConditioner(
            encoder_name=text_encoder_name,
            conditioning_dim=conditioning_dim,
            freeze_encoder=freeze_encoders,
            num_adaptive_layers=0,  # We use shared layers below
            skip_cross_attention_blocks=skip_cross_attention_blocks,
        )

        self.image_conditioner = ImageConditioner(
            encoder_name=image_encoder_name,
            conditioning_dim=conditioning_dim,
            freeze_encoder=freeze_encoders,
            num_adaptive_layers=0,  # We use shared layers below
            skip_cross_attention_blocks=skip_cross_attention_blocks,
        )

        # Shared adaptive layers for fused conditioning
        self.adaptive_layers = nn.ModuleList(
            [
                AdaptiveLayer(
                    hidden_dim=conditioning_dim, num_heads=8, dropout=0.1
                )
                for _ in range(num_adaptive_layers)
            ]
        )

        # Modality type embeddings
        self.text_type_embedding = nn.Parameter(
            torch.zeros(1, 1, conditioning_dim)
        )
        self.image_type_embedding = nn.Parameter(
            torch.zeros(1, 1, conditioning_dim)
        )
        nn.init.normal_(self.text_type_embedding, std=0.02)
        nn.init.normal_(self.image_type_embedding, std=0.02)

        _log.info(
            "MultiModalConditioner created: text=%s, image=%s, cond_dim=%d",
            text_encoder_name, image_encoder_name, conditioning_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        block_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Condition decoder hidden states on text and/or image features.

        At least one of text_input_ids or pixel_values must be provided.
        Early decoder blocks skip cross-attention per Text2CAD design.

        Args:
            hidden_states: [B, S, D] decoder hidden states.
            text_input_ids: [B, L_text] text token ids (optional).
            text_attention_mask: [B, L_text] text padding mask.
            pixel_values: [B, C, H, W] image tensors (optional).
            block_index: Optional zero-based decoder block index.

        Returns:
            Conditioned hidden states [B, S, D].
        """
        # Text2CAD skip: first N blocks bypass cross-attention
        if (
            block_index is not None
            and block_index < self.skip_cross_attention_blocks
        ):
            return hidden_states

        conditioning_parts: List[torch.Tensor] = []

        if text_input_ids is not None:
            text_cond = self.text_conditioner.encode_text(
                text_input_ids, text_attention_mask
            )
            text_cond = text_cond + self.text_type_embedding
            conditioning_parts.append(text_cond)

        if pixel_values is not None:
            image_cond = self.image_conditioner.encode_image(pixel_values)
            image_cond = image_cond + self.image_type_embedding
            conditioning_parts.append(image_cond)

        if not conditioning_parts:
            raise ValueError(
                "At least one of text_input_ids or pixel_values must be provided."
            )

        # Concatenate conditioning along sequence dimension
        conditioning = torch.cat(conditioning_parts, dim=1)  # [B, L_total, D]

        for layer in self.adaptive_layers:
            hidden_states = layer(hidden_states, conditioning)

        return hidden_states
