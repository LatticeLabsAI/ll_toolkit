"""
CLIP vision encoder with SDPA attention for LL-OCADR.

A faithful CLIP ViT image encoder (patch embedding + class token + learned
positional embeddings + pre-LayerNorm transformer) that uses PyTorch's
``scaled_dot_product_attention`` (SDPA) for the self-attention, matching the
DeepSeek-OCR "clip_sdpa" variant. It provides the *global / semantic* branch of
the rendered-image vision tower (the SAM branch in ``sam_vary_sdpa.py`` provides
the high-resolution branch).

Input:  pixel_values  ``[B, 3, H, W]``
Output: last_hidden_state ``[B, 1 + (H/patch)*(W/patch), embed_dim]``
        (index 0 is the class token).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CLIPVisionConfig:
    """Configuration for :class:`CLIPVisionSDPA`."""

    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5


class CLIPAttentionSDPA(nn.Module):
    """Multi-head self-attention using scaled_dot_product_attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, n, _ = hidden_states.shape

        def _shape(x: torch.Tensor) -> torch.Tensor:
            # [B, N, C] -> [B, heads, N, head_dim]
            return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        q = _shape(self.q_proj(hidden_states))
        k = _shape(self.k_proj(hidden_states))
        v = _shape(self.v_proj(hidden_states))

        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attn = attn.transpose(1, 2).contiguous().view(b, n, self.embed_dim)
        return self.out_proj(attn)


class CLIPMLP(nn.Module):
    """Feed-forward block (GELU)."""

    def __init__(self, embed_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CLIPEncoderLayerSDPA(nn.Module):
    """Pre-LayerNorm transformer encoder layer (CLIP style)."""

    def __init__(self, cfg: CLIPVisionConfig) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        self.self_attn = CLIPAttentionSDPA(cfg.embed_dim, cfg.num_heads, cfg.dropout)
        self.layer_norm2 = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        self.mlp = CLIPMLP(cfg.embed_dim, cfg.mlp_ratio)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states)
        )
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class CLIPVisionSDPA(nn.Module):
    """CLIP vision transformer (SDPA attention).

    Produces a sequence of patch embeddings (with a leading class token) from an
    image, used as the global/semantic branch of the LL-OCADR vision tower.
    """

    def __init__(self, cfg: CLIPVisionConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or CLIPVisionConfig()
        self.cfg = cfg

        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.num_positions = self.num_patches + 1  # + class token

        self.patch_embedding = nn.Conv2d(
            cfg.num_channels,
            cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=False,
        )
        self.class_embedding = nn.Parameter(torch.randn(cfg.embed_dim))
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_positions, cfg.embed_dim)
        )
        self.pre_layrnorm = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        self.layers = nn.ModuleList(
            [CLIPEncoderLayerSDPA(cfg) for _ in range(cfg.depth)]
        )
        self.post_layernorm = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)

    @property
    def embed_dim(self) -> int:
        return self.cfg.embed_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images.

        Args:
            pixel_values: ``[B, 3, H, W]`` image tensor.

        Returns:
            ``[B, 1 + num_patches, embed_dim]`` hidden states (index 0 = class
            token). When the input resolution differs from the configured
            ``image_size``, the learned positional embeddings are bicubically
            interpolated to match.
        """
        b = pixel_values.shape[0]

        patches = self.patch_embedding(pixel_values)  # [B, C, gh, gw]
        gh, gw = patches.shape[-2], patches.shape[-1]
        patches = patches.flatten(2).transpose(1, 2)  # [B, gh*gw, C]

        class_tokens = self.class_embedding.expand(b, 1, -1)
        hidden_states = torch.cat([class_tokens, patches], dim=1)
        hidden_states = hidden_states + self._interpolate_pos_embed(gh, gw)

        hidden_states = self.pre_layrnorm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.post_layernorm(hidden_states)

    def _interpolate_pos_embed(self, gh: int, gw: int) -> torch.Tensor:
        """Return positional embeddings matching a ``gh x gw`` patch grid."""
        num_patches = gh * gw
        if num_patches == self.num_patches:
            return self.position_embedding

        class_pos = self.position_embedding[:, :1]
        patch_pos = self.position_embedding[:, 1:]
        orig = int(self.num_patches**0.5)
        patch_pos = patch_pos.reshape(1, orig, orig, self.cfg.embed_dim).permute(
            0, 3, 1, 2
        )
        patch_pos = F.interpolate(
            patch_pos, size=(gh, gw), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, num_patches, -1)
        return torch.cat([class_pos, patch_pos], dim=1)


def build_clip_sdpa(
    image_size: int = 224,
    patch_size: int = 14,
    embed_dim: int = 1024,
    depth: int = 24,
    num_heads: int = 16,
) -> CLIPVisionSDPA:
    """Factory for a :class:`CLIPVisionSDPA` encoder (mirrors build_* siblings)."""
    return CLIPVisionSDPA(
        CLIPVisionConfig(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
    )
