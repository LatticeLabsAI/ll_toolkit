"""
SAM ("Vary") high-resolution vision encoder with SDPA attention for LL-OCADR.

A ViTDet/SAM-style image encoder — patch embedding, absolute positional
embeddings, a stack of transformer blocks that mostly use *windowed* attention
with a few *global*-attention blocks, and a convolutional neck — using PyTorch's
``scaled_dot_product_attention`` (SDPA). This is the "Vary" high-resolution
branch of the LL-OCADR rendered-image vision tower (the CLIP branch in
``clip_sdpa.py`` provides the global/semantic branch), mirroring DeepSeek-OCR's
SAM-with-SDPA encoder.

Input:  pixel_values ``[B, 3, H, W]``
Output: feature map ``[B, out_chans, H/patch, W/patch]``
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAMVaryConfig:
    """Configuration for :class:`SAMVaryViTSDPA` (SAM ViT-B defaults)."""

    image_size: int = 1024
    patch_size: int = 16
    num_channels: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    out_chans: int = 256
    window_size: int = 14
    # Blocks that use GLOBAL attention (all others use windowed attention).
    global_attn_indexes: tuple = (2, 5, 8, 11)
    layer_norm_eps: float = 1e-6


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over ``[B, C, H, W]`` (SAM neck convention)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def window_partition(
    x: torch.Tensor, window_size: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Partition ``[B, H, W, C]`` into non-overlapping windows with padding.

    Returns the windows ``[B*num_windows, window_size, window_size, C]`` and the
    padded ``(Hp, Wp)`` so the inverse can crop back.
    """
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w

    x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, c)
    )
    return windows, (hp, wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> torch.Tensor:
    """Inverse of :func:`window_partition`; crops back to ``[B, H, W, C]``."""
    hp, wp = pad_hw
    h, w = hw
    b = windows.shape[0] // ((hp // window_size) * (wp // window_size))
    x = windows.view(
        b, hp // window_size, wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, hp, wp, -1)
    if hp > h or wp > w:
        x = x[:, :h, :w, :].contiguous()
    return x


class SAMAttentionSDPA(nn.Module):
    """Multi-head self-attention over flattened spatial tokens (SDPA)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        b, h, w, c = x.shape
        n = h * w
        qkv = (
            self.qkv(x.reshape(b, n, c))
            .reshape(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, h, w, c)
        return self.proj(attn)


class SAMBlockSDPA(nn.Module):
    """Transformer block with optional windowed attention (SAM/ViTDet style)."""

    def __init__(self, cfg: SAMVaryConfig, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        self.attn = SAMAttentionSDPA(cfg.embed_dim, cfg.num_heads)
        self.norm2 = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)
        hidden = int(cfg.embed_dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            h, w = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            x = self.attn(x)
            x = window_unpartition(x, self.window_size, pad_hw, (h, w))
        else:
            x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SAMVaryViTSDPA(nn.Module):
    """SAM ("Vary") ViT image encoder with SDPA attention.

    Produces a dense spatial feature map suitable as the high-resolution branch
    of the LL-OCADR vision tower.
    """

    def __init__(self, cfg: SAMVaryConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or SAMVaryConfig()
        self.cfg = cfg
        grid = cfg.image_size // cfg.patch_size

        self.patch_embed = nn.Conv2d(
            cfg.num_channels,
            cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, grid, grid, cfg.embed_dim))

        self.blocks = nn.ModuleList(
            [
                SAMBlockSDPA(
                    cfg,
                    window_size=0
                    if i in cfg.global_attn_indexes
                    else cfg.window_size,
                )
                for i in range(cfg.depth)
            ]
        )

        self.neck = nn.Sequential(
            nn.Conv2d(cfg.embed_dim, cfg.out_chans, kernel_size=1, bias=False),
            LayerNorm2d(cfg.out_chans),
            nn.Conv2d(
                cfg.out_chans, cfg.out_chans, kernel_size=3, padding=1, bias=False
            ),
            LayerNorm2d(cfg.out_chans),
        )

    @property
    def out_chans(self) -> int:
        return self.cfg.out_chans

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images into a dense feature map.

        Args:
            pixel_values: ``[B, 3, H, W]``.

        Returns:
            ``[B, out_chans, H/patch, W/patch]`` feature map.
        """
        x = self.patch_embed(pixel_values)  # [B, C, gh, gw]
        x = x.permute(0, 2, 3, 1)  # [B, gh, gw, C]
        x = x + self._interpolate_pos_embed(x.shape[1], x.shape[2])

        for block in self.blocks:
            x = block(x)

        x = x.permute(0, 3, 1, 2)  # [B, C, gh, gw]
        return self.neck(x)

    def _interpolate_pos_embed(self, gh: int, gw: int) -> torch.Tensor:
        """Bicubically resize the absolute pos-embed to a ``gh x gw`` grid."""
        if self.pos_embed.shape[1] == gh and self.pos_embed.shape[2] == gw:
            return self.pos_embed
        pe = self.pos_embed.permute(0, 3, 1, 2)  # [1, C, H, W]
        pe = F.interpolate(pe, size=(gh, gw), mode="bicubic", align_corners=False)
        return pe.permute(0, 2, 3, 1)


def build_sam_vary_sdpa(
    image_size: int = 1024,
    patch_size: int = 16,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    out_chans: int = 256,
) -> SAMVaryViTSDPA:
    """Factory for a :class:`SAMVaryViTSDPA` encoder (mirrors build_* siblings)."""
    return SAMVaryViTSDPA(
        SAMVaryConfig(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            out_chans=out_chans,
        )
    )
