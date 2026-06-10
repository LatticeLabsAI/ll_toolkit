"""
Rendered-image vision tower for LL-OCADR.

Composes the two SDPA image encoders into a single dual-branch tower that mirrors
DeepSeek-OCR's DeepEncoder (a high-resolution SAM branch + a semantic CLIP
branch):

    pixel_values [B, 3, H, W]
      ├── SAM (sam_vary_sdpa)  -> [B, out_chans, gh, gw] --compress--> tokens
      └── CLIP (clip_sdpa)     -> [B, 1 + Np, clip_dim]  --drop CLS--> tokens
      concat tokens -> Linear -> [B, num_vision_tokens, n_embed]

The output tokens live in the LLM embedding space (``n_embed``) so they can be
spliced into the language model's input sequence exactly like the 3D mesh
tokens. Both encoders interpolate their positional embeddings, so the same image
resolution can feed both branches.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .clip_sdpa import CLIPVisionConfig, CLIPVisionSDPA
from .sam_vary_sdpa import SAMVaryConfig, SAMVaryViTSDPA


class VisionTower(nn.Module):
    """Dual-branch (SAM + CLIP) rendered-image encoder producing LLM tokens."""

    def __init__(
        self,
        n_embed: int,
        clip_config: CLIPVisionConfig | None = None,
        sam_config: SAMVaryConfig | None = None,
        sam_compress_stride: int = 2,
    ) -> None:
        super().__init__()
        clip_config = clip_config or CLIPVisionConfig()
        sam_config = sam_config or SAMVaryConfig()

        self.clip = CLIPVisionSDPA(clip_config)
        self.sam = SAMVaryViTSDPA(sam_config)

        # Spatially compress the dense SAM feature map, then project it to the
        # CLIP embedding width so the two branches share a token dimension.
        self.sam_compress = nn.Sequential(
            nn.Conv2d(
                self.sam.out_chans,
                self.sam.out_chans,
                kernel_size=3,
                stride=sam_compress_stride,
                padding=1,
            ),
            nn.GELU(),
        )
        self.sam_proj = nn.Linear(self.sam.out_chans, self.clip.embed_dim)

        # Fuse the concatenated tokens into the LLM embedding space.
        self.projector = nn.Linear(self.clip.embed_dim, n_embed)
        self.n_embed = n_embed

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into LLM-space vision tokens.

        Args:
            pixel_values: ``[B, 3, H, W]`` image tensor.

        Returns:
            ``[B, num_vision_tokens, n_embed]`` where ``num_vision_tokens`` is
            the SAM compressed-grid token count plus the CLIP patch count.
        """
        # --- CLIP branch: semantic patch tokens (drop the class token) ---
        clip_tokens = self.clip(pixel_values)[:, 1:, :]  # [B, Np, clip_dim]

        # --- SAM branch: high-res feature map -> compressed tokens ---
        sam_map = self.sam(pixel_values)  # [B, out_chans, gh, gw]
        sam_map = self.sam_compress(sam_map)  # [B, out_chans, gh', gw']
        sam_tokens = sam_map.flatten(2).transpose(1, 2)  # [B, Ns, out_chans]
        sam_tokens = self.sam_proj(sam_tokens)  # [B, Ns, clip_dim]

        # --- Fuse ---
        tokens = torch.cat([sam_tokens, clip_tokens], dim=1)  # [B, Ns+Np, clip_dim]
        return self.projector(tokens)  # [B, Ns+Np, n_embed]


def build_vision_tower(
    n_embed: int,
    image_size: int = 224,
    clip_patch_size: int = 14,
    clip_embed_dim: int = 1024,
    clip_depth: int = 24,
    clip_num_heads: int = 16,
    sam_patch_size: int = 16,
    sam_embed_dim: int = 768,
    sam_depth: int = 12,
    sam_num_heads: int = 12,
    sam_out_chans: int = 256,
    sam_compress_stride: int = 2,
) -> VisionTower:
    """Factory for a :class:`VisionTower` (mirrors the package's build_* helpers)."""
    return VisionTower(
        n_embed=n_embed,
        clip_config=CLIPVisionConfig(
            image_size=image_size,
            patch_size=clip_patch_size,
            embed_dim=clip_embed_dim,
            depth=clip_depth,
            num_heads=clip_num_heads,
        ),
        sam_config=SAMVaryConfig(
            image_size=image_size,
            patch_size=sam_patch_size,
            embed_dim=sam_embed_dim,
            depth=sam_depth,
            num_heads=sam_num_heads,
            out_chans=sam_out_chans,
        ),
        sam_compress_stride=sam_compress_stride,
    )
