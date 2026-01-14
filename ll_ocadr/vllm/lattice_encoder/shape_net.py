"""
ShapeNet - Global Shape Encoder for LL-OCADR.
Extracts high-level semantic shape features from full mesh context.
Based on Point-BERT / Point-MAE transformer architecture.
"""

import torch
import torch.nn as nn
import numpy as np


class PointPatchEmbedding(nn.Module):
    """
    Tokenize point cloud into spatial patches.
    Divides point cloud into groups and embeds each group.
    """

    def __init__(self, patch_size=32, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Local feature extractor (mini PointNet per patch)
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),  # Input: xyz + normals (6 dims)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, embed_dim, 1)
        )

    def forward(self, coords, normals):
        """
        Args:
            coords: [B, N, 3] vertex coordinates
            normals: [B, N, 3] vertex normals

        Returns:
            patch_tokens: [B, num_patches, embed_dim]
        """
        B, N, _ = coords.shape

        # Concatenate coords + normals
        points = torch.cat([coords, normals], dim=-1)  # [B, N, 6]
        points = points.transpose(1, 2)  # [B, 6, N]

        # Extract local features
        feature = self.first_conv(points)  # [B, 256, N]

        # Max pool to get patch-level features
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # [B, 256, 1]
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)  # [B, 512, N]

        feature = self.second_conv(feature)  # [B, embed_dim, N]

        # Group into patches
        num_patches = 256  # Fixed number of patches
        patch_size = N // num_patches

        if patch_size == 0:
            # Too few points, pad
            patch_size = 1
            num_patches = N

        # Reshape into patches and max pool within each patch
        patches = feature[:, :, :num_patches * patch_size].view(B, self.embed_dim, num_patches, patch_size)
        patch_tokens = torch.max(patches, dim=-1)[0]  # [B, embed_dim, num_patches]
        patch_tokens = patch_tokens.transpose(1, 2)  # [B, num_patches, embed_dim]

        return patch_tokens


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with self-attention and FFN.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, embed_dim]

        Returns:
            x: [B, N, embed_dim]
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # FFN with residual
        x = x + self.mlp(self.norm2(x))

        return x


class ShapeNet(nn.Module):
    """
    Global shape encoder for full mesh context.
    Equivalent to CLIP for images, extracts high-level semantic features.

    Architecture:
        - Patch-based tokenization of point cloud
        - Transformer encoder with positional encoding
        - CLS token for global shape representation

    Input: coords [B, N, 3], normals [B, N, 3]
    Output: features [B, 257, 768] - CLS + 256 patch tokens
    """

    def __init__(self, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim

        # Point cloud tokenizer
        self.patch_embed = PointPatchEmbedding(
            patch_size=32, embed_dim=embed_dim
        )

        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 257, embed_dim)  # 256 patches + CLS
        )

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, coords, normals):
        """
        Args:
            coords: [B, N, 3] downsampled mesh vertices
            normals: [B, N, 3] vertex normals

        Returns:
            features: [B, 257, 768] - CLS token + 256 patch tokens
        """
        B = coords.shape[0]

        # Tokenize point cloud into patches
        patch_tokens = self.patch_embed(coords, normals)  # [B, num_patches, embed_dim]

        # Handle variable number of patches
        num_patches = patch_tokens.shape[1]
        if num_patches < 256:
            # Pad with zeros
            padding = torch.zeros(B, 256 - num_patches, self.embed_dim, device=patch_tokens.device)
            patch_tokens = torch.cat([patch_tokens, padding], dim=1)
        elif num_patches > 256:
            # Truncate
            patch_tokens = patch_tokens[:, :256]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)  # [B, 257, embed_dim]

        # Add positional encoding
        tokens = tokens + self.pos_embed

        # Transformer encoding
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        return tokens  # [B, 257, 768]


def build_shape_net(embed_dim=768, depth=12, num_heads=12):
    """Build ShapeNet encoder."""
    return ShapeNet(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
