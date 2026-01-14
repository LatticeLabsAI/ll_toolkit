"""
MLP Projector for LL-OCADR.
Projects concatenated 3D geometry features to LLM embedding space.
Adapted from DeepSeek-OCR's build_linear.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpProjector(nn.Module):
    """
    Projects concatenated 3D features (GeometryNet + ShapeNet) to LLM embedding space.

    Input: [batch, num_tokens, input_dim] where input_dim = 1024 (256 + 768)
    Output: [batch, num_tokens, n_embed] where n_embed = 1280 (LLM dimension)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            # Simple linear projection: 1024 -> 1280
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            # MLP with GELU activation
            mlp_depth = cfg.get("depth", 1)
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        self.layers = modules

    def forward(self, x):
        """
        Args:
            x: [batch, num_tokens, input_dim] - concatenated 3D features

        Returns:
            [batch, num_tokens, n_embed] - projected to LLM embedding space
        """
        return self.layers(x)

    @staticmethod
    def get_flops_per_sample(cfg):
        """Calculate FLOPs for this projector."""
        if cfg.projector_type == "linear":
            fwd = 2 * cfg.input_dim * cfg.n_embed
        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            fwd = 2 * cfg.input_dim * cfg.n_embed + (mlp_depth - 1) * 2 * cfg.n_embed * cfg.n_embed
        else:
            fwd = 0
        return fwd * 3
