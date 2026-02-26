"""Vertex prediction heads for CAD mesh generation.

Provides neural network modules for predicting 3D vertex positions
from decoder hidden states.  Unlike :class:`ParameterHeads` (which
predict quantised parameter bins per command slot), these heads
directly output continuous 3D coordinates for mesh vertices.

Two modules are provided:

- :class:`VertexPredictionHead` — Predicts vertex presence + coarse
  positions from sequence hidden states.
- :class:`VertexRefinementHead` — Learned iterative refinement that
  takes coarse positions + context and produces sub-grid accurate
  positions via progressively smaller deltas.

Both can be used as supplementary outputs alongside
:class:`CompositeHead` in any generation architecture (VAE, VQ-VAE,
diffusion).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class VertexPredictionOutput:
    """Container for vertex prediction head outputs.

    Attributes:
        vertex_presence: ``[B, max_vertices]`` logits indicating
            whether each vertex slot is occupied.
        coarse_positions: ``[B, max_vertices, 3]`` initial position
            estimates.
        refined_positions: ``[B, max_vertices, 3]`` positions after
            learned refinement (may equal coarse if no refinement).
    """

    vertex_presence: torch.Tensor
    coarse_positions: torch.Tensor
    refined_positions: torch.Tensor


# ---------------------------------------------------------------------------
# VertexPredictionHead
# ---------------------------------------------------------------------------


class VertexPredictionHead(nn.Module):
    """Predict vertex positions from decoder hidden states.

    Architecture:

    1. **Presence head**: MLP → ``[B, max_vertices]`` logits.
       Answers *which* vertex slots are occupied.
    2. **Coarse position head**: MLP → ``[B, max_vertices, 3]``.
       Gives initial (x, y, z) estimates for each vertex.
    3. **Optional refinement**: A :class:`VertexRefinementHead` that
       iteratively improves positions if ``num_refinement_steps > 0``.

    The module pools the sequence dimension of the hidden states
    (mean pool) before predicting vertex positions, so the output
    is sequence-length independent.

    Args:
        embed_dim: Dimension of decoder hidden states.
        max_vertices: Maximum number of vertex slots per shape.
        hidden_dim: Hidden dimension for MLP layers.
        num_refinement_steps: Number of learned refinement iterations.
            Set to 0 to disable learned refinement.

    Example::

        head = VertexPredictionHead(embed_dim=256, max_vertices=512)
        output = head(hidden_states)  # hidden_states: [B, S, 256]
        # output.vertex_presence: [B, 512]
        # output.refined_positions: [B, 512, 3]
    """

    def __init__(
        self,
        embed_dim: int = 256,
        max_vertices: int = 512,
        hidden_dim: int = 256,
        num_refinement_steps: int = 3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_vertices = max_vertices
        self.hidden_dim = hidden_dim

        # Sequence → shape-level embedding via mean pool + projection
        self.pool_projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Stage 1: Predict which vertex slots are occupied
        self.presence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_vertices),
        )

        # Stage 2: Predict coarse (x, y, z) per vertex slot
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_vertices * 3),
        )

        # Stage 3: Optional learned refinement
        self.refiner: Optional[VertexRefinementHead] = None
        if num_refinement_steps > 0:
            self.refiner = VertexRefinementHead(
                vertex_dim=3,
                context_dim=hidden_dim,
                num_iterations=num_refinement_steps,
                hidden_dim=hidden_dim // 2,
            )

        _log.info(
            "VertexPredictionHead: embed_dim=%d, max_vertices=%d, "
            "hidden_dim=%d, refinement_steps=%d",
            embed_dim, max_vertices, hidden_dim, num_refinement_steps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> VertexPredictionOutput:
        """Predict vertex positions from decoder output.

        Args:
            hidden_states: ``[B, S, embed_dim]`` from the decoder.

        Returns:
            :class:`VertexPredictionOutput` with presence logits,
            coarse positions, and refined positions.
        """
        batch_size = hidden_states.shape[0]

        # Mean-pool across sequence dimension → shape embedding
        shape_embed = hidden_states.mean(dim=1)  # [B, embed_dim]
        shape_embed = self.pool_projection(shape_embed)  # [B, hidden_dim]

        # Predict vertex presence
        vertex_presence = self.presence_head(shape_embed)  # [B, max_v]

        # Predict coarse positions
        pos_flat = self.position_head(shape_embed)  # [B, max_v * 3]
        coarse_positions = pos_flat.view(
            batch_size, self.max_vertices, 3
        )  # [B, max_v, 3]

        # Clamp to [-1, 1] (common normalisation range for CAD)
        coarse_positions = torch.tanh(coarse_positions)

        # Optional learned refinement
        if self.refiner is not None:
            refined_positions = self.refiner(
                coarse_positions, context=shape_embed
            )
        else:
            refined_positions = coarse_positions

        return VertexPredictionOutput(
            vertex_presence=vertex_presence,
            coarse_positions=coarse_positions,
            refined_positions=refined_positions,
        )

    @torch.no_grad()
    def decode_vertices(
        self,
        output: VertexPredictionOutput,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode predicted vertices to numpy arrays.

        Filters to only the vertices whose presence logit exceeds
        ``threshold`` (after sigmoid).

        Args:
            output: Output from :meth:`forward`.
            threshold: Presence probability threshold.

        Returns:
            Tuple of:
                - ``vertices``: ``(K, 3)`` float32 array of active vertices.
                - ``presence_mask``: ``(max_vertices,)`` bool array.
        """
        # Take first sample in batch
        presence_probs = torch.sigmoid(output.vertex_presence[0])  # [max_v]
        mask = presence_probs > threshold  # [max_v]

        positions = output.refined_positions[0]  # [max_v, 3]
        active_positions = positions[mask]  # [K, 3]

        return (
            active_positions.cpu().numpy().astype(np.float32),
            mask.cpu().numpy(),
        )


# ---------------------------------------------------------------------------
# VertexRefinementHead
# ---------------------------------------------------------------------------


class VertexRefinementHead(nn.Module):
    """Learned iterative refinement of coarse vertex positions.

    Takes coarse vertex positions and a shape-level context vector,
    then applies ``num_iterations`` refinement steps.  Each step
    predicts a position delta that is added with a decreasing scale
    factor for stability.

    Architecture per iteration:

    1. Concatenate position ``[B, N, 3]`` with broadcasted context
       ``[B, N, context_dim]``.
    2. MLP → delta ``[B, N, 3]``.
    3. Update: ``pos = pos + alpha_i * delta`` where
       ``alpha_i = 1 / (i + 1)``.

    Args:
        vertex_dim: Coordinate dimension (3 for xyz).
        context_dim: Dimension of the shape-level context embedding.
        num_iterations: Number of refinement iterations.
        hidden_dim: Hidden dimension for the refinement MLPs.

    Example::

        refiner = VertexRefinementHead(context_dim=256, num_iterations=3)
        refined = refiner(coarse_positions, context=shape_embedding)
    """

    def __init__(
        self,
        vertex_dim: int = 3,
        context_dim: int = 256,
        num_iterations: int = 3,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.vertex_dim = vertex_dim
        self.context_dim = context_dim
        self.num_iterations = num_iterations

        # One refinement MLP per iteration
        self.refinement_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(vertex_dim + context_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, vertex_dim),
            )
            for _ in range(num_iterations)
        ])

        _log.info(
            "VertexRefinementHead: vertex_dim=%d, context_dim=%d, "
            "iterations=%d, hidden=%d",
            vertex_dim, context_dim, num_iterations, hidden_dim,
        )

    def forward(
        self,
        coarse_positions: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Iteratively refine vertex positions.

        Args:
            coarse_positions: ``[B, N, vertex_dim]`` initial estimates.
            context: ``[B, context_dim]`` shape-level embedding.  If
                ``None``, a zero context is used.

        Returns:
            Refined positions ``[B, N, vertex_dim]``.
        """
        batch_size, num_vertices, vdim = coarse_positions.shape
        positions = coarse_positions

        # Broadcast context → [B, N, context_dim]
        if context is not None:
            ctx = context.unsqueeze(1).expand(
                batch_size, num_vertices, self.context_dim
            )
        else:
            ctx = torch.zeros(
                batch_size, num_vertices, self.context_dim,
                device=positions.device, dtype=positions.dtype,
            )

        for i, layer in enumerate(self.refinement_layers):
            # Concatenate position with context
            inp = torch.cat([positions, ctx], dim=-1)  # [B, N, vdim + ctx_dim]

            # Predict delta
            delta = layer(inp)  # [B, N, vdim]

            # Apply with decreasing step size
            alpha = 1.0 / (i + 1.0)
            positions = positions + alpha * delta
            positions = torch.clamp(positions, -1.0, 1.0)

        return positions
