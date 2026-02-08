"""BRepNet coedge convolution encoder for B-Rep models.

Implements the BRepNet architecture that operates on coedge-level topology
for learning on B-Rep CAD models. Each topological edge in a B-Rep has two
oriented coedges (one per adjacent face), forming a natural graph structure
with next/prev/mate neighbor relationships.

Based on "BRepNet: A topological message passing system for solid models"
(Lambourne et al., 2021).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)


@dataclass
class CoedgeData:
    """Container for coedge-level graph data.

    Each topological edge in a B-Rep solid has two oriented coedges, one
    for each adjacent face. The coedges form a graph with three types of
    neighbor relationships:
      - next: the next coedge in the same face loop
      - prev: the previous coedge in the same face loop
      - mate: the opposing coedge on the adjacent face (same edge, opposite orientation)

    Attributes:
        features: [num_coedges, feature_dim] coedge feature tensor.
        next_indices: [num_coedges] index of the next coedge in the same face loop.
        prev_indices: [num_coedges] index of the previous coedge in the same face loop.
        mate_indices: [num_coedges] index of the mate coedge (opposing orientation).
        face_indices: [num_coedges] which face each coedge belongs to.
    """

    features: torch.Tensor
    next_indices: torch.Tensor
    prev_indices: torch.Tensor
    mate_indices: torch.Tensor
    face_indices: torch.Tensor

    def num_coedges(self) -> int:
        """Return the number of coedges."""
        return self.features.size(0)

    def num_faces(self) -> int:
        """Return the number of unique faces."""
        return int(self.face_indices.max().item()) + 1 if self.face_indices.numel() > 0 else 0

    def to(self, device: torch.device) -> CoedgeData:
        """Move all tensors to the given device.

        Args:
            device: Target device.

        Returns:
            New CoedgeData with tensors on the target device.
        """
        return CoedgeData(
            features=self.features.to(device),
            next_indices=self.next_indices.to(device),
            prev_indices=self.prev_indices.to(device),
            mate_indices=self.mate_indices.to(device),
            face_indices=self.face_indices.to(device),
        )


class CoedgeConvLayer(nn.Module):
    """Single coedge convolution layer.

    Aggregates information from three neighbor types (next, prev, mate)
    via learned linear transforms:

        h_i' = sigma(W_self * h_i + W_next * h_{next(i)}
                      + W_prev * h_{prev(i)} + W_mate * h_{mate(i)})

    Args:
        feature_dim: Input and output feature dimension.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()

        self.feature_dim = feature_dim

        self.W_self = nn.Linear(feature_dim, feature_dim)
        self.W_next = nn.Linear(feature_dim, feature_dim)
        self.W_prev = nn.Linear(feature_dim, feature_dim)
        self.W_mate = nn.Linear(feature_dim, feature_dim)

    def forward(self, coedge_data: CoedgeData) -> torch.Tensor:
        """Forward pass: aggregate from next, prev, mate neighbors.

        Args:
            coedge_data: CoedgeData containing features and neighbor indices.

        Returns:
            [num_coedges, feature_dim] updated coedge features.
        """
        h = coedge_data.features

        h_self = self.W_self(h)
        h_next = self.W_next(h[coedge_data.next_indices])
        h_prev = self.W_prev(h[coedge_data.prev_indices])
        h_mate = self.W_mate(h[coedge_data.mate_indices])

        h_out = F.relu(h_self + h_next + h_prev + h_mate)

        return h_out


class BRepNetEncoder(nn.Module):
    """BRepNet encoder with stacked coedge convolution layers.

    Architecture:
      - Input projection from input_dim to hidden_dim
      - num_layers stacked CoedgeConvLayers with residual connections + LayerNorm
      - Face-level pooling: aggregate coedge features per face (mean)
      - Graph-level pooling: attention pool over face embeddings

    Args:
        input_dim: Input coedge feature dimension (default 12).
        hidden_dim: Hidden feature dimension (default 128).
        output_dim: Output embedding dimension (default 128).
        num_layers: Number of CoedgeConvLayers (default 6).
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 6,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Stacked coedge convolution layers
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.conv_layers.append(CoedgeConvLayer(hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Output projection from hidden_dim to output_dim
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Attention pooling for graph-level embedding
        self.attn_gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 1),
        )

        _log.info(
            f"BRepNetEncoder: input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"output_dim={output_dim}, num_layers={num_layers}"
        )

    def forward(
        self, coedge_data: CoedgeData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            coedge_data: CoedgeData with coedge features and neighbor indices.

        Returns:
            face_embeddings: [num_faces, output_dim] per-face embeddings.
            graph_embedding: [output_dim] global graph embedding.
        """
        # Input projection
        h = self.input_proj(coedge_data.features)

        # Create a working copy of coedge_data with projected features
        working_data = CoedgeData(
            features=h,
            next_indices=coedge_data.next_indices,
            prev_indices=coedge_data.prev_indices,
            mate_indices=coedge_data.mate_indices,
            face_indices=coedge_data.face_indices,
        )

        # Apply stacked coedge conv layers with residual + norm
        for conv_layer, norm in zip(self.conv_layers, self.layer_norms):
            residual = working_data.features
            h_new = conv_layer(working_data)
            h_new = norm(h_new)

            # Residual connection
            h_new = h_new + residual

            # Update working data features
            working_data = CoedgeData(
                features=h_new,
                next_indices=coedge_data.next_indices,
                prev_indices=coedge_data.prev_indices,
                mate_indices=coedge_data.mate_indices,
                face_indices=coedge_data.face_indices,
            )

        # Output projection
        coedge_embeddings = self.output_proj(working_data.features)

        # Aggregate coedge features per face (mean pooling)
        num_faces = coedge_data.num_faces()
        face_embeddings = torch.zeros(
            num_faces, self.output_dim,
            device=coedge_embeddings.device,
            dtype=coedge_embeddings.dtype,
        )

        # Scatter mean: average coedge embeddings for each face
        face_idx_expanded = coedge_data.face_indices.unsqueeze(-1).expand_as(coedge_embeddings)
        face_embeddings.scatter_add_(0, face_idx_expanded, coedge_embeddings)

        # Count coedges per face for averaging
        face_counts = torch.zeros(
            num_faces, 1,
            device=coedge_embeddings.device,
            dtype=coedge_embeddings.dtype,
        )
        face_counts.scatter_add_(
            0,
            coedge_data.face_indices.unsqueeze(-1),
            torch.ones_like(coedge_data.face_indices.unsqueeze(-1), dtype=coedge_embeddings.dtype),
        )
        face_counts = face_counts.clamp(min=1.0)
        face_embeddings = face_embeddings / face_counts

        # Attention pooling for graph-level embedding
        attn_weights = F.softmax(self.attn_gate(face_embeddings), dim=0)
        graph_embedding = (attn_weights * face_embeddings).sum(dim=0)

        return face_embeddings, graph_embedding
