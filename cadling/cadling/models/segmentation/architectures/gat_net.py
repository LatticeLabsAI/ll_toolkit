"""Graph Attention Network for B-Rep face segmentation.

Implements multi-head graph attention for learning on B-Rep face adjacency graphs.
Based on "Graph Attention Networks" (Veličković et al., 2018) and BRepGAT architecture.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)


class GraphAttentionEncoder(nn.Module):
    """Multi-layer Graph Attention encoder for B-Rep faces.

    Architecture follows BRepGAT:
    - Multiple GAT layers with multi-head attention
    - Residual connections for gradient flow
    - Layer normalization for training stability

    Args:
        in_dim: Input node feature dimension
        hidden_dim: Hidden dimension per attention head
        num_layers: Number of GAT layers
        num_heads: Number of attention heads per layer
        dropout: Dropout rate
        edge_dim: Edge feature dimension (optional)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        try:
            from torch_geometric.nn import GATConv

            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim * num_heads),
                nn.LayerNorm(hidden_dim * num_heads),
                nn.ReLU(),
            )

            # GAT layers
            self.gat_layers = nn.ModuleList()
            self.layer_norms = nn.ModuleList()

            for i in range(num_layers):
                # Input dim for first layer is hidden_dim * num_heads
                if i == 0:
                    layer_in_dim = hidden_dim * num_heads
                else:
                    layer_in_dim = hidden_dim * num_heads

                # Last layer doesn't concatenate heads
                concat_heads = i < num_layers - 1

                gat_layer = GATConv(
                    in_channels=layer_in_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=concat_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                )

                self.gat_layers.append(gat_layer)

                # Layer norm after each GAT layer
                if concat_heads:
                    norm_dim = hidden_dim * num_heads
                else:
                    norm_dim = hidden_dim

                self.layer_norms.append(nn.LayerNorm(norm_dim))

        except ImportError:
            _log.error("torch_geometric not installed, cannot create GAT layers")
            raise

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [N, in_dim] node features
            edge_index: [2, E] edge indices
            edge_attr: [E, edge_dim] edge features (optional)

        Returns:
            [N, hidden_dim] node embeddings (last layer has no head concatenation)
        """
        # Input projection
        x = self.input_proj(x)

        # GAT layers with residual connections
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            # Store input for residual
            residual = x

            # GAT layer
            x = gat_layer(x, edge_index, edge_attr=edge_attr)

            # Layer norm
            x = layer_norm(x)

            # Residual connection (skip first layer if dims don't match)
            if i > 0 and residual.size(-1) == x.size(-1):
                x = x + residual

            # Activation (except last layer)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)

        return x


class HybridGATTransformer(nn.Module):
    """Hybrid GAT + Transformer architecture for B-Rep segmentation.

    Stage 1: GAT for local face relationships
    Stage 2: Transformer for global part context

    Args:
        in_dim: Input node feature dimension
        gat_hidden_dim: Hidden dimension for GAT
        gat_num_heads: Number of attention heads in GAT
        gat_num_layers: Number of GAT layers
        transformer_hidden_dim: Hidden dimension for Transformer
        transformer_num_layers: Number of Transformer layers
        transformer_num_heads: Number of attention heads in Transformer
        num_classes: Number of output classes
        dropout: Dropout rate
        edge_dim: Edge feature dimension
    """

    def __init__(
        self,
        in_dim: int = 24,
        gat_hidden_dim: int = 256,
        gat_num_heads: int = 8,
        gat_num_layers: int = 3,
        transformer_hidden_dim: int = 512,
        transformer_num_layers: int = 4,
        transformer_num_heads: int = 8,
        num_classes: int = 24,
        dropout: float = 0.1,
        edge_dim: Optional[int] = 8,
    ):
        super().__init__()

        # Stage 1: GAT for local topology
        self.gat_encoder = GraphAttentionEncoder(
            in_dim=in_dim,
            hidden_dim=gat_hidden_dim,
            num_layers=gat_num_layers,
            num_heads=gat_num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
        )

        # Project GAT output to transformer dimension
        self.gat_to_transformer = nn.Linear(gat_hidden_dim, transformer_hidden_dim)

        # Stage 2: Transformer for global context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim,
            nhead=transformer_num_heads,
            dim_feedforward=transformer_hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_num_layers,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim // 2),
            nn.LayerNorm(transformer_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [N, in_dim] node features
            edge_index: [2, E] edge indices
            edge_attr: [E, edge_dim] edge features
            batch: [N] batch assignment (for batched graphs)

        Returns:
            [N, num_classes] per-face class logits
        """
        # Stage 1: GAT for local topology
        x = self.gat_encoder(x, edge_index, edge_attr)  # [N, gat_hidden_dim]

        # Project to transformer dimension
        x = self.gat_to_transformer(x)  # [N, transformer_hidden_dim]

        # Stage 2: Transformer for global context
        # Reshape for transformer (expects [batch, seq_len, features])
        if batch is None:
            # Single graph
            x_seq = x.unsqueeze(0)  # [1, N, D]

            # Apply transformer
            x_transformed = self.transformer(x_seq)  # [1, N, D]

            # Flatten back
            x = x_transformed.squeeze(0)  # [N, D]
        else:
            # Batched graphs - PROPER PADDING FOR VARIABLE-SIZED GRAPHS
            unique_batches = torch.unique(batch, sorted=True)
            num_batches = len(unique_batches)

            # Find max nodes per graph for padding
            max_nodes = max((batch == b).sum().item() for b in unique_batches)

            # Create padded tensor [B, max_N, D]
            D = x.size(1)
            x_padded = torch.zeros(
                num_batches, max_nodes, D,
                device=x.device,
                dtype=x.dtype
            )

            # Store node counts for unpacking later
            node_counts = []

            # Fill in each graph's data with padding
            for i, b in enumerate(unique_batches):
                mask = batch == b
                x_b = x[mask]  # [N_b, D]
                n_nodes = x_b.size(0)
                node_counts.append(n_nodes)

                # Pad to max_nodes
                x_padded[i, :n_nodes] = x_b

            # Apply transformer to padded batch
            x_transformed = self.transformer(x_padded)  # [B, max_N, D]

            # Unpack back to original nodes
            x_list = []
            for i, n_nodes in enumerate(node_counts):
                # Extract only the valid (non-padded) nodes
                x_list.append(x_transformed[i, :n_nodes])  # [N_b, D]

            # Concatenate all graphs back to [total_N, D]
            x = torch.cat(x_list, dim=0)

        # Classification head
        logits = self.classifier(x)  # [N, num_classes]

        return logits


class GATWithEdgeFeatures(nn.Module):
    """Graph Attention Network that explicitly uses edge features.

    Extends standard GAT to incorporate edge attributes in attention computation.

    Args:
        in_channels: Input node feature dimension
        out_channels: Output node feature dimension
        edge_dim: Edge feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        concat: Whether to concatenate attention heads
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True,
    ):
        super().__init__()

        try:
            from torch_geometric.nn import GATConv

            self.gat = GATConv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=num_heads,
                concat=concat,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=True,
            )
        except ImportError:
            _log.error("torch_geometric not installed")
            raise

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge indices
            edge_attr: [E, edge_dim] edge features

        Returns:
            [N, out_channels * num_heads] (if concat) or [N, out_channels] (if not concat)
        """
        return self.gat(x, edge_index, edge_attr=edge_attr)
