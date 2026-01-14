"""EdgeConv-based graph neural network for mesh segmentation.

Implements EdgeConv layers for dynamic graph learning on mesh face graphs.
Based on "Dynamic Graph CNN for Learning on Point Clouds and Meshes" (Wang et al., 2019).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops

_log = logging.getLogger(__name__)


class EdgeConvBlock(MessagePassing):
    """EdgeConv block with dynamic edge features.

    Computes edge features as h_ij = MLP([h_i, h_j - h_i]) for each edge (i,j).
    Aggregates with max pooling: h_i' = max_j h_ij.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        aggr: Aggregation method ('max', 'mean', 'add')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "max",
    ):
        super().__init__(aggr=aggr)

        # Edge MLP: processes [h_i || (h_j - h_i)]
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge indices

        Returns:
            [N, out_channels] updated node features
        """
        # Propagate messages
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Compute edge features.

        Args:
            x_i: [E, D] source node features
            x_j: [E, D] target node features

        Returns:
            [E, out_channels] edge features
        """
        # Concatenate [h_i || (h_j - h_i)]
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)  # [E, 2*D]

        # Process through edge MLP
        return self.edge_mlp(edge_features)  # [E, out_channels]


class EdgeConvNet(nn.Module):
    """Multi-layer EdgeConv network for mesh segmentation.

    Architecture:
    - 5 EdgeConv blocks with skip connections
    - Progressive feature dimension: 64 → 128 → 256 → 512 → 512
    - Global max pooling + concatenation
    - Segmentation head with dropout

    Args:
        in_channels: Input node feature dimension (default: 7)
        num_classes: Number of segmentation classes
        hidden_dims: List of hidden dimensions for EdgeConv blocks
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int = 7,
        num_classes: int = 12,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512, 512]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
        )

        # EdgeConv blocks
        self.edge_convs = nn.ModuleList()
        prev_dim = hidden_dims[0]

        for hidden_dim in hidden_dims:
            self.edge_convs.append(
                EdgeConvBlock(in_channels=prev_dim, out_channels=hidden_dim)
            )
            prev_dim = hidden_dim

        # Global pooling dimension
        # Concatenate all block outputs + global feature
        total_dim = sum(hidden_dims) + hidden_dims[-1]

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge indices
            batch: [N] batch assignment (for batched graphs)

        Returns:
            [N, num_classes] per-node class logits
        """
        # Input projection
        x = self.input_proj(x)

        # Collect features from all EdgeConv blocks
        block_features = []

        for edge_conv in self.edge_convs:
            x = edge_conv(x, edge_index)
            block_features.append(x)

        # Concatenate all block features
        x_concat = torch.cat(block_features, dim=1)  # [N, sum(hidden_dims)]

        # Global max pooling
        if batch is None:
            # Single graph: pool over all nodes
            global_feature = torch.max(x_concat, dim=0, keepdim=True)[0]  # [1, D]
            global_feature = global_feature.expand(x_concat.size(0), -1)  # [N, D]
        else:
            # Batched graphs: pool per graph
            from torch_geometric.nn import global_max_pool

            global_feature = global_max_pool(x_concat, batch)  # [B, D]
            # Expand to match node count
            global_feature = global_feature[batch]  # [N, D]

        # Concatenate local + global features
        x_final = torch.cat([x_concat, global_feature], dim=1)  # [N, total_dim]

        # Segmentation head
        logits = self.seg_head(x_final)  # [N, num_classes]

        return logits


class MeshSegmentationGNN(nn.Module):
    """Complete mesh segmentation architecture with EdgeConv.

    Includes optional integration with pretrained encoders (ShapeNet/GeometryNet).

    Args:
        in_channels: Input node feature dimension
        num_classes: Number of segmentation classes
        hidden_dims: Hidden dimensions for EdgeConv blocks
        use_pretrained_encoders: Whether to use pretrained shape/geometry encoders
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int = 7,
        num_classes: int = 12,
        hidden_dims: list[int] = None,
        use_pretrained_encoders: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.use_pretrained_encoders = use_pretrained_encoders

        # EdgeConv backbone
        self.edge_conv_net = EdgeConvNet(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # Optional: Pretrained encoders for additional context
        if use_pretrained_encoders:
            try:
                from ll_ocadr.vllm.lattice_encoder.geometry_net import build_geometry_net
                from ll_ocadr.vllm.lattice_encoder.shape_net import build_shape_net

                self.geometry_net = build_geometry_net()
                self.shape_net = build_shape_net()

                # Freeze pretrained encoders
                for param in self.geometry_net.parameters():
                    param.requires_grad = False
                for param in self.shape_net.parameters():
                    param.requires_grad = False

                _log.info("Loaded pretrained ShapeNet and GeometryNet encoders")
            except ImportError:
                _log.warning(
                    "Could not import ll_ocadr encoders, continuing without pretrained features"
                )
                self.use_pretrained_encoders = False

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge indices
            batch: [N] batch assignment
            coords: [N, 3] node coordinates (for pretrained encoders)
            normals: [N, 3] node normals (for pretrained encoders)

        Returns:
            [N, num_classes] per-node class logits
        """
        # Optional: Extract features from pretrained encoders
        if self.use_pretrained_encoders and coords is not None and normals is not None:
            with torch.no_grad():
                # Proper batch handling - split nodes by batch assignment
                if batch is None:
                    # Single graph - use simple unsqueeze
                    coords_batch = coords.unsqueeze(0)  # [1, N, 3]
                    normals_batch = normals.unsqueeze(0)  # [1, N, 3]

                    # Extract global shape features
                    shape_features = self.shape_net(coords_batch, normals_batch)  # [1, 256, D]
                    geo_features = self.geometry_net(coords_batch, normals_batch)  # [1, 128, D]

                    # Pool and expand
                    shape_global = torch.mean(shape_features, dim=1)  # [1, D]
                    geo_local = torch.mean(geo_features, dim=1)  # [1, D]

                    shape_global = shape_global.expand(x.size(0), -1)
                    geo_local = geo_local.expand(x.size(0), -1)

                else:
                    # Multiple graphs - split by batch and pad
                    unique_batches = torch.unique(batch, sorted=True)
                    num_batches = len(unique_batches)

                    # Find max nodes per graph for padding
                    max_nodes = max((batch == b).sum().item() for b in unique_batches)

                    # Prepare padded batches
                    coords_batched = torch.zeros(num_batches, max_nodes, 3, device=coords.device)
                    normals_batched = torch.zeros(num_batches, max_nodes, 3, device=normals.device)
                    node_counts = []

                    for i, b in enumerate(unique_batches):
                        mask = batch == b
                        coords_b = coords[mask]
                        normals_b = normals[mask]
                        n_nodes = coords_b.size(0)
                        node_counts.append(n_nodes)

                        # Pad to max_nodes
                        coords_batched[i, :n_nodes] = coords_b
                        normals_batched[i, :n_nodes] = normals_b

                    # Extract features from encoders
                    shape_features = self.shape_net(coords_batched, normals_batched)  # [B, 256, D]
                    geo_features = self.geometry_net(coords_batched, normals_batched)  # [B, 128, D]

                    # Pool per graph
                    shape_global_batched = torch.mean(shape_features, dim=1)  # [B, D]
                    geo_local_batched = torch.mean(geo_features, dim=1)  # [B, D]

                    # Expand to all nodes in each graph
                    shape_global = torch.zeros(x.size(0), shape_global_batched.size(1), device=x.device)
                    geo_local = torch.zeros(x.size(0), geo_local_batched.size(1), device=x.device)

                    for i, b in enumerate(unique_batches):
                        mask = batch == b
                        shape_global[mask] = shape_global_batched[i]
                        geo_local[mask] = geo_local_batched[i]

                # Concatenate with input features
                x = torch.cat([x, shape_global, geo_local], dim=1)

        # Main EdgeConv network
        logits = self.edge_conv_net(x, edge_index, batch)

        return logits
