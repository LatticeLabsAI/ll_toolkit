"""Neural network architectures for CAD segmentation.

This module provides graph neural network components:
- EdgeConvNet: EdgeConv-based mesh segmentation GNN
- GraphAttentionEncoder: GAT layers for B-Rep face graphs
- InstanceSegmentationHead: Instance clustering head
"""

from __future__ import annotations

from .edge_conv_net import EdgeConvNet, EdgeConvBlock, MeshSegmentationGNN
from .gat_net import GraphAttentionEncoder, HybridGATTransformer, GATWithEdgeFeatures
from .instance_segmentation import (
    InstanceSegmentationHead,
    discriminative_loss,
    cluster_embeddings,
    compute_instance_iou,
)

__all__ = [
    "EdgeConvNet",
    "EdgeConvBlock",
    "MeshSegmentationGNN",
    "GraphAttentionEncoder",
    "HybridGATTransformer",
    "GATWithEdgeFeatures",
    "InstanceSegmentationHead",
    "discriminative_loss",
    "cluster_embeddings",
    "compute_instance_iou",
]
