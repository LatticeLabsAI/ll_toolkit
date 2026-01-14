"""Instance segmentation head for grouping faces into feature instances.

Implements discriminative loss for instance clustering based on embeddings.
Based on "Semantic Instance Segmentation with a Discriminative Loss Function" (De Brabandere et al., 2017).
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)


class InstanceSegmentationHead(nn.Module):
    """Instance segmentation head using embedding clustering.

    Learns per-face embeddings such that:
    - Faces in the same instance have similar embeddings (pull)
    - Faces in different instances have dissimilar embeddings (push)

    Args:
        in_dim: Input feature dimension
        embedding_dim: Embedding dimension
        num_layers: Number of MLP layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_dim: int,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Build MLP for embedding projection
        layers = []
        current_dim = in_dim

        for i in range(num_layers - 1):
            hidden_dim = max(embedding_dim * 2, current_dim // 2)
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        # Final projection to embedding space
        layers.append(nn.Linear(current_dim, embedding_dim))

        self.embedding_mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to embedding space.

        Args:
            x: [N, in_dim] input features

        Returns:
            [N, embedding_dim] embeddings
        """
        embeddings = self.embedding_mlp(x)

        # L2 normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


def discriminative_loss(
    embeddings: torch.Tensor,
    instance_labels: torch.Tensor,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute discriminative loss for instance segmentation.

    Loss has three components:
    1. Variance loss (L_var): Pull embeddings of same instance together
    2. Distance loss (L_dist): Push embeddings of different instances apart
    3. Regularization loss (L_reg): Keep cluster centers near origin

    Args:
        embeddings: [N, D] face embeddings
        instance_labels: [N] instance ID for each face
        delta_v: Variance margin (intra-cluster)
        delta_d: Distance margin (inter-cluster)

    Returns:
        total_loss: Combined loss
        var_loss: Variance component
        dist_loss: Distance component
        reg_loss: Regularization component
    """
    unique_labels = torch.unique(instance_labels)
    num_instances = len(unique_labels)

    if num_instances == 0:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    # Compute cluster centers (mean embeddings per instance)
    cluster_centers = []

    for label in unique_labels:
        mask = instance_labels == label
        cluster_mean = embeddings[mask].mean(dim=0)
        cluster_centers.append(cluster_mean)

    cluster_centers = torch.stack(cluster_centers)  # [K, D]

    # 1. Variance loss: Pull embeddings to cluster centers
    var_loss = 0.0

    for i, label in enumerate(unique_labels):
        mask = instance_labels == label
        cluster_embeddings = embeddings[mask]  # [N_i, D]
        cluster_center = cluster_centers[i]  # [D]

        # Distance to cluster center
        distances = torch.norm(cluster_embeddings - cluster_center, dim=1)  # [N_i]

        # Hinge loss: max(0, distance - delta_v)^2
        var_term = F.relu(distances - delta_v) ** 2
        var_loss += var_term.mean()

    var_loss /= num_instances

    # 2. Distance loss: Push cluster centers apart
    dist_loss = 0.0

    if num_instances > 1:
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                center_i = cluster_centers[i]
                center_j = cluster_centers[j]

                # Distance between cluster centers
                inter_distance = torch.norm(center_i - center_j)

                # Hinge loss: max(0, 2*delta_d - distance)^2
                dist_term = F.relu(2 * delta_d - inter_distance) ** 2
                dist_loss += dist_term

        # Normalize by number of pairs
        num_pairs = num_instances * (num_instances - 1) / 2
        dist_loss /= num_pairs

    # 3. Regularization loss: Keep cluster centers near origin
    reg_loss = torch.norm(cluster_centers, dim=1).mean()

    # Total loss
    total_loss = var_loss + dist_loss + 0.001 * reg_loss

    return total_loss, var_loss, dist_loss, reg_loss


def cluster_embeddings(
    embeddings: torch.Tensor,
    distance_threshold: float = 0.5,
    min_cluster_size: int = 3,
) -> torch.Tensor:
    """Cluster embeddings into instances using agglomerative clustering.

    Args:
        embeddings: [N, D] face embeddings
        distance_threshold: Distance threshold for merging clusters
        min_cluster_size: Minimum faces per cluster

    Returns:
        [N] predicted instance labels
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    import numpy as np

    # Convert to numpy for scipy
    embeddings_np = embeddings.detach().cpu().numpy()

    # Agglomerative clustering
    linkage_matrix = linkage(embeddings_np, method="average", metric="euclidean")

    # Cut dendrogram at threshold
    labels = fcluster(linkage_matrix, distance_threshold, criterion="distance")

    # Filter small clusters
    unique_labels, counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique_labels, counts):
        if count < min_cluster_size:
            # Assign to nearest large cluster
            mask = labels == label
            small_cluster_embeddings = embeddings_np[mask]

            # Find nearest large cluster center
            large_clusters = unique_labels[counts >= min_cluster_size]

            if len(large_clusters) > 0:
                min_dist = float("inf")
                nearest_label = large_clusters[0]

                for large_label in large_clusters:
                    large_mask = labels == large_label
                    large_center = embeddings_np[large_mask].mean(axis=0)

                    # Distance to small cluster center
                    small_center = small_cluster_embeddings.mean(axis=0)
                    dist = np.linalg.norm(large_center - small_center)

                    if dist < min_dist:
                        min_dist = dist
                        nearest_label = large_label

                labels[mask] = nearest_label

    return torch.from_numpy(labels).long().to(embeddings.device)


def compute_instance_iou(
    pred_labels: torch.Tensor,
    gt_labels: torch.Tensor,
) -> float:
    """Compute instance-level IoU (Intersection over Union).

    Args:
        pred_labels: [N] predicted instance labels
        gt_labels: [N] ground truth instance labels

    Returns:
        Mean IoU across all instances
    """
    unique_gt = torch.unique(gt_labels)
    ious = []

    for gt_label in unique_gt:
        gt_mask = gt_labels == gt_label

        # Find best matching predicted instance
        unique_pred = torch.unique(pred_labels[gt_mask])

        if len(unique_pred) == 0:
            ious.append(0.0)
            continue

        best_iou = 0.0

        for pred_label in unique_pred:
            pred_mask = pred_labels == pred_label

            # Compute IoU
            intersection = (gt_mask & pred_mask).sum().float()
            union = (gt_mask | pred_mask).sum().float()

            if union > 0:
                iou = intersection / union
                best_iou = max(best_iou, iou.item())

        ious.append(best_iou)

    return sum(ious) / len(ious) if ious else 0.0
