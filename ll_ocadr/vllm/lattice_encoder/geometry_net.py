"""
GeometryNet - Local Geometry Encoder for LL-OCADR.
Extracts fine-grained geometric features from mesh chunks.
Based on PointNet++ architecture with adaptations for CAD/Mesh processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each pair of points.

    Args:
        src: [B, N, C]
        dst: [B, M, C]

    Returns:
        dist: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def _farthest_point_sample_python(xyz, npoint):
    """
    Pure-Python fallback FPS. Used when torch_cluster is not available.

    Args:
        xyz: [B, N, 3] point cloud
        npoint: number of samples

    Returns:
        centroids: [B, npoint] sampled point indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


try:
    from torch_cluster import fps as _torch_cluster_fps
    _has_torch_cluster = True
except ImportError:
    _has_torch_cluster = False


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling for downsampling point clouds.

    Uses torch_cluster.fps (single fused CUDA kernel) when available,
    falling back to a Python loop implementation otherwise.

    Args:
        xyz: [B, N, 3] point cloud
        npoint: number of samples

    Returns:
        centroids: [B, npoint] sampled point indices
    """
    if not _has_torch_cluster:
        return _farthest_point_sample_python(xyz, npoint)

    B, N, _ = xyz.shape
    ratio = npoint / N

    # torch_cluster.fps expects [B*N, 3] with a batch vector [B*N]
    flat_xyz = xyz.reshape(B * N, -1)
    batch_vec = torch.arange(B, device=xyz.device).repeat_interleave(N)

    # fps returns flat indices into flat_xyz
    flat_idx = _torch_cluster_fps(flat_xyz, batch_vec, ratio=ratio, random_start=True)

    # Convert flat indices back to per-batch indices
    centroids = (flat_idx.reshape(B, npoint) % N)
    return centroids


def index_points(points, idx):
    """
    Index points based on indices.

    Args:
        points: [B, N, C]
        idx: [B, S] or [B, S, K]

    Returns:
        new_points: [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Find all points within radius from query points.

    Args:
        radius: local region radius
        nsample: max sample number in local region
        xyz: [B, N, 3] all points
        new_xyz: [B, S, 3] query points

    Returns:
        group_idx: [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Sample and group points.

    Args:
        npoint: number of centroids
        radius: ball query radius
        nsample: max number of samples per ball
        xyz: [B, N, 3] coordinates
        points: [B, N, D] point features

    Returns:
        new_xyz: [B, npoint, 3]
        new_points: [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint

    # Sample centroids
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)

    # Group points
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    Set Abstraction layer for hierarchical point cloud feature learning.
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Args:
            xyz: [B, N, 3] coordinates
            points: [B, N, D] features

        Returns:
            new_xyz: [B, npoint, 3]
            new_points: [B, npoint, mlp[-1]]
        """
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # new_points: [B, npoint, nsample, in_channel]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, in_channel, nsample, npoint]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Max pooling over nsample
        new_points = torch.max(new_points, 2)[0]  # [B, mlp[-1], npoint]
        new_points = new_points.permute(0, 2, 1)  # [B, npoint, mlp[-1]]

        return new_xyz, new_points


class GeometryNet(nn.Module):
    """
    Local geometry encoder for mesh chunks.
    Equivalent to SAM for images, extracts fine-grained geometric features.

    Architecture:
        - PointNet++ with 2 set abstraction layers
        - Multi-head attention for local context
        - Outputs 256-dimensional features per sampled point

    Input: coords [B, N, 3], normals [B, N, 3]
    Output: features [B, 128, 256]
    """

    def __init__(self):
        super().__init__()

        # Set abstraction layers (hierarchical sampling)
        # SA1: N -> 512 points, local region radius 0.2
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=6,  # xyz (3) + normals (3)
            mlp=[64, 64, 128]
        )

        # SA2: 512 -> 128 points, local region radius 0.4
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,  # previous features + xyz
            mlp=[128, 128, 256]
        )

        # Attention module for local context
        self.local_attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )

        # Layer norm
        self.norm = nn.LayerNorm(256)

    def forward(self, coords, normals):
        """
        Args:
            coords: [B, N, 3] vertex coordinates
            normals: [B, N, 3] vertex normals

        Returns:
            features: [B, 128, 256] - 128 sampled points with 256-dim features
        """
        # Concatenate coords + normals as input
        points = torch.cat([coords, normals], dim=-1)  # [B, N, 6]

        # Separate xyz from features
        xyz = coords  # [B, N, 3]
        features = normals  # [B, N, 3]

        # Hierarchical feature extraction
        xyz1, feat1 = self.sa1(xyz, features)  # [B, 512, 3], [B, 512, 128]
        xyz2, feat2 = self.sa2(xyz1, feat1)    # [B, 128, 3], [B, 128, 256]

        # Apply attention over local features
        feat2_attn, _ = self.local_attn(feat2, feat2, feat2)  # [B, 128, 256]

        # Add residual connection and normalize
        feat2 = self.norm(feat2 + feat2_attn)

        return feat2  # [B, 128, 256]


def build_geometry_net():
    """Build GeometryNet encoder."""
    return GeometryNet()
