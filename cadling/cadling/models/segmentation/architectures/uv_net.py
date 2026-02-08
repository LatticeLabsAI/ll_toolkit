"""UV-Net face encoder for B-Rep surfaces.

Implements UV parameter grid sampling + surface CNN for generating
per-face embeddings from B-Rep geometry. Based on the UV-Net architecture
for learning on CAD models via UV-map representations.

The pipeline:
1. UVGridSampler: Evaluates B-Rep faces on a regular UV grid to produce
   per-face (grid_size x grid_size x 7) tensors (xyz, normal, trim mask).
2. SurfaceCNN: Applies 2D convolutions over the UV grid to produce
   a fixed-dimensional face embedding.
3. UVNetEncoder: Combines SurfaceCNN face embeddings with graph message
   passing for context-aware per-face and graph-level embeddings.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)

# Lazy pythonocc availability check
_has_pythonocc = False
try:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepTools import breptools
    from OCC.Core.GeomAbs import (
        GeomAbs_BezierSurface,
        GeomAbs_BSplineSurface,
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Plane,
        GeomAbs_Sphere,
        GeomAbs_SurfaceOfExtrusion,
        GeomAbs_SurfaceOfRevolution,
        GeomAbs_Torus,
    )
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.gp import gp_Pnt, gp_Vec
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
    from OCC.Core.BRep import BRep_Tool

    _has_pythonocc = True
    _log.debug("pythonocc available for UV-Net sampling")
except ImportError:
    _log.debug("pythonocc not available, UV-Net will use placeholder grids")


class UVGridSampler:
    """Sample B-Rep faces on a regular UV parameter grid.

    For each face, evaluates the underlying surface at a grid of UV
    parameter values to produce a (grid_size, grid_size, 7) tensor:
      - channels 0-2: xyz position
      - channels 3-5: surface normal
      - channel 6: trim mask (1 if point is inside face boundary, 0 otherwise)

    When pythonocc is unavailable, generates a deterministic placeholder grid
    based on face index so that downstream modules can still be tested.

    Args:
        grid_size: Number of sample points per UV dimension (default 10).
    """

    def __init__(self, grid_size: int = 10) -> None:
        self.grid_size = grid_size

    def sample_face(
        self,
        face,
        grid_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample a single B-Rep face on a UV grid.

        Args:
            face: A TopoDS_Face (pythonocc) or an index/placeholder.
            grid_size: Override for the default grid resolution.

        Returns:
            Tensor of shape (grid_size, grid_size, 7).
        """
        gs = grid_size or self.grid_size

        if _has_pythonocc and self._is_topods_face(face):
            return self._sample_face_occ(face, gs)

        return self._sample_face_placeholder(face, gs)

    def sample_faces(
        self,
        faces: List,
        grid_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample multiple faces and stack into a batch tensor.

        Args:
            faces: List of TopoDS_Face objects or placeholder indices.
            grid_size: Override for the default grid resolution.

        Returns:
            Tensor of shape (num_faces, grid_size, grid_size, 7).
        """
        gs = grid_size or self.grid_size
        grids = [self.sample_face(f, gs) for f in faces]
        return torch.stack(grids, dim=0)

    @staticmethod
    def _is_topods_face(face) -> bool:
        """Check whether *face* is an OCC TopoDS_Face."""
        try:
            from OCC.Core.TopoDS import TopoDS_Face

            return isinstance(face, TopoDS_Face)
        except Exception:
            return False

    def _sample_face_occ(self, face, grid_size: int) -> torch.Tensor:
        """Evaluate the surface + normal at a UV grid using pythonocc.

        Args:
            face: TopoDS_Face object.
            grid_size: Number of sample points per UV axis.

        Returns:
            Tensor of shape (grid_size, grid_size, 7).
        """
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.gp import gp_Pnt, gp_Vec
        from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
        from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON

        adaptor = BRepAdaptor_Surface(face)
        u_min, u_max, v_min, v_max = (
            adaptor.FirstUParameter(),
            adaptor.LastUParameter(),
            adaptor.FirstVParameter(),
            adaptor.LastVParameter(),
        )

        grid = torch.zeros(grid_size, grid_size, 7, dtype=torch.float32)

        tol = 1e-6
        try:
            classifier = BRepTopAdaptor_FClass2d(face, tol)
        except Exception:
            classifier = None

        for i in range(grid_size):
            u = u_min + (u_max - u_min) * i / max(grid_size - 1, 1)
            for j in range(grid_size):
                v = v_min + (v_max - v_min) * j / max(grid_size - 1, 1)

                pnt = gp_Pnt()
                adaptor.D0(u, v, pnt)
                grid[i, j, 0] = pnt.X()
                grid[i, j, 1] = pnt.Y()
                grid[i, j, 2] = pnt.Z()

                try:
                    du = gp_Vec()
                    dv = gp_Vec()
                    pnt2 = gp_Pnt()
                    adaptor.D1(u, v, pnt2, du, dv)
                    normal = du.Crossed(dv)
                    mag = normal.Magnitude()
                    if mag > 1e-10:
                        normal.Divide(mag)
                    grid[i, j, 3] = normal.X()
                    grid[i, j, 4] = normal.Y()
                    grid[i, j, 5] = normal.Z()
                except Exception:
                    grid[i, j, 3:6] = 0.0

                if classifier is not None:
                    try:
                        from OCC.Core.gp import gp_Pnt2d

                        state = classifier.Perform(gp_Pnt2d(u, v))
                        grid[i, j, 6] = 1.0 if state in (TopAbs_IN, TopAbs_ON) else 0.0
                    except Exception:
                        grid[i, j, 6] = 1.0
                else:
                    grid[i, j, 6] = 1.0

        return grid

    def _sample_face_placeholder(self, face, grid_size: int) -> torch.Tensor:
        """Generate a deterministic placeholder UV grid.

        Uses a simple parametric surface (scaled by a seed derived from
        *face*) so the output is reproducible and non-zero.

        Args:
            face: Any hashable value (typically an int index).
            grid_size: Number of sample points per UV axis.

        Returns:
            Tensor of shape (grid_size, grid_size, 7).
        """
        seed = hash(face) % 10000
        rng = np.random.RandomState(seed)

        grid = np.zeros((grid_size, grid_size, 7), dtype=np.float32)

        amplitude = 0.1 + 0.9 * rng.random()
        offset = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)

        for i in range(grid_size):
            u = i / max(grid_size - 1, 1) * math.pi
            for j in range(grid_size):
                v = j / max(grid_size - 1, 1) * math.pi

                grid[i, j, 0] = offset[0] + math.cos(u) * math.cos(v) * amplitude
                grid[i, j, 1] = offset[1] + math.sin(u) * math.cos(v) * amplitude
                grid[i, j, 2] = offset[2] + math.sin(v) * amplitude

                nx = math.cos(u) * math.cos(v)
                ny = math.sin(u) * math.cos(v)
                nz = math.sin(v)
                mag = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-10
                grid[i, j, 3] = nx / mag
                grid[i, j, 4] = ny / mag
                grid[i, j, 5] = nz / mag

                grid[i, j, 6] = 1.0

        return torch.from_numpy(grid)


class SurfaceCNN(nn.Module):
    """2D CNN that encodes a UV-grid sample into a face embedding.

    Architecture:
        Conv2d(7 -> 32, k=3, pad=1) -> BN -> ReLU
        Conv2d(32 -> 64, k=3, pad=1) -> BN -> ReLU
        Conv2d(64 -> 64, k=3, pad=1) -> BN -> ReLU
        Global Average Pooling -> 64-dim face embedding

    Args:
        in_channels: Number of input channels (default 7: xyz + normal + trim).
        face_embed_dim: Output embedding dimension (default 64).
    """

    def __init__(
        self,
        in_channels: int = 7,
        face_embed_dim: int = 64,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.face_embed_dim = face_embed_dim

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, face_embed_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(face_embed_dim)

    def forward(self, uv_grid: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            uv_grid: [B, 7, H, W] UV-grid feature maps.

        Returns:
            [B, face_embed_dim] face embeddings.
        """
        x = F.relu(self.bn1(self.conv1(uv_grid)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global average pooling: [B, C, H, W] -> [B, C]
        x = x.mean(dim=[2, 3])

        return x


class _GraphAttentionLayer(nn.Module):
    """Single graph attention layer (no torch_geometric dependency).

    Implements multi-head attention over graph edges using scatter ops.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension per head.
        num_heads: Number of attention heads.
        edge_dim: Optional edge feature dimension.
        dropout: Dropout probability.
        concat: If True, concatenate heads; otherwise average.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 1,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1,
        concat: bool = True,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat

        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        attn_in_dim = 2 * out_dim + (edge_dim if edge_dim else 0)
        self.attn = nn.Parameter(torch.empty(num_heads, attn_in_dim))
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))

        self.edge_proj = nn.Linear(edge_dim, edge_dim) if edge_dim else None

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [N, in_dim] node features.
            edge_index: [2, E] edge indices.
            edge_attr: [E, edge_dim] edge features (optional).

        Returns:
            [N, out_dim * num_heads] if concat else [N, out_dim].
        """
        N = x.size(0)
        H = self.num_heads
        D = self.out_dim

        Wh = self.W(x).view(N, H, D)

        src, dst = edge_index[0], edge_index[1]

        Wh_i = Wh[src]
        Wh_j = Wh[dst]

        attn_input = torch.cat([Wh_i, Wh_j], dim=-1)

        if edge_attr is not None and self.edge_proj is not None:
            edge_feat = self.edge_proj(edge_attr)
            edge_feat = edge_feat.unsqueeze(1).expand(-1, H, -1)
            attn_input = torch.cat([attn_input, edge_feat], dim=-1)

        e = (attn_input * self.attn.unsqueeze(0)).sum(dim=-1)
        e = self.leaky_relu(e)

        # Scatter-based softmax for numerical stability
        e_max = torch.zeros(N, H, device=x.device, dtype=x.dtype)
        e_max.scatter_reduce_(
            0, dst.unsqueeze(-1).expand(-1, H), e, reduce="amax", include_self=True
        )
        e_stable = e - e_max[dst]
        e_exp = torch.exp(e_stable)

        e_sum = torch.zeros(N, H, device=x.device, dtype=x.dtype)
        e_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, H), e_exp)
        e_sum = e_sum.clamp(min=1e-10)

        alpha = e_exp / e_sum[dst]
        alpha = self.dropout(alpha)

        weighted = Wh_j * alpha.unsqueeze(-1)

        out = torch.zeros(N, H, D, device=x.device, dtype=x.dtype)
        out.scatter_add_(
            0,
            dst.unsqueeze(-1).unsqueeze(-1).expand(-1, H, D),
            weighted,
        )

        if self.concat:
            out = out.view(N, H * D)
        else:
            out = out.mean(dim=1)

        return out


class _AttentionPooling(nn.Module):
    """Attention-based graph pooling to produce a fixed-size graph embedding.

    Args:
        embed_dim: Node embedding dimension.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool node embeddings into a single graph embedding.

        Args:
            x: [N, D] node embeddings.

        Returns:
            [D] graph-level embedding.
        """
        attn_weights = F.softmax(self.gate(x), dim=0)
        graph_embed = (attn_weights * x).sum(dim=0)
        return graph_embed


class UVNetEncoder(nn.Module):
    """Complete UV-Net encoder for B-Rep models.

    Combines SurfaceCNN face embeddings with graph-level message passing
    to produce context-aware per-face embeddings and a global graph embedding.

    Args:
        grid_size: UV grid resolution per face.
        face_embed_dim: Face embedding dimension from SurfaceCNN (default 64).
        graph_embed_dim: Output graph/face embedding dimension (default 128).
        num_gnn_layers: Number of graph attention layers (default 4).
        num_heads: Number of attention heads per GNN layer (default 8).
        edge_dim: Optional edge feature dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        grid_size: int = 10,
        face_embed_dim: int = 64,
        graph_embed_dim: int = 128,
        num_gnn_layers: int = 4,
        num_heads: int = 8,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.grid_size = grid_size
        self.face_embed_dim = face_embed_dim
        self.graph_embed_dim = graph_embed_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_heads = num_heads

        self.sampler = UVGridSampler(grid_size=grid_size)

        self.surface_cnn = SurfaceCNN(
            in_channels=7,
            face_embed_dim=face_embed_dim,
        )

        gnn_dim = graph_embed_dim
        self.face_proj = nn.Sequential(
            nn.Linear(face_embed_dim, gnn_dim),
            nn.LayerNorm(gnn_dim),
            nn.ReLU(),
        )

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        for layer_idx in range(num_gnn_layers):
            concat = layer_idx < num_gnn_layers - 1

            if concat:
                head_dim = max(gnn_dim // num_heads, 1)
                layer = _GraphAttentionLayer(
                    in_dim=gnn_dim,
                    out_dim=head_dim,
                    num_heads=num_heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=True,
                )
                out_dim = head_dim * num_heads
            else:
                layer = _GraphAttentionLayer(
                    in_dim=gnn_dim,
                    out_dim=gnn_dim,
                    num_heads=num_heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=False,
                )
                out_dim = gnn_dim

            self.gnn_layers.append(layer)
            self.gnn_norms.append(nn.LayerNorm(out_dim))

        self.attn_pool = _AttentionPooling(graph_embed_dim)

        _log.info(
            f"UVNetEncoder: grid={grid_size}, face_dim={face_embed_dim}, "
            f"graph_dim={graph_embed_dim}, gnn_layers={num_gnn_layers}, "
            f"heads={num_heads}"
        )

    def forward(
        self,
        face_grids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            face_grids: [N, 7, H, W] UV-grid features for each face.
                N = number of faces, H = W = grid_size.
                Channel order: xyz(3) + normal(3) + trim(1).
            edge_index: [2, E] face adjacency edges.
            edge_attr: [E, edge_dim] optional edge features.

        Returns:
            per_face_embeddings: [N, graph_embed_dim] per-face embeddings.
            graph_embedding: [graph_embed_dim] global graph embedding.
        """
        face_embeds = self.surface_cnn(face_grids)
        x = self.face_proj(face_embeds)

        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            residual = x
            x = gnn_layer(x, edge_index, edge_attr)
            x = norm(x)

            if residual.size(-1) == x.size(-1):
                x = x + residual

            if i < len(self.gnn_layers) - 1:
                x = F.elu(x)

        per_face_embeddings = x
        graph_embedding = self.attn_pool(per_face_embeddings)

        return per_face_embeddings, graph_embedding
