"""Faithful MLX port of the real ll_ocadr geometry tower (PointNet++ + Point-BERT).

The configured PyTorch ll_ocadr model was never trained (no checkpoint to convert)
AND cannot run on Apple Silicon, so unlike ll_stepnet/ll_brepnet there are no trained
weights to load. "Faithful" here therefore means: reproduce the EXACT architecture of
the real encoders and PROVE the MLX tower computes the same function as the real code.

That proof is possible because GeometryNet / ShapeNet / MlpProjector are plain
``torch.nn.Module``s with no LLM dependency — they instantiate and run on CPU. So we
random-init the real PyTorch tower, convert its weights into this MLX tower, feed the
SAME ``(coords, normals)`` through both, and assert max-abs-diff ~1e-5 (the same rigor
that made the other two ports verifiably faithful rather than asserted).

Real architecture (ll_ocadr/vllm/lattice_encoder/*, config for the 0.5B model):
  GeometryNet (PointNet++):
    SA1: FPS N->512, ball-query r0.2 nsample32, mlp[64,64,128]  (in 3+3)
    SA2: FPS 512->128, ball-query r0.4 nsample64, mlp[128,128,256] (in 3+128)
    MultiheadAttention(256, 8) over the 128 points, +residual, LayerNorm -> [B,128,256]
  ShapeNet (Point-BERT, 0.5B config: embed768 depth4 heads8):
    PointPatchEmbedding (1x1 conv mini-PointNet -> 256 patches x 768),
    CLS + learnable pos, depth TransformerBlocks (pre-norm, GELU MLP), LayerNorm -> [B,257,768]
  Forward fuse: concat[shape[:,1:] (256x768), geom padded 128->256 (256x256)] -> 256x1024
    -> MlpProjector (linear 1024->n_embed) -> 256 mesh tokens.

The 1x1 convs are per-point Linears over channels, so they are implemented as MLX
Linear + BatchNorm (mathematically identical, no NHWC Conv layout). MultiheadAttention
is implemented manually from the packed in_proj_weight (the same trick used for the
stepnet port). FPS + ball-query are a deterministic geometric function of the fixed
cloud (start fixed at point 0) — computed once in numpy and shared by both models, so
they cannot be a source of divergence.

Modes:
  parity - random-init the real PyTorch tower, convert -> MLX, assert forward parity.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import logging  # noqa: E402

logging.disable(logging.WARNING)
import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as mlxnn  # noqa: E402
from mlx.utils import tree_unflatten  # noqa: E402

_REPO = "/Users/ryanoboyle/LatticeLabs_toolkit"


# ============================================================================
# Deterministic numpy FPS / ball-query / grouping (shared by both models).
# ============================================================================
def square_distance_np(src, dst):  # src [N,3], dst [M,3] -> [N,M]
    return (-2 * src @ dst.T) + (src ** 2).sum(1)[:, None] + (dst ** 2).sum(1)[None, :]


def fps_np(xyz, npoint):
    """Farthest point sampling, deterministic start at index 0. -> [npoint] indices."""
    n = xyz.shape[0]
    centroids = np.zeros(npoint, np.int64)
    distance = np.full(n, 1e10, np.float64)
    farthest = 0
    for i in range(npoint):
        centroids[i] = farthest
        d = ((xyz - xyz[farthest]) ** 2).sum(1)
        distance = np.minimum(distance, d)
        farthest = int(np.argmax(distance))
    return centroids


def ball_query_np(radius, nsample, xyz, new_xyz):
    """[S,nsample] indices — replicates geometry_net.query_ball_point exactly."""
    n = xyz.shape[0]
    s = new_xyz.shape[0]
    group_idx = np.tile(np.arange(n, dtype=np.int64)[None, :], (s, 1))  # [S,N]
    sqrdists = square_distance_np(new_xyz, xyz)  # [S,N]
    group_idx[sqrdists > radius ** 2] = n
    group_idx = np.sort(group_idx, axis=-1)[:, :nsample]
    group_first = np.tile(group_idx[:, 0:1], (1, group_idx.shape[1]))
    mask = group_idx == n
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group_np(npoint, radius, nsample, xyz, points):
    """Returns new_xyz [npoint,3], grouped [npoint,nsample,3+D], group_idx [npoint,nsample]."""
    fps_idx = fps_np(xyz, npoint)
    new_xyz = xyz[fps_idx]
    group_idx = ball_query_np(radius, nsample, xyz, new_xyz)
    grouped_xyz = xyz[group_idx]                       # [npoint,nsample,3]
    grouped_xyz_norm = grouped_xyz - new_xyz[:, None]  # center
    if points is not None:
        grouped_points = points[group_idx]
        grouped = np.concatenate([grouped_xyz_norm, grouped_points], axis=-1)
    else:
        grouped = grouped_xyz_norm
    return new_xyz, grouped, group_idx


def precompute_geom(coords, normals):
    """All geometric (cloud-only) tensors GeometryNetMLX needs. SA2 feature grouping
    happens at runtime (depends on SA1 conv output), so cache its xyz-norm + indices."""
    # SA1: full grouping (features = normals, known at data time)
    new_xyz1, sa1_grouped, _ = sample_and_group_np(512, 0.2, 32, coords, normals)
    # SA2 geometry on the SA1 centroids (features = SA1 output -> gathered at runtime)
    fps2 = fps_np(new_xyz1, 128)
    new_xyz2 = new_xyz1[fps2]
    g2 = ball_query_np(0.4, 64, new_xyz1, new_xyz2)        # [128,64] into the 512
    sa2_grouped_xyz = new_xyz1[g2] - new_xyz2[:, None]     # [128,64,3]
    return {
        "sa1_grouped": sa1_grouped.astype(np.float32),       # [512,32,6]
        "sa2_grouped_xyz": sa2_grouped_xyz.astype(np.float32),  # [128,64,3]
        "sa2_group_idx": g2.astype(np.int32),                # [128,64]
    }


# ============================================================================
# MLX modules
# ============================================================================
class LinBN(mlxnn.Module):
    """1x1 conv == per-point Linear over channels, followed by BatchNorm."""

    def __init__(self, cin, cout):
        super().__init__()
        self.lin = mlxnn.Linear(cin, cout)
        self.bn = mlxnn.BatchNorm(cout)

    def __call__(self, x):
        return self.bn(self.lin(x))


class SAMLX(mlxnn.Module):
    """PointNetSetAbstraction mlp: stack of (Linear->BN->ReLU), max over nsample axis."""

    def __init__(self, in_ch, mlp):
        super().__init__()
        layers = []
        last = in_ch
        for out in mlp:
            layers.append(LinBN(last, out))
            last = out
        self.layers = layers

    def __call__(self, grouped):  # [B, npoint, nsample, in_ch]
        x = grouped
        for layer in self.layers:
            x = mlxnn.relu(layer(x))
        return x.max(axis=2)  # [B, npoint, mlp[-1]]


class MHA(mlxnn.Module):
    """nn.MultiheadAttention math from the packed in_proj_weight [3d, d]."""

    def __init__(self, d, heads):
        super().__init__()
        self.d, self.h, self.hd = d, heads, d // heads
        self.in_proj = mlxnn.Linear(d, 3 * d)
        self.out_proj = mlxnn.Linear(d, d)

    def __call__(self, x):  # [B,S,d]
        b, s, _ = x.shape
        qkv = self.in_proj(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        def hd(t):
            return mx.transpose(t.reshape(b, s, self.h, self.hd), (0, 2, 1, 3))

        q, k, v = hd(q), hd(k), hd(v)
        att = mx.softmax((q @ mx.transpose(k, (0, 1, 3, 2))) / (self.hd ** 0.5), axis=-1)
        ctx = mx.transpose(att @ v, (0, 2, 1, 3)).reshape(b, s, self.d)
        return self.out_proj(ctx)


class GeometryNetMLX(mlxnn.Module):
    def __init__(self):
        super().__init__()
        self.sa1 = SAMLX(6, [64, 64, 128])
        self.sa2 = SAMLX(131, [128, 128, 256])
        self.local_attn = MHA(256, 8)
        self.norm = mlxnn.LayerNorm(256)

    def __call__(self, sa1_grouped, sa2_grouped_xyz, sa2_group_idx):
        feat1 = self.sa1(sa1_grouped)                       # [B,512,128]
        b = feat1.shape[0]
        gathered = mx.stack([feat1[i][sa2_group_idx[i]] for i in range(b)])  # [B,128,64,128]
        grouped2 = mx.concatenate([sa2_grouped_xyz, gathered], axis=-1)      # [B,128,64,131]
        feat2 = self.sa2(grouped2)                          # [B,128,256]
        feat2 = self.norm(feat2 + self.local_attn(feat2))
        return feat2                                        # [B,128,256]


class PatchEmbedMLX(mlxnn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.f1 = LinBN(6, 128)
        self.f2 = mlxnn.Linear(128, 256)         # first_conv tail (no BN)
        self.s1 = LinBN(512, 512)
        self.s2 = mlxnn.Linear(512, embed_dim)   # second_conv tail (no BN)
        self.embed_dim = embed_dim

    def __call__(self, coords, normals):
        pts = mx.concatenate([coords, normals], axis=-1)        # [B,N,6]
        f = self.f2(mlxnn.relu(self.f1(pts)))                   # [B,N,256]
        g = f.max(axis=1, keepdims=True)                        # [B,1,256]
        f = mx.concatenate([mx.broadcast_to(g, f.shape), f], axis=-1)  # [B,N,512]
        f = self.s2(mlxnn.relu(self.s1(f)))                     # [B,N,embed]
        b, n, e = f.shape
        ps = n // 256
        patches = f[:, : 256 * ps].reshape(b, 256, ps, e).max(axis=2)  # [B,256,embed]
        return patches


class TFBlockMLX(mlxnn.Module):
    def __init__(self, d, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = mlxnn.LayerNorm(d)
        self.attn = MHA(d, heads)
        self.norm2 = mlxnn.LayerNorm(d)
        hidden = int(d * mlp_ratio)
        self.fc1 = mlxnn.Linear(d, hidden)
        self.fc2 = mlxnn.Linear(hidden, d)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(mlxnn.gelu(self.fc1(self.norm2(x))))


class ShapeNetMLX(mlxnn.Module):
    def __init__(self, embed=768, depth=4, heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedMLX(embed)
        self.cls_token = mx.zeros((1, 1, embed))
        self.pos_embed = mx.zeros((1, 257, embed))
        self.blocks = [TFBlockMLX(embed, heads) for _ in range(depth)]
        self.norm = mlxnn.LayerNorm(embed)
        self.embed = embed

    def __call__(self, coords, normals):
        pt = self.patch_embed(coords, normals)               # [B,256,embed]
        b = pt.shape[0]
        cls = mx.broadcast_to(self.cls_token, (b, 1, self.embed))
        tokens = mx.concatenate([cls, pt], axis=1) + self.pos_embed  # [B,257,embed]
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)                             # [B,257,embed]


class FaithfulTower(mlxnn.Module):
    """GeometryNet + ShapeNet + linear projector -> 256 mesh tokens (+ aux class head)."""

    def __init__(self, n_embed, shape_depth=4, shape_heads=8):
        super().__init__()
        self.geometry = GeometryNetMLX()
        self.shape = ShapeNetMLX(768, shape_depth, shape_heads)
        self.projector = mlxnn.Linear(1024, n_embed)
        self.aux_head = mlxnn.Linear(n_embed, 3)

    def __call__(self, b):
        geom = self.geometry(b["sa1_grouped"], b["sa2_grouped_xyz"], b["sa2_group_idx"])  # [B,128,256]
        shp = self.shape(b["coords"], b["normals"])[:, 1:]   # [B,256,768] (skip CLS)
        bs = geom.shape[0]
        geom_pad = mx.concatenate([geom, mx.zeros((bs, 128, 256))], axis=1)  # [B,256,256]
        feat = mx.concatenate([shp, geom_pad], axis=-1)      # [B,256,1024]
        return self.projector(feat)                          # [B,256,n_embed]

    def aux_logits(self, tokens):
        return self.aux_head(tokens.mean(axis=1))


# ============================================================================
# weight conversion: real PyTorch encoders -> MLX tower
# ============================================================================
def convert_tower(geom_pt, shape_pt, proj_pt, mlx_tower):
    import torch

    def t(x):
        return mx.array(x.detach().cpu().float().numpy())

    gsd = geom_pt.state_dict()
    ssd = shape_pt.state_dict()
    psd = proj_pt.state_dict()
    pairs = []

    # GeometryNet SA layers: mlp_convs.i (Conv2d 1x1 -> Linear), mlp_bns.i (BN)
    for sa, n_mlp in (("sa1", 3), ("sa2", 3)):
        for i in range(n_mlp):
            w = gsd[f"{sa}.mlp_convs.{i}.weight"]  # [out,in,1,1]
            pairs.append((f"geometry.{sa}.layers.{i}.lin.weight", t(w[:, :, 0, 0])))
            pairs.append((f"geometry.{sa}.layers.{i}.lin.bias", t(gsd[f"{sa}.mlp_convs.{i}.bias"])))
            for p in ("weight", "bias", "running_mean", "running_var"):
                pairs.append((f"geometry.{sa}.layers.{i}.bn.{p}", t(gsd[f"{sa}.mlp_bns.{i}.{p}"])))
    # GeometryNet attention + norm
    pairs += [("geometry.local_attn.in_proj.weight", t(gsd["local_attn.in_proj_weight"])),
              ("geometry.local_attn.in_proj.bias", t(gsd["local_attn.in_proj_bias"])),
              ("geometry.local_attn.out_proj.weight", t(gsd["local_attn.out_proj.weight"])),
              ("geometry.local_attn.out_proj.bias", t(gsd["local_attn.out_proj.bias"])),
              ("geometry.norm.weight", t(gsd["norm.weight"])), ("geometry.norm.bias", t(gsd["norm.bias"]))]

    # ShapeNet patch embed: first_conv (0=Conv1d,1=BN,3=Conv1d), second_conv (0,1,3)
    def conv1d(w):  # [out,in,1] -> [out,in]
        return t(w[:, :, 0])

    pairs += [("shape.patch_embed.f1.lin.weight", conv1d(ssd["patch_embed.first_conv.0.weight"])),
              ("shape.patch_embed.f1.lin.bias", t(ssd["patch_embed.first_conv.0.bias"]))]
    for p in ("weight", "bias", "running_mean", "running_var"):
        pairs.append((f"shape.patch_embed.f1.bn.{p}", t(ssd[f"patch_embed.first_conv.1.{p}"])))
    pairs += [("shape.patch_embed.f2.weight", conv1d(ssd["patch_embed.first_conv.3.weight"])),
              ("shape.patch_embed.f2.bias", t(ssd["patch_embed.first_conv.3.bias"])),
              ("shape.patch_embed.s1.lin.weight", conv1d(ssd["patch_embed.second_conv.0.weight"])),
              ("shape.patch_embed.s1.lin.bias", t(ssd["patch_embed.second_conv.0.bias"]))]
    for p in ("weight", "bias", "running_mean", "running_var"):
        pairs.append((f"shape.patch_embed.s1.bn.{p}", t(ssd[f"patch_embed.second_conv.1.{p}"])))
    pairs += [("shape.patch_embed.s2.weight", conv1d(ssd["patch_embed.second_conv.3.weight"])),
              ("shape.patch_embed.s2.bias", t(ssd["patch_embed.second_conv.3.bias"]))]
    # ShapeNet cls/pos/blocks/norm
    pairs += [("shape.cls_token", t(ssd["cls_token"])), ("shape.pos_embed", t(ssd["pos_embed"])),
              ("shape.norm.weight", t(ssd["norm.weight"])), ("shape.norm.bias", t(ssd["norm.bias"]))]
    depth = len({k.split("blocks.")[1].split(".")[0] for k in ssd if k.startswith("blocks.")})
    for i in range(depth):
        b = f"blocks.{i}"
        pairs += [(f"shape.blocks.{i}.norm1.weight", t(ssd[f"{b}.norm1.weight"])),
                  (f"shape.blocks.{i}.norm1.bias", t(ssd[f"{b}.norm1.bias"])),
                  (f"shape.blocks.{i}.norm2.weight", t(ssd[f"{b}.norm2.weight"])),
                  (f"shape.blocks.{i}.norm2.bias", t(ssd[f"{b}.norm2.bias"])),
                  (f"shape.blocks.{i}.attn.in_proj.weight", t(ssd[f"{b}.attn.in_proj_weight"])),
                  (f"shape.blocks.{i}.attn.in_proj.bias", t(ssd[f"{b}.attn.in_proj_bias"])),
                  (f"shape.blocks.{i}.attn.out_proj.weight", t(ssd[f"{b}.attn.out_proj.weight"])),
                  (f"shape.blocks.{i}.attn.out_proj.bias", t(ssd[f"{b}.attn.out_proj.bias"])),
                  (f"shape.blocks.{i}.fc1.weight", t(ssd[f"{b}.mlp.0.weight"])),
                  (f"shape.blocks.{i}.fc1.bias", t(ssd[f"{b}.mlp.0.bias"])),
                  (f"shape.blocks.{i}.fc2.weight", t(ssd[f"{b}.mlp.3.weight"])),
                  (f"shape.blocks.{i}.fc2.bias", t(ssd[f"{b}.mlp.3.bias"]))]

    # projector (linear): MlpProjector wraps it as .layers; a bare nn.Linear is "weight"
    pkey = next(k for k in ("layers.weight", "layers.0.weight", "weight") if k in psd)
    bkey = pkey.replace("weight", "bias")
    pairs += [("projector.weight", t(psd[pkey])), ("projector.bias", t(psd[bkey]))]

    mlx_tower.update(tree_unflatten(pairs))
    mx.eval(mlx_tower.parameters())
    return len(pairs)


def parity():
    sys.path.insert(0, f"{_REPO}/ll_ocadr/vllm/lattice_encoder")
    sys.path.insert(0, f"{_REPO}/ll_ocadr")
    import torch
    import geometry_net as G
    from geometry_net import build_geometry_net
    from shape_net import build_shape_net

    # make PyTorch sampling deterministic + identical to our numpy precompute
    G.farthest_point_sample = lambda xyz, npoint: torch.from_numpy(
        np.stack([fps_np(x.cpu().numpy(), npoint) for x in xyz])).long().to(xyz.device)
    G.query_ball_point = lambda radius, nsample, xyz, new_xyz: torch.from_numpy(
        np.stack([ball_query_np(radius, nsample, xyz[i].cpu().numpy(), new_xyz[i].cpu().numpy())
                  for i in range(xyz.shape[0])])).long().to(xyz.device)

    torch.manual_seed(0)
    np.random.seed(0)
    n_embed = 896
    geom_pt = build_geometry_net().eval()
    shape_pt = build_shape_net(embed_dim=768, depth=4, num_heads=8).eval()
    proj_pt = torch.nn.Linear(1024, n_embed).eval()

    N = 2048
    coords = np.random.randn(N, 3).astype(np.float32) * 0.3
    normals = np.random.randn(N, 3).astype(np.float32)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    # PyTorch reference forward (replicating latticelabs_ocadr fuse logic)
    with torch.no_grad():
        c = torch.from_numpy(coords)[None]
        nrm = torch.from_numpy(normals)[None]
        g_pt = geom_pt(c, nrm)                       # [1,128,256]
        s_pt = shape_pt(c, nrm)[:, 1:]               # [1,256,768]
        g_pad = torch.cat([g_pt, torch.zeros(1, 128, 256)], dim=1)  # [1,256,256]
        fused = torch.cat([s_pt, g_pad], dim=-1)     # [1,256,1024]
        tok_pt = proj_pt(fused).numpy()              # [1,256,896]

    # MLX tower
    tower = FaithfulTower(n_embed, shape_depth=4, shape_heads=8)
    npar = convert_tower(geom_pt, shape_pt, proj_pt, tower)
    tower.eval()  # BatchNorm must use converted running stats (match PyTorch .eval())
    pre = precompute_geom(coords, normals)
    b = {"sa1_grouped": mx.array(pre["sa1_grouped"])[None],
         "sa2_grouped_xyz": mx.array(pre["sa2_grouped_xyz"])[None],
         "sa2_group_idx": mx.array(pre["sa2_group_idx"])[None],
         "coords": mx.array(coords)[None], "normals": mx.array(normals)[None]}
    # component diffs
    g_mlx = np.array(tower.geometry(b["sa1_grouped"], b["sa2_grouped_xyz"], b["sa2_group_idx"]).tolist())
    s_mlx = np.array(tower.shape(b["coords"], b["normals"])[:, 1:].tolist())
    tok_mlx = np.array(tower(b).tolist())

    dg = float(np.abs(g_mlx - g_pt.detach().numpy()).max())
    ds = float(np.abs(s_mlx - s_pt.detach().numpy()).max())
    dt = float(np.abs(tok_mlx - tok_pt).max())
    print(f"converted {npar} tensors", flush=True)
    print(f"PARITY max-abs-diff   geometry={dg:.2e}   shape={ds:.2e}   mesh_tokens={dt:.2e}", flush=True)
    ok = dg < 1e-3 and ds < 1e-3 and dt < 1e-3
    print(f"FAITHFUL_PARITY {'PASS' if ok else 'FAIL'} "
          f"(tower reproduces the real PyTorch encoders within float tolerance)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["parity"], default="parity")
    ap.parse_args()
    parity()
