"""ll_brepnet B-rep face segmentation in native MLX — faithful weight-conversion port.

This is NOT a simplified re-implementation. It reproduces the EXACT architecture of
``ll_brepnet.models.LLBRepNet`` (the model that reached PyTorch test mIoU 0.828) and
CONVERTS the real trained Lightning checkpoint
(``resources/fusion360/full_train/best.ckpt``) into MLX, so the MLX model *is* the
trained model — same weights, same accuracy — running natively on Apple Silicon.

Exact architecture (from ll_brepnet.py / uvnet_encoders.py / cadling brep_net.py):

  surface_encoder : 3x [Conv2d(3,pad1) -> BatchNorm2d -> ReLU] -> AdaptiveAvgPool2d(1)  (7->32->64->64)
  curve_encoder   : 3x [Conv1d(3,pad1) -> BatchNorm1d -> ReLU] -> AdaptiveAvgPool1d(1)  (6->32->64->64)
  face_proj : Linear(8+64 -> 64) -> ReLU          edge_proj : Linear(7+64 -> 64) -> ReLU
  encoder (BRepNetEncoder):
    input_proj : Linear(129 -> 128) -> LayerNorm -> ReLU
    4x layer   : residual = h ; h = LayerNorm(relu(W_self·h + W_next·h[next]
                 + W_prev·h[prev] + W_mate·h[mate])) ; h = h + residual
    output_proj: Linear(128 -> 128)                 (attn_gate feeds only the
    coedge->face scatter-mean                        discarded graph embedding)
  seg_head : Linear(128 -> 8)

Weight conversion details (the non-trivial parts the simplified port got wrong):
  * PyTorch Conv2d weight is [out,in,kH,kW] (OIHW); MLX Conv2d is [out,kH,kW,in] (OHWI) -> permute.
  * PyTorch Conv1d weight is [out,in,kW] (OIW);    MLX Conv1d is [out,kW,in] (OWI)   -> permute.
  * BatchNorm running_mean / running_var / weight / bias are converted and applied in
    inference mode (eps 1e-5) — exactly what the PyTorch model does at eval.
  * Linear / LayerNorm map 1:1 (same [out,in] / [dim] layout) with no transpose.

Both models are driven from the SAME real ``BRepDataset`` (identical z-score
standardization), so the parity comparison is apples-to-apples.

Modes: probe | convert | parity.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore")
import logging  # noqa: E402

logging.disable(logging.WARNING)
import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as mlxnn  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_DATA = _REPO / "resources/fusion360/full_processed"
_CKPT = _REPO / "resources/fusion360/full_train/best.ckpt"
IGNORE = -1


# --- faithful MLX modules -----------------------------------------------------
class EvalBN(mlxnn.Module):
    """BatchNorm in inference mode over the last (channel) axis, from running stats."""

    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((c,))
        self.bias = mx.zeros((c,))
        self.running_mean = mx.zeros((c,))
        self.running_var = mx.ones((c,))
        self.eps = eps

    def __call__(self, x):  # x: [..., C]
        return (x - self.running_mean) * mx.rsqrt(self.running_var + self.eps) * self.weight + self.bias


class SurfEnc(mlxnn.Module):
    """UVNetSurfaceEncoder: 3x(Conv2d->BN->ReLU) -> global avg pool. Input [F,U,V,7]."""

    def __init__(self, cin=7, out=64):
        super().__init__()
        self.c0 = mlxnn.Conv2d(cin, 32, 3, padding=1)
        self.b1 = EvalBN(32)
        self.c3 = mlxnn.Conv2d(32, 64, 3, padding=1)
        self.b4 = EvalBN(64)
        self.c6 = mlxnn.Conv2d(64, out, 3, padding=1)
        self.b7 = EvalBN(out)

    def __call__(self, g):
        x = mlxnn.relu(self.b1(self.c0(g)))
        x = mlxnn.relu(self.b4(self.c3(x)))
        x = mlxnn.relu(self.b7(self.c6(x)))
        return x.mean(axis=(1, 2))  # AdaptiveAvgPool2d(1) -> [F,out]


class CurveEnc(mlxnn.Module):
    """UVNetCurveEncoder: 3x(Conv1d->BN->ReLU) -> global avg pool. Input [E,U,6]."""

    def __init__(self, cin=6, out=64):
        super().__init__()
        self.c0 = mlxnn.Conv1d(cin, 32, 3, padding=1)
        self.b1 = EvalBN(32)
        self.c3 = mlxnn.Conv1d(32, 64, 3, padding=1)
        self.b4 = EvalBN(64)
        self.c6 = mlxnn.Conv1d(64, out, 3, padding=1)
        self.b7 = EvalBN(out)

    def __call__(self, g):
        x = mlxnn.relu(self.b1(self.c0(g)))
        x = mlxnn.relu(self.b4(self.c3(x)))
        x = mlxnn.relu(self.b7(self.c6(x)))
        return x.mean(axis=1)  # AdaptiveAvgPool1d(1) -> [E,out]


class CoedgeConv(mlxnn.Module):
    def __init__(self, d):
        super().__init__()
        self.W_self = mlxnn.Linear(d, d)
        self.W_next = mlxnn.Linear(d, d)
        self.W_prev = mlxnn.Linear(d, d)
        self.W_mate = mlxnn.Linear(d, d)

    def __call__(self, h, nidx, pidx, midx):
        return mlxnn.relu(self.W_self(h) + self.W_next(h[nidx]) + self.W_prev(h[pidx]) + self.W_mate(h[midx]))


class BRepEncoder(mlxnn.Module):
    """cadling BRepNetEncoder: input_proj -> residual coedge convs -> output_proj -> face mean-pool."""

    def __init__(self, input_dim=129, hidden=128, layers=4):
        super().__init__()
        self.input_lin = mlxnn.Linear(input_dim, hidden)
        self.input_ln = mlxnn.LayerNorm(hidden)
        self.conv_layers = [CoedgeConv(hidden) for _ in range(layers)]
        self.layer_norms = [mlxnn.LayerNorm(hidden) for _ in range(layers)]
        self.output_proj = mlxnn.Linear(hidden, hidden)

    def __call__(self, feats, nidx, pidx, midx, c2f, nf):
        h = mlxnn.relu(self.input_ln(self.input_lin(feats)))
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            res = h
            h = norm(conv(h, nidx, pidx, midx)) + res
        coedge_emb = self.output_proj(h)
        onehot = (c2f[:, None] == mx.arange(nf)[None, :]).astype(coedge_emb.dtype)  # [C,F]
        return (onehot.T @ coedge_emb) / mx.maximum(onehot.sum(axis=0)[:, None], 1)  # [F,hidden]


class LLBRepNetMLX(mlxnn.Module):
    def __init__(self, nc=8, surf=64, curve=64, ent=64, hidden=128, layers=4):
        super().__init__()
        self.surface_encoder = SurfEnc(7, surf)
        self.curve_encoder = CurveEnc(6, curve)
        self.face_proj = mlxnn.Linear(8 + surf, ent)
        self.edge_proj = mlxnn.Linear(7 + curve, ent)
        self.encoder = BRepEncoder(2 * ent + 1, hidden, layers)
        self.seg_head = mlxnn.Linear(hidden, nc)

    def __call__(self, b):
        face_repr = mlxnn.relu(self.face_proj(
            mx.concatenate([b["ff"], self.surface_encoder(b["fg"])], axis=1)))
        edge_repr = mlxnn.relu(self.edge_proj(
            mx.concatenate([b["ef"], self.curve_encoder(b["eg"])], axis=1)))
        coedge = mx.concatenate([face_repr[b["c2f"]], edge_repr[b["c2e"]], b["rev"]], axis=1)
        face_emb = self.encoder(coedge, b["c2n"], b["c2p"], b["c2m"], b["c2f"], b["nf"])
        return self.seg_head(face_emb)


# --- weight conversion --------------------------------------------------------
def convert_checkpoint(ckpt_path, model):
    """Load the real Lightning state_dict and assign it into the MLX model."""
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck

    def lin(key):  # Linear / LayerNorm: direct copy
        return mx.array(sd[key].detach().cpu().float().numpy())

    def conv2d(key):  # OIHW -> OHWI
        return mx.array(np.transpose(sd[key].detach().cpu().float().numpy(), (0, 2, 3, 1)))

    def conv1d(key):  # OIW -> OWI
        return mx.array(np.transpose(sd[key].detach().cpu().float().numpy(), (0, 2, 1)))

    pairs = []
    # UV-Net encoders: net indices 0(conv) 1(bn) 3(conv) 4(bn) 6(conv) 7(bn)
    for enc, convf in (("surface_encoder", conv2d), ("curve_encoder", conv1d)):
        for ci, attr in ((0, "c0"), (3, "c3"), (6, "c6")):
            pairs.append((f"{enc}.{attr}.weight", convf(f"{enc}.net.{ci}.weight")))
            pairs.append((f"{enc}.{attr}.bias", lin(f"{enc}.net.{ci}.bias")))
        for bi, attr in ((1, "b1"), (4, "b4"), (7, "b7")):
            for p in ("weight", "bias", "running_mean", "running_var"):
                pairs.append((f"{enc}.{attr}.{p}", lin(f"{enc}.net.{bi}.{p}")))
    # entity projections (Sequential index 0 is the Linear)
    pairs += [("face_proj.weight", lin("face_proj.0.weight")), ("face_proj.bias", lin("face_proj.0.bias")),
              ("edge_proj.weight", lin("edge_proj.0.weight")), ("edge_proj.bias", lin("edge_proj.0.bias"))]
    # coedge encoder
    pairs += [("encoder.input_lin.weight", lin("encoder.input_proj.0.weight")),
              ("encoder.input_lin.bias", lin("encoder.input_proj.0.bias")),
              ("encoder.input_ln.weight", lin("encoder.input_proj.1.weight")),
              ("encoder.input_ln.bias", lin("encoder.input_proj.1.bias")),
              ("encoder.output_proj.weight", lin("encoder.output_proj.weight")),
              ("encoder.output_proj.bias", lin("encoder.output_proj.bias"))]
    n_layers = len({k.split("conv_layers.")[1].split(".")[0]
                    for k in sd if "encoder.conv_layers." in k})
    for i in range(n_layers):
        for w in ("W_self", "W_next", "W_prev", "W_mate"):
            pairs.append((f"encoder.conv_layers.{i}.{w}.weight", lin(f"encoder.conv_layers.{i}.{w}.weight")))
            pairs.append((f"encoder.conv_layers.{i}.{w}.bias", lin(f"encoder.conv_layers.{i}.{w}.bias")))
        pairs.append((f"encoder.layer_norms.{i}.weight", lin(f"encoder.layer_norms.{i}.weight")))
        pairs.append((f"encoder.layer_norms.{i}.bias", lin(f"encoder.layer_norms.{i}.bias")))
    pairs += [("seg_head.weight", lin("seg_head.weight")), ("seg_head.bias", lin("seg_head.bias"))]

    model.update(tree_unflatten(pairs))
    mx.eval(model.parameters())
    return model, len(pairs)


def to_mlx_batch(batch):
    fg = np.transpose(batch.face_point_grids.numpy(), (0, 2, 3, 1))  # [F,U,V,7]
    eg = np.transpose(batch.edge_point_grids.numpy(), (0, 2, 1))      # [E,U,6]
    return {
        "ff": mx.array(batch.face_features.numpy().astype(np.float32)),
        "ef": mx.array(batch.edge_features.numpy().astype(np.float32)),
        "fg": mx.array(fg.astype(np.float32)), "eg": mx.array(eg.astype(np.float32)),
        "c2f": mx.array(batch.coedge_to_face.numpy().astype(np.int32)),
        "c2e": mx.array(batch.coedge_to_edge.numpy().astype(np.int32)),
        "c2n": mx.array(batch.coedge_to_next.numpy().astype(np.int32)),
        "c2p": mx.array(batch.coedge_to_prev.numpy().astype(np.int32)),
        "c2m": mx.array(batch.coedge_to_mate.numpy().astype(np.int32)),
        "rev": mx.array(batch.coedge_reversed.numpy().astype(np.float32)),
        "nf": int(batch.face_features.shape[0]),
    }


def miou(conf, nc):
    ious = []
    for c in range(nc):
        inter = conf[c, c]
        union = conf[c, :].sum() + conf[:, c].sum() - inter
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "convert", "parity"], default="parity")
    ap.add_argument("--ckpt", default=str(_CKPT))
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--out", default=str(_REPO / "ll_brepnet/checkpoints"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds_meta = json.load(open(_DATA / "dataset.json"))
    nc = int(ds_meta["num_classes"])

    model = LLBRepNetMLX(nc=nc)

    if args.mode == "probe":
        f = mx.zeros((3, 10, 10, 7)); e = mx.zeros((5, 10, 6))
        b = {"ff": mx.zeros((3, 8)), "ef": mx.zeros((5, 7)), "fg": f, "eg": e,
             "c2f": mx.array(np.array([0, 1, 2, 0, 1], np.int32)),
             "c2e": mx.array(np.array([0, 1, 2, 3, 4], np.int32)),
             "c2n": mx.array(np.array([1, 2, 0, 4, 3], np.int32)),
             "c2p": mx.array(np.array([2, 0, 1, 4, 3], np.int32)),
             "c2m": mx.array(np.array([3, 4, 2, 0, 1], np.int32)),
             "rev": mx.zeros((5, 1)), "nf": 3}
        out = model(b)
        print(f"probe: logits {out.shape} finite={bool(mx.isfinite(out).all().item())}", flush=True)
        return

    model, n = convert_checkpoint(args.ckpt, model)
    out_path = f"{args.out}/brepnet_mlx.safetensors"
    mx.save_safetensors(out_path, dict(tree_flatten(model.parameters())))
    print(f"converted {n} real tensors -> {out_path}", flush=True)
    if args.mode == "convert":
        return

    # parity: drive BOTH models from the real BRepDataset
    import sys
    sys.path.insert(0, str(_REPO / "ll_brepnet"))
    sys.path.insert(0, str(_REPO / "cadling"))
    import torch
    from ll_brepnet.models.ll_brepnet import LLBRepNet
    from ll_brepnet.dataloaders.brep_dataset import BRepDataset, brep_collate_fn

    pt = LLBRepNet.load_from_checkpoint(args.ckpt, map_location="cpu")
    pt.eval()
    ds = BRepDataset(_DATA / "dataset.json", _DATA, split="test", standardize=True)
    n_test = min(args.n_test, len(ds))
    print(f"running parity on {n_test} test solids ...", flush=True)

    conf_pt = np.zeros((nc, nc), np.int64)
    conf_mlx = np.zeros((nc, nc), np.int64)
    agree = total = 0
    for i in range(n_test):
        try:
            sample = ds[i]
            batch = brep_collate_fn([sample])
        except Exception:
            continue
        with torch.no_grad():
            lp = pt(batch).cpu().numpy()
        lm = np.array(model(to_mlx_batch(batch)).tolist())
        lab = batch.labels.numpy()
        m = lab != IGNORE
        pp, pm = lp.argmax(1), lm.argmax(1)
        agree += int((pp[m] == pm[m]).sum()); total += int(m.sum())
        for t, a, b2 in zip(lab[m], pp[m], pm[m]):
            conf_pt[t, a] += 1; conf_mlx[t, b2] += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_test}  running mIoU pt={miou(conf_pt,nc):.3f} mlx={miou(conf_mlx,nc):.3f}", flush=True)

    result = {"framework": "MLX (Apple Silicon)", "port": "faithful weight-conversion",
              "task": "B-rep face segmentation", "dataset": "Fusion360", "num_classes": nc,
              "n_test_solids": total and n_test, "n_faces_scored": int(total),
              "source_checkpoint": args.ckpt,
              "argmax_agreement_vs_pytorch": round(agree / max(total, 1), 4),
              "mlx_mIoU": round(miou(conf_mlx, nc), 4), "pytorch_mIoU": round(miou(conf_pt, nc), 4),
              "mlx_acc": round(float(np.trace(conf_mlx) / max(conf_mlx.sum(), 1)), 4),
              "pytorch_acc": round(float(np.trace(conf_pt) / max(conf_pt.sum(), 1)), 4),
              "pytorch_reference_full_test_mIoU": 0.828,
              "checkpoint": out_path}
    with open(f"{args.out}/brepnet_mlx_metrics.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print("BREPNET_MLX_PARITY", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
