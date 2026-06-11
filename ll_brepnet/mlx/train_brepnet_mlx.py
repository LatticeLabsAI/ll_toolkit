"""ll_brepnet B-rep face segmentation in native MLX (Apple Silicon).

MLX port of LLBRepNet: UV-Net face/edge encoders -> per-entity projections ->
coedge features (gather face/edge reprs by coedge topology) -> BRepNet coedge
message-passing (W_self·h + W_next·h[next] + W_prev·h[prev] + W_mate·h[mate],
x num_layers) -> coedge->face mean-pool (scatter-mean via one-hot matmul) ->
per-face segmentation head. Same Fusion360 data + task as the PyTorch trainer
(which reached test mIoU 0.828).

Graph batching: N solids are concatenated with offset arithmetic on the coedge/
face/edge indices, so a batch is one big graph. Trains entirely in MLX.

Modes: probe | train.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import logging  # noqa: E402

logging.disable(logging.WARNING)
import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as mlxnn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_DATA = _REPO / "resources/fusion360/full_processed"
IGNORE = -1


# --- data -------------------------------------------------------------------
def load_records(stems, fmean, fstd, emean, estd):
    recs = []
    for s in stems:
        npz = _DATA / f"{s}.npz"
        seg = _DATA / f"{s}.seg"
        if not npz.exists() or not seg.exists():
            continue
        try:
            z = np.load(npz)
            labels = np.loadtxt(seg, dtype=np.int64).reshape(-1)
            nf = int(z["num_faces"])
            if labels.shape[0] != nf:
                continue
            ff = (z["face_features"].astype(np.float32) - fmean) / fstd
            ef = (z["edge_features"].astype(np.float32) - emean) / estd
            # grids -> channels-last for MLX conv: face [F,10,10,7], edge [E,10,6]
            fg = np.transpose(z["face_point_grids"].astype(np.float32), (0, 2, 3, 1))
            eg = np.transpose(z["edge_point_grids"].astype(np.float32), (0, 2, 1))
            recs.append({
                "ff": ff, "ef": ef, "fg": fg, "eg": eg,
                "c2f": z["coedge_to_face"].astype(np.int32),
                "c2e": z["coedge_to_edge"].astype(np.int32),
                "c2n": z["coedge_to_next"].astype(np.int32),
                "c2p": z["coedge_to_prev"].astype(np.int32),
                "c2m": z["coedge_to_mate"].astype(np.int32),
                "rev": z["coedge_reversed"].astype(np.float32),
                "labels": labels.astype(np.int32), "nf": nf,
            })
        except Exception:
            continue
    return recs


def collate(batch):
    """Concatenate solids into one big graph with offset coedge/face/edge ids."""
    fo = co = eo = 0
    out = {k: [] for k in ["ff", "ef", "fg", "eg", "c2f", "c2e", "c2n", "c2p", "c2m", "rev", "labels"]}
    for r in batch:
        out["ff"].append(r["ff"]); out["ef"].append(r["ef"])
        out["fg"].append(r["fg"]); out["eg"].append(r["eg"])
        out["c2f"].append(r["c2f"] + fo); out["c2e"].append(r["c2e"] + eo)
        out["c2n"].append(r["c2n"] + co); out["c2p"].append(r["c2p"] + co); out["c2m"].append(r["c2m"] + co)
        out["rev"].append(r["rev"]); out["labels"].append(r["labels"])
        fo += r["nf"]; eo += r["ef"].shape[0]; co += r["c2f"].shape[0]
    cat = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    cat["n_faces"] = fo
    return cat


# --- model ------------------------------------------------------------------
class SurfEnc(mlxnn.Module):
    def __init__(self, cin=7, out=64):
        super().__init__()
        self.c1 = mlxnn.Conv2d(cin, 32, 3, padding=1)
        self.c2 = mlxnn.Conv2d(32, 64, 3, padding=1)
        self.lin = mlxnn.Linear(64, out)

    def __call__(self, g):  # [F,10,10,7]
        x = mlxnn.relu(self.c1(g)); x = mlxnn.relu(self.c2(x))
        x = x.mean(axis=(1, 2))  # global avg pool -> [F,64]
        return mlxnn.relu(self.lin(x))


class CurveEnc(mlxnn.Module):
    def __init__(self, cin=6, out=64):
        super().__init__()
        self.c1 = mlxnn.Conv1d(cin, 32, 3, padding=1)
        self.c2 = mlxnn.Conv1d(32, 64, 3, padding=1)
        self.lin = mlxnn.Linear(64, out)

    def __call__(self, g):  # [E,10,6]
        x = mlxnn.relu(self.c1(g)); x = mlxnn.relu(self.c2(x))
        x = x.mean(axis=1)  # [E,64]
        return mlxnn.relu(self.lin(x))


class CoedgeConv(mlxnn.Module):
    def __init__(self, d):
        super().__init__()
        self.ws = mlxnn.Linear(d, d); self.wn = mlxnn.Linear(d, d)
        self.wp = mlxnn.Linear(d, d); self.wm = mlxnn.Linear(d, d)

    def __call__(self, h, nidx, pidx, midx):
        return mlxnn.relu(self.ws(h) + self.wn(h[nidx]) + self.wp(h[pidx]) + self.wm(h[midx]))


class BRepNetMLX(mlxnn.Module):
    def __init__(self, num_classes=8, ent=64, hid=128, layers=6):
        super().__init__()
        self.surf = SurfEnc(7, 64)
        self.curve = CurveEnc(6, 64)
        self.face_proj = mlxnn.Linear(8 + 64, ent)
        self.edge_proj = mlxnn.Linear(7 + 64, ent)
        self.in_proj = mlxnn.Linear(2 * ent + 1, hid)
        self.convs = [CoedgeConv(hid) for _ in range(layers)]
        self.head = mlxnn.Linear(hid, num_classes)

    def __call__(self, b):
        face_repr = mlxnn.relu(self.face_proj(mx.concatenate([b["ff"], self.surf(b["fg"])], axis=1)))
        edge_repr = mlxnn.relu(self.edge_proj(mx.concatenate([b["ef"], self.curve(b["eg"])], axis=1)))
        h = mx.concatenate([face_repr[b["c2f"]], edge_repr[b["c2e"]], b["rev"][:, None]], axis=1)
        h = mlxnn.relu(self.in_proj(h))
        for conv in self.convs:
            h = conv(h, b["c2n"], b["c2p"], b["c2m"])
        # coedge -> face mean pool via one-hot matmul (scatter-mean)
        nf = b["n_faces"]
        onehot = (b["c2f"][:, None] == mx.arange(nf)[None, :]).astype(h.dtype)  # [C, F]
        face_emb = (onehot.T @ h) / mx.maximum(onehot.sum(axis=0)[:, None], 1)
        return self.head(face_emb)  # [F, num_classes]


def to_mx(b):
    return {
        "ff": mx.array(b["ff"]), "ef": mx.array(b["ef"]),
        "fg": mx.array(b["fg"]), "eg": mx.array(b["eg"]),
        "c2f": mx.array(b["c2f"]), "c2e": mx.array(b["c2e"]),
        "c2n": mx.array(b["c2n"]), "c2p": mx.array(b["c2p"]), "c2m": mx.array(b["c2m"]),
        "rev": mx.array(b["rev"]), "n_faces": int(b["n_faces"]),
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
    ap.add_argument("--mode", choices=["probe", "train"], default="train")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(_REPO / "ll_brepnet/checkpoints"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = json.load(open(_DATA / "dataset.json"))
    nc = int(ds["num_classes"])
    fs = ds["feature_standardization"]
    fmean = np.array([e["mean"] for e in fs["face_features"]], np.float32)
    fstd = np.array([e["standard_deviation"] for e in fs["face_features"]], np.float32) + 1e-8
    emean = np.array([e["mean"] for e in fs["edge_features"]], np.float32)
    estd = np.array([e["standard_deviation"] for e in fs["edge_features"]], np.float32) + 1e-8
    print(f"num_classes={nc} classes={ds['class_names']}", flush=True)

    train_stems = ds["training_set"][: args.n_train]
    val_stems = ds["validation_set"][: args.n_val]
    print(f"loading {len(train_stems)} train / {len(val_stems)} val records ...", flush=True)
    train = load_records(train_stems, fmean, fstd, emean, estd)
    val = load_records(val_stems, fmean, fstd, emean, estd)
    print(f"loaded {len(train)} train / {len(val)} val solids", flush=True)

    model = BRepNetMLX(num_classes=nc)

    if args.mode == "probe":
        b = to_mx(collate(train[:4]))
        out = model(b)
        print(f"probe: batch faces={b['n_faces']} logits={out.shape} finite={bool(mx.isfinite(out).all().item())}", flush=True)
        return

    opt = optim.Adam(learning_rate=args.lr)

    def loss_fn(b, labels):
        logits = model(b)
        mask = labels != IGNORE
        ce = mlxnn.losses.cross_entropy(logits, mx.where(mask, labels, 0), reduction="none") * mask
        return ce.sum() / mx.maximum(mask.sum(), 1)

    lg = mlxnn.value_and_grad(model, loss_fn)

    def evaluate(recs):
        conf = np.zeros((nc, nc), np.int64)
        for k in range(0, len(recs), args.bs):
            b = collate(recs[k:k + args.bs])
            logits = model(to_mx(b))
            pred = np.array(mx.argmax(logits, axis=1).tolist())
            lab = b["labels"]
            m = lab != IGNORE
            for p, t in zip(pred[m], lab[m]):
                conf[t, p] += 1
        acc = float(np.trace(conf) / max(conf.sum(), 1))
        return miou(conf, nc), acc

    best = -1.0
    for epoch in range(args.epochs):
        np.random.shuffle(train)
        tot = 0.0; nb = 0
        for k in range(0, len(train), args.bs):
            b = collate(train[k:k + args.bs])
            labels = mx.array(b["labels"])
            lv, g = lg(to_mx(b), labels)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state, lv)
            tot += float(lv.item()); nb += 1
        vmiou, vacc = evaluate(val)
        print(f"epoch {epoch+1}/{args.epochs} loss={tot/max(nb,1):.4f} val_mIoU={vmiou:.3f} val_acc={vacc:.3f}", flush=True)
        if vmiou > best:
            best = vmiou
            mx.save_safetensors(f"{args.out}/brepnet_mlx.safetensors", dict(tree_flatten(model.parameters())))

    result = {"framework": "MLX", "task": "B-rep face segmentation", "dataset": "Fusion360",
              "num_classes": nc, "n_train": len(train), "n_val": len(val), "epochs": args.epochs,
              "best_val_mIoU": round(best, 3), "pytorch_reference_test_mIoU": 0.828,
              "checkpoint": f"{args.out}/brepnet_mlx.safetensors"}
    with open(f"{args.out}/brepnet_mlx_metrics.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print("BREPNET_MLX_DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
