"""ll_stepnet STEP classifier in native MLX (Apple Silicon).

MLX port of stepnet.tasks.STEPForClassification: a token-embedding +
transformer-encoder + mean-pool + MLP-head classifier. Same task and data as the
PyTorch trainer (ll_stepnet/scripts/train_classification.py) — DeepCAD cad_vec ->
command-token sequence -> face-count complexity class — so the MLX result is
directly comparable (PyTorch reached val acc 0.976 vs 0.434 majority).

Trains entirely in MLX on Apple Silicon. Modes: probe | train.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")
import logging  # noqa: E402

logging.disable(logging.WARNING)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.use = lambda *a, **k: None

import numpy as np  # noqa: E402
import h5py  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as mlxnn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_DEEPCAD = str(_REPO / "resources/DeepCAD")
sys.path.insert(0, _DEEPCAD)
sys.path.insert(0, str(_REPO / "resources/ll_gen_proof"))

# --- DeepCAD cad_vec -> command-token sequence + face-count class (self-contained)
LEVELS, RANGE, MAX_LEN = 256, 2.0, 256
MASK = {"LINE": [0, 1, 2, 3], "ARC": [0, 1, 2, 3, 4, 5], "CIRCLE": [0, 1, 2],
        "EXTRUDE": [0, 1, 2, 3, 4, 5, 6, 7], "SOL": [], "EOS": []}
CMD_TOK = {"SOL": 6, "LINE": 7, "ARC": 8, "CIRCLE": 9, "EXTRUDE": 10, "EOS": 11}
VOCAB = 12 + LEVELS  # command/special tokens + param value tokens
BUCKETS = [(0, 4), (5, 6), (7, 9999)]
CLASS_NAMES = ["simple(<=4)", "box(5-6)", "complex(7+)"]


def _qc(g):
    return int(np.clip(round(float(g)), 0, LEVELS - 1))


def _qv(v):
    return int(np.clip(round((float(v) + RANGE) / (2 * RANGE) * (LEVELS - 1)), 0, LEVELS - 1))


def _cmds(cad, Circle, Arc):
    out = []
    for ext in cad.seq:
        for loop in ext.profile.children:
            out.append(("SOL", {}))
            for cv in loop.children:
                if isinstance(cv, Circle):
                    out.append(("CIRCLE", {0: _qc(cv.center[0]), 1: _qc(cv.center[1]),
                                           2: _qv(float(cv.radius) / (LEVELS - 1) * 2 * RANGE)}))
                elif isinstance(cv, Arc):
                    s, e, c = cv.start_point, cv.end_point, cv.center
                    out.append(("ARC", {0: _qc(s[0]), 1: _qc(s[1]), 2: _qc(e[0]), 3: _qc(e[1]),
                                        4: _qc(c[0]), 5: _qc(c[1])}))
                else:
                    s, e = cv.start_point, cv.end_point
                    out.append(("LINE", {0: _qc(s[0]), 1: _qc(s[1]), 2: _qc(e[0]), 3: _qc(e[1])}))
        out.append(("EXTRUDE", {0: _qv(float(np.clip((abs(float(ext.extent_one)) + abs(float(ext.extent_two))) * 4, 0.3, 2.0)))}))
    out.append(("EOS", {}))
    return out


def _command_dicts(cmds):
    out = []
    for name, slots in cmds:
        p = [0] * 16
        m = [False] * 16
        for j in MASK[name]:
            p[j] = int(slots.get(j, 0))
            m[j] = True
        out.append({"command_type": name, "parameters": p, "parameter_mask": m})
    return out


def _tokens(cmds):
    t = [1]
    for name, slots in cmds:
        t.append(CMD_TOK[name])
        for j in MASK[name]:
            t.append(12 + int(slots.get(j, 0)))
    t.append(2)
    t = t[:MAX_LEN]
    return t + [0] * (MAX_LEN - len(t))


def _bucket(nf):
    return 0 if nf <= 4 else (1 if nf <= 6 else 2)


def build_dataset(n_target, cache):
    if cache and os.path.exists(cache):
        d = np.load(cache)
        if d["tokens"].shape[0] >= n_target:
            return d["tokens"][:n_target], d["classes"][:n_target]
    from cadlib.extrude import CADSequence
    from cadlib.curves import Arc, Circle
    from ll_gen.proposals.command_proposal import CommandSequenceProposal
    from ll_gen.disposal.command_executor import execute_command_proposal
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE

    def nfaces(shape):
        c = 0
        e = TopExp_Explorer(shape, TopAbs_FACE)
        while e.More():
            c += 1
            e.Next()
        return c

    toks, classes = [], []
    for f in sorted(glob.glob(os.path.join(_DEEPCAD, "data/cad_vec/*/*.h5"))):
        if len(toks) >= n_target:
            break
        try:
            with h5py.File(f, "r") as h:
                vec = h["vec"][:].astype(int)
            cad = CADSequence.from_vector(vec, is_numerical=True, n=256)
            cmds = _cmds(cad, Circle, Arc)
            shape = execute_command_proposal(CommandSequenceProposal(
                command_dicts=_command_dicts(cmds), quantization_bits=8, normalization_range=2.0))
            if shape is None:
                continue
            nf = nfaces(shape)
            if nf < 1:
                continue
            toks.append(_tokens(cmds))
            classes.append(_bucket(nf))
        except Exception:
            continue
    toks = np.array(toks, np.int32)
    classes = np.array(classes, np.int32)
    if cache:
        np.savez(cache, tokens=toks, classes=classes)
    return toks, classes


# --- MLX transformer classifier (port of STEPForClassification) -------------
class Block(mlxnn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.attn = mlxnn.MultiHeadAttention(d, heads)
        self.n1 = mlxnn.LayerNorm(d)
        self.ffn = mlxnn.Sequential(mlxnn.Linear(d, d * 4), mlxnn.GELU(), mlxnn.Linear(d * 4, d))
        self.n2 = mlxnn.LayerNorm(d)

    def __call__(self, x):
        x = self.n1(x + self.attn(x, x, x))
        return self.n2(x + self.ffn(x))


class MLXClassifier(mlxnn.Module):
    def __init__(self, vocab=VOCAB, d=128, layers=2, heads=4, nclass=3, maxlen=MAX_LEN):
        super().__init__()
        self.embed = mlxnn.Embedding(vocab, d)
        self.pos = mx.random.normal((maxlen, d)) * 0.02
        self.blocks = [Block(d, heads) for _ in range(layers)]
        self.norm = mlxnn.LayerNorm(d)
        self.head = mlxnn.Sequential(mlxnn.Linear(d, 512), mlxnn.ReLU(), mlxnn.Linear(512, nclass))

    def __call__(self, ids):
        x = self.embed(ids) + self.pos[: ids.shape[1]][None]
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        m = (ids != 0).astype(x.dtype)[..., None]
        pooled = (x * m).sum(axis=1) / mx.maximum(m.sum(axis=1), 1)
        return self.head(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "train"], default="train")
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=str(_REPO / "ll_stepnet/checkpoints"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache = f"{args.out}/mlx_classifier_data.npz"

    if args.mode == "probe":
        toks, classes = build_dataset(40, None)
        model = MLXClassifier()
        out = model(mx.array(toks[:8]))
        import collections
        print(f"probe: tokens {toks.shape}, logits {out.shape}, dist {dict(collections.Counter(classes.tolist()))}", flush=True)
        return

    print("building/loading dataset ...", flush=True)
    toks, classes = build_dataset(args.n_train + args.n_val, cache)
    vt, vcl = toks[:args.n_val], classes[:args.n_val]
    tt, tcl = toks[args.n_val:], classes[args.n_val:]
    import collections
    cnt = np.bincount(tcl, minlength=3)
    majority = float(np.bincount(vcl, minlength=3).max() / len(vcl))
    print(f"built {tt.shape[0]} train / {vt.shape[0]} val; train dist={cnt.tolist()}; majority={majority:.3f}", flush=True)

    model = MLXClassifier()
    w = mx.array((cnt.sum() / (3 * np.clip(cnt, 1, None))).astype(np.float32))
    opt = optim.Adam(learning_rate=args.lr)

    def loss_fn(ids, y):
        logits = model(ids)
        ce = mlxnn.losses.cross_entropy(logits, y, reduction="none")
        return (ce * w[y]).mean()

    lg = mlxnn.value_and_grad(model, loss_fn)

    def acc(toks_, cls_):
        c = 0
        for k in range(0, toks_.shape[0], 256):
            p = np.array(mx.argmax(model(mx.array(toks_[k:k + 256])), axis=1).tolist())
            c += int((p == cls_[k:k + 256]).sum())
        return c / toks_.shape[0]

    best = -1.0
    n = tt.shape[0]
    for epoch in range(args.epochs):
        perm = np.random.permutation(n)
        tot = 0.0
        nb = 0
        for k in range(0, n, args.bs):
            idx = perm[k:k + args.bs]
            lv, g = lg(mx.array(tt[idx]), mx.array(tcl[idx]))
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state, lv)
            tot += float(lv.item())
            nb += 1
        a = acc(vt, vcl)
        print(f"epoch {epoch+1}/{args.epochs} loss={tot/max(nb,1):.4f} val_acc={a:.3f}", flush=True)
        if a > best:
            best = a
            mx.save_safetensors(f"{args.out}/stepnet_classifier_mlx.safetensors",
                                dict(tree_flatten(model.parameters())))

    result = {"framework": "MLX", "task": "STEP->face-count class (3)", "dataset": "DeepCAD cad_vec",
              "n_train": int(tt.shape[0]), "n_val": int(vt.shape[0]), "epochs": args.epochs,
              "majority_baseline": round(majority, 3), "best_val_acc": round(best, 3),
              "pytorch_reference_acc": 0.976,
              "checkpoint": f"{args.out}/stepnet_classifier_mlx.safetensors"}
    with open(f"{args.out}/stepnet_classifier_mlx_metrics.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print("STEPNET_MLX_DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
