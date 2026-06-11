"""ll_stepnet STEPForClassification in native MLX — faithful weight-conversion port.

This is NOT a simplified re-implementation. It reproduces the EXACT architecture of
``stepnet.tasks.STEPForClassification`` (the model that reached PyTorch val acc 0.976)
and CONVERTS the real trained PyTorch checkpoint
(``ll_stepnet/checkpoints/stepnet_classifier.pt``) into MLX, so the MLX model *is* the
trained model — same weights, same accuracy — running natively on Apple Silicon.

Architecture (reconstructed from the checkpoint state_dict, see encoder.py / tasks.py):

    STEPForClassification
      encoder (STEPEncoder)
        transformer_encoder (STEPTransformerEncoder)
          token_embedding : Embedding(50000, 256)
          pos_embedding   : (1, 5000, 256)            # trained nn.Parameter
          transformer     : 6 x nn.TransformerEncoderLayer  (post-norm, relu, 8 heads, ff=1024)
          layer_norm      : LayerNorm(256)            # applied after the stack
        graph_encoder (STEPGraphEncoder)             # UNUSED at inference: topology=None -> zeros(128)
        fusion : Linear(384->1024) -> ReLU -> Linear(1024->1024)
      classifier : Linear(1024->512) -> ReLU -> Dropout -> Linear(512->3)

    forward(token_ids):  # topology_data is None throughout training/eval
        x = transformer_encoder(token_ids)           # [B, S, 256]
        token_pooled = x.mean(dim=1)                 # [B, 256]  (UNMASKED mean over all positions)
        combined = cat([token_pooled, zeros(B,128)]) # [B, 384]
        return classifier(fusion(combined))          # [B, 3]

Every checkpoint tensor maps 1:1 onto an MLX Linear/Embedding/LayerNorm of identical
shape (MLX Linear.weight is [out, in] just like PyTorch), so conversion is a direct
array copy with NO transposes — the only non-trivial piece is multi-head attention,
implemented here manually from the packed ``in_proj_weight`` (768,256) = [Wq;Wk;Wv]
so it is bit-faithful to PyTorch's F.multi_head_attention_forward.

Modes:
  probe   - build the MLX model, run a forward, print shapes.
  convert - load the real .pt, convert weights, save MLX safetensors + parity report.
  parity  - convert AND run BOTH the PyTorch and MLX models on the same real DeepCAD
            val split; report argmax-agreement rate + each model's val accuracy.
  train   - train the faithful architecture from scratch in MLX (proves it is also
            natively trainable, not just a frozen import).
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
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore")
import logging  # noqa: E402

logging.disable(logging.WARNING)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.use = lambda *a, **k: None  # neutralize cadlib's TkAgg switch

import numpy as np  # noqa: E402
import h5py  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as mlxnn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_DEEPCAD = str(_REPO / "resources/DeepCAD")

# --- DeepCAD cad_vec -> command-token sequence + face-count class --------------
# IDENTICAL tokenization to ll_stepnet/scripts/train_classification.py (the trainer
# that produced the checkpoint), so the val data fed to the MLX model matches exactly.
LEVELS, RANGE, MAX_LEN = 256, 2.0, 256
NUM_SLOTS = 16
MASK = {"LINE": [0, 1, 2, 3], "ARC": [0, 1, 2, 3, 4, 5], "CIRCLE": [0, 1, 2],
        "EXTRUDE": [0, 1, 2, 3, 4, 5, 6, 7], "SOL": [], "EOS": []}
CMD_TOK = {"SOL": 6, "LINE": 7, "ARC": 8, "CIRCLE": 9, "EXTRUDE": 10, "EOS": 11}
BUCKETS = [(0, 4), (5, 6), (7, 9999)]
CLASS_NAMES = ["simple(<=4)", "box(5-6)", "complex(7+)"]
VOCAB = 50000  # matches the checkpoint's token_embedding (50000, 256)


def _q_coord(g):
    return int(np.clip(round(float(g)), 0, LEVELS - 1))


def _q_value(v):
    return int(np.clip(round((float(v) + RANGE) / (2 * RANGE) * (LEVELS - 1)), 0, LEVELS - 1))


def _translate(cad, Circle, Arc):
    cmds = []
    for ext in cad.seq:
        for loop in ext.profile.children:
            cmds.append(("SOL", {}))
            for cv in loop.children:
                if isinstance(cv, Circle):
                    r_mag = float(cv.radius) / (LEVELS - 1) * 2.0 * RANGE
                    cmds.append(("CIRCLE", {0: _q_coord(cv.center[0]), 1: _q_coord(cv.center[1]),
                                            2: _q_value(r_mag)}))
                elif isinstance(cv, Arc):
                    s, e, c = cv.start_point, cv.end_point, cv.center
                    cmds.append(("ARC", {0: _q_coord(s[0]), 1: _q_coord(s[1]), 2: _q_coord(e[0]),
                                         3: _q_coord(e[1]), 4: _q_coord(c[0]), 5: _q_coord(c[1])}))
                else:
                    s, e = cv.start_point, cv.end_point
                    cmds.append(("LINE", {0: _q_coord(s[0]), 1: _q_coord(s[1]),
                                          2: _q_coord(e[0]), 3: _q_coord(e[1])}))
        depth = abs(float(ext.extent_one)) + abs(float(ext.extent_two))
        cmds.append(("EXTRUDE", {0: _q_value(float(np.clip(depth * 4.0, 0.3, 2.0)))}))
    cmds.append(("EOS", {}))
    return cmds


def _command_dicts(cmds):
    out = []
    for name, slots in cmds:
        p = [0] * NUM_SLOTS
        m = [False] * NUM_SLOTS
        for j in MASK[name]:
            p[j] = int(slots.get(j, 0))
            m[j] = True
        out.append({"command_type": name, "parameters": p, "parameter_mask": m})
    return out


def _encode_tokens(cmds):
    toks = [1]
    for name, slots in cmds:
        toks.append(CMD_TOK[name])
        for j in MASK[name]:
            toks.append(12 + int(slots.get(j, 0)))
    toks.append(2)
    toks = toks[:MAX_LEN]
    return toks + [0] * (MAX_LEN - len(toks))


def _bucket(nf):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= nf <= hi:
            return i
    return len(BUCKETS) - 1


def build_val_split(n_val, cache):
    """Reproduce the PyTorch trainer's val split EXACTLY: scan sorted(files)[:need//6]
    and collect the first ``n_val`` that produce valid solids, same tokenization."""
    if cache and os.path.exists(cache):
        d = np.load(cache)
        if d["tokens"].shape[0] >= n_val:
            return d["tokens"][:n_val], d["classes"][:n_val]
    sys.path.insert(0, _DEEPCAD)
    sys.path.insert(0, str(_REPO / "resources/ll_gen_proof"))
    from cadlib.extrude import CADSequence
    from cadlib.curves import Arc, Circle
    from ll_gen.proposals.command_proposal import CommandSequenceProposal
    from ll_gen.disposal.command_executor import execute_command_proposal
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE

    def face_count(shape):
        c = 0
        e = TopExp_Explorer(shape, TopAbs_FACE)
        while e.More():
            c += 1
            e.Next()
        return c

    files = sorted(glob.glob(os.path.join(_DEEPCAD, "data/cad_vec", "*/*.h5")))
    need = (5000 + 1000) * 3 + 4000  # == 22000, mirrors the trainer
    files = files[: need // 6]
    toks, classes = [], []
    for f in files:
        if len(toks) >= n_val:
            break
        try:
            with h5py.File(f, "r") as h:
                vec = h["vec"][:].astype(int)
            cad = CADSequence.from_vector(vec, is_numerical=True, n=256)
            cmds = _translate(cad, Circle, Arc)
            shape = execute_command_proposal(CommandSequenceProposal(
                command_dicts=_command_dicts(cmds), quantization_bits=8, normalization_range=2.0))
            if shape is None:
                continue
            nf = face_count(shape)
            if nf < 1:
                continue
            toks.append(_encode_tokens(cmds))
            classes.append(_bucket(nf))
        except Exception:
            continue
    toks = np.array(toks, np.int32)
    classes = np.array(classes, np.int32)
    if cache:
        np.savez(cache, tokens=toks, classes=classes)
    return toks, classes


# --- faithful MLX model (exact port of STEPForClassification) ------------------
class FaithfulMHA(mlxnn.Module):
    """PyTorch nn.MultiheadAttention math from the packed in_proj_weight (768,256)."""

    def __init__(self, d=256, heads=8):
        super().__init__()
        self.d, self.h, self.hd = d, heads, d // heads
        self.in_proj = mlxnn.Linear(d, 3 * d)   # weight (768,256), bias (768,)
        self.out_proj = mlxnn.Linear(d, d)       # weight (256,256), bias (256,)

    def __call__(self, x):  # x: [B, S, d]
        B, S, _ = x.shape
        qkv = self.in_proj(x)                                  # [B,S,768]
        q, k, v = mx.split(qkv, 3, axis=-1)                    # 3 x [B,S,256]

        def heads(t):
            return mx.transpose(t.reshape(B, S, self.h, self.hd), (0, 2, 1, 3))  # [B,h,S,hd]

        q, k, v = heads(q), heads(k), heads(v)
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) / (self.hd ** 0.5)  # [B,h,S,S]
        ctx = mx.softmax(scores, axis=-1) @ v                            # [B,h,S,hd]
        ctx = mx.transpose(ctx, (0, 2, 1, 3)).reshape(B, S, self.d)      # [B,S,d]
        return self.out_proj(ctx)


class FaithfulLayer(mlxnn.Module):
    """nn.TransformerEncoderLayer (post-norm, activation=relu) — PyTorch defaults."""

    def __init__(self, d=256, heads=8, ff=1024):
        super().__init__()
        self.self_attn = FaithfulMHA(d, heads)
        self.linear1 = mlxnn.Linear(d, ff)
        self.linear2 = mlxnn.Linear(ff, d)
        self.norm1 = mlxnn.LayerNorm(d)
        self.norm2 = mlxnn.LayerNorm(d)

    def __call__(self, x):
        x = self.norm1(x + self.self_attn(x))
        return self.norm2(x + self.linear2(mlxnn.relu(self.linear1(x))))


class FaithfulTransformerEncoder(mlxnn.Module):
    def __init__(self, vocab=VOCAB, d=256, layers=6, heads=8, ff=1024):
        super().__init__()
        self.token_embedding = mlxnn.Embedding(vocab, d)
        self.pos_embedding = mx.zeros((1, 5000, d))  # overwritten by converted weights
        self.layers = [FaithfulLayer(d, heads, ff) for _ in range(layers)]
        self.layer_norm = mlxnn.LayerNorm(d)

    def __call__(self, ids):
        x = self.token_embedding(ids) + self.pos_embedding[:, : ids.shape[1], :]
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class FaithfulSTEPClassifier(mlxnn.Module):
    def __init__(self, vocab=VOCAB, nclass=3, d=256, output_dim=1024, graph_node_dim=128):
        super().__init__()
        self.te = FaithfulTransformerEncoder(vocab, d)
        self.graph_node_dim = graph_node_dim
        self.fusion0 = mlxnn.Linear(d + graph_node_dim, output_dim)
        self.fusion2 = mlxnn.Linear(output_dim, output_dim)
        self.cls0 = mlxnn.Linear(output_dim, 512)
        self.cls3 = mlxnn.Linear(512, nclass)

    def __call__(self, ids):
        x = self.te(ids)                                   # [B,S,256]
        token_pooled = x.mean(axis=1)                      # [B,256] unmasked mean
        zero_graph = mx.zeros((ids.shape[0], self.graph_node_dim))
        combined = mx.concatenate([token_pooled, zero_graph], axis=-1)  # [B,384]
        out = self.fusion2(mlxnn.relu(self.fusion0(combined)))          # [B,1024]
        return self.cls3(mlxnn.relu(self.cls0(out)))                    # [B,nclass]


# --- weight conversion: real PyTorch checkpoint -> MLX params ------------------
def convert_checkpoint(ckpt_path, model):
    """Load the real trained state_dict and assign it into the MLX model by an
    explicit name map. Returns (model, n_converted)."""
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)

    def arr(key):
        return mx.array(sd[key].detach().cpu().float().numpy())

    pairs = []
    te = "encoder.transformer_encoder"
    pairs.append(("te.token_embedding.weight", arr(f"{te}.token_embedding.weight")))
    pairs.append(("te.pos_embedding", arr(f"{te}.pos_embedding")))
    pairs.append(("te.layer_norm.weight", arr(f"{te}.layer_norm.weight")))
    pairs.append(("te.layer_norm.bias", arr(f"{te}.layer_norm.bias")))
    n_layers = len({k.split(".layers.")[1].split(".")[0]
                    for k in sd if f"{te}.transformer.layers." in k})
    for i in range(n_layers):
        s = f"{te}.transformer.layers.{i}"
        d = f"te.layers.{i}"
        pairs += [
            (f"{d}.self_attn.in_proj.weight", arr(f"{s}.self_attn.in_proj_weight")),
            (f"{d}.self_attn.in_proj.bias", arr(f"{s}.self_attn.in_proj_bias")),
            (f"{d}.self_attn.out_proj.weight", arr(f"{s}.self_attn.out_proj.weight")),
            (f"{d}.self_attn.out_proj.bias", arr(f"{s}.self_attn.out_proj.bias")),
            (f"{d}.linear1.weight", arr(f"{s}.linear1.weight")),
            (f"{d}.linear1.bias", arr(f"{s}.linear1.bias")),
            (f"{d}.linear2.weight", arr(f"{s}.linear2.weight")),
            (f"{d}.linear2.bias", arr(f"{s}.linear2.bias")),
            (f"{d}.norm1.weight", arr(f"{s}.norm1.weight")),
            (f"{d}.norm1.bias", arr(f"{s}.norm1.bias")),
            (f"{d}.norm2.weight", arr(f"{s}.norm2.weight")),
            (f"{d}.norm2.bias", arr(f"{s}.norm2.bias")),
        ]
    for src, dst in (("encoder.fusion.0", "fusion0"), ("encoder.fusion.2", "fusion2"),
                     ("classifier.0", "cls0"), ("classifier.3", "cls3")):
        pairs.append((f"{dst}.weight", arr(f"{src}.weight")))
        pairs.append((f"{dst}.bias", arr(f"{src}.bias")))

    model.update(tree_unflatten(pairs))
    mx.eval(model.parameters())
    return model, len(pairs)


def _pt_logits(ckpt_path, toks, nclass):
    """Run the real PyTorch STEPForClassification on the same tokens (for parity)."""
    import torch
    sys.path.insert(0, str(_REPO / "ll_stepnet"))
    from stepnet.tasks import STEPForClassification

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = STEPForClassification(num_classes=nclass)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    outs = []
    with torch.no_grad():
        for k in range(0, toks.shape[0], 256):
            t = torch.tensor(toks[k:k + 256], dtype=torch.long)
            outs.append(model(t).cpu().numpy())
    return np.concatenate(outs, axis=0)


def _mlx_logits(model, toks):
    outs = []
    for k in range(0, toks.shape[0], 256):
        outs.append(np.array(model(mx.array(toks[k:k + 256])).tolist()))
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "convert", "parity", "train"], default="parity")
    ap.add_argument("--ckpt", default=str(_REPO / "ll_stepnet/checkpoints/stepnet_classifier.pt"))
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=str(_REPO / "ll_stepnet/checkpoints"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    nclass = len(BUCKETS)

    if args.mode == "probe":
        model = FaithfulSTEPClassifier(nclass=nclass)
        ids = mx.array(np.random.randint(0, 268, (4, MAX_LEN)).astype(np.int32))
        out = model(ids)
        print(f"probe: logits {out.shape} finite={bool(mx.isfinite(out).all().item())}", flush=True)
        return

    if args.mode == "convert":
        model = FaithfulSTEPClassifier(nclass=nclass)
        model, n = convert_checkpoint(args.ckpt, model)
        out_path = f"{args.out}/stepnet_classifier_mlx.safetensors"
        mx.save_safetensors(out_path, dict(tree_flatten(model.parameters())))
        print(f"converted {n} tensors -> {out_path}", flush=True)
        return

    if args.mode == "parity":
        cache = f"{args.out}/mlx_parity_val.npz"
        print(f"building/loading {args.n_val} real val samples (same split as trainer) ...", flush=True)
        toks, cls = build_val_split(args.n_val, cache)
        print(f"val set: {toks.shape[0]} samples, dist={np.bincount(cls, minlength=nclass).tolist()}", flush=True)

        model = FaithfulSTEPClassifier(nclass=nclass)
        model, n = convert_checkpoint(args.ckpt, model)
        mx.save_safetensors(f"{args.out}/stepnet_classifier_mlx.safetensors",
                            dict(tree_flatten(model.parameters())))
        print(f"converted {n} real tensors into MLX", flush=True)

        lg_mlx = _mlx_logits(model, toks)
        lg_pt = _pt_logits(args.ckpt, toks, nclass)
        pred_mlx, pred_pt = lg_mlx.argmax(1), lg_pt.argmax(1)
        agree = float((pred_mlx == pred_pt).mean())
        max_logit_diff = float(np.abs(lg_mlx - lg_pt).max())
        acc_mlx = float((pred_mlx == cls).mean())
        acc_pt = float((pred_pt == cls).mean())
        per = {CLASS_NAMES[c]: round(float(((pred_mlx == cls) & (cls == c)).sum() /
                                            max((cls == c).sum(), 1)), 3) for c in range(nclass)}
        majority = float(np.bincount(cls, minlength=nclass).max() / len(cls))

        result = {"framework": "MLX (Apple Silicon)", "port": "faithful weight-conversion",
                  "task": "STEP->face-count complexity class (3)", "dataset": "DeepCAD cad_vec",
                  "n_val": int(toks.shape[0]), "source_checkpoint": args.ckpt,
                  "argmax_agreement_vs_pytorch": round(agree, 4),
                  "max_abs_logit_diff": round(max_logit_diff, 5),
                  "mlx_val_acc": round(acc_mlx, 4), "pytorch_val_acc": round(acc_pt, 4),
                  "majority_baseline": round(majority, 3), "mlx_per_class_acc": per,
                  "checkpoint": f"{args.out}/stepnet_classifier_mlx.safetensors"}
        with open(f"{args.out}/stepnet_classifier_mlx_metrics.json", "w") as fh:
            json.dump(result, fh, indent=2)
        print("STEPNET_MLX_PARITY", json.dumps(result), flush=True)
        return

    # mode == train : native MLX training of the faithful architecture from scratch
    cache = f"{args.out}/mlx_parity_val.npz"
    toks, cls = build_val_split(args.n_val, cache)
    n_tr = int(toks.shape[0] * 0.8)
    tt, tcl, vt, vcl = toks[:n_tr], cls[:n_tr], toks[n_tr:], cls[n_tr:]
    model = FaithfulSTEPClassifier(nclass=nclass)
    cnt = np.bincount(tcl, minlength=nclass)
    w = mx.array((cnt.sum() / (nclass * np.clip(cnt, 1, None))).astype(np.float32))
    opt = optim.Adam(learning_rate=args.lr)

    def loss_fn(ids, y):
        ce = mlxnn.losses.cross_entropy(model(ids), y, reduction="none")
        return (ce * w[y]).mean()

    lg = mlxnn.value_and_grad(model, loss_fn)
    for epoch in range(args.epochs):
        perm = np.random.permutation(tt.shape[0])
        for k in range(0, tt.shape[0], args.bs):
            idx = perm[k:k + args.bs]
            lv, g = lg(mx.array(tt[idx]), mx.array(tcl[idx]))
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state, lv)
        pv = _mlx_logits(model, vt).argmax(1)
        print(f"epoch {epoch+1}/{args.epochs} val_acc={float((pv == vcl).mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
