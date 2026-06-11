"""Autoregressive CAD-command generator in native MLX — produces MEASURED-valid CAD.

The existing ll_gen generators cannot produce valid CAD: the command-VAE's parallel
(z-broadcast) decoder is primitive-limited and posterior-collapses (validity ~0-12%),
and the diffusion path samples faces independently so they never mate (validity 0).
The robust, proven route to valid CAD (DeepCAD / Text2CAD) is to generate the
CONSTRUCTION PROGRAM autoregressively and execute it: the model learns the command
grammar from real data, and the OCC kernel builds the solid command-by-command.

Pipeline:
  real DeepCAD cad_vec -> translate -> command-token sequence (vocab 268: 0=PAD, 1=BOS,
  2=EOS, 6-11=command types, 12..267=quantised param values)
  -> causal-transformer LM, teacher-forced next-token training on real sequences
  -> autoregressive sampling (temperature + top-k)
  -> decode tokens -> command_dicts -> execute_command_proposal -> OCC solid

Validity is MEASURED through the real kernel and gated HONESTLY against the cylinder
trap: a sample counts as valid only if it forms a solid (solid_count >= 1) with
non-degenerate volume (> eps); we also report num_distinct (rounded bounding boxes)
and the volume spread, so a high rate with one repeated trivial shape is visible.

Modes: probe | train  (train trains, samples, and reports measured validity).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from collections import Counter
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

# --- tokenization (shared scheme with the stepnet/ocadr trainers) -------------
LEVELS, RANGE, MAX_LEN = 256, 2.0, 64
MASK = {"LINE": [0, 1, 2, 3], "ARC": [0, 1, 2, 3, 4, 5], "CIRCLE": [0, 1, 2],
        "EXTRUDE": [0, 1, 2, 3, 4, 5, 6, 7], "SOL": [], "EOS": []}
CMD_TOK = {"SOL": 6, "LINE": 7, "ARC": 8, "CIRCLE": 9, "EXTRUDE": 10, "EOS": 11}
TOK_CMD = {v: k for k, v in CMD_TOK.items()}
VOCAB = 12 + LEVELS
PAD, BOS, SEQ_EOS = 0, 1, 2


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
        out.append(("EXTRUDE", {0: _qv(float(np.clip((abs(float(ext.extent_one)) +
                                                      abs(float(ext.extent_two))) * 4, 0.3, 2.0)))}))
    out.append(("EOS", {}))
    return out


def encode_tokens(cmds):
    t = [BOS]
    for name, slots in cmds:
        t.append(CMD_TOK[name])
        for j in MASK[name]:
            t.append(12 + int(slots.get(j, 0)))
    t.append(SEQ_EOS)
    t = t[:MAX_LEN]
    return t + [PAD] * (MAX_LEN - len(t))


def decode_tokens(toks):
    """Token list -> list of (command_name, {slot: value}). Robust to malformed runs."""
    cmds = []
    i, n = 0, len(toks)
    while i < n:
        t = int(toks[i])
        if t == SEQ_EOS or t == PAD:
            break
        if t in TOK_CMD:
            name = TOK_CMD[t]
            if name == "EOS":
                break
            slots = {}
            ok = True
            for j in MASK[name]:
                i += 1
                if i < n and 12 <= int(toks[i]) < 12 + LEVELS:
                    slots[j] = int(toks[i]) - 12
                else:
                    ok = False
                    break
            if ok:
                cmds.append((name, slots))
            i += 1
        else:
            i += 1  # stray param token without a command — skip
    return cmds


def command_dicts(cmds):
    out = []
    for name, slots in cmds:
        p = [0] * 16
        m = [False] * 16
        for j in MASK[name]:
            p[j] = int(slots.get(j, 0))
            m[j] = True
        out.append({"command_type": name, "parameters": p, "parameter_mask": m})
    return out


def build_dataset(n_target, cache):
    if cache and os.path.exists(cache):
        d = np.load(cache)
        if d["tokens"].shape[0] >= n_target:
            return d["tokens"][:n_target]
    from cadlib.extrude import CADSequence
    from cadlib.curves import Arc, Circle

    toks = []
    for f in sorted(glob.glob(os.path.join(_DEEPCAD, "data/cad_vec/*/*.h5"))):
        if len(toks) >= n_target:
            break
        try:
            with h5py.File(f, "r") as h:
                vec = h["vec"][:].astype(int)
            cad = CADSequence.from_vector(vec, is_numerical=True, n=256)
            cmds = _cmds(cad, Circle, Arc)
            enc = encode_tokens(cmds)
            if enc[1] != PAD:  # non-empty
                toks.append(enc)
            if len(toks) % 5000 == 0 and len(toks):
                print(f"  built {len(toks)}/{n_target}", flush=True)
        except Exception:
            continue
    toks = np.array(toks, np.int32)
    if cache:
        np.savez(cache, tokens=toks)
    return toks


# --- MLX causal-transformer language model ------------------------------------
class CausalBlock(mlxnn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.qkv = mlxnn.Linear(d, 3 * d)
        self.proj = mlxnn.Linear(d, d)
        self.n1 = mlxnn.LayerNorm(d)
        self.n2 = mlxnn.LayerNorm(d)
        self.fc1 = mlxnn.Linear(d, ff)
        self.fc2 = mlxnn.Linear(ff, d)
        self.d = d

    def __call__(self, x, mask):
        b, s, _ = x.shape
        h = self.n1(x)
        q, k, v = mx.split(self.qkv(h), 3, axis=-1)

        def sp(t):
            return mx.transpose(t.reshape(b, s, self.h, self.hd), (0, 2, 1, 3))

        q, k, v = sp(q), sp(k), sp(v)
        att = (q @ mx.transpose(k, (0, 1, 3, 2))) / (self.hd ** 0.5) + mask
        ctx = mx.softmax(att, axis=-1) @ v
        ctx = mx.transpose(ctx, (0, 2, 1, 3)).reshape(b, s, self.d)
        x = x + self.proj(ctx)
        return x + self.fc2(mlxnn.gelu(self.fc1(self.n2(x))))


class ARGPT(mlxnn.Module):
    def __init__(self, vocab=VOCAB, d=256, layers=6, heads=8, ff=1024, maxlen=MAX_LEN):
        super().__init__()
        self.embed = mlxnn.Embedding(vocab, d)
        self.pos = mx.zeros((1, maxlen, d))
        self.blocks = [CausalBlock(d, heads, ff) for _ in range(layers)]
        self.norm = mlxnn.LayerNorm(d)
        self.head = mlxnn.Linear(d, vocab)
        self.maxlen = maxlen

    def __call__(self, ids):
        s = ids.shape[1]
        mask = mx.where(mx.triu(mx.ones((s, s)), k=1) > 0,
                        mx.array(-1e9, mx.float32), mx.array(0.0, mx.float32))[None, None]
        x = self.embed(ids) + self.pos[:, :s, :]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.norm(x))


def sample(model, n, temperature=1.0, top_k=20):
    """Autoregressive batch sampling -> list of token lists (BOS-stripped)."""
    cur = mx.full((n, 1), BOS, dtype=mx.int32)
    done = np.zeros(n, bool)
    seqs = [[] for _ in range(n)]
    for _ in range(MAX_LEN - 1):
        logits = model(cur)[:, -1, :] / temperature  # [n, vocab]
        if top_k:
            kth = mx.sort(logits, axis=-1)[:, -top_k][:, None]
            logits = mx.where(logits < kth, mx.array(-1e9, mx.float32), logits)
        nxt = mx.random.categorical(logits)  # [n]
        mx.eval(nxt)
        nxt_np = np.array(nxt.tolist())
        for i in range(n):
            if not done[i]:
                t = int(nxt_np[i])
                seqs[i].append(t)
                if t == SEQ_EOS or t == CMD_TOK["EOS"]:
                    done[i] = True
        cur = mx.concatenate([cur, nxt[:, None].astype(mx.int32)], axis=1)
        if done.all():
            break
    return seqs


# --- honest validity through the real OCC kernel ------------------------------
def make_evaluator():
    from ll_gen.proposals.command_proposal import CommandSequenceProposal
    from ll_gen.disposal.command_executor import execute_command_proposal
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    def evaluate(toks):
        """Return (is_valid_solid, volume, bbox_signature) for one token sequence."""
        cmds = decode_tokens(toks)
        if not cmds:
            return False, 0.0, None
        try:
            shape = execute_command_proposal(CommandSequenceProposal(
                command_dicts=command_dicts(cmds), quantization_bits=8, normalization_range=2.0))
        except Exception:
            return False, 0.0, None
        if shape is None:
            return False, 0.0, None
        nsolids = 0
        e = TopExp_Explorer(shape, TopAbs_SOLID)
        while e.More():
            nsolids += 1
            e.Next()
        if nsolids < 1:
            return False, 0.0, None
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        vol = abs(props.Mass())
        if vol <= 1e-4:           # reject zero-volume degenerates
            return False, vol, None
        box = Bnd_Box()
        brepbndlib.Add(shape, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        sig = (round(xmax - xmin, 1), round(ymax - ymin, 1), round(zmax - zmin, 1))
        return True, vol, sig

    return evaluate


def measure_validity(model, evaluate, n, temperature, top_k):
    seqs = sample(model, n, temperature, top_k)
    valid, vols, sigs = 0, [], []
    for s in seqs:
        ok, vol, sig = evaluate(s)
        if ok:
            valid += 1
            vols.append(vol)
            sigs.append(sig)
    distinct = len(set(sigs))
    # cylinder-trap guard: fraction of valid samples that are the single most common shape
    top_frac = (Counter(sigs).most_common(1)[0][1] / len(sigs)) if sigs else 0.0
    return {"n": n, "validity": valid / n, "num_valid": valid, "num_distinct": distinct,
            "top_shape_frac": round(top_frac, 3),
            "mean_volume": float(np.mean(vols)) if vols else 0.0,
            "vol_p10_p90": [float(np.percentile(vols, 10)), float(np.percentile(vols, 90))] if vols else [0, 0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "train"], default="train")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", default=str(_REPO / "ll_gen/checkpoints"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache = f"{args.out}/ar_tokens_cache.npz"

    if args.mode == "probe":
        toks = build_dataset(64, None)
        model = ARGPT()
        out = model(mx.array(toks[:4]))
        ev = make_evaluator()
        seqs = sample(model, 4)
        v0 = [ev(s)[0] for s in seqs]
        print(f"probe: data {toks.shape}, logits {out.shape}, sampled-valid(untrained)={sum(v0)}/4", flush=True)
        return

    print("building/loading real DeepCAD command sequences ...", flush=True)
    toks = build_dataset(args.n_train, cache)
    print(f"dataset: {toks.shape[0]} sequences", flush=True)
    n_val = min(2000, toks.shape[0] // 10)
    tr = toks[n_val:]
    model = ARGPT()
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(ids):
        logits = model(ids[:, :-1])
        tgt = ids[:, 1:]
        mask = (tgt != PAD).astype(mx.float32)
        ce = mlxnn.losses.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
        ce = ce.reshape(tgt.shape) * mask
        return ce.sum() / mx.maximum(mask.sum(), 1)

    lg = mlxnn.value_and_grad(model, loss_fn)
    evaluate = make_evaluator()
    print("measuring untrained baseline validity ...", flush=True)
    base = measure_validity(model, evaluate, args.n_eval, args.temperature, args.top_k)
    print(f"BASELINE (untrained): {json.dumps(base)}", flush=True)

    n = tr.shape[0]
    best = -1.0
    for epoch in range(args.epochs):
        perm = np.random.permutation(n)
        tot = 0.0
        nb = 0
        for k in range(0, n, args.bs):
            idx = perm[k:k + args.bs]
            lv, g = lg(mx.array(tr[idx]))
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state, lv)
            tot += float(lv.item())
            nb += 1
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            m = measure_validity(model, evaluate, args.n_eval, args.temperature, args.top_k)
            print(f"epoch {epoch+1}/{args.epochs} loss={tot/max(nb,1):.4f} "
                  f"validity={m['validity']:.3f} valid={m['num_valid']} distinct={m['num_distinct']} "
                  f"mean_vol={m['mean_volume']:.3f}", flush=True)
            if m["validity"] > best:
                best = m["validity"]
                mx.save_safetensors(f"{args.out}/ar_generator_mlx.safetensors",
                                    dict(tree_flatten(model.parameters())))
        else:
            print(f"epoch {epoch+1}/{args.epochs} loss={tot/max(nb,1):.4f}", flush=True)

    final = measure_validity(model, evaluate, max(args.n_eval, 256), args.temperature, args.top_k)
    result = {"framework": "MLX", "model": "autoregressive CAD-command transformer (DeepCAD-style)",
              "task": "generate valid CAD construction programs", "dataset": "DeepCAD cad_vec",
              "n_train": int(tr.shape[0]), "epochs": args.epochs, "vocab": VOCAB,
              "sampling": {"temperature": args.temperature, "top_k": args.top_k},
              "validity_gate": "is_solid AND volume>1e-4 (non-degenerate), measured via real OCC kernel",
              "baseline_untrained_validity": round(base["validity"], 4),
              "best_validity": round(best, 4), "final": final,
              "checkpoint": f"{args.out}/ar_generator_mlx.safetensors"}
    with open(f"{args.out}/ar_generator_mlx_metrics.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print("AR_GENERATOR_DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
