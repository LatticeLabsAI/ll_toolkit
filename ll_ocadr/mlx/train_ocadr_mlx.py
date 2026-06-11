"""ll_ocadr in MLX on Apple Silicon — FAITHFUL geometry tower + retrain.

Supersedes the earlier maxpool stand-in encoder. The trainable 3D tower is now the
REAL ll_ocadr architecture ported to MLX and forward-parity-proven against the
PyTorch encoders (see faithful_tower_mlx.py: GeometryNet PointNet++ + ShapeNet
Point-BERT + linear projector). It maps a CAD point cloud (coords + normals) into
256 mesh tokens in a 4-bit Qwen2's embedding space (mlx-lm ``input_embeddings``);
the LLM base stays frozen/quantized while LoRA adapters + the tower train jointly so
the LLM learns to attend to the injected mesh tokens (LLaVA-style).

There are NO pretrained ll_ocadr weights (the configured model was never trained and
can't run here), so this is an architecture-faithful RETRAIN, not a parity claim —
the faithfulness proof lives in faithful_tower_mlx.py. Task/data/metrics are held
IDENTICAL to the maxpool run for a direct comparison: balanced 3-way class from the point
cloud, class-only target, honest SHUFFLED-mesh baseline (feed the wrong mesh).

Modes: probe | train.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

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
from mlx_lm import load as mlx_load  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: E402

_REPO = "/Users/ryanoboyle/LatticeLabs_toolkit"
_DEEPCAD = f"{_REPO}/resources/DeepCAD"
sys.path.insert(0, _DEEPCAD)
sys.path.insert(0, f"{_REPO}/resources/ll_gen_proof")
sys.path.insert(0, f"{_REPO}/ll_ocadr/mlx")

from faithful_tower_mlx import FaithfulTower, precompute_geom  # noqa: E402

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh  # noqa: E402
from OCC.Core.TopExp import TopExp_Explorer  # noqa: E402
from OCC.Core.TopAbs import TopAbs_FACE  # noqa: E402
from OCC.Core.TopoDS import topods  # noqa: E402
from OCC.Core.BRep import BRep_Tool  # noqa: E402
from OCC.Core.TopLoc import TopLoc_Location  # noqa: E402

NUM_POINTS, NUM_MESH_TOKENS = 2048, 256
CLASS_NAMES = ["simple", "box", "complex"]


def _bucket(nf: int) -> int:
    return 0 if nf <= 4 else (1 if nf <= 6 else 2)


def _tri_nodes(tri, k):
    t = tri.Triangle(k)
    try:
        a, b, c = t.Get()
        return a, b, c
    except Exception:
        return t.Value(1), t.Value(2), t.Value(3)


def _sample_points(shape, N=NUM_POINTS):
    """Sample N (coords, normals) from the triangulated solid. Normals are
    area-weighted vertex normals — the real encoders consume xyz + normals (6 ch)."""
    BRepMesh_IncrementalMesh(shape, 0.05)
    verts, tris = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    nfaces = 0
    while exp.More():
        nfaces += 1
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            tr = loc.Transformation()
            base = len(verts)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(tr)
                verts.append([p.X(), p.Y(), p.Z()])
            for k in range(1, tri.NbTriangles() + 1):
                a, b, c = _tri_nodes(tri, k)
                tris.append([base + a - 1, base + b - 1, base + c - 1])
        exp.Next()
    if len(verts) < 3 or not tris:
        return None, None, 0
    verts = np.asarray(verts, np.float32)
    tris = np.asarray(tris, np.int64)
    nrm = np.zeros_like(verts)
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # area-weighted face normals
    for j in range(3):
        np.add.at(nrm, tris[:, j], fn)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(ln > 1e-8, nrm / (ln + 1e-8), np.array([0.0, 0.0, 1.0], np.float32))
    c = verts.mean(0)
    s = np.abs(verts - c).max() + 1e-6
    verts = (verts - c) / s
    idx = np.random.choice(len(verts), N, replace=len(verts) < N)
    return verts[idx].astype(np.float32), nrm[idx].astype(np.float32), nfaces


def build_dataset(n_target, deps, cache):
    if cache and os.path.exists(cache):
        d = np.load(cache)
        if int(d["classes"].shape[0]) >= n_target:
            return {k: d[k][:n_target] for k in
                    ("coords", "normals", "sa1", "sa2xyz", "sa2idx", "classes")}
    CADSequence, command_dicts, execute = deps
    files = sorted(glob.glob(os.path.join(_DEEPCAD, "data/cad_vec/*/*.h5")))
    coords, normals, sa1, sa2xyz, sa2idx, classes = [], [], [], [], [], []
    for f in files:
        if len(classes) >= n_target:
            break
        try:
            with h5py.File(f, "r") as h:
                vec = h["vec"][:].astype(int)
            cad = CADSequence.from_vector(vec, is_numerical=True, n=256)
            shape = execute(command_dicts(cad))
            if shape is None:
                continue
            pc, nrm, nf = _sample_points(shape)
            if pc is None or nf < 1:
                continue
            pre = precompute_geom(pc, nrm)
            coords.append(pc)
            normals.append(nrm)
            sa1.append(pre["sa1_grouped"])
            sa2xyz.append(pre["sa2_grouped_xyz"])
            sa2idx.append(pre["sa2_group_idx"])
            classes.append(_bucket(nf))
            if len(classes) % 200 == 0:
                print(f"  built {len(classes)}/{n_target}", flush=True)
        except Exception:
            continue
    out = {"coords": np.stack(coords), "normals": np.stack(normals),
           "sa1": np.stack(sa1), "sa2xyz": np.stack(sa2xyz),
           "sa2idx": np.stack(sa2idx), "classes": np.array(classes)}
    if cache:
        np.savez(cache, **out)
    return out


class OCADRMLX(mlxnn.Module):
    """Trainable faithful tower + LoRA-adapted (otherwise frozen) LLM, so mlx
    value_and_grad differentiates tower params AND LoRA params together."""

    def __init__(self, tower, llm):
        super().__init__()
        self.tower = tower
        self.llm = llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "train"], default="train")
    ap.add_argument("--llm", default="mlx-community/Qwen2-0.5B-Instruct-4bit")
    ap.add_argument("--n-train", type=int, default=2200)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-layers", type=int, default=8)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--out", default=f"{_REPO}/ll_ocadr/checkpoints")
    args = ap.parse_args()

    from cadlib.extrude import CADSequence
    from ll_gen.proposals.command_proposal import CommandSequenceProposal
    from ll_gen.disposal.command_executor import execute_command_proposal
    import deepcad_supervised_train as M

    def command_dicts(cad):
        cmds = M.translate(cad)
        return CommandSequenceProposal(command_dicts=M.cad_to_command_dicts(cmds),
                                       quantization_bits=8, normalization_range=2.0)
    deps = (CADSequence, command_dicts, execute_command_proposal)

    print(f"loading {args.llm} + LoRA (rank={args.lora_rank}, layers={args.lora_layers}) ...", flush=True)
    llm, tok = mlx_load(args.llm)
    llm.freeze()
    linear_to_lora_layers(llm, args.lora_layers, {"rank": args.lora_rank, "scale": 20.0, "dropout": 0.0})
    H = llm.args.hidden_size
    tower = FaithfulTower(H, shape_depth=4, shape_heads=8)  # 0.5B faithful config
    model = OCADRMLX(tower, llm)
    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable (faithful tower + LoRA): {n_trainable/1e6:.2f}M; LLM base frozen+quantized", flush=True)

    prompt_ids = mx.array(tok.encode("Describe this CAD part: "))
    Lp = int(prompt_ids.shape[0])

    def mesh_tokens(d, idx):
        b = {"sa1_grouped": mx.array(d["sa1"][idx]), "sa2_grouped_xyz": mx.array(d["sa2xyz"][idx]),
             "sa2_group_idx": mx.array(d["sa2idx"][idx].astype(np.int32)),
             "coords": mx.array(d["coords"][idx]), "normals": mx.array(d["normals"][idx])}
        return model.tower(b)  # [B,256,H]

    def make_batch(d, idx, texts):
        mesh = mesh_tokens(d, idx)
        prompt_emb = model.llm.model.embed_tokens(prompt_ids)
        resp = [mx.array(tok.encode(t + tok.eos_token)) for t in texts]
        maxLr = max(int(r.shape[0]) for r in resp)
        T = mesh.shape[1]
        seqs, labels = [], []
        for b in range(len(texts)):
            r = resp[b]
            Lr = int(r.shape[0])
            remb = model.llm.model.embed_tokens(r)
            emb = mx.concatenate([prompt_emb, mesh[b], remb], axis=0)
            pad = maxLr - Lr
            if pad:
                emb = mx.concatenate([emb, mx.zeros((pad, H))], axis=0)
            seqs.append(emb)
            labels.append([-100] * (Lp + T) + r.tolist() + [-100] * pad)
        return mx.stack(seqs), mx.array(np.array(labels)), mesh

    def loss_fn(d, idx, texts, classes):
        emb, labels, mesh = make_batch(d, idx, texts)
        logits = model.llm(mx.zeros(emb.shape[:2], dtype=mx.int32), input_embeddings=emb)
        lg = logits[:, :-1].reshape(-1, logits.shape[-1])
        tg = labels[:, 1:].reshape(-1)
        mask = tg != -100
        ce = mlxnn.losses.cross_entropy(lg, mx.where(mask, tg, 0), reduction="none") * mask
        lm_loss = ce.sum() / mx.maximum(mask.sum(), 1)
        aux = mlxnn.losses.cross_entropy(model.tower.aux_logits(mesh), mx.array(classes), reduction="mean")
        return lm_loss + 0.5 * aux

    if args.mode == "probe":
        d = build_dataset(6, deps, None)
        model.tower.train()
        lv, grads = mlxnn.value_and_grad(model, loss_fn)(
            d, np.arange(4), [CLASS_NAMES[c] for c in d["classes"][:4]], d["classes"][:4])
        gnorm = float(mx.sqrt(sum((g * g).sum() for _, g in tree_flatten(grads))).item())
        mx.eval(lv)
        print(f"probe loss={float(lv.item()):.4f} grad_norm={gnorm:.4f}", flush=True)
        return

    os.makedirs(args.out, exist_ok=True)
    cache = f"{args.out}/ocadr_faithful_data.npz"
    print("building/loading dataset (N=2048 pts + normals + cached FPS grouping) ...", flush=True)
    data = build_dataset(args.n_train + args.n_val, deps, cache)
    allcl = data["classes"]
    rng = np.random.default_rng(0)
    per = int(np.bincount(allcl, minlength=3).min())
    keep = np.concatenate([rng.permutation(np.where(allcl == c)[0])[:per] for c in range(3)])
    rng.shuffle(keep)
    data = {k: v[keep] for k, v in data.items()}
    allcl = data["classes"]
    nval = max(len(keep) // 6, 60)
    val_idx = np.arange(nval)
    tr_idx = np.arange(nval, len(keep))
    vcl, tcl = allcl[val_idx], allcl[tr_idx]
    majority = float(np.bincount(vcl, minlength=3).max() / len(vcl))
    print(f"balanced {len(tr_idx)} train / {len(val_idx)} val; per-class={per}; "
          f"val dist={np.bincount(vcl, minlength=3).tolist()} (majority={majority:.3f})", flush=True)

    opt = optim.Adam(learning_rate=args.lr)
    lg_fn = mlxnn.value_and_grad(model, loss_fn)

    def greedy_class(d, idx):
        B = len(idx)
        model.tower.eval()
        mesh = mesh_tokens(d, idx)
        pe = mx.broadcast_to(model.llm.model.embed_tokens(prompt_ids)[None], (B, Lp, H))
        cur = mx.concatenate([pe, mesh], axis=1)
        toks = [[] for _ in range(B)]
        for _ in range(8):
            logits = model.llm(mx.zeros(cur.shape[:2], dtype=mx.int32), input_embeddings=cur)
            nxt = mx.argmax(logits[:, -1], axis=-1)
            mx.eval(nxt)
            cur = mx.concatenate([cur, model.llm.model.embed_tokens(nxt)[:, None]], axis=1)
            for b in range(B):
                toks[b].append(int(nxt[b].item()))
        preds = np.full(B, -1, int)
        for b in range(B):
            txt = tok.decode(toks[b]).lower()
            pos = {nm: txt.find(nm) for nm in CLASS_NAMES if nm in txt}
            if pos:
                preds[b] = CLASS_NAMES.index(min(pos, key=pos.get))
        return preds

    def class_acc(d, idx, classes, shuffle=False):
        mesh_idx = idx[np.random.permutation(len(idx))] if shuffle else idx
        correct = 0
        for k in range(0, len(idx), 16):
            sl = slice(k, k + 16)
            correct += int((greedy_class(d, mesh_idx[sl]) == classes[sl]).sum())
        return correct / len(idx)

    def aux_acc(d, idx, classes):
        model.tower.eval()
        correct = 0
        for k in range(0, len(idx), 16):
            sl = slice(k, k + 16)
            mesh = mesh_tokens(d, idx[sl])
            pred = np.array(mx.argmax(model.tower.aux_logits(mesh), axis=1).tolist())
            correct += int((pred == classes[sl]).sum())
        return correct / len(idx)

    best = -1.0
    n = len(tr_idx)
    for epoch in range(args.epochs):
        model.tower.train()
        perm = np.random.permutation(n)
        tot = 0.0
        nb = 0
        for k in range(0, n, args.bs):
            bidx = tr_idx[perm[k:k + args.bs]]
            lv, grads = lg_fn(data, bidx, [CLASS_NAMES[c] for c in data["classes"][bidx]],
                              data["classes"][bidx])
            opt.update(model, grads)
            mx.eval(model.trainable_parameters(), opt.state, lv)
            tot += float(lv.item())
            nb += 1
        acc = class_acc(data, val_idx, vcl)
        shuf = class_acc(data, val_idx, vcl, shuffle=True)
        aux = aux_acc(data, val_idx, vcl)
        print(f"epoch {epoch+1}/{args.epochs} loss={tot/max(nb,1):.4f} "
              f"llm_gen_acc={acc:.3f} shuffled={shuf:.3f} encoder_aux_acc={aux:.3f}", flush=True)
        if acc > best:
            best = acc
            mx.save_safetensors(f"{args.out}/ocadr_mlx.safetensors",
                                dict(tree_flatten(model.trainable_parameters())))

    result = {"framework": "MLX (Apple Silicon)", "port": "faithful PointNet++/Point-BERT tower (parity-proven), retrained",
              "task": "CAD point-cloud -> class (simple/box/complex), 3-way balanced",
              "llm": args.llm, "llm_base": "frozen + 4-bit quantized",
              "trainable": "faithful GeometryNet+ShapeNet+projector + LoRA", "trainable_params_M": round(n_trainable / 1e6, 2),
              "n_train": int(len(tr_idx)), "n_val": int(len(val_idx)), "epochs": args.epochs,
              "num_mesh_tokens": NUM_MESH_TOKENS, "majority_baseline": round(majority, 3),
              "encoder_mesh_read_acc": round(aux_acc(data, val_idx, vcl), 3),
              "llm_generation_acc": round(class_acc(data, val_idx, vcl), 3),
              "shuffled_mesh_baseline": round(class_acc(data, val_idx, vcl, shuffle=True), 3),
              "checkpoint": f"{args.out}/ocadr_mlx.safetensors"}
    with open(f"{args.out}/ocadr_mlx_metrics.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print("OCADR_MLX_FAITHFUL_DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
