"""Train StructuredDiffusion's GeometryCodec on real DeepCAD B-rep geometry.

Make-real campaign (ll_gen diffusion path). The GeometryCodec is the latent<->
geometry autoencoder the diffusion denoisers operate in; it had never been
trained on real geometry. This trains it: DeepCAD cad_vec -> solid (validated
executor) -> sample each face as an 8x8x3 UV grid and each edge as a 12x3
polyline (unit-cube normalized, masked) -> codec masked-MSE reconstruction.
Writes the checkpoint + metrics to ll_gen/checkpoints/.

Result (4000 train / 600 val, 60 epochs MPS): recon MSE 0.40 (untrained) ->
0.0003 (trained). NOTE: this trains the codec only; the diffusion *denoisers*
do not yet converge (they plateau at predict-zero), so the generator does not
yet produce valid CAD — see docs/2026-06-10-ll-gen-make-real-findings.md.

Requires DeepCAD cad_vec under ``--deepcad`` (data.tar from
http://www.cs.columbia.edu/cg/deepcad/) + the ll_gen executor + pythonocc.

Run::

    python ll_gen/scripts/train_diffusion_codec.py --n-train 4000 --epochs 60 \
        --device mps
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.use = lambda *a, **k: None
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import h5py  # noqa: E402
import torch  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]

LEVELS, RANGE = 256, 2.0
MASK = {"LINE": [0, 1, 2, 3], "ARC": [0, 1, 2, 3, 4, 5], "CIRCLE": [0, 1, 2],
        "EXTRUDE": [0, 1, 2, 3, 4, 5, 6, 7], "SOL": [], "EOS": []}
NUM_FACES, NUM_EDGES, UV, MPTS = 8, 12, 8, 12


def _q_coord(g):
    return int(np.clip(round(float(g)), 0, LEVELS - 1))


def _q_value(v):
    return int(np.clip(round((float(v) + RANGE) / (2 * RANGE) * (LEVELS - 1)), 0, LEVELS - 1))


def _command_dicts(cad, Circle, Arc):
    out = []

    def emit(name, slots):
        p = [0] * 16
        m = [False] * 16
        for j in MASK[name]:
            p[j] = int(slots.get(j, 0))
            m[j] = True
        out.append({"command_type": name, "parameters": p, "parameter_mask": m})

    for ext in cad.seq:
        for loop in ext.profile.children:
            emit("SOL", {})
            for cv in loop.children:
                if isinstance(cv, Circle):
                    r_mag = float(cv.radius) / (LEVELS - 1) * 2.0 * RANGE
                    emit("CIRCLE", {0: _q_coord(cv.center[0]), 1: _q_coord(cv.center[1]), 2: _q_value(r_mag)})
                elif isinstance(cv, Arc):
                    s, e, c = cv.start_point, cv.end_point, cv.center
                    emit("ARC", {0: _q_coord(s[0]), 1: _q_coord(s[1]), 2: _q_coord(e[0]),
                                 3: _q_coord(e[1]), 4: _q_coord(c[0]), 5: _q_coord(c[1])})
                else:
                    s, e = cv.start_point, cv.end_point
                    emit("LINE", {0: _q_coord(s[0]), 1: _q_coord(s[1]), 2: _q_coord(e[0]), 3: _q_coord(e[1])})
        depth = abs(float(ext.extent_one)) + abs(float(ext.extent_two))
        emit("EXTRUDE", {0: _q_value(float(np.clip(depth * 4.0, 0.3, 2.0)))})
    emit("EOS", {})
    return out


def _extract_geometry(shape, occ):
    (TopExp_Explorer, TopAbs_FACE, TopAbs_EDGE, topods, BRepAdaptor_Surface,
     BRepAdaptor_Curve, Bnd_Box, brepbndlib) = occ
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
    scale = 2.0 / max(xmax - xmin, ymax - ymin, zmax - zmin, 1e-6)

    def n(p):
        return ((np.array([p.X(), p.Y(), p.Z()]) - center) * scale).astype(np.float32)

    fg = np.zeros((NUM_FACES, UV, UV, 3), np.float32)
    fm = np.ones((NUM_FACES,), bool)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    fi = 0
    while exp.More() and fi < NUM_FACES:
        s = BRepAdaptor_Surface(topods.Face(exp.Current()))
        u0, u1, v0, v1 = s.FirstUParameter(), s.LastUParameter(), s.FirstVParameter(), s.LastVParameter()
        for iu in range(UV):
            for iv in range(UV):
                fg[fi, iu, iv] = n(s.Value(u0 + (u1 - u0) * iu / (UV - 1), v0 + (v1 - v0) * iv / (UV - 1)))
        fm[fi] = False
        fi += 1
        exp.Next()
    if fi == 0:
        return None
    ep = np.zeros((NUM_EDGES, MPTS, 3), np.float32)
    em = np.ones((NUM_EDGES,), bool)
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    ei = 0
    while exp.More() and ei < NUM_EDGES:
        try:
            c = BRepAdaptor_Curve(topods.Edge(exp.Current()))
            t0, t1 = c.FirstParameter(), c.LastParameter()
            for k in range(MPTS):
                ep[ei, k] = n(c.Value(t0 + (t1 - t0) * k / (MPTS - 1)))
            em[ei] = False
            ei += 1
        except Exception:
            pass
        exp.Next()
    return fg, ep, fm, em


def load_geometry(files, limit, deps):
    CADSequence, Circle, Arc, CommandSequenceProposal, execute, occ = deps
    fg, ep, fm, em = [], [], [], []
    for f in files:
        if len(fg) >= limit:
            break
        try:
            with h5py.File(f, "r") as h:
                vec = h["vec"][:].astype(int)
            cad = CADSequence.from_vector(vec, is_numerical=True, n=256)
            shape = execute(CommandSequenceProposal(
                command_dicts=_command_dicts(cad, Circle, Arc), quantization_bits=8, normalization_range=2.0))
            if shape is None:
                continue
            g = _extract_geometry(shape, occ)
            if g is None:
                continue
            fg.append(g[0]); ep.append(g[1]); fm.append(g[2]); em.append(g[3])
        except Exception:
            continue
    if not fg:
        return None
    return (torch.from_numpy(np.stack(fg)), torch.from_numpy(np.stack(ep)),
            torch.from_numpy(np.stack(fm)), torch.from_numpy(np.stack(em)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepcad", default=str(_REPO / "resources/DeepCAD/data/cad_vec"))
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(_REPO / "ll_gen/checkpoints"))
    args = ap.parse_args()
    dev = args.device

    sys.path.insert(0, str(_REPO / "resources/DeepCAD"))
    from cadlib.extrude import CADSequence
    from cadlib.curves import Arc, Circle
    from ll_gen.proposals.command_proposal import CommandSequenceProposal
    from ll_gen.disposal.command_executor import execute_command_proposal
    from ll_gen.training.run import build_generator
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    occ = (TopExp_Explorer, TopAbs_FACE, TopAbs_EDGE, topods, BRepAdaptor_Surface,
           BRepAdaptor_Curve, Bnd_Box, brepbndlib)
    deps = (CADSequence, Circle, Arc, CommandSequenceProposal, execute_command_proposal, occ)

    gen = build_generator("diffusion", dev)
    if gen._model is None and hasattr(gen, "_init_model"):
        gen._init_model()
    codec = gen._model.geometry_codec.to(dev)

    files = sorted(glob.glob(os.path.join(args.deepcad, "*/*.h5")))
    if not files:
        raise SystemExit(f"No cad_vec h5 files under {args.deepcad}; download DeepCAD data.tar first.")
    need = (args.n_train + args.n_val) * 4 + 4000
    print(f"extracting geometry from up to {need} DeepCAD solids ...", flush=True)
    vfg, vep, vfm, vem = (t.to(dev) for t in load_geometry(files[:need // 6], args.n_val, deps))
    tfg, tep, tfm, tem = load_geometry(files[need // 6:], args.n_train, deps)
    print(f"extracted {tfg.shape[0]} train / {vfg.shape[0]} val solids on {dev}", flush=True)

    opt = torch.optim.Adam(codec.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "diffusion_codec.pt")
    n = tfg.shape[0]
    best = 1e9
    for epoch in range(args.epochs):
        codec.train()
        perm = torch.randperm(n)
        tot = 0.0
        nb = 0
        for k in range(0, n, args.bs):
            idx = perm[k:k + args.bs]
            out = codec.reconstruction_loss(tfg[idx].to(dev), tep[idx].to(dev),
                                            tfm[idx].to(dev), tem[idx].to(dev))
            loss = out["total_recon_loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        codec.eval()
        with torch.no_grad():
            v = codec.reconstruction_loss(vfg, vep, vfm, vem)
        vtot = float(v["total_recon_loss"])
        print(f"epoch {epoch+1}/{args.epochs} train_loss={tot/max(nb,1):.5f} "
              f"val_face_mse={float(v['face_recon_loss']):.5f} val_edge_mse={float(v['edge_recon_loss']):.5f}",
              flush=True)
        if vtot < best:
            best = vtot
            torch.save({"codec_state_dict": codec.state_dict()}, ckpt)

    result = {"n_train": int(tfg.shape[0]), "n_val": int(vfg.shape[0]), "epochs": args.epochs,
              "best_val_recon_mse": round(best, 6), "checkpoint": ckpt,
              "latent_dim": codec.latent_dim}
    with open(os.path.join(args.out, "diffusion_codec_metrics.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print("DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
