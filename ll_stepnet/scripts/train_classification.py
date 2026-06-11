"""Train ll_stepnet's STEPForClassification on real DeepCAD models.

First real trained ll_stepnet checkpoint (make-real campaign). Task: predict a
model's face-count complexity class (<=4 / 5-6 / 7+ faces) from its CAD command
sequence — a GEOMETRIC label derived from the reconstructed solid, so the encoder
must learn the construction->geometry relationship rather than count tokens.

Pipeline: DeepCAD cad_vec (h5) -> cadlib CADSequence (absolute geometry) ->
executor-schema command_dicts -> OCC solid (for the face-count label) + flattened
token sequence (for the input). Trains stepnet.tasks.STEPForClassification and
writes the checkpoint + metrics to ll_stepnet/checkpoints/.

Requires the canonical DeepCAD data extracted under ``--deepcad`` (default
``resources/DeepCAD/data/cad_vec``; download via
http://www.cs.columbia.edu/cg/deepcad/data.tar) and the ll_gen executor +
pythonocc (conda 'cadling' env).

Run::

    python ll_stepnet/scripts/train_classification.py --n-train 5000 --epochs 30 \
        --device mps
"""

from __future__ import annotations

import argparse
import collections
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
matplotlib.use = lambda *a, **k: None  # neutralize cadlib's TkAgg switch
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
import h5py  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------
# DeepCAD cad_vec -> executor-schema translation (self-contained; validated
# 30/30 real models -> valid solids). Mirrors the executor's symmetric
# [0,255]<->[-2,2] param quantization.
# ----------------------------------------------------------------------------
LEVELS, RANGE, MAX_CMDS, NUM_SLOTS = 256, 2.0, 60, 16
MASK = {"LINE": [0, 1, 2, 3], "ARC": [0, 1, 2, 3, 4, 5], "CIRCLE": [0, 1, 2],
        "EXTRUDE": [0, 1, 2, 3, 4, 5, 6, 7], "SOL": [], "EOS": []}
CMD_TOK = {"SOL": 6, "LINE": 7, "ARC": 8, "CIRCLE": 9, "EXTRUDE": 10, "EOS": 11}
MAX_LEN = 256
BUCKETS = [(0, 4), (5, 6), (7, 9999)]
CLASS_NAMES = ["simple(<=4)", "box(5-6)", "complex(7+)"]


def _q_coord(grid: float) -> int:
    return int(np.clip(round(float(grid)), 0, LEVELS - 1))


def _q_value(v: float) -> int:
    return int(np.clip(round((float(v) + RANGE) / (2 * RANGE) * (LEVELS - 1)), 0, LEVELS - 1))


def _translate(cad, Circle, Arc):
    """CADSequence (absolute) -> list of (command_name, {slot: quant})."""
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
    toks = [1]  # BOS
    for name, slots in cmds:
        toks.append(CMD_TOK[name])
        for j in MASK[name]:
            toks.append(12 + int(slots.get(j, 0)))
    toks.append(2)  # EOS
    toks = toks[:MAX_LEN]
    toks += [0] * (MAX_LEN - len(toks))
    return toks


def _bucket(nf):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= nf <= hi:
            return i
    return len(BUCKETS) - 1


def load_dataset(files, limit, deps):
    CADSequence, Circle, Arc, CommandSequenceProposal, execute, face_count = deps
    toks, labels = [], []
    for f in files:
        if len(toks) >= limit:
            break
        try:
            with h5py.File(f, "r") as h:
                vec = h["vec"][:].astype(int)
            cad = CADSequence.from_vector(vec, is_numerical=True, n=256)
            cmds = _translate(cad, Circle, Arc)
            shape = execute(CommandSequenceProposal(
                command_dicts=_command_dicts(cmds), quantization_bits=8, normalization_range=2.0))
            if shape is None:
                continue
            nf = face_count(shape)
            if nf < 1:
                continue
            toks.append(_encode_tokens(cmds))
            labels.append(_bucket(nf))
        except Exception:
            continue
    return torch.tensor(toks, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def accuracy(model, tok, lab, dev, bs=256):
    model.eval()
    corr = tot = 0
    per = np.zeros((len(BUCKETS), 2), dtype=int)
    with torch.no_grad():
        for k in range(0, tok.shape[0], bs):
            pred = model(tok[k:k + bs].to(dev)).argmax(-1).cpu()
            y = lab[k:k + bs]
            corr += int((pred == y).sum()); tot += int(y.shape[0])
            for c in range(len(BUCKETS)):
                msk = y == c
                per[c, 1] += int(msk.sum()); per[c, 0] += int(((pred == y) & msk).sum())
    return corr / max(tot, 1), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepcad", default=str(_REPO / "resources/DeepCAD/data/cad_vec"))
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(_REPO / "ll_stepnet/checkpoints"))
    args = ap.parse_args()
    dev = args.device

    sys.path.insert(0, str(_REPO / "resources/DeepCAD"))
    from cadlib.extrude import CADSequence
    from cadlib.curves import Arc, Circle
    from ll_gen.proposals.command_proposal import CommandSequenceProposal
    from ll_gen.disposal.command_executor import execute_command_proposal
    from stepnet.tasks import STEPForClassification
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE

    def face_count(shape):
        c = 0
        e = TopExp_Explorer(shape, TopAbs_FACE)
        while e.More():
            c += 1
            e.Next()
        return c

    deps = (CADSequence, Circle, Arc, CommandSequenceProposal, execute_command_proposal, face_count)
    files = sorted(glob.glob(os.path.join(args.deepcad, "*/*.h5")))
    if not files:
        raise SystemExit(f"No cad_vec h5 files under {args.deepcad}; download DeepCAD data.tar first.")

    need = (args.n_train + args.n_val) * 3 + 4000
    print(f"building dataset from up to {need} DeepCAD models ...", flush=True)
    val_tok, val_lab = load_dataset(files[:need // 6], args.n_val, deps)
    tr_tok, tr_lab = load_dataset(files[need // 6:], args.n_train, deps)
    print(f"built {tr_tok.shape[0]} train / {val_tok.shape[0]} val", flush=True)
    print(f"train label dist: {dict(sorted(collections.Counter(tr_lab.tolist()).items()))} {CLASS_NAMES}", flush=True)

    model = STEPForClassification(num_classes=len(BUCKETS)).to(dev)
    counts = np.bincount(tr_lab.numpy(), minlength=len(BUCKETS)).astype(float)
    w = torch.tensor(counts.sum() / (len(BUCKETS) * np.clip(counts, 1, None)), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    majority = float(np.max(counts) / counts.sum())
    base_acc, _ = accuracy(model, val_tok, val_lab, dev)
    print(f"BASELINE val_acc={base_acc:.3f} (majority-class baseline={majority:.3f})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "stepnet_classifier.pt")
    n = tr_tok.shape[0]
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        nb = 0
        for k in range(0, n, args.bs):
            idx = perm[k:k + args.bs]
            loss = crit(model(tr_tok[idx].to(dev)), tr_lab[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        acc, _ = accuracy(model, val_tok, val_lab, dev)
        print(f"epoch {epoch+1}/{args.epochs} loss={tot/max(nb,1):.4f} val_acc={acc:.3f}", flush=True)
        if acc > best:
            best = acc
            torch.save({"model_state_dict": model.state_dict(),
                        "num_classes": len(BUCKETS), "class_names": CLASS_NAMES}, ckpt)

    acc, per = accuracy(model, val_tok, val_lab, dev)
    per_class = {CLASS_NAMES[c]: [round(per[c, 0] / max(per[c, 1], 1), 3), int(per[c, 1])]
                 for c in range(len(BUCKETS))}
    result = {"task": "STEP->face-count complexity class (3)", "dataset": "DeepCAD cad_vec",
              "n_train": int(tr_tok.shape[0]), "n_val": int(val_tok.shape[0]), "epochs": args.epochs,
              "baseline_majority_acc": round(majority, 3), "best_val_acc": round(best, 3),
              "final_val_acc": round(acc, 3), "per_class_acc": per_class, "checkpoint": ckpt}
    with open(os.path.join(args.out, "stepnet_classifier_metrics.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print("DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
