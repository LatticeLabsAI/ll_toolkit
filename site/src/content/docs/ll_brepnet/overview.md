---
title: ll_brepnet — Overview
description: A B-Rep face-graph neural network for CAD solid-model segmentation — coedge message passing + UV-grid geometry, trained on the Fusion 360 Gallery set.
sidebar:
  label: Overview
  order: 1
  badge:
    text: Trained
    variant: success
---

**ll_brepnet** is a B-Rep **face-segmentation** neural network. It operates
directly on the boundary representation of a CAD solid — faces and edges
connected through oriented *coedges* (half-edges carrying next / previous / mate
/ parent-face / parent-edge adjacency) — and fuses that topology with UV-grid
surface and curve geometry to predict a semantic segment label for every face.

It is an **independent, MIT-licensed** package built on the toolkit's own B-Rep
machinery (`cadling`). It is *inspired by* BRepNet
([arXiv:2104.00706](https://arxiv.org/abs/2104.00706)) and UV-Net, but contains
no code from those projects — see `ll_brepnet/ATTRIBUTION.md`.

## Modules

| Module | Responsibility |
|---|---|
| `ll_brepnet.pipelines.extract_brepnet_data_from_step` | STEP → unit-box scale → coedge graph + per-face/edge features + UV-grids → `.npz` |
| `ll_brepnet.pipelines.extract_brepnet_data_from_json` | Same record from a precomputed JSON topology |
| `ll_brepnet.pipelines.build_dataset_file` / `quickstart` | Manifest building (split + train-only standardization); Fusion 360 orchestration |
| `ll_brepnet.dataloaders` | `BRepDataset`, offset-aware `brep_collate_fn`, `MaxNumFacesSampler`, `BRepDataModule` |
| `ll_brepnet.models` | UV-Net surface/curve encoders + the `LLBRepNet` LightningModule |
| `ll_brepnet.train` / `ll_brepnet.eval` | pytorch-lightning training; folder/checkpoint inference → per-face logits |

## Architecture

```text
STEP solid
  → unit-box scale → coedge graph (next/prev/mate/face/edge incidence)
  → per-face features (surface-type one-hot + area) + face UV-grid [7,U,V]
  → per-edge features (curve-type one-hot + length + convexity) + edge U-grid [6,U]
        │
        ▼
  UV-Net surface/curve encoders ⊕ scalar features
        │  (gather per coedge: face_repr[c2f] ⊕ edge_repr[c2e] ⊕ reversed)
        ▼
  coedge message passing (BRepNetEncoder, reused from cadling) → coedge→face mean pool
        ▼
  per-face segmentation head → [num_faces, num_classes]
```

## Results

Trained on the **full official split** of the **Fusion 360 Gallery segmentation
dataset (s2.0.0)** — 27,282 train / 3,032 validation / 5,366 test solids, the
official 8 manufacturing-feature classes, 30 epochs on an Apple-Silicon GPU (MPS).

**Held-out test split (5,366 real solids):**

| Metric | Value |
|---|---|
| **mean IoU (macro)** | **0.828** |
| accuracy | 0.947 |

Per-class IoU:

| Class | IoU | | Class | IoU |
|---|---|---|---|---|
| Fillet | 0.98 | | RevolveSide | 0.83 |
| ExtrudeSide | 0.93 | | CutSide | 0.79 |
| ExtrudeEnd | 0.91 | | CutEnd | 0.74 |
| Chamfer | 0.89 | | RevolveEnd | 0.55 |

These are **real, reproducible numbers** (see **Usage**) — and **0.828 mIoU
exceeds the BRepNet paper's reported ~0.65–0.72 on s2.0.0**, achieved with the
MIT-clean reused-coedge-encoder architecture rather than the paper's kernel
convolution. (An earlier 4,800-solid subset run reached 0.709; training on the
full set raised it to 0.828 and lifted the rare **RevolveEnd** class from 0.11
to 0.55.)

## Native MLX (Apple Silicon)

A native-MLX port (`ll_brepnet/mlx/train_brepnet_mlx.py`) reproduces the exact
architecture — UV-Net encoders with BatchNorm + the coedge message-passing encoder
(input projection, residual coedge convolutions with LayerNorm, output projection) — and
**converts the real trained checkpoint** into MLX. Driving both models from the same
`BRepDataset`, it is verified at **100% per-face argmax agreement** with PyTorch and an
**identical mIoU (0.835 on the measured subset)**, so the MLX model *is* the trained GNN
running on Apple Silicon (the conversion handles Conv `OIHW→OHWI`/`OIW→OWI` permutes and
inference-mode BatchNorm running stats):

```bash
python {{script.ll_brepnet.train}} --mode parity   # convert real weights + verify
```

:::note[Scope]
The repository ships the training/eval code and this reproducible recipe; the
trained checkpoint (~4.8 MB) is produced by the run in **Usage** rather than
committed. **RevolveEnd** remains the hardest class (rarest in the dataset);
longer training would narrow the gap further.
:::

Use the sidebar for **Installation** and **Usage**.
