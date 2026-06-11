---
title: ll_brepnet (Implemented)
description: The planned B-Rep face-graph neural network is now implemented and trained — see the ll_brepnet package docs.
sidebar:
  label: ll_brepnet → done
  order: 1
  badge:
    text: Done
    variant: success
---

:::tip[Implemented]
`ll_brepnet` is **no longer a roadmap item — it is built, trained, and
documented.** The package extracts the coedge graph + UV-grid geometry from STEP
solids and trains a face-segmentation model end-to-end.

On the **full official split** of the **Fusion 360 Gallery segmentation dataset**
(official 8 manufacturing-feature classes, 5,366-solid held-out test split) it
reaches **test mIoU ≈ 0.828 / accuracy ≈ 0.947** — **exceeding** the BRepNet
paper's reported ~0.65–0.72 mIoU, achieved with an MIT-clean architecture. A
**native-MLX port** (parity-verified at 100% per-face agreement with PyTorch) runs the
trained GNN on Apple Silicon.
:::

See the package documentation:

- [Overview](/ll_toolkit/ll_brepnet/overview/) — architecture, results, per-class IoU
- [Installation](/ll_toolkit/ll_brepnet/installation/)
- [Usage](/ll_toolkit/ll_brepnet/usage/) — prepare data, train, and segment STEP files

It is an independent, MIT-licensed implementation built on the toolkit's own
B-Rep machinery (`cadling`); inspired by BRepNet
([arXiv:2104.00706](https://arxiv.org/abs/2104.00706)) and UV-Net but containing
no code from those projects.
