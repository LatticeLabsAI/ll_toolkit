---
title: ll_brepnet — Installation
description: Install ll_brepnet — PyTorch + pythonocc-core + occwl via conda, plus pytorch-lightning.
sidebar:
  label: Installation
  order: 2
---

`ll_brepnet` needs PyTorch, `pythonocc-core`, and `occwl`, which on macOS must
come from conda to avoid the OpenMP/`libomp` conflict (see the repo `CLAUDE.md`).
It also uses `pytorch-lightning` for training and `cadling` for the shared B-Rep
extraction machinery.

## Option 1 — reuse the `cadling` environment (recommended)

The `cadling` conda environment already provides PyTorch, `pythonocc-core`, and
`occwl`. Add the training deps and install the package editable:

```bash
conda activate cadling
pip install pytorch-lightning tensorboard
pip install -e ./ll_brepnet
```

## Option 2 — standalone environment

```bash
conda env create -f ll_brepnet/environment.yaml
conda activate ll-brepnet
```

## Verify

```bash
python -c "import ll_brepnet; print(ll_brepnet.__version__)"
# 0.1.0
```

Run the test suite (skips automatically without pythonocc / torch):

```bash
pytest ll_brepnet/tests -q          # fast tests
pytest ll_brepnet/tests -q -m ""    # include the slow end-to-end test
```

## Dependencies

| Dependency | Source | Why |
|---|---|---|
| `pytorch` | conda-forge | model + training (conda only, OpenMP safety) |
| `pythonocc-core` | conda-forge | STEP loading + B-Rep traversal |
| `occwl` | pip | UV-grid sampling (`uvgrid` / `ugrid`) |
| `pytorch-lightning`, `torchmetrics`, `tensorboard` | pip | training loop, mIoU/accuracy, logging |
| `cadling` | editable (monorepo) | coedge extraction + reused encoder |
| `scikit-learn` | conda/pip | train/val/test splitting |
