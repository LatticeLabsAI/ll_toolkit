---
title: cadling — Installation
description: Install cadling and its optional dependency groups, with the conda-forge PyTorch caveat and pythonocc-core note.
sidebar:
  label: Installation
  order: 2
---

cadling requires **Python 3.9+** and **conda** (for `pythonocc-core` and, on
macOS, PyTorch).

## Environment setup

```bash
# Create the conda environment (installs PyTorch via conda-forge)
conda env create -f environment.yml
conda activate cadling

# Install cadling in development mode
pip install -e ".[all]"
```

:::danger[macOS: PyTorch must come from conda-forge]
PyPI's `torch` bundles a `libomp.dylib` that conflicts with conda's OpenMP
runtime, causing `OMP: Error #15` crashes. Always install PyTorch (and the rest
of the ML stack) from conda-forge — never `pip install torch`. See the
monorepo [Installation](/ll_toolkit/get-started/installation/) page.
:::

## Optional dependency groups

```bash
pip install -e ".[dev]"      # pytest, black, ruff, mypy, pre-commit
pip install -e ".[cad]"      # numpy-stl, trimesh, networkx
pip install -e ".[ml]"       # transformers (PyTorch via conda only)
pip install -e ".[vision]"   # transformers, easyocr, opencv-python
pip install -e ".[all]"      # Everything above
```

## pythonocc-core

The BRep backend and STEP geometry use `pythonocc-core`, which is only available
through conda (not PyPI) and is included in `environment.yml`. Tests that need it
are marked `requires_pythonocc` and skip automatically when it is absent.

## Neural integration (ll_stepnet)

cadling integrates [ll_stepnet](/ll_toolkit/ll_stepnet/overview/) for STEP neural
processing. It is installed as an editable dependency via `environment.yml`, or
manually:

```bash
cd ../ll_stepnet
pip install -e .
```

## Next

Continue to [Usage](/ll_toolkit/cadling/usage/).
