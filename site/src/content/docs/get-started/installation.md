---
title: Installation
description: Install the LatticeLabs Toolkit monorepo or individual packages, with the macOS-critical PyTorch/conda-forge caveat.
sidebar:
  order: 1
---

The LatticeLabs Toolkit is a monorepo. You can install everything at once through
the conda environment, or install individual packages with `pip`.

## Prerequisites

- **Python 3.9 – 3.12**
- **[Conda](https://docs.conda.io/)** (Miniconda or Miniforge recommended)

## macOS-critical: install PyTorch via conda-forge

:::danger[Do not `pip install torch`]
PyTorch **must** be installed via **conda-forge**, not pip. PyPI's `torch`
bundles its own `libomp.dylib`, which conflicts with conda's `llvm-openmp` and
crashes the process on macOS:

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

Always install PyTorch (and `torchvision`, `torchaudio`, `pytorch-geometric`)
from conda-forge. The `environment.yml` is the authoritative dependency source.
:::

## Full monorepo (recommended)

```bash
# Clone the repository
git clone https://github.com/LatticeLabsAI/ll_toolkit.git
cd ll_toolkit

# Create the conda environment (installs PyTorch, pythonocc, and all packages)
conda env create -f environment.yml
conda activate cadling
```

The environment installs `cadling`, `ll_stepnet`, and `geotoken` as editable
packages.

## Individual packages

Each package is independently installable with `pip` (run after activating the
conda environment so PyTorch is already present):

```bash
pip install -e ./cadling          # CAD document processing
pip install -e ./ll_stepnet       # STEP/BRep neural networks
pip install -e ./geotoken         # Geometric tokenizer
pip install -e ./ll_ocadr         # Optical CAD recognition
pip install -e ./ll_gen           # Generation orchestration
pip install -e ./ll_clouds        # Point-cloud processing
```

## Optional dependency groups

The root `pyproject.toml` defines extras you can install on top of the base
packages:

```bash
pip install -e ".[dev]"        # Testing, linting, docs
pip install -e ".[cad]"        # CAD processing (trimesh, networkx, numpy-stl)
pip install -e ".[ml]"         # ML (transformers, accelerate, einops)
pip install -e ".[vision]"     # Vision (opencv, easyocr, matplotlib)
pip install -e ".[hub]"        # HuggingFace Hub integration
pip install -e ".[drawings]"   # 2D drawings (DXF, PDF)
pip install -e ".[all]"        # Everything
```

:::note[pythonocc-core]
`pythonocc-core` (used by cadling's BRep backend and STEP geometry) is only
available through conda, not PyPI. It is included in `environment.yml`.
:::

## Next steps

Continue to the [Quickstart](/ll_toolkit/get-started/quickstart/) for a first
end-to-end run, or jump to any package's Overview from the sidebar.
