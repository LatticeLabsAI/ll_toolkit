---
title: Development
description: Development setup, coding standards, testing, and the git workflow for contributing to the LatticeLabs Toolkit.
sidebar:
  label: Development
  order: 1
---

This guide covers contributing across the monorepo. It generalizes cadling's
development guide to all packages.

## Setup

```bash
git clone https://github.com/LatticeLabsAI/ll_toolkit.git
cd ll_toolkit

# Conda environment (installs PyTorch via conda-forge, pythonocc, packages)
conda env create -f environment.yml
conda activate cadling

# Editable installs (as needed)
pip install -e ./cadling ./ll_stepnet ./geotoken ./ll_ocadr ./ll_gen ./ll_clouds
```

:::danger[Never `pip install torch`]
PyTorch and the rest of the ML stack must come from **conda-forge**. PyPI's torch
bundles a conflicting `libomp.dylib` that crashes on macOS with `OMP: Error #15`.
Each test `conftest.py` imports torch first and sets `OMP_NUM_THREADS=1`; torch-
using test modules use `pytest.importorskip("torch")` at module level.
:::

## Coding standards

- **Black** — line length 88.
- **Ruff** — `E, W, F, I, N, UP, B, C4` rules.
- **mypy** — Python 3.9 target.
- **Type hints** required on public functions/methods.
- **Google-style docstrings** on public APIs.
- `from __future__ import annotations` at the top of modules.
- `_log = logging.getLogger(__name__)` for logging.
- Lazy-import heavy deps (torch, pythonocc, trimesh) behind availability checks.

```bash
black <pkg>/ tests/
ruff check <pkg>/ tests/
mypy <pkg>/
```

## Testing

```bash
# From the repo root, or per package:
cd cadling   && pytest tests/unit/ -v
cd ll_stepnet && pytest tests/ -v
cd geotoken  && pytest tests/ -v

# Marker selection
pytest -m "not slow"
pytest -m "not requires_gpu"
pytest -m "not requires_pythonocc"
pytest -n auto              # parallel
```

Write unit tests with real numeric assertions (geometry validated against
closed-form cases — plane normals, sphere curvature, known transforms). Heavy
tests gate behind `requires_torch` / `requires_cadquery` / `slow`.

## Git workflow

Branch names: `feat/…`, `fix/…`, `refactor/…`, `docs/…`. Commit messages follow
[Conventional Commits](https://www.conventionalcommits.org/):

```text
feat(backend): add STEP backend with ll_stepnet integration
fix(ll_gen): unify VAE decode so RL gains reach generate()
docs(spec): add SPEC-2 documentation-site
```

PR process: branch from `main` → implement with tests → `pytest` green → `ruff`
+ `black --check` + `mypy` clean → update docs → open PR.

## Project conventions

- **If something is called but missing, implement it — do not remove the call.**
- Unused variables/methods/imports are intentional; use them as intended.
- No fabricated/hardcoded outputs where real logic belongs — failure paths raise
  or log, never return fake geometry.

## Working on the docs

To add or edit these documentation pages, see
[Working on the docs site](/ll_toolkit/contributing/docs-site/).
