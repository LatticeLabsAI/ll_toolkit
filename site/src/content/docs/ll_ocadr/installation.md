---
title: ll_ocadr — Installation
description: Install ll_ocadr for HF-native optical CAD recognition; STEP support needs pythonocc-core.
sidebar:
  label: Installation
  order: 2
---

```bash
pip install -e ./ll_ocadr
```

## Dependencies

Core inference needs `torch`, `transformers`, `trimesh`, `numpy`, `scipy`.

- **STEP (B-Rep) support** additionally needs `pythonocc-core` (conda-forge).
  STEP-file tests skip automatically when it is not installed.
- The declared `vllm` dependency is **only** for the experimental serving path,
  which is not functional today (see [Overview](/ll_toolkit/ll_ocadr/overview/)).

:::danger[PyTorch from conda-forge]
Install PyTorch via conda-forge before installing ll_ocadr on macOS — see the
monorepo [Installation](/ll_toolkit/get-started/installation/) page.
:::

## Next

Continue to [Usage](/ll_toolkit/ll_ocadr/usage/).
