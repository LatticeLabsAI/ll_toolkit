---
title: ll_stepnet — Installation
description: Install ll_stepnet (the stepnet package) for neural STEP/B-Rep processing.
sidebar:
  label: Installation
  order: 2
---

ll_stepnet installs the top-level **`stepnet`** package.

```bash
cd ll_stepnet
pip install -e .
```

:::danger[PyTorch from conda-forge]
ll_stepnet depends on PyTorch. On macOS, install PyTorch via conda-forge before
`pip install -e .` to avoid the OpenMP crash — see the monorepo
[Installation](/ll_toolkit/get-started/installation/) page.
:::

## Verify

```python
from stepnet import STEPTokenizer

print(STEPTokenizer().tokenize("#31=CONICAL_SURFACE('',#1837,2.6797,0.7854);"))
```

## Next

Continue to [Usage](/ll_toolkit/ll_stepnet/usage/).
