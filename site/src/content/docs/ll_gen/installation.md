---
title: ll_gen — Installation
description: Install ll_gen for generation orchestration — neural generators (torch + stepnet) and the CadQuery dispose sandbox.
sidebar:
  label: Installation
  order: 2
---

```bash
pip install -e ./ll_gen
```

## What each path needs

ll_gen has two proposal paths and a shared dispose stage:

- **Dispose (always)** — executes proposals in a sandboxed **CadQuery**
  subprocess. CadQuery (and its OpenCASCADE backend) is needed to turn proposals
  into real geometry and validate them.
- **Neural path** — needs **PyTorch** and the **`stepnet`** package, whose
  `STEPVAE`, `StructuredDiffusion`, `VQVAEModel`, and `CADGenerationPipeline` the
  neural generators drive. Install [ll_stepnet](/ll_toolkit/ll_stepnet/installation/) too.
- **Code path** — proposes CadQuery/OpenSCAD code; an LLM proposer is optional.

:::danger[PyTorch from conda-forge]
On macOS, install PyTorch via conda-forge before installing ll_gen — see the
monorepo [Installation](/ll_toolkit/get-started/installation/) page.
:::

## Verify

```python
from ll_gen import GenerationOrchestrator

orch = GenerationOrchestrator()   # uses default LLGenConfig
print(type(orch).__name__)
```

## Next

Continue to [Usage](/ll_toolkit/ll_gen/usage/).
