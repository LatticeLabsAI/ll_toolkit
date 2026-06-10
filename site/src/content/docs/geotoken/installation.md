---
title: geotoken — Installation
description: Install geotoken, optionally with mesh-processing or development extras.
sidebar:
  label: Installation
  order: 2
---

geotoken is a pure NumPy/Pydantic library — `trimesh` is optional (only for mesh
I/O), and PyTorch is **not** required.

```bash
# From the repository root
pip install -e ./geotoken

# With mesh processing support (trimesh)
pip install -e "./geotoken[mesh]"

# With development tools
pip install -e "./geotoken[dev]"
```

**Requirements**

- Python ≥ 3.9
- numpy ≥ 1.24
- pydantic ≥ 2.0
- trimesh ≥ 3.20 (optional, for mesh processing)

Heavy dependencies (scipy, trimesh) are imported lazily, so the core import is
light.

## Next

Continue to [Usage](/ll_toolkit/geotoken/usage/).
