---
title: ll_clouds — Installation
description: Install ll_clouds — a dependency-light NumPy/SciPy point-cloud library.
sidebar:
  label: Installation
  order: 2
---

ll_clouds is dependency-light: **NumPy + SciPy**, with **trimesh** used only for
mesh I/O. It does **not** require PyTorch, and `import ll_clouds` pulls in none of
cadling, ll_ocadr, or torch.

```bash
pip install -e ./ll_clouds
```

To sample point clouds from meshes, install trimesh (or use the monorepo's `cad`
extra):

```bash
pip install trimesh
```

## Verify

```python
import numpy as np
from ll_clouds import PointCloud, centroid

pc = PointCloud(points=np.random.rand(100, 3).astype(np.float32))
print(centroid(pc))
```

## Next

Continue to [Usage](/ll_toolkit/ll_clouds/usage/).
