---
title: geotoken — Overview
description: Geometric tokenizer with adaptive quantization — converts CAD/mesh geometry into discrete token sequences for transformer models.
sidebar:
  label: Overview
  order: 1
  badge:
    text: Stable
    variant: success
---

**geotoken** is a geometric tokenizer with adaptive quantization. It converts 3D
geometry — CAD (STEP, IGES, B-Rep) and meshes (STL, OBJ) — into discrete token
sequences suitable for transformer-based models, at three levels:

- **Mesh-level** — raw vertex/face geometry tokenization.
- **Parametric-level** — construction history (sketch-and-extrude command
  sequences, DeepCAD format).
- **Topology-level** — B-Rep graph structures with feature vectors.

## The key idea: adaptive precision

Adaptive precision quantization allocates **more bits to geometrically complex
regions** (high curvature, dense features) and **fewer bits to flat/simple
regions** — reducing token count while preserving the features that matter.

| Tier | Bits | Levels | Use case |
|---|---|---|---|
| `DRAFT` | 6 | 64 | Fast preview, low bandwidth |
| `STANDARD` | 8 | 256 | Balanced quality/size (default) |
| `PRECISION` | 10 | 1024 | High fidelity, lossless-adjacent |

## A first taste

```python
from geotoken import GeoTokenizer, QuantizationConfig, PrecisionTier
import numpy as np

vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)

config = QuantizationConfig(tier=PrecisionTier.STANDARD, adaptive=True)
tokenizer = GeoTokenizer(config)

tokens = tokenizer.tokenize(vertices, faces)
reconstructed = tokenizer.detokenize(tokens)

impact = tokenizer.analyze_impact(vertices, faces)
print(f"Mean error: {impact.mean_error:.6f}")
```

## Key classes

- `GeoTokenizer` — mesh tokenization.
- `CommandSequenceTokenizer` — CAD command sequences.
- `GraphTokenizer` — B-Rep topology graphs (consumes a cadling `TopologyGraph`).
- `CADVocabulary` — token → integer-ID encoding.

## Status

:::tip[Maturity: Stable]
geotoken is the toolkit's best-tested package (400+ tests) and is a pure
NumPy/Pydantic library — trimesh is optional, only for mesh I/O. It integrates
directly with [cadling](/ll_toolkit/cadling/overview/) topology and feeds
[ll_stepnet](/ll_toolkit/ll_stepnet/overview/).
:::

Use the sidebar for **Installation**, **Usage**, and the **API Reference**.
