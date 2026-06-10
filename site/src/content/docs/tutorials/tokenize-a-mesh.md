---
title: 'Tutorial: Tokenize a mesh'
description: Turn a mesh into adaptive tokens with geotoken, reconstruct it, and measure the quantization impact across precision tiers.
sidebar:
  label: Tokenize a mesh
  order: 3
---

In this tutorial you will tokenize a mesh with [geotoken](/ll_toolkit/geotoken/overview/),
reconstruct it, and compare precision tiers. Allow ~10 minutes. geotoken needs
only NumPy (trimesh is optional).

## 1. Build a mesh

```python
import numpy as np

# A unit tetrahedron
vertices = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
)
faces = np.array(
    [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64
)
```

To start from a real mesh file instead, load it with trimesh and pass
`mesh.vertices` / `mesh.faces`.

## 2. Tokenize with the default (STANDARD, adaptive) config

```python
from geotoken import GeoTokenizer, QuantizationConfig, PrecisionTier

config = QuantizationConfig(tier=PrecisionTier.STANDARD, adaptive=True)
tokenizer = GeoTokenizer(config)

tokens = tokenizer.tokenize(vertices, faces)
reconstructed = tokenizer.detokenize(tokens)

print("reconstructed vertices:\n", reconstructed)
```

The tokenizer normalizes into a unit cube, scores per-vertex complexity
(curvature + feature density), allocates bits accordingly, and prevents distinct
vertices from collapsing into the same quantized value.

## 3. Measure the quantization impact

```python
impact = tokenizer.analyze_impact(vertices, faces)
print(f"mean error:        {impact.mean_error:.6f}")
print(f"hausdorff distance: {impact.hausdorff_distance:.6f}")
```

## 4. Compare precision tiers

```python
for tier in (PrecisionTier.DRAFT, PrecisionTier.STANDARD, PrecisionTier.PRECISION):
    t = GeoTokenizer(QuantizationConfig(tier=tier, adaptive=True))
    impact = t.analyze_impact(vertices, faces)
    print(f"{tier.name:9s}  mean_error={impact.mean_error:.6f}")
```

Expect error to fall as bit-width rises (DRAFT 6-bit → STANDARD 8-bit →
PRECISION 10-bit), at the cost of more tokens.

## 5. Encode to integer IDs for a model

```python
from geotoken import CommandSequenceTokenizer, CADVocabulary

commands = [
    {"type": "SOL", "params": [0.5, 0.5] + [0] * 14},
    {"type": "LINE", "params": [0.0, 0.0, 0, 1.0, 0.0] + [0] * 11},
    {"type": "EXTRUDE", "params": [0] * 15 + [5.0]},
    {"type": "EOS", "params": [0] * 16},
]
seq = CommandSequenceTokenizer().tokenize(commands)
ids = CADVocabulary().encode(seq.command_tokens)
print("token ids:", ids[:10], "…")
```

## Where to next

- Understand why quantization is necessary:
  [How geometry becomes tokens](/ll_toolkit/concepts/tokenization/).
- Feed tokens to a model: [ll_stepnet Usage](/ll_toolkit/ll_stepnet/usage/).
