---
title: geotoken — Usage
description: Tokenize meshes, CAD command sequences, and B-Rep graphs; measure quantization impact; integrate with cadling and ll_stepnet.
sidebar:
  label: Usage
  order: 3
---

geotoken tokenizes geometry at three levels — mesh, parametric (command
sequences), and topology (B-Rep graphs) — using adaptive quantization.

## Mesh tokenization

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
print(f"Hausdorff distance: {impact.hausdorff_distance:.6f}")
```

## Command-sequence tokenization

```python
from geotoken import CommandSequenceTokenizer, CADVocabulary

commands = [
    {"type": "SOL", "params": [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"type": "LINE", "params": [0.0, 0.0, 0, 1.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"type": "EXTRUDE", "params": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0]},
    {"type": "EOS", "params": [0] * 16},
]

token_seq = CommandSequenceTokenizer().tokenize(commands)
token_ids = CADVocabulary().encode(token_seq.command_tokens)
```

## Graph tokenization (B-Rep topology)

```python
from geotoken import GraphTokenizer
import numpy as np

node_features = np.random.randn(10, 48).astype(np.float32)       # 10 nodes, 48-dim
edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)
edge_features = np.random.randn(4, 16).astype(np.float32)        # 4 edges, 16-dim

token_seq = GraphTokenizer().tokenize(node_features, edge_index, edge_features)
```

## Precision tiers

| Tier | Bits | Levels | Max error | Use case |
|---|---|---|---|---|
| `DRAFT` | 6 | 64 | < 0.5 | Preview, low bandwidth |
| `STANDARD` | 8 | 256 | < 0.2 | Default |
| `PRECISION` | 10 | 1024 | < 0.05 | High fidelity |

Adaptive allocation weights curvature (0.7) and feature density (0.3) into a
complexity score, then assigns bits by percentile interpolation. After
quantization, a spatial-hash pass detects and perturbs collapsed vertices so
distinct features stay distinct.

## Integration with cadling and ll_stepnet

The bridge `cadling.backend.geotoken_integration.GeoTokenIntegration` tokenizes a
whole `CADlingDocument`:

```python
from cadling.backend.geotoken_integration import GeoTokenIntegration

bridge = GeoTokenIntegration()
result = bridge.tokenize_document(
    doc, include_mesh=True, include_graph=True,
    include_commands=True, include_constraints=True,
)
mesh_tokens, graph_tokens = result.mesh_tokens, result.graph_tokens
token_ids = result.token_ids
```

geotoken and [ll_stepnet](/ll_toolkit/ll_stepnet/overview/) share the same
`CommandType` enum and feature dimensions (48-dim nodes, 16-dim edges, 16
command parameters), enabling **zero-adapter** integration. `ll_stepnet` provides
a `GeoTokenDataset` PyTorch wrapper for tokenized data.

## Vertex post-processing

```python
from geotoken.vertex import VertexValidator, VertexClusterer, VertexMerger

report = VertexValidator().validate(vertices, faces)
clustering = VertexClusterer(merge_distance=0.005).cluster(vertices)
merged_verts, clean_faces = VertexMerger.merge(vertices, faces, clustering)
```

## Related

- [Overview](/ll_toolkit/geotoken/overview/) · [Installation](/ll_toolkit/geotoken/installation/)
- Background: [How geometry becomes tokens](/ll_toolkit/concepts/tokenization/).
