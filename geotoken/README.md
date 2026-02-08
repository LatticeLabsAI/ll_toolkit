# GeoToken

Geometric tokenizer with adaptive quantization for CAD/mesh data. Converts 3D geometry into discrete token sequences suitable for transformer-based models.

## Overview

GeoToken transforms CAD geometry (STEP, IGES, B-Rep) and mesh data (STL, OBJ) into discrete tokens at multiple levels:

- **Mesh-level**: Raw vertex/face geometry tokenization
- **Parametric-level**: Construction history (sketch-and-extrude sequences)
- **Topology-level**: B-Rep graph structures with feature vectors

**Key Innovation**: Adaptive precision quantization allocates more bits to geometrically complex regions (high curvature, dense features) and fewer bits to flat/simple regions—reducing token count while preserving important features.

## Installation

```bash
# From repository root
pip install -e ./geotoken

# With mesh processing support
pip install -e "./geotoken[mesh]"

# With development tools
pip install -e "./geotoken[dev]"
```

**Requirements:**
- Python >= 3.9
- numpy >= 1.24
- pydantic >= 2.0
- trimesh >= 3.20 (optional, for mesh processing)

## Quick Start

### Mesh Tokenization

```python
from geotoken import GeoTokenizer, QuantizationConfig, PrecisionTier
import numpy as np

# Sample mesh data
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)

# Configure tokenizer
config = QuantizationConfig(
    tier=PrecisionTier.STANDARD,  # 8-bit precision
    adaptive=True                   # Geometry-aware bit allocation
)

# Tokenize
tokenizer = GeoTokenizer(config)
tokens = tokenizer.tokenize(vertices, faces)

# Reconstruct
reconstructed = tokenizer.detokenize(tokens)

# Analyze quality
impact = tokenizer.analyze_impact(vertices, faces)
print(f"Mean error: {impact.mean_error:.6f}")
print(f"Hausdorff distance: {impact.hausdorff_distance:.6f}")
```

### Command Sequence Tokenization

```python
from geotoken import CommandSequenceTokenizer, CADVocabulary

# DeepCAD-format construction history
commands = [
    {"type": "SOL", "params": [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"type": "LINE", "params": [0.0, 0.0, 0, 1.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"type": "LINE", "params": [1.0, 0.0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"type": "EXTRUDE", "params": [0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0]},
    {"type": "EOS", "params": [0]*16}
]

# Tokenize commands
tokenizer = CommandSequenceTokenizer()
token_seq = tokenizer.tokenize(commands)

# Encode to integer IDs for model input
vocab = CADVocabulary()
token_ids = vocab.encode(token_seq.command_tokens)
```

### Graph Tokenization (B-Rep Topology)

```python
from geotoken import GraphTokenizer
import numpy as np

# B-Rep topology data (from cadling TopologyGraph)
node_features = np.random.randn(10, 48).astype(np.float32)  # 10 nodes, 48-dim features
edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)  # 4 edges
edge_features = np.random.randn(4, 16).astype(np.float32)  # 4 edges, 16-dim features

# Tokenize graph
tokenizer = GraphTokenizer()
token_seq = tokenizer.tokenize(node_features, edge_index, edge_features)
```

## Precision Tiers

| Tier | Bits | Quantization Levels | Use Case |
|------|------|---------------------|----------|
| DRAFT | 6 | 64 | Fast preview, low bandwidth |
| STANDARD | 8 | 256 | Balanced quality/size (default) |
| PRECISION | 10 | 1024 | High fidelity, lossless-adjacent |

## Documentation

- [Architecture Guide](docs/architecture.md) - Core concepts and design decisions
- [API Reference](docs/api/README.md) - Complete API documentation
- [Integration Guide](docs/integration.md) - Using geotoken with cadling and ll_stepnet
- [Examples](docs/examples/README.md) - Runnable code examples

## Package Structure

```
geotoken/
├── geotoken/
│   ├── config.py              # Configuration dataclasses
│   ├── tokenizer/             # Tokenization pipelines
│   │   ├── geo_tokenizer.py   # Mesh tokenization
│   │   ├── command_tokenizer.py    # CAD command sequences
│   │   ├── graph_tokenizer.py      # B-Rep graph serialization
│   │   └── vocabulary.py           # Token ID encoding
│   ├── quantization/          # Quantization algorithms
│   │   ├── adaptive.py        # Geometry-aware quantization
│   │   ├── uniform.py         # Fixed-precision baseline
│   │   └── normalizer.py      # Coordinate normalization
│   ├── analysis/              # Geometric analysis
│   │   ├── curvature.py       # Curvature computation
│   │   └── feature_density.py # Local density metrics
│   ├── vertex/                # Vertex post-processing
│   │   ├── vertex_validation.py    # Mesh validity checks
│   │   ├── vertex_clustering.py    # Duplicate merging
│   │   └── vertex_refinement.py    # Coarse-to-fine optimization
│   └── impact/                # Quality assessment
│       └── analyzer.py        # Quantization impact metrics
├── tests/                     # 460+ test cases
└── pyproject.toml
```

## License

MIT
