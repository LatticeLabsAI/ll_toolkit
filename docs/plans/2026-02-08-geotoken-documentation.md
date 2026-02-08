# GeoToken Documentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create comprehensive, production-quality documentation for the geotoken package covering architecture, usage, data requirements, and integration patterns.

**Architecture:** Documentation will be structured as a main README.md with linked sub-documents for API reference, integration guide, and usage examples. All docs live in `geotoken/docs/` with the main README.md at package root.

**Tech Stack:** Markdown with Mermaid diagrams, Google-style docstrings reference, code examples tested against actual package.

---

## Task 1: Create Documentation Directory Structure

**Files:**
- Create: `geotoken/docs/` directory
- Create: `geotoken/docs/api/` directory
- Create: `geotoken/docs/examples/` directory

**Step 1: Create directory structure**

```bash
mkdir -p geotoken/docs/api geotoken/docs/examples
```

**Step 2: Verify directories exist**

Run: `ls -la geotoken/docs/`
Expected: `api/` and `examples/` subdirectories

**Step 3: Commit**

```bash
git add geotoken/docs/
git commit -m "docs(geotoken): add documentation directory structure"
```

---

## Task 2: Write Main README.md

**Files:**
- Modify: `geotoken/README.md`

**Step 1: Read existing README**

Run: `cat geotoken/README.md`
Expected: Current incomplete README content

**Step 2: Write comprehensive README**

Replace `geotoken/README.md` with:

```markdown
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
```

**Step 3: Verify README renders correctly**

Run: `head -50 geotoken/README.md`
Expected: Well-formatted markdown with all sections

**Step 4: Commit**

```bash
git add geotoken/README.md
git commit -m "docs(geotoken): rewrite README with comprehensive overview"
```

---

## Task 3: Write Architecture Guide

**Files:**
- Create: `geotoken/docs/architecture.md`

**Step 1: Write architecture documentation**

Create `geotoken/docs/architecture.md`:

```markdown
# GeoToken Architecture

This document describes the core architecture, design decisions, and data flow of the geotoken package.

## Design Philosophy

GeoToken follows these principles:

1. **Adaptive Precision**: Allocate more bits to geometrically complex regions
2. **Roundtrip Fidelity**: All transforms store inverse parameters for reconstruction
3. **Modular Quantization**: Pluggable quantizers (adaptive, uniform) via configuration
4. **Lazy Dependencies**: Heavy deps (scipy, trimesh) imported conditionally
5. **Native Format Alignment**: Token formats match ll_stepnet exactly (zero adapters)

## Core Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           GeoTokenizer Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: vertices (N,3), faces (F,3)                                     │
│     │                                                                   │
│     ▼                                                                   │
│  ┌─────────────────────┐                                                │
│  │ RelationshipPreserv │  Normalize to unit cube                        │
│  │ ingNormalizer       │  (uniform scale, preserves proportions)        │
│  └──────────┬──────────┘                                                │
│             │                                                           │
│             ▼                                                           │
│  ┌─────────────────────┐                                                │
│  │ CurvatureAnalyzer   │  Compute per-vertex curvature                  │
│  │ FeatureDensityAna   │  (discrete Laplace-Beltrami + edge variance)   │
│  └──────────┬──────────┘                                                │
│             │                                                           │
│             ▼                                                           │
│  ┌─────────────────────┐                                                │
│  │ BitAllocator        │  Map complexity → bits per vertex              │
│  │                     │  (percentile-based interpolation)              │
│  └──────────┬──────────┘                                                │
│             │                                                           │
│             ▼                                                           │
│  ┌─────────────────────┐                                                │
│  │ AdaptiveQuantizer   │  Quantize with variable precision              │
│  │                     │  + prevent feature collapse                    │
│  └──────────┬──────────┘                                                │
│             │                                                           │
│             ▼                                                           │
│  Output: TokenSequence (coordinate_tokens, geometry_tokens, metadata)   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Token Types

GeoToken produces six categories of tokens:

### 1. CoordinateToken
Quantized vertex positions with variable bit-width.

```python
@dataclass
class CoordinateToken:
    x: int          # Quantized X coordinate [0, 2^bits - 1]
    y: int          # Quantized Y coordinate
    z: int          # Quantized Z coordinate
    bits: int       # Bit-width used for this vertex
    vertex_index: int
```

### 2. GeometryToken
Structural tokens for faces, edges, vertices.

```python
@dataclass
class GeometryToken:
    token_type: str  # "FACE", "EDGE", "VERTEX"
    indices: list[int]  # Referenced vertex indices
```

### 3. CommandToken
CAD construction operations (DeepCAD-compatible).

```python
@dataclass
class CommandToken:
    command_type: CommandType  # SOL, LINE, ARC, CIRCLE, EXTRUDE, EOS
    parameters: list[int]      # 16-element quantized parameter array
    parameter_mask: list[bool] # Which parameters are active
```

**Command Types:**
| Type | ID | Active Parameters | Description |
|------|----|--------------------|-------------|
| SOL | 0 | 2 (x, y) | Start of loop |
| LINE | 1 | 4 (x1, y1, x2, y2) | Line segment |
| ARC | 2 | 6 (x1, y1, x2, y2, cx, cy) | Arc segment |
| CIRCLE | 3 | 3 (cx, cy, r) | Full circle |
| EXTRUDE | 4 | 1 (height) | Extrusion operation |
| EOS | 5 | 0 | End of sequence |

### 4. ConstraintToken
Geometric constraints between sketch entities.

```python
@dataclass
class ConstraintToken:
    constraint_type: ConstraintType  # COINCIDENT, TANGENT, PERPENDICULAR, etc.
    source_index: int
    target_index: int
```

### 5. GraphNodeToken / GraphEdgeToken
B-Rep topology graph serialization.

```python
@dataclass
class GraphNodeToken:
    node_index: int
    feature_tokens: list[int]  # Quantized 48-dim features
    node_type: str  # "FACE", "EDGE", "VERTEX"
    bits: int

@dataclass
class GraphEdgeToken:
    source: int
    target: int
    feature_tokens: list[int]  # Quantized 16-dim features
    bits: int
```

### 6. GraphStructureToken
Structural markers for autoregressive generation.

```python
class GraphStructureToken(Enum):
    GRAPH_START = "GRAPH_START"
    GRAPH_END = "GRAPH_END"
    NODE_START = "NODE_START"
    NODE_END = "NODE_END"
    EDGE = "EDGE"
```

## Adaptive Quantization Algorithm

The adaptive quantizer allocates bits based on geometric complexity:

```
complexity_score = w_c × normalized_curvature + w_f × normalized_density

where:
  w_c = 0.7 (curvature weight)
  w_f = 0.3 (feature density weight)
```

Bit allocation uses percentile-based interpolation:

```
if complexity < percentile_10:
    bits = base_bits (8)
elif complexity > percentile_90:
    bits = base_bits + max_additional (12)
else:
    bits = lerp(base_bits, base_bits + max_additional, normalized_percentile)
```

### Feature Collapse Prevention

After quantization, geotoken detects when distinct vertices map to identical quantized values using spatial hashing (O(n) complexity). When collisions occur, it perturbs one vertex by the minimum quantization step to maintain distinctness.

## Vocabulary Structure

CADVocabulary maps tokens to integer IDs for model input:

```
Total Vocabulary Size: 73,377 tokens

[0-4]       Special tokens: PAD, BOS, EOS, SEP, UNK
[5-24586]   Command tokens: 6 types × 16 params × 256 levels
[24587-56986] Constraint tokens: type × source × target combinations
[56987-73376] Graph tokens: structure markers + quantized features
```

## Configuration System

All behavior is controlled through configuration dataclasses:

```python
@dataclass
class QuantizationConfig:
    tier: PrecisionTier = PrecisionTier.STANDARD
    adaptive: bool = True
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    bit_allocation: AdaptiveBitAllocationConfig = field(default_factory=AdaptiveBitAllocationConfig)

@dataclass
class AdaptiveBitAllocationConfig:
    base_bits: int = 8           # Minimum bits for flat regions
    max_additional_bits: int = 4  # Extra bits for complex regions
    min_bits: int = 4            # Absolute minimum
    max_bits: int = 16           # Absolute maximum
    curvature_weight: float = 0.7
    density_weight: float = 0.3
    percentile_low: float = 10.0
    percentile_high: float = 90.0
```

## Reconstruction Quality

Expected reconstruction errors by tier:

| Tier | Max Error | Mean Error | Use Case |
|------|-----------|------------|----------|
| DRAFT (6-bit) | < 0.5 | < 0.2 | Preview |
| STANDARD (8-bit) | < 0.2 | < 0.05 | Default |
| PRECISION (10-bit) | < 0.05 | < 0.01 | High fidelity |

## Memory and Performance

- **Lazy imports**: scipy, trimesh loaded only when needed
- **Spatial hashing**: O(n) collision detection, not O(n²)
- **NumPy-first**: All operations use numpy (torch not required)
- **Metadata preservation**: Transform parameters stored for roundtrip
```

**Step 2: Verify file was created**

Run: `head -30 geotoken/docs/architecture.md`
Expected: Markdown header and design philosophy section

**Step 3: Commit**

```bash
git add geotoken/docs/architecture.md
git commit -m "docs(geotoken): add architecture guide"
```

---

## Task 4: Write Integration Guide

**Files:**
- Create: `geotoken/docs/integration.md`

**Step 1: Write integration documentation**

Create `geotoken/docs/integration.md`:

```markdown
# GeoToken Integration Guide

This guide explains how geotoken integrates with other LatticeLabs toolkit packages.

## Integration Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        LatticeLabs Toolkit                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                 │
│  │   cadling   │     │   geotoken  │     │  ll_stepnet │                 │
│  │             │     │             │     │             │                 │
│  │ CAD Parsing │────▶│ Tokenization│────▶│ ML Models   │                 │
│  │ & Analysis  │     │ & Encoding  │     │ & Training  │                 │
│  └─────────────┘     └─────────────┘     └─────────────┘                 │
│         │                   │                   │                        │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌─────────────────────────────────────────────────────┐                 │
│  │              GeoTokenIntegration Bridge             │                 │
│  │         (cadling/backend/geotoken_integration.py)   │                 │
│  └─────────────────────────────────────────────────────┘                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Primary Integration: GeoTokenIntegration Bridge

The central entry point for all geotoken interactions is `GeoTokenIntegration` in cadling:

```python
from cadling.backend.geotoken_integration import GeoTokenIntegration

bridge = GeoTokenIntegration()

# Tokenize a complete document
result = bridge.tokenize_document(
    doc,
    include_mesh=True,
    include_graph=True,
    include_commands=True,
    include_constraints=True
)

# Access results
mesh_tokens = result.mesh_tokens
graph_tokens = result.graph_tokens
command_tokens = result.command_tokens
token_ids = result.token_ids
metadata = result.metadata
```

### Bridge Methods

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `tokenize_document()` | CADlingDocument | GeoTokenResult | Complete document tokenization |
| `tokenize_mesh()` | vertices, faces | TokenSequence | Mesh geometry only |
| `tokenize_topology()` | TopologyGraph | TokenSequence | B-Rep graph only |
| `tokenize_sketch()` | Sketch2DItem | TokenSequence | Command sequence only |
| `encode_sequences()` | TokenSequence | list[int] | Token → ID encoding |
| `decode_token_ids()` | list[int] | TokenSequence | ID → Token decoding |
| `validate_roundtrip()` | original, reconstructed | dict | Quality metrics |

### Lazy Import Pattern

The bridge uses lazy imports for graceful degradation:

```python
def _try_import_tokenizers():
    """Import geotoken tokenizers, return None if unavailable."""
    try:
        from geotoken import GeoTokenizer, GraphTokenizer, CommandSequenceTokenizer
        return GeoTokenizer, GraphTokenizer, CommandSequenceTokenizer
    except ImportError:
        return None, None, None
```

When geotoken is unavailable, the bridge sets `degraded=True` in metadata and returns empty token sequences.

## Integration with cadling

### Data Flow: CADlingDocument → Tokens

```python
# 1. Parse CAD file with cadling
from cadling import DocumentConverter

converter = DocumentConverter()
doc = converter.convert("model.step")

# 2. Extract geometry items
mesh_item = doc.get_item_by_type("MeshItem")
vertices = mesh_item.to_numpy()  # Returns (N, 3) float32

topo_graph = doc.get_item_by_type("TopologyGraph")
node_features = topo_graph.to_numpy_node_features()  # (N, 48) float32
edge_index = topo_graph.to_edge_index()  # (2, M) int64
edge_features = topo_graph.to_numpy_edge_features()  # (M, 16) float32

sketch = doc.get_item_by_type("Sketch2DItem")
commands = sketch.to_geotoken_commands()  # List of command dicts

# 3. Tokenize via bridge
bridge = GeoTokenIntegration()
result = bridge.tokenize_document(doc)
```

### SegNetPipeline Integration

The SegNetPipeline uses geotoken for its three-stage process:

```python
from cadling.pipeline import SegNetPipeline

pipeline = SegNetPipeline(
    include_mesh_tokens=True,
    include_graph_tokens=True,
    include_command_tokens=True,
    include_constraints=True,
    vertex_merge_distance=0.005  # Post-process vertex clustering
)

result = pipeline.run(doc)
token_sequence = result.token_sequence
validation_report = result.validation_report
```

**Pipeline Stages:**
1. **Segment**: Parse document, run enrichment models
2. **Tokenize**: Use GeoTokenIntegration bridge
3. **Reconstruct**: Execute tokens via CommandExecutor

### GenerationPipeline Integration

For generative models, geotoken handles decoding:

```python
from cadling.generation import GenerationPipeline

pipeline = GenerationPipeline(backend="vae")

# Generate tokens from model
generated_ids = model.generate(prompt_embedding)

# Decode tokens to geometry
mesh = pipeline.decode_tokens_to_geometry(generated_ids)

# Validate generated mesh
validation = pipeline._validate_generated_mesh(mesh.vertices, mesh.faces)
```

## Integration with ll_stepnet

### Native Format Alignment

ll_stepnet and geotoken share the same CommandType enum:

```python
# geotoken/tokenizer/token_types.py
class CommandType(IntEnum):
    SOL = 0
    LINE = 1
    ARC = 2
    CIRCLE = 3
    EXTRUDE = 4
    EOS = 5

# ll_stepnet/stepnet/output_heads.py
class CommandType(IntEnum):
    SOL = 0
    LINE = 1
    ARC = 2
    CIRCLE = 3
    EXTRUDE = 4
    EOS = 5
```

This alignment enables zero-adapter integration.

### GeoTokenDataset

ll_stepnet provides a PyTorch Dataset wrapper:

```python
from ll_stepnet.stepnet.data import GeoTokenDataset, GeoTokenCollator

# Load tokenized data
dataset = GeoTokenDataset(token_sequences)
collator = GeoTokenCollator(pad_id=0)

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collator)

for batch in dataloader:
    command_types = batch["command_types"]  # (B, L)
    parameters = batch["parameters"]  # (B, L, 16)
    parameter_masks = batch["parameter_masks"]  # (B, L, 16)
```

### Feature Dimensions

Standard feature dimensions across packages:

| Feature | Dimensions | Used By |
|---------|------------|---------|
| Node features | 48 | cadling TopologyGraph, geotoken GraphTokenizer |
| Edge features | 16 | cadling TopologyGraph, geotoken GraphTokenizer |
| Command parameters | 16 | geotoken CommandToken, ll_stepnet output heads |

## Format Conversion

### DeepCAD ↔ cadling Format

geotoken includes a format converter for compatibility:

```python
from geotoken.tokenizer.command_format_converter import CommandFormatConverter

converter = CommandFormatConverter()

# DeepCAD format: compact, active-only parameters
deepcad_line = {"type": "LINE", "params": [0.0, 0.0, 1.0, 1.0]}

# cadling format: z-interleaved, 16-element padded
cadling_line = converter.deepcad_to_cadling("LINE", deepcad_line["params"])
# Result: [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Auto-detect format
format_type = converter.detect_format(commands)  # "deepcad" or "cadling"
```

## Vertex Post-Processing

Generated meshes often need post-processing:

```python
from geotoken.vertex import VertexValidator, VertexClusterer, VertexMerger

# Validate mesh quality
validator = VertexValidator()
report = validator.validate(vertices, faces)

if not report.manifold_check:
    print("Warning: Non-manifold mesh")

# Cluster and merge near-duplicate vertices
clusterer = VertexClusterer(merge_distance=0.005)
clustering = clusterer.cluster(vertices)
merged_verts, clean_faces = VertexMerger.merge(vertices, faces, clustering)

# Refine coarse predictions
from geotoken.vertex import CoarseToFineRefiner

refiner = CoarseToFineRefiner(
    max_iterations=20,
    learning_rate=0.1,
    convergence_threshold=1e-4
)
result = refiner.refine(merged_verts, target_points=reference)
```

## Quality Assessment

```python
from geotoken.impact import QuantizationImpactAnalyzer

analyzer = QuantizationImpactAnalyzer()
report = analyzer.analyze(original_vertices, faces, config)

print(f"Hausdorff distance: {report.hausdorff_distance}")
print(f"Mean error: {report.mean_error}")
print(f"Feature collapse rate: {report.feature_loss.collapse_rate}")
print(f"Relationship preservation: {report.relationship_preservation_rate}")
```

## SDG Integration

For synthetic data generation with text-CAD pairs:

```python
from cadling.sdg.qa.sequence_annotator import SequenceAnnotator

annotator = SequenceAnnotator()

# Generate text-token pairs for training
pairs = annotator.annotate(
    cad_file="model.step",
    annotation_level="detailed",
    include_constraints=True
)

# Output: JSONL with text annotations + command tokens
```

## Error Handling

The bridge handles missing dependencies gracefully:

```python
result = bridge.tokenize_document(doc)

if result.metadata.get("degraded"):
    print("Warning: geotoken not available, returning empty tokens")
    print(f"Errors: {result.errors}")
```

## Configuration Propagation

Pass configuration through the bridge:

```python
bridge = GeoTokenIntegration(config={
    "source_format": "deepcad",
    "include_constraints": True,
    "precision_tier": "STANDARD"
})
```
```

**Step 2: Verify file was created**

Run: `head -30 geotoken/docs/integration.md`
Expected: Integration guide header and architecture diagram

**Step 3: Commit**

```bash
git add geotoken/docs/integration.md
git commit -m "docs(geotoken): add integration guide for cadling and ll_stepnet"
```

---

## Task 5: Write API Reference Index

**Files:**
- Create: `geotoken/docs/api/README.md`

**Step 1: Write API index**

Create `geotoken/docs/api/README.md`:

```markdown
# GeoToken API Reference

## Package Exports

All public APIs are exported from the main package:

```python
from geotoken import (
    # Configuration
    PrecisionTier,
    QuantizationConfig,
    NormalizationConfig,
    AdaptiveBitAllocationConfig,
    CommandTokenizationConfig,
    GraphTokenizationConfig,

    # Tokenization
    GeoTokenizer,
    CommandSequenceTokenizer,
    GraphTokenizer,
    CADVocabulary,

    # Token Types
    TokenSequence,
    CoordinateToken,
    GeometryToken,
    CommandToken,
    CommandType,
    ConstraintToken,
    ConstraintType,
    BooleanOpToken,
    BooleanOpType,
    GraphNodeToken,
    GraphEdgeToken,
    GraphStructureToken,

    # Quantization
    AdaptiveQuantizer,
    UniformQuantizer,
    FeatureVectorQuantizer,
    UVGridQuantizer,

    # Special Token IDs
    PAD_TOKEN_ID,  # 0
    BOS_TOKEN_ID,  # 1
    EOS_TOKEN_ID,  # 2
    SEP_TOKEN_ID,  # 3
    UNK_TOKEN_ID,  # 4
)
```

## Configuration Classes

### PrecisionTier

```python
class PrecisionTier(Enum):
    DRAFT = 6       # 6-bit, 64 levels
    STANDARD = 8    # 8-bit, 256 levels (default)
    PRECISION = 10  # 10-bit, 1024 levels
```

### QuantizationConfig

```python
@dataclass
class QuantizationConfig:
    tier: PrecisionTier = PrecisionTier.STANDARD
    adaptive: bool = True
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    bit_allocation: AdaptiveBitAllocationConfig = field(default_factory=AdaptiveBitAllocationConfig)
```

### NormalizationConfig

```python
@dataclass
class NormalizationConfig:
    center: bool = True              # Center to origin
    uniform_scale: bool = True       # Maintain aspect ratio
    target_range: tuple = (0.0, 1.0) # Output coordinate range
```

### AdaptiveBitAllocationConfig

```python
@dataclass
class AdaptiveBitAllocationConfig:
    base_bits: int = 8              # Minimum bits for flat regions
    max_additional_bits: int = 4    # Extra bits for complex regions
    min_bits: int = 4               # Absolute minimum
    max_bits: int = 16              # Absolute maximum
    curvature_weight: float = 0.7   # Weight for curvature in complexity
    density_weight: float = 0.3     # Weight for feature density
    percentile_low: float = 10.0    # Below this → base_bits
    percentile_high: float = 90.0   # Above this → max bits
```

## Tokenizer Classes

### GeoTokenizer

Main tokenizer for mesh geometry.

```python
class GeoTokenizer:
    def __init__(self, config: QuantizationConfig = None):
        """Initialize with optional configuration."""

    def tokenize(
        self,
        vertices: np.ndarray,  # (N, 3) float32
        faces: np.ndarray      # (F, 3) int64
    ) -> TokenSequence:
        """Tokenize mesh geometry to discrete tokens."""

    def detokenize(self, tokens: TokenSequence) -> np.ndarray:
        """Reconstruct vertices from tokens. Returns (N, 3) float32."""

    def analyze_impact(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> ImpactReport:
        """Analyze quantization quality impact."""
```

### CommandSequenceTokenizer

Tokenizer for CAD construction history.

```python
class CommandSequenceTokenizer:
    def __init__(self, config: CommandTokenizationConfig = None):
        """Initialize with optional configuration."""

    def tokenize(
        self,
        construction_history: list[dict],
        constraints: list[dict] = None
    ) -> TokenSequence:
        """Tokenize CAD command sequence."""

    def parse_construction_history(self, commands: list[dict]) -> list:
        """Parse raw command dicts to internal format."""

    def normalize_sketches(self, parsed: list) -> list:
        """Apply 2D normalization to sketch commands."""

    def normalize_3d(self, parsed: list) -> list:
        """Apply 3D normalization to extrusion commands."""

    def quantize_parameters(self, parsed: list) -> list[CommandToken]:
        """Quantize command parameters to discrete values."""

    def dequantize_parameters(self, tokens: list[CommandToken]) -> list[dict]:
        """Reconstruct command parameters from tokens."""

    def pad_or_truncate(self, tokens: list[CommandToken]) -> list[CommandToken]:
        """Pad to max_length or truncate, preserving sketch pairs."""
```

### GraphTokenizer

Tokenizer for B-Rep topology graphs.

```python
class GraphTokenizer:
    def __init__(self, config: GraphTokenizationConfig = None):
        """Initialize with optional configuration."""

    def fit(
        self,
        node_features: np.ndarray,  # (N, 48) float32
        edge_features: np.ndarray   # (M, 16) float32
    ) -> None:
        """Pre-fit quantization parameters from sample data."""

    def tokenize(
        self,
        node_features: np.ndarray,  # (N, 48) float32
        edge_index: np.ndarray,     # (2, M) int64
        edge_features: np.ndarray,  # (M, 16) float32
        node_types: list[str] = None
    ) -> TokenSequence:
        """Serialize B-Rep graph to token sequence."""
```

### CADVocabulary

Token ↔ integer ID mapping.

```python
class CADVocabulary:
    def __init__(self):
        """Initialize vocabulary with all token types."""

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (73,377 tokens)."""

    def encode(self, command_tokens: list[CommandToken]) -> list[int]:
        """Encode command tokens to integer IDs."""

    def decode(self, token_ids: list[int]) -> list[CommandToken]:
        """Decode integer IDs back to command tokens."""

    def encode_constraint(self, token: ConstraintToken) -> int:
        """Encode single constraint token."""

    def encode_graph_node(self, token: GraphNodeToken) -> list[int]:
        """Encode graph node token to integer IDs."""

    def encode_full_sequence(self, token_seq: TokenSequence) -> list[int]:
        """Encode complete token sequence."""

    def save(self, path: str) -> None:
        """Save vocabulary to file."""

    @classmethod
    def load(cls, path: str) -> "CADVocabulary":
        """Load vocabulary from file."""
```

## Token Dataclasses

### TokenSequence

Container for all token types.

```python
@dataclass
class TokenSequence:
    coordinate_tokens: list[CoordinateToken] = field(default_factory=list)
    geometry_tokens: list[GeometryToken] = field(default_factory=list)
    command_tokens: list[CommandToken] = field(default_factory=list)
    constraint_tokens: list[ConstraintToken] = field(default_factory=list)
    boolean_op_tokens: list[BooleanOpToken] = field(default_factory=list)
    graph_node_tokens: list[GraphNodeToken] = field(default_factory=list)
    graph_edge_tokens: list[GraphEdgeToken] = field(default_factory=list)
    graph_structure_tokens: list[GraphStructureToken] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

### CommandToken

```python
@dataclass
class CommandToken:
    command_type: CommandType
    parameters: list[int]      # 16-element quantized array
    parameter_mask: list[bool] # Which parameters are active
```

### CommandType

```python
class CommandType(IntEnum):
    SOL = 0      # Start of loop (x, y)
    LINE = 1     # Line segment (x1, y1, x2, y2)
    ARC = 2      # Arc (x1, y1, x2, y2, cx, cy)
    CIRCLE = 3   # Circle (cx, cy, r)
    EXTRUDE = 4  # Extrusion (height)
    EOS = 5      # End of sequence
```

## Quantization Classes

### AdaptiveQuantizer

```python
class AdaptiveQuantizer:
    def __init__(self, config: QuantizationConfig):
        """Initialize with configuration."""

    def quantize(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> AdaptiveQuantizationResult:
        """Quantize with adaptive bit allocation."""

    def dequantize(self, result: AdaptiveQuantizationResult) -> np.ndarray:
        """Reconstruct vertices from quantization result."""
```

### UniformQuantizer

```python
class UniformQuantizer:
    def __init__(self, bits: int = 8):
        """Initialize with fixed bit-width."""

    def quantize(self, values: np.ndarray) -> np.ndarray:
        """Quantize to [0, 2^bits - 1] range."""

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """Reconstruct to [0, 1] range."""
```

## Analysis Classes

### CurvatureAnalyzer

```python
class CurvatureAnalyzer:
    def analyze_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> CurvatureResult:
        """Compute per-vertex curvature for mesh."""

    def analyze_point_cloud(
        self,
        points: np.ndarray,
        n_neighbors: int = 12
    ) -> CurvatureResult:
        """Compute curvature for point cloud via local PCA."""
```

### FeatureDensityAnalyzer

```python
class FeatureDensityAnalyzer:
    def analyze(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> FeatureDensityResult:
        """Compute per-vertex feature density."""
```

## Vertex Processing

### VertexValidator

```python
class VertexValidator:
    def validate(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> VertexValidationReport:
        """Run all validation checks."""
```

**VertexValidationReport fields:**
- `bounds_check: bool` - All coordinates within range
- `collision_check: bool` - No overlapping vertices
- `manifold_check: bool` - Each edge shared by exactly 2 faces
- `degeneracy_check: bool` - No zero-area faces
- `winding_check: bool` - Consistent face orientation
- `euler_check: bool` - Topology consistency (V - E + F = 2)

### VertexClusterer

```python
class VertexClusterer:
    def __init__(self, merge_distance: float = 0.005):
        """Initialize with merge threshold."""

    def cluster(self, vertices: np.ndarray) -> ClusteringResult:
        """Find clusters of near-duplicate vertices."""
```

### CoarseToFineRefiner

```python
class CoarseToFineRefiner:
    def __init__(
        self,
        max_iterations: int = 20,
        learning_rate: float = 0.1,
        convergence_threshold: float = 1e-4,
        face_quality_weight: float = 1.0,
        smoothness_weight: float = 0.5
    ):
        """Initialize refiner with optimization parameters."""

    def refine(
        self,
        coarse_vertices: np.ndarray,
        target_points: np.ndarray = None,
        face_indices: np.ndarray = None,
        constraints: list[dict] = None
    ) -> RefinementResult:
        """Refine coarse vertices to sub-quantization accuracy."""
```

## Impact Analysis

### QuantizationImpactAnalyzer

```python
class QuantizationImpactAnalyzer:
    def analyze(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        config: QuantizationConfig
    ) -> ImpactReport:
        """Analyze quantization quality impact."""
```

**ImpactReport fields:**
- `tier: PrecisionTier`
- `hausdorff_distance: float` - Max bidirectional error
- `mean_error: float` - Mean per-vertex error
- `max_error: float` - Max per-vertex error
- `relationship_preservation_rate: float` - Preserved geometric relationships
- `feature_loss: FeatureLossMetric` - Collapsed vertex statistics
- `total_bits_used: int`
- `mean_bits_per_vertex: float`
```

**Step 2: Verify file was created**

Run: `head -50 geotoken/docs/api/README.md`
Expected: API reference header and exports section

**Step 3: Commit**

```bash
git add geotoken/docs/api/README.md
git commit -m "docs(geotoken): add API reference"
```

---

## Task 6: Write Data Requirements Guide

**Files:**
- Create: `geotoken/docs/data-requirements.md`

**Step 1: Write data requirements documentation**

Create `geotoken/docs/data-requirements.md`:

```markdown
# GeoToken Data Requirements

This document specifies the expected data formats, shapes, and constraints for all geotoken inputs and outputs.

## Mesh Data

### Vertices

| Property | Requirement |
|----------|-------------|
| Type | `numpy.ndarray` |
| Shape | `(N, 3)` where N >= 1 |
| Dtype | `float32` or `float64` (converted internally) |
| Range | Any finite values (normalized internally) |
| NaN/Inf | Not allowed |

```python
# Valid vertex arrays
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
vertices = np.random.randn(100, 3).astype(np.float32)

# Invalid
vertices = np.array([[0, 0], [1, 1]])  # Wrong shape (N, 2)
vertices = np.array([[[0, 0, 0]]])  # Wrong dimensions (1, 1, 3)
```

### Faces

| Property | Requirement |
|----------|-------------|
| Type | `numpy.ndarray` |
| Shape | `(F, 3)` where F >= 0 |
| Dtype | `int32` or `int64` |
| Range | `[0, N-1]` where N is vertex count |
| Winding | Consistent (CCW or CW) |

```python
# Valid face arrays
faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
faces = np.zeros((0, 3), dtype=np.int64)  # Empty is valid

# Invalid
faces = np.array([[0, 1, 2, 3]])  # Wrong shape (F, 4) - quads not supported
faces = np.array([[0, 1, 100]])   # Index 100 out of bounds
```

## Command Sequences

### Command Dictionary Format

```python
command = {
    "type": str,      # "SOL", "LINE", "ARC", "CIRCLE", "EXTRUDE", "EOS"
    "params": list    # 16-element float list
}
```

### Command Types and Parameters

| Type | Active Params | Parameter Positions | Description |
|------|---------------|---------------------|-------------|
| SOL | 2 | [0:x, 1:y] | Start of loop |
| LINE | 4 | [0:x1, 1:y1, 3:x2, 4:y2] | Line segment |
| ARC | 6 | [0:x1, 1:y1, 3:x2, 4:y2, 6:cx, 7:cy] | Arc segment |
| CIRCLE | 3 | [0:cx, 1:cy, 6:r] | Full circle |
| EXTRUDE | 1 | [15:height] | Extrusion |
| EOS | 0 | - | End of sequence |

**Note:** Positions 2, 5, 8, etc. are reserved for z-coordinates in cadling format.

### Example Command Sequence

```python
commands = [
    {"type": "SOL", "params": [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"type": "LINE", "params": [0.0, 0.0, 0, 1.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"type": "LINE", "params": [1.0, 0.0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"type": "LINE", "params": [1.0, 1.0, 0, 0.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"type": "LINE", "params": [0.0, 1.0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"type": "EXTRUDE", "params": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0]},
    {"type": "EOS", "params": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
]
```

### DeepCAD Format (Compact)

GeoToken also accepts DeepCAD's compact format:

```python
# DeepCAD format - active parameters only
commands = [
    {"type": "LINE", "params": [0.0, 0.0, 1.0, 0.0]},  # x1, y1, x2, y2
    {"type": "CIRCLE", "params": [0.5, 0.5, 0.3]}      # cx, cy, r
]

# Auto-detected and converted internally
tokenizer.tokenize(commands)  # Works with either format
```

## B-Rep Graph Data

### Node Features

| Property | Requirement |
|----------|-------------|
| Type | `numpy.ndarray` |
| Shape | `(N, 48)` where N is node count |
| Dtype | `float32` |
| Range | Typically normalized to [-1, 1] or [0, 1] |

The 48-dimensional feature vector encodes:
- Positions 0-2: Centroid (x, y, z)
- Positions 3-5: Normal vector
- Positions 6-8: Bounding box min
- Positions 9-11: Bounding box max
- Positions 12-47: Surface type one-hot, curvature, area, etc.

### Edge Index

| Property | Requirement |
|----------|-------------|
| Type | `numpy.ndarray` |
| Shape | `(2, M)` where M is edge count |
| Dtype | `int64` |
| Range | `[0, N-1]` where N is node count |

```python
# 4 edges connecting nodes 0→1, 1→2, 2→3, 3→0
edge_index = np.array([[0, 1, 2, 3],
                       [1, 2, 3, 0]], dtype=np.int64)
```

### Edge Features

| Property | Requirement |
|----------|-------------|
| Type | `numpy.ndarray` |
| Shape | `(M, 16)` where M matches edge_index |
| Dtype | `float32` |
| Range | Typically normalized |

The 16-dimensional edge feature vector encodes:
- Convexity type
- Dihedral angle
- Edge length
- Adjacent face relationship

### Node Types (Optional)

| Property | Requirement |
|----------|-------------|
| Type | `list[str]` |
| Length | N (matching node_features) |
| Values | "FACE", "EDGE", "VERTEX" |

```python
node_types = ["FACE", "FACE", "EDGE", "EDGE", "VERTEX"]
```

## Constraint Data

### Constraint Dictionary Format

```python
constraint = {
    "type": str,           # Constraint type
    "source_index": int,   # Source entity index
    "target_index": int    # Target entity index (optional for some types)
}
```

### Constraint Types

| Type | Description | Requires Target |
|------|-------------|-----------------|
| COINCIDENT | Points coincide | Yes |
| TANGENT | Curves tangent | Yes |
| PERPENDICULAR | Lines perpendicular | Yes |
| PARALLEL | Lines parallel | Yes |
| HORIZONTAL | Line is horizontal | No |
| VERTICAL | Line is vertical | No |
| EQUAL | Equal length/radius | Yes |
| SYMMETRIC | Symmetric about axis | Yes |
| CONCENTRIC | Circles share center | Yes |
| MIDPOINT | Point at midpoint | Yes |

## Output Token Shapes

### TokenSequence Contents

After tokenization, expect these shapes:

| Field | Type | Typical Shape |
|-------|------|---------------|
| coordinate_tokens | list[CoordinateToken] | Length N (vertex count) |
| geometry_tokens | list[GeometryToken] | Length F (face count) |
| command_tokens | list[CommandToken] | Length 60 (padded/truncated) |
| constraint_tokens | list[ConstraintToken] | Variable |
| graph_node_tokens | list[GraphNodeToken] | Length N (node count) |
| graph_edge_tokens | list[GraphEdgeToken] | Length M (edge count) |

### Encoded Token IDs

After vocabulary encoding:

```python
token_ids = vocab.encode_full_sequence(token_seq)
# Returns: list[int] with IDs in [0, 73376]
```

## Edge Cases

### Empty Inputs

```python
# Empty faces (point cloud)
vertices = np.random.randn(100, 3).astype(np.float32)
faces = np.zeros((0, 3), dtype=np.int64)
tokens = tokenizer.tokenize(vertices, faces)  # Works

# Empty commands
tokens = tokenizer.tokenize([])  # Returns EOS-only sequence

# Single vertex
vertices = np.array([[0, 0, 0]], dtype=np.float32)
faces = np.zeros((0, 3), dtype=np.int64)
tokens = tokenizer.tokenize(vertices, faces)  # Works
```

### Large Inputs

| Input Type | Soft Limit | Hard Limit |
|------------|------------|------------|
| Vertices | 10,000 | 100,000 |
| Faces | 20,000 | 200,000 |
| Commands | 60 | 60 (truncated) |
| Graph nodes | 1,000 | 10,000 |
| Graph edges | 5,000 | 50,000 |

### Precision Limits

| Tier | Bits | Quantization Error |
|------|------|-------------------|
| DRAFT | 6 | ±1/64 ≈ 0.0156 |
| STANDARD | 8 | ±1/256 ≈ 0.0039 |
| PRECISION | 10 | ±1/1024 ≈ 0.00098 |

## Validation

Use `VertexValidator` to check mesh quality:

```python
from geotoken.vertex import VertexValidator

validator = VertexValidator()
report = validator.validate(vertices, faces)

assert report.bounds_check, "Vertices out of expected range"
assert report.manifold_check, "Non-manifold mesh"
assert report.degeneracy_check, "Degenerate faces present"
```
```

**Step 2: Verify file was created**

Run: `head -50 geotoken/docs/data-requirements.md`
Expected: Data requirements header and mesh data section

**Step 3: Commit**

```bash
git add geotoken/docs/data-requirements.md
git commit -m "docs(geotoken): add data requirements specification"
```

---

## Task 7: Write Usage Examples

**Files:**
- Create: `geotoken/docs/examples/README.md`
- Create: `geotoken/docs/examples/mesh_tokenization.py`
- Create: `geotoken/docs/examples/command_tokenization.py`
- Create: `geotoken/docs/examples/integration_example.py`

**Step 1: Write examples index**

Create `geotoken/docs/examples/README.md`:

```markdown
# GeoToken Examples

Runnable examples demonstrating geotoken usage patterns.

## Examples

| File | Description |
|------|-------------|
| [mesh_tokenization.py](mesh_tokenization.py) | Basic mesh tokenization and reconstruction |
| [command_tokenization.py](command_tokenization.py) | CAD command sequence tokenization |
| [integration_example.py](integration_example.py) | Full cadling integration workflow |

## Running Examples

```bash
cd geotoken/docs/examples
python mesh_tokenization.py
python command_tokenization.py
python integration_example.py
```

## Prerequisites

```bash
pip install -e "../../[mesh,dev]"
```
```

**Step 2: Write mesh tokenization example**

Create `geotoken/docs/examples/mesh_tokenization.py`:

```python
#!/usr/bin/env python3
"""Example: Mesh tokenization with adaptive quantization."""
from __future__ import annotations

import numpy as np

from geotoken import (
    GeoTokenizer,
    QuantizationConfig,
    PrecisionTier,
    AdaptiveBitAllocationConfig,
)


def create_cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Create a unit cube mesh."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # Top face
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 6, 5], [4, 7, 6],  # Top
        [0, 4, 5], [0, 5, 1],  # Front
        [2, 6, 7], [2, 7, 3],  # Back
        [0, 3, 7], [0, 7, 4],  # Left
        [1, 5, 6], [1, 6, 2],  # Right
    ], dtype=np.int64)

    return vertices, faces


def main():
    print("=== GeoToken Mesh Tokenization Example ===\n")

    # Create sample mesh
    vertices, faces = create_cube_mesh()
    print(f"Input mesh: {len(vertices)} vertices, {len(faces)} faces")

    # Configure adaptive quantization
    config = QuantizationConfig(
        tier=PrecisionTier.STANDARD,
        adaptive=True,
        bit_allocation=AdaptiveBitAllocationConfig(
            base_bits=8,
            max_additional_bits=4,
            curvature_weight=0.7,
            density_weight=0.3,
        )
    )

    # Tokenize
    tokenizer = GeoTokenizer(config)
    tokens = tokenizer.tokenize(vertices, faces)

    print(f"\nTokenization results:")
    print(f"  Coordinate tokens: {len(tokens.coordinate_tokens)}")
    print(f"  Geometry tokens: {len(tokens.geometry_tokens)}")

    # Show bits per vertex
    bits = [t.bits for t in tokens.coordinate_tokens]
    print(f"  Bits per vertex: min={min(bits)}, max={max(bits)}, mean={np.mean(bits):.1f}")

    # Reconstruct
    reconstructed = tokenizer.detokenize(tokens)

    # Calculate error
    error = np.linalg.norm(vertices - reconstructed, axis=1)
    print(f"\nReconstruction error:")
    print(f"  Mean: {np.mean(error):.6f}")
    print(f"  Max: {np.max(error):.6f}")

    # Analyze impact
    impact = tokenizer.analyze_impact(vertices, faces)
    print(f"\nImpact analysis:")
    print(f"  Hausdorff distance: {impact.hausdorff_distance:.6f}")
    print(f"  Total bits: {impact.total_bits_used}")
    print(f"  Mean bits/vertex: {impact.mean_bits_per_vertex:.1f}")

    # Compare precision tiers
    print("\n=== Precision Tier Comparison ===")
    for tier in [PrecisionTier.DRAFT, PrecisionTier.STANDARD, PrecisionTier.PRECISION]:
        cfg = QuantizationConfig(tier=tier, adaptive=True)
        tok = GeoTokenizer(cfg)
        tokens = tok.tokenize(vertices, faces)
        recon = tok.detokenize(tokens)
        err = np.max(np.linalg.norm(vertices - recon, axis=1))
        print(f"  {tier.name}: max_error={err:.6f}")


if __name__ == "__main__":
    main()
```

**Step 3: Write command tokenization example**

Create `geotoken/docs/examples/command_tokenization.py`:

```python
#!/usr/bin/env python3
"""Example: CAD command sequence tokenization."""
from __future__ import annotations

from geotoken import (
    CommandSequenceTokenizer,
    CADVocabulary,
    CommandType,
)


def create_square_extrude() -> list[dict]:
    """Create commands for a square extrusion."""
    return [
        {"type": "SOL", "params": [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"type": "LINE", "params": [0.0, 0.0, 0, 1.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"type": "LINE", "params": [1.0, 0.0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"type": "LINE", "params": [1.0, 1.0, 0, 0.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"type": "LINE", "params": [0.0, 1.0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"type": "EXTRUDE", "params": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0]},
        {"type": "EOS", "params": [0] * 16},
    ]


def main():
    print("=== GeoToken Command Tokenization Example ===\n")

    # Create sample commands
    commands = create_square_extrude()
    print(f"Input: {len(commands)} commands")
    for cmd in commands:
        active = [p for p in cmd["params"] if p != 0]
        print(f"  {cmd['type']}: {active if active else '(no params)'}")

    # Tokenize
    tokenizer = CommandSequenceTokenizer()
    token_seq = tokenizer.tokenize(commands)

    print(f"\nTokenization results:")
    print(f"  Command tokens: {len(token_seq.command_tokens)}")

    # Show token details
    print("\nToken breakdown:")
    for i, tok in enumerate(token_seq.command_tokens[:7]):  # Show first 7
        cmd_name = CommandType(tok.command_type).name
        active_count = sum(tok.parameter_mask)
        print(f"  [{i}] {cmd_name}: {active_count} active params")

    # Encode to integer IDs
    vocab = CADVocabulary()
    token_ids = vocab.encode(token_seq.command_tokens)

    print(f"\nVocabulary encoding:")
    print(f"  Vocab size: {vocab.vocab_size}")
    print(f"  Encoded sequence length: {len(token_ids)}")
    print(f"  Token IDs (first 10): {token_ids[:10]}")

    # Decode back
    decoded = vocab.decode(token_ids)
    print(f"\nRoundtrip verification:")
    print(f"  Decoded tokens: {len(decoded)}")

    # Dequantize parameters
    reconstructed = tokenizer.dequantize_parameters(token_seq.command_tokens[:7])
    print(f"\nDequantized commands:")
    for cmd in reconstructed:
        active = [f"{p:.3f}" for p in cmd["params"] if abs(p) > 1e-6]
        print(f"  {cmd['type']}: {active if active else '(no params)'}")


if __name__ == "__main__":
    main()
```

**Step 4: Write integration example**

Create `geotoken/docs/examples/integration_example.py`:

```python
#!/usr/bin/env python3
"""Example: Full integration with cadling and ll_stepnet data formats."""
from __future__ import annotations

import numpy as np

from geotoken import (
    GeoTokenizer,
    GraphTokenizer,
    CommandSequenceTokenizer,
    CADVocabulary,
    QuantizationConfig,
    PrecisionTier,
)
from geotoken.vertex import VertexValidator, VertexClusterer, VertexMerger


def simulate_cadling_data():
    """Simulate data that would come from cadling."""
    # Mesh data (from MeshItem.to_numpy())
    vertices = np.random.randn(50, 3).astype(np.float32) * 10
    faces = np.random.randint(0, 50, (100, 3)).astype(np.int64)

    # Topology graph data (from TopologyGraph)
    node_features = np.random.randn(10, 48).astype(np.float32)
    edge_index = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=np.int64)
    edge_features = np.random.randn(10, 16).astype(np.float32)

    # Command sequence (from Sketch2DItem)
    commands = [
        {"type": "SOL", "params": [0.5, 0.5] + [0]*14},
        {"type": "LINE", "params": [0.0, 0.0, 0, 1.0, 0.0] + [0]*11},
        {"type": "EXTRUDE", "params": [0]*15 + [5.0]},
        {"type": "EOS", "params": [0]*16},
    ]

    return {
        "vertices": vertices,
        "faces": faces,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "commands": commands,
    }


def main():
    print("=== GeoToken Integration Example ===\n")

    # Simulate cadling data
    data = simulate_cadling_data()
    print("Simulated cadling data:")
    print(f"  Vertices: {data['vertices'].shape}")
    print(f"  Faces: {data['faces'].shape}")
    print(f"  Node features: {data['node_features'].shape}")
    print(f"  Edge features: {data['edge_features'].shape}")
    print(f"  Commands: {len(data['commands'])}")

    # Initialize tokenizers
    mesh_tokenizer = GeoTokenizer(QuantizationConfig(tier=PrecisionTier.STANDARD))
    graph_tokenizer = GraphTokenizer()
    command_tokenizer = CommandSequenceTokenizer()
    vocab = CADVocabulary()

    # Tokenize all components
    print("\n--- Mesh Tokenization ---")
    mesh_tokens = mesh_tokenizer.tokenize(data["vertices"], data["faces"])
    print(f"  Coordinate tokens: {len(mesh_tokens.coordinate_tokens)}")

    print("\n--- Graph Tokenization ---")
    graph_tokens = graph_tokenizer.tokenize(
        data["node_features"],
        data["edge_index"],
        data["edge_features"]
    )
    print(f"  Node tokens: {len(graph_tokens.graph_node_tokens)}")
    print(f"  Edge tokens: {len(graph_tokens.graph_edge_tokens)}")

    print("\n--- Command Tokenization ---")
    cmd_tokens = command_tokenizer.tokenize(data["commands"])
    print(f"  Command tokens: {len(cmd_tokens.command_tokens)}")

    # Encode to IDs (for ll_stepnet model input)
    mesh_ids = vocab.encode_full_sequence(mesh_tokens)
    graph_ids = vocab.encode_full_sequence(graph_tokens)
    cmd_ids = vocab.encode(cmd_tokens.command_tokens)

    print("\n--- Vocabulary Encoding (for ll_stepnet) ---")
    print(f"  Mesh token IDs: {len(mesh_ids)}")
    print(f"  Graph token IDs: {len(graph_ids)}")
    print(f"  Command token IDs: {len(cmd_ids)}")

    # Post-processing example
    print("\n--- Vertex Post-Processing ---")
    validator = VertexValidator()
    report = validator.validate(data["vertices"], data["faces"])
    print(f"  Bounds check: {report.bounds_check}")
    print(f"  Manifold check: {report.manifold_check}")

    # Cluster duplicates
    clusterer = VertexClusterer(merge_distance=0.01)
    clustering = clusterer.cluster(data["vertices"])
    print(f"  Clusters found: {clustering.n_clusters}")

    # Reconstruct and verify
    print("\n--- Roundtrip Verification ---")
    reconstructed = mesh_tokenizer.detokenize(mesh_tokens)
    error = np.linalg.norm(data["vertices"] - reconstructed, axis=1)
    print(f"  Mean reconstruction error: {np.mean(error):.6f}")
    print(f"  Max reconstruction error: {np.max(error):.6f}")

    print("\n=== Integration Complete ===")


if __name__ == "__main__":
    main()
```

**Step 5: Verify all example files created**

Run: `ls -la geotoken/docs/examples/`
Expected: README.md and three .py files

**Step 6: Run examples to verify they work**

Run: `cd geotoken && python docs/examples/mesh_tokenization.py`
Expected: Output showing tokenization results

**Step 7: Commit**

```bash
git add geotoken/docs/examples/
git commit -m "docs(geotoken): add runnable usage examples"
```

---

## Task 8: Final Review and Update Root CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (repository root)

**Step 1: Read current CLAUDE.md**

Run: `cat CLAUDE.md`
Expected: Current repository instructions

**Step 2: Add geotoken section**

Add to the end of `CLAUDE.md`:

```markdown

## geotoken Package

Geometric tokenizer for CAD/mesh data. See `geotoken/README.md` for full documentation.

**Quick Commands:**
```bash
# Install
pip install -e ./geotoken

# Run tests
cd geotoken && pytest tests/ -v

# Run examples
python geotoken/docs/examples/mesh_tokenization.py
```

**Key Classes:**
- `GeoTokenizer`: Mesh tokenization
- `CommandSequenceTokenizer`: CAD command sequences
- `GraphTokenizer`: B-Rep topology graphs
- `CADVocabulary`: Token → ID encoding

**Integration:**
- cadling: `from cadling.backend.geotoken_integration import GeoTokenIntegration`
- ll_stepnet: `from ll_stepnet.stepnet.data import GeoTokenDataset`
```

**Step 3: Verify CLAUDE.md updated**

Run: `tail -20 CLAUDE.md`
Expected: New geotoken section visible

**Step 4: Commit all documentation**

```bash
git add CLAUDE.md
git commit -m "docs: add geotoken section to repository CLAUDE.md"
```

---

## Task 9: Verify Complete Documentation

**Files:**
- Verify: All documentation files exist and are consistent

**Step 1: List all documentation files**

Run: `find geotoken/docs -name "*.md" -o -name "*.py" | sort`
Expected:
```
geotoken/docs/api/README.md
geotoken/docs/architecture.md
geotoken/docs/data-requirements.md
geotoken/docs/examples/README.md
geotoken/docs/examples/command_tokenization.py
geotoken/docs/examples/integration_example.py
geotoken/docs/examples/mesh_tokenization.py
geotoken/docs/integration.md
```

**Step 2: Verify README links are correct**

Run: `grep -n "\](docs/" geotoken/README.md`
Expected: Links to architecture.md, api/README.md, integration.md, examples/README.md

**Step 3: Run all example scripts**

Run: `cd geotoken && python docs/examples/mesh_tokenization.py && python docs/examples/command_tokenization.py`
Expected: Both scripts complete without errors

**Step 4: Final commit with tag**

```bash
git add -A
git commit -m "docs(geotoken): complete documentation suite

- README.md: Comprehensive overview with quick start
- docs/architecture.md: Core concepts and design decisions
- docs/api/README.md: Complete API reference
- docs/integration.md: cadling and ll_stepnet integration guide
- docs/data-requirements.md: Input/output specifications
- docs/examples/: Runnable Python examples"
```

---

## Summary

This plan creates:

| Document | Purpose | Lines |
|----------|---------|-------|
| README.md | Package overview, quick start | ~200 |
| docs/architecture.md | Core concepts, pipelines, token types | ~300 |
| docs/api/README.md | Complete API reference | ~350 |
| docs/integration.md | cadling/ll_stepnet integration | ~300 |
| docs/data-requirements.md | Data formats and constraints | ~250 |
| docs/examples/*.py | Runnable examples | ~200 |

**Total: ~1,600 lines of documentation**
