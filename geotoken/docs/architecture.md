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
