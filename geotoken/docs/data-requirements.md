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
# 4 edges connecting nodes 0->1, 1->2, 2->3, 3->0
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
