# CAD Graph Construction Utilities

This module provides utilities for converting CAD data (meshes and B-Rep) to PyTorch Geometric graphs for training graph neural networks (GNNs).

## Overview

The `cadling.lib.graph` module replaces placeholder random data (`torch.randn()`) with **real geometric features** extracted from CAD geometry. This enables training ML models on actual geometric and topological properties of CAD models.

## Modules

### 1. `features.py` - Geometric Utilities
Low-level geometric computations for faces, edges, and vertices.

**Functions:**
- `compute_face_centroid(vertices, face)` - Compute centroid of a triangle
- `compute_face_normal(vertices, face)` - Compute face normal vector
- `compute_face_area(vertices, face)` - Compute triangle area
- `compute_dihedral_angle(normal1, normal2)` - Angle between faces
- `compute_edge_length(vertices, edge)` - Edge length
- `compute_vertex_curvature(mesh, vertex_idx)` - Estimate curvature

### 2. `mesh_graph.py` - Mesh to PyG Conversion
Converts `trimesh` objects to PyTorch Geometric `Data` objects.

**Main Function:**
```python
mesh_to_pyg_graph(
    mesh: trimesh.Trimesh,
    use_face_graph: bool = True,
    include_normals: bool = True,
    include_curvature: bool = False,
    labels: Optional[np.ndarray] = None
) -> Data
```

**Output:**
- `x`: Node features `[N, 7]` (centroid (3) + normal (3) + area (1))
- `edge_index`: Face adjacency `[2, E]`
- `edge_attr`: Edge features `[E, 2]` (dihedral angle + distance)
- `pos`: Face centroids `[N, 3]`
- `y`: Labels `[N]` (if provided)

### 3. `brep_graph.py` - B-Rep to PyG Conversion
Converts STEP B-Rep entities to PyTorch Geometric `Data` objects.

**Main Function:**
```python
brep_to_pyg_graph(
    entities: Dict[int, Dict],
    face_labels: Optional[np.ndarray] = None
) -> Data
```

**Output:**
- `x`: Node features `[N, 24]`
  - Surface type one-hot (10 types): PLANE, CYLINDRICAL, CONICAL, SPHERICAL, etc.
  - Geometric features (14): area, centroid (3), normal (3), curvature (2), bbox (3), params (2)
- `edge_index`: Face adjacency `[2, E]`
- `edge_attr`: Edge features `[E, 8]`
  - Edge type (1): concave/convex/tangent
  - Dihedral angle (1)
  - Edge length (1)
  - Edge centroid (3)
  - Edge curvature (2)
- `y`: Labels `[N]` (if provided)

## Usage Examples

### Example 1: Mesh Graph Construction

```python
import trimesh
from cadling.lib.graph import mesh_to_pyg_graph

# Load STL/mesh file
mesh = trimesh.load("model.stl")

# Convert to PyG graph
graph = mesh_to_pyg_graph(mesh, use_face_graph=True)

print(f"Nodes: {graph.num_nodes}")
print(f"Edges: {graph.num_edges}")
print(f"Node features shape: {graph.x.shape}")  # [N, 7]
print(f"Edge features shape: {graph.edge_attr.shape}")  # [E, 2]

# Verify features are real (not random)
graph2 = mesh_to_pyg_graph(mesh, use_face_graph=True)
assert torch.allclose(graph.x, graph2.x)  # Deterministic!
```

### Example 2: B-Rep Graph Construction

```python
from cadling.backend.step import STEPParser
from cadling.lib.graph import brep_to_pyg_graph
import numpy as np

# Parse STEP file
parser = STEPParser()
with open("model.step") as f:
    step_content = f.read()

entities = parser.parse(step_content)

# Convert to PyG graph
graph = brep_to_pyg_graph(entities)

print(f"Faces: {graph.num_nodes}")
print(f"Adjacencies: {graph.num_edges // 2}")  # Undirected
print(f"Node features shape: {graph.x.shape}")  # [N, 24]
print(f"Edge features shape: {graph.edge_attr.shape}")  # [E, 8]

# Verify features are deterministic
graph2 = brep_to_pyg_graph(entities)
assert torch.allclose(graph.x, graph2.x)  # Real features, not random!
```

### Example 3: Training with Real Features

```python
from cadling.models.segmentation.training.streaming_pipeline import (
    create_streaming_pipeline,
    build_brep_graph  # Now uses real graph construction!
)

# Create pipeline with real graph construction
pipeline = create_streaming_pipeline(
    dataset_name="mfcad-dataset",
    graph_builder=build_brep_graph,  # Uses brep_to_pyg_graph internally
    dataset_type="mfcad",
    batch_size=16,
    streaming=True
)

# Train model on real geometric features
for batch in pipeline:
    # batch.x contains REAL surface features, not torch.randn()!
    # batch.edge_attr contains REAL edge features!
    model_output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    loss = criterion(model_output, batch.y)
    # ...
```

## Feature Specifications

### Mesh Node Features (7 dimensions)
1. **Centroid** (3): `[x, y, z]` - Face centroid position
2. **Normal** (3): `[nx, ny, nz]` - Unit normal vector
3. **Area** (1): Face area

### Mesh Edge Features (2 dimensions)
1. **Dihedral angle** (1): Angle between adjacent face normals `[0, π]`
2. **Distance** (1): Distance between face centroids

### B-Rep Node Features (24 dimensions)
1. **Surface type one-hot** (10): PLANE, CYLINDRICAL, CONICAL, SPHERICAL, TOROIDAL, B_SPLINE, REVOLUTION, EXTRUSION, SWEPT, UNKNOWN
2. **Geometric properties** (14):
   - Area (1)
   - Centroid (3): `[x, y, z]`
   - Normal (3): `[nx, ny, nz]`
   - Curvature (2): mean and Gaussian
   - Bounding box (3): `[width, height, depth]`
   - Surface parameters (2): u_range, v_range

### B-Rep Edge Features (8 dimensions)
1. **Edge type** (1): 0=concave, 0.5=tangent, 1=convex
2. **Dihedral angle** (1): Angle between adjacent face normals
3. **Edge length** (1): Distance between face centroids (proxy)
4. **Edge centroid** (3): `[x, y, z]` midpoint
5. **Edge curvature** (2): Average mean and Gaussian curvature

## Integration with Training Pipeline

### Before (Placeholder Implementation)
```python
# streaming_pipeline.py - OLD
from ..graph_utils import mesh_to_pyg_graph  # Module doesn't exist!

def build_brep_graph(sample):
    num_faces = sample.get("num_faces", 10)
    return Data(
        x=torch.randn(num_faces, 24),  # RANDOM!
        edge_attr=torch.randn(num_faces * 3, 8),  # RANDOM!
    )
```

### After (Real Implementation)
```python
# streaming_pipeline.py - NEW
from cadling.lib.graph import brep_to_pyg_graph  # Real module!

def build_brep_graph(sample):
    entities = sample.get("entities")  # From data loader
    face_labels = sample.get("face_labels")
    return brep_to_pyg_graph(entities=entities, face_labels=face_labels)
    # Returns REAL geometric features!
```

## Dependencies

- **PyTorch** - For tensor operations
- **PyTorch Geometric** - For `Data` objects
- **trimesh** - For mesh processing (mesh_graph.py only)
- **numpy** - For numerical computations
- **cadling.backend.step** - For STEP parsing and feature extraction (brep_graph.py only)

## Testing

Run unit tests:
```bash
pytest cadling/tests/unit/lib/graph/test_graph_utils.py -v
```

Run integration tests with streaming pipeline:
```bash
pytest cadling/tests/unit/models/segmentation/training/test_streaming_pipeline.py::TestGraphBuilders -v
```

## Performance Considerations

### Mesh Graphs
- **Fast**: O(F) for F faces, uses efficient adjacency building
- **Memory**: ~50 bytes per face for features + adjacency

### B-Rep Graphs
- **Slower**: Requires STEP parsing and feature extraction
- **Memory**: ~200 bytes per face for features + adjacency
- **Caching recommended**: Store pre-computed graphs to disk

### Optimization Tips
1. **Enable caching** in `StreamingCADDataset` to avoid recomputing graphs
2. **Precompute graphs** for datasets and save to disk
3. **Use batch processing** with multiprocessing for large datasets
4. **Disable curvature** in mesh graphs if not needed (faster)

## Troubleshooting

### ImportError: torch not found
```bash
pip install torch torch-geometric
```

### ImportError: trimesh not found
```bash
pip install trimesh
```

### No face entities found in STEP
- Ensure STEP file contains `FACE_SURFACE` or similar face entities
- Check STEP parser output: `entities.keys()`
- Verify STEP file is valid AP203/AP214 format

### Features are all zeros
- Check if STEP entities have required geometry data
- Verify `STEPFeatureExtractor` can extract features
- Enable debug logging: `feature_extractor.extract_features(entities)`

## References

- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **trimesh**: https://trimesh.org/
- **STEP ISO 10303**: https://en.wikipedia.org/wiki/ISO_10303

## License

Part of the cadling library. See main LICENSE file.
