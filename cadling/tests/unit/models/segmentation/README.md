# Segmentation Models Unit Tests

This directory contains unit tests for the CAD segmentation models.

## Test Coverage

### Graph Utilities (`test_graph_utils.py`)
- **TestMeshToGraph**: Tests mesh-to-graph conversion
  - Face adjacency graph construction
  - Vertex graph construction
  - Node feature extraction (centroids, normals, areas, curvature)
  - Edge feature extraction (dihedral angles, edge lengths)

- **TestEdgeFeatures**: Tests edge feature computation
  - Dihedral angle calculation
  - Edge length computation

- **TestNodeFeatures**: Tests node feature extraction
  - Geometric feature computation
  - Curvature estimation

### Model Tests (`test_models.py`)
- **TestMeshSegmentationModel**: Tests MeshSegmentationModel
  - Model initialization with/without checkpoint
  - MeshItem processing
  - Segmentation result storage

- **TestBRepSegmentationModel**: Tests BRepSegmentationModel
  - Model initialization
  - 24 manufacturing feature classes
  - Feature class completeness

- **TestManufacturingFeatureRecognizer**: Tests feature recognition
  - Model initialization
  - Geometric detectors (holes, pockets, bosses, fillets, chamfers)
  - Feature parameter extraction

- **TestVisionTextAssociation**: Tests VisionTextAssociationModel
  - Model initialization with/without API keys
  - Manufacturing feature description generation
  - Prompt creation

- **TestArchitectures**: Tests neural network components
  - EdgeConvBlock forward pass
  - GraphAttentionEncoder forward pass
  - InstanceSegmentationHead forward pass
  - Embedding normalization

## Running Tests

### Run all segmentation unit tests:
```bash
pytest cadling/tests/unit/models/segmentation/ -v
```

### Run specific test file:
```bash
pytest cadling/tests/unit/models/segmentation/test_graph_utils.py -v
pytest cadling/tests/unit/models/segmentation/test_models.py -v
```

### Run specific test class:
```bash
pytest cadling/tests/unit/models/segmentation/test_graph_utils.py::TestMeshToGraph -v
pytest cadling/tests/unit/models/segmentation/test_models.py::TestMeshSegmentationModel -v
```

### Run specific test method:
```bash
pytest cadling/tests/unit/models/segmentation/test_graph_utils.py::TestMeshToGraph::test_mesh_to_pyg_graph_face_graph -v
```

### Run with coverage:
```bash
pytest cadling/tests/unit/models/segmentation/ --cov=cadling.models.segmentation --cov-report=html
```

## Dependencies

These tests require the following packages:
- `pytest` - Testing framework
- `numpy` - Array operations
- `torch` - PyTorch for neural networks
- `torch_geometric` - Graph neural networks
- `trimesh` - Mesh processing

Optional (for specific tests):
- `anthropic` - Claude API (for VLM tests)
- `openai` - OpenAI API (for VLM tests)

Install test dependencies:
```bash
pip install pytest numpy torch torch-geometric trimesh
```

## Test Markers

Tests use pytest markers to skip tests when dependencies are unavailable:
- `pytest.importorskip("trimesh")` - Skips if trimesh not installed
- `pytest.importorskip("torch_geometric")` - Skips if PyG not installed
- `pytest.importorskip("torch")` - Skips if PyTorch not installed

## Mock Strategy

Tests use mocking extensively to avoid requiring:
- Trained model checkpoints
- API keys for VLM services
- Large test datasets
- GPU hardware

Mocks are used for:
- Model checkpoints (using empty state dicts)
- Neural network forward passes (returning zeros)
- VLM API calls (returning mock responses)
- Graph construction (using simple test graphs)

## Expected Output

All tests should pass with pytest output like:
```
cadling/tests/unit/models/segmentation/test_graph_utils.py::TestMeshToGraph::test_mesh_to_pyg_graph_face_graph PASSED
cadling/tests/unit/models/segmentation/test_graph_utils.py::TestMeshToGraph::test_mesh_to_pyg_graph_vertex_graph PASSED
...
======================== N passed in X.XXs ========================
```

## Troubleshooting

### ImportError: No module named 'torch_geometric'
Install PyTorch Geometric:
```bash
pip install torch-geometric
```

### ImportError: No module named 'trimesh'
Install trimesh:
```bash
pip install trimesh
```

### Tests fail with "CUDA not available"
Tests should work on CPU. If you see CUDA errors, ensure models use:
```python
checkpoint = torch.load(path, map_location="cpu")
```

### VLM tests fail
VLM tests should skip gracefully without API keys. If they fail:
- Ensure `vlm_client` is None when no API key provided
- Check that tests mock VLM API calls

## Adding New Tests

When adding new segmentation features, add corresponding tests:

1. **Unit tests** - Test individual components
2. **Integration tests** - Test in full pipeline (see `tests/integration/segmentation/`)
3. **Mock heavy dependencies** - Avoid requiring GPUs, trained models, API keys
4. **Use pytest fixtures** - Share test data across tests
5. **Document expected behavior** - Use clear test names and docstrings
