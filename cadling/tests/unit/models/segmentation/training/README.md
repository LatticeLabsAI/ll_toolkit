# Training Infrastructure Unit Tests

Unit tests for CAD segmentation model training infrastructure.

## Test Coverage

### Data Loaders (`test_data_loaders.py`)

Tests HuggingFace dataset loaders with streaming support.

**TestBaseDataLoader**
- Initialization with streaming/non-streaming modes
- Iteration over dataset samples
- `take()` method for limiting samples
- `shuffle()` functionality

**TestMFCADDataLoader**
- MFCAD++ dataset initialization
- Preprocessing of face annotations
- Feature class mapping (24 classes)
- Unknown label handling

**TestMFInstSegDataLoader**
- MFInstSeg dataset initialization
- Instance mask preprocessing
- Boundary mask handling

**TestABCDataLoader**
- ABC Dataset initialization
- Mesh data preprocessing
- Assembly hierarchy extraction

**TestFusion360DataLoader**
- Fusion 360 Gallery initialization
- B-Rep graph loading

**TestGetDataLoader**
- Auto-detection of dataset types
- Explicit dataset type specification
- Custom configuration options

### Streaming Pipeline (`test_streaming_pipeline.py`)

Tests efficient data streaming with on-the-fly graph construction.

**TestStreamingCADDataset**
- Streaming dataset initialization
- Iteration over samples
- Graph caching functionality
- Error handling for failed graph construction

**TestStreamingDataPipeline**
- Pipeline initialization
- Batching with PyTorch Geometric
- `take()` method for batch limiting
- Shuffle configuration

**TestCreateStreamingPipeline**
- Convenience function for pipeline creation
- Configuration options

**TestGraphBuilders**
- `build_mesh_graph()` - Convert mesh to PyG graph
- `build_brep_graph()` - Convert B-Rep to PyG graph
- `build_instance_seg_graph()` - Instance segmentation graphs

**TestPipelineIntegration**
- Integration with data loaders
- Caching performance benefits
- Multi-split dataset support

### Training Loop (`test_train.py`)

Tests complete training infrastructure.

**TestTrainingConfig**
- Default configuration values
- Custom configuration
- All hyperparameter options

**TestSegmentationTrainer**
- Trainer initialization
- Device selection (CPU/CUDA auto-detection)
- Optimizer setup (Adam)
- Learning rate schedulers (plateau, cosine)
- Mixed precision training (AMP)
- Custom criterion support

**TestTrainerMethods**
- `train_epoch()` - Single epoch training
- `validate()` - Validation metrics
- `save_checkpoint()` - Checkpoint persistence
- `load_checkpoint()` - Checkpoint restoration
- `train()` - Complete training loop

**TestTrainingFeatures**
- Early stopping
- Gradient clipping
- Learning rate scheduling
- Checkpoint frequency
- Training without validation

**TestTrainingIntegration**
- Full training cycle (multiple epochs)
- Checkpoint persistence across sessions
- History tracking

### Dataset Builders (`test_dataset_builders.py`)

Tests dataset conversion to HuggingFace format.

**TestMFCADDatasetBuilder**
- Builder `_info()` method
- Feature class definitions (24 classes)
- Split generators (train/val/test)
- `_generate_examples()` method
- Missing STEP file handling
- Unknown label handling
- Default parameter values

**TestCreateHFDataset**
- Creating MFCAD dataset from local files
- Creating custom datasets
- Upload to HuggingFace Hub (mocked)

**TestGenericDatasetCreation**
- Generic dataset builder
- Multiple split support

**TestDatasetBuilderIntegration**
- End-to-end dataset creation
- Integration with HuggingFace `load_dataset`
- Realistic multi-split datasets

## Running Tests

### Run all training tests:
```bash
pytest cadling/tests/unit/models/segmentation/training/ -v
```

### Run specific test file:
```bash
pytest cadling/tests/unit/models/segmentation/training/test_data_loaders.py -v
pytest cadling/tests/unit/models/segmentation/training/test_streaming_pipeline.py -v
pytest cadling/tests/unit/models/segmentation/training/test_train.py -v
pytest cadling/tests/unit/models/segmentation/training/test_dataset_builders.py -v
```

### Run specific test class:
```bash
pytest cadling/tests/unit/models/segmentation/training/test_data_loaders.py::TestMFCADDataLoader -v
pytest cadling/tests/unit/models/segmentation/training/test_train.py::TestSegmentationTrainer -v
```

### Run with coverage:
```bash
pytest cadling/tests/unit/models/segmentation/training/ \
    --cov=cadling.models.segmentation.training \
    --cov-report=html
```

### Run tests requiring specific dependencies:
```bash
# Tests requiring HuggingFace datasets
pytest cadling/tests/unit/models/segmentation/training/test_data_loaders.py -v

# Tests requiring PyTorch + PyTorch Geometric
pytest cadling/tests/unit/models/segmentation/training/test_streaming_pipeline.py -v
pytest cadling/tests/unit/models/segmentation/training/test_train.py -v
```

## Dependencies

These tests require:
- `pytest` - Testing framework
- `torch` - PyTorch for neural networks
- `torch_geometric` - Graph neural networks
- `datasets` - HuggingFace datasets library
- `numpy` - Array operations
- `trimesh` - Mesh processing (for graph builder tests)

Optional:
- `huggingface_hub` - For upload tests

Install test dependencies:
```bash
pip install pytest torch torch-geometric datasets numpy trimesh huggingface_hub
```

## Test Markers

Tests use `pytest.importorskip()` to skip tests when dependencies are unavailable:
- `pytest.importorskip("torch")` - Skips if PyTorch not installed
- `pytest.importorskip("torch_geometric")` - Skips if PyG not installed
- `pytest.importorskip("datasets")` - Skips if HF datasets not installed
- `pytest.importorskip("trimesh")` - Skips if trimesh not installed

## Mock Strategy

Tests extensively use mocking to avoid requiring:
- Actual large datasets (MFCAD++, ABC, etc.)
- Trained model checkpoints
- GPU hardware
- HuggingFace Hub authentication
- Network access for dataset downloads

### Mocked Components:
- `load_dataset()` - HuggingFace dataset loading
- Dataset iteration - Returns mock samples
- Neural network forward passes
- HuggingFace Hub uploads (`push_to_hub`)
- Graph construction (in some tests)

## Expected Output

All tests should pass with output like:
```
cadling/tests/unit/models/segmentation/training/test_data_loaders.py::TestMFCADDataLoader::test_mfcad_loader_initialization PASSED
cadling/tests/unit/models/segmentation/training/test_streaming_pipeline.py::TestStreamingCADDataset::test_streaming_dataset_iteration PASSED
cadling/tests/unit/models/segmentation/training/test_train.py::TestSegmentationTrainer::test_train_epoch PASSED
...
======================== N passed in X.XXs ========================
```

## Troubleshooting

### ImportError: No module named 'datasets'
Install HuggingFace datasets:
```bash
pip install datasets
```

### ImportError: No module named 'torch_geometric'
Install PyTorch Geometric:
```bash
pip install torch-geometric
```

### Tests fail with "CUDA not available"
Tests should work on CPU. Ensure models use:
```python
model.to(torch.device("cpu"))
checkpoint = torch.load(path, map_location="cpu")
```

### Temporary directory cleanup errors
Tests use `tempfile.TemporaryDirectory()` for isolated test environments. These are automatically cleaned up, but may fail on Windows if files are still open.

## Adding New Tests

When adding new training features, add corresponding tests:

1. **Unit tests** - Test individual components in isolation
2. **Integration tests** - Test components working together
3. **Mock heavy dependencies** - Avoid requiring actual datasets, GPUs, network
4. **Use pytest fixtures** - Share test setup across tests
5. **Clear test names** - Use descriptive test method names

Example:
```python
class TestNewFeature:
    """Test new training feature."""

    def test_feature_initialization(self):
        """Test feature can be initialized."""
        # Test setup and initialization
        pass

    def test_feature_functionality(self):
        """Test feature works correctly."""
        # Test core functionality
        pass

    def test_feature_error_handling(self):
        """Test feature handles errors gracefully."""
        # Test error cases
        pass
```

## Performance Considerations

Some tests may be slow due to:
- Creating temporary files on disk
- Initializing PyTorch models
- Iterating over mock datasets

To speed up tests:
- Use small mock datasets (5-10 samples)
- Use small models for training tests
- Cache fixtures when possible
- Skip slow integration tests during development with `-m "not slow"`

## Coverage Goals

Target coverage for training infrastructure:
- Data loaders: 90%+ line coverage
- Streaming pipeline: 85%+ line coverage
- Training loop: 80%+ line coverage
- Dataset builders: 85%+ line coverage

Check current coverage:
```bash
pytest cadling/tests/unit/models/segmentation/training/ \
    --cov=cadling.models.segmentation.training \
    --cov-report=term-missing
```

## See Also

- [../README.md](../README.md) - Segmentation models test documentation
- [../../README.md](../../README.md) - Main model tests documentation
- [../../../../cadling/models/segmentation/training/README.md](../../../../cadling/models/segmentation/training/README.md) - Training infrastructure documentation
