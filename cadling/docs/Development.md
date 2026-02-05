# CADling Development Guide

This document provides development guidelines, coding standards, and best practices for contributing to cadling.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Coding Standards](#coding-standards)
4. [Architecture Patterns](#architecture-patterns)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation Standards](#documentation-standards)
7. [Git Workflow](#git-workflow)
8. [Common Tasks](#common-tasks)
9. [Troubleshooting](#troubleshooting)

---

## Development Setup

### Prerequisites

- Python 3.9+
- pythonocc-core (requires conda for installation)
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd cadling

# Create conda environment (recommended for pythonocc-core)
conda env create -f environment.yml
conda activate cadling

# Or use pip (may require manual pythonocc-core setup)
pip install -e ".[dev]"

# Install ll_stepnet
cd ../ll_stepnet
pip install -e .
cd ../cadling

# Verify installation
pytest tests/
```

### IDE Setup

#### VS Code

Recommended extensions:
- Python
- Pylance
- Python Test Explorer
- Ruff
- Black Formatter

Settings (`.vscode/settings.json`):

```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true
}
```

---

## Project Structure

```
cadling/
├── cadling/                    # Main package
│   ├── backend/                # Format-specific backends
│   │   ├── __init__.py
│   │   ├── abstract_backend.py
│   │   ├── step_backend.py
│   │   ├── stl_backend.py
│   │   ├── brep_backend.py
│   │   └── step/               # STEP-specific modules
│   │       └── stepnet_integration.py
│   ├── datamodel/              # Data structures
│   │   ├── __init__.py
│   │   ├── base_models.py
│   │   ├── step.py
│   │   ├── stl.py
│   │   └── pipeline_options.py
│   ├── pipeline/               # Processing pipelines
│   │   ├── __init__.py
│   │   ├── base_pipeline.py
│   │   ├── simple_pipeline.py
│   │   ├── vlm_pipeline.py
│   │   └── hybrid_pipeline.py
│   ├── models/                 # Enrichment models
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── classification.py
│   │   ├── property_prediction.py
│   │   └── similarity.py
│   ├── chunker/                # Chunking for RAG
│   │   ├── __init__.py
│   │   ├── base_chunker.py
│   │   ├── hybrid_chunker.py
│   │   └── step_chunker/
│   ├── sdg/                    # Synthetic data generation
│   │   ├── __init__.py
│   │   └── qa_generator.py
│   ├── cli/                    # Command-line interface
│   │   ├── __init__.py
│   │   └── main.py
│   ├── utils/                  # Utilities
│   │   └── __init__.py
│   └── document_converter.py   # Main entry point
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   │   ├── backend/
│   │   ├── datamodel/
│   │   ├── pipeline/
│   │   └── models/
│   ├── integration/            # Integration tests
│   │   └── test_end_to_end.py
│   ├── fixtures/               # Test data
│   │   ├── step_files/
│   │   └── stl_files/
│   └── conftest.py             # Pytest configuration
├── docs/                       # Documentation
│   ├── api/
│   ├── guides/
│   └── examples/
├── examples/                   # Example scripts
│   ├── convert_step.py
│   ├── chunk_for_rag.py
│   └── generate_qa.py
├── pyproject.toml              # Project configuration
├── README.md
├── Plan.md
├── Overview.md
├── Development.md
└── Adjustments.md
```

### Module Organization

- **backend/**: One file per format + subdirectories for complex backends
- **datamodel/**: One file per major data structure
- **pipeline/**: One file per pipeline type
- **models/**: One file per enrichment model type

---

## Coding Standards

### Style Guide

Follow [PEP 8](https://pep8.org/) with these specifics:

- **Line length**: 88 characters (Black default)
- **Imports**: Use absolute imports, sorted with `isort`
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Google style docstrings

### Code Formatting

Use **Black** for automatic formatting:

```bash
black cadling/ tests/
```

### Linting

Use **Ruff** for linting:

```bash
ruff check cadling/ tests/
```

### Type Checking

Use **mypy** for static type checking:

```bash
mypy cadling/
```

### Example Code Style

```python
"""Module docstring explaining purpose."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from cadling.datamodel.base_models import CADItem, CADlingDocument

_log = logging.getLogger(__name__)


class MyClass(BaseModel):
    """Class docstring.

    Attributes:
        field_name: Description of field.
        another_field: Description of another field.
    """

    field_name: str
    another_field: Optional[int] = None

    def my_method(
        self,
        param1: str,
        param2: int = 10,
    ) -> Dict[str, Any]:
        """Short description.

        Longer description if needed, explaining what this method does,
        any important details, edge cases, etc.

        Args:
            param1: Description of param1.
            param2: Description of param2. Defaults to 10.

        Returns:
            Description of return value.

        Raises:
            ValueError: When param1 is empty.
        """
        if not param1:
            raise ValueError("param1 cannot be empty")

        _log.debug(f"Processing with param1={param1}, param2={param2}")

        return {"result": param1 * param2}
```

---

## Architecture Patterns

### 1. Backend Pattern

All backends must inherit from abstract base classes:

```python
class MyFormatBackend(DeclarativeCADBackend):
    """Backend for MY_FORMAT files"""

    @classmethod
    def supports_text_parsing(cls) -> bool:
        return True

    @classmethod
    def supports_rendering(cls) -> bool:
        return False

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.MY_FORMAT}

    def is_valid(self) -> bool:
        # Validate file format
        return self.content.startswith("MY_FORMAT_MAGIC")

    def convert(self) -> CADlingDocument:
        # Parse and convert to CADlingDocument
        doc = CADlingDocument(name=self.file.name)
        # ... parsing logic ...
        return doc
```

**Key Points**:
- Inherit from appropriate base class (`DeclarativeCADBackend`, `RenderableCADBackend`, or both)
- Implement all abstract methods
- Use class methods for format capabilities
- Lazy-load heavy dependencies (pythonocc-core, ll_stepnet)

### 2. Data Model Pattern

Use Pydantic for all data models:

```python
class MyItem(CADItem):
    """Custom CAD item"""

    item_type: str = "my_item"

    # Required fields
    my_field: str

    # Optional fields with defaults
    optional_field: Optional[int] = None

    # Computed fields
    @property
    def computed_property(self) -> float:
        return len(self.my_field) * 2.0

    # Validators
    @validator("my_field")
    def validate_my_field(cls, v):
        if not v:
            raise ValueError("my_field cannot be empty")
        return v
```

**Key Points**:
- Inherit from appropriate base model (`CADItem`, `CADlingDocument`, etc.)
- Use type hints for all fields
- Use `Optional` for nullable fields
- Add validators for data integrity
- Use `@property` for computed fields

### 3. Pipeline Pattern

Pipelines follow the Build → Assemble → Enrich pattern:

```python
class MyPipeline(BaseCADPipeline):
    """Custom pipeline"""

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Build: Parse and extract structure"""
        backend = conv_res.input._backend

        # Backend-specific logic
        if isinstance(backend, MyFormatBackend):
            conv_res.document = backend.convert()
        else:
            raise ValueError("MyPipeline requires MyFormatBackend")

        return conv_res

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Assemble: Combine components (optional)"""
        # Custom assembly logic
        return conv_res

    def _enrich_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Enrich: Apply models"""
        # Call parent enrichment
        return super()._enrich_document(conv_res)
```

**Key Points**:
- Inherit from `BaseCADPipeline`
- Implement `_build_document` (required)
- Override `_assemble_document` and `_enrich_document` as needed
- Use `ConversionResult` wrapper throughout
- Handle errors gracefully, populate `conv_res.errors`

### 4. Enrichment Model Pattern

```python
class MyEnrichmentModel(EnrichmentModel):
    """Custom enrichment model"""

    def __init__(self, artifacts_path: Path):
        super().__init__()

        # Load model (lazy-load if possible)
        self.model = self._load_model(artifacts_path)

    def __call__(self, doc: CADlingDocument, item_batch: List[CADItem]):
        """Enrich items"""

        for item in item_batch:
            if isinstance(item, MyItem):
                # Run inference
                prediction = self.model(item)

                # Add to item properties
                item.properties["my_prediction"] = prediction
```

**Key Points**:
- Inherit from `EnrichmentModel`
- Load models in `__init__`
- Implement `__call__(doc, item_batch)`
- Add predictions to `item.properties`
- Batch processing for efficiency

---

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                       # Fast, isolated tests
│   ├── backend/
│   │   ├── test_abstract_backend.py
│   │   ├── test_step_backend.py
│   │   └── test_stl_backend.py
│   ├── datamodel/
│   │   ├── test_base_models.py
│   │   └── test_step.py
│   └── pipeline/
│       └── test_simple_pipeline.py
├── integration/                # End-to-end tests
│   └── test_step_conversion.py
└── fixtures/                   # Test data
    ├── step_files/
    │   ├── simple_cube.step
    │   └── complex_assembly.step
    └── stl_files/
        └── mesh.stl
```

### Writing Unit Tests

```python
import pytest
from pathlib import Path

from cadling.backend.step_backend import STEPBackend
from cadling.datamodel.base_models import CADInputDocument, InputFormat


class TestSTEPBackend:
    """Test STEP backend"""

    @pytest.fixture
    def simple_step_file(self, tmp_path):
        """Create simple STEP file"""
        step_content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('test'), '2;1');
FILE_NAME('test.step', ...);
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
ENDSEC;
END-ISO-10303-21;
"""
        step_file = tmp_path / "test.step"
        step_file.write_text(step_content)
        return step_file

    def test_backend_initialization(self, simple_step_file):
        """Test backend can be initialized"""
        in_doc = CADInputDocument(
            file=simple_step_file,
            format=InputFormat.STEP
        )

        backend = STEPBackend(in_doc, simple_step_file, options=None)

        assert backend is not None
        assert backend.is_valid()

    def test_convert_simple_step(self, simple_step_file):
        """Test conversion of simple STEP file"""
        in_doc = CADInputDocument(
            file=simple_step_file,
            format=InputFormat.STEP
        )

        backend = STEPBackend(in_doc, simple_step_file, options=None)
        doc = backend.convert()

        assert doc is not None
        assert len(doc.items) == 1
        assert doc.items[0].entity_type == "CARTESIAN_POINT"

    def test_invalid_file(self, tmp_path):
        """Test backend rejects invalid files"""
        invalid_file = tmp_path / "invalid.step"
        invalid_file.write_text("NOT A STEP FILE")

        in_doc = CADInputDocument(
            file=invalid_file,
            format=InputFormat.STEP
        )

        backend = STEPBackend(in_doc, invalid_file, options=None)

        assert not backend.is_valid()
```

**Key Points**:
- Use pytest fixtures for test data
- One test class per class being tested
- Test both happy path and error cases
- Use descriptive test names
- Keep tests fast and isolated

### Writing Integration Tests

```python
import pytest
from cadling import DocumentConverter
from cadling.datamodel.base_models import ConversionStatus, InputFormat


def test_step_to_json_conversion(fixtures_path):
    """Test end-to-end STEP to JSON conversion"""

    step_file = fixtures_path / "step_files" / "simple_cube.step"

    converter = DocumentConverter(
        allowed_formats=[InputFormat.STEP]
    )

    result = converter.convert(step_file)

    assert result.status == ConversionStatus.SUCCESS
    assert result.document is not None

    json_output = result.document.export_to_json()

    assert "items" in json_output
    assert len(json_output["items"]) > 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cadling --cov-report=html

# Run specific test file
pytest tests/unit/backend/test_step_backend.py

# Run specific test
pytest tests/unit/backend/test_step_backend.py::TestSTEPBackend::test_convert_simple_step

# Run with verbose output
pytest -v

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

### Test Markers

Use markers to categorize tests:

```python
@pytest.mark.slow
def test_large_assembly():
    """Test with large CAD assembly"""
    pass

@pytest.mark.requires_gpu
def test_gpu_acceleration():
    """Test GPU acceleration"""
    pass
```

---

## Documentation Standards

### Docstring Style

Use **Google style docstrings**:

```python
def my_function(param1: str, param2: int = 10) -> Dict[str, Any]:
    """Short one-line description.

    Longer description providing more context, explaining what the function
    does, any important details, algorithms used, etc.

    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 10.

    Returns:
        A dictionary with keys:
            - "result": The computed result.
            - "status": Status code.

    Raises:
        ValueError: If param1 is empty.
        TypeError: If param2 is not an integer.

    Example:
        >>> my_function("test", 5)
        {"result": "test5", "status": 0}
    """
    pass
```

### Module Documentation

Every module should have a module-level docstring:

```python
"""STEP backend for parsing STEP files.

This module provides the STEPBackend class which integrates with ll_stepnet
for STEP file parsing and feature extraction. It supports both text-based
parsing and rendering using pythonocc-core.

Classes:
    STEPBackend: Main backend for STEP files.
    STEPNetIntegration: Integration layer for ll_stepnet.

Example:
    from cadling.backend.step_backend import STEPBackend

    backend = STEPBackend(in_doc, path, options)
    doc = backend.convert()
"""
```

### README Updates

Keep README.md up-to-date with:
- Installation instructions
- Quick start examples
- Key features
- Links to full documentation

---

## Git Workflow

### Branch Naming

- `feature/short-description` - New features
- `bugfix/issue-number-description` - Bug fixes
- `refactor/short-description` - Refactoring
- `docs/short-description` - Documentation

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:

```
feat(backend): add STEP backend with ll_stepnet integration

Implement STEPBackend class that uses ll_stepnet for tokenization,
feature extraction, and topology building.

Closes #42
```

```
fix(pipeline): handle empty documents in simple pipeline

Added validation to check if backend returns empty document
and raise appropriate error.

Fixes #58
```

### Pull Request Process

1. Create feature branch from `main`
2. Implement changes with tests
3. Run full test suite: `pytest`
4. Run linting: `ruff check . && black --check .`
5. Update documentation if needed
6. Push and create PR
7. Address review comments
8. Squash merge to `main`

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings

## Related Issues
Closes #issue_number
```

---

## Common Tasks

### Adding a New CAD Format

1. Create data model in `cadling/datamodel/my_format.py`
2. Create backend in `cadling/backend/my_format_backend.py`
3. Add format to `InputFormat` enum
4. Create `MyFormatOption` in `document_converter.py`
5. Add tests
6. Update documentation

### Adding a New Enrichment Model

1. Create model in `cadling/models/my_model.py`
2. Inherit from `EnrichmentModel`
3. Implement `__call__` method
4. Add tests
5. Document in README

### Adding a New Chunker

1. Create chunker in `cadling/chunker/my_chunker.py`
2. Inherit from `BaseCADChunker`
3. Implement `chunk` method
4. Add tests
5. Add example usage

### Debugging ll_stepnet Integration

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

from cadling.backend.step.stepnet_integration import STEPNetIntegration

stepnet = STEPNetIntegration()

# Test tokenization
tokens = stepnet.tokenizer.encode(step_text)
print(f"Tokens: {tokens}")

# Test feature extraction
features = stepnet.feature_extractor.extract_geometric_features(entity)
print(f"Features: {features}")

# Test topology building
topology = stepnet.topology_builder.build_complete_topology(features_list)
print(f"Topology: num_nodes={topology.num_nodes}, num_edges={topology.num_edges}")
```

---

## Troubleshooting

### pythonocc-core Installation Issues

**Problem**: pythonocc-core fails to install with pip

**Solution**: Use conda

```bash
conda install -c conda-forge pythonocc-core
```

### ll_stepnet Import Errors

**Problem**: `ModuleNotFoundError: No module named 'll_stepnet'`

**Solution**: Install ll_stepnet in development mode

```bash
cd ../ll_stepnet
pip install -e .
```

### CUDA Out of Memory

**Problem**: GPU runs out of memory during enrichment

**Solution**: Reduce batch size or use CPU

```python
pipeline_options = PipelineOptions(
    enrichment_batch_size=1,  # Reduce batch size
    device="cpu"  # Use CPU instead of GPU
)
```

### Slow Tests

**Problem**: Test suite takes too long

**Solution**: Run only fast tests

```bash
pytest -m "not slow"
```

Or parallelize:

```bash
pip install pytest-xdist
pytest -n auto
```

---

## Performance Best Practices

1. **Lazy-load heavy dependencies**: Import pythonocc-core and ll_stepnet only when needed
2. **Batch processing**: Process items in batches for enrichment
3. **Caching**: Cache expensive computations (topology graphs, embeddings)
4. **Profiling**: Use `cProfile` to identify bottlenecks

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
result = converter.convert("large_assembly.step")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(20)
```

---

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Build package: `python -m build`
5. Test package: `pip install dist/cadling-*.whl`
6. Tag release: `git tag v0.1.0`
7. Push tag: `git push origin v0.1.0`
8. Create GitHub release
9. Publish to PyPI: `twine upload dist/*`

---

## Resources

- [Docling Documentation](https://github.com/DS4SD/docling)
- [PythonOCC Documentation](https://github.com/tpaviot/pythonocc-core)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)

---

## Questions?

For questions or issues:
- Check [Plan.md](Plan.md) for implementation roadmap
- Check [Adjustments.md](Adjustments.md) for docling adaptations
- Open an issue on GitHub
- Contact the team

Happy coding!
