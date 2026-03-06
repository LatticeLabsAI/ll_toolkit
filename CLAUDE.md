# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

- If something is called but missing, it should be **implemented, not removed**.
- Unused variables, methods, or imports are **always intentional** — use them appropriately as they are critical to operations.
- This is not a production server — all setup/run commands must be explicitly provided.

## Repository Structure

This is a monorepo with multiple packages:

| Package | Path | Purpose |
|---------|------|---------|
| **cadling** | `cadling/` | Main CAD document processing toolkit (docling-inspired) |
| **ll-stepnet** | `ll_stepnet/` | Neural network for STEP/B-Rep files |
| **ll-ocadr** | `ll_ocadr/` | Optical CAD Recognition system |
| **geotoken** | `geotoken/` | Geometric tokenizer |
| **lib/** | `lib/{vertdict,cadrec,ocadr,segnet}/` | Supporting libraries |

## Environment Setup

PyTorch **must** be installed via conda-forge, not pip — PyPI torch bundles `libomp.dylib` that conflicts with conda's OpenMP on macOS.

```bash
# Create environment (installs PyTorch via conda-forge)
cd cadling
conda env create -f environment.yml
conda activate cadling

# ll_stepnet is installed as editable dep via environment.yml, but can also be installed manually:
cd ../ll_stepnet && pip install -e . && cd ../cadling
```

**Never run** `pip install torch` — it will cause `OMP: Error #15` crashes.

## OpenMP / PyTorch Import Order (macOS Critical)

On macOS (especially Apple Silicon), mixing different OpenMP library sources causes a fatal crash:

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

**Root Cause:** PyTorch bundles `libomp.dylib` while conda-forge packages use `llvm-openmp`. Loading both crashes the process.

**Rules to prevent this:**

1. **ALL packages use `conda-forge` channel only** — NEVER use the `pytorch` channel
2. **Each `tests/conftest.py` imports torch FIRST** before any other imports (except stdlib)
3. **Test files use `torch = pytest.importorskip("torch")` at module level** — not inside methods
4. **Set `OMP_NUM_THREADS=1` on macOS** as extra protection (conftest.py does this automatically)

**Affected Files:**

- `ll_stepnet/environment.yml` — must use `conda-forge` only
- `ll_ocadr/environment.yml` — must use `conda-forge` only
- `ll_stepnet/tests/conftest.py` — imports torch first for OpenMP protection

**References:**

- [PyTorch Issue #44282](https://github.com/pytorch/pytorch/issues/44282)
- [PyTorch Issue #132372](https://github.com/pytorch/pytorch/issues/132372)

## Build & Test Commands

All commands run from `cadling/` directory:

```bash
# Install cadling in dev mode
pip install -e ".[dev,cad,ml,vision]"

# Run all tests
pytest

# Run specific test file
pytest tests/unit/backend/test_step_backend.py

# Run specific test
pytest tests/unit/backend/test_step_backend.py::TestSTEPBackend::test_convert_simple_step

# Skip slow/GPU/pythonocc tests
pytest -m "not slow"
pytest -m "not requires_gpu"
pytest -m "not requires_pythonocc"

# Parallel test execution
pytest -n auto

# Lint and format
ruff check cadling/ tests/
black cadling/ tests/
mypy cadling/
```

## CLI Entry Points

```bash
# Main CLI (Click-based)
cadling convert <file> --format json|markdown
cadling chunk <file>
cadling generate-qa <file>

# SDG CLI (Typer-based)
python -m cadling.sdg.cli.main qa sample <file>
python -m cadling.sdg.cli.main qa generate <file>
```

## Architecture (cadling)

CADling mirrors [docling](https://github.com/DS4SD/docling)'s architecture adapted for 3D CAD geometry. The processing flow is:

```text
DocumentConverter (entry point: backend/document_converter.py)
  → Format Detection → Backend Selection
  → Backend (format-specific parsing)
  → Pipeline (Build → Assemble → Enrich)
  → CADlingDocument (central data structure)
  → Output: Chunking (RAG), SDG (Q&A), Export (JSON/Markdown)
```

### Layer Details

**Backends** (`cadling/backend/`) — Format-specific parsing. Inherit from `AbstractCADBackend` → `DeclarativeCADBackend` (text) or `RenderableCADBackend` (vision). Major backends: STEP (`step/`), STL (`stl/`), BRep (`brep/`), IGES.

**Data Models** (`cadling/datamodel/`) — Pydantic v2 models. Core types in `base_models.py`: `CADlingDocument`, `CADItem`, `InputFormat`, `ConversionResult`. Format-specific models in `step.py`, `stl.py`, `brep.py`.

**Pipelines** (`cadling/pipeline/`) — Orchestrate Build → Assemble → Enrich. `BaseCADPipeline` is the ABC. Variants: `SimpleCADPipeline` (text-only), `VisionPipeline` (rendered images), `HybridPipeline` (text + vision).

**Enrichment Models** (`cadling/models/`) — Optional post-processing chain. Inherit from `EnrichmentModel`. Include geometry analysis, topology validation, mesh quality, segmentation (`segmentation/` subdirectory with GNN architectures).

**Chunkers** (`cadling/chunker/`) — RAG chunking strategies. Inherit from `BaseCADChunker`. Format-specific chunkers in subdirectories, plus `hybrid_chunker.py` and `hierarchical_chunker.py`.

**SDG** (`cadling/sdg/`) — Synthetic data generation for Q&A pairs from CAD documents. CLI in `cli/`, generation logic in `qa/`.

**Experimental** (`cadling/experimental/`) — Experimental pipelines (assembly hierarchy, multi-view fusion, threaded VLM) and models (design intent, feature recognition, PMI extraction, manufacturability).

### ll_stepnet Integration

`ll_stepnet` (`ll_stepnet/stepnet/`) provides neural processing for STEP files. Integrated into cadling at two layers:

- **Backend layer**: `cadling/backend/step/stepnet_integration.py` — tokenization, feature extraction, topology building
- **Model layer**: Classification, property prediction, similarity via enrichment models

## Key Patterns

- **Lazy imports**: Heavy deps (pythonocc, trimesh) imported conditionally with `has_pythonocc`/`has_trimesh` flags
- **Pydantic v2**: All data models use Pydantic with `arbitrary_types_allowed = True` for torch tensors
- **`_log = logging.getLogger(__name__)`**: Standard logging pattern throughout
- **`from __future__ import annotations`**: Used in most modules for forward reference support
- **Google-style docstrings**: Required for public APIs
- **Conventional Commits**: `feat(scope):`, `fix(scope):`, `refactor(scope):` etc.

## Tooling Config

Configured in `cadling/pyproject.toml`:

- **Black**: line-length 88, target py39-py312
- **Ruff**: E, W, F, I, N, UP, B, C4 rules; E501 ignored (Black handles it)
- **Mypy**: py39, lenient (disallow_untyped_defs=false), ignores missing imports
- **Pytest**: strict markers, coverage enabled, testpaths=["tests"]

## Known State

See `cadling/docs/RequiredToBeCorrected.md` for ~200 methods with placeholder/incomplete implementations that need production-quality code. High priority: backend abstract method implementations, geometry analysis, graph builder, geometry extractors, assembly mate detection.

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
