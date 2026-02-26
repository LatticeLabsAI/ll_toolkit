# LatticeLabs Toolkit

A monorepo of Python packages for CAD document processing, neural networks for 3D geometry, geometric tokenization, optical CAD recognition, and generative CAD modeling.

## Packages

| Package | Path | Description |
|---------|------|-------------|
| **cadling** | [`cadling/`](cadling/) | CAD document processing toolkit (docling-inspired). Multi-format parsing (STEP, STL, BRep, IGES), topology analysis, RAG-ready chunking, and synthetic data generation. |
| **ll-stepnet** | [`ll_stepnet/`](ll_stepnet/) | Neural network package for STEP/B-Rep CAD files. Tokenization, feature extraction, topology encoding, and task-specific models. |
| **geotoken** | [`geotoken/`](geotoken/) | Geometric tokenizer with adaptive quantization for CAD and mesh data. Mesh, parametric, and topology-level tokenization. |
| **ll-ocadr** | [`ll_ocadr/`](ll_ocadr/) | Optical CAD Recognition system. DeepSeek-OCR-inspired 3D geometry processing for LLMs with tiled chunks and global context. |
| **ll-gen** | [`ll_gen/`](ll_gen/) | Generation orchestration: neural propose, deterministic dispose. Coordinates neural proposal and deterministic disposal engines for CAD generation. |

## Quick Start

### Prerequisites

- Python 3.9 - 3.12
- [Conda](https://docs.conda.io/) (Miniconda or Miniforge recommended)

### Installation

PyTorch **must** be installed via conda-forge (not pip) to avoid OpenMP library conflicts on macOS.

```bash
# Clone the repository
git clone https://github.com/LatticeLabsAI/ll_toolkit.git
cd ll_toolkit

# Create the conda environment (installs PyTorch, pythonocc, and all packages)
conda env create -f environment.yml
conda activate cadling
```

The environment installs `cadling`, `ll_stepnet`, and `geotoken` as editable packages. To install individual packages manually:

```bash
pip install -e ./cadling          # CAD document processing
pip install -e ./ll_stepnet       # STEP/BRep neural networks
pip install -e ./geotoken         # Geometric tokenizer
pip install -e ./ll_ocadr         # Optical CAD recognition
pip install -e ./ll_gen           # Generation orchestration
```

### Optional dependency groups (from root pyproject.toml)

```bash
pip install -e ".[dev]"        # Testing, linting, docs
pip install -e ".[cad]"        # CAD processing (trimesh, networkx, numpy-stl)
pip install -e ".[ml]"         # ML (transformers, accelerate, einops)
pip install -e ".[vision]"     # Vision (opencv, easyocr, matplotlib)
pip install -e ".[hub]"        # HuggingFace Hub integration
pip install -e ".[drawings]"   # 2D drawings (DXF, PDF)
pip install -e ".[all]"        # Everything
```

## Usage

### cadling CLI

```bash
# Convert a CAD file to JSON or Markdown
cadling convert model.step --format json

# Chunk a CAD file for RAG
cadling chunk model.step

# Generate synthetic Q&A pairs
cadling generate-qa model.step
```

### Python API

```python
from cadling.backend.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("model.step")
```

```python
from geotoken import GeoTokenizer

tokenizer = GeoTokenizer()
tokens = tokenizer.tokenize(mesh)
```

```python
from stepnet.encoder import StepNetEncoder

encoder = StepNetEncoder()
embeddings = encoder.encode(step_data)
```

## Architecture

```
Input CAD File (STEP / STL / BRep / IGES)
  |
  v
cadling: DocumentConverter
  -> Format Detection -> Backend Selection
  -> Backend (format-specific parsing)
  -> Pipeline (Build -> Assemble -> Enrich)
  -> CADlingDocument
  |
  +-> Chunking (RAG)           # cadling.chunker
  +-> SDG (Q&A pairs)          # cadling.sdg
  +-> Export (JSON / Markdown)
  |
  +-> geotoken: tokenize geometry for neural models
  +-> ll_stepnet: neural STEP/BRep processing
  +-> ll_ocadr: optical CAD recognition
  +-> ll_gen: generative CAD modeling
```

## Development

### Running Tests

```bash
# All packages (from repo root)
pytest

# Individual packages
cd cadling && pytest tests/unit/ -v
cd ll_stepnet && pytest tests/ -v
cd geotoken && pytest tests/ -v
```

### Linting and Formatting

```bash
ruff check .
black .
mypy cadling/cadling ll_stepnet/stepnet geotoken/geotoken
```

### Test Markers

```bash
pytest -m "not slow"                # Skip slow tests
pytest -m "not requires_gpu"        # Skip GPU tests
pytest -m "not requires_pythonocc"  # Skip pythonocc tests
pytest -n auto                      # Parallel execution
```

## Project Structure

```
ll_toolkit/
  cadling/           # CAD document processing toolkit
    cadling/         #   Python package
    tests/           #   Tests
  ll_stepnet/        # Neural networks for STEP/BRep
    stepnet/         #   Python package
    tests/
  geotoken/          # Geometric tokenizer
    geotoken/        #   Python package
    tests/
  ll_ocadr/          # Optical CAD recognition
    tests/
  ll_gen/            # Generation orchestration
    ll_gen/          #   Python package
    tests/
  docs/              # Research docs and plans
  pyproject.toml     # Root config (tooling, shared deps)
  environment.yml    # Conda environment definition
```

## License

MIT

## Links

- [GitHub](https://github.com/LatticeLabsAI/ll_toolkit)
