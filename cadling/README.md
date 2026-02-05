# CADling

A [docling](https://github.com/DS4SD/docling)-inspired toolkit for CAD document processing that enables optical CAD recognition, STEP/STL/BRep/IGES parsing, intelligent chunking, and structured export for LLM/ML/AI compatibility.

CADling brings proven document-processing methodologies to the world of 3D Computer-Aided Design. Just as docling bridges document formats and AI systems, CADling makes CAD files first-class citizens in the AI ecosystem.

## Key Capabilities

- **Dual-Modality Processing** -- text-based parsing of CAD file code (STEP entities, STL vertices) and vision-based recognition from rendered CAD images, with hybrid fusion of both modalities
- **Multi-Format Support** -- STEP (ISO 10303-21), STL (ASCII and binary), BRep (OpenCASCADE boundary representation), IGES, and rendered CAD images
- **Topology Analysis** -- graph-based analysis of entity relationships, assembly hierarchies, and geometric connectivity
- **RAG-Ready Chunking** -- multiple chunking strategies (hybrid, hierarchical, topology-aware) with metadata for vector databases
- **Synthetic Data Generation** -- LLM-powered Q&A pair generation from CAD documents for training CAD-aware models
- **Neural Network Integration** -- ll_stepnet integration for tokenization, feature extraction, classification, property prediction, and embeddings
- **Enrichment Models** -- pluggable geometry analysis, topology validation, mesh quality, interference checking, surface analysis, pattern detection, and segmentation
- **Structured Export** -- JSON and Markdown output with Pydantic-validated data models

## Installation

### Prerequisites

- Python 3.9+
- Conda (required for pythonocc-core and PyTorch on macOS)

### Environment Setup

```bash
# Create the conda environment (installs PyTorch via conda-forge)
conda env create -f environment.yml
conda activate cadling

# Install cadling in development mode
pip install -e ".[all]"
```

> **macOS Critical**: PyTorch **must** be installed via conda-forge, not pip. PyPI's torch bundles a `libomp.dylib` that conflicts with conda's OpenMP runtime, causing `OMP: Error #15` crashes. See `CLAUDE.md` for details.

### Optional Dependency Groups

```bash
pip install -e ".[dev]"      # pytest, black, ruff, mypy, pre-commit
pip install -e ".[cad]"      # numpy-stl, trimesh, networkx
pip install -e ".[ml]"       # transformers (PyTorch via conda only)
pip install -e ".[vision]"   # transformers, easyocr, opencv-python
pip install -e ".[all]"      # Everything above
```

### ll_stepnet (STEP Neural Network)

ll_stepnet is installed as an editable dependency via `environment.yml`. To install manually:

```bash
cd ../ll_stepnet
pip install -e .
```

## Quick Start

### Python API

```python
from cadling import DocumentConverter, InputFormat, ConversionStatus

# Basic conversion
converter = DocumentConverter()
result = converter.convert("part.step")

if result.status == ConversionStatus.SUCCESS:
    doc = result.document
    print(f"Parsed {len(doc.items)} items")

    # Export
    json_data = doc.export_to_json()
    markdown = doc.export_to_markdown()
```

### With Format Options

```python
from cadling import DocumentConverter, FormatOption, InputFormat
from cadling.backend.step.step_backend import STEPBackend
from cadling.pipeline.hybrid_pipeline import HybridPipeline

converter = DocumentConverter(
    allowed_formats=[InputFormat.STEP],
    format_options={
        InputFormat.STEP: FormatOption(
            backend=STEPBackend,
            pipeline_cls=HybridPipeline,
        )
    }
)

result = converter.convert("assembly.step")
```

### Chunking for RAG

```python
from cadling import DocumentConverter
from cadling.chunker.hybrid_chunker import CADHybridChunker

converter = DocumentConverter()
result = converter.convert("part.step")

chunker = CADHybridChunker(max_tokens=512, overlap_tokens=50)
chunks = list(chunker.chunk(result.document))

for chunk in chunks:
    print(f"Chunk {chunk.chunk_id}: {len(chunk.meta.entity_ids)} entities")
    # chunk.text         -> text representation
    # chunk.meta         -> entity types, topology subgraph, embeddings, bbox
    # chunk.metadata     -> additional metadata dict
```

### Synthetic Data Generation

```python
from pathlib import Path
from cadling.sdg.qa import (
    CADPassageSampler, CADGenerator, CADJudge,
    CADSampleOptions, CADGenerateOptions, CADCritiqueOptions,
    LlmProvider,
)

# 1. Sample passages from CAD files
sample_opts = CADSampleOptions(sample_file=Path("samples.jsonl"))
sampler = CADPassageSampler(sample_opts)
sampler.sample([Path("part.step"), Path("assembly.step")])

# 2. Generate Q&A pairs
gen_opts = CADGenerateOptions(
    provider=LlmProvider.OPENAI,
    model_id="gpt-4o",
    generated_file=Path("generated.jsonl"),
)
generator = CADGenerator(gen_opts)
generator.generate(Path("samples.jsonl"))

# 3. Critique and improve
crit_opts = CADCritiqueOptions(
    provider=LlmProvider.OPENAI,
    model_id="gpt-4o",
    critiqued_file=Path("critiqued.jsonl"),
)
judge = CADJudge(crit_opts)
judge.critique(Path("generated.jsonl"))
```

## CLI

### Main CLI (`cadling`)

```bash
# Convert CAD file to JSON or Markdown
cadling convert part.step --format json --pretty -o part.json
cadling convert mesh.stl --format markdown -o mesh.md

# Chunk CAD file for RAG
cadling chunk part.step --max-tokens 512 --overlap 50 -o chunks.jsonl

# Generate synthetic Q&A pairs
cadling generate-qa part.step -n 100 -m gpt-4 -o qa.jsonl
cadling generate-qa assembly.step -n 50 --no-critique -o qa.jsonl

# Show file information
cadling info part.step
```

### SDG CLI (`cadling-sdg`)

```bash
# Sample passages from CAD files
cadling-sdg qa sample part.step assembly.step \
  --chunker hybrid --max-passages 50 --max-tokens 512 -o samples.jsonl

# Generate Q&A pairs from sampled passages
cadling-sdg qa generate samples.jsonl \
  -p openai -m gpt-4o --max-qac 100 --temperature 0.7 \
  -q geometry -q topology -o generated.jsonl

# Critique and improve Q&A quality
cadling-sdg qa critique generated.jsonl \
  -p openai -m gpt-4o --threshold 3 --rewrite -o critiqued.jsonl

# Generate conceptual Q&A from description
cadling-sdg qa conceptual "A flanged bearing housing with bolt holes" \
  -p anthropic -m claude-3-opus --num-topics 10 --questions-per-topic 5 -o conceptual.jsonl

# Options
cadling-sdg --verbose qa ...    # Verbose output
cadling-sdg --debug qa ...      # Debug logging
cadling-sdg version             # Show version
```

## Architecture

CADling mirrors docling's layered architecture, adapted for 3D CAD geometry:

```
                    DocumentConverter
                     (entry point)
                          |
                    Format Detection
                          |
         +--------+-------+-------+--------+
         |        |       |       |        |
       STEP     STL    BRep    IGES   CAD_IMAGE
      Backend  Backend Backend Backend  Backend
         |        |       |       |        |
         +--------+-------+-------+--------+
                          |
                 Pipeline (orchestration)
                  Build -> Assemble -> Enrich
                          |
                   CADlingDocument
                  (central data model)
                          |
              +-----------+-----------+
              |           |           |
          Chunking      SDG       Export
          (for RAG)   (Q&A gen)  (JSON/MD)
```

### Processing Flow

1. **DocumentConverter** receives a CAD file and detects its format
2. A **Backend** parses the file into raw data (entities, meshes, topology)
3. A **Pipeline** orchestrates three stages:
   - **Build**: Parse and extract structure from the backend
   - **Assemble**: Combine components and resolve references
   - **Enrich**: Apply enrichment models (classification, analysis, etc.)
4. The result is a **CADlingDocument** containing items, topology, segments, and embeddings
5. Downstream consumers **chunk** for RAG, **generate** synthetic data, or **export** to JSON/Markdown

### Layer Details

**Backends** (`cadling/backend/`) parse format-specific files. All inherit from `AbstractCADBackend` with two specializations:
- `DeclarativeCADBackend` -- text-based parsing (STEP entities, STL vertices)
- `RenderableCADBackend` -- vision-based rendering (multi-view images for VLM analysis)

STEP and STL backends support both modalities (dual-mode).

**Pipelines** (`cadling/pipeline/`) orchestrate the Build/Assemble/Enrich stages. Available pipelines:

| Pipeline | Description |
|----------|-------------|
| `SimpleCADPipeline` | Basic text-only parsing |
| `STEPPipeline` | STEP-specific processing |
| `STLPipeline` | STL-specific processing |
| `VisionPipeline` | Vision-only (rendered images via VLM) |
| `VlmPipeline` | Vision-language model analysis |
| `HybridPipeline` | Fused text + vision processing |

**Enrichment Models** (`cadling/models/`) are optional post-processing modules. All inherit from `EnrichmentModel` and implement `__call__(doc, item_batch)`:

| Model | Purpose |
|-------|---------|
| `GeometryAnalysisModel` | Geometric property computation |
| `TopologyValidationModel` | Topology integrity checking |
| `MeshQualityModel` | Mesh quality metrics |
| `SurfaceAnalysisModel` | Surface type classification |
| `InterferenceCheckModel` | Collision/interference detection |
| `ConstraintDetectionModel` | Geometric constraint discovery |
| `PatternDetectionModel` | Repeated feature recognition |
| `GeometryNormalizationModel` | Coordinate normalization |
| Segmentation models | Face-level GNN segmentation (`segmentation/`) |

**Chunkers** (`cadling/chunker/`) split documents for RAG. All inherit from `BaseCADChunker`:

| Chunker | Strategy |
|---------|----------|
| `CADHybridChunker` | Entity-level + semantic grouping |
| `CADHierarchicalChunker` | Assembly hierarchy-aware |
| Format-specific chunkers | STEP, STL, BRep tailored chunking |

**SDG** (`cadling/sdg/`) generates synthetic Q&A training data:
- `CADPassageSampler` -- samples passages from CAD documents
- `CADGenerator` -- generates Q&A pairs via LLM (OpenAI, Anthropic, vLLM, Ollama, any OpenAI-compatible endpoint)
- `CADJudge` -- critiques and rewrites low-quality pairs
- `CADConceptualGenerator` -- generates topic-based Q&A from text descriptions

## Data Models

### Core Types

```python
InputFormat          # STEP | STL | BREP | IGES | CAD_IMAGE
ConversionStatus     # SUCCESS | PARTIAL | FAILURE
CADlingDocument      # Central document with items, topology, segments, embeddings
CADItem              # Base item (step_entity, mesh, assembly, annotation)
ConversionResult     # Wrapper with status, document, and errors
TopologyGraph        # Entity reference graph (adjacency list)
BoundingBox3D        # 3D axis-aligned bounding box
Segment              # Semantic region (feature, face, component)
```

### CADItem Hierarchy

```
CADItem (base)
 +-- STEPEntityItem      entity_id, entity_type, numeric/reference params, features
 +-- MeshItem             vertices, normals, facets, manifold/watertight flags
 +-- AssemblyItem         component references and hierarchy
 +-- AnnotationItem       annotation_type, value, confidence, source_view
```

### SDG Types

```python
LlmProvider          # openai | anthropic | vllm | ollama | openai_compatible
QuestionType         # fact_single | geometry | topology | manufacturing |
                     # material | assembly | dimension | tolerance
ChunkerType          # hybrid | step | stl | brep | topology
CADSampleOptions     # Sampling configuration
CADGenerateOptions   # Generation configuration
CADCritiqueOptions   # Critique configuration (dimensions, threshold, rewrite)
CADConceptualOptions # Conceptual generation options
```

## Docling Adaptations

CADling adopts docling's Backend/Pipeline/Enrichment architecture but diverges where 3D geometry demands it:

| Aspect | Docling (2D Documents) | CADling (3D CAD) |
|--------|----------------------|------------------|
| Input | Pages of text, images, tables | 3D geometry, meshes, parametric entities |
| Parsing | Text extraction, OCR, layout | Entity parsing, mesh processing, topology building |
| Bounding boxes | 2D (x, y, width, height) | 3D (x/y/z min/max) |
| Layout | Reading order, columns, sections | Topology graph, assembly hierarchy |
| Vision | Document layout understanding | Optical CAD recognition (dimensions, GD&T, tolerances) |
| Chunking | Text-based semantic chunks | Geometric/topological chunks with 3D metadata |
| Output | DoclingDocument (structured text) | CADlingDocument (structured geometry + topology) |

## ll_stepnet Integration

[ll_stepnet](../ll_stepnet/) provides neural processing for STEP files, integrated at two layers:

**Backend layer** (`cadling/backend/step/stepnet_integration.py`):
- `STEPTokenizer` -- converts STEP text to token IDs
- `STEPFeatureExtractor` -- extracts 128+ geometric/topological features per entity
- `TopologyBuilder` -- constructs entity reference graphs

**Model layer** (enrichment models):
- Part classification, property prediction, similarity embeddings, and captioning

## Experimental Features

The `cadling/experimental/` directory contains advanced capabilities under active development:

**Pipelines**:
- `AssemblyHierarchyPipeline` -- multi-part assembly processing with BOM generation
- `MultiViewFusionPipeline` -- multi-view rendering with cross-view consensus
- `ThreadedGeometryVLMPipeline` -- parallel geometry + vision-language analysis

**Models**:
- `DesignIntentInferenceModel` -- infer functional purpose from geometry
- `FeatureRecognitionVLMModel` -- vision-based machining feature detection
- `PMIExtractionModel` -- extract dimensions, tolerances, GD&T from rendered views
- `ManufacturabilityAssessmentModel` -- DFM analysis and cost estimation
- `CADToTextGenerationModel` -- natural language descriptions of CAD geometry
- `GeometricConstraintModel` -- implicit geometric constraint detection

## Testing

```bash
# Run all tests (from cadling/ directory)
pytest

# With coverage
pytest --cov=cadling --cov-report=html

# Run specific test file or test
pytest tests/unit/backend/test_step_backend.py
pytest tests/unit/backend/test_step_backend.py::TestSTEPBackend::test_convert_simple_step

# Skip slow, GPU, or pythonocc-dependent tests
pytest -m "not slow"
pytest -m "not requires_gpu"
pytest -m "not requires_pythonocc"

# Parallel execution
pytest -n auto
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.slow` | Long-running tests |
| `@pytest.mark.requires_gpu` | Needs CUDA GPU |
| `@pytest.mark.requires_pythonocc` | Needs pythonocc-core |

## Development

### Code Quality

```bash
black cadling/ tests/          # Format
ruff check cadling/ tests/     # Lint
mypy cadling/                  # Type check
pre-commit run --all-files     # All hooks
```

### Style

- **Formatter**: Black (line-length 88)
- **Linter**: Ruff (E, W, F, I, N, UP, B, C4 rules)
- **Type checker**: mypy (Python 3.9 target)
- **Docstrings**: Google style
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) (`feat(scope):`, `fix(scope):`, etc.)

### Adding a New CAD Format

1. Add value to `InputFormat` enum in `datamodel/base_models.py`
2. Create data model in `datamodel/my_format.py`
3. Create backend in `backend/my_format_backend.py` inheriting from `DeclarativeCADBackend` and/or `RenderableCADBackend`
4. Register default option in `DocumentConverter._get_default_format_option()`
5. Add tests

### Adding an Enrichment Model

1. Create model in `models/my_model.py` inheriting from `EnrichmentModel`
2. Implement `__call__(self, doc: CADlingDocument, item_batch: list[CADItem])`
3. Add predictions to `item.properties`
4. Add tests

### Adding a Chunker

1. Create chunker in `chunker/my_chunker.py` inheriting from `BaseCADChunker`
2. Implement `chunk(self, doc: CADlingDocument) -> Iterator[CADChunk]`
3. Add tests

## Use Cases

- **Engineering RAG** -- chunk CAD files, index in vector databases, query with LLMs
- **Automated Design Review** -- validate topology, check interference, assess manufacturability
- **Part Similarity Search** -- compute embeddings, find similar parts in CAD repositories
- **CAD-Aware LLM Training** -- generate structured training data from CAD documents
- **Optical CAD Recognition** -- extract dimensions, tolerances, and GD&T from rendered drawings via VLMs

## Project Structure

```
cadling/
  cadling/
    __init__.py                  # Public API: DocumentConverter, InputFormat, etc.
    backend/
      abstract_backend.py        # AbstractCADBackend, DeclarativeCADBackend, RenderableCADBackend
      document_converter.py      # DocumentConverter (main entry point)
      step/                      # STEP backend + ll_stepnet integration
      stl/                       # STL backend (ASCII + binary)
      brep/                      # BRep backend (pythonocc-core)
      iges_backend.py            # IGES backend
      cadling_parse_backend.py   # Image-based backend
    datamodel/
      base_models.py             # CADlingDocument, CADItem, InputFormat, ConversionResult
      step.py, stl.py, brep.py   # Format-specific models
      pipeline_options.py        # Pipeline configuration
      backend_options.py         # Backend configuration
    pipeline/
      base_pipeline.py           # BaseCADPipeline (Build/Assemble/Enrich)
      simple_pipeline.py         # SimpleCADPipeline
      hybrid_pipeline.py         # HybridPipeline (text + vision)
      vision_pipeline.py         # VisionPipeline
      vlm_pipeline.py            # VlmPipeline
      step_pipeline.py           # STEPPipeline
      stl_pipeline.py            # STLPipeline
    models/
      base_model.py              # EnrichmentModel abstract base
      geometry_analysis.py       # Geometric property analysis
      topology_validation.py     # Topology validation
      mesh_quality.py            # Mesh quality metrics
      surface_analysis.py        # Surface analysis
      interference_check.py      # Interference detection
      segmentation/              # GNN segmentation (EdgeConv, GAT, instance seg)
    chunker/
      base_chunker.py            # BaseCADChunker
      hybrid_chunker.py          # Hybrid chunking strategy
      hierarchical_chunker.py    # Hierarchy-aware chunking
      step_chunker/, stl_chunker/, brep_chunker/
      tokenizer/, serializer/, visualizer/
    sdg/
      qa/                        # Q&A generation pipeline
        sample.py                # CADPassageSampler
        generate.py              # CADGenerator
        critique.py              # CADJudge
        conceptual_generate.py   # CADConceptualGenerator
        prompts/                 # LLM prompt templates
      cli/                       # Typer CLI (cadling-sdg)
    cli/                         # Click CLI (cadling)
    lib/
      graph/                     # Graph construction and PyG export
      geometry/                  # Geometry utilities
    experimental/                # Advanced pipelines, models, and options
  tests/
    unit/, functional/, integration/
    fixtures/                    # Test CAD files
    conftest.py                  # Pytest fixtures
  docs/                          # Architecture docs, plans, adjustments
  pyproject.toml                 # Project config, tool settings, entry points
  environment.yml                # Conda environment (authoritative dep source)
  requirements.txt               # Pip dependencies (excludes PyTorch)
  CLAUDE.md                      # AI assistant guidelines
```

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Core | Pydantic 2.0+, NumPy, Pillow |
| CAD Processing | pythonocc-core (conda), numpy-stl, trimesh, NetworkX |
| Machine Learning | PyTorch (conda-forge), PyTorch Geometric, Transformers |
| Vision | EasyOCR, OpenCV |
| CLI | Click (main), Typer (SDG) |
| Development | pytest, Black, Ruff, mypy, pre-commit |

## License

MIT
