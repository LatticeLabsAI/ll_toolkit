# CADling Overview

## Vision

CADling is a **docling-inspired toolkit for CAD files** that brings document processing methodologies to Computer-Aided Design (CAD) and mesh data. It enables optical CAD recognition, STEP/STL code parsing, chunking, and understanding for LLM/ML/AI compatibility.

## What is CADling?

CADling applies proven document processing patterns from [docling](https://github.com/DS4SD/docling) to the CAD domain, supporting:

- **STEP files** (ISO 10303, text and rendered)
- **STL files** (ASCII and binary mesh formats)
- **BRep files** (Boundary Representation geometry)
- **IGES files** (Initial Graphics Exchange Specification)

### Key Capabilities

1. **Dual-Modality Processing**
   - **Text-based parsing**: Direct parsing of CAD file formats (STEP entities, STL vertices)
   - **Vision-based recognition**: OCR and vision model analysis of rendered CAD images

2. **Optical CAD Recognition**
   - Vision-language models analyze rendered CAD views
   - OCR extracts dimensions, tolerances, and annotations
   - Multiple view rendering (front, top, isometric, etc.)

3. **Neural Network Integration**
   - **ll_stepnet** integration at two layers:
     - Backend layer: Tokenization, feature extraction, topology building
     - Model layer: Classification, property prediction, similarity, captioning

4. **LLM-Compatible Output**
   - Structured representations for RAG (Retrieval-Augmented Generation)
   - Semantic chunking for vector databases
   - Markdown/JSON export for LLM consumption

5. **Synthetic Data Generation**
   - Q&A pair generation from CAD documents
   - Training data for CAD-specific models
   - Similar to docling-sdg but for geometric data

---

## Architecture Overview

CADling mirrors docling's proven layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     DocumentConverter                        │
│  Entry point for CAD file conversion with format routing    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Input Processing                          │
│  CADInputDocument → Format Detection → Backend Selection    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ STEPBackend  │  │  STLBackend  │  │ BRepBackend  │      │
│  │ (ll_stepnet) │  │   (mesh)     │  │ (pythonocc)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  Text Parsing + Rendering + Feature Extraction               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Simple     │  │   Vision     │  │   Hybrid     │      │
│  │  Pipeline    │  │  Pipeline    │  │  Pipeline    │      │
│  │ (text only)  │  │ (rendered)   │  │ (text+vis)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  Build → Assemble → Enrich                                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  CADlingDocument                             │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Hierarchical Items:                              │       │
│  │  • STEPEntityItem (geometric entities)            │       │
│  │  • MeshItem (vertices, facets, normals)           │       │
│  │  • AssemblyItem (multi-part structures)           │       │
│  │  • AnnotationItem (dimensions, tolerances)        │       │
│  └──────────────────────────────────────────────────┘       │
│  ┌──────────────────────────────────────────────────┐       │
│  │  CAD-Specific Data:                               │       │
│  │  • TopologyGraph (entity references)              │       │
│  │  • Embeddings (ll_stepnet vectors)                │       │
│  │  • 3D Bounding Boxes                              │       │
│  │  • Physical Properties (volume, mass, area)       │       │
│  └──────────────────────────────────────────────────┘       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   Output & Applications                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Chunking   │  │     SDG      │  │  Export      │      │
│  │   (RAG)      │  │  (Q&A Gen)   │  │ (JSON/MD)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Backend Layer

Backends handle format-specific parsing and feature extraction.

#### Backend Hierarchy

```python
AbstractCADBackend
├── DeclarativeCADBackend (direct text parsing)
│   └── Examples: STEP text, ASCII STL
└── RenderableCADBackend (supports rendering to images)
    └── Examples: STEP with pythonocc, BRep, IGES
```

#### Dual-Purpose Backends

Some backends inherit from **both** interfaces:

```python
class STEPBackend(DeclarativeCADBackend, RenderableCADBackend):
    """Hybrid backend supporting both text parsing and rendering"""

    def convert(self) -> CADlingDocument:
        # Text-based parsing using ll_stepnet
        pass

    def render_view(self, view_name: str) -> Image:
        # Render using pythonocc-core
        pass
```

#### ll_stepnet Integration (Backend Layer)

Backends use ll_stepnet for **parsing and feature extraction**:

- **STEPTokenizer**: Parse STEP text into tokens
- **STEPFeatureExtractor**: Extract geometric features (coordinates, radii, axes)
- **STEPTopologyBuilder**: Build entity reference graph

This data flows into `CADlingDocument` structure.

### 2. Data Models

#### CADlingDocument

The central data structure, equivalent to docling's `DoclingDocument`:

```python
class CADlingDocument(BaseModel):
    # Metadata
    name: str
    format: InputFormat  # STEP, STL, BREP, IGES
    origin: CADDocumentOrigin
    hash: str

    # Hierarchical content
    items: List[CADItem]

    # CAD-specific data
    topology: Optional[TopologyGraph]  # Entity references
    embeddings: Optional[torch.Tensor]  # ll_stepnet embeddings

    # Methods
    def export_to_json() -> Dict[str, Any]
    def export_to_markdown() -> str
```

#### CADItem Hierarchy

```python
CADItem (base)
├── STEPEntityItem
│   ├── entity_type: "CARTESIAN_POINT", "CIRCLE", "ADVANCED_FACE"
│   ├── numeric_params: [x, y, z, radius, ...]
│   └── features: ll_stepnet extracted features
├── MeshItem
│   ├── vertices: [[x,y,z], ...]
│   ├── normals: [[nx,ny,nz], ...]
│   └── facets: [[v1,v2,v3], ...]
├── AssemblyItem
│   └── components: References to child items
└── AnnotationItem
    ├── annotation_type: "dimension", "tolerance", "note"
    └── value: Extracted from rendered view (OCR/VLM)
```

### 3. Pipeline Layer

Pipelines orchestrate the conversion workflow: **Build → Assemble → Enrich**.

#### Pipeline Types

1. **SimpleCADPipeline**
   - For text-based formats (STEP text, ASCII STL)
   - Direct conversion without rendering
   - Uses `DeclarativeCADBackend.convert()`

2. **CADVlmPipeline** (Vision Pipeline)
   - For rendered CAD images
   - Renders multiple views (front, top, isometric)
   - Applies vision-language model for "optical CAD recognition"
   - Extracts `AnnotationItem` objects (dimensions, tolerances)

3. **HybridCADPipeline**
   - **Key Innovation**: Combines text parsing + vision analysis
   - Parse STEP entities using ll_stepnet
   - Render views and apply VLM
   - Fuse information into unified `CADlingDocument`

#### Pipeline Selection

```python
# DocumentConverter automatically selects pipeline

STEP file → SimpleCADPipeline (text parsing)
STEP file + render_views=True → HybridCADPipeline (text + vision)
STL file → SimpleCADPipeline (mesh parsing)
CAD image → CADVlmPipeline (vision only)
```

### 4. Model Layer (Enrichment)

Enrichment models add predictions to `CADlingDocument` using ll_stepnet task models.

#### ll_stepnet Integration (Model Layer)

Models use ll_stepnet for **inference tasks**:

- **STEPForClassification**: Part classification (bracket, housing, shaft)
- **STEPForPropertyPrediction**: Physical properties (volume, mass, surface area)
- **STEPForSimilarity**: Generate embeddings for RAG/search
- **STEPForCaptioning**: Generate text descriptions of parts
- **STEPForQA**: Question answering about CAD geometry

Example:

```python
class CADPartClassifier(EnrichmentModel):
    """Classify CAD parts using ll_stepnet"""

    def __init__(self, artifacts_path: Path):
        self.model = STEPForClassification.from_pretrained(
            artifacts_path / "step_classifier.pt"
        )

    def __call__(self, doc: CADlingDocument, item_batch: List[CADItem]):
        # Run inference and add predictions to items
        logits = self.model(token_ids, topology_data=doc.topology)
        predicted_class = torch.argmax(logits, dim=1)
        item.properties["predicted_class"] = predicted_class.item()
```

### 5. Chunking (cadling-core)

Semantic chunking for RAG systems, adapted from docling-core.

#### CAD-Specific Chunkers

- **CADHybridChunker**: Combines entity-level and semantic chunking
- **CADHierarchicalChunker**: Respects assembly hierarchy
- **CADTopologyChunker**: Groups entities by topological relationships

Each chunk contains:
- Text representation of CAD entities
- Topology subgraph
- Embeddings (if enriched with ll_stepnet)
- Metadata (entity types, properties)

### 6. Synthetic Data Generation (cadling-sdg)

Generate training data from CAD documents, similar to docling-sdg.

#### CADQAGenerator

```python
from cadling_sdg.qa import CADQAGenerator

qa_generator = CADQAGenerator(llm_model="gpt-4")
qa_pairs = qa_generator.generate_qa_pairs(doc, num_pairs=100)

# Example questions:
# - "What is the radius of the cylindrical surface at entity #31?"
# - "How many ADVANCED_FACE entities are in this part?"
# - "What is the estimated volume of this part?"
```

---

## Use Cases

### 1. CAD File Parsing and Conversion

Convert CAD files to LLM-compatible formats:

```python
from cadling import DocumentConverter, InputFormat

converter = DocumentConverter(
    allowed_formats=[InputFormat.STEP, InputFormat.STL]
)

result = converter.convert("part.step")
doc = result.document  # CADlingDocument

# Export to JSON
doc.export_to_json("part.json")
```

### 2. Chunking for RAG Systems

Break down CAD files for vector databases:

```python
from cadling_core.transforms.chunker import CADHybridChunker

chunker = CADHybridChunker(max_tokens=512)
chunks = list(chunker.chunk(doc))

# Each chunk has text, topology, embeddings, metadata
for chunk in chunks:
    vector_db.add(chunk.text, chunk.meta.embedding)
```

### 3. Synthetic Data Generation

Generate Q&A pairs for training:

```python
from cadling_sdg.qa import CADQAGenerator

qa_generator = CADQAGenerator()
qa_pairs = qa_generator.generate_qa_pairs(doc, num_pairs=100)
```

### 4. Optical CAD Recognition

Extract dimensions and annotations from rendered CAD images:

```python
converter = DocumentConverter(
    format_options={
        InputFormat.STEP: STEPFormatOption(
            pipeline_cls=HybridCADPipeline,
            render_views=True,
            vlm_enabled=True
        )
    }
)

result = converter.convert("technical_drawing.step")

# Extract annotations
annotations = [
    item for item in result.document.items
    if isinstance(item, AnnotationItem)
]
```

---

## Comparison: Docling vs CADling

| Aspect | Docling | CADling |
|--------|---------|---------|
| **Domain** | Documents (PDF, DOCX, HTML) | CAD files (STEP, STL, BRep, IGES) |
| **Input** | 2D documents with text, tables, images | 3D geometry, meshes, parametric data |
| **Parsing** | Text extraction, layout analysis | Entity parsing, mesh processing, topology building |
| **Vision** | Document layout understanding | Optical CAD recognition (dimensions, annotations) |
| **Output** | DoclingDocument (text, tables, figures) | CADlingDocument (entities, meshes, topology) |
| **Chunking** | Semantic text chunks | Geometric/topological chunks |
| **Neural Models** | TableFormer, OCR, VLM | ll_stepnet (STEP-aware neural network) |
| **Use Cases** | Document Q&A, RAG, parsing | CAD Q&A, RAG, design understanding |

---

## Technology Stack

### Core Dependencies

- **pythonocc-core**: CAD kernel for STEP/BRep/IGES parsing and rendering
- **ll_stepnet**: STEP-aware neural network for tokenization, features, and inference
- **pydantic**: Data validation and serialization
- **torch**: Neural network inference
- **numpy-stl** / **trimesh**: STL mesh processing

### Optional Dependencies

- **Vision models**: For optical CAD recognition (GPT-4V, Claude, open-source VLMs)
- **OCR**: For text extraction from rendered CAD (EasyOCR, Tesseract)
- **Graph libraries**: NetworkX, PyTorch Geometric (for topology graphs)

---

## Project Structure

```
cadling/
├── cadling/
│   ├── backend/              # Format-specific parsers
│   │   ├── abstract_backend.py
│   │   ├── step_backend.py
│   │   ├── stl_backend.py
│   │   └── brep_backend.py
│   ├── datamodel/            # Data structures
│   │   ├── base_models.py
│   │   ├── step.py
│   │   ├── stl.py
│   │   └── pipeline_options.py
│   ├── pipeline/             # Conversion workflows
│   │   ├── base_pipeline.py
│   │   ├── simple_pipeline.py
│   │   └── vlm_pipeline.py
│   ├── models/               # ll_stepnet task models
│   │   ├── classification.py
│   │   ├── property_prediction.py
│   │   └── similarity.py
│   ├── chunker/              # Semantic chunking
│   │   ├── step_chunker/
│   │   └── mesh_chunker/
│   ├── sdg/                  # Synthetic data generation
│   │   └── qa_generator.py
│   └── cli/                  # Command-line interface
└── README.md
```

---

## Next Steps

See [Plan.md](Plan.md) for the detailed implementation roadmap.

See [Development.md](Development.md) for development guidelines and best practices.

See [Adjustments.md](Adjustments.md) for specific docling → cadling adaptations.

---

## References

- **Docling**: https://github.com/DS4SD/docling
- **Docling-core**: https://github.com/DS4SD/docling-core
- **Docling-sdg**: https://github.com/DS4SD/docling-sdg
- **ll_stepnet**: CAD-specific neural network (local module)
- **PythonOCC**: https://github.com/tpaviot/pythonocc-core

---

## Recent Research

### Neural Networks for Mesh Understanding (2025)

- **InfoGNN**: End-to-end framework for 3D mesh classification and segmentation using InfoConv and InfoMP modules
- **MeshCNN**: Convolutional neural network for triangular meshes with edge-based operations
- **MeshNet**: Face-unit and feature splitting for 3D shape representation
- **3D Mesh Transformers**: Hierarchical neural networks with local shape tokens

Sources:
- [InfoGNN: End-to-end deep learning on mesh via graph neural networks](https://arxiv.org/html/2503.02414v1)
- [MeshCNN](https://ranahanocka.github.io/MeshCNN/)
- [MeshNet | AAAI 2019](https://dl.acm.org/doi/10.1609/aaai.v33i01.33018279)
- [3D mesh transformer: A hierarchical neural network](https://www.sciencedirect.com/science/article/abs/pii/S0925231222012383)
