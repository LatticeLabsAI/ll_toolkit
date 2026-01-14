# Docling → CADling Adjustments

This document details the specific adaptations made when translating docling's architecture to the CAD domain.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Model Adjustments](#data-model-adjustments)
3. [Backend Adjustments](#backend-adjustments)
4. [Pipeline Adjustments](#pipeline-adjustments)
5. [Chunking Adjustments](#chunking-adjustments)
6. [Model Integration Adjustments](#model-integration-adjustments)
7. [Format-Specific Adjustments](#format-specific-adjustments)
8. [Rationale Summary](#rationale-summary)

---

## Overview

Docling is designed for **2D documents** (PDF, DOCX, HTML) with text, tables, and images. CADling adapts this architecture for **3D CAD files** (STEP, STL, BRep, IGES) with geometry, meshes, and parametric data.

### Core Differences

| Aspect | Docling | CADling |
|--------|---------|---------|
| **Domain** | 2D documents | 3D CAD files |
| **Input** | Pages of text, images, tables | 3D geometry, meshes, parametric entities |
| **Parsing** | Text extraction, OCR, layout analysis | Entity parsing, mesh processing, topology building |
| **Output** | DoclingDocument (structured text) | CADlingDocument (structured geometry) |
| **Chunking** | Text-based semantic chunks | Geometric/topological chunks |
| **Vision** | Document layout understanding | Optical CAD recognition (dimensions, annotations) |

---

## Data Model Adjustments

### DoclingDocument → CADlingDocument

#### Docling Structure

```python
class DoclingDocument(BaseModel):
    name: str
    origin: DocumentOrigin
    items: List[DocItem]  # TextItem, TableItem, PictureItem, etc.
    metadata: Dict[str, Any]
```

#### CADling Adaptation

```python
class CADlingDocument(BaseModel):
    name: str
    format: InputFormat  # NEW: STEP, STL, BREP, IGES
    origin: CADDocumentOrigin
    items: List[CADItem]  # STEPEntityItem, MeshItem, AssemblyItem, AnnotationItem

    # NEW: CAD-specific data
    topology: Optional[TopologyGraph]  # Entity reference graph
    embeddings: Optional[torch.Tensor]  # ll_stepnet embeddings

    # NEW: CAD-specific methods
    def export_to_json() -> Dict[str, Any]
    def export_to_markdown() -> str
```

**Key Adjustments**:

1. **Added `format` field**: CAD files come in multiple formats with different parsing strategies
2. **Added `topology` field**: CAD entities reference each other (e.g., #31 references #15)
3. **Added `embeddings` field**: ll_stepnet generates neural embeddings for entities
4. **Export methods**: CAD data needs special serialization (3D coordinates, topology graphs)

### DocItem → CADItem

#### Docling Items

```python
DocItem (base)
├── TextItem (text blocks)
├── TableItem (tabular data)
├── PictureItem (images, figures)
├── SectionHeaderItem
└── ListItem
```

#### CADling Items

```python
CADItem (base)
├── STEPEntityItem (geometric entities: CARTESIAN_POINT, CIRCLE, FACE)
├── MeshItem (vertices, facets, normals)
├── AssemblyItem (multi-part hierarchies)
└── AnnotationItem (dimensions, tolerances from vision)
```

**Key Adjustments**:

1. **3D Bounding Boxes**: `BoundingBox3D` instead of 2D bounding boxes
2. **Geometric Properties**: Volume, mass, surface area instead of text properties
3. **Entity References**: CAD entities reference other entities (topology)
4. **ll_stepnet Features**: Geometric features extracted by neural network

```python
class STEPEntityItem(CADItem):
    item_type: str = "step_entity"

    # NEW: STEP-specific fields
    entity_id: int  # e.g., #31
    entity_type: str  # e.g., "CYLINDRICAL_SURFACE"
    numeric_params: List[float]  # e.g., [0.0, 0.0, 1.0, 5.0]
    reference_params: List[int]  # e.g., [#15, #22]

    # NEW: ll_stepnet features
    features: Optional[Dict[str, Any]]

    # Inherited from CADItem
    bbox: Optional[BoundingBox3D]  # 3D instead of 2D
    properties: Dict[str, Any]  # volume, mass, etc.
```

### BoundingBox → BoundingBox3D

#### Docling 2D BoundingBox

```python
class BoundingBox(BaseModel):
    l: float  # left
    t: float  # top
    r: float  # right
    b: float  # bottom
```

#### CADling 3D BoundingBox

```python
class BoundingBox3D(BaseModel):
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2
        )

    @property
    def volume(self) -> float:
        return (
            (self.x_max - self.x_min) *
            (self.y_max - self.y_min) *
            (self.z_max - self.z_min)
        )
```

**Rationale**: CAD files contain 3D geometry requiring 3D spatial representation.

---

## Backend Adjustments

### Backend Hierarchy Changes

#### Docling Backends

```python
AbstractDocumentBackend (base)
├── DeclarativeDocumentBackend (direct conversion)
│   └── Examples: DOCX, HTML, Markdown
└── PaginatedDocumentBackend (page-level processing)
    └── Examples: PDF
```

#### CADling Backends

```python
AbstractCADBackend (base)
├── DeclarativeCADBackend (text parsing)
│   └── Examples: STEP text, ASCII STL
├── RenderableCADBackend (supports rendering)
│   └── Examples: STEP with pythonocc, BRep
└── Hybrid backends (inherit from both)
    └── Example: STEPBackend (text + rendering)
```

**Key Adjustments**:

1. **Renamed "Paginated" → "Renderable"**: CAD files don't have "pages" but can be rendered as multiple views
2. **Dual-inheritance support**: Some backends support BOTH text parsing AND rendering
3. **View concept**: Instead of pages, CAD backends render "views" (front, top, isometric)

### PdfDocumentBackend → STEPBackend Example

#### Docling PDF Backend

```python
class PdfDocumentBackend(PaginatedDocumentBackend):
    """PDF backend with page-level processing"""

    def load_page(self, page_num: int) -> PageBackend:
        """Load a specific page"""
        pass

    def num_pages(self) -> int:
        """Total number of pages"""
        pass

    def render_page(self, page_num: int) -> Image:
        """Render page to image"""
        pass
```

#### CADling STEP Backend

```python
class STEPBackend(DeclarativeCADBackend, RenderableCADBackend):
    """STEP backend with text parsing + rendering"""

    # DeclarativeCADBackend interface
    def convert(self) -> CADlingDocument:
        """Parse STEP text using ll_stepnet"""
        pass

    # RenderableCADBackend interface
    def available_views(self) -> List[str]:
        """Available views instead of pages"""
        return ["front", "top", "right", "isometric"]

    def render_view(self, view_name: str) -> Image:
        """Render view using pythonocc-core"""
        pass
```

**Rationale**: STEP files have structured text (parsable) AND 3D geometry (renderable).

### Backend Method Changes

| Docling Method | CADling Equivalent | Change Rationale |
|----------------|-------------------|------------------|
| `load_page(page_num)` | `load_view(view_name)` | CAD has views, not pages |
| `num_pages()` | `available_views()` | Returns list of view names |
| `render_page(page_num)` | `render_view(view_name)` | Named views instead of numbered pages |

---

## Pipeline Adjustments

### Pipeline Hierarchy Changes

#### Docling Pipelines

```python
BasePipeline (abstract)
├── SimplePipeline (declarative backends)
├── StandardPdfPipeline (PDF processing)
└── VlmPipeline (vision-language model analysis)
```

#### CADling Pipelines

```python
BaseCADPipeline (abstract)
├── SimpleCADPipeline (text-based CAD)
├── CADVlmPipeline (optical CAD recognition)
└── HybridCADPipeline (text + vision) [NEW]
```

**Key Addition**: `HybridCADPipeline` combines text parsing + vision analysis, a pattern unique to CAD where we can leverage BOTH modalities.

### StandardPdfPipeline → HybridCADPipeline

#### Docling PDF Pipeline

```python
class StandardPdfPipeline(BasePipeline):
    """PDF processing with layout analysis"""

    def _build_document(self, conv_res):
        # Process page by page
        for page_num in range(backend.num_pages()):
            page = backend.load_page(page_num)
            # Extract text, tables, figures
            items = self._extract_items(page)
            conv_res.document.items.extend(items)
        return conv_res
```

#### CADling Hybrid Pipeline

```python
class HybridCADPipeline(BaseCADPipeline):
    """Combines text parsing + vision analysis"""

    def _build_document(self, conv_res):
        backend = conv_res.input._backend

        # Phase 1: Text parsing (if supported)
        if isinstance(backend, DeclarativeCADBackend):
            doc = backend.convert()  # Parse STEP entities
        else:
            doc = CADlingDocument(name=backend.file.name)

        # Phase 2: Vision analysis (if supported)
        if isinstance(backend, RenderableCADBackend):
            for view_name in backend.available_views():
                view_image = backend.render_view(view_name)
                # Extract dimensions, annotations using VLM
                annotations = self._extract_annotations(view_image)
                doc.items.extend(annotations)

        # Phase 3: Fuse information
        doc = self._fuse_text_and_vision(doc)

        conv_res.document = doc
        return conv_res
```

**Rationale**: CAD files can be processed in TWO ways simultaneously:
1. Parse structured text (STEP entities)
2. Analyze rendered images (dimensions, annotations)

This dual-modality approach is unique to CAD and doesn't exist in docling.

### Pipeline Stage Adjustments

| Docling Stage | CADling Equivalent | Adjustments |
|---------------|-------------------|-------------|
| Build | Build | Extract geometry instead of text |
| Assemble | Assemble | Build topology graph instead of document structure |
| Enrich | Enrich | Apply ll_stepnet models instead of TableFormer |

---

## Chunking Adjustments

### DoclingChunker → CADChunker

#### Docling Chunking

```python
from docling_core.transforms.chunker import HybridChunker

chunker = HybridChunker(tokenizer="...", max_tokens=512)

for chunk in chunker.chunk(doc):
    print(chunk.text)  # Text content
    print(chunk.meta.headings)  # Section headings
```

**Focus**: Semantic text chunking based on document structure (headings, paragraphs).

#### CADling Chunking

```python
from cadling.chunker import CADHybridChunker

chunker = CADHybridChunker(max_tokens=512)

for chunk in chunker.chunk(doc):
    print(chunk.text)  # Text representation of entities
    print(chunk.meta.entity_types)  # STEP entity types
    print(chunk.meta.topology_subgraph)  # Entity reference graph
    print(chunk.meta.embedding)  # ll_stepnet embeddings
```

**Focus**: Geometric/topological chunking with:
1. **Entity grouping**: Group related geometric entities
2. **Topology preservation**: Include entity reference subgraph
3. **Embeddings**: Include neural embeddings for similarity search

### ChunkMeta Adjustments

#### Docling ChunkMeta

```python
class ChunkMeta(BaseModel):
    doc_items: List[RefItem]  # Document items in chunk
    headings: List[str]  # Section headings
    origin: PageOrigin  # Source page
```

#### CADling ChunkMeta

```python
class CADChunkMeta(BaseModel):
    doc_items: List[RefItem]  # CAD items in chunk

    # NEW: CAD-specific metadata
    entity_types: List[str]  # e.g., ["CARTESIAN_POINT", "CIRCLE"]
    entity_ids: List[int]  # e.g., [#31, #42, #58]
    topology_subgraph: Optional[TopologyGraph]  # Entity references
    embedding: Optional[np.ndarray]  # Averaged ll_stepnet embeddings
    properties: Dict[str, Any]  # Volume, mass, etc.
```

**Rationale**: RAG for CAD requires geometric metadata and topology context.

### Chunking Strategy Differences

| Aspect | Docling | CADling |
|--------|---------|---------|
| **Chunk boundary** | Semantic (headings, paragraphs) | Topological (entity relationships) |
| **Chunk size** | Token count | Token count + topology complexity |
| **Metadata** | Headings, page numbers | Entity types, topology, embeddings |
| **Use case** | Text retrieval | Geometric similarity search |

---

## Model Integration Adjustments

### Docling Models → ll_stepnet Integration

#### Docling Enrichment Models

```python
# Examples from docling
- TableFormer: Table structure recognition
- LayoutModel: Document layout analysis
- OCR models: Text extraction
- VLM models: Vision-language understanding
```

**Focus**: Text and layout understanding.

#### CADling Enrichment Models

```python
# ll_stepnet task models
- STEPForClassification: Part classification (bracket, housing, shaft)
- STEPForPropertyPrediction: Physical properties (volume, mass)
- STEPForSimilarity: Embeddings for RAG
- STEPForCaptioning: Generate text descriptions
- STEPForQA: Question answering
```

**Focus**: Geometric understanding and property prediction.

### Two-Layer Integration

**Key Difference**: ll_stepnet is used at TWO layers in cadling, whereas docling models are only used in the enrichment stage.

#### Backend Layer (Parsing)

```python
# In STEPBackend.convert()
from ll_stepnet import STEPTokenizer, STEPFeatureExtractor, STEPTopologyBuilder

# Use ll_stepnet for PARSING
tokenizer = STEPTokenizer()
feature_extractor = STEPFeatureExtractor()
topology_builder = STEPTopologyBuilder()

tokens = tokenizer.encode(step_text)
features = feature_extractor.extract_geometric_features(entity)
topology = topology_builder.build_complete_topology(features_list)
```

**Purpose**: Parse and extract structure from CAD files.

#### Model Layer (Enrichment)

```python
# In enrichment pipeline
from ll_stepnet.models import STEPForClassification

# Use ll_stepnet for INFERENCE
model = STEPForClassification.from_pretrained("path/to/model")
logits = model(token_ids, topology_data=topology)
predicted_class = torch.argmax(logits, dim=1)
```

**Purpose**: Add predictions and embeddings to parsed CAD documents.

**Rationale**: ll_stepnet is both a parser AND a neural network, unlike docling's models which are only used for enrichment.

---

## Format-Specific Adjustments

### PDF → STEP

| Aspect | PDF (Docling) | STEP (CADling) |
|--------|---------------|----------------|
| **Structure** | Pages of text/images | Entities with references |
| **Parsing** | Text extraction, OCR | Entity parsing, topology building |
| **Backend** | PdfDocumentBackend | STEPBackend (dual inheritance) |
| **Output** | TextItem, TableItem | STEPEntityItem |
| **Key challenge** | Layout understanding | Entity reference resolution |

### DOCX → STL

| Aspect | DOCX (Docling) | STL (CADling) |
|--------|----------------|---------------|
| **Structure** | Structured XML | Mesh (vertices, facets) |
| **Parsing** | XML parsing | Mesh parsing (ASCII/binary) |
| **Backend** | MsWordDocumentBackend | STLBackend |
| **Output** | TextItem, TableItem | MeshItem |
| **Key challenge** | Style preservation | Mesh validation (manifold, watertight) |

### Image → CAD Image (Rendered)

| Aspect | Image (Docling) | CAD Image (CADling) |
|--------|-----------------|---------------------|
| **Structure** | Raster image | Rendered CAD view |
| **Processing** | VLM for content | VLM for dimensions/annotations |
| **Backend** | ImageDocumentBackend | CADImageBackend |
| **Output** | PictureItem | AnnotationItem |
| **Key challenge** | Object detection | Dimension/tolerance extraction |

---

## Rationale Summary

### Why These Adjustments?

#### 1. 3D vs 2D

**Problem**: Documents are 2D (pages), CAD is 3D (geometry).

**Solution**:
- `BoundingBox3D` instead of `BoundingBox`
- `available_views()` instead of `num_pages()`
- `render_view(name)` instead of `render_page(num)`

#### 2. Topology Graphs

**Problem**: CAD entities reference each other (e.g., a FACE references EDGEs), forming a graph structure.

**Solution**:
- Add `TopologyGraph` to `CADlingDocument`
- Include topology in chunk metadata
- Use ll_stepnet's topology builder

#### 3. Dual Modality

**Problem**: CAD files can be both text (STEP entities) AND visual (rendered images).

**Solution**:
- Dual-inheritance backends (`DeclarativeCADBackend` + `RenderableCADBackend`)
- `HybridCADPipeline` that combines both modalities
- Information fusion strategies

#### 4. ll_stepnet Integration

**Problem**: CAD parsing requires domain-specific neural network.

**Solution**:
- Use ll_stepnet at backend layer (parsing)
- Use ll_stepnet at model layer (enrichment)
- Two-layer integration pattern

#### 5. Geometric Metadata

**Problem**: CAD chunks need geometric context for RAG.

**Solution**:
- Add `entity_types`, `topology_subgraph`, `embeddings` to chunk metadata
- Chunk based on topological relationships
- Include geometric properties (volume, mass)

---

## Migration Guide: Docling Code → CADling Code

### Example: Converting a Document

#### Docling

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")

# Export to markdown
markdown = result.document.export_to_markdown()
```

#### CADling

```python
from cadling.document_converter import DocumentConverter
from cadling.datamodel.base_models import InputFormat

converter = DocumentConverter(
    allowed_formats=[InputFormat.STEP]
)
result = converter.convert("part.step")

# Export to markdown
markdown = result.document.export_to_markdown()
```

**Changes**: Added `allowed_formats` and `InputFormat` enum.

### Example: Chunking for RAG

#### Docling

```python
from docling_core.transforms.chunker import HybridChunker

chunker = HybridChunker(tokenizer="...", max_tokens=512)
chunks = list(chunker.chunk(doc))

for chunk in chunks:
    vector_db.add(chunk.text, metadata=chunk.meta)
```

#### CADling

```python
from cadling.chunker import CADHybridChunker

chunker = CADHybridChunker(max_tokens=512)
chunks = list(chunker.chunk(doc))

for chunk in chunks:
    vector_db.add(
        chunk.text,
        embedding=chunk.meta.embedding,  # NEW: ll_stepnet embeddings
        metadata={
            "entity_types": chunk.meta.entity_types,  # NEW
            "topology": chunk.meta.topology_subgraph,  # NEW
        }
    )
```

**Changes**: Added embeddings and CAD-specific metadata.

### Example: Adding Enrichment Models

#### Docling

```python
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.models.table_former import TableFormer

pipeline_options = PipelineOptions(
    enrichment_models=[TableFormer()]
)

result = converter.convert("document.pdf", pipeline_options=pipeline_options)
```

#### CADling

```python
from cadling.pipeline.simple_pipeline import SimpleCADPipeline
from cadling.models.classification import CADPartClassifier

pipeline_options = PipelineOptions(
    enrichment_models=[
        CADPartClassifier(artifacts_path)  # ll_stepnet model
    ]
)

result = converter.convert("part.step", pipeline_options=pipeline_options)
```

**Changes**: Replace docling models with ll_stepnet-based CAD models.

---

## Conclusion

CADling successfully adapts docling's proven architecture for the CAD domain by:

1. **Extending data models** for 3D geometry and topology
2. **Adapting backends** to support dual modalities (text + vision)
3. **Creating hybrid pipelines** that fuse text and visual information
4. **Integrating ll_stepnet** at both parsing and enrichment layers
5. **Adjusting chunking** for geometric similarity search

These adjustments preserve docling's core patterns (backend → pipeline → enrichment) while addressing the unique challenges of CAD file processing.

For implementation details, see:
- [Overview.md](Overview.md) - Architecture overview
- [Plan.md](Plan.md) - Implementation roadmap
- [Development.md](Development.md) - Development guidelines
