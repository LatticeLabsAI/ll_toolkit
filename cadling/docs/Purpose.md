# Purpose of CADling

## What is CADling?

**CADling** is a comprehensive toolkit for CAD (Computer-Aided Design) document processing, inspired by the architecture and design patterns of [Docling](https://github.com/DS4SD/docling). Just as Docling bridges the gap between document formats and AI/ML systems by enabling intelligent document parsing and understanding, CADling extends this paradigm to the world of 3D CAD files.

CADling provides:
- **Optical CAD Recognition**: Vision-based extraction of annotations, dimensions, and tolerances from rendered CAD images
- **Deep Code Parsing**: Text-based parsing of STEP entities, STL meshes, and other CAD formats
- **Topology Analysis**: Graph-based analysis of entity relationships and geometric hierarchies
- **Chunking for RAG**: Intelligent splitting of CAD documents for Retrieval-Augmented Generation
- **LLM/ML/AI Compatibility**: Standardized data structures optimized for AI consumption

## Why Does CADling Exist?

### The Problem

CAD files are ubiquitous in engineering, manufacturing, and product design, yet they remain largely opaque to modern AI systems:

1. **Inaccessible to LLMs**: CAD files (STEP, STL, BRep, IGES) cannot be directly processed by Large Language Models. Unlike text documents or even PDFs, there's no standard pipeline for converting CAD geometry into AI-readable formats.

2. **Locked Visual Information**: Engineering drawings contain critical information in visual annotations—dimensions, tolerances (GD&T), surface finishes, welding symbols—that are not encoded in the geometric data itself.

3. **Complex Topology**: CAD models encode sophisticated topological relationships (solids→shells→faces→edges→vertices) that require specialized understanding to extract and represent.

4. **Fragmented Tooling**: Existing CAD processing tools are either:
   - Format-specific (only handle STEP or only STL)
   - Rendering-focused (for visualization, not understanding)
   - Proprietary (closed-source or commercial-only)
   - Not AI-ready (don't output ML-friendly formats)

5. **No RAG Support**: There's no established method for chunking CAD files for vector databases or retrieval systems, preventing the use of RAG architectures with engineering data.

### The Opportunity

The convergence of several technologies creates a unique opportunity:

- **Vision-Language Models (VLMs)**: GPT-4V, Claude Vision, and open-source VLMs can now "read" engineering drawings
- **Open CAD Libraries**: pythonocc-core, trimesh, and other open-source tools provide CAD processing primitives
- **Proven Architecture**: Docling has validated a modular, pipeline-based approach for document AI
- **Engineering AI Demand**: Manufacturing, product design, and engineering teams increasingly need AI systems that understand CAD

## What CADling Aims to Accomplish

### Core Mission

**Make CAD files first-class citizens in the AI ecosystem.**

CADling aims to do for CAD what Docling does for documents: provide a robust, extensible, open-source toolkit that bridges the gap between engineering file formats and modern AI systems.

### Key Goals

#### 1. **Multimodal CAD Understanding**

Combine complementary processing modalities:
- **Text-based parsing**: Extract precise geometric data (coordinates, topology, entity types) from STEP, STL, BRep formats
- **Vision-based recognition**: Use VLMs to extract visual annotations (dimensions, tolerances, notes) from rendered CAD views
- **Hybrid fusion**: Merge text and vision modalities for comprehensive understanding

#### 2. **Format Coverage**

Support the most important CAD formats:
- ✅ **STEP** (ISO 10303-21): The lingua franca of CAD interoperability
- ✅ **STL**: The universal mesh format for 3D printing and manufacturing
- ✅ **BRep**: OpenCASCADE's native boundary representation format
- 🔄 **IGES**: Legacy CAD exchange format (future)
- 🔄 **OBJ**: Common mesh format with materials (future)
- 🔄 **DXF**: 2D drawing exchange format (future)

#### 3. **AI-Native Architecture**

Design specifically for AI/ML consumption:
- **Structured Output**: Pydantic models with JSON/Markdown export
- **Topology Graphs**: NetworkX-compatible adjacency lists for GNN processing
- **Feature Vectors**: Fixed-size embeddings for similarity search
- **Chunking Strategies**: Multiple strategies (sequential, entity-type, spatial, connectivity) for RAG
- **Provenance Tracking**: Full lineage tracking for AI explainability

#### 4. **ll_stepnet Integration**

Integrate STEP-aware neural networks for:
- **Part Classification**: Classify CAD parts by function (bracket, shaft, housing, etc.)
- **Property Prediction**: Predict manufacturing properties (mass, volume, surface area)
- **Similarity Search**: Find similar parts in CAD repositories
- **Feature Recognition**: Identify machining features (holes, slots, pockets)

Unlike existing tools that just wrap ll_stepnet, CADling implements:
- **From-scratch parsing**: Complete STEP parser with no external dependencies
- **Custom tokenization**: Token-level encoding for transformer models
- **Feature extraction**: 128+ geometric and topological features per entity
- **Topology analysis**: Multi-level graph analysis (entity, feature, assembly)

#### 5. **Production-Ready Quality**

Follow best practices for production software:
- **Docling-inspired architecture**: Proven patterns (Backend→Pipeline→Enrichment)
- **Comprehensive testing**: Unit tests, integration tests, end-to-end tests
- **Type safety**: Full type hints and Pydantic validation
- **Logging and monitoring**: Structured logging with performance metrics
- **Error handling**: Graceful degradation with detailed error reporting
- **Documentation**: Inline docs, examples, and tutorials

## How CADling Relates to Docling

### Inspiration

CADling adopts Docling's core architectural patterns:

```
Docling Pattern:              CADling Adaptation:
────────────────              ───────────────────
Document Formats      →       CAD Formats (STEP, STL, BRep)
PDF Pages             →       CAD Views (front, top, isometric)
OCR + Layout          →       Parsing + Topology + VLM
Document Items        →       CAD Items (entities, meshes, annotations)
Chunking for RAG      →       Spatial/Topology-aware chunking
Export to Markdown    →       Export to JSON/Markdown/Graph formats
```

### Key Adaptations

1. **3D vs 2D**:
   - Documents have pages (sequential 2D sheets)
   - CAD has views (projected 2D views of 3D geometry)

2. **Topology vs Layout**:
   - Documents have layout (reading order, columns, sections)
   - CAD has topology (entity references, assembly hierarchy, geometric relationships)

3. **Text vs Geometry**:
   - Documents primarily contain text with some images
   - CAD primarily contains geometry with some annotations

4. **Bounding Boxes**:
   - Documents use 2D bounding boxes (x, y, width, height)
   - CAD uses 3D bounding boxes (x_min, y_min, z_min, x_max, y_max, z_max)

### Divergence

While inspired by Docling, CADling diverges where necessary:

- **Custom data models**: STEPEntityItem, MeshItem, TopologyGraph
- **CAD-specific backends**: STEP parser, STL parser, BRep loader
- **Geometric features**: Coordinates, radii, volumes, surface areas
- **Topology analysis**: Entity reference graphs, connected components, Euler characteristic
- **Spatial chunking**: Octree-based subdivision for 3D spatial reasoning

## Vision for the Future

### Short-Term (Next 3-6 months)

1. **Format Expansion**: Add IGES, DXF, OBJ support
2. **ll_stepnet Models**: Train and integrate task-specific models
3. **Rendering Improvements**: Better off-screen rendering with pythonocc
4. **Performance Optimization**: Parallel processing, caching, streaming
5. **Documentation**: Comprehensive tutorials and examples

### Medium-Term (6-12 months)

1. **Assembly Understanding**: Multi-part assembly processing and relationships
2. **PMI Extraction**: Product Manufacturing Information (GD&T, tolerances, finishes)
3. **CAD Search**: Semantic search over CAD repositories
4. **CAD Generation**: LLM-to-CAD generation (text→STEP)
5. **Cloud Deployment**: Scalable cloud service for CAD processing

### Long-Term Vision

**CADling as the standard library for CAD AI.**

We envision CADling becoming:
- The default toolkit for processing CAD in AI pipelines
- The foundation for CAD-aware LLMs and multimodal models
- A bridge between engineering tools (SolidWorks, Fusion360, FreeCAD) and AI systems
- An enabler for "AI-native" CAD tools that understand design intent
- A catalyst for automated design review, optimization, and generation

## Use Cases

### 1. **Engineering RAG Systems**

Build retrieval-augmented generation systems for engineering:
```python
from cadling import DocumentConverter
from cadling.chunking import EntityTypeChunker

# Convert CAD files to chunks
converter = DocumentConverter()
result = converter.convert("assembly.step")

chunker = EntityTypeChunker(chunk_size=100)
chunks = chunker.chunk(result.document)

# Index in vector database
for chunk in chunks:
    vector_db.add(chunk.text, embeddings=chunk.embeddings)

# Query with LLM
query = "Find all mounting brackets in the assembly"
relevant_chunks = vector_db.search(query)
llm_response = llm.generate(query, context=relevant_chunks)
```

### 2. **Automated Design Review**

AI-powered design review and validation:
```python
from cadling import DocumentConverter
from cadling.models import GeometricPropertyEnricher, EntityClassifier

converter = DocumentConverter()
result = converter.convert("part.step")
doc = result.document

# Check for design issues
for item in doc.items:
    if item.properties["complexity_category"] == "complex":
        print(f"Complex geometry may be difficult to manufacture: {item.label}")

    if item.properties["entity_class"] == "topology" and item.has_error:
        print(f"Topology error detected: {item.label}")
```

### 3. **Part Similarity Search**

Find similar parts in large CAD repositories:
```python
from cadling import DocumentConverter
from cadling.models import SimilarityModel

# Index repository
similarity_model = SimilarityModel()
repository = []

for cad_file in repository_files:
    result = converter.convert(cad_file)
    embeddings = similarity_model.compute_embeddings(result.document)
    repository.append((cad_file, embeddings))

# Search for similar parts
query_result = converter.convert("query_part.step")
query_embeddings = similarity_model.compute_embeddings(query_result.document)
similar_parts = similarity_model.find_similar(query_embeddings, repository)
```

### 4. **CAD-Aware LLMs**

Fine-tune LLMs on CAD data:
```python
from cadling import DocumentConverter
from cadling.chunking import SequentialChunker

# Prepare training data
training_data = []
for cad_file, description in dataset:
    result = converter.convert(cad_file)
    chunker = SequentialChunker(chunk_size=512)
    chunks = chunker.chunk(result.document)

    for chunk in chunks:
        training_data.append({
            "text": chunk.text,
            "description": description,
            "topology": chunk.metadata["topology_features"]
        })

# Fine-tune LLM
model.train(training_data)
```

### 5. **Optical CAD Recognition**

Extract dimensions and tolerances from engineering drawings:
```python
from cadling.pipeline import HybridPipeline
from cadling.models import ApiVlmModel

vlm = ApiVlmModel(provider="openai", model="gpt-4-vision-preview")
options = HybridPipelineOptions(
    vlm_model=vlm,
    views_to_render=["front", "top", "right"]
)

pipeline = HybridPipeline(options)
result = pipeline.execute(input_doc)

# Extract annotations
for item in result.document.items:
    if item.item_type == "annotation":
        print(f"{item.annotation_type}: {item.text} from {item.source_view}")
```

## Success Metrics

CADling will be considered successful when:

1. **Adoption**: Used by 100+ organizations for CAD processing
2. **Coverage**: Supports 90%+ of common CAD formats
3. **Performance**: Processes typical CAD files in <5 seconds
4. **Accuracy**: >95% accuracy on entity extraction and topology analysis
5. **Integration**: Default choice for CAD processing in 3+ major AI frameworks
6. **Community**: 1000+ GitHub stars, active contributor community
7. **Impact**: Enables 10+ research papers on CAD AI

## Contributing

CADling is open source and welcomes contributions:

- **Code**: Backends, pipelines, models, optimizations
- **Documentation**: Tutorials, examples, API docs
- **Testing**: Unit tests, integration tests, test CAD files
- **Research**: Novel algorithms, benchmarks, evaluations
- **Use cases**: Real-world applications and success stories

## Conclusion

CADling represents a fundamental shift in how we think about CAD files in the age of AI. By treating CAD as a first-class modality alongside text and images, and by providing robust, open-source tooling for CAD understanding, CADling aims to unlock the vast corpus of engineering knowledge encoded in CAD files for AI systems.

Just as Docling has become essential infrastructure for document AI, we envision CADling becoming essential infrastructure for engineering AI—enabling a new generation of AI-powered design tools, automated manufacturing systems, and intelligent engineering assistants.

---

**CADling**: Making CAD files AI-ready, one polygon at a time. 🔧🤖
