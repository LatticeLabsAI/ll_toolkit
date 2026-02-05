# CADling Unimplemented Features

This document tracks all features, components, and functionality described in `Plan.md` that have not yet been fully implemented in the CADling codebase.

**Last Updated**: 2026-01-10

---

## Table of Contents

1. [Phase 1: Core Infrastructure](#phase-1-core-infrastructure)
2. [Phase 2: STEP Backend with ll_stepnet](#phase-2-step-backend-with-ll_stepnet)
3. [Phase 3: STL and Mesh Support](#phase-3-stl-and-mesh-support)
4. [Phase 4: Vision Pipeline](#phase-4-vision-pipeline)
5. [Phase 5: Hybrid Pipeline](#phase-5-hybrid-pipeline)
6. [Phase 6: Enrichment Models](#phase-6-enrichment-models)
7. [Phase 7: Chunking System](#phase-7-chunking-system)
8. [Phase 8: Synthetic Data Generation](#phase-8-synthetic-data-generation)
9. [Phase 9: Additional Formats](#phase-9-additional-formats)
10. [Phase 10: CLI and Documentation](#phase-10-cli-and-documentation)

---

## Phase 1: Core Infrastructure

### Status: PARTIALLY IMPLEMENTED

#### 1.1 Base Data Models ✅ MOSTLY IMPLEMENTED
**File**: `cadling/datamodel/base_models.py` (486 lines)

**Implemented**:
- [x] `InputFormat` enum
- [x] `CADDocumentOrigin`
- [x] `CADItem` base class
- [x] `CADlingDocument` with basic methods
- [x] `ConversionResult`
- [x] `BoundingBox3D` model

**NOT Implemented**:
- [ ] `CADlingDocument.export_to_json()` - **CRITICAL** - Method exists but may be incomplete
- [ ] `CADlingDocument.export_to_markdown()` - **CRITICAL** - Method exists but may be incomplete
- [ ] Full test coverage for all data models

#### 1.2 Abstract Backend Interfaces ✅ MOSTLY IMPLEMENTED
**File**: `cadling/backend/abstract_backend.py` (335 lines)

**Implemented**:
- [x] `AbstractCADBackend` base class
- [x] `DeclarativeCADBackend` class
- [x] `RenderableCADBackend` class
- [x] `CADInputDocument` class

**NOT Implemented**:
- [ ] Complete backend option models
- [ ] Full test coverage for all backends

#### 1.3 Base Pipeline ✅ MOSTLY IMPLEMENTED
**File**: `cadling/pipeline/base_pipeline.py` (317 lines)

**Implemented**:
- [x] `BaseCADPipeline` class
- [x] `PipelineOptions` datamodel

**NOT Implemented**:
- [ ] Enrichment model interface integration
- [ ] Complete error handling for all edge cases
- [ ] Full test coverage

#### 1.4 Simple Pipeline ⚠️ STATUS UNKNOWN
**File**: `cadling/pipeline/simple_pipeline.py` (unknown lines)

**Status**: File exists but implementation status unknown

**NOT Implemented**:
- [ ] Verify `SimpleCADPipeline` is complete
- [ ] Full test coverage

#### 1.5 Document Converter ✅ MOSTLY IMPLEMENTED
**File**: `cadling/document_converter.py` (381 lines)

**Implemented**:
- [x] `DocumentConverter` class
- [x] Format detection logic
- [x] `FormatOption` model

**NOT Implemented**:
- [ ] **CRITICAL** - Four NotImplementedError exceptions still present in code
- [ ] Integration tests for all format conversions
- [ ] Error handling for malformed files

#### 1.6 Configuration Files ⚠️ INCOMPLETE
**Status**: Partial

**NOT Implemented**:
- [ ] Complete `pyproject.toml` with all dependencies
- [ ] `requirements.txt` with pinned versions
- [ ] `environment.yml` for conda users
- [ ] Optional dependencies properly separated (vision, dev, etc.)

---

## Phase 2: STEP Backend with ll_stepnet

### Status: NOT IMPLEMENTED

#### 2.1 STEP Data Models ⚠️ PARTIAL
**File**: `cadling/datamodel/step.py` (exists)

**NOT Implemented**:
- [ ] `STEPEntityItem` class with all properties
- [ ] `TopologyGraph` class with adjacency matrix/edge index
- [ ] `STEPDocument` class extending CADlingDocument
- [ ] `STEPHeader` model
- [ ] Full test coverage

#### 2.2 ll_stepnet Integration Layer ❌ NOT IMPLEMENTED
**File**: `cadling/backend/step/stepnet_integration.py` **MISSING**

**NOT Implemented**:
- [ ] **CRITICAL** - `STEPNetIntegration` class
- [ ] Entity extraction logic using ll_stepnet
- [ ] Feature extraction wrapper
- [ ] Topology building wrapper
- [ ] Integration with ll_stepnet tokenizer
- [ ] Integration with ll_stepnet feature extractor
- [ ] Integration with ll_stepnet topology builder
- [ ] All integration tests

#### 2.3 STEP Backend Implementation ❌ NOT IMPLEMENTED
**File**: `cadling/backend/step_backend.py` (0 lines - EMPTY FILE)

**NOT Implemented**:
- [ ] **CRITICAL** - Complete `STEPBackend` class implementation
- [ ] STEP file loading and parsing
- [ ] Entity-to-CADItem conversion
- [ ] Topology integration
- [ ] STEP header parsing
- [ ] Validation logic (`is_valid()` method)
- [ ] All unit tests
- [ ] Integration tests with sample STEP files

#### 2.4 STEP Format Option ❌ NOT IMPLEMENTED
**File**: `cadling/document_converter.py` (needs update)

**NOT Implemented**:
- [ ] `STEPFormatOption` class
- [ ] Registration in default format options
- [ ] End-to-end STEP conversion testing

---

## Phase 3: STL and Mesh Support

### Status: PARTIALLY IMPLEMENTED

#### 3.1 Mesh Data Models ⚠️ PARTIAL
**File**: `cadling/datamodel/mesh.py` (exists)

**NOT Implemented**:
- [ ] Complete `MeshItem` class with all properties
- [ ] Mesh property calculations (manifold, watertight)
- [ ] Volume and surface area calculations
- [ ] Full test coverage

#### 3.2 STL Backend ⚠️ STATUS UNKNOWN
**File**: `cadling/backend/stl_backend.py` (exists)

**NOT Implemented**:
- [ ] **Verify implementation status** - File exists but completeness unknown
- [ ] ASCII/binary STL detection
- [ ] Mesh extraction using numpy-stl or trimesh
- [ ] Mesh validation (manifold checking, watertight checking)
- [ ] Property calculations (volume, surface area)
- [ ] Unit tests
- [ ] Integration tests with sample STL files

#### 3.3 STL Format Option ❌ NOT IMPLEMENTED

**NOT Implemented**:
- [ ] `STLFormatOption` class
- [ ] Registration in DocumentConverter
- [ ] End-to-end STL conversion testing

---

## Phase 4: Vision Pipeline

### Status: NOT IMPLEMENTED

#### 4.1 Annotation Data Models ❌ NOT IMPLEMENTED
**File**: `cadling/datamodel/base_models.py` (needs update)

**NOT Implemented**:
- [ ] **CRITICAL** - `AnnotationItem` class
- [ ] Annotation types (dimension, tolerance, note, label)
- [ ] Image bounding box integration
- [ ] Source view tracking
- [ ] Confidence scores
- [ ] Unit tests

#### 4.2 VLM Model Integration ❌ NOT IMPLEMENTED
**Files**:
- `cadling/models/vlm_model.py` **MISSING**
- `cadling/models/__init__.py` **MISSING**

**NOT Implemented**:
- [ ] **CRITICAL** - `VlmModel` base class
- [ ] **CRITICAL** - `ApiVlmModel` for GPT-4V, Claude
- [ ] **CRITICAL** - `InlineVlmModel` for local models (LLaVA, Qwen-VL)
- [ ] OCR integration using EasyOCR
- [ ] Response parsing and validation
- [ ] All unit tests

#### 4.3 Rendering Support for STEP Backend ❌ NOT IMPLEMENTED
**File**: `cadling/backend/step_backend.py` (needs major update)

**NOT Implemented**:
- [ ] **CRITICAL** - pythonocc-core rendering integration
- [ ] Multiple view rendering (front, top, right, isometric, etc.)
- [ ] Camera positioning for each view
- [ ] Lighting and resolution options
- [ ] Image export functionality
- [ ] Unit tests
- [ ] Integration tests

#### 4.4 Vision Pipeline Implementation ❌ NOT IMPLEMENTED
**File**: `cadling/pipeline/vlm_pipeline.py` **MISSING**

**NOT Implemented**:
- [ ] **CRITICAL** - `CADVlmPipeline` class
- [ ] Multi-view rendering orchestration
- [ ] VLM prompting for CAD annotation extraction
- [ ] Annotation parsing from VLM responses
- [ ] `CADVlmPipelineOptions` datamodel
- [ ] All unit tests
- [ ] Integration tests with rendered images

#### 4.5 Vision Format Option ❌ NOT IMPLEMENTED

**NOT Implemented**:
- [ ] `CADImageBackend` for loading CAD images directly
- [ ] `CADImageFormatOption` class
- [ ] Testing with CAD screenshot images

---

## Phase 5: Hybrid Pipeline

### Status: NOT IMPLEMENTED

#### 5.1 Hybrid Pipeline Implementation ❌ NOT IMPLEMENTED
**File**: `cadling/pipeline/hybrid_pipeline.py` **MISSING**

**NOT Implemented**:
- [ ] **CRITICAL** - `HybridCADPipeline` class
- [ ] Text parsing + vision analysis orchestration
- [ ] Information fusion logic between text and vision
- [ ] Spatial linking of annotations to entities
- [ ] `HybridPipelineOptions` datamodel
- [ ] All unit tests
- [ ] Integration tests

---

## Phase 6: Enrichment Models

### Status: NOT IMPLEMENTED

#### 6.1 Enrichment Model Interface ❌ NOT IMPLEMENTED
**File**: `cadling/models/base_model.py` **MISSING**

**NOT Implemented**:
- [ ] **CRITICAL** - `EnrichmentModel` base class
- [ ] Interface for enrichment pipeline integration

#### 6.2 Classification Model ❌ NOT IMPLEMENTED
**File**: `cadling/models/classification.py` **MISSING**

**NOT Implemented**:
- [ ] **CRITICAL** - `CADPartClassifier` class
- [ ] ll_stepnet STEPForClassification integration
- [ ] Model loading and inference
- [ ] Batch processing support
- [ ] Unit tests

#### 6.3 Property Prediction Model ❌ NOT IMPLEMENTED
**File**: `cadling/models/property_prediction.py` **MISSING**

**NOT Implemented**:
- [ ] **CRITICAL** - `CADPropertyPredictor` class
- [ ] Volume, mass, and geometric property prediction
- [ ] ll_stepnet integration
- [ ] Unit tests

#### 6.4 Similarity Model ❌ NOT IMPLEMENTED
**File**: `cadling/models/similarity.py` **MISSING**

**NOT Implemented**:
- [ ] **CRITICAL** - `CADSimilarityEmbedder` class
- [ ] Embedding generation for RAG
- [ ] ll_stepnet STEPForSimilarity integration
- [ ] Document-level embedding aggregation
- [ ] Unit tests

#### 6.5 Pipeline Integration ❌ NOT IMPLEMENTED

**NOT Implemented**:
- [ ] Enrichment model support in pipelines
- [ ] Pipeline options for enrichment configuration
- [ ] Testing enrichment pipeline end-to-end

---

## Phase 7: Chunking System

### Status: NOT IMPLEMENTED

#### 7.1 Base Chunker ❌ NOT IMPLEMENTED
**File**: `cadling/chunker/base_chunker.py` **MISSING**

**NOT Implemented**:
- [ ] **CRITICAL** - `BaseCADChunker` abstract class
- [ ] **CRITICAL** - `CADChunk` datamodel
- [ ] **CRITICAL** - `CADChunkMeta` datamodel
- [ ] Iterator interface for chunking
- [ ] Unit tests

#### 7.2 Hybrid Chunker ❌ NOT IMPLEMENTED
**File**: `cadling/chunker/hybrid_chunker.py` **MISSING**

**NOT Implemented**:
- [ ] **CRITICAL** - `CADHybridChunker` class
- [ ] Entity-level + semantic chunking logic
- [ ] Token counting with tokenizers
- [ ] Chunk text generation from items
- [ ] Topology subgraph extraction
- [ ] Embedding aggregation
- [ ] Unit tests
- [ ] Integration tests with STEP documents

#### 7.3 Hierarchical Chunker ❌ NOT IMPLEMENTED
**File**: `cadling/chunker/hierarchical_chunker.py` **MISSING**

**NOT Implemented**:
- [ ] **CRITICAL** - `CADHierarchicalChunker` class
- [ ] Assembly hierarchy respect
- [ ] Multi-part CAD handling
- [ ] Unit tests

---

## Phase 8: Synthetic Data Generation

### Status: NOT IMPLEMENTED

#### 8.1 QA Generator ❌ NOT IMPLEMENTED
**File**: `cadling/sdg/qa_generator.py` **MISSING**

**NOT Implemented**:
- [ ] **CRITICAL** - `CADQAGenerator` class
- [ ] LLM integration (OpenAI, Anthropic)
- [ ] Question generation from chunks
- [ ] Answer generation
- [ ] Critique and improvement loop
- [ ] `QAPair` datamodel
- [ ] Unit tests
- [ ] Example Q&A pairs generation

---

## Phase 9: Additional Formats

### Status: PARTIALLY IMPLEMENTED

#### 9.1 BRep Backend ⚠️ STATUS UNKNOWN
**File**: `cadling/backend/brep_backend.py` (exists)

**NOT Implemented**:
- [ ] **Verify implementation status** - File exists but completeness unknown
- [ ] Complete `BRepBackend` class
- [ ] BRep data models (if not in datamodel/brep.py)
- [ ] pythonocc-core integration
- [ ] Unit tests

#### 9.2 IGES Backend ❌ NOT IMPLEMENTED
**File**: `cadling/backend/iges_backend.py` **MISSING**

**NOT Implemented**:
- [ ] **CRITICAL** - `IGESBackend` class
- [ ] IGES parsing and loading
- [ ] IGES data models
- [ ] Unit tests

---

## Phase 10: CLI and Documentation

### Status: NOT IMPLEMENTED

#### 10.1 CLI Implementation ❌ NOT IMPLEMENTED
**File**: `cadling/cli/main.py` **MISSING**

**NOT Implemented**:
- [ ] **CRITICAL** - CLI command structure (click or argparse)
- [ ] `convert` command for file conversion
- [ ] `chunk` command for RAG chunking
- [ ] `generate-qa` command for synthetic data
- [ ] Options and help text
- [ ] CLI tests

#### 10.2 Documentation ❌ INCOMPLETE

**NOT Implemented**:
- [ ] API reference documentation (Sphinx or similar)
- [ ] User guide with examples
- [ ] Architecture documentation
- [ ] Contribution guidelines
- [ ] Comprehensive README with usage examples
- [ ] Example Jupyter notebooks
- [ ] Tutorial videos or screencasts

---

## Critical Missing Components Summary

### Immediate Priority (P0)

1. **STEP Backend Implementation** (`cadling/backend/step_backend.py`)
   - Currently 0 lines - completely empty
   - Blocks all STEP file functionality
   - Requires ll_stepnet integration

2. **ll_stepnet Integration Layer** (`cadling/backend/step/stepnet_integration.py`)
   - Missing entirely
   - Critical for STEP parsing, tokenization, topology building
   - Core functionality dependency

3. **Document Converter NotImplementedErrors**
   - Four NotImplementedError exceptions in `document_converter.py`
   - Blocks actual file conversion

4. **Export Methods**
   - `CADlingDocument.export_to_json()`
   - `CADlingDocument.export_to_markdown()`
   - Without these, cannot output converted documents

### High Priority (P1)

5. **Enrichment Models** (entire `cadling/models/` directory missing)
   - Classification, property prediction, similarity
   - Essential for RAG and downstream tasks

6. **Chunking System** (entire `cadling/chunker/` implementation missing)
   - Required for RAG integration
   - Blocks document preparation for embeddings

7. **Vision Pipeline** (`cadling/pipeline/vlm_pipeline.py`)
   - Optical CAD recognition
   - VLM integration for annotations

8. **CLI** (`cadling/cli/main.py`)
   - User-facing tool for conversions
   - Makes toolkit usable without Python knowledge

### Medium Priority (P2)

9. **Hybrid Pipeline** (`cadling/pipeline/hybrid_pipeline.py`)
   - Advanced functionality
   - Combines text + vision

10. **Synthetic Data Generation** (`cadling/sdg/qa_generator.py`)
    - Training data generation
    - Q&A pair creation

11. **Additional Formats** (IGES backend completely missing)
    - Expand format support
    - BRep backend status unknown

---

## Testing Gaps

### Unit Tests
- Most components lack comprehensive unit tests
- Need pytest fixtures for sample data
- Target: >80% code coverage

### Integration Tests
- End-to-end conversion workflows not tested
- Need sample CAD file test suite
- Pipeline combination testing missing

### Test Data
- Need to collect diverse CAD samples:
  - Simple STEP parts (primitives)
  - Complex STEP assemblies
  - ASCII and binary STL files
  - BRep and IGES files
  - Rendered CAD images with annotations

---

## Blockers and Dependencies

### External Dependencies Not Verified
- [ ] pythonocc-core installation and setup
- [ ] ll_stepnet availability and version compatibility
- [ ] numpy-stl or trimesh for STL processing
- [ ] VLM API access (GPT-4V, Claude)
- [ ] transformers library for local VLMs
- [ ] EasyOCR for optical character recognition

### Internal Dependencies
- Phase 2 (STEP backend) is blocked until ll_stepnet integration is complete
- Phase 4 (Vision) depends on rendering support (Phase 4.3)
- Phase 5 (Hybrid) depends on both Phase 2 and Phase 4
- Phase 6 (Enrichment) depends on Phase 2
- Phase 7 (Chunking) depends on Phase 2
- Phase 8 (SDG) depends on Phase 7

---

## Configuration and Infrastructure Gaps

### Project Setup
- [ ] Complete `pyproject.toml` with all dependencies
- [ ] Lock file (requirements.txt or poetry.lock)
- [ ] Conda environment.yml
- [ ] Docker containerization for reproducibility
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Pre-commit hooks (black, ruff, mypy)

### Development Tools
- [ ] Automated testing on push
- [ ] Code coverage reporting
- [ ] Documentation generation (Sphinx)
- [ ] Release automation

---

## Estimated Implementation Effort

Based on the plan timeline and current state:

- **Phase 1**: 70% complete → ~2-3 days remaining
- **Phase 2**: 5% complete → ~2 weeks
- **Phase 3**: 30% complete → ~1 week
- **Phase 4**: 0% complete → ~2-3 weeks
- **Phase 5**: 0% complete → ~1 week
- **Phase 6**: 0% complete → ~2 weeks
- **Phase 7**: 0% complete → ~1.5 weeks
- **Phase 8**: 0% complete → ~1 week
- **Phase 9**: 20% complete → ~1 week
- **Phase 10**: 0% complete → ~1 week

**Total Estimated Remaining**: ~12-15 weeks of focused development

---

## Recommendations

### Immediate Actions
1. Implement STEP backend (`step_backend.py`)
2. Create ll_stepnet integration layer
3. Fix NotImplementedError exceptions in document converter
4. Implement export methods (JSON and Markdown)
5. Write comprehensive unit tests for Phase 1 components

### Short-Term (Next 2-4 Weeks)
1. Complete Phase 2 (STEP support)
2. Verify and complete Phase 3 (STL support)
3. Begin Phase 6 (Enrichment models) in parallel
4. Set up CI/CD pipeline

### Medium-Term (1-3 Months)
1. Implement Phase 4 (Vision pipeline)
2. Implement Phase 5 (Hybrid pipeline)
3. Complete Phase 7 (Chunking system)
4. Implement Phase 10 (CLI)

### Long-Term (3+ Months)
1. Implement Phase 8 (SDG)
2. Complete Phase 9 (Additional formats)
3. Write comprehensive documentation
4. Create example notebooks and tutorials
5. Production hardening and optimization

---

## Notes

- This document should be updated regularly as features are implemented
- Mark items with ✅ when fully complete with tests
- Mark items with ⚠️ when partially complete or status uncertain
- Mark items with ❌ when not started
- Add new items as they are discovered during implementation

**Contributors**: Add your name when you implement a feature to track ownership.

---

**End of Document**
