# CADling Implementation Plan

This document outlines the phased implementation strategy for the cadling toolkit.

---

## Table of Contents

1. [Overview](#overview)
2. [Implementation Phases](#implementation-phases)
3. [Phase 1: Core Infrastructure](#phase-1-core-infrastructure)
4. [Phase 2: STEP Backend with ll_stepnet](#phase-2-step-backend-with-ll_stepnet)
5. [Phase 3: STL and Mesh Support](#phase-3-stl-and-mesh-support)
6. [Phase 4: Vision Pipeline](#phase-4-vision-pipeline)
7. [Phase 5: Hybrid Pipeline](#phase-5-hybrid-pipeline)
8. [Phase 6: Enrichment Models](#phase-6-enrichment-models)
9. [Phase 7: Chunking System](#phase-7-chunking-system)
10. [Phase 8: Synthetic Data Generation](#phase-8-synthetic-data-generation)
11. [Phase 9: Additional Formats](#phase-9-additional-formats)
12. [Phase 10: CLI and Documentation](#phase-10-cli-and-documentation)
13. [Testing Strategy](#testing-strategy)
14. [Milestones](#milestones)

---

## Overview

The implementation follows a **bottom-up** approach:
1. Build foundational abstractions (backends, data models)
2. Implement one complete vertical slice (STEP format)
3. Expand to other formats and capabilities
4. Add advanced features (vision, chunking, SDG)

### Guiding Principles

- **Mirror docling structure**: Reuse proven patterns wherever possible
- **Incremental delivery**: Each phase produces working, testable components
- **Integration-first**: Prioritize ll_stepnet integration early
- **Test-driven**: Write tests alongside implementation
- **Documentation**: Document as we build

---

## Implementation Phases

### Phase Timeline

| Phase | Deliverable | Dependencies | Priority |
|-------|-------------|--------------|----------|
| 1 | Core infrastructure | None | P0 |
| 2 | STEP backend + ll_stepnet | Phase 1 | P0 |
| 3 | STL/mesh support | Phase 1 | P0 |
| 4 | Vision pipeline | Phase 1, 2 | P1 |
| 5 | Hybrid pipeline | Phase 2, 4 | P1 |
| 6 | Enrichment models | Phase 2 | P1 |
| 7 | Chunking system | Phase 2, 3 | P1 |
| 8 | Synthetic data generation | Phase 2, 7 | P2 |
| 9 | BRep/IGES formats | Phase 1 | P2 |
| 10 | CLI + documentation | All phases | P2 |

---

## Phase 1: Core Infrastructure

**Goal**: Establish foundational abstractions that all other components depend on.

### 1.1 Base Data Models

**File**: `cadling/datamodel/base_models.py`

Implement core data structures:

```python
# Input format enumeration
class InputFormat(Enum):
    STEP = "step"
    STL = "stl"
    BREP = "brep"
    IGES = "iges"
    CAD_IMAGE = "cad_image"

# CAD-specific document origin
class CADDocumentOrigin(BaseModel):
    filename: str
    format: InputFormat
    binary_hash: str
    mimetype: Optional[str]

# Base CAD item
class CADItem(BaseModel):
    item_type: str
    label: CADItemLabel
    text: Optional[str]
    bbox: Optional[BoundingBox3D]
    properties: Dict[str, Any] = {}
    parent: Optional[str]
    children: List[str] = []
    prov: List[ProvenanceItem] = []

# Central document structure
class CADlingDocument(BaseModel):
    name: str
    format: InputFormat
    origin: CADDocumentOrigin
    hash: str
    items: List[CADItem] = []
    topology: Optional[TopologyGraph] = None
    embeddings: Optional[torch.Tensor] = None
    processing_history: List[ProcessingStep] = []

    def add_item(self, item: CADItem):
        self.items.append(item)

    def export_to_json(self) -> Dict[str, Any]:
        # Serialize to JSON
        pass

    def export_to_markdown(self) -> str:
        # Convert to markdown representation
        pass

# Conversion result wrapper
class ConversionResult(BaseModel):
    input: CADInputDocument
    document: Optional[CADlingDocument]
    status: ConversionStatus
    errors: List[ErrorItem] = []
    pages: List[CADPage] = []  # For renderable backends
```

**Subtasks**:
- [ ] Define `InputFormat` enum
- [ ] Implement `CADDocumentOrigin`
- [ ] Implement `CADItem` base class
- [ ] Implement `CADlingDocument` with methods
- [ ] Implement `ConversionResult`
- [ ] Add `TopologyGraph` placeholder
- [ ] Add `BoundingBox3D` model
- [ ] Write unit tests

### 1.2 Abstract Backend Interfaces

**File**: `cadling/backend/abstract_backend.py`

Define backend contracts:

```python
class AbstractCADBackend(ABC):
    """Base backend for all CAD formats"""

    def __init__(self, in_doc: CADInputDocument, path_or_stream, options):
        self.file = in_doc.file
        self.path_or_stream = path_or_stream
        self.document_hash = in_doc.document_hash
        self.input_format = in_doc.format
        self.options = options

    @classmethod
    @abstractmethod
    def supports_text_parsing(cls) -> bool:
        """Whether backend can parse text representation"""
        pass

    @classmethod
    @abstractmethod
    def supports_rendering(cls) -> bool:
        """Whether backend can render to images"""
        pass

    @classmethod
    @abstractmethod
    def supported_formats(cls) -> set[InputFormat]:
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        pass


class DeclarativeCADBackend(AbstractCADBackend):
    """For formats with direct text-to-document conversion"""

    @abstractmethod
    def convert(self) -> CADlingDocument:
        """Parse and convert to CADlingDocument"""
        pass


class RenderableCADBackend(AbstractCADBackend):
    """For backends that support rendering to images"""

    @abstractmethod
    def load_view(self, view_name: str):
        """Load a specific view"""
        pass

    @abstractmethod
    def available_views(self) -> List[str]:
        """List available views (front, top, iso, etc.)"""
        pass

    @abstractmethod
    def render_view(self, view_name: str, resolution: int) -> Image:
        """Render view to image"""
        pass
```

**Subtasks**:
- [ ] Implement `AbstractCADBackend`
- [ ] Implement `DeclarativeCADBackend`
- [ ] Implement `RenderableCADBackend`
- [ ] Add `CADInputDocument` class
- [ ] Add backend option models
- [ ] Write unit tests

### 1.3 Base Pipeline

**File**: `cadling/pipeline/base_pipeline.py`

Define pipeline orchestration:

```python
class BaseCADPipeline(ABC):
    """Base pipeline for CAD conversion"""

    def __init__(self, pipeline_options: PipelineOptions):
        self.pipeline_options = pipeline_options
        self.enrichment_pipe: List[EnrichmentModel] = []

    def execute(self, in_doc: CADInputDocument) -> ConversionResult:
        conv_res = ConversionResult(input=in_doc)

        try:
            # Build: Parse and extract structure
            conv_res = self._build_document(conv_res)

            # Assemble: Combine components
            conv_res = self._assemble_document(conv_res)

            # Enrich: Apply models
            conv_res = self._enrich_document(conv_res)

            conv_res.status = self._determine_status(conv_res)
        except Exception as e:
            conv_res.status = ConversionStatus.FAILURE
            conv_res.errors.append(ErrorItem(...))

        return conv_res

    @abstractmethod
    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Backend-specific document building"""
        pass

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Default: no-op, can be overridden"""
        return conv_res

    def _enrich_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Apply enrichment models"""
        for model in self.enrichment_pipe:
            model(conv_res.document, conv_res.document.items)
        return conv_res
```

**Subtasks**:
- [ ] Implement `BaseCADPipeline`
- [ ] Implement `PipelineOptions` datamodel
- [ ] Add enrichment model interface
- [ ] Add error handling
- [ ] Write unit tests

### 1.4 Simple Pipeline

**File**: `cadling/pipeline/simple_pipeline.py`

First concrete pipeline:

```python
class SimpleCADPipeline(BaseCADPipeline):
    """Pipeline for declarative backends (text parsing)"""

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        # Backend directly converts to document
        backend = conv_res.input._backend

        if not isinstance(backend, DeclarativeCADBackend):
            raise ValueError("SimpleCADPipeline requires DeclarativeCADBackend")

        conv_res.document = backend.convert()
        return conv_res
```

**Subtasks**:
- [ ] Implement `SimpleCADPipeline`
- [ ] Write unit tests

### 1.5 Document Converter

**File**: `cadling/document_converter.py`

Main entry point:

```python
class DocumentConverter:
    """Main converter class, similar to docling DocumentConverter"""

    def __init__(
        self,
        allowed_formats: List[InputFormat] = None,
        format_options: Dict[InputFormat, FormatOption] = None
    ):
        self.allowed_formats = allowed_formats or list(InputFormat)
        self.format_options = format_options or {}

    def convert(
        self,
        source: Union[Path, str, BytesIO]
    ) -> ConversionResult:
        # Detect format
        input_doc = self._create_input_document(source)

        # Select backend and pipeline
        format_option = self._get_format_option(input_doc.format)
        backend = format_option.backend(input_doc, source, format_option.backend_options)
        pipeline = format_option.pipeline_cls(format_option.pipeline_options)

        # Set backend on input doc
        input_doc._backend = backend

        # Execute pipeline
        return pipeline.execute(input_doc)
```

**Subtasks**:
- [ ] Implement `DocumentConverter`
- [ ] Add format detection logic
- [ ] Add `FormatOption` model
- [ ] Write integration tests

### 1.6 Configuration Files

Setup project dependencies:

**File**: `pyproject.toml`

```toml
[project]
name = "cadling"
version = "0.1.0"
description = "CAD document processing toolkit inspired by docling"
dependencies = [
    "pydantic>=2.0",
    "numpy>=1.24",
    "torch>=2.0",
    "pythonocc-core>=7.7",
    "numpy-stl>=3.0",
    "trimesh>=4.0",
    "Pillow>=10.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1",
]
vision = [
    "transformers>=4.35",
    "easyocr>=1.7",
]
```

**Subtasks**:
- [ ] Write `pyproject.toml`
- [ ] Write `requirements.txt`
- [ ] Write `environment.yml` (conda)

### Phase 1 Deliverables

- [x] Base data models (`CADlingDocument`, `CADItem`, etc.)
- [x] Abstract backend interfaces
- [x] Base pipeline and simple pipeline
- [x] Document converter skeleton
- [x] Configuration files
- [x] Unit tests for all components

---

## Phase 2: STEP Backend with ll_stepnet

**Goal**: Implement complete STEP file support with ll_stepnet integration.

### 2.1 STEP Data Models

**File**: `cadling/datamodel/step.py`

```python
class STEPEntityItem(CADItem):
    """STEP entity (e.g., CARTESIAN_POINT, CIRCLE)"""

    item_type: str = "step_entity"
    entity_id: int  # e.g., #31
    entity_type: str  # e.g., "CYLINDRICAL_SURFACE"

    # Parsed parameters
    numeric_params: List[float] = []
    reference_params: List[int] = []  # Entity references

    # ll_stepnet features
    features: Optional[Dict[str, Any]] = None


class TopologyGraph(BaseModel):
    """Graph structure from ll_stepnet"""

    num_nodes: int
    num_edges: int
    adjacency_matrix: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None  # PyG format
    node_features: Optional[torch.Tensor] = None


class STEPDocument(CADlingDocument):
    """STEP-specific document"""

    format: InputFormat = InputFormat.STEP
    header: STEPHeader
    entity_index: Dict[int, STEPEntityItem] = {}
```

**Subtasks**:
- [ ] Implement `STEPEntityItem`
- [ ] Implement `TopologyGraph`
- [ ] Implement `STEPDocument`
- [ ] Add `STEPHeader` model
- [ ] Write unit tests

### 2.2 ll_stepnet Integration Layer

**File**: `cadling/backend/step/stepnet_integration.py`

Wrapper around ll_stepnet components:

```python
class STEPNetIntegration:
    """Integration layer for ll_stepnet"""

    def __init__(self):
        from ll_stepnet.tokenizer import STEPTokenizer
        from ll_stepnet.feature_extractor import STEPFeatureExtractor
        from ll_stepnet.topology_builder import STEPTopologyBuilder

        self.tokenizer = STEPTokenizer()
        self.feature_extractor = STEPFeatureExtractor()
        self.topology_builder = STEPTopologyBuilder()

    def parse_step_file(self, step_text: str) -> Tuple[List[Entity], TopologyGraph]:
        """Parse STEP file using ll_stepnet"""

        # Tokenize
        token_ids = self.tokenizer.encode(step_text)

        # Extract entities and features
        entities = self._extract_entities(step_text)

        features_list = []
        for entity in entities:
            features = self.feature_extractor.extract_geometric_features(entity)
            features_list.append(features)

        # Build topology
        topology = self.topology_builder.build_complete_topology(features_list)

        return entities, topology
```

**Subtasks**:
- [ ] Implement `STEPNetIntegration` class
- [ ] Add entity extraction logic
- [ ] Add feature extraction wrapper
- [ ] Add topology building wrapper
- [ ] Write integration tests

### 2.3 STEP Backend Implementation

**File**: `cadling/backend/step_backend.py`

```python
class STEPBackend(DeclarativeCADBackend):
    """STEP file backend with ll_stepnet integration"""

    def __init__(self, in_doc, path_or_stream, options):
        super().__init__(in_doc, path_or_stream, options)

        # Load ll_stepnet components
        self.stepnet = STEPNetIntegration()

        # Parse STEP file
        self.step_text = self._load_step_text()
        self.entities, self.topology = self.stepnet.parse_step_file(self.step_text)

    @classmethod
    def supports_text_parsing(cls) -> bool:
        return True

    @classmethod
    def supports_rendering(cls) -> bool:
        return False  # Add in Phase 4

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.STEP}

    def is_valid(self) -> bool:
        return self.step_text.startswith("ISO-10303-21")

    def convert(self) -> CADlingDocument:
        """Convert STEP to CADlingDocument"""

        doc = STEPDocument(
            name=self.file.name,
            origin=CADDocumentOrigin(
                filename=self.file.name,
                format=InputFormat.STEP,
                binary_hash=self.document_hash
            ),
            hash=self.document_hash
        )

        # Convert entities to items
        for entity in self.entities:
            item = STEPEntityItem(
                entity_id=entity.id,
                entity_type=entity.type,
                numeric_params=entity.numeric_params,
                reference_params=entity.reference_params,
                features=entity.features,
                text=str(entity)
            )
            doc.add_item(item)
            doc.entity_index[entity.id] = item

        # Add topology
        doc.topology = self.topology

        return doc
```

**Subtasks**:
- [ ] Implement `STEPBackend` class
- [ ] Add STEP file loading
- [ ] Add entity conversion
- [ ] Add topology integration
- [ ] Add STEP header parsing
- [ ] Write unit tests
- [ ] Write integration tests with sample STEP files

### 2.4 STEP Format Option

**File**: `cadling/document_converter.py` (update)

```python
class STEPFormatOption(FormatOption):
    pipeline_cls: Type = SimpleCADPipeline
    backend: Type[AbstractCADBackend] = STEPBackend
    backend_options: Optional[STEPBackendOptions] = None
```

**Subtasks**:
- [ ] Add `STEPFormatOption`
- [ ] Add to default format options
- [ ] Test end-to-end conversion

### Phase 2 Deliverables

- [x] STEP data models
- [x] ll_stepnet integration layer
- [x] STEP backend implementation
- [x] End-to-end STEP conversion working
- [x] Unit and integration tests
- [x] Sample STEP files for testing

---

## Phase 3: STL and Mesh Support

**Goal**: Add STL file support with mesh processing.

### 3.1 Mesh Data Models

**File**: `cadling/datamodel/mesh.py`

```python
class MeshItem(CADItem):
    """Mesh data (STL, OBJ)"""

    item_type: str = "mesh"

    vertices: List[List[float]]  # [[x,y,z], ...]
    normals: List[List[float]]
    facets: List[List[int]]  # [[v1,v2,v3], ...]

    num_vertices: int
    num_facets: int
    is_manifold: bool
    is_watertight: bool


class STLDocument(CADlingDocument):
    """STL-specific document"""

    format: InputFormat = InputFormat.STL
    is_ascii: bool
    mesh: MeshItem
```

**Subtasks**:
- [ ] Implement `MeshItem`
- [ ] Implement `STLDocument`
- [ ] Add mesh property calculations
- [ ] Write unit tests

### 3.2 STL Backend

**File**: `cadling/backend/stl_backend.py`

```python
class STLBackend(DeclarativeCADBackend):
    """STL file backend"""

    def __init__(self, in_doc, path_or_stream, options):
        super().__init__(in_doc, path_or_stream, options)

        # Parse STL (using numpy-stl or trimesh)
        import numpy as np
        from stl import mesh as stl_mesh

        self.mesh_data = stl_mesh.Mesh.from_file(str(path_or_stream))

    def convert(self) -> CADlingDocument:
        """Convert STL to CADlingDocument"""

        # Extract vertices, normals, facets
        vertices = self._extract_vertices()
        normals = self.mesh_data.normals.tolist()
        facets = self._extract_facets()

        # Create mesh item
        mesh_item = MeshItem(
            vertices=vertices,
            normals=normals,
            facets=facets,
            num_vertices=len(vertices),
            num_facets=len(facets),
            is_manifold=self._check_manifold(),
            is_watertight=self._check_watertight(),
            properties={
                "volume": self.mesh_data.get_mass_properties()[0],
                "surface_area": self._compute_surface_area()
            }
        )

        doc = STLDocument(
            name=self.file.name,
            is_ascii=self._is_ascii(),
            mesh=mesh_item
        )
        doc.add_item(mesh_item)

        return doc
```

**Subtasks**:
- [ ] Implement `STLBackend`
- [ ] Add ASCII/binary detection
- [ ] Add mesh extraction
- [ ] Add mesh validation (manifold, watertight)
- [ ] Add property calculations
- [ ] Write unit tests
- [ ] Test with sample STL files

### 3.3 STL Format Option

Add to document converter.

**Subtasks**:
- [ ] Add `STLFormatOption`
- [ ] Test end-to-end STL conversion

### Phase 3 Deliverables

- [x] Mesh data models
- [x] STL backend implementation
- [x] End-to-end STL conversion working
- [x] Unit and integration tests
- [x] Sample STL files for testing

---

## Phase 4: Vision Pipeline

**Goal**: Enable optical CAD recognition from rendered images.

### 4.1 Annotation Data Models

**File**: `cadling/datamodel/base_models.py` (update)

```python
class AnnotationItem(CADItem):
    """Dimension, tolerance, note from rendered CAD"""

    item_type: str = "annotation"
    annotation_type: str  # "dimension", "tolerance", "note", "label"
    value: Optional[str]

    # Image provenance
    image_bbox: Optional[BoundingBox] = None  # 2D bbox
    source_view: Optional[str] = None  # "front", "top", etc.
    confidence: Optional[float] = None
```

**Subtasks**:
- [ ] Implement `AnnotationItem`
- [ ] Write unit tests

### 4.2 VLM Model Integration

**File**: `cadling/models/vlm_model.py`

```python
class VlmModel(ABC):
    """Base class for vision-language models"""

    @abstractmethod
    def predict(self, image: Image, prompt: str) -> VlmResponse:
        pass


class ApiVlmModel(VlmModel):
    """API-based VLM (GPT-4V, Claude)"""

    def __init__(self, options: ApiVlmOptions):
        self.api_key = options.api_key
        self.model_name = options.model_name
        self.client = self._initialize_client()

    def predict(self, image: Image, prompt: str) -> VlmResponse:
        # Call API
        pass


class InlineVlmModel(VlmModel):
    """Local VLM (e.g., LLaVA, Qwen-VL)"""

    def __init__(self, options: InlineVlmOptions):
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.processor = AutoProcessor.from_pretrained(options.model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(options.model_path)

    def predict(self, image: Image, prompt: str) -> VlmResponse:
        # Run inference
        pass
```

**Subtasks**:
- [ ] Implement `VlmModel` interface
- [ ] Implement `ApiVlmModel` (GPT-4V, Claude)
- [ ] Implement `InlineVlmModel` (transformers)
- [ ] Add OCR integration (EasyOCR)
- [ ] Write unit tests

### 4.3 Rendering Support for STEP Backend

**File**: `cadling/backend/step_backend.py` (update)

Make `STEPBackend` inherit from `RenderableCADBackend`:

```python
class STEPBackend(DeclarativeCADBackend, RenderableCADBackend):
    """Hybrid backend: text parsing + rendering"""

    def __init__(self, in_doc, path_or_stream, options):
        super().__init__(in_doc, path_or_stream, options)

        # ... existing code ...

        # Load 3D shape for rendering (if enabled)
        if options.enable_rendering:
            from OCC.Core.STEPControl import STEPControl_Reader

            reader = STEPControl_Reader()
            reader.ReadFile(str(path_or_stream))
            reader.TransferRoots()
            self.shape = reader.OneShape()

    @classmethod
    def supports_rendering(cls) -> bool:
        return True

    def available_views(self) -> List[str]:
        return ["front", "top", "right", "isometric", "bottom", "left", "back"]

    def render_view(self, view_name: str, resolution: int = 1024) -> Image:
        """Render using pythonocc-core"""
        from OCC.Display.SimpleGui import init_display
        from OCC.Display.OCCViewer import Viewer3d

        # Setup view
        viewer = Viewer3d()
        viewer.Create()
        viewer.DisplayShape(self.shape, update=True)

        # Set camera based on view_name
        if view_name == "front":
            viewer.View.SetProj(0, 1, 0)
        elif view_name == "top":
            viewer.View.SetProj(0, 0, 1)
        # ... other views ...

        # Render to image
        viewer.FitAll()
        image_bytes = viewer.GetImageData(resolution, resolution)
        image = Image.frombytes('RGB', (resolution, resolution), image_bytes)

        return image
```

**Subtasks**:
- [ ] Add pythonocc-core rendering
- [ ] Implement view camera positioning
- [ ] Add rendering options (resolution, lighting, etc.)
- [ ] Write unit tests
- [ ] Test rendering with sample STEP files

### 4.4 Vision Pipeline Implementation

**File**: `cadling/pipeline/vlm_pipeline.py`

```python
class CADVlmPipeline(BaseCADPipeline):
    """Vision pipeline for optical CAD recognition"""

    def __init__(self, options: CADVlmPipelineOptions):
        super().__init__(options)

        # Initialize vision model
        self.vlm_model = self._get_vlm_model(options.vlm_options)

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        backend = conv_res.input._backend

        if not isinstance(backend, RenderableCADBackend):
            raise ValueError("CADVlmPipeline requires RenderableCADBackend")

        doc = CADlingDocument(name=backend.file.name)

        # Render multiple views
        for view_name in backend.available_views():
            view_image = backend.render_view(view_name, resolution=1024)

            # Extract annotations using VLM
            annotations = self._extract_annotations_from_image(
                view_image, view_name
            )

            for ann in annotations:
                doc.add_item(ann)

        conv_res.document = doc
        return conv_res

    def _extract_annotations_from_image(
        self, image: Image, view_name: str
    ) -> List[AnnotationItem]:
        """Extract dimensions, tolerances using VLM + OCR"""

        # Prompt for VLM
        prompt = """
        Analyze this CAD technical drawing and extract:
        1. Dimensions with values and units
        2. Tolerances
        3. Notes and labels

        Return as structured JSON.
        """

        # Run VLM
        response = self.vlm_model.predict(image, prompt)

        # Parse response to AnnotationItem objects
        annotations = []
        for item in response.annotations:
            ann = AnnotationItem(
                annotation_type=item.type,
                value=item.value,
                image_bbox=item.bbox,
                source_view=view_name,
                confidence=item.confidence
            )
            annotations.append(ann)

        return annotations
```

**Subtasks**:
- [ ] Implement `CADVlmPipeline`
- [ ] Add VLM prompting logic
- [ ] Add annotation parsing
- [ ] Add `CADVlmPipelineOptions`
- [ ] Write unit tests
- [ ] Test with rendered CAD images

### 4.5 Vision Format Option

Add support for CAD images as input:

```python
class CADImageFormatOption(FormatOption):
    pipeline_cls: Type = CADVlmPipeline
    backend: Type[AbstractCADBackend] = CADImageBackend
```

**Subtasks**:
- [ ] Implement `CADImageBackend` (loads image directly)
- [ ] Add `CADImageFormatOption`
- [ ] Test with CAD screenshot images

### Phase 4 Deliverables

- [x] Annotation data models
- [x] VLM integration (API + local)
- [x] pythonocc-core rendering support
- [x] Vision pipeline implementation
- [x] End-to-end optical CAD recognition working
- [x] Unit and integration tests

---

## Phase 5: Hybrid Pipeline

**Goal**: Combine text parsing + vision analysis for comprehensive understanding.

### 5.1 Hybrid Pipeline Implementation

**File**: `cadling/pipeline/hybrid_pipeline.py`

```python
class HybridCADPipeline(BaseCADPipeline):
    """Combines text parsing + vision analysis"""

    def __init__(self, options: HybridPipelineOptions):
        super().__init__(options)

        self.text_enabled = options.enable_text_parsing
        self.vision_enabled = options.enable_vision
        self.vlm_model = self._get_vlm_model(options.vlm_options) if self.vision_enabled else None

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        backend = conv_res.input._backend

        # Phase 1: Text parsing
        if self.text_enabled and isinstance(backend, DeclarativeCADBackend):
            doc = backend.convert()
        else:
            doc = CADlingDocument(name=backend.file.name)

        # Phase 2: Vision analysis
        if self.vision_enabled and isinstance(backend, RenderableCADBackend):
            for view_name in backend.available_views():
                view_image = backend.render_view(view_name)
                annotations = self._extract_annotations_from_image(view_image, view_name)

                for ann in annotations:
                    doc.add_item(ann)

        # Phase 3: Fusion (optional)
        if self.text_enabled and self.vision_enabled:
            doc = self._fuse_information(doc)

        conv_res.document = doc
        return conv_res

    def _fuse_information(self, doc: CADlingDocument) -> CADlingDocument:
        """Fuse text-extracted and vision-extracted information"""

        # Example: Link annotations to STEP entities based on spatial proximity
        # This is domain-specific logic

        return doc
```

**Subtasks**:
- [ ] Implement `HybridCADPipeline`
- [ ] Add information fusion logic
- [ ] Add `HybridPipelineOptions`
- [ ] Write unit tests
- [ ] Test with STEP files (text + rendering)

### Phase 5 Deliverables

- [x] Hybrid pipeline implementation
- [x] Information fusion logic
- [x] End-to-end hybrid conversion working
- [x] Unit and integration tests

---

## Phase 6: Enrichment Models

**Goal**: Add ll_stepnet task models for enrichment.

### 6.1 Enrichment Model Interface

**File**: `cadling/models/base_model.py`

```python
class EnrichmentModel(ABC):
    """Base class for enrichment models"""

    @abstractmethod
    def __call__(self, doc: CADlingDocument, item_batch: List[CADItem]):
        """Enrich items with predictions"""
        pass
```

### 6.2 Classification Model

**File**: `cadling/models/classification.py`

```python
class CADPartClassifier(EnrichmentModel):
    """Classify CAD parts using ll_stepnet"""

    def __init__(self, artifacts_path: Path):
        from ll_stepnet.models import STEPForClassification

        self.model = STEPForClassification.from_pretrained(
            artifacts_path / "step_classifier.pt"
        )
        self.stepnet = STEPNetIntegration()

    def __call__(self, doc: CADlingDocument, item_batch: List[CADItem]):
        for item in item_batch:
            if isinstance(item, STEPEntityItem):
                # Prepare input
                token_ids = self.stepnet.tokenizer.encode(item.text)
                topology = doc.topology

                # Run inference
                logits = self.model(token_ids, topology_data=topology)
                predicted_class = torch.argmax(logits, dim=1)

                # Add to item
                item.properties["predicted_class"] = predicted_class.item()
                item.properties["class_confidence"] = torch.softmax(logits, dim=1).max().item()
```

**Subtasks**:
- [ ] Implement `CADPartClassifier`
- [ ] Add model loading logic
- [ ] Write unit tests

### 6.3 Property Prediction Model

**File**: `cadling/models/property_prediction.py`

Similar to classification, predict volume, mass, etc.

**Subtasks**:
- [ ] Implement `CADPropertyPredictor`
- [ ] Write unit tests

### 6.4 Similarity Model

**File**: `cadling/models/similarity.py`

```python
class CADSimilarityEmbedder(EnrichmentModel):
    """Generate embeddings for RAG"""

    def __init__(self, artifacts_path: Path):
        from ll_stepnet.models import STEPForSimilarity

        self.model = STEPForSimilarity.from_pretrained(
            artifacts_path / "similarity_model.pt"
        )
        self.stepnet = STEPNetIntegration()

    def __call__(self, doc: CADlingDocument, item_batch: List[CADItem]):
        # Generate embeddings for each item
        embeddings = []

        for item in item_batch:
            if isinstance(item, STEPEntityItem):
                token_ids = self.stepnet.tokenizer.encode(item.text)
                embedding = self.model(token_ids, topology_data=doc.topology)
                embeddings.append(embedding)
                item.properties["embedding"] = embedding.cpu().numpy()

        # Store document-level embeddings
        if embeddings:
            doc.embeddings = torch.stack(embeddings)
```

**Subtasks**:
- [ ] Implement `CADSimilarityEmbedder`
- [ ] Write unit tests

### 6.5 Pipeline Integration

Update pipelines to support enrichment:

```python
# In DocumentConverter
converter = DocumentConverter(
    format_options={
        InputFormat.STEP: STEPFormatOption(
            pipeline_options=PipelineOptions(
                enrichment_models=[
                    CADPartClassifier(artifacts_path),
                    CADPropertyPredictor(artifacts_path),
                    CADSimilarityEmbedder(artifacts_path)
                ]
            )
        )
    }
)
```

**Subtasks**:
- [ ] Add enrichment support to pipelines
- [ ] Test enrichment pipeline

### Phase 6 Deliverables

- [x] Enrichment model interface
- [x] Classification model
- [x] Property prediction model
- [x] Similarity/embedding model
- [x] Pipeline integration
- [x] Unit and integration tests

---

## Phase 7: Chunking System

**Goal**: Semantic chunking for RAG systems.

### 7.1 Base Chunker

**File**: `cadling/chunker/base_chunker.py`

```python
class BaseCADChunker(ABC):
    """Base chunker for CAD documents"""

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens

    @abstractmethod
    def chunk(self, doc: CADlingDocument) -> Iterator[CADChunk]:
        """Generate chunks from document"""
        pass


class CADChunk(BaseModel):
    """Single chunk for RAG"""

    text: str
    meta: CADChunkMeta


class CADChunkMeta(BaseModel):
    """Metadata for chunk"""

    entity_types: List[str]
    entity_ids: List[int]
    topology_subgraph: Optional[TopologyGraph]
    embedding: Optional[np.ndarray]
    properties: Dict[str, Any]
```

**Subtasks**:
- [ ] Implement `BaseCADChunker`
- [ ] Implement `CADChunk` and `CADChunkMeta`
- [ ] Write unit tests

### 7.2 Hybrid Chunker

**File**: `cadling/chunker/hybrid_chunker.py`

```python
class CADHybridChunker(BaseCADChunker):
    """Hybrid chunker combining entity-level and semantic chunking"""

    def __init__(self, tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2", max_tokens: int = 512):
        super().__init__(max_tokens)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def chunk(self, doc: CADlingDocument) -> Iterator[CADChunk]:
        """Generate chunks"""

        current_chunk_items = []
        current_tokens = 0

        for item in doc.items:
            # Count tokens
            item_text = self._item_to_text(item)
            item_tokens = len(self.tokenizer.encode(item_text))

            if current_tokens + item_tokens > self.max_tokens and current_chunk_items:
                # Yield current chunk
                yield self._create_chunk(current_chunk_items, doc)
                current_chunk_items = []
                current_tokens = 0

            current_chunk_items.append(item)
            current_tokens += item_tokens

        # Yield final chunk
        if current_chunk_items:
            yield self._create_chunk(current_chunk_items, doc)

    def _create_chunk(self, items: List[CADItem], doc: CADlingDocument) -> CADChunk:
        """Create chunk from items"""

        # Combine text
        text = "\n".join(self._item_to_text(item) for item in items)

        # Extract metadata
        entity_types = [item.entity_type for item in items if isinstance(item, STEPEntityItem)]
        entity_ids = [item.entity_id for item in items if isinstance(item, STEPEntityItem)]

        # Extract topology subgraph
        topology_subgraph = self._extract_topology_subgraph(items, doc.topology)

        # Average embeddings
        embeddings = [item.properties.get("embedding") for item in items if "embedding" in item.properties]
        avg_embedding = np.mean(embeddings, axis=0) if embeddings else None

        meta = CADChunkMeta(
            entity_types=entity_types,
            entity_ids=entity_ids,
            topology_subgraph=topology_subgraph,
            embedding=avg_embedding,
            properties={}
        )

        return CADChunk(text=text, meta=meta)
```

**Subtasks**:
- [ ] Implement `CADHybridChunker`
- [ ] Add text generation from items
- [ ] Add topology subgraph extraction
- [ ] Write unit tests
- [ ] Test with sample STEP documents

### 7.3 Hierarchical Chunker

**File**: `cadling/chunker/hierarchical_chunker.py`

Respects assembly hierarchy for multi-part CAD.

**Subtasks**:
- [ ] Implement `CADHierarchicalChunker`
- [ ] Write unit tests

### Phase 7 Deliverables

- [x] Base chunker interface
- [x] Hybrid chunker implementation
- [x] Hierarchical chunker implementation
- [x] Chunk metadata and text generation
- [x] Unit and integration tests

---

## Phase 8: Synthetic Data Generation

**Goal**: Generate Q&A pairs for training.

### 8.1 QA Generator

**File**: `cadling/sdg/qa_generator.py`

```python
class CADQAGenerator:
    """Generate Q&A pairs from CAD documents"""

    def __init__(
        self,
        llm_model: str = "gpt-4",
        critique_enabled: bool = True
    ):
        self.llm_model = llm_model
        self.critique_enabled = critique_enabled
        self.client = self._initialize_llm()

    def generate_qa_pairs(
        self,
        doc: CADlingDocument,
        num_pairs: int = 100
    ) -> List[QAPair]:
        """Generate Q&A pairs"""

        qa_pairs = []

        # Sample entities/chunks
        chunks = list(CADHybridChunker().chunk(doc))

        for chunk in chunks[:num_pairs]:
            # Generate question
            question = self._generate_question(chunk)

            # Generate answer
            answer = self._generate_answer(chunk, question)

            # Critique (optional)
            if self.critique_enabled:
                question, answer = self._critique_qa(chunk, question, answer)

            qa_pairs.append(QAPair(
                question=question,
                answer=answer,
                context=chunk.text,
                metadata=chunk.meta
            ))

        return qa_pairs

    def _generate_question(self, chunk: CADChunk) -> str:
        """Generate question from chunk"""

        prompt = f"""
        Given this CAD entity information:
        {chunk.text}

        Generate a specific question about geometric properties, dimensions, or relationships.
        """

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    def _generate_answer(self, chunk: CADChunk, question: str) -> str:
        """Generate answer"""
        # Similar prompting
        pass

    def _critique_qa(self, chunk: CADChunk, question: str, answer: str) -> Tuple[str, str]:
        """Critique and improve Q&A pair"""
        # Similar to docling-sdg critique step
        pass
```

**Subtasks**:
- [ ] Implement `CADQAGenerator`
- [ ] Add question generation prompts
- [ ] Add answer generation prompts
- [ ] Add critique logic
- [ ] Write unit tests
- [ ] Test with sample documents

### Phase 8 Deliverables

- [x] QA generator implementation
- [x] Sample → Generate → Critique pipeline
- [x] Integration with chunking
- [x] Unit and integration tests
- [x] Example generated Q&A pairs

---

## Phase 9: Additional Formats

**Goal**: Add BRep and IGES support.

### 9.1 BRep Backend

**File**: `cadling/backend/brep_backend.py`

Similar to STEP backend, uses pythonocc-core.

**Subtasks**:
- [ ] Implement `BRepBackend`
- [ ] Add BRep data models
- [ ] Write unit tests

### 9.2 IGES Backend

**File**: `cadling/backend/iges_backend.py`

**Subtasks**:
- [ ] Implement `IGESBackend`
- [ ] Add IGES data models
- [ ] Write unit tests

### Phase 9 Deliverables

- [x] BRep backend and data models
- [x] IGES backend and data models
- [x] Unit and integration tests

---

## Phase 10: CLI and Documentation

**Goal**: Polish the toolkit with CLI and comprehensive docs.

### 10.1 CLI Implementation

**File**: `cadling/cli/main.py`

```python
import click
from cadling import DocumentConverter

@click.group()
def cli():
    """CADling CLI"""
    pass

@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path())
@click.option("--format", "-f", type=click.Choice(["json", "markdown"]))
def convert(input_file, output, format):
    """Convert CAD file"""

    converter = DocumentConverter()
    result = converter.convert(input_file)

    if result.status == ConversionStatus.SUCCESS:
        if format == "json":
            output_data = result.document.export_to_json()
        else:
            output_data = result.document.export_to_markdown()

        if output:
            with open(output, "w") as f:
                f.write(output_data)
        else:
            click.echo(output_data)
    else:
        click.echo(f"Conversion failed: {result.errors}")

@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--max-tokens", default=512)
def chunk(input_file, max_tokens):
    """Chunk CAD file for RAG"""

    converter = DocumentConverter()
    result = converter.convert(input_file)

    from cadling.chunker import CADHybridChunker
    chunker = CADHybridChunker(max_tokens=max_tokens)

    for i, chunk in enumerate(chunker.chunk(result.document)):
        click.echo(f"=== Chunk {i} ===")
        click.echo(chunk.text)
        click.echo()
```

**Subtasks**:
- [ ] Implement CLI commands (convert, chunk, generate-qa)
- [ ] Add options and help text
- [ ] Write CLI tests

### 10.2 Documentation

- [ ] API reference (auto-generated from docstrings)
- [ ] User guide with examples
- [ ] Architecture documentation
- [ ] Contribution guidelines
- [ ] README updates

### Phase 10 Deliverables

- [x] CLI implementation
- [x] Comprehensive documentation
- [x] Example notebooks
- [x] README with usage examples

---

## Testing Strategy

### Unit Tests

- Test each component in isolation
- Use pytest fixtures for sample data
- Aim for >80% code coverage

### Integration Tests

- Test end-to-end conversion workflows
- Test with real CAD files (STEP, STL, etc.)
- Test pipeline combinations

### Test Data

- Collect sample CAD files:
  - Simple STEP parts (cube, cylinder, sphere)
  - Complex STEP assemblies
  - ASCII and binary STL files
  - BRep and IGES files
  - Rendered CAD images with annotations

### Continuous Integration

- GitHub Actions for automated testing
- Test across Python 3.9, 3.10, 3.11
- Code quality checks (black, ruff, mypy)

---

## Milestones

### M1: Foundational Infrastructure (Phases 1-2)
- Target: Week 4
- Deliverable: STEP file conversion working end-to-end
- Success criteria: Convert STEP → CADlingDocument → JSON

### M2: Multi-Format Support (Phase 3)
- Target: Week 6
- Deliverable: STEP + STL conversion working
- Success criteria: Handle both text CAD and mesh formats

### M3: Vision Capabilities (Phases 4-5)
- Target: Week 10
- Deliverable: Optical CAD recognition working
- Success criteria: Extract annotations from rendered CAD images

### M4: Enrichment & RAG (Phases 6-7)
- Target: Week 14
- Deliverable: Full enrichment pipeline + chunking
- Success criteria: Generate embeddings and chunks for RAG

### M5: Complete Toolkit (Phases 8-10)
- Target: Week 18
- Deliverable: All features, CLI, documentation
- Success criteria: Production-ready toolkit

---

## Next Steps

1. Review this plan with stakeholders
2. Set up development environment
3. Begin Phase 1 implementation
4. Establish CI/CD pipeline
5. Start collecting test CAD files

See [Development.md](Development.md) for coding guidelines and best practices.
