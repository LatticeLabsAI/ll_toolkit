# CADling Experimental Features

Experimental features for advanced CAD processing, inspired by [Docling's](https://github.com/DS4SD/docling) experimental architecture. These features leverage CADling's existing infrastructure (backends, models, pipelines) to add sophisticated capabilities for 3D CAD processing.

## ⚠️ Stability Warning

**These features are experimental and may change without notice.**
- APIs may be modified or removed in future versions
- Performance characteristics may vary
- Not recommended for production use without thorough testing
- Feedback and contributions are welcome!

## 📋 Feature Overview

### Datamodels & Options (Features 10-12)
Configuration options for experimental processing:
- **CADAnnotationOptions**: PMI and annotation extraction configuration
- **MultiViewOptions**: Multi-view rendering and fusion settings
- **AssemblyAnalysisOptions**: Assembly processing parameters

### Vision-Based Models (Features 4-6)
VLM-powered enrichment models:
- **PMIExtractionModel**: Extract dimensions, tolerances, GD&T symbols
- **FeatureRecognitionVlmModel**: Identify machining features (holes, pockets, etc.)
- **ManufacturabilityAssessmentModel**: DFM analysis and cost estimation

### AI Understanding Models (Features 7-9)
AI-powered semantic analysis:
- **DesignIntentInferenceModel**: Infer functional purpose and design intent
- **CADToTextGenerationModel**: Generate natural language descriptions
- **GeometricConstraintModel**: Extract implicit geometric constraints

### Advanced Pipelines (Features 1-3)
Experimental processing pipelines:
- **ThreadedGeometryVlmPipeline**: Two-stage geometry + VLM analysis
- **MultiViewFusionPipeline**: Multi-view rendering with result fusion
- **AssemblyHierarchyPipeline**: Assembly-aware processing with BOM generation

## 🚀 Quick Start

### Basic Usage

```python
from cadling.experimental import (
    CADAnnotationOptions,
    ThreadedGeometryVlmPipeline,
)

# Configure options
options = CADAnnotationOptions(
    vlm_model="gpt-4-vision",
    annotation_types=["dimension", "tolerance", "gdt"],
    views_to_process=["front", "top", "isometric"],
    min_confidence=0.7,
)

# Create pipeline
pipeline = ThreadedGeometryVlmPipeline(options)

# Execute on CAD file
from cadling.datamodel import CADInputDocument
input_doc = CADInputDocument(file="path/to/model.step")
result = pipeline.execute(input_doc)

# Access results
for item in result.document.items:
    pmi = item.properties.get("pmi_annotations", [])
    for annotation in pmi:
        print(f"{annotation['type']}: {annotation['text']}")
```

## 📖 Detailed Feature Documentation

### Feature 1: ThreadedGeometryVlmPipeline

Two-stage pipeline combining geometric analysis with VLM processing.

**Stage 1: Geometric Analysis**
- Parse CAD file (STEP/STL/BRep)
- Extract topology and features
- Build BRep graph structure
- Render views for VLM

**Stage 2: VLM with Context**
- Inject geometric features into VLM prompts
- Extract PMI and annotations
- Cross-validate across views

```python
from cadling.experimental import ThreadedGeometryVlmPipeline, CADAnnotationOptions

options = CADAnnotationOptions(
    vlm_model="gpt-4-vision",
    include_geometric_context=True,
)
pipeline = ThreadedGeometryVlmPipeline(options)
result = pipeline.execute(input_doc)
```

### Feature 2: MultiViewFusionPipeline

Renders multiple views and fuses VLM results using consensus algorithms.

**Fusion Strategies:**
- `weighted_consensus`: Weight by confidence, require majority
- `majority_vote`: Simple majority across views
- `hierarchical`: Priority-based (front/top > others)

```python
from cadling.experimental import MultiViewFusionPipeline, MultiViewOptions, ViewConfig

options = MultiViewOptions(
    views=[
        ViewConfig(name="front", azimuth=0, elevation=0),
        ViewConfig(name="top", azimuth=0, elevation=90),
        ViewConfig(name="right", azimuth=90, elevation=0),
        ViewConfig(name="isometric", azimuth=45, elevation=35.264),
    ],
    fusion_strategy="weighted_consensus",
    resolution=2048,
)
pipeline = MultiViewFusionPipeline(options)
result = pipeline.execute(input_doc)

# Access fused features
fused = result.document.items[0].properties.get("fused_machining_features")
```

### Feature 3: AssemblyHierarchyPipeline

Processes multi-part assemblies with hierarchy awareness and BOM generation.

**Capabilities:**
- Component detection
- Mate relationship extraction
- Hierarchical processing
- Bill of Materials generation
- Interference checking (optional)

```python
from cadling.experimental import AssemblyHierarchyPipeline, AssemblyAnalysisOptions

options = AssemblyAnalysisOptions(
    detect_components=True,
    extract_mates=True,
    generate_bom=True,
    group_identical_parts=True,
)
pipeline = AssemblyHierarchyPipeline(options)
result = pipeline.execute(input_doc)

# Access BOM
bom = result.document.properties.get("bill_of_materials")
for entry in bom:
    print(f"{entry['name']}: qty {entry['quantity']}")

# Access assembly tree
tree = result.document.properties.get("assembly_tree")
```

### Feature 4: PMIExtractionModel

Extracts Product Manufacturing Information using VLM.

**Extracted Information:**
- Dimensions (linear, angular, radius, diameter)
- Tolerances (±, limit, fit callouts)
- GD&T symbols and feature control frames
- Surface finish specifications
- Material callouts
- Welding symbols

```python
from cadling.experimental import PMIExtractionModel, CADAnnotationOptions

options = CADAnnotationOptions(
    annotation_types=["dimension", "tolerance", "gdt", "surface_finish"],
    vlm_model="gpt-4-vision",
    min_confidence=0.8,
)
model = PMIExtractionModel(options)

# Use in pipeline or standalone
model(doc, doc.items)

# Access extracted PMI
for item in doc.items:
    annotations = item.properties.get("pmi_annotations", [])
    for ann in annotations:
        print(f"{ann['type']}: {ann['text']} (confidence: {ann['confidence']:.2f})")
```

### Feature 5: FeatureRecognitionVlmModel

Identifies machining features using VLM analysis.

**Recognized Features:**
- **Holes**: through, blind, counterbore, countersunk, threaded
- **Pockets**: rectangular, circular, blind, through
- **Slots**: through, blind, T-slots
- **Edge Features**: fillets, chamfers, rounds
- **Protrusions**: bosses, ribs, lugs

```python
from cadling.experimental import FeatureRecognitionVlmModel, CADAnnotationOptions

options = CADAnnotationOptions(
    annotation_types=["hole", "pocket", "fillet", "chamfer"],
    vlm_model="gpt-4-vision",
    enable_cross_view_validation=True,
)
model = FeatureRecognitionVlmModel(options)
model(doc, doc.items)

# Access detected features
for item in doc.items:
    features = item.properties.get("machining_features", [])
    for feature in features:
        print(f"{feature['feature_type']}: {feature['parameters']}")
```

### Feature 6: ManufacturabilityAssessmentModel

Assesses manufacturability using DFM rules and VLM analysis.

**Assessment Criteria:**
- Thin wall detection
- Deep pocket analysis
- Sharp internal corner identification
- Small hole detection
- Tool access evaluation
- Setup complexity estimation

```python
from cadling.experimental import ManufacturabilityAssessmentModel, CADAnnotationOptions

options = CADAnnotationOptions(vlm_model="gpt-4-vision")
model = ManufacturabilityAssessmentModel(options)
model(doc, doc.items)

# Access manufacturability report
for item in doc.items:
    report = item.properties.get("manufacturability_report")
    print(f"Score: {report['overall_score']}/100")
    print(f"Difficulty: {report['estimated_difficulty']}")
    for issue in report['issues']:
        print(f"- {issue['description']}")
```

### Feature 7: DesignIntentInferenceModel

Infers design intent and functional purpose from geometry and visual analysis.

**Intent Categories:**
- Structural (load-bearing)
- Mounting (attachment)
- Alignment (positioning)
- Sealing (containment)
- Thermal (heat management)
- Motion (articulation)

```python
from cadling.experimental import DesignIntentInferenceModel, CADAnnotationOptions

options = CADAnnotationOptions(vlm_model="gpt-4-vision")
model = DesignIntentInferenceModel(options)
model(doc, doc.items)

# Access inferred intent
for item in doc.items:
    intent = item.properties.get("design_intent")
    print(f"Primary intent: {intent['primary_intent']}")
    print(f"Function: {intent['functional_description']}")
    print(f"Load-bearing: {intent['is_load_bearing']}")
```

### Feature 8: CADToTextGenerationModel

Generates comprehensive natural language descriptions of CAD parts.

**Generated Content:**
- Summary (1-sentence overview)
- Detailed description (2-3 paragraphs)
- Key features list
- Dimensions summary
- Manufacturing notes
- Assembly instructions

```python
from cadling.experimental import CADToTextGenerationModel, CADAnnotationOptions

options = CADAnnotationOptions(vlm_model="gpt-4-vision")
model = CADToTextGenerationModel(options)
model(doc, doc.items)

# Access generated description
for item in doc.items:
    desc = item.properties.get("text_description")
    print(desc['summary'])
    print(desc['detailed_description'])
    for feature in desc['key_features']:
        print(f"- {feature}")
```

### Feature 9: GeometricConstraintModel

Extracts implicit geometric constraints and relationships.

**Detected Constraints:**
- **Orientation**: parallel, perpendicular, tangent
- **Alignment**: concentric, coaxial, coincident
- **Symmetry**: bilateral, rotational
- **Dimensional**: distance, angle, equal length/radius

```python
from cadling.experimental import GeometricConstraintModel

model = GeometricConstraintModel(tolerance=0.001, min_confidence=0.7)
model(doc, doc.items)

# Access constraint graph
for item in doc.items:
    constraints = item.properties.get("constraints", [])
    for constraint in constraints:
        print(f"{constraint['constraint_type']}: {constraint['description']}")

    graph = item.properties.get("constraint_graph")
    print(f"Graph has {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")
```

## 🔧 Configuration Options

### CADAnnotationOptions

```python
from cadling.experimental import CADAnnotationOptions

options = CADAnnotationOptions(
    annotation_types=["dimension", "tolerance", "gdt", "surface_finish", "material", "welding"],
    min_confidence=0.7,
    vlm_model="gpt-4-vision",
    views_to_process=["front", "top", "right"],
    enable_cross_view_validation=True,
    extraction_resolution=2048,
    include_geometric_context=True,
)
```

### MultiViewOptions

```python
from cadling.experimental import MultiViewOptions, ViewConfig

options = MultiViewOptions(
    views=[
        ViewConfig(name="front", azimuth=0, elevation=0),
        ViewConfig(name="top", azimuth=0, elevation=90),
        # Add more views...
    ],
    resolution=2048,
    fusion_strategy="weighted_consensus",
    conflict_threshold=0.5,
    enable_lighting=True,
    render_edges=True,
    parallel_rendering=True,
)
```

### AssemblyAnalysisOptions

```python
from cadling.experimental import AssemblyAnalysisOptions

options = AssemblyAnalysisOptions(
    detect_components=True,
    extract_mates=True,
    generate_bom=True,
    check_interference=False,
    max_components=1000,
    bom_include_metadata=True,
    bom_include_properties=True,
    process_subassemblies=True,
    extract_fasteners=True,
    group_identical_parts=True,
)
```

## 🧪 Integration Examples

### Using Multiple Models Together

```python
from cadling.experimental import (
    ThreadedGeometryVlmPipeline,
    CADAnnotationOptions,
    PMIExtractionModel,
    FeatureRecognitionVlmModel,
    ManufacturabilityAssessmentModel,
)

# Configure
options = CADAnnotationOptions(vlm_model="gpt-4-vision")

# Create models
pmi_model = PMIExtractionModel(options)
feature_model = FeatureRecognitionVlmModel(options)
mfg_model = ManufacturabilityAssessmentModel(options)

# Add to pipeline
options.enrichment_models = [pmi_model, feature_model, mfg_model]
pipeline = ThreadedGeometryVlmPipeline(options)

# Execute
result = pipeline.execute(input_doc)

# Access all results
for item in result.document.items:
    pmi = item.properties.get("pmi_annotations")
    features = item.properties.get("machining_features")
    mfg_report = item.properties.get("manufacturability_report")
```

### Custom Pipeline with Experimental Features

```python
from cadling.pipeline import BaseCADPipeline
from cadling.experimental import PMIExtractionModel, CADAnnotationOptions

class CustomPipeline(BaseCADPipeline):
    def __init__(self, options):
        super().__init__(options)
        self.pmi_model = PMIExtractionModel(options)

    def _build_document(self, conv_res):
        # Custom build logic
        backend = conv_res.input._backend
        conv_res.document = backend.convert()
        return conv_res

    def _enrich_document(self, conv_res):
        # Apply experimental model
        if conv_res.document:
            self.pmi_model(conv_res.document, conv_res.document.items)
        return conv_res
```

## 🐛 Troubleshooting

### VLM API Keys

Most experimental models require VLM API keys:

```bash
# For OpenAI models (gpt-4-vision, etc.)
export OPENAI_API_KEY="sk-..."

# For Anthropic models (claude-3-opus, etc.)
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Memory Issues

For large assemblies or high-resolution rendering:
- Reduce `resolution` in MultiViewOptions
- Limit `max_components` in AssemblyAnalysisOptions
- Disable `parallel_rendering` if needed

### Performance Optimization

- Use `haiku` or smaller VLM models for faster processing
- Reduce `views_to_process` to minimum required
- Disable `enable_cross_view_validation` for speed
- Set `parallel_rendering=True` for multi-view pipelines

## 📚 API Reference

For detailed API documentation, see docstrings in:
- `cadling/experimental/datamodel/` - Options classes
- `cadling/experimental/models/` - Enrichment models
- `cadling/experimental/pipeline/` - Pipeline classes

## 🤝 Contributing

Contributions are welcome! To add new experimental features:

1. Follow existing patterns (see Feature 1-12 for examples)
2. Inherit from appropriate base classes
3. Include comprehensive docstrings
4. Add tests in `tests/experimental/`
5. Update this README with usage examples

## 📄 License

Same as parent CADling project.

## 🙏 Acknowledgments

Inspired by [Docling's experimental architecture](https://github.com/DS4SD/docling) but adapted for 3D CAD processing.
