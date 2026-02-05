"""CADling Experimental Features.

This module provides experimental features for advanced CAD processing,
inspired by Docling's experimental architecture. These features are subject
to change and may be promoted to core functionality or deprecated based on
user feedback and performance.

## Feature Categories

### Datamodels & Options (Features 10-12)
Configuration options for experimental features:
- CADAnnotationOptions: PMI and annotation extraction
- MultiViewOptions: Multi-view rendering and fusion
- AssemblyAnalysisOptions: Assembly processing

### Vision-Based Models (Features 4-6)
VLM-powered enrichment models:
- PMIExtractionModel: Extract dimensions, tolerances, GD&T
- FeatureRecognitionVlmModel: Identify machining features
- ManufacturabilityAssessmentModel: DFM analysis

### AI Understanding Models (Features 7-9)
AI-powered semantic analysis:
- DesignIntentInferenceModel: Infer design purpose
- CADToTextGenerationModel: Generate descriptions
- GeometricConstraintModel: Extract geometric relationships

### Advanced Pipelines (Features 1-3)
Experimental processing pipelines:
- ThreadedGeometryVlmPipeline: Two-stage geometry + VLM
- MultiViewFusionPipeline: Multi-view fusion
- AssemblyHierarchyPipeline: Assembly-aware processing

## Usage

```python
from cadling.experimental import (
    CADAnnotationOptions,
    PMIExtractionModel,
    ThreadedGeometryVlmPipeline,
)

# Configure options
options = CADAnnotationOptions(
    vlm_model="gpt-4-vision",
    annotation_types=["dimension", "tolerance", "gdt"],
    views_to_process=["front", "top", "isometric"],
)

# Create pipeline
pipeline = ThreadedGeometryVlmPipeline(options)

# Execute
result = pipeline.execute(input_doc)
```

## Stability Warning

**These features are experimental and may change without notice.**
- APIs may be modified or removed
- Performance characteristics may vary
- Not recommended for production use without thorough testing
- Feedback and contributions welcome!

## Documentation

For detailed documentation on each feature, see:
- README.md in this directory
- Docstrings in individual modules
- Example scripts in examples/
"""

# Datamodels & Options (Phase 1)
from .datamodel import (
    AssemblyAnalysisOptions,
    CADAnnotationOptions,
    MateType,
    MultiViewOptions,
    ViewConfig,
)

# Vision-Based Models (Phase 2)
from .models import (
    DFMIssue,
    DFMRule,
    FeatureRecognitionVlmModel,
    IssueSeverity,
    MachiningFeature,
    ManufacturabilityAssessmentModel,
    ManufacturabilityReport,
    ManufacturingProcess,
    PMIExtractionModel,
)

# AI Understanding Models (Phase 3)
from .models import (
    CADDescription,
    CADToTextGenerationModel,
    ConstraintGraph,
    ConstraintType,
    DesignIntent,
    DesignIntentInferenceModel,
    GeometricConstraint,
    GeometricConstraintModel,
    IntentCategory,
    LoadType,
)

# Advanced Pipelines (Phase 4)
from .pipeline import (
    AssemblyHierarchyPipeline,
    AssemblyNode,
    MultiViewFusionPipeline,
    ThreadedGeometryVlmPipeline,
)

__all__ = [
    # === Datamodels & Options (Features 10-12) ===
    "CADAnnotationOptions",
    "MultiViewOptions",
    "ViewConfig",
    "AssemblyAnalysisOptions",
    "MateType",
    # === Vision-Based Models (Features 4-6) ===
    "PMIExtractionModel",
    "FeatureRecognitionVlmModel",
    "MachiningFeature",
    "ManufacturabilityAssessmentModel",
    "ManufacturabilityReport",
    "DFMIssue",
    "DFMRule",
    "IssueSeverity",
    "ManufacturingProcess",
    # === AI Understanding Models (Features 7-9) ===
    "DesignIntentInferenceModel",
    "DesignIntent",
    "IntentCategory",
    "LoadType",
    "CADToTextGenerationModel",
    "CADDescription",
    "GeometricConstraintModel",
    "GeometricConstraint",
    "ConstraintType",
    "ConstraintGraph",
    # === Advanced Pipelines (Features 1-3) ===
    "ThreadedGeometryVlmPipeline",
    "MultiViewFusionPipeline",
    "AssemblyHierarchyPipeline",
    "AssemblyNode",
]

__version__ = "0.1.0-experimental"
