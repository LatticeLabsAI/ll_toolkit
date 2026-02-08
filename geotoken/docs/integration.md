# GeoToken Integration Guide

This guide explains how geotoken integrates with other LatticeLabs toolkit packages.

## Integration Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        LatticeLabs Toolkit                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                 │
│  │   cadling   │     │   geotoken  │     │  ll_stepnet │                 │
│  │             │     │             │     │             │                 │
│  │ CAD Parsing │────▶│ Tokenization│────▶│ ML Models   │                 │
│  │ & Analysis  │     │ & Encoding  │     │ & Training  │                 │
│  └─────────────┘     └─────────────┘     └─────────────┘                 │
│         │                   │                   │                        │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │              GeoTokenIntegration Bridge                             │ │
│  │         (cadling/backend/geotoken_integration.py)                   │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Primary Integration: GeoTokenIntegration Bridge

The central entry point for all geotoken interactions is `GeoTokenIntegration` in cadling:

```python
from cadling.backend.geotoken_integration import GeoTokenIntegration

bridge = GeoTokenIntegration()

# Tokenize a complete document
result = bridge.tokenize_document(
    doc,
    include_mesh=True,
    include_graph=True,
    include_commands=True,
    include_constraints=True
)

# Access results
mesh_tokens = result.mesh_tokens
graph_tokens = result.graph_tokens
command_tokens = result.command_tokens
token_ids = result.token_ids
metadata = result.metadata
```

### Bridge Methods

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `tokenize_document()` | CADlingDocument | GeoTokenResult | Complete document tokenization |
| `tokenize_mesh()` | vertices, faces | TokenSequence | Mesh geometry only |
| `tokenize_topology()` | TopologyGraph | TokenSequence | B-Rep graph only |
| `tokenize_sketch()` | Sketch2DItem | TokenSequence | Command sequence only |
| `encode_sequences()` | TokenSequence | list[int] | Token → ID encoding |
| `decode_token_ids()` | list[int] | TokenSequence | ID → Token decoding |
| `validate_roundtrip()` | original, reconstructed | dict | Quality metrics |

### Lazy Import Pattern

The bridge uses lazy imports for graceful degradation:

```python
def _try_import_tokenizers():
    """Import geotoken tokenizers, return None if unavailable."""
    try:
        from geotoken import GeoTokenizer, GraphTokenizer, CommandSequenceTokenizer
        return GeoTokenizer, GraphTokenizer, CommandSequenceTokenizer
    except ImportError:
        return None, None, None
```

When geotoken is unavailable, the bridge sets `degraded=True` in metadata and returns empty token sequences.

## Integration with cadling

### Data Flow: CADlingDocument → Tokens

```python
# 1. Parse CAD file with cadling
from cadling import DocumentConverter

converter = DocumentConverter()
doc = converter.convert("model.step")

# 2. Extract geometry items
mesh_item = doc.get_item_by_type("MeshItem")
vertices = mesh_item.to_numpy()  # Returns (N, 3) float32

topo_graph = doc.get_item_by_type("TopologyGraph")
node_features = topo_graph.to_numpy_node_features()  # (N, 48) float32
edge_index = topo_graph.to_edge_index()  # (2, M) int64
edge_features = topo_graph.to_numpy_edge_features()  # (M, 16) float32

sketch = doc.get_item_by_type("Sketch2DItem")
commands = sketch.to_geotoken_commands()  # List of command dicts

# 3. Tokenize via bridge
bridge = GeoTokenIntegration()
result = bridge.tokenize_document(doc)
```

### SegNetPipeline Integration

The SegNetPipeline uses geotoken for its three-stage process:

```python
from cadling.pipeline import SegNetPipeline

pipeline = SegNetPipeline(
    include_mesh_tokens=True,
    include_graph_tokens=True,
    include_command_tokens=True,
    include_constraints=True,
    vertex_merge_distance=0.005  # Post-process vertex clustering
)

result = pipeline.run(doc)
token_sequence = result.token_sequence
validation_report = result.validation_report
```

**Pipeline Stages:**
1. **Segment**: Parse document, run enrichment models
2. **Tokenize**: Use GeoTokenIntegration bridge
3. **Reconstruct**: Execute tokens via CommandExecutor

### GenerationPipeline Integration

For generative models, geotoken handles decoding:

```python
from cadling.generation import GenerationPipeline

pipeline = GenerationPipeline(backend="vae")

# Generate tokens from model
generated_ids = model.generate(prompt_embedding)

# Decode tokens to geometry
mesh = pipeline.decode_tokens_to_geometry(generated_ids)

# Validate generated mesh
validation = pipeline._validate_generated_mesh(mesh.vertices, mesh.faces)
```

## Integration with ll_stepnet

### Native Format Alignment

ll_stepnet and geotoken share the same CommandType enum:

```python
# geotoken/tokenizer/token_types.py
class CommandType(IntEnum):
    SOL = 0
    LINE = 1
    ARC = 2
    CIRCLE = 3
    EXTRUDE = 4
    EOS = 5

# ll_stepnet/stepnet/output_heads.py
class CommandType(IntEnum):
    SOL = 0
    LINE = 1
    ARC = 2
    CIRCLE = 3
    EXTRUDE = 4
    EOS = 5
```

This alignment enables zero-adapter integration.

### GeoTokenDataset

ll_stepnet provides a PyTorch Dataset wrapper:

```python
from ll_stepnet.stepnet.data import GeoTokenDataset, GeoTokenCollator

# Load tokenized data
dataset = GeoTokenDataset(token_sequences)
collator = GeoTokenCollator(pad_id=0)

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collator)

for batch in dataloader:
    command_types = batch["command_types"]  # (B, L)
    parameters = batch["parameters"]  # (B, L, 16)
    parameter_masks = batch["parameter_masks"]  # (B, L, 16)
```

### Feature Dimensions

Standard feature dimensions across packages:

| Feature | Dimensions | Used By |
|---------|------------|---------|
| Node features | 48 | cadling TopologyGraph, geotoken GraphTokenizer |
| Edge features | 16 | cadling TopologyGraph, geotoken GraphTokenizer |
| Command parameters | 16 | geotoken CommandToken, ll_stepnet output heads |

## Format Conversion

### DeepCAD ↔ cadling Format

geotoken includes a format converter for compatibility:

```python
from geotoken.tokenizer.command_format_converter import CommandFormatConverter

converter = CommandFormatConverter()

# DeepCAD format: compact, active-only parameters
deepcad_line = {"type": "LINE", "params": [0.0, 0.0, 1.0, 1.0]}

# cadling format: z-interleaved, 16-element padded
cadling_line = converter.deepcad_to_cadling("LINE", deepcad_line["params"])
# Result: [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Auto-detect format
format_type = converter.detect_format(commands)  # "deepcad" or "cadling"
```

## Vertex Post-Processing

Generated meshes often need post-processing:

```python
from geotoken.vertex import VertexValidator, VertexClusterer, VertexMerger

# Validate mesh quality
validator = VertexValidator()
report = validator.validate(vertices, faces)

if not report.manifold_check:
    print("Warning: Non-manifold mesh")

# Cluster and merge near-duplicate vertices
clusterer = VertexClusterer(merge_distance=0.005)
clustering = clusterer.cluster(vertices)
merged_verts, clean_faces = VertexMerger.merge(vertices, faces, clustering)

# Refine coarse predictions
from geotoken.vertex import CoarseToFineRefiner

refiner = CoarseToFineRefiner(
    max_iterations=20,
    learning_rate=0.1,
    convergence_threshold=1e-4
)
result = refiner.refine(merged_verts, target_points=reference)
```

## Quality Assessment

```python
from geotoken.impact import QuantizationImpactAnalyzer

analyzer = QuantizationImpactAnalyzer()
report = analyzer.analyze(original_vertices, faces, config)

print(f"Hausdorff distance: {report.hausdorff_distance}")
print(f"Mean error: {report.mean_error}")
print(f"Feature collapse rate: {report.feature_loss.collapse_rate}")
print(f"Relationship preservation: {report.relationship_preservation_rate}")
```

## SDG Integration

For synthetic data generation with text-CAD pairs:

```python
from cadling.sdg.qa.sequence_annotator import SequenceAnnotator

annotator = SequenceAnnotator()

# Generate text-token pairs for training
pairs = annotator.annotate(
    cad_file="model.step",
    annotation_level="detailed",
    include_constraints=True
)

# Output: JSONL with text annotations + command tokens
```

## Error Handling

The bridge handles missing dependencies gracefully:

```python
result = bridge.tokenize_document(doc)

if result.metadata.get("degraded"):
    print("Warning: geotoken not available, returning empty tokens")
    print(f"Errors: {result.errors}")
```

## Configuration Propagation

Pass configuration through the bridge:

```python
bridge = GeoTokenIntegration(config={
    "source_format": "deepcad",
    "include_constraints": True,
    "precision_tier": "STANDARD"
})
```
