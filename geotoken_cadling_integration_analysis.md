# GeoToken ↔ Cadling Integration Analysis

## Executive Summary

GeoToken and Cadling are currently **completely decoupled** — no imports, no shared interfaces, no data format contracts. While both modules are individually mature, integrating them reveals **7 critical gaps** where data formats don't align, **3 missing geotoken capabilities** that cadling's output requires, and **2 architectural decisions** that need to be made about where the integration layer lives.

This document maps every integration point, identifies every mismatch, and proposes what needs to change in each module.

---

## 1. The Two Modules at a Glance

### What Cadling Produces

Cadling is a CAD document processing toolkit. Given a STEP, STL, or DXF file, it produces:

| Output | Format | Shape |
|--------|--------|-------|
| `CADlingDocument` | Pydantic model | Hierarchical container |
| `TopologyGraph` | Adjacency list + features | N nodes, M edges |
| Node features | `np.ndarray` | `[num_faces, 48]` float32 |
| Edge features | `np.ndarray` | `[num_edges, 16]` float32 |
| Face UV-grids | `Dict[int, np.ndarray]` | `{face_idx: [10, 10, 7]}` |
| Edge UV-grids | `Dict[int, np.ndarray]` | `{edge_idx: [10, 6]}` |
| PyG Data object | `torch_geometric.data.Data` | Graph-ready tensors |
| 2D command sequences | `List[Dict]` | `{"type": "LINE", "params": [16 floats]}` |
| Geometric constraints | `List[Dict]` | `{"type": "PARALLEL", ...}` |
| STEP token IDs | `List[int]` | From STEPTokenizer (50K vocab) |

### What GeoToken Consumes

GeoToken is a geometric tokenizer. It has two main entry points:

| Entry Point | Expected Input | Shape |
|-------------|---------------|-------|
| `GeoTokenizer.tokenize()` | vertices + faces | `(N, 3)` float + `(F, 3)` int |
| `CommandSequenceTokenizer.tokenize()` | construction history | DeepCAD JSON format |
| `CADVocabulary.encode()` | `List[CommandToken]` | From CommandSequenceTokenizer |

---

## 2. Integration Point Map

There are **5 natural integration points** where cadling's output can flow into geotoken's input. Each has different alignment status.

### Integration Point 1: STL/Mesh Pipeline → GeoTokenizer

**Path:** `DocumentConverter.convert("file.stl")` → mesh vertices/faces → `GeoTokenizer.tokenize()`

**Status: GAP — Data extraction needed**

Cadling's STL backend produces `MeshItem` objects inside `CADlingDocument.items`. These items contain mesh data, but geotoken expects raw `np.ndarray` vertices `(N, 3)` and faces `(F, 3)`.

**The problem:** Cadling's `MeshItem` stores mesh data inside `properties` dict as serialized metadata — not as raw numpy arrays ready for geotoken. There is no standard method like `mesh_item.get_vertices()` that returns `np.ndarray(N, 3)`.

**What's needed:**

- A bridge function that extracts vertices and faces from a `CADlingDocument` containing mesh items
- Or: `MeshItem` needs `.vertices` and `.faces` properties that return numpy arrays
- GeoTokenizer works perfectly once it receives `(N, 3)` vertices — the normalization, adaptive quantization, and tokenization pipeline all work

### Integration Point 2: 2D Sketch Extraction → CommandSequenceTokenizer

**Path:** `SketchGeometryExtractor` → command sequences → `CommandSequenceTokenizer.tokenize()`

**Status: PARTIAL MATCH — Format mismatch on params**

Cadling's `SketchGeometryExtractor` produces commands in this format:

```python
{"type": "LINE", "params": [x1, y1, 0, x2, y2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
```

GeoToken's `CommandSequenceTokenizer` expects DeepCAD format:

```python
{"type": "Line", "params": [x1, y1, x2, y2]}  # 4 params only
```

**The mismatches:**

1. **Parameter count**: Cadling pads to 16 floats with trailing zeros. GeoToken expects compact params (LINE=4, ARC=6, CIRCLE=3, SOL=2, EXTRUDE=8).

2. **3D vs 2D**: Cadling inserts z-coordinates (always 0.0 for 2D sketches) interleaved with x,y. GeoToken expects pure 2D parameters.

3. **Command type casing**: Cadling uses uppercase (`"LINE"`, `"ARC"`, `"CIRCLE"`). GeoToken accepts both but normalizes to its own `CommandType` enum internally — this actually works.

4. **Missing `EXTRUDE` commands**: Cadling's sketch extractor produces only sketch-level commands (SOL, LINE, ARC, CIRCLE, EOS). It doesn't produce EXTRUDE commands because those come from the 3D construction history, not the 2D sketch. GeoToken's `CommandSequenceTokenizer` expects full construction histories including extrusions.

5. **No `"sequence"` wrapper**: GeoToken accepts either `{"sequence": [...]}` or a bare list — cadling stores commands as a bare list in `item.properties["command_sequence"]`, which is fine.

**What's needed:**

- A format adapter that strips the trailing zeros and z-coordinates from cadling's 16-param format to geotoken's compact format
- Or: modify `SketchGeometryExtractor` to optionally produce DeepCAD-compatible compact params
- The EXTRUDE gap is architectural — cadling would need a separate module to reconstruct full construction histories from B-Rep

### Integration Point 3: Geometric Constraints → ConstraintToken

**Path:** `SketchGeometryExtractor` constraints → `ConstraintToken`

**Status: GAP — GeoToken infrastructure exists but is unused**

Cadling's sketch extractor detects geometric constraints:

```python
{"type": "PARALLEL", "entity_a": 0, "entity_b": 2, "confidence": 0.95}
```

GeoToken defines `ConstraintToken` and `ConstraintType`:

```python
@dataclass
class ConstraintToken:
    constraint_type: ConstraintType
    source_index: int
    target_index: int
    value: Optional[float] = None
```

With `ConstraintType` enum: `COINCIDENT, PARALLEL, PERPENDICULAR, TANGENT, EQUAL, HORIZONTAL, VERTICAL, FIXED, CONCENTRIC, SYMMETRIC, COLINEAR`.

**The problems:**

- GeoToken's `CommandSequenceTokenizer` has `include_constraints: bool = False` in its config — constraints are **disabled by default** and the constraint tokenization code path is **not implemented** (the config exists but no code reads constraints from input)
- GeoToken's `CADVocabulary` doesn't encode constraint tokens — `encode()` only handles `CommandToken` objects
- Cadling produces constraints that map well to GeoToken's types (PARALLEL, PERPENDICULAR, CONCENTRIC, TANGENT, EQUAL_RADIUS), but there's no code path to actually use them

**What's needed:**

- GeoToken needs a `ConstraintTokenizer` or the `CommandSequenceTokenizer` needs to actually implement constraint handling
- `CADVocabulary` needs to be extended to encode `ConstraintToken` objects into token IDs
- A mapping from cadling's constraint format to geotoken's `ConstraintToken`

### Integration Point 4: B-Rep Topology Graph → GeoTokenizer (Enhanced)

**Path:** `TopologyGraph` + 48-dim node features + 16-dim edge features → ???

**Status: NO PATH EXISTS — GeoToken has no graph tokenizer**

This is the **biggest gap**. Cadling's crown jewel output is the enriched B-Rep topology graph with 48-dimensional node features, 16-dimensional edge features, UV-grids, and face-to-face adjacency. This is exactly what CAD AI models (BRepNet, UV-Net, SolidGen) consume.

GeoToken has **no capability to tokenize graph-structured data**. Its two tokenizers handle:

- Raw vertex/face meshes (`GeoTokenizer`) — just coordinates, no features
- Command sequences (`CommandSequenceTokenizer`) — parametric construction history

Neither can handle a topology graph with heterogeneous node/edge features.

**What's needed (this is significant new functionality):**

- A `GraphTokenizer` class that can convert a topology graph with rich node/edge features into a token sequence
- Token types for graph structure (node tokens, edge tokens, adjacency tokens)
- A strategy for serializing 48-dim features into discrete tokens (could use geotoken's existing quantization)
- UV-grid tokenization — how to represent `[10, 10, 7]` grids as tokens
- Vocabulary extension to handle graph tokens

### Integration Point 5: STEP Token IDs → ll_stepnet → Embeddings → GeoToken

**Path:** Cadling STEP backend → `STEPTokenizer` → token IDs → ll_stepnet encoder → embeddings → ???

**Status: INDIRECT — Embeddings don't have a geotoken consumer**

Cadling stores embeddings in `CADlingDocument.embeddings: List[List[float]]`. These come from ll_stepnet's transformer encoder. GeoToken has no concept of consuming pre-computed embeddings — it always works from raw geometry.

**What's needed:**

- This is more of a design question: should geotoken be extended to accept pre-computed embeddings as additional features, or should the embedding → token pipeline live outside geotoken?
- If geotoken handles it: a method like `tokenize_with_embeddings(vertices, faces, embeddings)` that can incorporate learned features into the quantization decisions

---

## 3. Detailed Gap Analysis

### Gap 1: No Mesh Extraction API in Cadling

**Severity: Medium**
**Affects: Integration Point 1**

Cadling doesn't expose a clean `.vertices` / `.faces` property on `MeshItem`. The mesh data is stored in various ways depending on the backend (STL stores raw triangles, STEP stores B-Rep shells). There's no unified method to get `np.ndarray(N, 3)` vertices and `np.ndarray(F, 3)` face indices.

**Fix location:** `cadling/datamodel/base_models.py` — add vertex/face extraction to `MeshItem`, or create a utility in the integration layer.

### Gap 2: Command Sequence Format Mismatch

**Severity: Medium**
**Affects: Integration Point 2**

Cadling's 16-param padded format with interleaved z-coordinates doesn't match geotoken's compact DeepCAD format.

**Fix options:**

- **Option A:** Add a `to_deepcad_format()` method to cadling's output
- **Option B:** Make geotoken's `CommandSequenceTokenizer` accept cadling's format with a `source_format="cadling"` config option
- **Option C:** Write a standalone adapter function in the integration layer

Option B is cleanest — it keeps both modules independent while adding flexibility to geotoken.

### Gap 3: No Graph Tokenization in GeoToken

**Severity: Critical**
**Affects: Integration Point 4**

GeoToken cannot tokenize graph-structured data with rich features. This is the most important data cadling produces for ML, and geotoken has no way to consume it.

**This is the single biggest piece of work needed for useful integration.**

A `GraphTokenizer` would need to:

1. Accept `TopologyGraph` + node features `(N, 48)` + edge features `(M, 16)` + UV-grids
2. Quantize continuous features into discrete tokens using existing adaptive quantization
3. Serialize the graph structure (nodes, edges, adjacency) into a flat token sequence
4. Handle variable-size graphs with padding/truncation
5. Support detokenization back to graph structure

### Gap 4: Constraint Tokenization Not Implemented

**Severity: Low-Medium**
**Affects: Integration Point 3**

GeoToken defines the data structures (`ConstraintToken`, `ConstraintType`) and config option (`include_constraints`), but the actual tokenization code path doesn't exist. The vocabulary also can't encode constraints.

**Fix location:** `geotoken/tokenizer/command_tokenizer.py` — implement constraint parsing and encoding. `geotoken/tokenizer/vocabulary.py` — extend to handle constraint tokens.

### Gap 5: No Integration Layer / Bridge Module

**Severity: Medium**
**Affects: All integration points**

There is no `cadling_geotoken_bridge.py` or similar integration module. Cadling has `stepnet_integration.py` for ll_stepnet, but nothing equivalent for geotoken.

**Fix:** Create an integration module (in cadling, in geotoken, or as a top-level package) that handles data format conversion between the two.

### Gap 6: UV-Grid Tokenization Missing from GeoToken

**Severity: Medium**
**Affects: Integration Point 4**

Cadling produces rich UV-grids: `[10, 10, 7]` per face (points + normals + trimming) and `[10, 6]` per edge (points + tangents). These are the standard representation used by UV-Net and subsequent B-Rep ML models.

GeoToken has no concept of UV-grid tokenization. Its `GeoTokenizer` only handles raw vertex coordinates, not structured surface samplings.

**What's needed:** Either a dedicated UV-grid quantizer in geotoken, or UV-grids are flattened into the node/edge feature vectors before tokenization (cadling already does this partially — 18 UV-grid statistics are in the 48-dim node features).

### Gap 7: Vocabulary Incompatibility

**Severity: Low-Medium**

Cadling has two tokenizer systems:

- `STEPTokenizer` with ~50,000 vocab (STEP entity types, keywords, operators)
- `CADTokenizer` interface (Simple, GPT, HuggingFace backends)

GeoToken has:

- `CADVocabulary` with ~24,581 vocab (command types × parameters × quantization levels)

These vocabularies serve different purposes and are completely incompatible. If a downstream transformer needs to process both STEP text tokens and geometric command tokens, there's no unified vocabulary.

**Fix options:**

- Keep vocabularies separate (multimodal model with separate embedding tables)
- Create a merged vocabulary that encompasses both
- Use geotoken's vocabulary for geometry and cadling's for text/structure

---

## 4. Impact on Cadling

### Changes Required in Cadling

| Change | Files Affected | Effort |
|--------|---------------|--------|
| Add mesh vertex/face extraction to MeshItem | `cadling/datamodel/base_models.py` | Small |
| Add `to_deepcad_format()` to command sequences | `cadling/models/segmentation/sketch_geometry_extractor.py` | Small |
| Create geotoken integration module | New: `cadling/backend/geotoken_integration.py` | Medium |
| Expose topology graph in geotoken-consumable format | `cadling/lib/graph/brep_graph.py` | Medium |
| Add geotoken as optional dependency | `pyproject.toml` | Trivial |

### What Cadling Should NOT Change

- The 48-dim node features and 16-dim edge features are well-designed and should remain as-is
- The UV-grid format `[10, 10, 7]` is industry-standard (matches UV-Net paper) — keep it
- The `TopologyGraph` adjacency list representation is fine
- The `CADlingDocument` model shouldn't embed geotoken types — keep the integration at the boundary

### Risk Assessment

**Low risk:** Adding mesh extraction properties and format adapters are additive changes — they won't break existing functionality.

**Medium risk:** Creating the integration module means cadling would import geotoken (or vice versa), introducing a dependency between previously independent packages. This should be an **optional** dependency with graceful fallback, following the pattern cadling already uses with ll_stepnet.

---

## 5. Impact on GeoToken

### Changes Required in GeoToken

| Change | Files Affected | Effort |
|--------|---------------|--------|
| Support cadling's 16-param command format | `geotoken/tokenizer/command_tokenizer.py` | Small |
| Implement constraint tokenization | `geotoken/tokenizer/command_tokenizer.py`, `vocabulary.py` | Medium |
| Create `GraphTokenizer` class | New: `geotoken/tokenizer/graph_tokenizer.py` | **Large** |
| Create graph token types | `geotoken/tokenizer/token_types.py` | Medium |
| Extend vocabulary for graph + constraint tokens | `geotoken/tokenizer/vocabulary.py` | Medium |
| Add UV-grid quantization | New: `geotoken/quantization/uv_grid_quantizer.py` | Medium |
| Add feature vector quantization | `geotoken/quantization/adaptive.py` or new file | Medium |
| Update public API | `geotoken/__init__.py` | Small |

### What GeoToken Handles Well Already

- **Normalization**: `RelationshipPreservingNormalizer` works for any `(N, 3)` vertex data — cadling's mesh vertices can go straight in
- **Adaptive quantization**: Works perfectly for mesh tokenization — curvature-aware bit allocation is exactly what STL data needs
- **Command tokenization pipeline**: The normalize → quantize → pad pipeline is solid and just needs format flexibility
- **Vocabulary encoding**: The token ID scheme is well-designed and extensible
- **Impact analysis**: Hausdorff distance and reconstruction metrics can validate any tokenization quality

### The Big Missing Piece: GraphTokenizer

This is the largest piece of work. A `GraphTokenizer` would be a new class roughly equivalent in scope to the existing `GeoTokenizer` + `CommandSequenceTokenizer` combined. It would need:

1. **Node tokenization**: Quantize 48-dim features into a fixed-length token subsequence per node
2. **Edge tokenization**: Quantize 16-dim features into tokens per edge
3. **Adjacency encoding**: Represent graph connectivity — options include adjacency list tokens, edge index tokens, or implicit ordering (nodes sorted by connectivity)
4. **UV-grid tokenization**: Optionally embed quantized UV-grid samples per node/edge
5. **Graph-level metadata**: Number of nodes, edges, graph properties
6. **Variable-size handling**: Padding/truncation for batch processing
7. **Detokenization**: Reconstruct graph structure from token sequence

**Estimated scope:** ~400-600 lines of new code plus ~200 lines of tests.

---

## 6. Architectural Decision: Where Does Integration Live?

### Option A: Integration Layer in Cadling

```
cadling/
  backend/
    geotoken_integration.py   ← Bridge module (like stepnet_integration.py)
```

**Pros:** Follows the existing pattern with ll_stepnet. Cadling is the "orchestrator" that calls out to processing modules.
**Cons:** Cadling grows larger. GeoToken remains unaware of its primary consumer.

### Option B: Integration Layer in GeoToken

```
geotoken/
  adapters/
    cadling_adapter.py   ← Accepts cadling data types
```

**Pros:** GeoToken becomes self-contained for cadling data. Clean API for cadling to call.
**Cons:** GeoToken gains a dependency on cadling's data model (even if optional).

### Option C: Top-Level Integration Package

```
LatticeLabs_toolkit/
  integration/
    cadling_geotoken.py   ← Standalone bridge
```

**Pros:** Neither module depends on the other. Clean separation. Easy to test independently.
**Cons:** A third package to maintain. Users must import from three places.

**Recommendation: Option A** — it matches the existing ll_stepnet pattern, keeps geotoken independent, and cadling is already designed as the integration hub.

---

## 7. Proposed Integration Architecture

```
STEP/STL/DXF File
       │
       ▼
┌──────────────┐
│   Cadling     │
│  DocumentConverter.convert()
│               │
│  ┌────────────┤
│  │ CADlingDocument
│  │  ├── items (MeshItem, STEPEntityItem, Sketch2DItem)
│  │  ├── topology (TopologyGraph + 48d/16d features)
│  │  ├── segments (semantic features)
│  │  └── embeddings (from ll_stepnet)
│  └────────────┤
└───────┬───────┘
        │
        ▼
┌───────────────────────┐
│  geotoken_integration  │  ← New bridge module in cadling
│                        │
│  ┌─ extract_mesh()─────┼──→ GeoTokenizer.tokenize(vertices, faces)
│  │                     │      → TokenSequence (coordinate + geometry tokens)
│  │                     │
│  ├─ extract_commands()─┼──→ CommandSequenceTokenizer.tokenize(history)
│  │                     │      → TokenSequence (command tokens)
│  │                     │
│  ├─ extract_graph()────┼──→ GraphTokenizer.tokenize(topology, features)  [NEW]
│  │                     │      → TokenSequence (graph + feature tokens)
│  │                     │
│  └─ extract_constraints()──→ ConstraintTokenizer  [NEW]
│                        │      → TokenSequence (constraint tokens)
└───────────┬────────────┘
            │
            ▼
     CADVocabulary.encode()
            │
            ▼
     Token IDs → Transformer
```

---

## 8. Priority Ranking

Based on ML utility and implementation effort:

| Priority | Task | Value | Effort |
|----------|------|-------|--------|
| **P0** | Command sequence format adapter | Unlocks 2D sketch tokenization | Small |
| **P0** | Mesh extraction from CADlingDocument | Unlocks STL/mesh tokenization | Small |
| **P1** | GraphTokenizer for B-Rep topology | Unlocks the most valuable cadling output | Large |
| **P1** | Feature vector quantization (48d/16d) | Required by GraphTokenizer | Medium |
| **P2** | Constraint tokenization implementation | Adds sketch constraint awareness | Medium |
| **P2** | UV-grid quantization | Adds surface detail to graph tokens | Medium |
| **P3** | Unified vocabulary strategy | Needed for multimodal transformer | Medium |
| **P3** | Embedding integration | Incorporates ll_stepnet learned features | Small |

---

## 9. Concrete Next Steps

### Phase 1: Quick Wins (enable basic integration)

1. Add `get_mesh_data()` utility to cadling that returns `(vertices, faces)` numpy arrays from MeshItem
2. Add `source_format="cadling"` support to geotoken's `CommandSequenceTokenizer` that handles 16-param padded format
3. Create `cadling/backend/geotoken_integration.py` with bridge functions
4. Write integration tests using cadling's existing test data

### Phase 2: Graph Tokenization (the big value-add)

1. Design the `GraphTokenizer` API and token format
2. Implement feature vector quantization (extend adaptive quantizer for N-dim features)
3. Implement graph structure serialization to token sequences
4. Extend `CADVocabulary` for graph tokens
5. Write comprehensive tests

### Phase 3: Polish and Constraints

1. Implement constraint tokenization in `CommandSequenceTokenizer`
2. Add UV-grid quantization (optional, high-detail mode)
3. Create unified vocabulary option for multimodal models
4. Add embedding-aware tokenization (optional)

---

## 10. Summary of Issues

### Will Not Work Without Changes

1. **GeoToken cannot consume cadling's B-Rep topology graph** — no graph tokenizer exists
2. **Cadling's command sequence format doesn't match** — 16-param padded vs compact DeepCAD
3. **No mesh extraction API** — can't get numpy arrays from CADlingDocument
4. **Constraint tokenization is defined but unimplemented** — config exists, code path doesn't

### Works But Needs a Thin Adapter

1. **Vocabulary systems are separate** — different purposes, need a strategy for unified models
2. **UV-grids have no tokenization path** — geotoken doesn't handle structured surface samples
3. **ll_stepnet embeddings aren't geotoken-aware** — could inform quantization but no mechanism exists

### Works Out of the Box

1. **STL mesh vertices → GeoTokenizer** — once extracted, adaptive tokenization works perfectly
2. **Normalization** — geotoken's relationship-preserving normalization handles any 3D geometry
3. **Impact analysis** — Hausdorff distance, reconstruction error work for any tokenized geometry
4. **Precision tiers** — DRAFT/STANDARD/PRECISION map naturally to cadling's different use cases
