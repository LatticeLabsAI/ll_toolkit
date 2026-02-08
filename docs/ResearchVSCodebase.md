# Research → Codebase Mapping: LatticeLabs Toolkit

A systematic comparison of the research document ("How Neural Networks Generate CAD") against the current state of the LatticeLabs toolkit. Each section walks through what the research describes, where it maps in the codebase, what's implemented, and what's missing.

---

## 1. Tokenization & Quantization

### What the research describes

The research identifies three tokenization paradigms for CAD data:

1. **Sketch-and-extrude sequences** (DeepCAD) — 6 command types + 16 continuous params per command, all quantized to 8-bit integers (256 levels) after normalizing to a 2×2×2 cube. Key insight: discrete classification preserves geometric relationships that continuous regression breaks.

2. **Disentangled codebooks** (SkexGen) — separate topology codes (500), geometry codes (1000), and extrusion codes (1000). Each model compresses to just 10 discrete codes.

3. **B-rep hierarchical latents** (BrepGen) — faces compressed to 48-dim latents via VAE from UV-sampled surface points, edges to 12-dim latents. Topology encoded implicitly through geometry similarity (mating duplication).

Critical research parameters:

- Normalization to unit cube (2×2×2 or 1×1×1)
- 6-bit (64 levels) for draft, 8-bit (256) for standard — matches DeepCAD exactly
- Minimum feature threshold of 0.05 to prevent collapse (BrepGen)
- Quantization is **classification not regression** — the core insight

### Where it maps in the codebase

**`geotoken/`** — the geometric tokenizer package

| File | What it does |
|------|-------------|
| `config.py` | `PrecisionTier` enum: DRAFT=6-bit, STANDARD=8-bit, PRECISION=10-bit |
| `tokenizer/token_types.py` | `CoordinateToken` (quantized xyz + bit width), `GeometryToken` (face/edge/vertex), `TokenSequence` |
| `tokenizer/geo_tokenizer.py` | Main pipeline: normalize → analyze → allocate → quantize → tokenize |
| `quantization/adaptive.py` | Per-vertex bit allocation based on curvature + feature density |
| `quantization/uniform.py` | Fixed-bit quantization (like DeepCAD's 8-bit) |
| `quantization/normalizer.py` | `RelationshipPreservingNormalizer` — centers + scales to unit cube |
| `quantization/bit_allocator.py` | Maps complexity scores to bit widths |
| `analysis/curvature.py` | Vertex curvature analysis for adaptive allocation |
| `analysis/feature_density.py` | Feature density analysis |
| `impact/analyzer.py` | `QuantizationImpactAnalyzer` — measures quality degradation |

### What's implemented ✅

- **Precision tiers** matching the research exactly: 6-bit draft (64 levels), 8-bit standard (256 levels), plus a 10-bit precision tier
- **Unit cube normalization** with configurable range — `RelationshipPreservingNormalizer`
- **Feature collapse prevention** with `minimum_feature_threshold = 0.05` — matches BrepGen's minimum threshold
- **Adaptive quantization** that allocates more bits to high-curvature regions — this goes *beyond* the research literature, which mostly uses uniform quantization
- **Round-trip tokenize/detokenize** with stored normalization metadata
- **Impact analysis** tooling to measure quantization quality — not found in any of the referenced papers

### What's missing / gaps 🔲

- **No sketch-and-extrude command vocabulary** — geotoken operates on mesh vertices/faces, not the DeepCAD-style command sequences (line, arc, circle, extrude). There's no `CommandToken` type for sketch primitives
- **No VQ-VAE / codebook-based compression** — SkexGen's disentangled codebooks (topology vs geometry vs extrusion) aren't represented. The tokenizer is purely geometric, no learned discrete latents
- **No B-rep surface latent encoding** — BrepGen's approach of UV-sampling faces into 32×32×3 grids and compressing via VAE to 48-dim codes isn't implemented. The tokenizer works at vertex level, not face-surface level
- **Token types are limited** — `GeometryToken` has a string `token_type` field for "face"/"edge"/"vertex"/"separator" but no support for parametric commands, boolean operations, or constraint tokens
- **No sequence length management** — DeepCAD normalizes to 60-command fixed-length sequences; geotoken has no concept of target sequence length for transformer consumption

---

## 2. Neural Network Architectures

### What the research describes

Four architecture families:

1. **Transformer encoder-decoder VAE** (DeepCAD) — 4 layers, 8 heads, 512 FFN, 256-dim latent space + WGAN-gp for latent sampling
2. **Parallel codebook decoders** (SkexGen) — 3 separate transformers, one per codebook, with VQ-EMA updates
3. **Structured diffusion** (BrepGen) — 4 separate transformer denoisers (12 layers, 12 heads, 1024 hidden), 1000 training steps, PNDM 200-step inference
4. **Graph neural networks** (UV-Net, BRepNet) — face UV-grid CNNs + graph message passing over face-adjacency; coedge convolution for oriented edge walks

### Where it maps in the codebase

**`ll_stepnet/stepnet/encoder.py`** — Neural architectures

| Class | Architecture | Matches |
|-------|-------------|---------|
| `STEPTransformerEncoder` | 6-layer, 8-head transformer encoder, 256-dim embeddings, 1024 FFN | Close to DeepCAD's encoder (4-layer, 8-head, 512 FFN) |
| `STEPTransformerDecoder` | Causal transformer decoder with autoregressive masking | Maps to DeepCAD's decoder concept |
| `STEPGraphEncoder` | Simple GNN with message passing over adjacency matrix | Simplified version of UV-Net/BRepNet graph processing |
| `STEPEncoder` | Combined: transformer for tokens + GNN for topology → fusion | Novel hybrid not directly in the research |

**`cadling/models/segmentation/architectures/`** — GNN architectures

| File | Architecture |
|------|-------------|
| `gat_net.py` | Multi-head Graph Attention Network (GAT) using `torch_geometric` — 4 layers, 8 heads, residual connections |
| `edge_conv_net.py` | EdgeConv network |
| `instance_segmentation.py` | Instance segmentation architecture |

**`cadling/models/segmentation/brep_graph_builder.py`** — Builds face adjacency graphs from B-Rep topology (nodes = faces, edges = shared edges), with node features: surface type, area, curvature, normal, centroid

### What's implemented ✅

- **Transformer encoder** with architecture close to DeepCAD's (slightly deeper at 6 layers vs 4)
- **Autoregressive decoder** with causal masking — the right foundation for sequence generation
- **Graph neural network** for topology — both a simple version in stepnet and a proper GAT with torch_geometric in cadling
- **Hybrid fusion** of token-level and graph-level features — this dual-encoder approach isn't common in the papers but is architecturally sound
- **B-Rep face graph construction** from STEP topology with meaningful node/edge features

### What's missing / gaps 🔲

- **No VAE latent space** — DeepCAD's 256-dim VAE with KL divergence isn't implemented. The encoder produces embeddings but there's no variational bottleneck for generation
- **No latent GAN** — DeepCAD's WGAN-gp for mapping noise to the VAE latent space (enables unconditional sampling) is absent
- **No diffusion denoiser** — BrepGen's 4-denoiser architecture with 1000-step DDPM isn't present. No noise scheduling, no PNDM sampler
- **No VQ-VAE** — SkexGen's vector quantization with EMA codebook updates isn't implemented
- **No cross-attention conditioning** — Text2CAD's BERT cross-attention into the decoder isn't present
- **GNN is undersized** — `STEPGraphEncoder` uses simple linear layers for message passing, not the specialized coedge convolution from BRepNet or the UV-grid CNNs from UV-Net
- **No decoder output heads** — DeepCAD has separate linear heads for command type vs parameters; the decoder here has no task-specific prediction heads

---

## 3. DFS Reserialization (STEP-LLM)

### What the research describes

STEP-LLM tackles processing raw STEP files by DFS-based reserialization that linearizes graph cross-references while preserving locality. Achieves >95% renderability on generated outputs.

### Where it maps in the codebase

**`ll_stepnet/stepnet/reserialization.py`** — Full implementation

| Component | Status |
|-----------|--------|
| `STEPEntityGraph.parse()` | Regex-based STEP entity parsing → reference graph |
| `STEPDFSSerializer.serialize()` | DFS traversal with configurable root strategy |
| Root strategies | `no_incoming`, `type_hierarchy` (B-rep type weights), `both` |
| ID renumbering | Sequential reassignment after DFS ordering |
| Float normalization | Configurable precision |
| Orphan handling | Appended after DFS traversal |
| Branch pruning | Each entity visited exactly once |

**`cadling/chunker/dfs_chunker/`** — Consumes reserialization for RAG chunking

### What's implemented ✅

- **Complete DFS reserialization pipeline** — this is one of the most research-aligned pieces in the codebase
- **B-Rep type hierarchy scoring** for root selection — `MANIFOLD_SOLID_BREP` (100) down to `GEOMETRIC_SET` (40)
- **ID renumbering** preserving locality — exactly what STEP-LLM describes
- **Float normalization** with configurable precision
- **Integration into chunking** — DFS-ordered chunks keep subtrees together for RAG

### What's missing / gaps 🔲

- **No sequence length limits for transformer input** — STEP-LLM discusses chunking for context windows; the DFS serializer produces variable-length output without target token budgets
- **No learned entity embeddings** — the reserialization is purely structural; no neural feature injection into the serialized output

---

## 4. Training Data & Datasets

### What the research describes

| Dataset | Size | What it provides |
|---------|------|-----------------|
| ABC | 1M models | Final STEP geometry only |
| DeepCAD | 178K models | Full sketch-and-extrude sequences as JSON |
| Fusion 360 Gallery | 8,625 models | Validated human design sequences + assemblies |
| SketchGraphs | 15M sketches | Constraint graphs (coincidence, tangency, etc.) |
| Text2CAD | 660K annotations | Multi-level text descriptions paired with CAD |

### Where it maps in the codebase

**`cadling/sdg/`** — Synthetic Data Generation

| Component | What it does |
|-----------|-------------|
| `qa/sample.py` | Samples passages from processed CAD documents |
| `qa/generate.py` | `CADGenerator` — generates Q&A pairs using LLMs |
| `qa/conceptual_generate.py` | Conceptual Q&A generation |
| `qa/critique.py` | Quality filtering of generated pairs |
| `qa/prompts/` | Prompt templates for different question types |
| `qa/utils.py` | `ChatAgent` abstraction for LLM backends |

**`cadling/experimental/models/cad_to_text_generation_model.py`** — Generates natural language descriptions from CAD (the reverse direction of Text2CAD)

### What's implemented ✅

- **SDG pipeline** for generating Q&A pairs from CAD documents — this is the foundation for creating training data
- **Multi-level annotation** support (annotation levels match Text2CAD's complexity tiers conceptually)
- **LLM-backed generation** with `ChatAgent` supporting multiple providers
- **CAD-to-text model** that generates structured descriptions (summary, features, dimensions, manufacturing notes) — reverse of Text2CAD's text-to-CAD direction
- **Critique/filtering** pipeline for quality control

### What's missing / gaps 🔲

- **No dataset loaders** for ABC, DeepCAD, Fusion 360, SketchGraphs, or Text2CAD datasets — the SDG system generates *new* training data from CAD files rather than consuming existing research datasets
- **No sketch-and-extrude sequence extraction** — parsing OnShape construction history (as DeepCAD does) isn't implemented
- **No constraint graph extraction** — SketchGraphs-style constraint graph parsing (coincidence, tangency, perpendicularity edges) isn't present, though `experimental/models/geometric_constraint_model.py` detects constraints from geometry rather than parsing design history
- **No text-CAD pairing at scale** — the SDG generates Q&A pairs but not the multi-level text descriptions (abstract → expert) that Text2CAD uses for conditioning

---

## 5. Conditioning Mechanisms (Text/Image → CAD)

### What the research describes

- **Text2CAD**: Frozen BERT → adaptive layer → cross-attention in CAD decoder (first 2 blocks skip cross-attention)
- **Zoo.dev**: LLM generates KCL code (executable CAD language)
- **LLM code generation**: GPT-4 achieves 96.5% compile rate on OpenSCAD/CadQuery; AIDL language improves LLM performance
- **Image conditioning**: DINOv2/CLIP embeddings; Img2CAD uses finetuned Llama3.2 VLM

### Where it maps in the codebase

**No text-to-CAD or image-to-CAD conditioning exists.**

The closest pieces:

- `cadling/models/vlm_model.py` — VLM integration (API-based and inline) for *understanding* CAD, not generating it
- `cadling/experimental/models/feature_recognition_vlm_model.py` — VLM-based feature recognition
- `cadling/sdg/qa/utils.py` — `ChatAgent` with LLM provider abstraction (could be repurposed for code generation)
- `ll_stepnet/stepnet/encoder.py` — `STEPTransformerDecoder` has the right architecture but no cross-attention from text embeddings

### What's missing / gaps 🔲

- **No cross-attention conditioning layer** — the decoder has self-attention but no mechanism to attend to external embeddings (BERT, CLIP, etc.)
- **No BERT/CLIP text encoder integration** — no frozen encoder producing conditioning vectors
- **No code generation backend** — no KCL, CadQuery, or OpenSCAD generation pipeline (this was Tier 2 priority in the roadmap)
- **No image encoder** for conditioning — DINOv2/CLIP not integrated for generation purposes
- **This is the biggest gap** between the research and the codebase — the toolkit is entirely understanding-direction (CAD → text/features) with no generation-direction (text → CAD) pipeline

---

## 6. Output Reconstruction & Validity

### What the research describes

- Sequence methods execute predicted commands in OpenCASCADE
- BrepGen decodes latents to point grids → fits B-spline surfaces → merges duplicate nodes → sews into solids
- Constraint solving achieves 93% fully-constrained sketches with RL alignment
- Validity rates: BrepGen 62.9% watertight (DeepCAD distribution), DeepCAD 46.1%, SolidGen 60.3%
- Common failures: non-watertight solids, self-intersections, dangling edges

### Where it maps in the codebase

**`cadling/models/topology_validation.py`** — Validation model

| Check | Status |
|-------|--------|
| Manifoldness (each edge shared by ≤2 faces) | Defined in docstring |
| Euler characteristic (V - E + F = 2 - 2g) | Defined in docstring |
| Watertightness | Defined in docstring |
| Self-intersection detection | Configurable flag |
| Orientation consistency | Defined in docstring |

**`cadling/models/geometry_analysis.py`** — Geometry analysis enrichment
**`cadling/models/mesh_quality.py`** — Mesh quality checks
**`cadling/backend/step/`** — STEP backend with OpenCASCADE integration (via pythonocc)

### What's implemented ✅

- **Topology validation framework** with the right checks defined (manifoldness, Euler, watertightness, self-intersection)
- **Per-entity finding tracking** with severity levels — `ValidationFinding` tracks which specific entities fail and why
- **pythonocc integration** for OpenCASCADE geometry operations
- **Mesh quality analysis** module

### What's missing / gaps 🔲

- **Validation is for *analysis*, not *generation repair*** — the topology validation model checks existing geometry but doesn't include the BrepGen-style reconstruction pipeline (decode latents → fit B-splines → merge nodes → sew)
- **No B-spline surface fitting** from point samples (`GeomAPI_PointsToBSplineSurface`)
- **No mating duplication / node merging** logic — BrepGen's scheme of detecting similar geometry (bbox distance < 0.08, shape threshold < 0.2) and merging to recover topology
- **No constraint solver** — SketchGen's Newton's method solver for sketch constraint satisfaction isn't present
- **No generation-loop validation** — no feedback loop where validation results feed back into a generator (the RL alignment achieving 93% constrained sketches)
- **Implementation status of validation checks is unclear** — the docstring says they exist but the file header notes this is in `RequiredToBeCorrected.md` (~200 methods with placeholder implementations)

---

## 7. Graph Neural Networks for B-Rep

### What the research describes

- **UV-Net**: Processes faces as 10×10 UV-grids through surface CNNs (7 channels: xyz + normals + trim mask), then graph message passing over face-adjacency
- **BRepNet**: Coedge convolution — oriented edges with next/prev/mate pointers enabling walks that capture manufacturing patterns
- These produce 64-128 dim shape embeddings for retrieval, segmentation, and conditioning generation

### Where it maps in the codebase

| File | What | Research analog |
|------|------|----------------|
| `cadling/models/segmentation/brep_graph_builder.py` | Builds face adjacency graphs from STEP B-Rep | Matches UV-Net's graph structure |
| `cadling/models/segmentation/architectures/gat_net.py` | Multi-head GAT with residual connections | Standard GNN, not UV-Net or BRepNet specific |
| `cadling/models/segmentation/architectures/edge_conv_net.py` | EdgeConv network | Dynamic graph convolution |
| `cadling/models/segmentation/geometry_extractors.py` | Extracts manufacturing features (holes, pockets, etc.) | Feature recognition, not graph encoding |
| `ll_stepnet/stepnet/encoder.py` → `STEPGraphEncoder` | Simple GNN with adjacency message passing | Minimal version |

### What's implemented ✅

- **Face adjacency graph construction** with proper node features (surface type, area, curvature, normal, centroid) and edge features (dihedral angle, edge type)
- **GAT architecture** with multi-head attention, residual connections — a solid base architecture
- **Manufacturing feature extraction** from graph structure — holes, pockets, bosses, fillets, chamfers
- **Graph-level + token-level fusion** in `STEPEncoder`

### What's missing / gaps 🔲

- **No UV-grid face encoding** — UV-Net's core innovation of sampling each face on a 10×10 UV grid with 7 channels and processing through a surface CNN isn't implemented. The graph uses extracted features per face, not learned per-face representations
- **No coedge convolution** — BRepNet's oriented edge walking (next/prev/mate coedge pointers) isn't present. The adjacency is face-level, not coedge-level
- **No learned embeddings for generation conditioning** — the GNN outputs are used for segmentation tasks, not as conditioning vectors for generative models
- **Graph scale is limited** — the research handles shapes with 30-80 faces and notes attention scales quadratically; no scalability mechanisms are present

---

## Summary: Coverage Heat Map

| Research Area | Understanding Direction | Generation Direction |
|--------------|------------------------|---------------------|
| **Tokenization** | ✅ Strong (geotoken with adaptive quantization) | 🔲 No command vocabulary, no VQ-VAE |
| **Transformer architecture** | ✅ Encoder implemented | 🔲 No VAE, no diffusion, no conditioning |
| **Graph networks** | ✅ GAT + face graph builder | 🔲 No UV-Net, no coedge conv, no gen conditioning |
| **DFS reserialization** | ✅ Complete (STEP-LLM aligned) | ✅ Same code usable for both directions |
| **Training data / SDG** | ✅ Q&A generation pipeline | 🔲 No research dataset loaders, no text-CAD pairing |
| **Text/image conditioning** | N/A | 🔲 Nothing — biggest gap |
| **Topology validation** | ✅ Framework defined | 🔲 No generation repair loop |
| **Output reconstruction** | N/A | 🔲 No B-spline fitting, no node merging, no constraint solving |

### The pattern is clear

The toolkit has strong **CAD → Understanding** infrastructure (parsing, feature extraction, graph construction, tokenization, validation, SDG) but essentially **zero** generation-direction capability. Every piece of the generation pipeline from the research — VAE latent spaces, diffusion denoisers, cross-attention conditioning, output reconstruction, constraint solving — is absent.

The good news: the understanding infrastructure is exactly what you need as a foundation. The encoder architectures, graph builders, tokenizers, and validators are the building blocks that generation systems need. The gap is in connecting them into a generation loop.
