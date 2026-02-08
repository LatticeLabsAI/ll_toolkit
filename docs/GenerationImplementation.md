# Generation Pipeline Implementation Outline

A comprehensive outline of every change and new module needed to close the gaps between the CAD generation research and the LatticeLabs toolkit. Organized from foundational infrastructure up through the full generation loop.

Each section explains **what** needs to be built, **why** it's needed (mapped to the research), **where** it goes in the existing codebase, and **what existing code** it connects to.

---

## Phase 1: Extend the Tokenizer for CAD Command Sequences

### Why this comes first

The research is unambiguous: the single most impactful representation choice is **sketch-and-extrude command sequences** (DeepCAD, SkexGen, Text2CAD all use them). Right now `geotoken` operates on raw mesh vertices and faces — it can quantize geometry, but it can't represent the *construction history* that created that geometry. Every generation architecture downstream depends on having a command-level vocabulary.

### 1.1 — New: Command Token Types

**Where**: `geotoken/geotoken/tokenizer/token_types.py`

**What changes**: Add new token dataclasses alongside the existing `CoordinateToken` and `GeometryToken`.

New types needed:

- **`CommandToken`** — represents a single CAD operation
  - `command_type`: enum field — one of `SOL` (start of loop), `LINE`, `ARC`, `CIRCLE`, `EXTRUDE`, `EOS` (end of sequence). These are DeepCAD's 6 command types.
  - `parameters`: list of quantized integers (up to 16 values per command — coordinates, angles, distances, boolean op flags)
  - `parameter_mask`: which parameters are active for this command type (LINE uses xy endpoints, CIRCLE uses center + radius, etc.)

- **`ConstraintToken`** — represents geometric constraints between primitives
  - `constraint_type`: enum — `COINCIDENT`, `TANGENT`, `PERPENDICULAR`, `PARALLEL`, `CONCENTRIC`, `EQUAL_LENGTH`, `EQUAL_RADIUS`, `DISTANCE`, `ANGLE`
  - `source_index`: index of first primitive in sequence
  - `target_index`: index of second primitive
  - `value`: optional quantized constraint value (for distance/angle constraints)
  - Maps to SketchGraphs' constraint edges

- **`BooleanOpToken`** — for CSG-style operations
  - `op_type`: `UNION`, `INTERSECTION`, `SUBTRACTION`
  - `operand_indices`: which bodies are being combined

- **`SequenceConfig`** — metadata for a command sequence
  - `max_commands`: target sequence length (DeepCAD uses 60)
  - `quantization_bits`: bits for parameter quantization
  - `coordinate_range`: normalization range (2×2×2 cube)
  - `padding_token_id`: for fixed-length padding

**What stays the same**: `CoordinateToken`, `GeometryToken`, `TokenSequence` remain unchanged — they handle the mesh-level tokenization path. The new command tokens are an alternative representation path for parametric CAD.

### 1.2 — New: Command Sequence Tokenizer

**Where**: `geotoken/geotoken/tokenizer/command_tokenizer.py` (new file)

**What it does**: Converts parametric CAD construction history into fixed-length command token sequences.

Pipeline steps:

1. **Parse construction history** — accept input as either:
   - DeepCAD-format JSON (list of sketch + extrude operations)
   - OnShape API feature tree responses
   - CadQuery script AST (parsed Python)

2. **Normalize sketches** — translate each sketch to origin, scale to 2×2 square, canonicalize loop ordering (counter-clockwise, starting from bottom-left vertex). This matches DeepCAD's normalization exactly.

3. **Normalize 3D** — scale the full solid to a 2×2×2 cube using the existing `RelationshipPreservingNormalizer` from `geotoken/quantization/normalizer.py`.

4. **Quantize parameters** — map each continuous parameter to discrete levels using the existing `PrecisionTier` system (6-bit/64 levels for draft, 8-bit/256 for standard). This is *classification not regression* — the core insight from the research.

5. **Pad/truncate to fixed length** — target 60 commands (DeepCAD standard). Shorter sequences get `PAD` tokens, longer sequences get truncated with priority to keeping complete sketch-extrude pairs intact.

6. **Output** — returns an extended `TokenSequence` with both `command_tokens` and the associated `coordinate_tokens`.

**What existing code it uses**:

- `RelationshipPreservingNormalizer` for unit cube normalization
- `PrecisionTier` for bit-width selection
- `minimum_feature_threshold = 0.05` for collapse prevention
- `QuantizationImpactAnalyzer` to measure command quantization quality

### 1.3 — New: Vocabulary Builder

**Where**: `geotoken/geotoken/tokenizer/vocabulary.py` (new file)

**What it does**: Builds and manages the discrete token vocabulary that transformers consume.

Components:

- **`CADVocabulary`** class:
  - Maps every possible (command_type, parameter_index, quantized_value) triple to a unique integer token ID
  - Special tokens: `<PAD>`, `<BOS>`, `<EOS>`, `<SEP>` (sketch/extrude boundary), `<UNK>`
  - With 6 command types × 16 parameters × 256 levels = ~24,576 possible tokens + specials
  - `encode(command_tokens) → List[int]` — command sequence to integer ID sequence
  - `decode(token_ids) → List[CommandToken]` — integer IDs back to command tokens
  - Serializable to JSON for checkpoint portability

- This is what the `STEPTokenizer` in `ll_stepnet/stepnet/tokenizer.py` currently does for STEP text tokens — the vocabulary builder does the analogous thing for CAD command tokens.

### 1.4 — Update: Config Extensions

**Where**: `geotoken/geotoken/config.py`

**What changes**: Add new config dataclasses:

- **`CommandTokenizationConfig`**:
  - `max_sequence_length: int = 60`
  - `coordinate_quantization: PrecisionTier = STANDARD`
  - `parameter_quantization: PrecisionTier = STANDARD`
  - `normalization_range: float = 2.0` (2×2×2 cube)
  - `canonicalize_loops: bool = True`
  - `include_constraints: bool = False` (SketchGraphs-style)
  - `pad_to_max_length: bool = True`

---

## Phase 2: VAE and Generative Architecture in ll_stepnet

### Why this phase

The encoder and decoder in `ll_stepnet/stepnet/encoder.py` are structurally close to DeepCAD's architecture but missing the generative components — no variational bottleneck, no learned prior, no output prediction heads. This phase adds what's needed to turn the understanding encoder into a generative encoder-decoder.

### 2.1 — New: VAE Wrapper

**Where**: `ll_stepnet/stepnet/vae.py` (new file)

**What it does**: Wraps the existing `STEPTransformerEncoder` and `STEPTransformerDecoder` with a variational autoencoder bottleneck.

Architecture (following DeepCAD):

- **Encoder path**: `STEPTransformerEncoder` → mean pool → `μ_head` (Linear → 256-dim) + `σ_head` (Linear → 256-dim) → reparameterization trick → latent `z`
  - Uses the existing 6-layer, 8-head encoder as-is
  - Adds two new linear projection heads for mean and log-variance

- **Decoder path**: `z` → linear projection to (1, embed_dim) → expand to (seq_len, embed_dim) with learned positional embeddings → `STEPTransformerDecoder` → prediction heads
  - Uses the existing causal decoder as-is
  - Adds output prediction heads (see 2.2)

- **Loss**: reconstruction loss (cross-entropy on predicted command tokens) + KL divergence (β-VAE weighting with β warmup from 0 → 1 over first 10 epochs to prevent posterior collapse)

- **Config**: new `VAEConfig` dataclass in `ll_stepnet/stepnet/config.py`:
  - `latent_dim: int = 256`
  - `kl_weight: float = 1.0`
  - `kl_warmup_epochs: int = 10`
  - `encoder_config: STEPEncoderConfig`
  - `decoder_layers: int = 6` (matches encoder)

**What existing code it modifies**: None directly — it imports and wraps `STEPTransformerEncoder` and `STEPTransformerDecoder` without changing them.

### 2.2 — New: Decoder Output Heads

**Where**: `ll_stepnet/stepnet/output_heads.py` (new file)

**What it does**: Task-specific prediction layers that sit on top of the decoder's hidden states.

Following DeepCAD's separate-head design:

- **`CommandTypeHead`**: Linear(embed_dim, 6) → predicts which command type (SOL, LINE, ARC, CIRCLE, EXTRUDE, EOS)
- **`ParameterHeads`**: one `Linear(embed_dim, num_levels)` per parameter slot (16 slots × 256 levels each) → predicts each quantized parameter independently as classification
- **`CompositeHead`**: combines the above, applies command-type masking (LINE only predicts xy endpoints, CIRCLE only predicts center+radius, etc.)

This is the piece the research document calls out as critical: "DeepCAD has separate linear heads for command type vs parameters; the decoder here has no task-specific prediction heads."

**Integration with existing `tasks.py`**: The existing `STEPForCaptioning`, `STEPForClassification`, etc. in `tasks.py` are the understanding-direction heads. The new output heads are the generation-direction analog. They follow the same pattern (encoder → head) but output CAD commands instead of text/labels.

### 2.3 — New: Latent GAN

**Where**: `ll_stepnet/stepnet/latent_gan.py` (new file)

**What it does**: Trains a WGAN-gp that maps random noise to the VAE's learned latent distribution, enabling unconditional sampling of new CAD models.

Architecture (following DeepCAD exactly):

- **Generator**: `z_noise` (256-dim Gaussian) → 3-layer MLP (256→512→512→256) → `z_fake` in latent space
- **Discriminator**: `z` (256-dim) → 3-layer MLP (256→512→512→1) → real/fake score
- **Training**: WGAN-gp loss with gradient penalty λ=10, 5 critic updates per generator update
- Trains after the VAE is converged, using encoded latents from the training set as "real" samples

**Why it matters**: Without the latent GAN, the VAE's latent space has holes and low-density regions that produce garbage when sampled randomly. The research notes this explicitly: "Without this, the VAE's posterior collapse produces limited variety."

### 2.4 — New: Diffusion Denoiser (BrepGen-style)

**Where**: `ll_stepnet/stepnet/diffusion.py` (new file)

**What it does**: Implements the denoising diffusion framework for structured CAD generation. This is the more advanced alternative to the VAE path.

Components:

- **`DDPMScheduler`**: Linear beta schedule from 1e-4 to 0.02 over 1000 timesteps. Forward process adds Gaussian noise. Implements PNDM sampler for fast 200-step inference.

- **`CADDenoiser`**: Transformer-based denoiser (12 self-attention layers, 12 heads, 1024 hidden dim — matching BrepGen). Takes noisy latent + timestep embedding → predicts noise to subtract. Conditioned on timestep via sinusoidal embedding added to each layer.

- **`StructuredDiffusion`**: Orchestrates sequential denoising across CAD hierarchy:
  1. Denoise face positions (conditioned on nothing)
  2. Denoise face geometry (conditioned on face positions)
  3. Denoise edge positions (conditioned on faces)
  4. Denoise edge-vertex geometry (conditioned on edges + faces)
  
  Each stage has its own `CADDenoiser` instance.

- **Config**: `DiffusionConfig` in `config.py`:
  - `num_timesteps: int = 1000`
  - `beta_start: float = 1e-4`
  - `beta_end: float = 0.02`
  - `inference_steps: int = 200`
  - `denoiser_layers: int = 12`
  - `denoiser_heads: int = 12`
  - `denoiser_hidden_dim: int = 1024`

**Relationship to VAE**: These are two alternative generation strategies. The VAE path (2.1-2.3) is simpler and faster to implement. The diffusion path (2.4) produces higher quality results but is more complex. Both share the same tokenization (Phase 1) and output reconstruction (Phase 5) infrastructure.

### 2.5 — New: Cross-Attention Conditioning Module

**Where**: `ll_stepnet/stepnet/conditioning.py` (new file)

**What it does**: Adds the ability to condition generation on external embeddings (text, images). This is the "biggest gap" identified in the comparison.

Architecture (following Text2CAD):

- **`TextConditioner`**:
  - Wraps a frozen BERT/CLIP text encoder (loaded from HuggingFace)
  - `AdaptiveLayer`: single transformer encoder block that refines text embeddings for the CAD domain
  - Injects into the decoder via cross-attention: CAD token queries attend to text embedding keys/values
  - First 2 decoder blocks skip cross-attention (Text2CAD's design — lets initial structure form before conditioning)

- **`ImageConditioner`**:
  - Wraps a frozen DINOv2 or CLIP vision encoder
  - Produces visual embeddings that enter the same cross-attention pathway
  - For multi-view: averages embeddings from N views before conditioning

- **`MultiModalConditioner`**:
  - Combines text and image conditioners
  - Concatenates or attends to both embedding sequences

**What existing code it modifies**:

- `STEPTransformerDecoder` in `encoder.py` needs an update — currently it uses `nn.TransformerDecoder` with `memory=x` (self-attention only). The update adds a `cross_attention_memory` parameter so the decoder can attend to external conditioning embeddings when provided. When `cross_attention_memory=None`, behavior is unchanged (backward compatible).

### 2.6 — Update: Config Extensions

**Where**: `ll_stepnet/stepnet/config.py`

New config dataclasses to add:

- `VAEConfig` (for 2.1)
- `LatentGANConfig` (for 2.3)  
- `DiffusionConfig` (for 2.4)
- `ConditioningConfig` (for 2.5):
  - `text_encoder_name: str = "bert-base-uncased"`
  - `image_encoder_name: str = "facebook/dinov2-base"`
  - `conditioning_dim: int = 1024`
  - `skip_cross_attention_blocks: int = 2`
  - `freeze_encoder: bool = True`

---

## Phase 3: VQ-VAE and Codebook Module

### Why this is separate

SkexGen's disentangled codebook approach is architecturally different from the standard VAE (Phase 2). It uses vector quantization to compress CAD models into very compact discrete codes (just 10 per model). This enables a different kind of controllable generation — swap topology codes to change structure while keeping geometry style.

### 3.1 — New: Vector Quantizer

**Where**: `ll_stepnet/stepnet/vqvae.py` (new file)

Components:

- **`VectorQuantizer`**: Core VQ layer
  - Codebook of K learnable embedding vectors (e.g., K=500 for topology, K=1000 for geometry)
  - Input continuous vector → find nearest codebook entry → output discrete code + quantized vector
  - Training: exponential moving average codebook updates (decay=0.99), matching SkexGen
  - Commitment loss with β=0.25
  - Skip quantization for first 25 epochs (SkexGen's stabilization trick)

- **`DisentangledCodebooks`**: SkexGen's 3-codebook system
  - Topology codebook (500 codes): encodes curve type sequences
  - Geometry codebook (1000 codes): encodes 2D point positions on 64×64 grid
  - Extrusion codebook (1000 codes): encodes 3D extrusion operations
  - Each CAD model → 10 total codes (split across codebooks)

- **`CodebookDecoder`**: One transformer decoder per codebook (4 layers, 8 heads each), generates codes autoregressively within each codebook

**Relationship to Phase 2**: The VQ-VAE replaces the continuous VAE bottleneck (2.1) with discrete codebooks. It's an alternative, not an addition — you'd choose one or the other for a given generation approach.

---

## Phase 4: GNN Upgrades for Generation-Quality Embeddings

### Why this matters

The existing GAT in `cadling/models/segmentation/architectures/gat_net.py` and the `STEPGraphEncoder` in `ll_stepnet` are designed for understanding tasks (segmentation, classification). Generation requires richer per-face representations — specifically UV-Net's approach of learning from the actual surface geometry, not just extracted features.

### 4.1 — New: UV-Net Face Encoder

**Where**: `cadling/models/segmentation/architectures/uv_net.py` (new file)

**What it does**: Implements UV-Net's core innovation — sampling each B-Rep face on a UV parameter grid and processing through a surface CNN.

Architecture:

- **UV Sampling**: For each face in the B-Rep, sample a 10×10 grid in UV parameter space. At each sample point, compute 7 channels: xyz position (3), surface normal (3), trim mask (1, whether the point is inside the face boundary).
  - Uses pythonocc's `BRepAdaptor_Surface` to evaluate surfaces at UV points
  - This is where `cadling/backend/step/` pythonocc integration comes in

- **Surface CNN**: Per-face 2D convolution network processing the 10×10×7 grid
  - 3 conv layers (7→32→64→64), kernel size 3, batch norm, ReLU
  - Global average pool → 64-dim face embedding

- **Graph Message Passing**: Feed face embeddings into the existing `GraphAttentionEncoder` from `gat_net.py`
  - Node features = UV-Net face embeddings (64-dim) instead of manually extracted features
  - Edge features = dihedral angles (from existing `brep_graph_builder.py`)
  - Output: 128-dim per-face embeddings + graph-level embedding via attention pooling

**What existing code it uses**:

- `BRepFaceGraphBuilder` in `brep_graph_builder.py` — already builds the face adjacency graph
- `GraphAttentionEncoder` in `gat_net.py` — already implements multi-head GAT
- pythonocc integration via `cadling/backend/step/` — already has surface evaluation

### 4.2 — New: Coedge Convolution (BRepNet-style)

**Where**: `cadling/models/segmentation/architectures/brep_net.py` (new file)

**What it does**: Implements BRepNet's oriented coedge convolution — the most topologically-aware GNN for B-Rep data.

Architecture:

- **Coedge data structure**: Each edge in a B-Rep has two oriented coedges (one per adjacent face). Each coedge maintains pointers to:
  - `next`: next coedge in the same face loop
  - `prev`: previous coedge in the same face loop  
  - `mate`: the opposing coedge on the adjacent face

- **Coedge convolution layer**: At each step, a coedge aggregates features from its `next`, `prev`, and `mate` neighbors via learned linear transformations + activation. This enables "walks" around face boundaries and across face boundaries.

- **Graph construction**: Builds the coedge adjacency from STEP topology
  - Extends `STEPTopologyBuilder` in `ll_stepnet/stepnet/topology.py` with coedge pointers
  - Also extends `BRepFaceGraphBuilder` to optionally output coedge-level graphs

**Why this matters for generation**: Coedge convolution captures manufacturing-relevant patterns (the sequence of surfaces around a hole, the transition between fillets and planar faces) that face-level message passing misses. These patterns are exactly what the generation model needs to learn.

### 4.3 — Update: Graph Builder Extensions

**Where**: `cadling/models/segmentation/brep_graph_builder.py`

**What changes**: Add methods to support the new architectures:

- `build_uv_sampled_graph()` — returns graph with UV-grid face features instead of extracted features. Uses pythonocc to evaluate surfaces at UV parameter points.
- `build_coedge_graph()` — returns coedge-level graph with next/prev/mate pointers
- Existing `build_face_graph()` stays unchanged

---

## Phase 5: Output Reconstruction Pipeline

### Why this is critical

The research is clear: "Converting neural outputs to valid solids remains the hardest problem." Validity rates top out around 63%. This phase builds the pipeline that takes raw neural predictions and produces valid STEP files.

### 5.1 — New: Command Sequence Executor

**Where**: `cadling/generation/reconstruction/command_executor.py` (new directory + file)

**What it does**: Takes a predicted command token sequence and executes it in OpenCASCADE to produce B-Rep geometry.

Pipeline:

1. **Decode tokens** → CommandToken sequence via the vocabulary from Phase 1
2. **Dequantize parameters** → continuous values using stored normalization metadata
3. **Build sketches** → for each sketch group (SOL...LINE/ARC/CIRCLE...):
   - Create 2D wire from primitives using pythonocc `BRepBuilderAPI_MakeWire`
   - Close wire into face using `BRepBuilderAPI_MakeFace`
4. **Apply extrusions** → for each EXTRUDE command:
   - `BRepPrimAPI_MakePrism` for linear extrusion
   - Apply boolean operation (union/intersection/subtraction) with existing body
5. **Validate result** → feed through the existing `TopologyValidationModel` from `cadling/models/topology_validation.py`
6. **Export** → write to STEP via pythonocc `STEPControl_Writer`

**What existing code it uses**:

- `geotoken` vocabulary and dequantization (Phase 1)
- `cadling/backend/step/` pythonocc integration
- `TopologyValidationModel` for validity checking
- `cadling/backend/pythonocc_core_backend.py` for OCC utilities

### 5.2 — New: B-Spline Surface Fitter

**Where**: `cadling/generation/reconstruction/surface_fitter.py` (new file)

**What it does**: For diffusion-based generation (BrepGen path), converts denoised point grids into B-spline surfaces that OpenCASCADE can work with.

Pipeline:

1. **Receive decoded point grids** — face VAE outputs 32×32×3 point arrays
2. **Fit B-spline surface** — `GeomAPI_PointsToBSplineSurface` from OCC
3. **Trim surface** — use edge curves to trim the surface to the face boundary
4. **Quality check** — compute Chamfer distance between fitted surface and original points

### 5.3 — New: Topology Recovery via Node Merging

**Where**: `cadling/generation/reconstruction/topology_merger.py` (new file)

**What it does**: Implements BrepGen's mating duplication recovery — detecting which separately-generated face/edge nodes should actually be merged to form topological adjacency.

Algorithm:

1. For each pair of edge nodes across different faces:
   - Compute bounding box distance
   - Compute shape feature similarity (Chamfer distance between edge curves)
2. If bbox distance < 0.08 AND shape similarity < 0.2 → merge
3. Average vertex positions across merged duplicates
4. Scale/translate edges to match merged vertex positions
5. Adjust face surfaces to minimize Chamfer distance to connected edges
6. Sew trimmed faces into closed solid via `BRepBuilderAPI_Sewing`

**Thresholds are from BrepGen's ablation studies** — the research found optimal at 0.06-0.1 for bbox and ~0.2 for shape.

### 5.4 — New: Constraint Solver

**Where**: `cadling/generation/reconstruction/constraint_solver.py` (new file)

**What it does**: Takes generated 2D sketches and enforces geometric constraints (parallelism, perpendicularity, tangency, coincidence) that the neural network approximately predicted.

Approach:

1. **Detect near-constraints** — from the generated sketch, identify pairs of primitives that are *almost* parallel, *almost* tangent, etc. (within configurable tolerance)
2. **Build constraint system** — each constraint becomes an equation: `parallel(line_i, line_j)` means `angle(line_i, line_j) = 0`
3. **Newton's method solver** — iteratively adjust primitive parameters to satisfy all constraints simultaneously. Convergence = valid sketch. Failure to converge = inconsistent geometry (reject).

**What existing code it connects to**:

- `cadling/experimental/models/geometric_constraint_model.py` — already defines `ConstraintType` enum and constraint detection logic. The solver adds the *satisfaction* side.
- `cadling/models/constraint_detection.py` — assembly-level constraint detection (different scope but shared types)

### 5.5 — New: Validation Feedback Loop

**Where**: `cadling/generation/reconstruction/validation_loop.py` (new file)

**What it does**: Runs the full reconstruction pipeline and feeds validation results back to the generator for iterative refinement.

Loop:

1. Generate command sequence (or denoised latents)
2. Reconstruct solid via executor (5.1) or surface fitter (5.2) + merger (5.3)
3. Validate via `TopologyValidationModel`
4. If valid → accept
5. If invalid → report specific failures (`ValidationFinding` with entity IDs) → feed back to generator as negative reward signal

The RL alignment approach (achieving 93% fully-constrained sketches in the research) would use this loop to train the generator with validation feedback as reward.

---

## Phase 6: Code Generation Backend (Fastest Path to Text-to-CAD)

### Why this is strategically important

The research shows GPT-4 achieves 96.5% compile rate on CadQuery code. This approach completely sidesteps the topology gap — the CAD kernel handles all validity, and the LLM just needs to write valid code. This is the fastest practical path to text→CAD generation.

### 6.1 — New: CadQuery Code Generator

**Where**: `cadling/generation/codegen/cadquery_generator.py` (new directory + file)

**What it does**: Uses LLMs to generate CadQuery Python scripts from text descriptions, then executes them to produce STEP files.

Components:

- **`CadQueryGenerator`**:
  - Uses the existing `ChatAgent` from `cadling/sdg/qa/utils.py` for LLM API calls
  - System prompt with CadQuery API reference, examples, and constraints
  - Input: text description + optional reference images
  - Output: CadQuery Python script string

- **`CadQueryExecutor`**:
  - Sandboxed execution of generated CadQuery scripts
  - Captures the resulting `cadquery.Workplane` object
  - Exports to STEP via `cq.exporters.export`
  - Error capture: syntax errors, runtime errors, OCC exceptions

- **`CadQueryValidator`**:
  - Runs the generated STEP through `TopologyValidationModel`
  - Compile rate tracking (percentage of scripts that produce valid solids)
  - If failed: feeds error message back to LLM for retry (up to N attempts)

**What existing code it uses**:

- `ChatAgent` from `cadling/sdg/qa/utils.py` — already supports OpenAI, Anthropic, vLLM, Ollama
- `TopologyValidationModel` for output validation
- STEP backend for parsing/validating the output file

### 6.2 — New: OpenSCAD Code Generator

**Where**: `cadling/generation/codegen/openscad_generator.py` (new file)

**What it does**: Same approach as 6.1 but targeting OpenSCAD, which has simpler syntax that LLMs handle well.

- OpenSCAD scripts → execute via `openscad` CLI → produces STL
- STL → convert to STEP via pythonocc if parametric output needed

### 6.3 — New: Code Generation Prompt Library

**Where**: `cadling/generation/codegen/prompts/` (new directory)

Prompt templates:

- `cadquery_system.txt` — system prompt with CadQuery API reference
- `openscad_system.txt` — system prompt with OpenSCAD syntax reference
- `text_to_code.txt` — text description → code generation template
- `image_to_code.txt` — image + description → code generation template
- `repair.txt` — error message → code fix template

### 6.4 — New: Generation CLI

**Where**: `cadling/cli/generate.py` (new file, extending existing CLI)

New CLI commands:

```bash
cadling generate --from-text "a mounting bracket with 4 bolt holes" --output bracket.step
cadling generate --from-text "..." --backend cadquery|openscad --output part.step
cadling generate --from-image render.png --output part.step
cadling generate --from-text "..." --validate --max-retries 3
```

Integrates with the existing Click-based CLI in `cadling/cli/cli.py`.

---

## Phase 7: Research Dataset Loaders

### Why this matters

The SDG pipeline generates new Q&A pairs from CAD files, but the generation models need to train on the established research datasets (DeepCAD's 178K models, ABC's 1M models, etc.). Without dataset loaders, there's no training data for the VAE, diffusion, or VQ-VAE models.

### 7.1 — New: DeepCAD Dataset Loader

**Where**: `cadling/data/datasets/deepcad_loader.py` (new directory + file)

**What it does**: Loads and preprocesses DeepCAD's 178K sketch-and-extrude sequences.

- Downloads/caches from the DeepCAD repo
- Parses JSON command sequences into `CommandToken` sequences (Phase 1)
- Applies normalization, quantization, and padding
- Returns PyTorch Dataset/DataLoader compatible with `STEPTrainer`
- Train/val/test splits matching the paper's setup

### 7.2 — New: ABC Dataset Loader

**Where**: `cadling/data/datasets/abc_loader.py` (new file)

**What it does**: Loads ABC dataset STEP files for B-Rep tasks.

- Streams from HuggingFace or local cache
- Parses STEP files via the existing STEP backend
- Builds face adjacency graphs via `BRepFaceGraphBuilder`
- Filters invalid/trivial models

**Connects to**: existing `BaseDataLoader` in `cadling/models/segmentation/training/data_loaders.py` — follows the same HuggingFace streaming pattern.

### 7.3 — New: Text2CAD Dataset Loader

**Where**: `cadling/data/datasets/text2cad_loader.py` (new file)

**What it does**: Loads Text2CAD's 660K text-annotated CAD models.

- Multi-level annotations: abstract, intermediate, detailed, expert
- Pairs text descriptions with command sequences
- Returns (text, command_tokens) pairs for training text-conditioned generation

### 7.4 — New: SketchGraphs Loader

**Where**: `cadling/data/datasets/sketchgraphs_loader.py` (new file)

**What it does**: Loads SketchGraphs' 15M parametric sketches as constraint graphs.

- Nodes = sketch primitives (lines, arcs, circles)
- Edges = designer-imposed constraints (coincident, tangent, perpendicular, etc.)
- Returns PyTorch Geometric `Data` objects compatible with the GNN architectures

### 7.5 — Update: Dataset Registry

**Where**: `cadling/data/__init__.py` (new)

Central registry:

```python
DATASETS = {
    "deepcad": DeepCADLoader,
    "abc": ABCLoader, 
    "text2cad": Text2CADLoader,
    "sketchgraphs": SketchGraphsLoader,
    "mfcad": MFCADLoader,  # already exists in dataset_builders/
}
```

---

## Phase 8: Training Infrastructure for Generation

### Why this is separate

The existing `STEPTrainer` in `ll_stepnet/stepnet/trainer.py` handles basic train/eval/checkpoint loops. Generation training requires additional infrastructure: VAE-specific losses, GAN training loops, diffusion noise scheduling, and generation-quality evaluation metrics.

### 8.1 — New: VAE Trainer

**Where**: `ll_stepnet/stepnet/training/vae_trainer.py` (new directory + file)

Extends `STEPTrainer` with:

- Reconstruction loss (cross-entropy on command tokens) + KL divergence
- β-VAE warmup scheduling
- Latent space visualization (t-SNE/UMAP of encoded latents per epoch)
- Reconstruction quality metrics: exact command match rate, parameter MSE

### 8.2 — New: GAN Trainer

**Where**: `ll_stepnet/stepnet/training/gan_trainer.py` (new file)

- Alternating generator/discriminator training
- WGAN-gp loss with gradient penalty
- FID-style metrics for latent distributions

### 8.3 — New: Diffusion Trainer

**Where**: `ll_stepnet/stepnet/training/diffusion_trainer.py` (new file)

- Random timestep sampling per batch
- Noise prediction loss (MSE between predicted and actual noise)
- EMA model averaging for stable generation
- Sampling visualization during training

### 8.4 — New: Generation Evaluation Metrics

**Where**: `cadling/evaluation/generation_metrics.py` (new directory + file)

Metrics from the research:

- **Validity rate**: percentage of generated outputs that produce watertight solids
- **Coverage (COV)**: fraction of test set shapes matched by a generated shape (Chamfer distance threshold)
- **Minimum Matching Distance (MMD)**: average distance from each test shape to nearest generated shape
- **Jensen-Shannon Divergence (JSD)**: distribution similarity between generated and test sets
- **Novelty**: fraction of generated shapes not close to any training example
- **Compile rate**: for code generation path, percentage of scripts that execute without error

---

## Phase 9: Text-CAD Paired Data Generation

### Why this extends the existing SDG

The SDG pipeline in `cadling/sdg/` generates Q&A pairs — questions about existing CAD parts. For text→CAD training, you need the reverse: text descriptions paired with the command sequences that produce those parts. This extends SDG from understanding-direction to generation-direction data.

### 9.1 — Update: Multi-Level Text Annotator

**Where**: `cadling/sdg/qa/text_cad_annotator.py` (new file)

**What it does**: Generates Text2CAD-style multi-level annotations for CAD models.

Following Text2CAD's pipeline:

1. Render multi-view images of the CAD model (4 views)
2. Feed to VLM (using existing `VlmModel` from `cadling/models/vlm_model.py`) → shape description
3. Feed description to LLM (using existing `ChatAgent`) → generate 4 annotation levels:
   - **Abstract**: "two concentric cylinders"
   - **Intermediate**: "a hollow tube with outer diameter 50mm"
   - **Detailed**: "cylindrical shell, OD=50mm, ID=40mm, height=100mm, 4 mounting holes"
   - **Expert**: full coordinate + parameter specifications

Output: `(text_level_1, text_level_2, text_level_3, text_level_4, command_sequence)` tuples

### 9.2 — New: Sequence Annotation Pipeline

**Where**: `cadling/sdg/qa/sequence_annotator.py` (new file)

**What it does**: For models where construction history is available, generates paired (text, command_sequence) training examples.

- Input: STEP file with parametric history (from OnShape or Fusion 360)
- Extract command sequence via Phase 1 tokenizer
- Generate text annotations via 9.1
- Output formatted training pairs

---

## Phase 10: Integration — The Generation Pipeline

### Where everything comes together

This phase wires all the previous phases into a unified generation pipeline that follows cadling's existing architectural patterns.

### 10.1 — New: Generation Pipeline

**Where**: `cadling/generation/pipeline.py` (new file)

**What it does**: Orchestrates the full text→CAD generation flow, following the same Build→Assemble→Enrich pattern as cadling's understanding pipelines.

```
Input (text/image prompt)
  → Conditioning (Phase 2.5 — encode text/image)
  → Generation (Phase 2 — VAE/diffusion decode to command tokens)
  → Reconstruction (Phase 5 — execute commands in OCC)
  → Validation (existing TopologyValidationModel)
  → Output: STEP file + validity report
```

### 10.2 — New: Generation Data Models

**Where**: `cadling/datamodel/generation.py` (new file)

Pydantic models for:

- `GenerationRequest`: text prompt, image path, generation config
- `GenerationResult`: STEP file path, validity report, command sequence, generation metadata
- `GenerationConfig`: which backend (vae/diffusion/codegen), num samples, temperature, etc.

### 10.3 — New: Package Structure

New directories to create:

```
cadling/
  generation/                  # NEW — all generation-direction code
    __init__.py
    pipeline.py                # 10.1 — main generation pipeline
    codegen/                   # Phase 6 — code generation
      __init__.py
      cadquery_generator.py
      openscad_generator.py
      prompts/
    reconstruction/            # Phase 5 — output reconstruction
      __init__.py
      command_executor.py
      surface_fitter.py
      topology_merger.py
      constraint_solver.py
      validation_loop.py
  data/                        # Phase 7 — dataset loaders
    __init__.py
    datasets/
      deepcad_loader.py
      abc_loader.py
      text2cad_loader.py
      sketchgraphs_loader.py
  evaluation/                  # Phase 8.4 — generation metrics
    __init__.py
    generation_metrics.py

geotoken/geotoken/
  tokenizer/
    command_tokenizer.py       # 1.2 — command sequence tokenizer
    vocabulary.py              # 1.3 — vocabulary builder
    token_types.py             # 1.1 — updated with new token types

ll_stepnet/stepnet/
  vae.py                       # 2.1 — VAE wrapper
  output_heads.py              # 2.2 — decoder output heads
  latent_gan.py                # 2.3 — latent GAN
  diffusion.py                 # 2.4 — diffusion denoiser
  conditioning.py              # 2.5 — cross-attention conditioning
  vqvae.py                     # 3.1 — vector quantization
  training/                    # Phase 8 — generation training
    __init__.py
    vae_trainer.py
    gan_trainer.py
    diffusion_trainer.py

cadling/models/segmentation/architectures/
  uv_net.py                    # 4.1 — UV-Net face encoder
  brep_net.py                  # 4.2 — BRepNet coedge convolution
```

---

## Dependency Order

The phases have clear dependencies. Here's the build order:

```
Phase 1 (Tokenization)     ← Foundation, no dependencies
    ↓
Phase 7 (Dataset Loaders)  ← Needs Phase 1 for tokenizing datasets
    ↓
Phase 2 (VAE/Diffusion)    ← Needs Phase 1 tokens as input
Phase 3 (VQ-VAE)           ← Alternative to Phase 2, same dependency
    ↓
Phase 4 (GNN Upgrades)     ← Independent, but feeds into Phase 2 conditioning
    ↓
Phase 5 (Reconstruction)   ← Needs Phase 1 (decode tokens) + existing pythonocc
    ↓
Phase 8 (Training)          ← Needs Phases 1, 2, 5, 7
    ↓
Phase 6 (Code Generation)  ← Independent fast-track, needs only existing SDG infra
    ↓
Phase 9 (Text-CAD SDG)     ← Needs Phase 1 + existing SDG
    ↓
Phase 10 (Integration)     ← Needs everything above
```

### Recommended implementation sequence

1. **Phase 1** → Tokenizer extensions (everything else depends on this)
2. **Phase 6** → Code generation backend (fastest path to working text→CAD)
3. **Phase 5** → Reconstruction pipeline (needed for neural generation AND to validate codegen output)
4. **Phase 7** → Dataset loaders (needed before training)
5. **Phase 2.1-2.2** → VAE + output heads (first neural generation)
6. **Phase 8.1** → VAE trainer (train the model)
7. **Phase 2.5** → Cross-attention conditioning (add text/image input)
8. **Phase 9** → Text-CAD data generation (training data for conditioned generation)
9. **Phase 4** → GNN upgrades (improve embedding quality)
10. **Phase 2.3** → Latent GAN (improve unconditional sampling)
11. **Phase 2.4** → Diffusion (higher quality generation)
12. **Phase 3** → VQ-VAE (controllable generation)
13. **Phase 10** → Full integration pipeline

---

## Files Modified (Existing)

| File | Change | Phase |
|------|--------|-------|
| `geotoken/geotoken/tokenizer/token_types.py` | Add `CommandToken`, `ConstraintToken`, `BooleanOpToken` | 1.1 |
| `geotoken/geotoken/config.py` | Add `CommandTokenizationConfig` | 1.4 |
| `ll_stepnet/stepnet/encoder.py` | Add `cross_attention_memory` param to decoder | 2.5 |
| `ll_stepnet/stepnet/config.py` | Add `VAEConfig`, `DiffusionConfig`, `ConditioningConfig`, etc. | 2.6 |
| `ll_stepnet/stepnet/topology.py` | Add coedge pointer construction | 4.2 |
| `cadling/models/segmentation/brep_graph_builder.py` | Add `build_uv_sampled_graph()`, `build_coedge_graph()` | 4.3 |
| `cadling/cli/cli.py` | Add `generate` command group | 6.4 |

## Files Created (New)

| Count | Location | Phase |
|-------|----------|-------|
| 2 | `geotoken/geotoken/tokenizer/` | 1.2, 1.3 |
| 5 | `ll_stepnet/stepnet/` | 2.1-2.5 |
| 1 | `ll_stepnet/stepnet/` | 3.1 |
| 3 | `ll_stepnet/stepnet/training/` | 8.1-8.3 |
| 2 | `cadling/models/segmentation/architectures/` | 4.1, 4.2 |
| 5 | `cadling/generation/reconstruction/` | 5.1-5.5 |
| 5 | `cadling/generation/codegen/` | 6.1-6.3 |
| 1 | `cadling/generation/` | 10.1 |
| 1 | `cadling/cli/` | 6.4 |
| 4 | `cadling/data/datasets/` | 7.1-7.4 |
| 1 | `cadling/evaluation/` | 8.4 |
| 1 | `cadling/datamodel/` | 10.2 |
| 2 | `cadling/sdg/qa/` | 9.1-9.2 |
| **33** | **total new files** | |
