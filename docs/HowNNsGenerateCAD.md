# How Neural Networks Generate CAD: A Mechanistic Deep Dive

Modern CAD generation models transform natural language or images into valid 3D solid geometry through a sophisticated pipeline that bridges continuous neural predictions with the discrete, exact world of parametric CAD. The core challenge is fundamental: neural networks output soft probability distributions over continuous spaces, while CAD requires exact integer topologies (face counts, edge connectivity) and precise geometric parameters. The field has converged on two dominant strategies—treating CAD construction as language modeling with discrete tokens, or using latent diffusion over structured geometric representations—with validity rates now reaching **63%** for complex B-rep generation, up from under 50% just two years ago.

## The tokenization problem shapes everything downstream

The choice of CAD representation fundamentally constrains what neural networks can learn and generate. Three major paradigms have emerged, each with distinct trade-offs between expressiveness, generation quality, and validity guarantees.

**Sketch-and-extrude sequences** dominate the research landscape, pioneered by DeepCAD in 2021. This approach mirrors how engineers actually design: draw 2D profiles, then extrude them into 3D solids. DeepCAD's vocabulary contains just six command types—start-of-loop, line, arc, circle, extrude, and end-of-sequence—plus **16 continuous parameters per command** covering coordinates, angles, extrusion distances, and boolean operations. All continuous values are quantized to 8-bit integers (256 levels) after normalizing solids to a 2×2×2 cube. This quantization is not merely a convenience—experiments showed that direct regression fails catastrophically because small MSE errors break critical geometric relationships like parallel and perpendicular lines, while discrete classification preserves learned constraints.

SkexGen advanced this representation by **disentangling topology from geometry**. Instead of interleaving curve types with coordinates, SkexGen maintains separate codebooks: 500 topology codes capturing curve type sequences, 1000 geometry codes encoding 2D point positions on a 64×64 grid, and 1000 extrusion codes for 3D operations. Each CAD model compresses to just 10 discrete codes. This separation enables controllable generation—changing topology codes produces structurally different parts while preserving geometric style.

**B-rep (Boundary Representation) tokenization** directly encodes the final solid geometry: faces, edges, and vertices with their topological connectivity. BrepGen, published at SIGGRAPH 2024, represents B-rep as a hierarchical tree where each face node contains a **48-dimensional latent code** (compressed from 32×32×3 UV-sampled surface points via VAE) plus a 6D bounding box. Edge nodes similarly compress curve geometry to 12-dimensional latents. The critical innovation is *mating duplication*: when two faces share an edge, that edge appears as a child node under both faces with identical features. This implicitly encodes topology through geometry similarity—during post-processing, nodes with similar features (bounding box distance < 0.08, shape threshold < 0.2) are merged to recover the true topological structure.

**CSG (Constructive Solid Geometry) trees** represent shapes as boolean operation hierarchies over geometric primitives. CSGNet encodes primitives as location/size parameters discretized to 8-unit spacing, with union, intersection, and subtraction operations forming the tree structure. CSG-Stump simplifies this to a fixed three-layer architecture: complement, intersection, then union operations, with binary connection matrices learned during training. While elegant, CSG struggles with organic shapes and complex topology.

## Autoregressive transformers dominate but diffusion is closing the gap

The architecture landscape splits between sequence-modeling approaches treating CAD as a language problem and structured generative models designed for geometric data.

**DeepCAD established the transformer-VAE paradigm** with a straightforward encoder-decoder architecture: both encoder and decoder use 4-layer transformer blocks with 8 attention heads and 512 feed-forward dimensions, compressing sequences to a 256-dimensional latent space. The decoder generates commands autoregressively, predicting command types and parameters through separate linear heads. A critical addition is the *latent GAN*—a WGAN-gp trained to map random noise to the VAE's latent distribution, enabling unconditional sampling. Without this, the VAE's posterior collapse produces limited variety.

**SkexGen splits generation across three parallel decoders**, one per codebook. Each is a standard transformer (4 layers, 8 heads) that generates codes autoregressively. The vector quantization uses exponential moving average codebook updates with decay 0.99, skipping quantization for the first 25 epochs to stabilize training. The commitment loss coefficient β=0.25 prevents encoders from straying too far from codebook vectors while avoiding posterior collapse.

**BrepGen's diffusion architecture** represents the frontier of structured CAD generation. Four separate transformer-based denoisers generate components sequentially: face positions, face latent geometry (conditioned on positions), edge positions (conditioned on faces), and edge-vertex geometry (conditioned on edges and faces). Each denoiser uses **12 self-attention layers with 12 heads and 1024 hidden dimension**—significantly larger than autoregressive approaches. The diffusion process runs 1000 steps during training with a linear beta schedule from 1e-4 to 0.02, using PNDM with 200 steps for fast inference. Generation takes 5-10 seconds per model on an RTX A5000.

**Graph neural networks** excel at B-rep *encoding* rather than generation. UV-Net processes faces as 10×10 UV-grids through surface CNNs (7 input channels: xyz, normals, trim mask), then aggregates via graph message passing over the face-adjacency graph. BRepNet innovates with *coedge convolution*—oriented edges maintain pointers to next/previous coedges in parent faces and mating coedges across face boundaries, enabling walks that capture manufacturing-relevant topological patterns. These networks produce 64-128 dimensional shape embeddings used for retrieval, segmentation, and conditioning generation models.

## Training requires reconstructing parametric history from billions of shapes

The field's progress depends critically on datasets capturing not just final geometry but the construction sequences that created it.

**ABC Dataset** provides scale with 1 million CAD models from OnShape public documents, but only final STEP geometry—no parametric history. Researchers must query OnShape's API to recover construction sequences, a process limited by sparse documentation. The dataset is uncurated, containing trivial and invalid models alongside complex designs, biased toward mechanical parts with sharp features.

**DeepCAD Dataset** fills the history gap with **178,238 models** including full sketch-and-extrude sequences parsed from OnShape. Each model is stored as JSON command sequences plus pre-vectorized representations for fast loading. Sequences are normalized to fixed length (60 commands), with all parameters quantized. Despite its size, significant duplication exists, and the distribution skews toward rectangular and cylindrical shapes.

**Fusion 360 Gallery** offers the highest-quality human design sequences—8,625 carefully filtered models where reconstruction has been validated against originals. The dataset includes assembly relationships (154,468 parts across 8,251 assemblies) with joint connectivity. Operations beyond sketch-and-extrude are suppressed to expand usable data volume, though this limits what models can learn.

**SketchGraphs** provides the largest 2D sketch corpus: **15 million parametric sketches** represented as constraint graphs where nodes are primitives and edges are designer-imposed constraints (coincidence, tangency, perpendicularity). The augmented CPTSketchGraphs expands this to 80 million through constraint-preserving transformations.

**Text2CAD Dataset** (2024) extends DeepCAD with ~660,000 text annotations at four complexity levels, from abstract descriptions ("two concentric cylinders") to expert specifications with precise coordinates. Annotations were generated via a two-stage pipeline: LLaVA-NeXT produces shape descriptions from multi-view renders, then Mixtral-50B generates multi-level prompts. The full dataset occupies 605GB including RGB and depth images.

Data preparation involves normalizing sketches to 2×2 squares (3D to 2×2×2 cubes), canonicalizing loop orderings (counter-clockwise from bottom-left), and filtering duplicates via geometry signatures. Most approaches quantize coordinates to 64-256 discrete levels, trading precision for tractable vocabulary sizes.

## Text and image conditioning bridge natural language to geometric parameters

The conditioning mechanisms for text-to-CAD span from simple cross-attention to sophisticated multi-modal LLM alignment.

**Text2CAD's architecture** (NeurIPS 2024 Spotlight) exemplifies the cross-attention approach. A frozen pre-trained BERT encoder processes text into 1024-dimensional contextual embeddings. An *adaptive layer*—a single transformer encoder block—refines these embeddings for the CAD domain. The CAD decoder contains 8 transformer blocks, each performing self-attention on CAD tokens followed by cross-attention where CAD queries attend to text keys and values. The first two decoder blocks skip cross-attention to allow initial CAD structure formation before conditioning kicks in. Training uses teacher forcing for 160 epochs on A100-80GB GPUs.

**Zoo.dev's Text-to-CAD** takes a fundamentally different approach: their ML model generates **KCL (KittyCAD Language) code** rather than raw geometry. KCL is a domain-specific language designed for LLM friendliness—text-based with built-in geometry primitives and plain-English units. The text-to-KCL model is trained on proprietary feature-tree data extracted from CAD files. Because KCL is executable code, outputs can be edited, parameterized, and validated through their geometry engine. Enterprise customers can fine-tune on converted NX, Creo, or SolidWorks files.

**LLM-based code generation** represents a parallel paradigm where GPT-4, Claude, or specialized code models generate OpenSCAD or CadQuery scripts. GPT-4 achieves **96.5% compile rate** on the CADPrompt benchmark, though geometric accuracy is harder to ensure. The AIDL language from MIT CSAIL improves LLM performance through implicit geometry references and declarative constraints that external solvers satisfy—LLMs perform better with AIDL than OpenSCAD despite zero training data for the new language.

**Image-to-CAD conditioning** typically uses DINOv2 or CLIP encoders to produce visual embeddings that condition generation. Img2CAD employs a two-stage factorization: a finetuned Llama3.2 VLM predicts discrete structural decisions (base shape, semantic parts), then a transformer predicts continuous geometric attributes conditioned on structure. MV2Cyl reconstructs 3D extrusion cylinders from multi-view images through joint curve and surface segmentation—2D U-Net predictions provide extrusion axes and sketch geometry.

## Converting neural outputs to valid solids remains the hardest problem

The reconstruction pipeline transforms predicted tokens or latent codes into manufacturable STEP files through geometric kernels, constraint solving, and extensive post-processing.

**For sequence-based methods**, predicted command sequences are executed directly in CAD kernels. DeepCAD's predicted tokens decode to sketch primitives (lines, arcs, circles with quantized coordinates) and extrusion parameters, which OpenCASCADE executes to produce B-rep geometry. The quantization from 256 levels limits precision but enables exact command reproduction.

**For B-rep diffusion**, the pipeline is more complex. BrepGen's denoised latent codes are decoded by shape VAEs into 32×32×3 point grids for faces and N×3 point arrays for edges. OpenCASCADE's `GeomAPI_PointsToBSplineSurface` fits B-spline surfaces to these point samples. The critical post-processing step merges duplicate nodes: faces that shared edges in the original topology have children with identical features, detected via thresholding and merged to reconstruct adjacency relationships. Vertices are averaged across duplicates, edges are scaled/translated to match endpoint vertices, and faces are adjusted to minimize Chamfer distance to connected edges. Finally, B-spline sewing joins trimmed faces into closed solids.

**Constraint solving** addresses 2D sketch validity. Generated sketches often violate geometric constraints (tangency, perpendicularity, coincidence) that would be enforced during manual design. SketchGen-style approaches that generate constraints alongside primitives, combined with reinforcement learning alignment using solver feedback, achieve **93% fully-constrained sketches** versus 8.9% without alignment. Newton's method solvers iteratively satisfy constraint systems—failure to converge indicates genuinely inconsistent geometry.

**Validity rates remain modest**: BrepGen achieves **62.9% watertight solids** on the DeepCAD distribution, **48.2% on ABC**. DeepCAD achieves 46.1%, SolidGen 60.3%. Common failure modes include non-watertight solids (missing faces leaving open regions), self-intersecting geometry, dangling edges shared by fewer than 2 faces, and broken face-edge-vertex connectivity. Mesh repair tools like ManifoldPlus can convert triangle soups to watertight manifolds but lose parametric information.

## The topology gap and geometric precision limit practical deployment

The fundamental tension between continuous neural predictions and discrete CAD requirements creates several persistent challenges.

**The topology gap** is perhaps the hardest: B-rep requires exact integer face counts and binary adjacency relationships that cannot be meaningfully interpolated. BrepGen's node duplication elegantly sidesteps explicit topology prediction by encoding adjacency through geometry similarity, but multinomial diffusion approaches that directly predict discrete topology achieve only **6.2% validity**—demonstrating the problem's severity.

**Quantization effects** propagate through the entire pipeline. 6-bit coordinate quantization (64 levels) on a 2×2×2 normalized cube provides ~0.03 unit resolution—adequate for rough shape but insufficient for precision engineering. BrepGen's 0.05 minimum threshold (7-bit equivalent) prevents distinct geometric features from collapsing into identical quantization bins. Finer quantization expands vocabulary size quadratically, straining transformer attention.

**Numerical precision** compounds at boolean operations. Surface-surface intersections accumulate floating-point errors, creating "pretty short edges" and sliver faces that cause topological problems. Edge intersection tolerance tuning is critical—BrepGen found optimal thresholds of 0.06-0.1 for bounding boxes and ~0.2 for shape features.

**Scalability to complex assemblies** remains largely unsolved. Current methods handle single-body solids with 30-80 faces and 600-1500 total sequence tokens. Attention complexity scales quadratically with sequence length. Multi-part assembly generation—predicting how components relate and connect—is an open research frontier, with datasets like JoinABLe and AutoMate beginning to address assembly mating but not generation.

## The 2023-2025 research landscape shows rapid architecture evolution

**BrepGen** (SIGGRAPH 2024) represents the current state-of-the-art for unconditional B-rep generation, combining VAE-compressed geometric features with structured diffusion over hierarchical trees. Its node duplication scheme for implicit topology encoding influenced subsequent work.

**Text2CAD** (NeurIPS 2024 Spotlight) established the BERT+cross-attention paradigm for text-conditioned CAD, contributing the first large-scale text-CAD paired dataset with 660K annotations across complexity levels.

**CAD-MLLM** aligns text, images, and point clouds to CAD sequences through a Vicuna-7B backbone fine-tuned with LoRA, demonstrating that large language model alignment can unify multi-modal CAD generation.

**HNC-CAD** introduces hierarchical neural codebooks at loop, profile, and solid levels, with masked skip connections preventing posterior collapse while maintaining controllable multi-scale generation.

**STEP-LLM** tackles the challenge of processing raw STEP files by DFS-based reserialization that linearizes graph cross-references while preserving locality, achieving >95% renderability on generated outputs.

**Zoo.dev's production system** represents the commercial frontier, generating editable KCL code from text prompts with enterprise fine-tuning on proprietary CAD data—demonstrating the viability of code-generation approaches for manufacturing-ready output.

## Conclusion

CAD generation has evolved from proof-of-concept VAEs to sophisticated production systems in under five years. The field has converged on key insights: quantized tokenization outperforms regression, disentangled representations enable control, and implicit topology encoding via geometry similarity sidesteps intractable discrete prediction. Yet fundamental challenges persist—validity rates plateau below 70%, complex assemblies remain out of reach, and the topology gap between continuous neural outputs and discrete CAD requirements has no clean solution. The most promising near-term path appears to be hybrid approaches where neural networks generate CAD code (KCL, CadQuery) that geometric kernels execute and validate, leveraging the reliability of traditional CAD systems while benefiting from neural network flexibility. For researchers and practitioners, the mechanistic understanding is clear: success in this domain requires joint mastery of geometric modeling, sequence modeling, and the subtle art of bridging continuous probabilities with exact geometric constraints.
