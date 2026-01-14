# STEP-Aware Unsupervised Pre-training

## Overview

LL-STEPNet uses **STEP-aware pre-training** that goes beyond simple token prediction. Unlike vanilla language models (GPT/BERT) that treat STEP files as pure text, our models understand the **geometric and topological structure** of CAD data.

## Why STEP-Aware Architecture?

STEP files are not just text - they encode:
- **Geometric primitives** (points, curves, surfaces with numeric parameters)
- **Entity references** (connections between geometric elements)
- **Topology** (how surfaces form solids, faces form shells, etc.)
- **Hierarchical structure** (assemblies, parts, sub-assemblies)

A vanilla transformer sees: `#123 = CIRCLE('', #124, 5.0);` as tokens
A STEP-aware transformer sees: **Entity 123**, type **CIRCLE**, references **Entity 124**, radius **5.0**, with topological relationships

## Architecture Components

### 1. Token Sequence Modeling

**STEPTransformerEncoder** (for Masked LM):
- Bidirectional attention over token sequences
- Learns token co-occurrence patterns
- Standard transformer architecture (like BERT)

**Causal Decoder** (for Causal LM):
- Unidirectional attention (predict next token)
- Autoregressive generation capability
- Standard decoder architecture (like GPT)

### 2. Topology/Geometry Understanding

**STEPGraphEncoder** (Graph Neural Network):
- Processes entity reference graph
- Message passing over topology edges
- Learns structural patterns (e.g., "circles reference points for centers")

**Node Features** (129-dimensional):
- 128 dimensions: Numeric geometric parameters (radius, coordinates, etc.)
- 1 dimension: Entity type embedding (hashed)

### 3. Multi-Modal Fusion

**Fusion Layer**:
- Combines token sequence features + graph topology features
- Projects to language model head
- Enables reasoning about both syntax AND structure

## Pre-training Tasks

### Causal Language Modeling (GPT-style)

**Objective**: Predict next token given previous tokens + topology

```python
# Input: #123 = CIRCLE('', #124,
# Label: 5.0
# Topology: Entity #123 references Entity #124 (CARTESIAN_POINT)
# Model learns: "CIRCLE referencing a POINT likely followed by radius parameter"
```

**Loss**: Cross-entropy on next token prediction

### Masked Language Modeling (BERT-style)

**Objective**: Predict masked tokens from bidirectional context + topology

```python
# Input: #123 = [MASK]('', #124, 5.0);
# Label: CIRCLE
# Topology: Entity #123 references Entity #124, has numeric param 5.0
# Model learns: "Entity with point reference + single numeric param → CIRCLE"
```

**Loss**: Cross-entropy on masked token prediction

## Data Requirements

**ZERO LABELS NEEDED!** Just point to a directory of raw STEP files:

```bash
data/
├── part1.step
├── part2.step
├── assembly3.step
└── subfolder/
    ├── part4.step
    └── part5.stp
```

The model learns from:
1. **Token patterns** (syntax of STEP language)
2. **Geometric parameters** (typical radius values, coordinate ranges)
3. **Topology structure** (which entities reference which)

## Training Example

```python
from stepnet.pretrain import STEPForCausalLM, STEPForMaskedLM
from examples.pretrain_unsupervised import train_causal_lm, train_masked_lm

# GPT-style pre-training (autoregressive)
train_causal_lm(
    data_dir='data/raw_step_files',
    output_dir='checkpoints/pretrain_causal',
    num_epochs=10
)

# BERT-style pre-training (masked prediction)
train_masked_lm(
    data_dir='data/raw_step_files',
    output_dir='checkpoints/pretrain_masked',
    num_epochs=10
)
```

## What the Model Learns

### 1. Geometric Patterns
- "CIRCLE entities have 1-2 numeric parameters (radius, maybe thickness)"
- "B_SPLINE_CURVE_WITH_KNOTS has many control points"
- "Coordinates typically in range [-1000, 1000] mm"

### 2. Structural Patterns
- "ADVANCED_FACE references EDGE_LOOP"
- "EDGE_LOOP contains multiple ORIENTED_EDGE"
- "ORIENTED_EDGE references EDGE_CURVE and VERTEX_POINT"

### 3. Contextual Understanding
- "After 'PRODUCT_DEFINITION', expect 'PRODUCT_DEFINITION_SHAPE'"
- "MECHANICAL_DESIGN entities cluster together"
- "Assembly files have more NEXT_ASSEMBLY_USAGE_OCCURRENCE"

## Comparison: Vanilla vs STEP-Aware

| Aspect | Vanilla Transformer | STEP-Aware (ours) |
|--------|-------------------|-------------------|
| Input | Token IDs only | Tokens + Topology graph |
| Entity understanding | No - just text patterns | Yes - geometric meaning |
| Reference resolution | No - treats #123 as arbitrary | Yes - follows entity links |
| Numeric parameters | Treats as random tokens | Extracts as features |
| Topology awareness | None | Full graph structure |
| Pre-training signal | Token co-occurrence | Tokens + Structure |

## Benefits for Downstream Tasks

After STEP-aware pre-training, the model can be fine-tuned for:

1. **Classification** - Understands part geometry + topology
2. **Property Prediction** - Relates structure to physical properties
3. **Similarity Search** - Compares topology, not just syntax
4. **Captioning** - Describes geometric features accurately
5. **Question Answering** - Reasons about entity relationships

## Technical Implementation

### Forward Pass (Causal LM)

```python
def forward(self, input_ids, topology_data, labels):
    # 1. Token embeddings + positional encoding
    token_embeds = self.token_embedding(input_ids) + self.position_embedding(positions)

    # 2. Causal transformer (predict next token)
    hidden_states = self.transformer(token_embeds, causal_mask)

    # 3. Graph encoder on topology
    if topology_data:
        adj_matrix = topology_data['adjacency_matrix']
        node_features = topology_data['node_features']
        graph_features = self.graph_encoder(node_features, adj_matrix)
        graph_pooled = graph_features.mean(dim=0)  # Pool to single vector

        # 4. Fuse token + graph
        combined = torch.cat([hidden_states, graph_pooled.expand(...)])
        hidden_states = self.fusion(combined)

    # 5. Language model head
    logits = self.lm_head(hidden_states)

    # 6. Compute loss (shifted by 1 for next-token prediction)
    loss = CrossEntropyLoss(logits[:-1], labels[1:])

    return {'logits': logits, 'loss': loss}
```

### Dataset Processing

```python
class RawSTEPDataset:
    def __getitem__(self, idx):
        # 1. Read STEP file
        step_content = read_step_file(self.step_files[idx])

        # 2. Tokenize (syntax)
        token_ids = self.tokenizer.encode(step_content)

        # 3. Extract features (geometry)
        features_list = self.feature_extractor.extract_features_from_chunk(step_content)

        # 4. Build topology (structure)
        topology_data = self.topology_builder.build_complete_topology(features_list)

        return {
            'input_ids': token_ids,
            'topology_data': topology_data
        }
```

## Model Sizes

| Configuration | Parameters | Embed Dim | Layers | Heads | Graph Layers |
|---------------|-----------|-----------|--------|-------|--------------|
| Small | ~50M | 512 | 6 | 8 | 3 |
| Medium | ~300M | 768 | 12 | 12 | 4 |
| Large | ~1B | 1024 | 24 | 16 | 6 |

## Training Tips

1. **Start Small**: Use 6-layer models for initial experiments
2. **Batch Size**: Process samples individually (topology graphs vary in size)
3. **Learning Rate**: 1e-4 works well with AdamW
4. **Masking Probability**: 15% (BERT standard) for masked LM
5. **Topology Extraction**: Wraps in try/except - some files may fail
6. **GPU Memory**: Topology adds ~20-30% memory overhead

## Future Enhancements

1. **Hierarchical Topology**: Multi-level graph (entities → faces → solids)
2. **Attention over Graph**: Direct attention between tokens and nodes
3. **Contrastive Learning**: Similar parts should have similar embeddings
4. **Multi-Task Pre-training**: Combine causal + masked + contrastive
5. **Transfer Learning**: Pre-train on large corpus, fine-tune on domain-specific

## References

- **STEP ISO 10303**: CAD data exchange standard
- **BERT**: Masked language modeling (Devlin et al., 2018)
- **GPT**: Causal language modeling (Radford et al., 2018)
- **Graph Neural Networks**: Message passing (Gilmer et al., 2017)
- **PointNet++**: 3D geometry understanding (Qi et al., 2017)

---

**Key Insight**: STEP files are structured geometric data, not just text. Pre-training must capture both syntax (tokens) and semantics (topology + geometry) to truly understand CAD models.
