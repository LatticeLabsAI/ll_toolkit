---
title: ll_stepnet — Overview
description: Neural network package for STEP/B-Rep CAD files — tokenizer, feature extractor, topology builder, transformer+GNN encoder, and task heads.
sidebar:
  label: Overview
  order: 1
  badge:
    text: Trained + MLX
    variant: success
---

**ll_stepnet** is a neural-network package for processing STEP / B-Rep CAD
files, built with a clean separation of concerns. It turns raw STEP text into
token IDs, geometric features, and a topology graph, fuses them in a
transformer + graph-neural-network encoder, and exposes task-specific heads for
classification, property prediction, similarity, captioning, and QA.

The installed top-level package is **`stepnet`** (import `from stepnet import ...`).

## Modules

| Module | Responsibility |
|---|---|
| `stepnet.tokenizer` | Convert STEP text → token IDs |
| `stepnet.features` | Extract geometric properties per entity |
| `stepnet.topology` | Build entity-reference graphs |
| `stepnet.encoder` | Transformer + GNN encoder fusing all representations |
| `stepnet.tasks` | Task-specific prediction heads |
| `stepnet.data` | Dataset / DataLoader helpers |
| `stepnet.trainer` | Training loop |

## Model architecture

```text
STEP File
  → Tokenizer        → Token IDs → Transformer Encoder → Token Embedding
  → Feature Extractor → Geometric Features
  → Topology Builder  → Graph → Graph Neural Network → Graph Embedding

Token Embedding + Graph Embedding → Fusion Layer → Final Encoding → Task Head
```

## A first taste

```python
from stepnet import STEPTokenizer, STEPFeatureExtractor

tokenizer = STEPTokenizer()
step_text = "#31=CONICAL_SURFACE('',#1837,2.6797,0.7854);"
token_ids = tokenizer.encode(step_text)

extractor = STEPFeatureExtractor()
features = extractor.extract_geometric_features(step_text)
print(features["entity_type"], features["numeric_params"])
```

## Status

:::tip[First real trained checkpoint + native MLX]
ll_stepnet now ships a **trained classifier**: `STEPForClassification` trained on real
DeepCAD models to predict face-count complexity (≤4 / 5–6 / 7+ faces, a geometric label
derived from the reconstructed solid). **Validation accuracy 0.976** vs a 0.436
majority-class baseline (per-class 0.991 / 0.978 / 0.933). A **native-MLX port**
(`ll_stepnet/mlx/train_classification_mlx.py`) loads those exact trained weights and
reproduces the PyTorch model on Apple Silicon — verified at **100% argmax agreement,
identical 0.976 accuracy** (`--mode parity`). The other task heads (property prediction,
similarity, captioning, QA) provide architectures + trainers; train them on your data
before relying on their outputs.
:::

`stepnet` also provides the generative models (`STEPVAE`, `StructuredDiffusion`,
`VQVAEModel`, `CADGenerationPipeline`). The valid-CAD generation path now lives in
[ll_gen](/ll_toolkit/ll_gen/overview/) as trained, program-based generators
(autoregressive command model + latent diffusion).

Use the sidebar for **Installation**, **Usage**, and the **API Reference**.
