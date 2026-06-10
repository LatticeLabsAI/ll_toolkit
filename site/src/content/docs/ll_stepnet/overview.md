---
title: ll_stepnet — Overview
description: Neural network package for STEP/B-Rep CAD files — tokenizer, feature extractor, topology builder, transformer+GNN encoder, and task heads.
sidebar:
  label: Overview
  order: 1
  badge:
    text: Untrained
    variant: caution
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

:::caution[Maturity: architectures present, models untrained]
ll_stepnet provides full model architectures and a trainer, but **ships no
trained checkpoints**. A randomly-initialized model produces meaningless
predictions until you train it on STEP data. See **Usage** (in the sidebar) for
the training loop.
:::

`stepnet` also provides the generative models (`STEPVAE`, `StructuredDiffusion`,
`VQVAEModel`, `CADGenerationPipeline`) that [ll_gen](/ll_toolkit/ll_gen/overview/)
drives for neural CAD generation.

Use the sidebar for **Installation**, **Usage**, and the **API Reference**.
