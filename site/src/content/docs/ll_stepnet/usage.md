---
title: ll_stepnet — Usage
description: Tokenize, extract features, build topology, encode, and train task-specific models on STEP/B-Rep data.
sidebar:
  label: Usage
  order: 3
---

ll_stepnet (imported as `stepnet`) composes four representations — tokens,
geometric features, a topology graph, and a fused encoding — and exposes
task-specific heads on top.

## Tokenization, features, topology

```python
from stepnet import STEPTokenizer, STEPFeatureExtractor, STEPTopologyBuilder

tokenizer = STEPTokenizer()
extractor = STEPFeatureExtractor()
builder = STEPTopologyBuilder()

step_text = "#31=CONICAL_SURFACE('',#1837,2.6797,0.7854);"
token_ids = tokenizer.encode(step_text)

features = extractor.extract_geometric_features(step_text)
print(features["entity_type"], features["numeric_params"], features["references"])

features_list = extractor.extract_features_from_chunk(chunk_text)
topology = builder.build_complete_topology(features_list)
print(topology["num_nodes"], topology["num_edges"])
```

## Encoding

```python
import torch
from stepnet import STEPEncoder, STEPTokenizer, STEPFeatureExtractor, STEPTopologyBuilder

tokenizer, extractor, builder, encoder = (
    STEPTokenizer(), STEPFeatureExtractor(), STEPTopologyBuilder(), STEPEncoder()
)

token_ids = torch.tensor([tokenizer.encode(chunk_text)])
topology = builder.build_complete_topology(extractor.extract_features_from_chunk(chunk_text))

output = encoder(token_ids, topology_data=topology)  # → [1, 1024]
```

## Task-specific models

```python
from stepnet import (
    STEPForClassification, STEPForPropertyPrediction,
    STEPForSimilarity, STEPForCaptioning, STEPForQA,
)

clf = STEPForClassification(vocab_size=50000, num_classes=10, output_dim=1024)
logits = clf(token_ids, topology_data=topology)

props = STEPForPropertyPrediction(vocab_size=50000, num_properties=6, output_dim=1024)
# returns [volume, surface_area, mass, bbox_x, bbox_y, bbox_z]

sim = STEPForSimilarity(vocab_size=50000, embedding_dim=512)
embedding = sim(token_ids, topology_data=topology)  # L2-normalized
```

## Training

```python
from stepnet import STEPForClassification, create_dataloader, STEPTrainer

train_loader = create_dataloader(file_paths=train_files, labels=train_labels,
                                 batch_size=8, use_topology=True)
val_loader = create_dataloader(file_paths=val_files, labels=val_labels,
                               batch_size=8, use_topology=True)

trainer = STEPTrainer(model=STEPForClassification(num_classes=10),
                      train_dataloader=train_loader, val_dataloader=val_loader,
                      checkpoint_dir="checkpoints")
trainer.train(num_epochs=10, save_every=2)
```

:::caution[Train before you trust outputs]
The models ship **untrained**. A randomly-initialized network produces
meaningless predictions — train it on STEP data (labels as integers for
classification, or float vectors for property prediction) before relying on
outputs.
:::

## Generative models

`stepnet` also exports the generative models — `STEPVAE`, `StructuredDiffusion`,
`VQVAEModel`, `CADGenerationPipeline` — that [ll_gen](/ll_toolkit/ll_gen/overview/)
drives for neural CAD generation.

## Related

- [Overview](/ll_toolkit/ll_stepnet/overview/) · [Installation](/ll_toolkit/ll_stepnet/installation/)
- Tokenize inputs with [geotoken](/ll_toolkit/geotoken/overview/) (native format alignment, zero adapters).
