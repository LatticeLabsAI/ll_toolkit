---
title: Quickstart
description: A first end-to-end run — parse a CAD file with cadling, then tokenize and encode it.
sidebar:
  order: 2
---

This page gets you from a CAD file to structured data in a few lines. It assumes
you have [installed](/ll_toolkit/get-started/installation/) the toolkit.

## Convert a CAD file (CLI)

`cadling` ships a command-line entry point:

```bash
# Convert a CAD file to JSON or Markdown
cadling convert model.step --format json -o model.json

# Chunk a CAD file for RAG
cadling chunk model.step --max-tokens 512 --overlap 50 -o chunks.jsonl

# Show file information
cadling info model.step
```

## Convert a CAD file (Python)

```python
from cadling import DocumentConverter, ConversionStatus

converter = DocumentConverter()
result = converter.convert("model.step")

if result.status == ConversionStatus.SUCCESS:
    doc = result.document
    print(f"Parsed {len(doc.items)} items")

    json_data = doc.export_to_json()
    markdown = doc.export_to_markdown()
```

## Tokenize geometry

```python
from geotoken import GeoTokenizer

tokenizer = GeoTokenizer()
tokens = tokenizer.tokenize(vertices, faces)
```

## Encode a STEP file with a neural model

```python
import torch
from stepnet import STEPEncoder, STEPTokenizer, STEPFeatureExtractor, STEPTopologyBuilder

tokenizer, extractor, builder, encoder = (
    STEPTokenizer(), STEPFeatureExtractor(), STEPTopologyBuilder(), STEPEncoder()
)

token_ids = torch.tensor([tokenizer.encode(step_text)])
topology = builder.build_complete_topology(
    extractor.extract_features_from_chunk(step_text)
)
embedding = encoder(token_ids, topology_data=topology)  # [1, 1024]
```

:::note[Models ship untrained]
The neural packages (`ll_stepnet`, `ll_ocadr`, `ll_gen`) ship with
architectures but **no trained checkpoints**. Until you train them, their
outputs are not meaningful predictions. Each package's pages explain how to
train or run a proof-of-life model.
:::

## Where to go next

- **[cadling](/ll_toolkit/cadling/overview/)** — full CAD parsing, chunking, and SDG.
- **[geotoken](/ll_toolkit/geotoken/overview/)** — adaptive geometric tokenization.
- **[ll_stepnet](/ll_toolkit/ll_stepnet/overview/)** — neural STEP/B-Rep models.
- **[ll_ocadr](/ll_toolkit/ll_ocadr/overview/)** — geometry-aware LLM input.
- **[ll_gen](/ll_toolkit/ll_gen/overview/)** — generative CAD orchestration.
- **[ll_clouds](/ll_toolkit/ll_clouds/overview/)** — point-cloud processing.
