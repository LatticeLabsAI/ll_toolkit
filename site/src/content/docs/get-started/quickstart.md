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

:::note[Which models are trained]
Several models now ship **trained** with reproducible, honest metrics:
**ll_brepnet** (B-Rep segmentation, test mIoU 0.828), **ll_stepnet** (face-count
classifier, val acc 0.976), **ll_ocadr** (geometry-grounded, 0.919 vs 0.313 shuffled),
and **ll_gen**'s program-based generators (valid CAD: AR {{metric.ll_gen.ar.validity}} / latent diffusion {{metric.ll_gen.latentDiffusion.sampledZValidity}}).
The neural models train and run natively in **MLX on Apple Silicon** as well as PyTorch.
Remaining task heads (e.g. ll_stepnet property prediction/QA) ship as architectures —
train them on your data before relying on their outputs.
:::

## Train / run natively on Apple Silicon (MLX)

Each neural package has an `mlx/` trainer that runs on Apple Silicon. The ones with
existing PyTorch checkpoints convert the real weights and prove parity:

```bash
python {{script.ll_stepnet.classification}} --mode parity   # acc 0.976, argmax 1.0 vs PyTorch
python {{script.ll_brepnet.train}}        --mode parity   # mIoU parity vs PyTorch
python {{script.ll_gen.arGenerator}}             --mode train    # valid CAD generation {{metric.ll_gen.ar.validity}}
python {{script.ll_gen.latentDiffusion}}         --mode train    # latent-diffusion generation {{metric.ll_gen.latentDiffusion.sampledZValidity}}
python {{script.ll_ocadr.train}}            --mode train    # geometry-grounded multimodal
```

## Where to go next

- **[cadling](/ll_toolkit/cadling/overview/)** — full CAD parsing, chunking, and SDG.
- **[geotoken](/ll_toolkit/geotoken/overview/)** — adaptive geometric tokenization.
- **[ll_stepnet](/ll_toolkit/ll_stepnet/overview/)** — neural STEP/B-Rep models.
- **[ll_ocadr](/ll_toolkit/ll_ocadr/overview/)** — geometry-aware LLM input.
- **[ll_gen](/ll_toolkit/ll_gen/overview/)** — generative CAD orchestration.
- **[ll_clouds](/ll_toolkit/ll_clouds/overview/)** — point-cloud processing.
