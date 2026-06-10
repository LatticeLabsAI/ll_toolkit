---
title: cadling — Usage
description: Convert, chunk, and generate Q&A from CAD files — the CLI, the Python API, and the Backend / Pipeline / Enrichment layers.
sidebar:
  label: Usage
  order: 3
---

cadling's processing flow is: **DocumentConverter → Backend → Pipeline
(Build → Assemble → Enrich) → CADlingDocument → Chunking / SDG / Export.**

## Python API

### Basic conversion

```python
from cadling import DocumentConverter, ConversionStatus

converter = DocumentConverter()
result = converter.convert("part.step")

if result.status == ConversionStatus.SUCCESS:
    doc = result.document
    print(f"Parsed {len(doc.items)} items")

    json_data = doc.export_to_json()
    markdown = doc.export_to_markdown()
```

### With format options

```python
from cadling import DocumentConverter, FormatOption, InputFormat
from cadling.backend.step.step_backend import STEPBackend
from cadling.pipeline.hybrid_pipeline import HybridPipeline

converter = DocumentConverter(
    allowed_formats=[InputFormat.STEP],
    format_options={
        InputFormat.STEP: FormatOption(
            backend=STEPBackend,
            pipeline_cls=HybridPipeline,
        )
    },
)
result = converter.convert("assembly.step")
```

### Chunking for RAG

```python
from cadling import DocumentConverter
from cadling.chunker.hybrid_chunker import CADHybridChunker

result = DocumentConverter().convert("part.step")

chunker = CADHybridChunker(max_tokens=512, overlap_tokens=50)
for chunk in chunker.chunk(result.document):
    print(f"Chunk {chunk.chunk_id}: {len(chunk.meta.entity_ids)} entities")
    # chunk.text  → text representation
    # chunk.meta  → entity types, topology subgraph, embeddings, bbox
```

### Synthetic data generation (SDG)

```python
from pathlib import Path
from cadling.sdg.qa import (
    CADPassageSampler, CADGenerator, CADJudge,
    CADSampleOptions, CADGenerateOptions, CADCritiqueOptions, LlmProvider,
)

# 1. Sample passages from CAD files
sampler = CADPassageSampler(CADSampleOptions(sample_file=Path("samples.jsonl")))
sampler.sample([Path("part.step"), Path("assembly.step")])

# 2. Generate Q&A pairs
gen = CADGenerator(CADGenerateOptions(
    provider=LlmProvider.OPENAI, model_id="gpt-4o",
    generated_file=Path("generated.jsonl"),
))
gen.generate(Path("samples.jsonl"))

# 3. Critique and improve
judge = CADJudge(CADCritiqueOptions(
    provider=LlmProvider.OPENAI, model_id="gpt-4o",
    critiqued_file=Path("critiqued.jsonl"),
))
judge.critique(Path("generated.jsonl"))
```

## CLI

### `cadling` (main)

```bash
cadling convert part.step --format json --pretty -o part.json
cadling chunk part.step --max-tokens 512 --overlap 50 -o chunks.jsonl
cadling generate-qa part.step -n 100 -m gpt-4 -o qa.jsonl
cadling info part.step
```

### `cadling-sdg`

```bash
cadling-sdg qa sample part.step assembly.step --chunker hybrid -o samples.jsonl
cadling-sdg qa generate samples.jsonl -p openai -m gpt-4o -o generated.jsonl
cadling-sdg qa critique generated.jsonl -p openai -m gpt-4o --rewrite -o critiqued.jsonl
```

## The layers

| Layer | Where | Role |
|---|---|---|
| **Backends** | `cadling/backend/` | Format-specific parsing. `DeclarativeCADBackend` (text) and `RenderableCADBackend` (vision). STEP and STL are dual-mode. |
| **Pipelines** | `cadling/pipeline/` | Orchestrate Build → Assemble → Enrich. `SimpleCADPipeline`, `STEPPipeline`, `STLPipeline`, `VisionPipeline`, `VlmPipeline`, `HybridPipeline`. |
| **Enrichment models** | `cadling/models/` | Optional post-processing: geometry analysis, topology validation, mesh quality, surface analysis, interference, GNN segmentation. |
| **Chunkers** | `cadling/chunker/` | RAG chunking: `CADHybridChunker`, `CADHierarchicalChunker`, format-specific. |
| **SDG** | `cadling/sdg/` | `CADPassageSampler` → `CADGenerator` → `CADJudge` (+ `CADConceptualGenerator`). |

## Core data models

```python
InputFormat       # STEP | STL | BREP | IGES | CAD_IMAGE
ConversionStatus  # SUCCESS | PARTIAL | FAILURE
CADlingDocument   # items, topology, segments, embeddings
CADItem           # STEPEntityItem | MeshItem | AssemblyItem | AnnotationItem
TopologyGraph     # entity reference graph (adjacency list)
```

## Related

- [Overview](/ll_toolkit/cadling/overview/) · [Installation](/ll_toolkit/cadling/installation/)
- Tokenize cadling output with [geotoken](/ll_toolkit/geotoken/overview/).
