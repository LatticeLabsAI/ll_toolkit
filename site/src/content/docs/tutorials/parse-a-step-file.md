---
title: 'Tutorial: Parse a STEP file'
description: Convert a STEP/STL file into a structured CADlingDocument, inspect its items, and export it to JSON and Markdown.
sidebar:
  label: Parse a STEP file
  order: 2
---

In this tutorial you will parse a CAD file with [cadling](/ll_toolkit/cadling/overview/),
inspect the resulting document, chunk it for RAG, and export it. Allow ~10
minutes.

## Prerequisites

- cadling installed (`pip install -e ".[all]"` inside the conda env — see
  [Installation](/ll_toolkit/cadling/installation/)).
- A CAD file. Any `.step`, `.stl`, `.brep`, or `.iges` file works. The repo
  ships an example `part.step` at its root.

## 1. Convert the file

```python
from cadling import DocumentConverter, ConversionStatus

converter = DocumentConverter()
result = converter.convert("part.step")

assert result.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL)
doc = result.document
print(f"Parsed {len(doc.items)} items from a {result.status.name} conversion")
```

`DocumentConverter` detects the format, selects a backend, and runs the
Build → Assemble → Enrich pipeline, returning a `CADlingDocument`.

## 2. Inspect the document

```python
for item in doc.items[:10]:
    print(type(item).__name__, getattr(item, "entity_type", ""))

# Topology: the entity-reference graph
topo = doc.topology
if topo is not None:
    print("topology nodes:", len(topo.adjacency_list))
```

Items are typed: `STEPEntityItem`, `MeshItem`, `AssemblyItem`, `AnnotationItem`.

## 3. Chunk it for RAG

```python
from cadling.chunker.hybrid_chunker import CADHybridChunker

chunker = CADHybridChunker(max_tokens=512, overlap_tokens=50)
chunks = list(chunker.chunk(doc))
print(f"{len(chunks)} chunks")
print(chunks[0].text[:200])
```

Each chunk carries `meta` with entity types, a topology subgraph, embeddings, and
a 3D bounding box — ready to index in a vector database.

## 4. Export

```python
import json
from pathlib import Path

# export_to_json() returns a dict; export_to_markdown() returns a string.
Path("part.json").write_text(json.dumps(doc.export_to_json(), indent=2))
Path("part.md").write_text(doc.export_to_markdown())
```

## Or do it all from the CLI

```bash
cadling convert part.step --format json --pretty -o part.json
cadling chunk part.step --max-tokens 512 --overlap 50 -o chunks.jsonl
cadling info part.step
```

## Where to next

- Tokenize this geometry: [Tokenize a mesh](/ll_toolkit/tutorials/tokenize-a-mesh/).
- Generate Q&A training data: see [cadling Usage → SDG](/ll_toolkit/cadling/usage/).
