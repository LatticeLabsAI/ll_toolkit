---
title: 'How to: chunk a CAD file for RAG'
description: Turn a CAD file into retrievable, metadata-rich chunks ready to embed and index in a vector database.
sidebar:
  label: Chunk a CAD file for RAG
  order: 2
---

**Goal:** convert a CAD file into chunks you can embed and store in a vector
database for retrieval-augmented generation.

This assumes you can already [parse a file](/ll_toolkit/tutorials/parse-a-step-file/)
with cadling.

## 1. Parse, then chunk

```python
from cadling import DocumentConverter
from cadling.chunker.hybrid_chunker import CADHybridChunker

doc = DocumentConverter().convert("assembly.step").document

chunker = CADHybridChunker(max_tokens=512, overlap_tokens=50)
chunks = list(chunker.chunk(doc))
```

Pick the chunker for your retrieval need:

| Chunker | Strategy |
|---|---|
| `CADHybridChunker` | entity-level + semantic grouping (good default) |
| `CADHierarchicalChunker` | assembly-hierarchy-aware (preserves BOM structure) |

## 2. Use the chunk text and metadata

Each chunk carries text plus structured metadata you should store alongside the
vector — it makes retrieval filterable and the context richer.

```python
for chunk in chunks:
    text = chunk.text                 # what you embed
    meta = chunk.meta                 # entity types, topology subgraph, bbox
    print(chunk.chunk_id, len(meta.entity_ids), "entities")
```

## 3. Embed and index

Use any embedding model and vector store. The pattern:

```python
records = []
for chunk in chunks:
    records.append({
        "id": chunk.chunk_id,
        "text": chunk.text,
        "vector": embed(chunk.text),          # your embedding function
        "metadata": {
            "entity_ids": list(chunk.meta.entity_ids),
            "source": "assembly.step",
        },
    })

vector_store.upsert(records)                  # your vector DB client
```

Store the `metadata` so you can filter retrieval (e.g. by entity type) and cite
the source CAD file in answers.

## 4. From the CLI instead

```bash
cadling chunk assembly.step --max-tokens 512 --overlap 50 -o chunks.jsonl
```

Each line of `chunks.jsonl` is one chunk with its text and metadata, ready to
feed an embedding/indexing job.

## See also

- [cadling Usage](/ll_toolkit/cadling/usage/) — all chunkers and the SDG pipeline.
- [Generate synthetic Q&A](/ll_toolkit/cadling/usage/) to build training data
  from the same documents.
