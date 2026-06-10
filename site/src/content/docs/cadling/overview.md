---
title: cadling — Overview
description: A docling-inspired toolkit for CAD document processing — parsing, topology analysis, RAG chunking, and synthetic data generation.
sidebar:
  label: Overview
  order: 1
  badge:
    text: Beta
    variant: note
---

**cadling** is a [docling](https://github.com/DS4SD/docling)-inspired toolkit
for CAD document processing. It parses STEP / STL / BRep / IGES files (and
rendered CAD images), builds a structured document model, and exposes that model
for RAG chunking, synthetic Q&A generation, and JSON/Markdown export — making
CAD files first-class citizens in LLM/ML pipelines.

## What it does

- **Dual-modality processing** — text-based parsing of CAD file contents (STEP
  entities, STL vertices) and vision-based recognition from rendered images,
  with hybrid fusion of both.
- **Multi-format support** — STEP (ISO 10303-21), STL (ASCII + binary), BRep
  (OpenCASCADE), IGES, and rendered CAD images.
- **Topology analysis** — graph-based analysis of entity relationships,
  assembly hierarchies, and geometric connectivity.
- **RAG-ready chunking** — hybrid, hierarchical, and topology-aware chunkers
  with 3D metadata for vector databases.
- **Synthetic data generation (SDG)** — LLM-powered Q&A pair generation from CAD
  documents, with sampling, generation, and critique stages.
- **Enrichment models** — pluggable geometry analysis, topology validation, mesh
  quality, interference checking, surface analysis, and GNN segmentation.

## Architecture at a glance

```text
DocumentConverter (entry point)
  → Format Detection → Backend Selection
  → Backend (format-specific parsing)
  → Pipeline (Build → Assemble → Enrich)
  → CADlingDocument (central data model)
  → Chunking (RAG) · SDG (Q&A) · Export (JSON / Markdown)
```

The processing flow mirrors docling's Backend / Pipeline / Enrichment layering,
adapted for 3D geometry. See **Usage** (in the sidebar) for the layer details and
worked examples.

## A first taste

```python
from cadling import DocumentConverter, ConversionStatus

converter = DocumentConverter()
result = converter.convert("part.step")

if result.status == ConversionStatus.SUCCESS:
    doc = result.document
    print(f"Parsed {len(doc.items)} items")
    markdown = doc.export_to_markdown()
```

## Status

:::note[Maturity: Beta]
cadling is the most complete package in the toolkit — broad format coverage, a
full CLI, chunkers, and an SDG pipeline. Some enrichment/geometry methods are
still being hardened (tracked inside the repo). Where a computation is not yet
implemented, the code raises or logs rather than returning fabricated geometry.
:::

Use the sidebar for **Installation**, **Usage**, and the **API Reference**.
