---
title: ll_brepnet (Planned)
description: A planned B-Rep face-graph neural network. Not yet implemented — the package is currently an empty scaffold.
sidebar:
  label: ll_brepnet
  order: 1
  badge:
    text: Planned
    variant: caution
---

:::caution[Not implemented yet]
`ll_brepnet` is **planned, not built**. The directory exists in the repository,
but every file in it is currently empty (0 bytes) — including `pyproject.toml`,
`requirements.txt`, and all of its Python modules. It is **not** installable,
importable, or runnable today, and it is intentionally **not** listed among the
toolkit's shipping packages.

This page exists so the project's status is honest: the toolkit documents what
its code actually does. When `ll_brepnet` has a real implementation, it will get
full Overview / Installation / Usage / API Reference pages like the other
packages — and a roadmap entry will no longer be needed.
:::

## What it is intended to be

Based on the scaffolded module layout, `ll_brepnet` is intended to be a **B-Rep
face-graph neural network** in the lineage of UV-Net / BRepNet — models that
operate directly on the boundary-representation graph of a CAD solid (faces as
nodes, edges as connections) with UV-grid surface features.

The planned structure (currently empty placeholders) sketches:

| Area | Planned modules |
|---|---|
| Data loading | `dataloaders/brep_dataset.py`, `dataloaders/max_num_faces_loader.py` |
| Models | `models/ll_brepnet.py`, `models/uvnet_encoders.py` |
| Pipelines | `pipelines/extract_brepnet_data_from_step.py`, `extract_brepnet_data_from_json.py`, `entity_mapper.py`, `build_dataset_file.py` |
| Evaluation | `eval/evaluate.py` |

## How it would fit the toolkit

A B-Rep face-graph model complements the existing neural packages:

- [cadling](/ll_toolkit/cadling/overview/) already builds a B-Rep topology graph
  during parsing — a natural input source.
- [geotoken](/ll_toolkit/geotoken/overview/) already provides B-Rep
  graph tokenization (`GraphTokenizer`).
- [ll_stepnet](/ll_toolkit/ll_stepnet/overview/) covers STEP-text + topology
  fusion; `ll_brepnet` would specialize in pure face-graph learning.

## Status

There is no code, no tests, and no checkpoints. Follow the repository for
updates: <https://github.com/LatticeLabsAI/ll_toolkit>.
