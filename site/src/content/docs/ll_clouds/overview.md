---
title: ll_clouds — Overview
description: Point-cloud processing and analysis for the LatticeLabs CAD toolkit — a dependency-light NumPy/SciPy library.
sidebar:
  label: Overview
  order: 1
  badge:
    text: Core
    variant: note
---

**ll_clouds** is point-cloud processing and analysis for the LatticeLabs CAD
toolkit — a standalone, dependency-light library (NumPy + SciPy; trimesh only for
mesh I/O) covering the core point-cloud workflow.

## What it does

- **I/O** — read/write PLY, PCD, XYZ; sample point clouds from meshes.
- **Preprocessing** — normalize (center + unit-scale), voxel downsample,
  farthest-point downsample (FPS), statistical outlier removal.
- **Features** — per-point normals (k-NN PCA), curvature, geometry statistics.
- **Registration** — point-to-point ICP.
- **Segmentation** — RANSAC plane fitting, Euclidean clustering.

## Bridges to sibling packages

Optional bridges convert documents/inputs from [`cadling`](/ll_toolkit/cadling/overview/)
(CAD document processing) and [`ll_ocadr`](/ll_toolkit/ll_ocadr/overview/)
(optical CAD recognition) into a `PointCloud`. These imports are **lazy**, so the
core library has no hard dependency on those packages — `import ll_clouds` pulls
in neither cadling, ll_ocadr, nor torch.

## Data model

The central type is a Pydantic `PointCloud` (points, optional
normals/colors/labels, metadata) with `arbitrary_types_allowed=True` for NumPy
arrays — consistent with the rest of the monorepo.

## Status

:::note[Maturity: core library]
ll_clouds is a real, installable core library with a full test suite. It uses
NumPy/SciPy/trimesh only — open3d is intentionally **not** a dependency.
:::

Use the sidebar for **Installation**, **Usage**, and the **API Reference**.
