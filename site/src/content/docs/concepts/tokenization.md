---
title: How geometry becomes tokens
description: Why continuous 3D geometry must be quantized into discrete tokens, the three representation levels, and how geotoken allocates precision adaptively.
sidebar:
  label: Tokenization
  order: 2
---

Transformer models consume sequences of discrete tokens. CAD geometry is
continuous — coordinates, angles, surface parameters. **Tokenization** is the
bridge, and the way you tokenize constrains everything a model can learn.

## Why quantize at all? Why not just regress coordinates?

It is tempting to have a network predict raw float coordinates directly. In
practice this fails. DeepCAD's ablation is the clearest evidence: replacing
256-level classification with continuous regression degraded median Chamfer
distance by ~2.7×. The reason is that small mean-squared errors **break exact
geometric relationships** — lines meant to be parallel or perpendicular drift
just enough to be wrong. Classification over quantized levels lets a model snap
to discrete constraints (exactly 90°, exactly parallel) because it can represent
a multi-modal distribution rather than one blurred Gaussian mode.

So the field quantizes: normalize the solid into a fixed cube (commonly 2×2×2),
then map each continuous value to one of N discrete levels (often 64–256).

## Three levels of representation

| Level | What it encodes | Example |
|---|---|---|
| **Mesh** | Raw vertex/face geometry | quantized vertex positions |
| **Parametric** | Construction history | sketch-and-extrude command sequences (DeepCAD) |
| **Topology** | B-Rep graph | faces/edges/vertices + connectivity (UV-Net, BrepGen) |

The **sketch-and-extrude** representation dominates research because it mirrors
how engineers design: draw 2D profiles, extrude into 3D. DeepCAD's vocabulary is
just six commands — start-of-loop, line, arc, circle, extrude,
end-of-sequence — each carrying a unified 16-parameter vector.

## The cost of quantization

Quantization is lossy, and the loss propagates. Six-bit coordinates (64 levels)
on a 2×2×2 cube give ~0.03-unit resolution — fine for rough shape, too coarse for
precision engineering. Finer quantization expands the vocabulary quadratically,
straining attention. And distinct features can **collapse** into the same bin,
silently merging geometry that should stay separate.

## How geotoken handles this: adaptive precision

[geotoken](/ll_toolkit/geotoken/overview/) addresses the precision/size tradeoff
by spending bits where they matter. Its pipeline:

```text
vertices, faces
  → RelationshipPreservingNormalizer   (uniform scale into unit cube)
  → CurvatureAnalyzer + FeatureDensity  (per-vertex complexity)
  → BitAllocator                        (complexity → bits per vertex)
  → AdaptiveQuantizer                   (variable precision + collapse prevention)
  → TokenSequence
```

A complexity score combines curvature (weight 0.7) and feature density (0.3);
bits are assigned by percentile interpolation — flat regions get the base 8
bits, complex regions up to 12. After quantization, a spatial-hash pass (O(n))
detects vertices that collapsed onto the same value and perturbs one by the
minimum step so distinct features stay distinct.

geotoken offers three precision tiers — DRAFT (6-bit), STANDARD (8-bit, default),
and PRECISION (10-bit) — and produces six token categories (coordinate,
geometry, command, constraint, graph-node/edge, and graph-structure tokens),
encoded to integer IDs by a `CADVocabulary` of ~73k tokens.

## Topology is the hard part

Mesh and parameter quantization are tractable. **Topology** — the exact integer
structure of which faces share which edges — is not, because it cannot be
meaningfully interpolated. Approaches that directly predict discrete topology
achieve very low validity; the trick that works (BrepGen) is to encode adjacency
*implicitly* through geometry similarity — duplicate a shared edge under both
parent faces, then merge near-identical nodes in post-processing.

## See also

- [geotoken usage](/ll_toolkit/geotoken/usage/) — the tokenizers in code.
- [How neural networks generate CAD](/ll_toolkit/concepts/how-neural-cad-generation-works/)
  — what happens to these tokens downstream.
