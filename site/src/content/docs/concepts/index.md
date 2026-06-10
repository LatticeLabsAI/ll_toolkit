---
title: Concepts
description: Background explainers on how neural networks read and generate CAD — tokenization, generation architectures, and the honest state of the field.
sidebar:
  label: Overview
  order: 1
---

These pages explain the ideas the LatticeLabs Toolkit is built on. They are
**conceptual background** — orientation, not API docs — adapted from the
project's research notes.

A note on honesty: where these pages cite validity rates or benchmark numbers,
those come from the **published research literature** (DeepCAD, BrepGen,
Text2CAD, and others), not from LatticeLabs' own models. The toolkit's neural
packages ship **untrained**; the numbers describe what the *field* has achieved
and frame what is realistic, not what this codebase outputs today.

## The pages

- **[How geometry becomes tokens](/ll_toolkit/concepts/tokenization/)** — why
  continuous 3D geometry has to be quantized into discrete tokens, and how
  [geotoken](/ll_toolkit/geotoken/overview/) does it adaptively.
- **[How neural networks generate CAD](/ll_toolkit/concepts/how-neural-cad-generation-works/)**
  — the end-to-end pipeline from a prompt to a valid solid, and the two dominant
  strategies (token sequences vs. code generation).
- **[Inside CAD generation models](/ll_toolkit/concepts/inside-cad-generation-models/)**
  — the architectures themselves: transformers, diffusion, VQ-VAE, and the GNN
  encoders that read B-Rep.
- **[The reality of AI CAD generation](/ll_toolkit/concepts/the-reality-of-ai-cad-generation/)**
  — what actually ships in production, why "propose → dispose" is the only
  reliable pattern, and why [ll_gen](/ll_toolkit/ll_gen/overview/) is built that
  way.

## The one idea to take away

Neural networks output soft probability distributions over continuous spaces;
CAD requires **exact** integer topology (face counts, edge connectivity) and
precise parameters. Every technique below is, at bottom, a way to bridge that
gap — by quantizing geometry into tokens, by encoding topology implicitly, or by
making the network write *code* that a battle-tested CAD kernel executes.
