---
title: ll_gen — Overview
description: Generation orchestration for CAD — neural propose, deterministic dispose in a sandbox. Ships DeepCAD-trained generators that produce measured-valid CAD via the construction-program route.
sidebar:
  label: Overview
  order: 1
  badge:
    text: Generates valid CAD
    variant: success
---

**ll_gen** orchestrates generative CAD modeling around a simple idea: **propose,
then dispose**. A generator proposes a candidate (a neural sample or a deterministic
template), and a deterministic *dispose* stage executes that proposal in a sandboxed
geometry kernel ([OCC](https://dev.opencascade.org/) /
[CadQuery](https://cadquery.readthedocs.io/)) to produce — and verify — real
geometry. The verification result feeds an RL alignment loop.

## How it works

```text
GenerationOrchestrator.generate(prompt)
  → propose
      ├─ autoregressive command generator   (the construction PROGRAM)
      ├─ latent diffusion over a program autoencoder
      └─ deterministic template
  → dispose  (OCC / CadQuery sandbox)   → real solid + validity
  → verification + feedback (reward)
```

## Generators that produce valid CAD

The robust route to valid CAD — the one DeepCAD/Text2CAD take — is to generate the
**construction program** (sketch + extrude commands) and *execute* it, so the kernel
builds a watertight solid, rather than generating B-rep faces that must be sewn. Two
trained generators take this route and run natively in **MLX on Apple Silicon**:

- **Autoregressive command generator** (`ll_gen/mlx/ar_generator_mlx.py`) — a causal
  transformer over the CAD command vocabulary, trained on ~38k real DeepCAD programs,
  sampled token-by-token → executed. **Measured validity {{metric.ll_gen.ar.validity}}**
  ({{metric.ll_gen.ar.validFraction}}), **{{metric.ll_gen.ar.distinct}} distinct** shapes.
- **Latent diffusion** (`ll_gen/mlx/latent_diffusion_mlx.py`) — diffuses the latent of
  a program autoencoder and decodes autoregressively. **Sampled-z validity
  {{metric.ll_gen.latentDiffusion.sampledZValidity}}** ({{metric.ll_gen.latentDiffusion.validFraction}}),
  **{{metric.ll_gen.latentDiffusion.distinct}} distinct**. The validity comes from the
  execution-respecting decoder; the diffusion contributes the diverse latent prior
  ({{metric.ll_gen.latentDiffusion.distinct}} distinct vs a predict-the-mean baseline's
  {{metric.ll_gen.latentDiffusion.baselineDistinct}}).

Two earlier routes are superseded because of their *representation*, not their
training: the command-VAE's parallel (non-autoregressive) decoder is primitive-limited
(~0–12% valid, posterior-collapses), and the raw-geometry diffusion produced **0** valid
solids — it denoised independent face grids that never mate when sewn. A faithful MLX
port of the command-VAE (`ll_gen/mlx/vae_mlx.py`) reproduces the PyTorch model exactly
(parity-verified) but inherits that limitation.

## Honest validity

Validity is **measured through the real kernel** and gated on a **non-degenerate solid**
(a closed solid with positive volume) — `BRepCheck` alone passes volume-less shells, so
gating on it would over-report. The harness also reports `num_distinct` (rounded
bounding boxes) as a mode-collapse guard, so a high rate from one repeated trivial shape
is visible, not hidden. (`GenerationMetrics.is_valid_solid`.)

## Running it

```bash
# train + measure the autoregressive command generator (Apple Silicon / MLX)
python {{script.ll_gen.arGenerator}} --mode train

# train the latent-diffusion generator and measure sampled-z validity
python {{script.ll_gen.latentDiffusion}} --mode train

# the orchestration / RL training entry point
python -m ll_gen.training.run --help
```

## Status

:::tip[Generators trained; validity measured through the real kernel]
The orchestration, dispose sandbox, verification, and RL loop run end-to-end, and
ll_gen now ships **trained generators that produce measured-valid CAD** on the DeepCAD
distribution — the autoregressive command generator ({{metric.ll_gen.ar.validity}} valid)
and the latent diffusion ({{metric.ll_gen.latentDiffusion.sampledZValidity}} valid), each
gated on real non-degenerate solids. Native-MLX trainers
run on Apple Silicon. Scope is stated honestly: these are trained on DeepCAD parametric
command sequences (sketch + extrude), and validity is measured on that distribution.
:::

Use the sidebar for **Installation**, **Usage**, and the **API Reference**.
