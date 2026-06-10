---
title: Inside CAD generation models
description: The generation architectures themselves — transformer-VAE, structured diffusion, disentangled VQ-VAE codebooks, and the GNN encoders that read B-Rep.
sidebar:
  label: Inside the models
  order: 4
---

This page looks inside the architectures, with the concrete shapes and choices
that make them work. The numbers are drawn from the primary research papers and
describe the field — not LatticeLabs' (untrained) models.

## The four architectural families

### Transformer-VAE (DeepCAD)

DeepCAD's autoencoder is, notably, *not* a true VAE — there is no KL term or
reparameterization. A 4-layer / 8-head transformer encoder turns a 60-command
sequence into a 256-dim latent via average pooling. The decoder operates
**non-autoregressively** (DETR-style learned queries cross-attend to the latent),
predicting all commands in parallel through two heads: command type (6-class) and
parameters (16 × 256-class). Because the latent space has no tractable sampling
distribution, a separate **WGAN-gp** is trained to map noise → latent for
unconditional generation.

### Structured diffusion (BrepGen)

BrepGen denoises B-Rep latents in a four-stage top-down cascade — face boxes,
face geometry, edge boxes, then edge-vertex geometry — each conditioned on the
previous. Each denoiser is a 12-layer / 12-head transformer; training uses
standard DDPM (1000 steps), inference uses PNDM (200 steps). Topology is encoded
**implicitly** by duplicating shared edges/vertices under each parent face and
merging near-identical nodes afterward.

### Disentangled VQ-VAE (SkexGen)

SkexGen factors a model into three independent codebooks — topology, geometry,
extrusion — maintained with EMA updates (decay 0.99) and a commitment loss
(β = 0.25), skipping quantization for the first 25 epochs to stabilize training.
A whole CAD model compresses to ~10 discrete codes, and changing topology codes
while holding geometry codes produces structurally different parts in the same
style — **controllable** generation.

### GNN encoders (UV-Net, BRepNet)

Graph neural networks excel at *reading* B-Rep, not generating it. UV-Net samples
each face on a 10×10 UV grid through surface CNNs, then message-passes over the
face-adjacency graph. BRepNet convolves over oriented coedges using topological
walks (next / previous / mating coedge / parent face) as kernels — with only
~360K parameters it reaches 77% IoU on Fusion 360 segmentation. These encoders
produce 64–128-dim embeddings used for retrieval, segmentation, and conditioning.

## Why classification beats regression (again)

A theme worth repeating: predicting each geometric parameter as a **classification
over quantized levels** outperforms regressing a float, because classification
represents multi-modal distributions and snaps to discrete constraints. The
effect is architecture-dependent — when structure is predicted first and
parameters regressed afterward (Img2CAD), the advantage can reverse — but for
independent per-parameter prediction, quantize-and-classify wins.

## How this maps to the toolkit

- [ll_stepnet](/ll_toolkit/ll_stepnet/overview/) exports the generative models
  the toolkit uses — `STEPVAE`, `StructuredDiffusion`, `VQVAEModel` — plus a
  `CADGenerationPipeline`, and the transformer + GNN **encoder** used for
  classification, property prediction, similarity, and captioning.
- [geotoken](/ll_toolkit/concepts/tokenization/) provides the UV-grid / graph
  tokenization these models consume.
- [ll_gen](/ll_toolkit/ll_gen/overview/) drives the generators and closes the
  loop with RL.

## The 2024–2025 shift

The recent wave replaced bespoke transformers with **pretrained LLMs**
(STEP-LLM, FlexCAD, CAD-Llama), made **reinforcement learning** standard, and
matured direct B-Rep generation (single-stage diffusion; topology/geometry
decoupling; neural intersection). The direction of travel is clear: let large
models propose, let kernels and solvers verify.
