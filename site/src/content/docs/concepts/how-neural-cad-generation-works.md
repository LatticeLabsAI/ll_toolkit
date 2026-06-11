---
title: How neural networks generate CAD
description: The end-to-end pipeline from prompt to valid solid — the two dominant strategies, conditioning, reconstruction, and why validity is hard.
sidebar:
  label: How NNs generate CAD
  order: 3
---

Modern CAD generation turns text or images into valid 3D solid geometry through
a pipeline that bridges continuous neural predictions with the discrete, exact
world of parametric CAD. The core challenge is fundamental: networks output soft
probability distributions; CAD requires exact integer topology and precise
parameters.

## Two dominant strategies

The field has converged on two approaches:

1. **Direct neural prediction** — train a specialized network (transformer,
   diffusion model, VQ-VAE) to output a tokenized CAD representation, then
   reconstruct it into a solid. Validity rates of roughly 46–83% depending on
   method.
2. **Code generation** — have a pretrained LLM emit CadQuery / OpenSCAD / KCL
   code, and let a CAD kernel execute it. Compilation/validity rates of 90–100%
   because the kernel guarantees topological consistency.

Both are represented in the LatticeLabs Toolkit:
[ll_stepnet](/ll_toolkit/ll_stepnet/overview/) provides direct-prediction models
(VAE/diffusion/VQ-VAE), and [ll_gen](/ll_toolkit/ll_gen/overview/) can take the
code-generation route — proposing code that a CadQuery sandbox executes.

## The pipeline, end to end

```text
prompt / image
  → conditioning (text/image → dense vectors)
  → generation (token sequence OR code OR latent sample)
  → reconstruction (CAD kernel executes → B-Rep solid)
  → validation (watertight? manifold? reasonable?)  → optional feedback loop
```

### Conditioning

Text and images become steering signals via cross-attention (Text2CAD uses a
frozen BERT encoder + an adaptive layer feeding a transformer decoder), adaptive
normalization, or token concatenation (CAD-MLLM concatenates text, image, and
point-cloud tokens as prefixes to an LLM). Image conditioning typically uses
DINOv2 or CLIP encoders.

### Generation

- **Autoregressive transformers** (DeepCAD, SkexGen) emit command tokens one at
  a time.
- **Diffusion** (BrepGen) denoises structured B-Rep latents in a cascade — face
  positions, then face geometry, then edges, then vertices.
- **VQ-VAE** (SkexGen, HNC-CAD) compresses to a small set of discrete codes from
  learned codebooks.

### Reconstruction

For sequence methods, predicted commands are executed directly in OpenCASCADE —
loops build wires, extrudes build solids. For B-Rep diffusion, decoded point
grids are fit to B-spline surfaces (`GeomAPI_PointsToBSplineSurface`), duplicate
nodes are merged to recover topology, and trimmed faces are sewn into a solid.

## Why validity is hard

Validity rates remain modest because of the **topology gap**: B-Rep requires
exact integer face counts and binary adjacency that cannot be interpolated. Even
one missing edge breaks watertightness. Reported validity (watertight solids)
clusters around 46–63% for direct B-Rep methods on the DeepCAD distribution,
with newer holistic-latent approaches pushing past 80%. Common failure modes:
non-watertight solids, self-intersections, dangling edges, broken
face-edge-vertex connectivity.

:::note[Measured in this toolkit]
This shows up concretely in [ll_gen](/ll_toolkit/ll_gen/overview/): a diffusion that
denoises **independent face grids** and sews them reaches **0** valid solids on the
honest solid+volume gate — the sampled faces never mate. Re-targeting the same idea to
diffuse a **construction-program** latent and decode it autoregressively (so the kernel
*builds* the solid) reaches **0.934** sampled-z validity, and the autoregressive command
model reaches **0.914** — both measured through the real kernel. These are the toolkit's
own numbers, not the literature figures above.
:::

## The role of reinforcement learning

A recurring result: **RL alignment with solver/kernel feedback** unlocks large
gains. Aligning a sketch-constraint model with solver rewards moved
fully-constrained output from ~9% (no alignment) to ~93%. The same idea — reward
the model for producing geometry the kernel accepts — is exactly what
[ll_gen](/ll_toolkit/ll_gen/overview/)'s REINFORCE loop does, rewarding proposals
that dispose into a valid closed solid.

## The pragmatic conclusion

The most reliable production path is **hybrid**: the network writes code, and a
proven CAD kernel executes and validates it. This is the
[propose → dispose pattern](/ll_toolkit/concepts/the-reality-of-ai-cad-generation/)
the toolkit follows.
