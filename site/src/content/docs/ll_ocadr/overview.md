---
title: ll_ocadr — Overview
description: Optical CAD Recognition — encode global + tiled-local 3D geometry into an LLM's embedding space. HF-native inference today; vLLM serving is experimental.
sidebar:
  label: Overview
  order: 1
  badge:
    text: HF-native
    variant: note
---

**ll_ocadr** (LatticeLabs Optical CAD Recognition) is a DeepSeek-OCR-inspired
pipeline for feeding 3D CAD/mesh geometry into a large language model. Instead of
document images, it processes CAD (STEP) and mesh (STL/OBJ/PLY) files: it encodes
a **global, full-resolution** view of the object together with **tiled local
chunks**, projects those features into the LLM's embedding space, and lets the
LLM reason over the combined tokens.

## Architecture

```text
mesh / STEP file
   │  (process/: chunkers, mesh/STEP loaders)
   ▼
GeometryNet (PointNet++ local features) + ShapeNet (ViT global features)
   │  concatenate → MLP projector → LLM embedding space
   ▼
LatticelabsOCADRForCausalLM → HF language model (forward / generate)
```

- `vllm/latticelabs_ocadr.py` — the multimodal model `LatticelabsOCADRForCausalLM`.
- `vllm/lattice_encoder/` — `GeometryNet`, `ShapeNet`, and the MLP projector.
- `vllm/process/` — file-content chunkers, mesh/STEP loaders, tokenizer glue.

## Inference (HF-native, supported)

```bash
python ll_ocadr/run_ll_ocadr_hf.py \
    --model Qwen/Qwen2-1.8B \
    --mesh part.step \
    --prompt "Describe this CAD part: <mesh>" \
    --max-new-tokens 64
```

The `<mesh>` placeholder token is registered (and the LM embeddings resized) at
build time; `n_embed` is derived automatically from the chosen language model.

## vLLM serving — experimental / future

:::caution[vLLM is not functional today]
The classes under `vllm/` named after vLLM's integration points
(`LLOCADRProcessingInfo`, `LLOCADRMultiModalProcessor`) and the
`run_ll_ocadr.py` / `run_ll_ocadr_eval_batch.py` scripts are **experimental and
not functional today**. `LatticelabsOCADRForCausalLM` is **not** registered via
vLLM's `ModelRegistry` and does **not** implement `SupportsMultiModal`, so it
cannot be served by the vLLM engine as written. Use the HF-native path above.
:::

## Status

:::note[Maturity: HF-native runnable; models untrained]
The HF-native model, encoders, and inference script are real and tested. Like the
other neural packages, ll_ocadr **ships no trained weights** — a tiny offline LM
is used in tests. vLLM serving is planned future work.
:::

Use the sidebar for **Installation**, **Usage**, and the **API Reference**.
