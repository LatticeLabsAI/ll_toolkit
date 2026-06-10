---
title: ll_ocadr — Usage
description: Run HF-native optical CAD recognition — CLI and programmatic — feeding global + tiled-local geometry into an LLM.
sidebar:
  label: Usage
  order: 3
---

ll_ocadr encodes a global, full-resolution view of an object together with tiled
local chunks, projects those features into a language model's embedding space,
and lets the LLM reason over the combined tokens.

## CLI (HF-native)

```bash
python ll_ocadr/run_ll_ocadr_hf.py \
    --model Qwen/Qwen2-1.8B \
    --mesh part.step \
    --prompt "Describe this CAD part: <mesh>" \
    --max-new-tokens 64
```

- The `<mesh>` placeholder token is registered and the LM embeddings are resized
  at build time.
- `n_embed` is derived automatically from the chosen language model.
- `--no-cropping` uses the global view only; otherwise the mesh is spatially
  chunked into local tiles.

## Programmatic

```python
from ll_ocadr.run_ll_ocadr_hf import build_model_and_tokenizer, run_inference

model, tokenizer, config, processor = build_model_and_tokenizer("Qwen/Qwen2-1.8B")
text = run_inference(model, processor, tokenizer, "part.stl", "Describe <mesh>")
print(text)
```

## How the pieces fit

```text
mesh / STEP file
   │  vllm/process/  (chunkers, mesh/STEP loaders, tokenizer glue)
   ▼
GeometryNet (PointNet++ local) + ShapeNet (ViT global)
   │  concatenate → MLP projector → LLM embedding space
   ▼
LatticelabsOCADRForCausalLM → HF language model (forward / generate)
```

- `vllm/latticelabs_ocadr.py` — `LatticelabsOCADRForCausalLM`.
- `vllm/lattice_encoder/` — `GeometryNet`, `ShapeNet`, MLP projector.

## Tests

```bash
# Fast unit tests (encoders, chunkers, fixtures) — no model download:
cd ll_ocadr && pytest tests/ -m "not slow"

# End-to-end + CLI tests (tiny offline LM, still no network):
cd ll_ocadr && pytest tests/ -m slow
```

:::caution[vLLM serving is experimental / not functional]
The classes named after vLLM's integration points and the `run_ll_ocadr.py` /
`run_ll_ocadr_eval_batch.py` scripts do **not** run today —
`LatticelabsOCADRForCausalLM` is not registered via vLLM's `ModelRegistry` and
does not implement `SupportsMultiModal`. Use the HF-native path above. See the
[Overview](/ll_toolkit/ll_ocadr/overview/).
:::

## Related

- [Overview](/ll_toolkit/ll_ocadr/overview/) · [Installation](/ll_toolkit/ll_ocadr/installation/)
- Background: [The reality of AI CAD generation](/ll_toolkit/concepts/the-reality-of-ai-cad-generation/).
