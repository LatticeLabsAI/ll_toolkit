---
title: 'Tutorial: OCADR HF inference'
description: Feed a mesh or STEP file plus a prompt into a language model with ll_ocadr and read its description, using the HF-native path.
sidebar:
  label: OCADR HF inference
  order: 5
---

In this tutorial you will run [ll_ocadr](/ll_toolkit/ll_ocadr/overview/)'s
HF-native pipeline: encode a 3D object's geometry into a language model's
embedding space and have the model describe it. Allow ~15 minutes (first run
downloads the chosen LM).

:::note[This HF path's projector is untrained — the trained model is the MLX one]
This tutorial exercises the **HF-native** PyTorch path with an **untrained**
geometry→text projector — a small object runs end to end, but treat the text as a
mechanism demo, not an accurate caption. A **genuinely trained, geometry-grounded model**
exists via the native-MLX path (`ll_ocadr/mlx/train_ocadr_mlx.py`): it reads the geometry
and verbalizes the correct class at **0.919 vs a 0.313 shuffled-mesh control**. The vLLM
serving path is experimental and not functional. See the
[Overview](/ll_toolkit/ll_ocadr/overview/).
:::

## 1. Run the CLI

The `<mesh>` placeholder in the prompt marks where the encoded geometry tokens go.

```bash
python ll_ocadr/run_ll_ocadr_hf.py \
    --model Qwen/Qwen2-1.8B \
    --mesh part.stl \
    --prompt "Describe this CAD part: <mesh>" \
    --max-new-tokens 64
```

- `--model` is any HF causal LM; `n_embed` is derived from it automatically.
- `--mesh` accepts a mesh (STL/OBJ/PLY) or a STEP file (STEP needs
  `pythonocc-core`).
- `--no-cropping` uses the global view only; otherwise the object is chunked into
  local tiles plus a global view.

## 2. Do it programmatically

```python
from ll_ocadr.run_ll_ocadr_hf import build_model_and_tokenizer, run_inference

model, tokenizer, config, processor = build_model_and_tokenizer("Qwen/Qwen2-1.8B")
text = run_inference(model, processor, tokenizer, "part.stl", "Describe <mesh>")
print(text)
```

`build_model_and_tokenizer` constructs `LatticelabsOCADRForCausalLM`, registers
the `<mesh>` token, and resizes the LM embeddings. `run_inference` chunks the
geometry, encodes it (GeometryNet PointNet++ local + ShapeNet ViT global →
MLP projector), splices the geometry embeddings into the prompt, and calls the
LM's `generate`.

## 3. Run the test suite (no network)

The fast tests exercise the encoders and chunkers with no model download; the
slow tests run an end-to-end forward/generate on a tiny offline LM.

```bash
cd ll_ocadr
pytest tests/ -m "not slow"   # encoders, chunkers, fixtures
pytest tests/ -m slow         # e2e + CLI on a tiny offline LM
```

## Where to next

- [ll_ocadr Usage](/ll_toolkit/ll_ocadr/usage/) for the architecture details.
- [The reality of AI CAD generation](/ll_toolkit/concepts/the-reality-of-ai-cad-generation/)
  for how geometry-aware LLM input fits the bigger picture.
