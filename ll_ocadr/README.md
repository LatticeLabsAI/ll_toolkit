# LatticeLabs Optical CAD Recognition (ll_ocadr)

A DeepSeek-OCR-inspired pipeline for feeding 3D CAD/mesh geometry into a large
language model. Instead of document images, it processes CAD (STEP) and mesh
(STL/OBJ/PLY) files: it encodes a **global full-resolution** view of the object
together with **tiled local chunks**, projects those features into the LLM's
embedding space, and lets the LLM reason over the combined tokens.

## Architecture

```
mesh / STEP file
   │  (process/: chunkers, mesh/STEP loaders)
   ▼
GeometryNet (PointNet++ local features)  +  ShapeNet (ViT global features)
   │  concatenate → MLP projector → LLM embedding space
   ▼
LatticelabsOCADRForCausalLM  →  HF language model (forward / generate)
```

- `vllm/latticelabs_ocadr.py` — the multimodal model (`LatticelabsOCADRForCausalLM`).
- `vllm/lattice_encoder/` — `GeometryNet`, `ShapeNet`, and the MLP projector.
- `vllm/process/` — file content chunkers, mesh/STEP loaders, tokenizer glue.

## Inference (supported: HF-native)

The supported, tested inference path uses plain HuggingFace `transformers` — no
vLLM required:

```bash
python ll_ocadr/run_ll_ocadr_hf.py \
    --model Qwen/Qwen2-1.8B \
    --mesh part.step \
    --prompt "Describe this CAD part: <mesh>" \
    --max-new-tokens 64
```

Programmatic use:

```python
from ll_ocadr.run_ll_ocadr_hf import build_model_and_tokenizer, run_inference

model, tokenizer, config, processor = build_model_and_tokenizer("Qwen/Qwen2-1.8B")
text = run_inference(model, processor, tokenizer, "part.stl", "Describe <mesh>")
print(text)
```

`n_embed` is derived automatically from the chosen language model, and the
`<mesh>` placeholder token is registered (with the LM embeddings resized) at
build time. The `--no-cropping` flag uses the global view only; otherwise the
mesh is spatially chunked into local tiles.

## vLLM serving — experimental / future work

The classes under `vllm/` named after vLLM's integration points
(`LLOCADRProcessingInfo`, `LLOCADRMultiModalProcessor`) and the
`run_ll_ocadr.py` / `run_ll_ocadr_eval_batch.py` scripts are **experimental and
not functional today**:

- `LatticelabsOCADRForCausalLM` is **not** registered via vLLM's
  `ModelRegistry` and does **not** implement vLLM's `SupportsMultiModal`
  interface, so it cannot be served by the vLLM engine as written.
- Wiring the model into the vLLM runtime (registration + the multimodal
  interface + KV-cache token accounting) is planned future work.

Until then, use the HF-native path above.

## Tests

```bash
# Fast unit tests (encoders, chunkers, fixtures) — no model download:
cd ll_ocadr && pytest tests/ -m "not slow"

# End-to-end + CLI tests (tiny offline LM, still no network):
cd ll_ocadr && pytest tests/ -m slow
```

STEP-file tests require `pythonocc-core` (conda-only) and skip automatically
when it is not installed.

## Dependencies

Core inference needs `torch`, `transformers`, `trimesh`, `numpy`, `scipy`.
STEP (B-Rep) support additionally needs `pythonocc-core` (conda-forge). The
declared `vllm` dependency is only for the experimental serving path above.
