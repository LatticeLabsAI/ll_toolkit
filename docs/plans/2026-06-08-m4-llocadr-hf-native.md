# Plan M4 — `ll_ocadr` HF-Native Inference + Pytest Suite

| | |
|---|---|
| **Spec** | SPEC-1 §5 M4, §3.1 FR-O1…O4, §4.2 |
| **Goal** | `LatticelabsOCADRForCausalLM` runs end-to-end via plain HuggingFace with a real pytest suite (encoders, chunkers, processor, e2e forward/generate). vLLM documented as future — no false "it works" claims. |
| **Depends on** | — (independent of M1–M3) |
| **Owner** | Maintainer · **Mode** Inline/sequential · **Tests** TDD · **Commits** per task |
| **Status** | Not started · **Decision baked in:** HF-native only; vLLM = documented future |

## Context (verified)
- Real model: `LatticelabsOCADRForCausalLM(nn.Module)` (latticelabs_ocadr.py:17), `forward` (:257), HF-native `generate` (:311 → `language_model.generate(inputs_embeds=...)` :354), `get_input_embeddings` (:217), `_mesh_to_embedding` (:54), `_format_chunk_grid` (:182).
- Real encoders: `GeometryNet` PointNet++ (`vllm/lattice_encoder/geometry_net.py`, FPS + set abstraction), `ShapeNet` ViT (`shape_net.py`).
- Real processors: `vllm/process/{file_content_chunker,mesh_process,step_process,step_tokenizer,ngram_norepeat}.py`.
- Gaps: model **not** registered into vLLM (`ModelRegistry`/`SupportsMultiModal` absent); **no pytest** — `ll_ocadr/test_ll_ocadr.py` is a manual `__main__` script.

## Pre-flight
- Read `latticelabs_ocadr.py` fully (model + `LLOCADRProcessingInfo` :372, `LLOCADRMultiModalProcessor` :426).
- Read `vllm/config.py` (model config; pick the smallest HF causal LM for tests — resolve **OQ4**).
- Read the encoders' forward signatures and expected input shapes.

## Tasks

### T4.1 — Test scaffold (OpenMP-safe)
- Create `ll_ocadr/tests/conftest.py` (torch imported first, `OMP_NUM_THREADS=1`) and `ll_ocadr/tests/__init__.py`.
- Add fixtures: a synthetic point cloud `(N,6)` tensor, a tiny synthetic mesh file (write a few-triangle STL to a tmp path), and a "tiny LM" config (OQ4).
- **Commit:** `test(ll_ocadr): add pytest scaffold + synthetic fixtures`

### T4.2 — Encoder unit tests (TDD against real shapes)
- `tests/unit/test_geometry_net.py`: `GeometryNet().forward(points)` returns the documented shape (e.g. `[B, C, npoint]`); gradients flow; FPS path works with and without `torch_cluster` (skip fast path if absent).
- `tests/unit/test_shape_net.py`: `ShapeNet`/`PointPatchEmbedding` forward shape + cls token presence; document the known `num_patches=256` hardcode (shape_net.py) — test the actual behavior, don't assert the ignored `patch_size` param.
- **Commit:** `test(ll_ocadr): unit tests for GeometryNet + ShapeNet encoders`

### T4.3 — Processor/chunker unit tests
- `tests/unit/test_chunkers.py`: `file_content_chunker` binary-STL-vs-ASCII detection (size heuristic), `mesh_process` mesh→BRepData fields, `step_process` topology extraction on `part.step` (repo root has one).
- Cover the known edge cases from `Review.md`: OBJ chunk vertex inclusion, entity-ID precision, `_chunk_brep` index alignment — assert current behavior and note any lossy path.
- **Commit:** `test(ll_ocadr): unit tests for chunkers + processors`

### T4.4 — End-to-end forward/generate smoke test
- `tests/integration/test_e2e_generate.py` (mark `slow`/`requires_torch`): build `LatticelabsOCADRForCausalLM` with the tiny LM (OQ4) + a tiny encoder config; feed a synthetic mesh + prompt; run `forward` (assert logits shape) then `generate(max_new_tokens=4)` (assert tokens produced, no exception).
- If a real defect blocks e2e (e.g., shape mismatch in `_format_chunk_grid`), fix it as part of the task — this is the "make it actually run" step.
- **Commit:** `test(ll_ocadr): end-to-end forward+generate smoke test with tiny LM`

### T4.5 — HF-native inference script
- Add `ll_ocadr/run_ll_ocadr_hf.py`: args `--model`, `--mesh` (STEP/STL/OBJ), `--prompt`, `--max-new-tokens`, `--device`; load model via HF, process the file, run `generate`, print text. No vLLM import.
- Add a `tests/integration/test_inference_script.py` invoking it on `part.step` with the tiny LM (mark `slow`).
- **Commit:** `feat(ll_ocadr): HF-native inference script run_ll_ocadr_hf.py`

### T4.6 — Honest docs + vLLM-as-future
- Update `ll_ocadr/README.md`: state HF-native is the supported path; vLLM registration (`ModelRegistry` + `SupportsMultiModal`) is **future work**, not functional today.
- In `latticelabs_ocadr.py` / `run_ll_ocadr*.py` docstrings, mark the vLLM entry points as experimental/not-wired. Remove any text implying working vLLM serving.
- **Commit:** `docs(ll_ocadr): document HF-native path; mark vLLM as future work`

## Verification
```bash
cd ll_ocadr && pytest tests/ -v -m "not slow"
pytest tests/ -v -m slow         # tiny-LM e2e (network/model download permitting)
python run_ll_ocadr_hf.py --model <tiny-lm> --mesh ../part.step --prompt "Describe this part." --max-new-tokens 16
```

## Milestone risks
- **R3 (e2e slow/flaky)** — use the smallest possible LM; mark `slow`/`requires_torch`; keep encoder/processor unit tests as the fast default suite.
- **Model download in CI** — pin a tiny model; if offline, `importorskip`/skip with a clear reason; never fake a pass.
- **Real shape bugs surface in T4.4** — expected; fix in place (that's the point of "make it run").

## Done checklist
- [ ] `pytest ll_ocadr/tests -m "not slow"` green (encoders + processors).
- [ ] e2e forward+generate smoke test green with a tiny LM.
- [ ] `run_ll_ocadr_hf.py` produces text on `part.step`.
- [ ] README + docstrings: no false vLLM claims; vLLM marked future.
