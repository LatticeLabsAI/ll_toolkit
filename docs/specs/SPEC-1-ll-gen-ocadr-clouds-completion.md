# SPEC-1: Completion of `ll_gen`, `ll_ocadr`, and `ll_clouds`

| Field | Value |
|---|---|
| **Spec ID** | SPEC-1 |
| **Title** | Make `ll_gen`, `ll_ocadr`, and `ll_clouds` 100% working and real |
| **Status** | In progress — M1 ✅ (PR #4, merged) · M4 ✅ (PR #5, merged) · M5 ✅ (PR #6, merged) · M2 ✅ (PR #7, in review) · M6 ◐ (quality gate + truth-up) · M3 ⬜ (training, pending data+compute) |
| **Author** | LatticeLabs (LayerDynamics) |
| **Owner** | Maintainer (solo) |
| **Created** | 2026-06-08 |
| **Source material** | `STATUS.md` (verified viability/status audit), `cadling/docs/RequiredToBeCorrected.md`, `Review.md`, direct code reading (file:line citations throughout) |
| **Related** | `docs/GenerationImplementation.md`, `docs/HowNNsGenerateCAD.md`, `docs/AICADGenerationDoesNotWorkHowYouThink.md`, `docs/training_data_requirements.md` |

---

## 1. Problem Statement

Three packages in the LatticeLabs monorepo are not production-real today. The defects are verified, not assumed:

1. **`ll_gen` — PARTIAL.** The deterministic-dispose path and the LLM-code-propose path are real and run. The **neural-latent propose + RL track is dead code**: every neural generator imports classes from modules that do not exist.
   - `ll_gen/ll_gen/generators/neural_vae.py:430` → `from ll_stepnet.stepnet.models import STEPVAE` — module `ll_stepnet/stepnet/models.py` **does not exist**.
   - `ll_gen/ll_gen/generators/neural_vae.py:431`, `neural_vqvae.py:425` → `from ll_stepnet.stepnet.pipeline import CADGenerationPipeline` — module is actually `ll_stepnet/stepnet/generation_pipeline.py`.
   - `ll_gen/ll_gen/generators/neural_diffusion.py:418` → `from ll_stepnet.stepnet.models import StructuredDiffusion` — wrong module.
   - `ll_gen/ll_gen/generators/neural_vqvae.py:424` → `from ll_stepnet.stepnet.models import VQVAEModel` — wrong module.
   - `ll_gen/ll_gen/generators/neural_vqvae.py:222` → `from ll_stepnet.stepnet.vqvae import DisentangledCodebooks` — correct submodule, but wrong `ll_stepnet.` prefix.
   - Root cause: the installed top-level package is **`stepnet`** (per `ll_stepnet/pyproject.toml:6` `name = "ll-stepnet"`; the package dir is `stepnet/`; `README.md` uses `from stepnet.encoder import ...`). The `ll_stepnet.stepnet.*` import path is wrong on two counts: the prefix, and the non-existent `models`/`pipeline` modules. The real classes live in `stepnet/vae.py` (`STEPVAE`, vae.py:35), `stepnet/diffusion.py` (`StructuredDiffusion`, diffusion.py:348), `stepnet/vqvae.py` (`VQVAEModel` vqvae.py:749, `DisentangledCodebooks` vqvae.py:269), and `stepnet/generation_pipeline.py` (`CADGenerationPipeline`, generation_pipeline.py:58) — all re-exported from `stepnet/__init__.py` (lines 69, 82, 108, 109).
   - **Consequence:** `GenerationOrchestrator._propose_neural_*` (orchestrator.py:447+) and `RLAlignmentTrainer.train_step` → `generator.generate_for_training(prompt)` (rl_trainer.py:170) raise `ImportError` the moment the neural path is exercised. The REINFORCE loop (rl_trainer.py:182–219) is correct but unreachable.

2. **`ll_ocadr` — PARTIAL.** `LatticelabsOCADRForCausalLM` (latticelabs_ocadr.py:17) is a genuine HF multimodal `nn.Module` with real PointNet++ (`vllm/lattice_encoder/geometry_net.py`) and ViT (`shape_net.py`) encoders, a real `forward` (latticelabs_ocadr.py:257) and a working HF-native `generate` (latticelabs_ocadr.py:311, delegates to `language_model.generate(inputs_embeds=...)`). But: (a) the "vLLM integration" headline is aspirational — the model is **not** registered via vLLM `ModelRegistry` and does not inherit `SupportsMultiModal`, so it cannot run under the vLLM engine; (b) there is **no pytest suite** — `ll_ocadr/test_ll_ocadr.py` is a manual `__main__` CLI script.

3. **`ll_clouds` — EMPTY.** Only `ll_clouds/pyproject.toml` exists (description: *"Pointcloud Processing and Analyzing for the LatticeLabs CAD toolkit"*). Zero source files. Cannot be installed, imported, or tested.

A cross-cutting fact applies to all neural work: **no trained checkpoints exist** for any project model (tree-wide search finds only third-party `resources/.../TITAN-1M.bin` and `.../best.ckpt`). Until training is run, every neural output is noise.

---

## 2. Goals and Non-Goals

### 2.1 Goals (definition of "100% working and real", per maintainer decisions)

- **G1 — `ll_gen` neural track runs end-to-end.** All broken imports fixed; `GenerationOrchestrator.generate(...)` completes for VAE, diffusion, and VQ-VAE paths without import/wiring errors; `generate_for_training` produces a proposal with log-probs; `RLAlignmentTrainer.train_step` performs a real gradient update.
- **G2 — Small real checkpoints exist.** Train VAE, diffusion, and VQ-VAE on a **small public dataset subset** (DeepCAD/ABC) to produce committed-or-downloadable checkpoints that generate **non-noise** output (valid-CAD rate measurably above random-init baseline). Not production quality — proof of life.
- **G3 — `ll_ocadr` is HF-native runnable + tested.** `LatticelabsOCADRForCausalLM.generate(...)` runs through a documented HF path with a real pytest suite covering encoders, chunking, processor, and an end-to-end forward/generate smoke test (small model). vLLM is documented as a future path, not claimed as working.
- **G4 — `ll_clouds` is a real core point-cloud library.** Implements I/O, preprocessing, features, and registration/segmentation primitives with Pydantic data models and a full pytest suite.
- **G5 — Quality gates green.** For all three packages: `ruff`, `black --check`, `mypy` clean; pytest suites pass; no `raise NotImplementedError`, no stub/`pass`-only method bodies, no TODO markers; no fabricated/hardcoded outputs where real logic belongs.

### 2.2 Non-Goals

- **NG1 — Full production training.** No multi-week convergence runs, no full-dataset training, no SOTA benchmark chasing. (Deferred; G2 is "proof of life" only.)
- **NG2 — vLLM runtime registration.** Not in scope for `ll_ocadr`; HF-native only (documented future path).
- **NG3 — New model architectures.** Use the existing `stepnet` VAE/diffusion/VQ-VAE and `ll_ocadr` encoders as-is; fix wiring/training, do not redesign.
- **NG4 — Changes to `cadling`, `geotoken`, `ll_stepnet` internals** beyond what's required to expose existing classes (e.g., confirming `stepnet/__init__.py` re-exports). No refactor of those packages.
- **NG5 — The DeepSeek-OCR marketing framing.** Out of scope here (a docs/positioning matter tracked separately).

---

## 3. Requirements

### 3.1 Functional Requirements

#### `ll_gen`
- **FR-G1** Replace all broken imports in `generators/neural_vae.py`, `neural_diffusion.py`, `neural_vqvae.py` with correct paths: import `STEPVAE`, `StructuredDiffusion`, `VQVAEModel`, `DisentangledCodebooks`, `CADGenerationPipeline` from `stepnet` (top-level re-exports) — e.g. `from stepnet import STEPVAE, CADGenerationPipeline`. Remove every `ll_stepnet.stepnet.*` reference.
- **FR-G2** `NeuralVAEGenerator`, `NeuralDiffusionGenerator`, `NeuralVQVAEGenerator` each instantiate their `stepnet` model, move it to device, and run a real decode/sample producing a typed proposal (`CommandSequenceProposal` / `LatentProposal`).
- **FR-G3** `generate_for_training(prompt)` returns a proposal carrying accumulated `log_probs` and `entropy` on the live computation graph (VQ-VAE AR decoder sampling at neural_vqvae.py:~395–420 is the reference pattern).
- **FR-G4** `RLAlignmentTrainer.train_step` (rl_trainer.py:132) executes: generate → dispose (CadQuery subprocess) → reward (`feedback/reward_signal.py`) → advantage → `loss.backward()` → `optimizer.step()`, returning a metrics dict including `reward`, `advantage`, `baseline`, `loss`.
- **FR-G5** `GenerationOrchestrator.generate(...)` routes to neural paths via `_propose_neural_vae/diffusion/vqvae` and completes the full propose→dispose→verify loop, with `checkpoint_path` loading the G2 checkpoints when provided.
- **FR-G6** A runnable training entry point (CLI or `python -m`) wires a dataset loader (`datasets/deepcad_loader.py` `load_deepcad` / `datasets/abc_loader.py` `load_abc`) → generator → `RLAlignmentTrainer.train_epoch`, with checkpoint save/load.

#### `ll_ocadr`
- **FR-O1** `LatticelabsOCADRForCausalLM` is constructible from a small config (small HF LLM) and runs `forward` and `generate` end-to-end on a synthetic mesh/point-cloud input.
- **FR-O2** Mesh→embedding path (`_mesh_to_embedding` latticelabs_ocadr.py:54) and chunk formatting (`_format_chunk_grid` :182) produce correctly-shaped tensors feeding `get_input_embeddings` (:217).
- **FR-O3** A documented HF-native inference entry point exists (script under `ll_ocadr/` mirroring `run_ll_ocadr.py` but without vLLM), taking a mesh/STEP/STL file + prompt and returning generated text.
- **FR-O4** vLLM path is explicitly documented as future work (docstring + README note); no code claims it runs today.

#### `ll_clouds`
- **FR-C1 I/O** Read/write PLY, PCD, XYZ; sample point clouds from meshes (trimesh) and from cadling/STEP geometry.
- **FR-C2 Preprocessing** Normalize (center+unit-scale), voxel downsample, farthest-point downsample (reuse FPS pattern compatible with `ll_ocadr` geometry_net), statistical outlier removal.
- **FR-C3 Features** Per-point normals (k-NN/PCA), curvature estimate, bounding box / centroid / extent stats.
- **FR-C4 Registration/Segmentation primitives** ICP (point-to-point) registration; RANSAC plane segmentation; Euclidean/region clustering.
- **FR-C5 Data models** Pydantic v2 models for `PointCloud` (points, optional normals/colors/labels, metadata) consistent with monorepo conventions (`arbitrary_types_allowed=True` for ndarray).
- **FR-C6 Bridges (light)** Helper to convert a cadling `CADlingDocument`/mesh and an `ll_ocadr` input into an `ll_clouds.PointCloud` (and back), without `ll_clouds` taking a hard dependency on those packages (lazy import).

### 3.2 Non-Functional Requirements
- **NFR-1 Conventions** `from __future__ import annotations`; `_log = logging.getLogger(__name__)`; Google-style docstrings; lazy imports for heavy deps (torch, trimesh, pythonocc, open3d-if-used) guarded by `_X_AVAILABLE` flags with graceful degradation.
- **NFR-2 OpenMP safety** Any test importing torch follows the repo rule: torch imported first in `conftest.py`, `OMP_NUM_THREADS=1`, `pytest.importorskip("torch")` at module level. PyTorch assumed conda-forge.
- **NFR-3 Determinism** Training/sampling accept a seed; RNG seeded via dedicated generators (not global `np.random`), per existing `reward_signal`/SDG patterns.
- **NFR-4 No fabricated data** Failure paths raise meaningful errors or skip with explicit logging; no hardcoded geometry returned as if computed (matches the `cadling` remediation standard).
- **NFR-5 Security** Generated CAD code continues to execute only in the existing subprocess sandbox (`disposal/code_executor.py`); no new `exec` of model output in-process.
- **NFR-6 Test markers** Reuse the marker taxonomy already declared in `ll_clouds/pyproject.toml` (`slow`, `requires_gpu`, `requires_pythonocc`, `requires_cadquery`, `requires_torch`, `requires_cadling`, `requires_geotoken`, `integration`, `unit`).

---

## 4. Architecture

### 4.1 `ll_gen` neural track (after fix)

```
GenerationOrchestrator.generate(prompt)            pipeline/orchestrator.py:113
  └─ _propose(...)                                  :372
       ├─ _propose_neural_vae      → NeuralVAEGenerator      generators/neural_vae.py
       ├─ _propose_neural_diffusion→ NeuralDiffusionGenerator generators/neural_diffusion.py
       └─ _propose_neural_vqvae    → NeuralVQVAEGenerator    generators/neural_vqvae.py
                                          │ imports (FIXED): from stepnet import STEPVAE,
                                          │   StructuredDiffusion, VQVAEModel,
                                          │   CADGenerationPipeline, DisentangledCodebooks
                                          ▼
                              stepnet models (REAL, untrained → G2 trained)
  └─ disposal (CadQuery subprocess sandbox)          disposal/code_executor.py
  └─ verification + feedback                          pipeline/verification.py, feedback/

Training:
RLAlignmentTrainer.train_epoch / train_step          training/rl_trainer.py:368 / :132
  └─ generator.generate_for_training(prompt)          :170  (log_probs on live graph)
  └─ reward_signal(...)                                feedback/reward_signal.py
  └─ advantage = reward − EMA baseline → loss.backward() → optimizer.step()   :182–219
Data: load_deepcad / load_abc                          datasets/deepcad_loader.py, abc_loader.py
```

### 4.2 `ll_ocadr` HF-native inference

```
mesh/STEP/STL file ─► file_content_chunker / mesh_process / step_process   vllm/process/*
        │ point clouds + chunks
        ▼
GeometryNet (PointNet++)  +  ShapeNet (ViT)          vllm/lattice_encoder/{geometry_net,shape_net}.py
        │ local + global features → MLP projector
        ▼
LatticelabsOCADRForCausalLM.get_input_embeddings ─► forward / generate   latticelabs_ocadr.py:217/257/311
        │ inputs_embeds
        ▼
AutoModelForCausalLM (HF) .generate(...)             latticelabs_ocadr.py:354
(vLLM ModelRegistry/SupportsMultiModal = DOCUMENTED FUTURE, not wired)
```

### 4.3 `ll_clouds` package layout (new)

```
ll_clouds/ll_clouds/
  __init__.py            # public exports
  datamodel.py           # Pydantic PointCloud + result models
  io.py                  # FR-C1: read/write PLY/PCD/XYZ, mesh sampling
  preprocess.py          # FR-C2: normalize, voxel/FPS downsample, outlier removal
  features.py            # FR-C3: normals, curvature, stats
  registration.py        # FR-C4: ICP
  segmentation.py        # FR-C4: RANSAC plane, clustering
  bridges.py             # FR-C6: cadling/ll_ocadr lazy converters
ll_clouds/tests/
  conftest.py            # OpenMP-safe torch import, fixtures
  unit/test_*.py         # one per module
  integration/test_*.py  # round-trips, bridges
```

---

## 5. Implementation Plan & Milestones

Owner for all milestones: **Maintainer**. Each milestone ends with its quality gate green.

### M1 — `ll_gen` wiring fix (G1) — *highest leverage, smallest effort*
- T1.1 Fix imports in `generators/neural_vae.py:430-431`, `neural_diffusion.py:418`, `neural_vqvae.py:222,424-425` → import from `stepnet`. **Owner:** Maintainer.
- T1.2 Add regression test `ll_gen/tests/test_neural_imports.py`: import + instantiate each generator (untrained) and run one `generate`/`generate_for_training` on CPU; assert no `ImportError` and proposal shape.
- T1.3 Add orchestrator smoke test: `GenerationOrchestrator.generate` for each neural mode reaches dispose stage.
- **Exit:** neural track imports and runs untrained; tests green.

### M2 — `ll_gen` training entry + RL loop proven (G1 cont.)
- T2.1 Implement/verify `generate_for_training` returns live-graph `log_probs` for all three generators.
- T2.2 Implement runnable training CLI (`python -m ll_gen.training.run` or Click) wiring `load_deepcad` → generator → `RLAlignmentTrainer.train_epoch`, with `--max-samples`, `--epochs`, `--save`.
- T2.3 Test: one `train_step` on CPU with a tiny synthetic batch performs a real gradient update (param delta ≠ 0).
- **Exit:** `train_step`/`train_epoch` run end-to-end; metrics returned; checkpoint saved/loaded.

### M3 — `ll_gen` small checkpoints (G2)
- T3.1 Acquire DeepCAD/ABC subset (see §6); document the exact command and the subset size.
- T3.2 Train VAE, diffusion, VQ-VAE to "proof of life" (short run); save checkpoints to `checkpoints/` and document download/reproduce steps.
- T3.3 Eval: measure valid-CAD rate (dispose success) of trained vs random-init; assert trained > random by a documented margin.
- **Exit:** three checkpoints produce non-noise output above baseline; eval recorded in spec appendix / a results file.

### M4 — `ll_ocadr` HF-native + tests (G3)
- T4.1 Add `ll_ocadr/tests/` with `conftest.py` (OpenMP-safe) and unit tests: `GeometryNet.forward`, `ShapeNet.forward`, chunkers (`file_content_chunker`, `mesh_process`, `step_process`), processor classes.
- T4.2 End-to-end smoke test: build `LatticelabsOCADRForCausalLM` with a tiny HF model, run `forward` + `generate` on a synthetic mesh; assert output tokens produced.
- T4.3 Implement HF-native inference script (`ll_ocadr/run_ll_ocadr_hf.py`): file + prompt → text, no vLLM.
- T4.4 Docs: mark vLLM as future; remove "vLLM integration works" claims.
- **Exit:** pytest suite passes; HF inference script runs; no false vLLM claims.

### M5 — `ll_clouds` core library (G4)
- T5.1 `datamodel.py` + `io.py` (FR-C1, FR-C5) + tests (round-trip PLY/PCD/XYZ, mesh sampling).
- T5.2 `preprocess.py` (FR-C2) + tests (normalize idempotence, downsample counts, outlier removal correctness on synthetic data).
- T5.3 `features.py` (FR-C3) + tests (normals on a plane → constant normal; curvature on sphere ≈ 1/r).
- T5.4 `registration.py` + `segmentation.py` (FR-C4) + tests (ICP recovers a known transform within tolerance; RANSAC finds a planted plane; clustering separates two blobs).
- T5.5 `bridges.py` (FR-C6) + integration tests (cadling mesh → PointCloud; ll_ocadr input → PointCloud), lazy-imported.
- **Exit:** `pip install -e ./ll_clouds` works; full suite passes; coverage ≥ 80% on `ll_clouds/ll_clouds`.

### M6 — Cross-cutting quality gate (G5)
- T6.1 `ruff check`, `black --check`, `mypy` clean for all three packages.
- T6.2 Stub scan: zero `NotImplementedError`, zero `pass`-only bodies, zero TODO across the three packages.
- T6.3 Update root `README.md` package table + `STATUS.md` to reflect new state.
- **Exit:** all gates green; status docs accurate.

### Dependency order
M1 → M2 → M3 (sequential). M4, M5 independent of the M1–M3 chain (can interleave). M6 last.

---

## 6. Data & Training Plan (G2)

- **Datasets:** loaders already exist and are real — `datasets/deepcad_loader.py` (`DeepCADDataset`, `load_deepcad`), `datasets/abc_loader.py` (`ABCDataset`, `load_abc`), `datasets/text2cad_loader.py`, `datasets/sketchgraphs_loader.py` (each with `__len__`/`__getitem__`). Use a **subset** (e.g., a few thousand DeepCAD command sequences) via the loader's limiting args.
- **Compute:** single GPU (or CPU for the smoke-scale run) — proof-of-life scale, not convergence. Document the actual machine used.
- **Procedure:** (a) optional supervised warm-start of VAE/VQ-VAE reconstruction on the subset using `stepnet` trainers (`stepnet/trainer.py`, `stepnet/training/`); (b) RL alignment via `RLAlignmentTrainer.train_epoch` with `reward_signal` from real dispose results.
- **Acceptance metric:** dispose-success (valid CAD) rate on a held-out prompt set, trained checkpoint vs random-init, with the margin recorded. Reference reality check (`STATUS.md`): unconditional parametric validity in the literature ranges DeepCAD ~24% → HNC-CAD ~81%; "proof of life" target is simply *above random-init*, not SOTA.
- **Checkpoints:** saved under `checkpoints/`; if too large to commit, document a reproducible train command + (optional) HF Hub upload.

---

## 7. Testing Strategy

- **Unit:** every new/fixed module has unit tests asserting real numeric behavior (no mock-only tests). Geometry math validated against closed-form cases (plane normals, sphere curvature, known ICP transforms).
- **Integration:** `ll_gen` propose→dispose→reward loop on CPU with a tiny model; `ll_ocadr` forward+generate with a tiny HF model; `ll_clouds` cadling/ll_ocadr bridges.
- **Regression:** `ll_gen/tests/test_neural_imports.py` permanently guards against the import regression that caused this spec.
- **Markers/CI:** heavy tests gated behind `requires_torch`/`requires_cadquery`/`slow`; the default suite runs on CPU without optional CAD deps.
- **Coverage targets:** `ll_clouds` ≥ 80%; `ll_ocadr` new suite covers all encoder/processor modules + e2e; `ll_gen` neural generators + trainer covered.

---

## 8. Risks & Mitigations

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | `stepnet` model `__init__` signatures differ from what generators pass (e.g. `STEPVAE(**asdict(vae_config))`) → `TypeError` after import fix | High | M1.T1.2 instantiates each model with defaults first; reconcile config dataclasses against real `__init__` signatures; add a config-mapping shim if needed. |
| R2 | Proof-of-life training does not beat random-init within the time budget | Medium | Warm-start reconstruction before RL; shrink subset/architecture; if still flat, record the negative result honestly and reduce G2 to "training runs + loss decreases" (owner decision point, not silent). |
| R3 | `ll_ocadr` tiny-HF-model e2e test is slow/flaky in CI | Medium | Use the smallest available causal LM; mark `slow`/`requires_torch`; keep encoder unit tests as the fast default. |
| R4 | `ll_clouds` heavy dep choice (open3d vs pure numpy/scipy/trimesh) | Medium | Default to numpy/scipy/trimesh (already in monorepo) for portability; make open3d an optional accelerator behind a flag, never required. |
| R5 | OpenMP crash on macOS when torch + conda mix in new tests | High | Enforce NFR-2 in every new `conftest.py`; `pytest.importorskip("torch")`. |
| R6 | DeepCAD/ABC subset download/availability | Medium | Loaders support local paths; document acquisition; provide a tiny synthetic fallback dataset for CI so tests never depend on external download. |
| R7 | Scope creep into full training (NG1) | Medium | Acceptance metric is explicitly "above random-init", not SOTA; M3 exit criteria bound the effort. |

---

## 9. Acceptance Criteria (Definition of Done)

A milestone-complete spec is satisfied when **all** hold:

1. `from stepnet import STEPVAE, StructuredDiffusion, VQVAEModel, CADGenerationPipeline, DisentangledCodebooks` succeeds, and no `ll_stepnet.stepnet.*` string remains in `ll_gen/`.
2. `GenerationOrchestrator.generate` completes for VAE, diffusion, VQ-VAE modes through to dispose, untrained and (when `checkpoint_path` given) trained.
3. `RLAlignmentTrainer.train_step` returns real metrics and changes model parameters (verified by a test).
4. Three `ll_gen` checkpoints exist and beat random-init on dispose-success by a recorded margin; reproduce command documented.
5. `ll_ocadr` pytest suite passes (encoders, chunkers, processor, e2e forward+generate); HF-native inference script runs; no code/docs claim working vLLM.
6. `ll_clouds` installs editable, exposes the FR-C1…C6 API, suite passes with ≥80% coverage.
7. `ruff`/`black --check`/`mypy` clean across the three packages; zero `NotImplementedError`/`pass`-only/TODO; no fabricated outputs.
8. `README.md` and `STATUS.md` updated to reflect the achieved state (no over-claiming).

---

## 10. Open Questions (each with an owner — no TBD)

| # | Question | Owner | Default if unanswered |
|---|---|---|---|
| OQ1 | Which exact DeepCAD/ABC subset + size for G2 training? | Maintainer | ⬜ OPEN (M3) — to be decided at M3.T3.1 when data+compute are available. |
| OQ2 | Point-cloud backend for `ll_clouds`: numpy/scipy/trimesh only, or optional open3d accelerator? | Maintainer | ✅ RESOLVED (M5) — numpy/scipy/trimesh required; open3d not used (optional accelerator only). |
| OQ3 | Commit checkpoints to git or host on HF Hub? | Maintainer | ⬜ OPEN (M3) — decide when the first checkpoints exist. |
| OQ4 | Smallest HF LLM to use for `ll_ocadr` tests/inference default? | Maintainer | ✅ RESOLVED (M4) — a tiny **offline** GPT-2 (n_embd 64) built in the test fixtures; no network/download. |
| OQ5 | Should `ll_clouds` bridges live in `ll_clouds` or in `cadling`/`ll_ocadr`? | Maintainer | ✅ RESOLVED (M5) — in `ll_clouds`, lazily imported; verified `import ll_clouds` pulls in none of cadling/ll_ocadr/torch/trimesh. |

---

## 11. Verification Log (evidence this spec is grounded, not assumed)

- Broken imports confirmed by reading `ll_gen/ll_gen/generators/neural_vae.py:430-431`, `neural_diffusion.py:418`, `neural_vqvae.py:222,424-425`.
- Non-existence of `ll_stepnet/stepnet/models.py` and `pipeline.py` confirmed via `ls` (no file, no dir, no shim).
- Real class locations confirmed: `stepnet/vae.py:35`, `diffusion.py:348`, `vqvae.py:269,749`, `generation_pipeline.py:58`; re-exports at `stepnet/__init__.py:69,82,108,109`.
- Orchestrator neural routing confirmed at `pipeline/orchestrator.py:104-106,372,447+`.
- RL loop confirmed real at `training/rl_trainer.py:132,170,182-219,368,448`.
- `ll_ocadr` real `forward`/`generate` confirmed at `latticelabs_ocadr.py:257,311,354`; absence of vLLM registration confirmed (no `ModelRegistry`/`SupportsMultiModal`).
- Dataset loaders confirmed real: `abc_loader.py` (366 L), `deepcad_loader.py` (215 L), `text2cad_loader.py` (255 L), `sketchgraphs_loader.py` (433 L), each with `__len__`/`__getitem__`/`load_*`.
- `ll_clouds` emptiness + intended scope confirmed via `ll_clouds/pyproject.toml`.
- Zero trained checkpoints confirmed via tree-wide weight-file search.
