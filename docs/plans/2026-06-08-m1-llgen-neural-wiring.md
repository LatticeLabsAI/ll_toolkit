# Plan M1 — `ll_gen` Neural Wiring Fix

| | |
|---|---|
| **Spec** | SPEC-1 §5 M1, §3.1 FR-G1/G2, §8 R1 |
| **Goal** | The neural-propose track (VAE / diffusion / VQ-VAE) imports and runs end-to-end on CPU, untrained, without `ImportError` or `TypeError`. |
| **Depends on** | — (start here; highest leverage, smallest effort) |
| **Owner** | Maintainer · **Mode** Inline/sequential · **Tests** TDD · **Commits** per task |
| **Status** | Not started |

## Why this milestone matters
Fixing ~a dozen lines resurrects the entire neural-propose + REINFORCE track — the project's most market-validated bet (per `STATUS.md`). It is currently 100% dead code.

## Root-cause summary (verified)
Two defects, not one:
1. **Wrong import paths.** Generators import from `ll_stepnet.stepnet.models` / `ll_stepnet.stepnet.pipeline`, which don't exist. Correct: top-level package is `stepnet`; classes are re-exported in `stepnet/__init__.py`.
   - `STEPVAE` → `stepnet.vae` (re-export `stepnet/__init__.py:108`)
   - `StructuredDiffusion` → `stepnet.diffusion` (`__init__.py:56`)
   - `VQVAEModel`, `DisentangledCodebooks` → `stepnet.vqvae` (`__init__.py:109`)
   - `CADGenerationPipeline` → `stepnet.generation_pipeline` (`__init__.py:69`)
2. **Constructor mismatch (R1).** Even after fixing imports, current call sites pass the wrong args:
   - `neural_vae.py:~444` calls `STEPVAE()` but `STEPVAE.__init__` **requires** `encoder_config` (vae.py:20, no default).
   - `neural_vqvae.py` calls `VQVAEModel()` but `VQVAEModel.__init__` **requires** `input_dim` (vqvae.py:781, no default).
   - `neural_diffusion.py:~430` calls `StructuredDiffusion(**self.diffusion_config.__dict__)` but `StructuredDiffusion.__init__(self, config)` takes a **single config object** (diffusion.py:371), not `**kwargs`.

## Pre-flight (read before touching code)
- `ll_stepnet/stepnet/__init__.py` (confirm the 5 re-exports above).
- `ll_stepnet/stepnet/config.py` — find the encoder config class `STEPVAE` expects (`encoder_config` with `token_embed_dim`, `vocab_size`; see vae.py:35-36) and the `DiffusionConfig` (diffusion.py:375).
- `ll_gen/ll_gen/config.py` — `GeneratorConfig` (:258), `vae_config`/`diffusion_config` fields, to map ll_gen config → stepnet constructor args.
- `ll_gen/ll_gen/generators/neural_vae.py:425-465`, `neural_diffusion.py:415-440`, `neural_vqvae.py:218-440` — the `_init_model` bodies.

## Tasks

### T1.1 — Red: failing regression test for all three generators
- Create `ll_gen/tests/conftest.py` (OpenMP-safe: import torch first, `os.environ["OMP_NUM_THREADS"]="1"`).
- Create `ll_gen/tests/test_neural_imports.py`:
  - For each of `NeuralVAEGenerator`, `NeuralDiffusionGenerator`, `NeuralVQVAEGenerator`: instantiate with default config on CPU, call `_init_model()` (or trigger lazy init), assert no exception and `generator._model` is an `nn.Module`.
  - Mark `requires_torch`; `pytest.importorskip("torch")` at module top.
- **Run it. Confirm it FAILS** with `ImportError`/`TypeError` (the bug). Record the failure output in the commit body.
- **Commit (suggest):** `test(ll_gen): add failing regression test for neural generator init`

### T1.2 — Green VAE: fix import + `encoder_config` reconciliation
- In `neural_vae.py`: change `from ll_stepnet.stepnet.models import STEPVAE` → `from stepnet import STEPVAE`; `from ll_stepnet.stepnet.pipeline import CADGenerationPipeline` → `from stepnet import CADGenerationPipeline`.
- Fix construction: build the `encoder_config` STEPVAE requires from `self.vae_config` (or a sensible default `stepnet` encoder config) and pass it; map remaining `asdict(vae_config)` fields to STEPVAE's named kwargs (`latent_dim`, `kl_weight`, `num_command_types`, `num_param_levels`, `max_seq_len`). No bare `STEPVAE()`.
- Run `test_neural_imports.py::*vae*` → green.
- **Commit:** `fix(ll_gen): wire NeuralVAEGenerator to stepnet.STEPVAE with valid encoder_config`

### T1.3 — Green diffusion: fix import + single-config constructor
- In `neural_diffusion.py`: `from ll_stepnet.stepnet.models import StructuredDiffusion` → `from stepnet import StructuredDiffusion`.
- Replace `StructuredDiffusion(**self.diffusion_config.__dict__)` with `StructuredDiffusion(config=self.diffusion_config)` (pass the object; `__init__` reads attrs via `getattr`). If `diffusion_config` is `None`, `StructuredDiffusion()` is already valid (config defaults internally).
- Run diffusion test → green.
- **Commit:** `fix(ll_gen): wire NeuralDiffusionGenerator to stepnet.StructuredDiffusion config object`

### T1.4 — Green VQ-VAE: fix imports + `input_dim`
- In `neural_vqvae.py`: fix `from ll_stepnet.stepnet.models import VQVAEModel` → `from stepnet import VQVAEModel`; `from ll_stepnet.stepnet.pipeline import CADGenerationPipeline` → `from stepnet import CADGenerationPipeline`; `from ll_stepnet.stepnet.vqvae import DisentangledCodebooks` → `from stepnet import DisentangledCodebooks` (line 222).
- Fix construction: pass the required `input_dim` (derive from the vocab/feature dim used elsewhere in ll_gen, or expose it on `GeneratorConfig`); pass codebook sizes if configured. No bare `VQVAEModel()`.
- Run vqvae test → green.
- **Commit:** `fix(ll_gen): wire NeuralVQVAEGenerator to stepnet.VQVAEModel with input_dim`

### T1.5 — Orchestrator smoke test (propose → dispose reachable)
- Add `ll_gen/tests/test_orchestrator_neural.py`: for each neural mode, call `GenerationOrchestrator.generate(prompt, mode=...)` (orchestrator.py:113, routing :372/447+) on CPU with a trivial prompt; assert it reaches the dispose stage and returns a result object (untrained output may be invalid CAD — that's expected; assert *no wiring error*, not validity).
- TDD: write test (may already pass after T1.2-T1.4); if it surfaces a routing/config gap, fix it.
- **Commit:** `test(ll_gen): smoke-test orchestrator neural propose→dispose for all 3 modes`

### T1.6 — Grep guard: no `ll_stepnet.` references remain
- `grep -rn "ll_stepnet\." ll_gen/` → must be empty. Fix any stragglers.
- **Commit (if changes):** `fix(ll_gen): remove residual ll_stepnet.* import references`

## Verification
```bash
cd ll_gen && pytest tests/test_neural_imports.py tests/test_orchestrator_neural.py -v
grep -rn "ll_stepnet\." ll_gen/        # expect: no output
```

## Milestone risks
- **R1 (signatures)** — handled directly by T1.2–T1.4. If a config field has no clean mapping, expose it on `GeneratorConfig` rather than hardcoding.
- **`stepnet` not importable in the active env** — confirm `pip install -e ./ll_stepnet` (package name `ll-stepnet`, import `stepnet`). If torch missing, tests `importorskip` and this milestone can't be verified — flag to maintainer, don't fake a pass.

## Done checklist
- [ ] `test_neural_imports.py` green for VAE, diffusion, VQ-VAE.
- [ ] Orchestrator smoke test green for all 3 modes.
- [ ] `grep -rn "ll_stepnet\." ll_gen/` empty.
- [ ] No bare `STEPVAE()` / `VQVAEModel()` / `**__dict__` diffusion construction remains.
