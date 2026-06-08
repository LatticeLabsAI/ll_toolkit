# Plan M3 — `ll_gen` Small Real Checkpoints (Proof of Life)

| | |
|---|---|
| **Spec** | SPEC-1 §5 M3, §6 Data & Training, §3.1 FR-G5, §9.4 |
| **Goal** | Train VAE, diffusion, VQ-VAE on a small public dataset subset to produce committed-or-hosted checkpoints whose dispose-success (valid-CAD) rate **beats random-init by a recorded margin**. Proof of life, not SOTA. |
| **Depends on** | M2 (runnable, checkpointing training loop) |
| **Owner** | Maintainer · **Mode** Inline/sequential · **Tests** TDD for harness; training is a measured run · **Commits** per task |
| **Status** | Not started · **Decision baked in:** real data only (no synthetic CI fallback) |

## Pre-flight
- Confirm dataset acquisition path for DeepCAD and/or ABC subset (loaders: `datasets/deepcad_loader.py`, `datasets/abc_loader.py`). Resolve **OQ1** (subset + size) and **OQ3** (commit vs HF Hub) before T3.2.
- Confirm compute: a GPU is strongly preferred; record the exact machine used.
- Read `feedback/reward_signal.py` (defines what "valid CAD" reward means) and `pipeline/verification.py`.

## Tasks

### T3.1 — Acquire + register the dataset subset
- Download the chosen DeepCAD/ABC subset to a documented local path; record the exact commands and the sample count in this plan's "Run log" section.
- Add `tests/test_dataset_loads.py` (mark `slow`/`integration`): assert the loader yields the expected count and that `__getitem__(0)` returns the expected keys/shapes.
- **Commit:** `chore(ll_gen): document dataset subset acquisition + loader smoke test`

### T3.2 — Establish the random-init baseline (TDD the eval harness first)
- **Red:** write `ll_gen/ll_gen/training/evaluate_validity.py` + `tests/test_validity_eval.py`. The eval function takes a generator + a held-out prompt set, runs propose→dispose, returns `valid_rate = #valid / #total`. Test it on a deterministic fixture (e.g., a generator that always emits a known-valid script → 1.0; known-invalid → 0.0).
- **Green:** implement `evaluate_validity`.
- Run it on **random-init** VAE/diffusion/VQ-VAE; record baseline `valid_rate` per model in the Run log.
- **Commit:** `feat(ll_gen): add validity evaluation harness + record random-init baseline`

### T3.3 — (Optional) supervised warm-start
- If RL-from-scratch is too sparse, warm-start VAE/VQ-VAE reconstruction on the subset using the `stepnet` trainers (`stepnet/trainer.py`, `stepnet/training/`). Save warm-start checkpoint.
- **Commit:** `feat(ll_gen): supervised warm-start of VAE/VQ-VAE on subset`

### T3.4 — Proof-of-life training runs
- Run `python -m ll_gen.training.run` for each model on the subset, short budget (document epochs/steps/wall-clock). Save checkpoints to `checkpoints/`.
- Capture training curves (loss, mean reward, mean advantage) to a results file under `results/`.
- **Commit:** `feat(ll_gen): proof-of-life checkpoints for VAE/diffusion/VQ-VAE`

### T3.5 — Eval trained vs baseline + acceptance gate
- Run `evaluate_validity` on each trained checkpoint over the same held-out prompts.
- **Acceptance:** trained `valid_rate` > random-init `valid_rate` by a margin recorded in the Run log (reference: literature unconditional validity spans DeepCAD ~24% → HNC-CAD ~81%; target is simply *above baseline*).
- If a model does **not** beat baseline within budget (risk R2): document the negative result honestly, and escalate the choice — extend budget, shrink model, or reduce that model's G2 target to "loss decreases + runs." Do **not** silently claim success.
- **Commit:** `docs(ll_gen): record trained-vs-baseline validity results`

### T3.6 — Checkpoint distribution
- Per OQ3: if checkpoints >50 MB, upload to HF Hub and document the download command; else commit under `checkpoints/`. Add a reproduce command to this plan.
- Wire `GenerationOrchestrator` / generators to load these checkpoints via `checkpoint_path` (verify FR-G5).
- **Commit:** `chore(ll_gen): publish checkpoints + reproduce instructions`

## Verification
```bash
cd ll_gen
python -m ll_gen.training.evaluate_validity --checkpoint checkpoints/vqvae.pt --prompts eval/heldout.jsonl
# trained valid_rate must exceed the recorded random-init baseline
```

## Run log (fill during execution — no TBD at completion)
| Model | Dataset subset (size) | Machine | Budget | Baseline valid_rate | Trained valid_rate | Margin | Checkpoint location |
|---|---|---|---|---|---|---|---|
| VAE | | | | | | | |
| Diffusion | | | | | | | |
| VQ-VAE | | | | | | | |

## Milestone risks
- **R2 (no improvement in budget)** — mitigations: warm-start (T3.3), shrink subset/model, more steps; final fallback is an explicit maintainer decision recorded in the Run log.
- **R6 (data availability)** — loaders accept local paths; if a source is unavailable, switch DeepCAD↔ABC and document.
- **Compute** — if no GPU, runs may be infeasible at useful scale; flag to maintainer rather than reporting a vacuous CPU result.

## Done checklist
- [ ] Dataset subset acquired + documented.
- [ ] Random-init baseline recorded per model.
- [ ] Trained checkpoints exist and load via `checkpoint_path`.
- [ ] Trained valid_rate > baseline (or honest negative result + escalation) recorded in Run log.
- [ ] Reproduce command documented.
