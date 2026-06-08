# Plan M2 — `ll_gen` Training Entry + RL Loop Proven

| | |
|---|---|
| **Spec** | SPEC-1 §5 M2, §3.1 FR-G3/G4/G6 |
| **Goal** | `generate_for_training` returns live-graph log-probs; `RLAlignmentTrainer.train_step` performs a real gradient update; a runnable training CLI wires dataset → generator → `train_epoch` with checkpoint save/load. |
| **Depends on** | M1 (generators must import + construct) |
| **Owner** | Maintainer · **Mode** Inline/sequential · **Tests** TDD · **Commits** per task |
| **Status** | Not started |

## Context (verified)
- `RLAlignmentTrainer` real and correct: `train_step` (rl_trainer.py:132), `generator.generate_for_training(prompt)` (:170), advantage/baseline (:182-191), `total_loss.backward()` → `optimizer.step()` (:210-219), `train_epoch` (:368), `evaluate` (:448).
- VQ-VAE already samples on a live graph accumulating log-probs/entropy (neural_vqvae.py:~395-420) — the reference pattern for the other generators.
- Dataset loaders are real: `load_deepcad` (deepcad_loader.py:151), `load_abc` (abc_loader.py:314), each backed by a `Dataset` with `__len__`/`__getitem__`.
- Reward from real dispose results: `feedback/reward_signal.py`.

## Pre-flight
- Read `ll_gen/ll_gen/training/rl_trainer.py` fully (`_init_training` :91, `train_step` :132, `train_epoch` :368, `_extract_token_ids` :274, `_get_log_probs` :340).
- Read each generator's `generate_for_training` (or confirm absence and which need it added — VAE/diffusion vs the VQ-VAE reference).
- Read `feedback/reward_signal.py` and `disposal/code_executor.py` interface used by the trainer.

## Tasks

### T2.1 — Red: log-probs contract test per generator
- `ll_gen/tests/test_generate_for_training.py`: for each neural generator, call `generate_for_training(prompt)` on CPU; assert the returned proposal carries `log_probs` that (a) are a tensor, (b) `requires_grad`, (c) are connected to model params (`torch.autograd.grad(log_probs.sum(), model.parameters())` yields at least one non-None grad).
- Run → expect failures for any generator lacking live-graph log-probs.
- **Commit:** `test(ll_gen): assert generate_for_training returns live-graph log-probs`

### T2.2 — Green: implement/repair `generate_for_training` for VAE and diffusion
- Mirror the VQ-VAE AR-sampling pattern: sample tokens from `Categorical(logits=...)`, accumulate `dist.log_prob` and `dist.entropy` on the live graph, attach to the proposal.
- Do **not** detach the sampling graph. No `torch.no_grad()` around the sampled path used for RL.
- Run T2.1 → green.
- **Commit:** `feat(ll_gen): live-graph log-prob accumulation in VAE/diffusion generate_for_training`

### T2.3 — Red+Green: real gradient update in `train_step`
- `ll_gen/tests/test_rl_trainer.py`: snapshot a model param, run one `train_step` with a tiny synthetic prompt and a stub-but-real dispose path (real subprocess CadQuery on a trivial valid script, or a deterministic reward fn injected via the trainer's reward hook), assert (a) returned dict has `reward`/`advantage`/`baseline`/`loss`, (b) at least one param changed (`not torch.allclose(before, after)`).
- Fix any gap so the test passes (advantage seeding at step 0 per rl_trainer.py:185-191 should already avoid zero-gradient first step — verify).
- **Commit:** `test(ll_gen): prove train_step performs a real gradient update`

### T2.4 — Training CLI entry point
- Add `ll_gen/ll_gen/training/run.py` (runnable as `python -m ll_gen.training.run`): args `--dataset {deepcad,abc}`, `--data-path`, `--generator {vae,diffusion,vqvae}`, `--max-samples`, `--epochs`, `--lr`, `--save PATH`, `--resume PATH`, `--device`, `--seed`.
- Wire: `load_deepcad/load_abc(max_samples=...)` → generator (M1) → `RLAlignmentTrainer.train_epoch` → checkpoint save.
- Seed RNG via dedicated generators (NFR-3), not global `np.random`.
- **Commit:** `feat(ll_gen): add python -m ll_gen.training.run training entry point`

### T2.5 — Checkpoint save/load round-trip
- `tests/test_checkpoint_roundtrip.py`: train 1 micro-step, save, construct a fresh generator, `load_checkpoint`, assert state_dict matches and a forward produces identical output for a fixed seed.
- Verify `load_checkpoint` actually restores (cross-check against the known `STEPNetTrainer.load_checkpoint` always-returns-0 bug class — ensure ll_gen's path restores epoch/optimizer too if applicable).
- **Commit:** `test(ll_gen): checkpoint save/load round-trip for neural generators`

### T2.6 — Smoke `train_epoch` on a tiny real subset
- `tests/test_train_epoch_smoke.py` (mark `slow`, `requires_cadquery`): run `train_epoch` for 1 epoch on `max_samples<=4` from a local DeepCAD/ABC sample (or the repo's `part.step`-derived minimal set); assert it completes and returns epoch metrics; assert loss is finite.
- **Commit:** `test(ll_gen): smoke train_epoch on a 4-sample subset`

## Verification
```bash
cd ll_gen && pytest tests/test_generate_for_training.py tests/test_rl_trainer.py tests/test_checkpoint_roundtrip.py -v
python -m ll_gen.training.run --dataset deepcad --data-path <path> --generator vqvae --max-samples 4 --epochs 1 --save /tmp/ck.pt
```

## Milestone risks
- **Reward needs real dispose (CadQuery).** Gate `requires_cadquery`; for the pure-unit gradient test (T2.3) inject a deterministic reward so it runs without CadQuery.
- **Diffusion is not autoregressive-token** — its "log-prob" may need a different RL formulation (score/ELBO). If live-graph log-prob isn't natural for diffusion, scope diffusion RL out of M2 (document it) and prove the loop on VAE + VQ-VAE; raise to maintainer rather than forcing an unsound metric.

## Done checklist
- [ ] `generate_for_training` returns live-graph log-probs (VAE, VQ-VAE at minimum; diffusion or documented exception).
- [ ] `train_step` proven to change params.
- [ ] `python -m ll_gen.training.run` runs a 1-epoch micro-train and saves a checkpoint.
- [ ] Checkpoint save/load round-trip test green.
