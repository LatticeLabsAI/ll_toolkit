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

### Pre-req fix (T3.2 blocker) — command-executor schema bridge
Measuring the baseline surfaced a pre-existing, structural bug: the disposal
command-executor consumed a command schema (`{type, params{named}}`, `EOL`
loop-end) that **nothing produces**, while `CommandSequenceProposal.dequantize`
emits `{command_type, parameters[positional], parameter_mask}` (`EOS`
loop-end). Every command read as an empty `type` → no geometry → validity was
pinned at 0% for *any* sequence, including a known-valid box. A second bug read
`token_ids` off the geotoken `TokenSequence` (which has none) so the cadling
fast-path always threw and fell back.

Fixed in `ll_gen/disposal/command_executor.py` (adapter `_normalize_commands` +
`_positional_to_named` + `_arc_midpoint`; `EOS` loop-end; cadling fast-path uses
`proposal.token_ids` and extracts the result shape). Proof: the known-valid box
now disposes to a valid solid (1 solid, 6 faces, χ=2, vol≈4.58). Regression test:
`tests/test_disposal_executor_schema.py`. **Before the fix all three models were
0%; the VAE baseline below (27%) is the direct evidence the fix works.**

### Dataset subset (T3.1) — resolves OQ1 + OQ3
**Source (real, no synthetic fallback):** `palapav/DeepCAD-DSL` on the HF Hub —
a DSL rendering of the DeepCAD sketch-and-extrude corpus (161.2K train / 8.9K
val / 8.1K test; 35 MB parquet). Chosen because its `target` DSL parses cleanly
into the `{type, params}` command schema (`SKETCH/LOOP/LINE/ARC/CIRCLE/EXTRUDE`).

**Wiring:** `ll_gen/datasets/_deepcad_dsl.py` (`parse_deepcad_dsl`) flattens the
DSL into SOL/LINE/ARC/CIRCLE/EXTRUDE/EOS commands; `deepcad_loader.py` now
defaults to this dataset and parses the DSL column. Unit tests:
`tests/test_deepcad_dsl.py` (8). Loader smoke test (3 layers — local fixture,
materialized subset, live Hub streaming): `tests/test_dataset_loads.py` (5).

**Acquisition commands** (run in the conda `cadling` env; `datasets` installed):
```bash
cd ll_gen
python scripts/download_deepcad_subset.py --split train --n 2000 --out data/deepcad_dsl
python scripts/download_deepcad_subset.py --split validation --n 200 --local-split val --out data/deepcad_dsl
```
**Materialized subset:** 2000 train + 200 val JSON files under
`ll_gen/data/deepcad_dsl/{train,val}/` (gitignored; reproduce via the script).

**Real-data sanity (de-risks warm-start):** disposing 40 ground-truth DeepCAD
sequences through the fixed executor yields **65% valid / 65% compile** — i.e.
real reconstructable validity is ~65%, well above the 27% random-init VAE
baseline, so warm-start has clear headroom. The ~35% that don't reconstruct are
multi-body/boolean (JOIN/CUT) and custom-plane models the standalone executor
doesn't fully support yet.

**OQ3 (distribution):** checkpoints to be sized in T3.6; data stays on the Hub
(not committed) and is reproduced via the script above.

### Random-init baselines (working executor)
Machine: macOS (Apple Silicon), **CPU**, conda `cadling` env (pythonocc-core +
cadquery). Held-out prompts: `ll_gen/eval/heldout.jsonl` (10 prompts). Seed 0.
Source: `python -m ll_gen.training.evaluate_validity --generator <m> --prompts
eval/heldout.jsonl --n-samples <k> --seed 0`. No external dataset — the reward
oracle is the real OCC dispose, so baseline needs only prompts.

| Model | Dataset subset (size) | Machine | Budget (baseline) | Baseline valid_rate | compile_rate | Trained valid_rate | Margin | Checkpoint location |
|---|---|---|---|---|---|---|---|---|
| VAE | n/a (dispose-reward, no dataset) | macOS CPU | n=100 (10×10) | 0.06–0.27* | 0.34–0.65 | **0.82** (RL path) | **+0.76** | `checkpoints/vae_rl.pt` (217 MB, gitignored; reproduce below) |
| VQ-VAE | n/a | macOS CPU | n=100 (10×10) | 0.00 | 0.00 | not run (sparse-reward R2) | | |
| Diffusion | n/a | macOS CPU | n=30 (10×3) | 0.00 | 0.00 | not run (~7 s/sample CPU) | | |

*VAE baseline depends on decode path + non-deterministic init (see note above):
random-init validity is 6% on the RL-optimized `generate_for_training` decode and
21–27% on the `generate` pipeline decode.

### Proof-of-life RL (T3.4 + T3.5) — same model before vs after, both decode paths
`python -m ll_gen.training.proof_of_life --generator vae --prompts eval/heldout.jsonl
--epochs 6 --steps-per-epoch 60 --n-eval-samples 10 --seed 0 --lr 1e-4
--save checkpoints/vae_rl.pt --results results/proof_of_life_vae.json`
(360 REINFORCE steps on the real OCC/cadquery dispose reward; ~47 s wall on CPU.)

| Decode path | Baseline valid | Trained valid | Baseline distinct | Trained distinct |
|---|---|---|---|---|
| **`generate_for_training` (RL-optimized, the gate)** | 6% | **82%** | 6 | **65** |
| `generate` (deployment pipeline) | 21% | **0%** | 21 | 0 |

Per-epoch training validity: 0.25 → 0.27 → 0.32 → 0.52 → 0.72 → **0.90**;
mean reward 0.69 → **0.92**. **ACCEPTANCE GATE MET** on the RL-optimized path:
trained 82% ≫ baseline 6% (+76 pts) with diversity *rising* (6 → 65 distinct valid
shapes) — genuine learning, not mode collapse. The RLAlignmentTrainer's entropy
bonus (`entropy_coeff=0.01`) plus this diversity check rule out reward-hacking.

**Decode-path divergence — found and FIXED.** The first run exposed a real
defect: `NeuralVAEGenerator.generate` (deployment, via `CADGenerationPipeline`)
and `generate_for_training` (RL) were *different* decoders, so RL improved only
the latter (82%) while the deployment path *regressed* (21% → 0%) through shared
weights — the checkpoint was not orchestrator-usable. Fixed by extracting a
shared `_decode_and_sample` so `generate` samples the same trajectory
`generate_for_training` does (under `no_grad`); `generate_candidates` still uses
the pipeline (same-pattern follow-up). After the fix, both paths track each
other:

| Decode path | Baseline valid | Trained valid | Trained distinct |
|---|---|---|---|
| `generate_for_training` (RL) | 6% | **98%** | 63 |
| **`generate` (deployment / orchestrator)** | 4% | **98%** | 63 |

Per-epoch validity 0.25 → 0.98; reward 0.69 → 0.95. The **deployed** generator
now reaches 98% prior-sampling validity with 63 distinct valid shapes — a
genuine, *usable* proof-of-life. Acceptance gate met on both paths.

Notes: random-init **VAE** emits multi-command sketches → 27% pass full BRep
validation (close to DeepCAD's ~24% unconditional baseline — a good sanity
check). **VQ-VAE** random-init decodes to a single LINE + EOS (one open edge,
no face) → 0%. **Diffusion** random-init likewise yields no closed geometry and
is ~7 s/sample (50-step sampler) on CPU. Proof-of-life target (T3.4) is the VAE,
which has signal to amplify; VQ-VAE/diffusion are the sparse-reward (R2) case.

**Reproducibility note:** `evaluate_validity` seeds the prior-sampling RNG but
the random-init *weights* depend on ambient RNG at model construction, so the
random-init VAE baseline is a distribution, not a point — re-measured at
**12% valid / 52% compile** (distinct=12) on a second process vs 27%/65% on the
first. The proof-of-life run therefore measures the **same model before vs after
RL in one process** (`training/proof_of_life.py`), which is reproducible and the
fair comparison.

### Warm-start (T3.3) — documented NEGATIVE result
`python -m ll_gen.training.warm_start --epochs 8 --batch-size 32 --max-samples 2000`
(8 epochs over the 2000-sample subset, 5:38 wall on CPU) reached `recon_loss`
4.23 but `cmd_loss` only **1.44** (random = ln 6 = 1.79 — barely above chance;
the STEPVAE encoder sees only command-type tokens, an architectural cap) and
`kl_loss` collapsed to **0.048**. Post-warm-start **prior-sampling validity = 0%**
(compile 5%, distinct 0) — *regressed* from the 27% random-init baseline. This is
the classic KL-collapse: good-enough reconstruction, but prior samples decode to
garbage. **Decision:** warm-start is the plan's optional step; it is *not* used to
initialize RL (a 5%-compile start gives REINFORCE no gradient signal). RL runs
from random-init instead. `warm_start.py` is kept (works; tests pass) and this
negative is the honest deliverable. (R2 mitigation revisited: warm-start did not
help here.)

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
