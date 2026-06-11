---
title: ll_gen — Usage
description: Generate CAD with the propose→dispose orchestrator, and train the neural generators with the REINFORCE loop or the proof-of-life run.
sidebar:
  label: Usage
  order: 3
---

ll_gen's entry point is the `GenerationOrchestrator`: it routes a prompt to a
proposal path, disposes the proposal into real geometry, and (on failure) feeds
structured error context back for a retry.

## Generate from a prompt

```python
from ll_gen import GenerationOrchestrator

orch = GenerationOrchestrator()           # default LLGenConfig
result = orch.generate(
    "a 20 mm cube with a 5 mm hole through the center",
    export=True,                           # write STEP/STL for valid results
    render=False,
)

print(result.is_valid)                     # did it dispose to a valid solid?
print(result.geometry_report)              # volume, bbox, face/edge/solid counts
print(result.step_path)                    # exported STEP path (if valid + export=True)
```

`generate()` runs the full pipeline:

```text
1. Route    decide Code (Path A) vs Neural (Path B) from the prompt
2. Propose  produce a typed proposal via the selected path
3. Dispose  execute + validate + repair in the CadQuery sandbox
4. Feedback on failure, build structured feedback and retry with error context
5. Export   write STEP/STL for valid results
```

You can force a route or cap retries:

```python
from ll_gen import GenerationRoute
# Routes: CODE_CADQUERY, CODE_OPENSCAD, CODE_PYTHONOCC,
#         NEURAL_VAE, NEURAL_DIFFUSION, NEURAL_VQVAE

result = orch.generate(
    "a hex bolt M6", force_route=GenerationRoute.CODE_CADQUERY, max_retries=3
)
```

## The proposal types

| Path | Proposer | Proposal type |
|---|---|---|
| Code | `CadQueryProposer`, `OpenSCADProposer` | `CodeProposal` |
| Neural | `NeuralVAEGenerator`, `NeuralVQVAEGenerator`, `NeuralDiffusionGenerator` | `LatentProposal` / `CommandSequenceProposal` |

All paths converge on the `DisposalEngine`, which executes the proposal,
validates the result (`DisposalResult`, `GeometryReport`), and can apply
`RepairAction`s.

## Trained generators that produce valid CAD (native MLX)

Two trained generators take the **construction-program** route — generate the command
program and *execute* it — so the kernel builds a watertight solid. They run natively in
MLX on Apple Silicon and report validity **measured through the real kernel**, gated on a
non-degenerate solid (closed solid with positive volume):

```bash
# Autoregressive command generator — trained on real DeepCAD programs.
# Result: validity 0.914 (234/256), 104 distinct, non-degenerate.
python ll_gen/mlx/ar_generator_mlx.py --mode train

# Latent diffusion over a program autoencoder.
# Result: sampled-z validity 0.934 (239/256), 138 distinct.
python ll_gen/mlx/latent_diffusion_mlx.py --mode train
```

For the latent diffusion the headline metric is **sampled-z** validity
(noise → denoise → decode → execute), reported against a `z=0` predict-the-mean
baseline so a diverse generator is distinguishable from one that repeats the mean shape.
A faithful MLX port of the command-VAE (`python ll_gen/mlx/vae_mlx.py --mode parity`)
reproduces the PyTorch model exactly, but that parallel-decoder VAE is itself a weak
generator (~0–12% valid) — the program-based generators above are the valid-CAD path.

## Train the neural generators (REINFORCE)

The RL alignment loop rewards proposals that dispose into valid solids:

```bash
python -m ll_gen.training.run \
    --generator vae \
    --dataset deepcad --data-path <hf-id-or-path> \
    --max-samples 2000 --epochs 1 --lr 1e-5 \
    --device cpu --save checkpoints/vae_rl.pt
```

`--generator` is one of `vae`, `vqvae`, `diffusion`. Provide training records via
`--dataset {deepcad,abc}` (+ `--data-path`) or a `--prompts-file` JSONL. The
command prints a metrics JSON including reward, advantage, baseline, and loss.

## Proof-of-life run (before/after validity)

The proof-of-life run measures prior-sampling validity of the **same** model
before and after the REINFORCE loop, so a real gain is distinguishable from mode
collapse:

```bash
python -m ll_gen.training.proof_of_life \
    --generator vae \
    --prompts eval/heldout.jsonl \
    --epochs 5 --steps-per-epoch 80 --n-eval-samples 100 \
    --seed 0 --save checkpoints/vae_rl.pt \
    --results results/proof_of_life_vae.json
```

It reports `validity_rate` **and** `num_distinct_valid` at both points plus the
per-epoch curve.

:::note[Two generator generations — know which you're running]
The **program-based** generators (`ar_generator_mlx.py`, `latent_diffusion_mlx.py`) are
**trained** and produce measured-valid CAD (0.914 / 0.934 valid). The **legacy**
neural generators reachable from the orchestrator (`vae`, `vqvae`, `diffusion` via the
REINFORCE loop) are randomly initialized out of the box — their prior samples are mostly
invalid until trained, and the raw-geometry diffusion is limited by representation
(independently-sampled faces don't mate). See
[The reality of AI CAD generation](/ll_toolkit/concepts/the-reality-of-ai-cad-generation/)
for why generating the program and executing it is the reliable route.
:::

## Related

- [Overview](/ll_toolkit/ll_gen/overview/) · [Installation](/ll_toolkit/ll_gen/installation/)
- Tutorial: [Generate CAD with ll_gen](/ll_toolkit/tutorials/generate-cad/).
