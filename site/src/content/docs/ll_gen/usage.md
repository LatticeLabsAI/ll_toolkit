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

:::caution[Models ship untrained]
Out of the box the neural generators are randomly initialized — their prior
samples are mostly invalid. The dispose stage and the RL loop are real and run
end-to-end; meaningful generation requires training. See
[The reality of AI CAD generation](/ll_toolkit/concepts/the-reality-of-ai-cad-generation/)
for why the propose→dispose design is the reliable part.
:::

## Related

- [Overview](/ll_toolkit/ll_gen/overview/) · [Installation](/ll_toolkit/ll_gen/installation/)
- Tutorial: [Generate CAD with ll_gen](/ll_toolkit/tutorials/generate-cad/).
