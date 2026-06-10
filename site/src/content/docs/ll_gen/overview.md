---
title: ll_gen — Overview
description: Generation orchestration for CAD — neural propose, deterministic dispose in a CadQuery sandbox, with a REINFORCE training loop. Models ship untrained.
sidebar:
  label: Overview
  order: 1
  badge:
    text: Untrained
    variant: caution
---

**ll_gen** orchestrates generative CAD modeling around a simple idea: **propose,
then dispose**. A generator proposes a candidate (neural latent sample, an LLM
code proposal, or a deterministic template), and a deterministic *dispose* stage
executes that proposal in a sandboxed [CadQuery](https://cadquery.readthedocs.io/)
subprocess to produce — and verify — real geometry. A REINFORCE training loop
closes the alignment loop using the verification result as reward.

## How it works

```text
GenerationOrchestrator.generate(prompt)
  → propose
      ├─ neural VAE / diffusion / VQ-VAE   (stepnet models)
      ├─ LLM code proposal
      └─ deterministic template
  → dispose  (CadQuery subprocess sandbox)   → real solid + validity
  → verification + feedback (reward)

Training:
  RLAlignmentTrainer.train_step
    → generate_for_training (log-probs on the live graph)
    → reward (dispose success / validity)
    → advantage = reward − baseline → loss.backward() → optimizer.step()
```

The neural generators import their models from the **`stepnet`** package
(`STEPVAE`, `StructuredDiffusion`, `VQVAEModel`, `CADGenerationPipeline`). Code
proposals are always executed in the subprocess sandbox — never `exec`'d in
process.

## Running a training loop

ll_gen exposes a runnable training entry point:

```bash
python -m ll_gen.training.run --help
```

## Status

:::caution[Maturity: pipeline real, models untrained]
The orchestration, dispose sandbox, verification, and REINFORCE loop are real and
run end-to-end. The neural generators **ship untrained** — a proof-of-life VAE
training run (`python -m ll_gen.training.proof_of_life`) is documented, but
production-quality generation requires real training. Reward is gated on
producing a **closed solid**, so the metric reflects genuine CAD validity rather
than reward-hacked non-solids.
:::

Use the sidebar for **Installation**, **Usage**, and the **API Reference**.
