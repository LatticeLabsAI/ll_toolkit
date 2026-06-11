---
title: 'Tutorial: Generate CAD with ll_gen'
description: Run the propose→dispose generation loop, inspect a DisposalResult, and train a proof-of-life neural generator.
sidebar:
  label: Generate CAD
  order: 4
---

In this tutorial you will run [ll_gen](/ll_toolkit/ll_gen/overview/)'s
propose→dispose loop and then train a proof-of-life neural generator. Allow
~20 minutes (training is short and CPU-friendly).

:::note[Two generator generations]
This tutorial walks the **orchestrator + RL loop** with a from-scratch generator so you
can see the propose→dispose loop and a before/after validity measurement end to end. For
generators that already **produce valid CAD**, jump to
[Generate valid CAD (trained, MLX)](#5-generate-valid-cad-trained-mlx) below — the
autoregressive command generator (0.914 valid) and latent diffusion (0.934 valid). Read
[The reality of AI CAD generation](/ll_toolkit/concepts/the-reality-of-ai-cad-generation/)
for why generating the program and executing it is the reliable route.
:::

## 1. Generate from a prompt

```python
from ll_gen import GenerationOrchestrator

orch = GenerationOrchestrator()
result = orch.generate("a 20 mm cube with a 5 mm hole through the center")

print("valid solid:", result.is_valid)
print("geometry:", result.geometry_report)
```

The orchestrator routes the prompt (code vs. neural), proposes a candidate,
disposes it in the CadQuery sandbox, and — if invalid — retries with structured
error feedback.

## 2. Force the code path

The code path proposes CadQuery/OpenSCAD code, which the kernel executes — the
most reliable route for simple mechanical parts.

```python
from ll_gen import GenerationRoute

result = orch.generate(
    "an M6 hex bolt, 30 mm long",
    force_route=GenerationRoute.CODE_CADQUERY,
    max_retries=3,
    export=True,   # write STEP/STL on success
)
print(result.is_valid, result.step_path)
```

## 3. Train a proof-of-life neural generator

Measure prior-sampling validity of the **same** VAE before and after the
REINFORCE dispose-reward loop. You need a small JSONL of eval prompts
(`eval/heldout.jsonl`, one `{"prompt": "..."}` per line).

```bash
python -m ll_gen.training.proof_of_life \
    --generator vae \
    --prompts eval/heldout.jsonl \
    --epochs 5 --steps-per-epoch 80 --n-eval-samples 100 \
    --seed 0 --save checkpoints/vae_rl.pt \
    --results results/proof_of_life_vae.json
```

Read the results:

```python
import json

r = json.load(open("results/proof_of_life_vae.json"))
print("baseline validity:", r["baseline"]["validity_rate"],
      "distinct:", r["baseline"]["num_distinct_valid"])
print("trained  validity:", r["trained"]["validity_rate"],
      "distinct:", r["trained"]["num_distinct_valid"])
```

`num_distinct_valid` matters: a validity gain from one valid shape repeated
(mode collapse) is visible here rather than mistaken for success.

## 4. Full RL training run

To train on a dataset rather than the proof-of-life harness:

```bash
python -m ll_gen.training.run \
    --generator vae \
    --dataset deepcad --data-path <hf-id-or-path> \
    --max-samples 2000 --epochs 1 --lr 1e-5 \
    --device cpu --save checkpoints/vae_rl.pt
```

The command prints a metrics JSON with reward, advantage, baseline, and loss.

## 5. Generate valid CAD (trained, MLX)

The generators that actually produce valid CAD take the **construction-program** route —
generate the command program and execute it, so the kernel builds a watertight solid.
They train and run natively in MLX on Apple Silicon:

```bash
# Autoregressive command generator: trains on real DeepCAD programs, then samples + executes.
# Reports validity through the real kernel, gated on a non-degenerate solid.
python ll_gen/mlx/ar_generator_mlx.py --mode train
# -> validity 0.914 (234/256), distinct 104, non-degenerate

# Latent diffusion over a program autoencoder: sample z -> decode -> execute.
python ll_gen/mlx/latent_diffusion_mlx.py --mode train
# -> sampled-z validity 0.934 (239/256), distinct 138  (vs a z=0 mean baseline: 14 distinct)
```

Both report `num_distinct` alongside validity, so a high rate from one repeated shape
(mode collapse) is visible. The latent-diffusion run prints **sampled-z** validity (noise
→ denoise → decode → execute) against a `z=0` predict-the-mean baseline — the comparison
that proves the diffusion adds diversity rather than repeating the mean shape.

## Where to next

- [ll_gen Usage](/ll_toolkit/ll_gen/usage/) for the full API.
- [Inside CAD generation models](/ll_toolkit/concepts/inside-cad-generation-models/)
  for what the VAE/diffusion/VQ-VAE generators are doing.
