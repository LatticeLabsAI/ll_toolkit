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

:::caution[Set expectations first]
ll_gen's neural generators ship **untrained** — random weights produce mostly
invalid geometry. The *dispose* stage (CadQuery execution + validation) and the
RL loop are real. This tutorial shows the loop working end to end and a
before/after validity measurement, not a production model. Read
[The reality of AI CAD generation](/ll_toolkit/concepts/the-reality-of-ai-cad-generation/)
for the why.
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

## Where to next

- [ll_gen Usage](/ll_toolkit/ll_gen/usage/) for the full API.
- [Inside CAD generation models](/ll_toolkit/concepts/inside-cad-generation-models/)
  for what the VAE/diffusion/VQ-VAE generators are doing.
