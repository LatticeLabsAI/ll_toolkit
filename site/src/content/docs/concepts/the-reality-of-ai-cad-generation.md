---
title: The reality of AI CAD generation
description: What actually ships in production, why the propose-then-dispose pattern is the only reliable approach, and how that shapes ll_gen.
sidebar:
  label: The honest picture
  order: 5
---

It is worth being blunt about the state of the art, because it explains why the
toolkit is built the way it is.

## What actually ships

As of early 2026, **no major CAD vendor ships neural-network-based 3D model
generation.** What is marketed as "AI CAD" falls into three categories that are
often conflated:

1. **Topology optimization rebranded as "generative design"** — shipping for
   years, FEA-based, *not* neural networks.
2. **Workflow assistants / copilots** — documentation chatbots (Onshape AI
   Advisor explicitly "does not generate designs").
3. **Actual LLM-based CAD generation** — one substantive production product
   (Zoo.dev), a handful of early startups, zero verified enterprise
   manufacturing deployments.

Even the leading product is candid about limits: it works best for "traditional,
simple mechanical parts — fasteners, bearings, connectors," is stochastic (same
prompt → different result), and independent testing shows quality "drops off
sharply with medium and high-complexity designs."

## The universal architecture: code → kernel → validate

Every shipping and near-shipping system follows the same three phases:

```text
stochastic phase   LLM generates code in a DSL (CadQuery / OpenSCAD / KCL)
deterministic phase a CAD kernel executes that code → exact B-Rep
validation phase   check watertight / manifold / sane; errors feed back
```

**No neural network writes directly into a CAD kernel's data structures.** The
indirection is the point: it leverages decades of proven kernel engineering and
gives the network a tractable code-writing task instead of an intractable
geometry-synthesis task. The cost is that the system inherits LLM code
generation's limits — stochastic output, prompt sensitivity, failure on complex
specs, and no engineering guarantees.

## Why ll_gen is "propose, then dispose"

[ll_gen](/ll_toolkit/ll_gen/overview/) implements exactly this pattern:

- **Propose** — a generator (neural latent sample, an LLM code proposal, or a
  deterministic template) produces a candidate.
- **Dispose** — the candidate is executed in a sandboxed **CadQuery** subprocess,
  which produces real geometry and reports whether it is a valid closed solid.
- **Align** — a REINFORCE loop rewards proposals that dispose into valid solids,
  pushing the generator toward geometry the kernel accepts.

This is the same code-through-kernel discipline the production systems use, and
the same RL-from-kernel-feedback technique that took sketch-constraint
satisfaction from ~9% to ~93% in the literature.

## Generative *design* ≠ generative *AI CAD*

These are architecturally unrelated and routinely confused:

| | Generative design (topology opt.) | Generative AI CAD |
|---|---|---|
| Input | fully specified loads/constraints | natural language |
| Method | gradient-based physics optimization | neural network trained on data |
| Determinism | same input → same output | same input → different outputs |
| Maturity | shipping ~a decade | one production system since late 2023 |
| Output | optimized geometry, engineering meaning | a starting point needing validation |

## What to expect from this toolkit

The toolkit's neural packages ship **untrained** — they are architectures and
training loops, not finished models. Treat generated output as a starting point
to be validated, never as a manufacturing-ready part. The honest near-term value
of neural CAD is prototyping, concept exploration, and education — with a human
doing the majority of the design effort. The toolkit is built to make the
*reliable* part (kernel execution + validation) load-bearing, and the
*unreliable* part (the neural proposal) improvable through training and RL.

## See also

- [How neural networks generate CAD](/ll_toolkit/concepts/how-neural-cad-generation-works/)
- [ll_gen overview](/ll_toolkit/ll_gen/overview/) and
  [ll_ocadr overview](/ll_toolkit/ll_ocadr/overview/)
