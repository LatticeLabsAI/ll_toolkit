"""Generation path routing — automatic Path A/B selection.

The router analyzes the input prompt (and optionally image) to
decide which generation path to use:

- **Code generation** (Path A): CadQuery or OpenSCAD, best for
  prismatic mechanical geometry described with standard CAD vocabulary.
- **Neural generation** (Path B): VAE, diffusion, or VQ-VAE, best
  for freeform/organic shapes or latent space exploration.
"""
from ll_gen.routing.router import GenerationRouter, RoutingDecision

__all__ = ["GenerationRouter", "RoutingDecision"]
