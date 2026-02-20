"""Neural generator wrappers — bridge ll_stepnet models to the proposal protocol."""
from __future__ import annotations

from ll_gen.generators.base import BaseNeuralGenerator
from ll_gen.generators.latent_sampler import LatentSampler
from ll_gen.generators.neural_diffusion import NeuralDiffusionGenerator
from ll_gen.generators.neural_vae import NeuralVAEGenerator
from ll_gen.generators.neural_vqvae import NeuralVQVAEGenerator

__all__ = [
    "BaseNeuralGenerator",
    "NeuralVAEGenerator",
    "NeuralDiffusionGenerator",
    "NeuralVQVAEGenerator",
    "LatentSampler",
]
