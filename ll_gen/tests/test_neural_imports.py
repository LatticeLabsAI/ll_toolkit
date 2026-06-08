"""Regression test for neural generator initialization (SPEC-1 M1, T1.1).

Guards against the wiring defect where ``ll_gen``'s neural generators import
``stepnet`` model classes from non-existent modules
(``ll_stepnet.stepnet.models`` / ``ll_stepnet.stepnet.pipeline``) and call
their constructors with the wrong arguments.

Each generator must be able to build a real ``torch.nn.Module`` on CPU with
default (untrained) configuration. Output quality is NOT asserted here — only
that the propose track is wired and constructs without ``ImportError`` /
``TypeError``.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ll_gen.generators.neural_diffusion import NeuralDiffusionGenerator
from ll_gen.generators.neural_vae import NeuralVAEGenerator
from ll_gen.generators.neural_vqvae import NeuralVQVAEGenerator


@pytest.mark.requires_torch
class TestNeuralGeneratorInit:
    """Each neural generator must construct a real nn.Module on CPU."""

    def test_vae_init_builds_module(self) -> None:
        gen = NeuralVAEGenerator(device="cpu")
        gen._init_model()
        assert isinstance(gen._model, torch.nn.Module), (
            "NeuralVAEGenerator._model must be a torch.nn.Module after init"
        )
        # A real module exposes parameters.
        assert any(True for _ in gen._model.parameters()), "VAE model has no parameters"

    def test_diffusion_init_builds_module(self) -> None:
        gen = NeuralDiffusionGenerator(device="cpu")
        gen._init_model()
        assert isinstance(gen._model, torch.nn.Module), (
            "NeuralDiffusionGenerator._model must be a torch.nn.Module after init"
        )
        assert any(True for _ in gen._model.parameters()), "Diffusion model has no parameters"

    def test_vqvae_init_builds_module(self) -> None:
        gen = NeuralVQVAEGenerator(device="cpu")
        gen._init_model()
        assert isinstance(gen._model, torch.nn.Module), (
            "NeuralVQVAEGenerator._model must be a torch.nn.Module after init"
        )
        assert any(True for _ in gen._model.parameters()), "VQ-VAE model has no parameters"
