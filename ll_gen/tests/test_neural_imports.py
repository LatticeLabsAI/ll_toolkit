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


@pytest.mark.requires_torch
class TestNeuralGeneratorConfigWiring:
    """Non-default configs must be forwarded into the underlying stepnet models.

    These validate the new constructor contracts beyond "it builds": the
    config values must actually reach STEPVAE / StructuredDiffusion / VQVAEModel.
    Attribute names below are the real ones stored by those models (verified
    against stepnet), not the generator-side field names.
    """

    def test_vae_non_default_config_wiring(self) -> None:
        """A non-default stepnet VAEConfig flows into STEPVAE + its encoder config.

        STEPVAE keeps ``embed_dim`` (= STEPEncoderConfig.token_embed_dim, which
        the generator maps from ``encoder_embed_dim``) plus the scalar VAE
        hyper-params, so assert on those.
        """
        from stepnet.config import VAEConfig

        vc = VAEConfig(
            encoder_embed_dim=128,  # -> STEPEncoderConfig.token_embed_dim -> embed_dim
            encoder_layers=2,
            latent_dim=64,
            kl_weight=0.5,
            max_seq_len=40,
        )
        gen = NeuralVAEGenerator(vae_config=vc, device="cpu")
        gen._init_model()

        assert gen._model.embed_dim == 128
        assert gen._model.latent_dim == 64
        assert gen._model.kl_weight == 0.5
        assert gen._model.max_seq_len == 40

    def test_diffusion_non_default_config_wiring(self) -> None:
        """A non-default DiffusionConfig reaches StructuredDiffusion and its denoisers.

        Exercises the new ``config=`` path: the config object is stored verbatim
        and its values propagate into the per-stage CADDenoiser modules.
        (512 / 8 = 64 head_dim keeps attention construction valid.)
        """
        from stepnet.config import DiffusionConfig

        dc = DiffusionConfig(
            denoiser_heads=8,
            denoiser_hidden_dim=512,
            latent_dim=128,
            denoiser_layers=1,
            num_timesteps=10,
            inference_steps=2,
        )
        gen = NeuralDiffusionGenerator(diffusion_config=dc, device="cpu")
        gen._init_model()

        # Config object stored verbatim (the new config= constructor path).
        assert gen._model.config.denoiser_heads == 8
        # Values propagated into the built denoiser modules.
        denoiser = next(iter(gen._model.denoisers.values()))
        assert denoiser.hidden_dim == 512
        assert denoiser.latent_dim == 128

    def test_vqvae_non_default_config_wiring(self) -> None:
        """max_seq_len and codebook_dim drive VQVAEModel.input_dim / code_dim.

        ``input_dim`` is derived as max_seq_len * (1 command type + 16 param
        slots); ``code_dim`` is the generator's codebook_dim (64 / 8 = 8 keeps
        the codebook-decoder attention valid).
        """
        gen = NeuralVQVAEGenerator(max_seq_len=40, codebook_dim=64, device="cpu")
        gen._init_model()

        assert gen._model.input_dim == 40 * (1 + 16)
        assert gen._model.code_dim == 64
