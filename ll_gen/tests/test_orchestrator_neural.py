"""Orchestrator neural-route smoke test (SPEC-1 M1, T1.5).

Verifies that the GenerationOrchestrator can route to each neural propose
path (VAE / diffusion / VQ-VAE), generate a proposal, and reach the disposal
stage — returning a ``DisposalResult`` without any wiring error
(``ImportError`` / ``TypeError`` / ``AttributeError``).

Geometry validity is NOT asserted: the models are untrained and the
deterministic disposal engine needs pythonocc/cadquery (conda-only, absent in
the CPU test environment), so ``is_valid`` is expected to be ``False``. The
contract under test is that the neural propose track is wired end-to-end and
the pipeline degrades gracefully to an invalid result instead of crashing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ll_gen.config import GenerationRoute
from ll_gen.pipeline.orchestrator import GenerationOrchestrator
from ll_gen.proposals.disposal_result import DisposalResult


def _fast_diffusion_generator():
    """Build a tiny-step diffusion generator so the smoke test stays fast.

    The default StructuredDiffusion runs 200 denoising steps across 4 stages,
    which is far too slow for a unit test. A tiny config exercises the same
    wiring in a fraction of a second.
    """
    from stepnet.config import DiffusionConfig

    from ll_gen.generators.neural_diffusion import NeuralDiffusionGenerator

    tiny = DiffusionConfig(
        num_timesteps=10,
        inference_steps=2,
        denoiser_layers=1,
        denoiser_hidden_dim=64,
        denoiser_heads=4,
        latent_dim=64,
    )
    return NeuralDiffusionGenerator(diffusion_config=tiny, device="cpu")


@pytest.mark.requires_torch
class TestOrchestratorNeuralRoutes:
    """Each neural route must propose and reach dispose without wiring errors."""

    def test_neural_vae_reaches_dispose(self) -> None:
        orch = GenerationOrchestrator()
        result = orch.generate(
            "a 20mm cube",
            force_route=GenerationRoute.NEURAL_VAE,
            max_retries=1,
            export=False,
        )
        assert isinstance(result, DisposalResult)

    def test_neural_vqvae_reaches_dispose(self) -> None:
        orch = GenerationOrchestrator()
        result = orch.generate(
            "a 20mm cube",
            force_route=GenerationRoute.NEURAL_VQVAE,
            max_retries=1,
            export=False,
        )
        assert isinstance(result, DisposalResult)

    def test_neural_diffusion_reaches_dispose(self) -> None:
        orch = GenerationOrchestrator()
        # Pre-seed the cached diffusion generator with a tiny-step config so the
        # orchestrator does not build the slow 200-step default.
        orch._diffusion_generator = _fast_diffusion_generator()
        result = orch.generate(
            "a 20mm cube",
            force_route=GenerationRoute.NEURAL_DIFFUSION,
            max_retries=1,
            export=False,
        )
        assert isinstance(result, DisposalResult)
