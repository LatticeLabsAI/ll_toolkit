"""Contract tests for generate_for_training log-probs (SPEC-1 M2, T2.1/T2.2).

The REINFORCE loop in ``RLAlignmentTrainer.train_step`` uses
``proposal.log_probs`` as the policy term. For a correct (unbiased) policy
gradient those log-probs must be a differentiable tensor sampled on the same
trajectory and connected to the model parameters.

- VAE and VQ-VAE sample autoregressively from live logits, so their log-probs
  flow back into the model parameters.
- Diffusion is a documented exception: ``StructuredDiffusion`` exposes no
  noise-conditioned / log-prob sampling API, so the signal is the prior
  log-prob of an independent N(0, I) draw — differentiable but decoupled from
  the model parameters (see ``neural_diffusion.generate_for_training``). M2's
  RL loop is therefore proven on VAE + VQ-VAE; diffusion RL is future work.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ll_gen.generators.neural_diffusion import NeuralDiffusionGenerator
from ll_gen.generators.neural_vae import NeuralVAEGenerator
from ll_gen.generators.neural_vqvae import NeuralVQVAEGenerator


def _connected_to_params(log_probs, model) -> bool:
    """True if a gradient flows from log_probs into at least one model param."""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        log_probs.sum(), params, retain_graph=False, allow_unused=True
    )
    return any(g is not None and g.abs().sum().item() > 0 for g in grads)


def _tiny_diffusion_generator() -> NeuralDiffusionGenerator:
    from stepnet.config import DiffusionConfig

    cfg = DiffusionConfig(
        num_timesteps=10,
        inference_steps=2,
        denoiser_layers=1,
        denoiser_hidden_dim=64,
        denoiser_heads=4,
        latent_dim=64,
    )
    return NeuralDiffusionGenerator(diffusion_config=cfg, device="cpu")


@pytest.mark.requires_torch
class TestLiveGraphLogProbs:
    def test_vae_log_probs_connected_to_params(self) -> None:
        gen = NeuralVAEGenerator(device="cpu")
        proposal = gen.generate_for_training("a 20mm cube")
        assert proposal.log_probs is not None
        assert proposal.log_probs.requires_grad
        assert proposal.entropy is not None
        assert _connected_to_params(proposal.log_probs, gen._model)

    def test_vqvae_log_probs_connected_to_params(self) -> None:
        gen = NeuralVQVAEGenerator(device="cpu")
        proposal = gen.generate_for_training("a 20mm cube")
        assert proposal.log_probs is not None
        assert proposal.log_probs.requires_grad
        assert proposal.entropy is not None
        assert _connected_to_params(proposal.log_probs, gen._model)

    def test_diffusion_log_probs_present_but_decoupled(self) -> None:
        """Documented contract: diffusion's log-prob is differentiable but not
        connected to model parameters (prior log-prob of an independent noise
        sample). Locks in the known limitation so a future change is noticed."""
        gen = _tiny_diffusion_generator()
        proposal = gen.generate_for_training("a 20mm cube")
        assert proposal.log_probs is not None
        assert proposal.log_probs.requires_grad
        assert not _connected_to_params(proposal.log_probs, gen._model)
