"""Regression tests for the diffusion DDPO policy-gradient path.

Guards the H1 fix: the diffusion RL signal was previously DECOUPLED from the
model — ``generate_for_training`` returned the log-prob of an independent
``N(0, I)`` noise draw, so ``optimizer.step()`` updated **zero** parameters
while logging a finite loss. The fix adds a real DDPO sampler
(``StructuredDiffusion.sample_with_log_prob``) whose per-step Gaussian
log-probabilities flow through the denoiser network, so the REINFORCE update
actually trains the model.

These tests assert the gradient genuinely reaches the parameters and that one
RL step changes them.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stepnet.config import DiffusionConfig
from stepnet.diffusion import GeometryCodec, StructuredDiffusion

from ll_gen.generators.neural_diffusion import NeuralDiffusionGenerator
from ll_gen.training.rl_trainer import RLAlignmentTrainer


def _small_config() -> DiffusionConfig:
    """A tiny diffusion config so the trajectory is cheap to differentiate."""
    return DiffusionConfig(
        num_timesteps=50,
        inference_steps=4,
        denoiser_layers=2,
        denoiser_heads=4,
        denoiser_hidden_dim=32,
        latent_dim=16,
        num_faces=6,
        num_edges=8,
        uv_grid_size=8,
        edge_num_points=12,
        codec_hidden_dim=64,
    )


def _policy_params(model):
    """The RL policy = the denoisers + stage conditioning projections.

    The geometry codec is deliberately excluded: it maps latents <-> geometry
    and is trained by the reconstruction objective in ``forward_train``, NOT by
    the REINFORCE policy gradient (the geometry decode happens after sampling,
    outside the trajectory log-prob).
    """
    params = list(model.denoisers.parameters())
    params += list(model.cond_projections.parameters())
    return [p for p in params if p.requires_grad]


def _num_params_with_gradient(scalar, model) -> tuple[int, int]:
    """Return (num policy params receiving a non-zero gradient, total policy params)."""
    params = _policy_params(model)
    grads = torch.autograd.grad(
        scalar.sum(), params, retain_graph=False, allow_unused=True
    )
    connected = sum(
        1 for g in grads if g is not None and g.abs().sum().item() > 0
    )
    return connected, len(params)


@pytest.mark.requires_torch
class TestDDPOSampler:
    def test_sample_with_log_prob_is_differentiable_to_params(self) -> None:
        torch.manual_seed(0)
        model = StructuredDiffusion(_small_config())
        model.eval()

        results, log_prob, entropy = model.sample_with_log_prob(
            batch_size=2, device="cpu", num_inference_steps=4, eta=1.0
        )

        assert log_prob.shape == (2,)
        assert log_prob.requires_grad
        assert bool(torch.isfinite(log_prob).all())
        assert entropy.shape == (2,)
        # results carries the per-stage token latents plus decoded geometry.
        assert set(StructuredDiffusion.STAGE_NAMES).issubset(set(results))
        assert "face_grids" in results and "edge_points" in results

        connected, total = _num_params_with_gradient(log_prob, model)
        # The whole point of the fix: the trajectory log-prob must reach the
        # model parameters (it reached zero of them before).
        assert connected == total and total > 0, (
            f"only {connected}/{total} params received gradient"
        )

    def test_eta_zero_is_coerced_to_stochastic(self) -> None:
        """A deterministic (eta=0) trajectory has a degenerate policy; the
        sampler must coerce it to a usable stochastic one."""
        torch.manual_seed(0)
        model = StructuredDiffusion(_small_config())
        model.eval()

        _, log_prob, _ = model.sample_with_log_prob(
            batch_size=1, device="cpu", num_inference_steps=4, eta=0.0
        )
        assert log_prob.requires_grad
        assert bool(torch.isfinite(log_prob).all())
        # A non-degenerate stochastic trajectory has a non-zero log-prob.
        assert float(log_prob.abs().sum()) > 0.0


@pytest.mark.requires_torch
class TestGeneratorTrainingSignal:
    def test_generate_for_training_log_probs_reach_params(self) -> None:
        gen = NeuralDiffusionGenerator(
            diffusion_config=_small_config(), device="cpu", inference_steps=4, eta=0.0
        )
        proposal = gen.generate_for_training("a 20mm cube")

        assert proposal.log_probs is not None
        assert proposal.log_probs.requires_grad
        assert proposal.entropy is not None

        connected, total = _num_params_with_gradient(
            proposal.log_probs, gen._model
        )
        assert connected == total and total > 0, (
            f"only {connected}/{total} model params received gradient"
        )


@pytest.mark.requires_torch
class TestGeometryDecoder:
    """The diffusion model must decode its latents into usable B-Rep geometry
    (face UV grids + edge polylines), not surface a raw flat latent."""

    def test_codec_roundtrip_shapes(self) -> None:
        codec = GeometryCodec(
            latent_dim=16, uv_grid_size=8, edge_num_points=12, hidden_dim=64
        )
        face = torch.randn(2, 6, 8, 8, 3)
        edge = torch.randn(2, 8, 12, 3)
        face_rec = codec.decode_faces(codec.encode_faces(face))
        edge_rec = codec.decode_edges(codec.encode_edges(edge))
        assert face_rec.shape == face.shape
        assert edge_rec.shape == edge.shape

    def test_codec_is_trainable(self) -> None:
        """Overfitting a single sample must drive the reconstruction loss down —
        proving the decoder learns (it is not an identity/stub head)."""
        torch.manual_seed(0)
        codec = GeometryCodec(
            latent_dim=16, uv_grid_size=8, edge_num_points=12, hidden_dim=64
        )
        codec.train()
        opt = torch.optim.Adam(codec.parameters(), lr=1e-3)
        face = torch.randn(1, 6, 8, 8, 3)
        edge = torch.randn(1, 8, 12, 3)
        first = codec.reconstruction_loss(face, edge)["total_recon_loss"].item()
        for _ in range(200):
            opt.zero_grad()
            loss = codec.reconstruction_loss(face, edge)["total_recon_loss"]
            loss.backward()
            opt.step()
        last = codec.reconstruction_loss(face, edge)["total_recon_loss"].item()
        assert last < first * 0.5

    def test_sample_returns_decoded_geometry(self) -> None:
        torch.manual_seed(0)
        model = StructuredDiffusion(_small_config())
        model.eval()
        out = model.sample(batch_size=1, device="cpu")
        assert out["face_grids"].shape == (1, 6, 8, 8, 3)
        assert out["edge_points"].shape == (1, 8, 12, 3)

    def test_generator_emits_valid_per_primitive_geometry(self) -> None:
        gen = NeuralDiffusionGenerator(
            diffusion_config=_small_config(), device="cpu", inference_steps=4, eta=0.0
        )
        proposal = gen.generate_for_training("a cube")
        # 6 face grids of [U, V, 3] and 8 edge polylines of [M, 3].
        assert len(proposal.face_grids) == 6
        assert proposal.face_grids[0].shape == (8, 8, 3)
        assert len(proposal.edge_points) == 8
        assert proposal.edge_points[0].shape == (12, 3)
        # validate_shapes() enforces ndim==3 / last-dim==3 on face grids.
        assert proposal.validate_shapes() == []

    def test_forward_train_geometry_trains_codec(self) -> None:
        """forward_train with geometry must return a finite reconstruction loss
        and produce gradients for the codec parameters."""
        torch.manual_seed(0)
        model = StructuredDiffusion(_small_config())
        model.train()
        geometry = {
            "face_grids": torch.randn(2, 6, 8, 8, 3),
            "edge_points": torch.randn(2, 8, 12, 3),
        }
        losses = model.forward_train(geometry=geometry)
        assert "face_recon_loss" in losses and "edge_recon_loss" in losses
        assert torch.isfinite(losses["total_loss"])
        losses["total_loss"].backward()
        codec_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.geometry_codec.parameters()
        )
        assert codec_grad, "codec received no gradient from forward_train"


@pytest.mark.requires_torch
class TestRLStepTrainsModel:
    def test_one_train_step_changes_parameters(self) -> None:
        gen = NeuralDiffusionGenerator(
            diffusion_config=_small_config(), device="cpu", inference_steps=4, eta=0.0
        )
        gen.generate_for_training("warmup")  # lazy-init the model
        # The RL step trains the policy (denoisers + conditioning), not the
        # codec (trained separately by reconstruction). Track the policy tensors.
        policy_keys = [
            k
            for k in gen._model.state_dict()
            if k.startswith("denoisers.") or k.startswith("cond_projections.")
        ]
        before = {k: gen._model.state_dict()[k].clone() for k in policy_keys}

        trainer = RLAlignmentTrainer(generator=gen)
        result = trainer.train_step("a 20mm cube")

        after = gen._model.state_dict()
        changed = sum(
            1 for k in policy_keys if not torch.equal(before[k].float(), after[k].float())
        )
        # The H1 regression: before the fix this was 0. Now every policy tensor
        # moves under one REINFORCE update.
        assert changed == len(policy_keys) and len(policy_keys) > 0, (
            f"only {changed}/{len(policy_keys)} policy tensors changed — RL not training"
        )
        assert "loss" in result and "reward" in result
