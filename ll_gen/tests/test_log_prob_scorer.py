"""Regression tests for the teacher-forcing sequence scorer (M-finish).

``RLAlignmentTrainer._get_log_probs`` was previously a ``NotImplementedError``
stub. It now delegates to ``generator.score_token_sequence`` — a real
teacher-forcing scorer that re-decodes fresh policy logits and gathers the
log-probability of a *given* token sequence.

Contract:
- VAE / VQ-VAE (command-sequence generators) return a differentiable scalar
  log-prob connected to the model parameters, plus a finite entropy.
- Diffusion (no command-token decoder) returns ``(None, 0.0)`` — honest
  "not applicable", not a stub.
- The scorer is an EVALUATION score: it is a fresh forward pass, NOT the RL
  gradient (that stays on ``proposal.log_probs`` from generate_for_training).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ll_gen.generators.neural_diffusion import NeuralDiffusionGenerator
from ll_gen.generators.neural_vae import NeuralVAEGenerator
from ll_gen.generators.neural_vqvae import NeuralVQVAEGenerator
from ll_gen.training.rl_trainer import RLAlignmentTrainer


def _connected_to_params(log_probs, model) -> bool:
    """True if a gradient flows from log_probs into at least one model param."""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        log_probs.sum(), params, retain_graph=False, allow_unused=True
    )
    return any(g is not None and g.abs().sum().item() > 0 for g in grads)


@pytest.mark.requires_torch
class TestSequenceScorer:
    def test_vae_scores_its_own_sequence(self) -> None:
        gen = NeuralVAEGenerator(device="cpu")
        proposal = gen.generate_for_training("a 20mm cube")
        assert proposal.token_ids, "expected a decoded token sequence"

        log_prob, entropy = gen.score_token_sequence(proposal.token_ids)

        assert log_prob is not None
        assert log_prob.requires_grad
        assert torch.isfinite(log_prob)
        assert isinstance(entropy, float)
        assert entropy >= 0.0
        # Real teacher-forcing: gradient must reach the model parameters.
        assert _connected_to_params(log_prob, gen._model)

    def test_vqvae_scores_its_own_sequence(self) -> None:
        gen = NeuralVQVAEGenerator(device="cpu")
        proposal = gen.generate_for_training("a bracket")
        assert proposal.token_ids, "expected a decoded token sequence"

        log_prob, entropy = gen.score_token_sequence(proposal.token_ids)

        assert log_prob is not None
        assert log_prob.requires_grad
        assert torch.isfinite(log_prob)
        # The VQ-VAE command-token distribution is produced by the pipeline's
        # projection heads (the codebook policy is sampled before this and is
        # scored on-policy by generate_for_training). So the teacher-forcing
        # gradient must reach those projection-head parameters.
        assert _connected_to_params(log_prob, gen._pipeline)

    def test_diffusion_scoring_not_applicable(self) -> None:
        # Diffusion has no command-token decoder: decode_command_logits is the
        # base default (None), so scoring returns (None, 0.0) — not a stub.
        gen = NeuralDiffusionGenerator(device="cpu")
        gen._init_model()

        assert gen.decode_command_logits() is None
        log_prob, entropy = gen.score_token_sequence([1, 6, 12, 13, 2])
        assert log_prob is None
        assert entropy == 0.0

    def test_trainer_get_log_probs_delegates(self, tmp_path) -> None:
        gen = NeuralVAEGenerator(device="cpu")
        gen._init_model()
        trainer = RLAlignmentTrainer(
            gen, learning_rate=1e-2, device="cpu", output_dir=str(tmp_path), seed=0
        )
        proposal = gen.generate_for_training("a 20mm cube")

        log_prob, entropy = trainer._get_log_probs(proposal.token_ids)

        assert log_prob is not None
        assert log_prob.requires_grad
        assert _connected_to_params(log_prob, gen._model)

    def test_scorer_handles_unscoreable_sequence(self) -> None:
        # A sequence with no recognisable command tokens scores to (None, 0.0)
        # rather than raising.
        gen = NeuralVAEGenerator(device="cpu")
        gen._init_model()
        log_prob, entropy = gen.score_token_sequence([1, 2])  # BOS, EOS only
        # EOS immediately after BOS still scores the EOS command token.
        assert (log_prob is None) or log_prob.requires_grad
        assert isinstance(entropy, float)

    def test_own_latent_score_is_deterministic(self) -> None:
        # Scoring from the proposal's OWN latent is deterministic (a stable
        # reconstruction-likelihood); the default fresh-prior path is not.
        gen = NeuralVAEGenerator(device="cpu")
        proposal = gen.generate_for_training("a 20mm cube")
        z = proposal.latent_vector
        assert z is not None

        lp1, _ = gen.score_token_sequence(proposal.token_ids, latent=z)
        lp2, _ = gen.score_token_sequence(proposal.token_ids, latent=z)
        assert lp1 is not None and lp2 is not None
        assert torch.allclose(lp1, lp2), "own-latent score must be deterministic"

        # The fresh-prior default varies across calls (one-sample estimate).
        fresh = [gen.score_token_sequence(proposal.token_ids)[0] for _ in range(5)]
        fresh_vals = {round(float(x), 4) for x in fresh if x is not None}
        assert len(fresh_vals) > 1, "fresh-prior scores should not be constant"


@pytest.mark.requires_torch
class TestEvalHarnessIntegration:
    def test_evaluate_validity_reports_sequence_log_prob(self, tmp_path) -> None:
        # The scorer is wired into the eval harness: evaluate_validity reports a
        # mean_sequence_log_prob, scored from each proposal's own latent.
        from ll_gen.proposals.disposal_result import DisposalResult
        from ll_gen.training.evaluate_validity import evaluate_validity

        gen = NeuralVAEGenerator(device="cpu")
        gen._init_model()

        # Deterministic, CadQuery-free dispose so the test is a unit test of the
        # metric wiring, not of the OCC kernel.
        def fake_dispose(proposal):
            return DisposalResult(is_valid=True, reward_signal=1.0)

        metrics = evaluate_validity(
            gen,
            prompts=["a 20mm cube", "a bracket"],
            n_samples=2,
            decode_mode="inference",
            output_dir=str(tmp_path),
            dispose_fn=fake_dispose,
        )
        # A real, finite reconstruction-likelihood was aggregated and reported.
        assert "mean_sequence_log_prob" in metrics.summary()
        assert metrics.mean_sequence_log_prob != 0.0
        assert metrics.mean_sequence_log_prob < 0.0  # log-prob of a real seq
