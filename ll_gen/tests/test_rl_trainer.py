"""RLAlignmentTrainer.train_step gradient-update test (SPEC-1 M2, T2.3).

Proves a single train_step performs a real REINFORCE update: it runs the live
generator, computes a policy-gradient loss from proposal.log_probs, backprops,
and changes the model parameters. The disposal/reward oracle is stubbed with a
deterministic reward so the test needs no cadquery/pythonocc — the gradient
mechanics under test are entirely real.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import ll_gen.training.rl_trainer as rl_mod
from ll_gen.generators.neural_vae import NeuralVAEGenerator
from ll_gen.training.rl_trainer import RLAlignmentTrainer


@pytest.mark.requires_torch
def test_train_step_updates_params(tmp_path, monkeypatch) -> None:
    gen = NeuralVAEGenerator(device="cpu")
    gen._init_model()
    trainer = RLAlignmentTrainer(
        gen, learning_rate=1e-2, device="cpu", output_dir=str(tmp_path), seed=0
    )
    trainer._init_training()

    # Inject a deterministic reward path: dispose returns a stub result and
    # compute_reward returns a fixed nonzero reward (no cadquery/OCC needed).
    monkeypatch.setattr(
        trainer._disposal_engine,
        "dispose",
        lambda proposal, export=False: types.SimpleNamespace(is_valid=False),
    )
    monkeypatch.setattr(
        rl_mod,
        "compute_reward",
        lambda result, config=None, target_dimensions=None: 1.0,
    )

    before = [p.detach().clone() for p in gen._model.parameters()]
    result = trainer.train_step("a 20mm cube")

    # Metrics contract.
    assert {"reward", "advantage", "baseline", "loss"} <= set(result)
    assert result["reward"] == 1.0
    # Step 0: advantage = reward - baseline(0.0) = 1.0 (nonzero policy gradient).
    assert result["advantage"] == pytest.approx(1.0)
    assert not result["failed"]

    # At least one parameter must have moved (those on the log-prob graph).
    after = list(gen._model.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert changed, "no model parameter changed after train_step"


@pytest.mark.requires_torch
def test_train_step_zero_reward_no_first_step_collapse(tmp_path, monkeypatch) -> None:
    """With reward 0 at step 0, advantage is 0 -> loss 0 (no spurious update).
    Confirms the baseline-before-update ordering doesn't crash and reports 0."""
    gen = NeuralVAEGenerator(device="cpu")
    gen._init_model()
    trainer = RLAlignmentTrainer(
        gen, learning_rate=1e-2, device="cpu", output_dir=str(tmp_path), seed=0
    )
    trainer._init_training()
    monkeypatch.setattr(
        trainer._disposal_engine,
        "dispose",
        lambda proposal, export=False: types.SimpleNamespace(is_valid=False),
    )
    monkeypatch.setattr(
        rl_mod,
        "compute_reward",
        lambda result, config=None, target_dimensions=None: 0.0,
    )
    result = trainer.train_step("a 20mm cube")
    assert result["reward"] == 0.0
    assert result["advantage"] == pytest.approx(0.0)
