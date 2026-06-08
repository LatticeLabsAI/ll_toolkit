"""Checkpoint save/load round-trip for the RL trainer (SPEC-1 M2, T2.5).

After one micro training step the trainer's state (model weights, baseline,
step count) is saved and loaded into a *fresh* trainer/generator; the restored
model must match the saved one bit-for-bit, and the bookkeeping must round-trip.
Guards against the "load always restarts from scratch" class of bug.
"""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import ll_gen.training.rl_trainer as rl_mod
from ll_gen.generators.neural_vae import NeuralVAEGenerator
from ll_gen.training.rl_trainer import RLAlignmentTrainer


def _stub_reward(trainer: RLAlignmentTrainer, monkeypatch) -> None:
    monkeypatch.setattr(
        trainer._disposal_engine,
        "dispose",
        lambda proposal, export=False: types.SimpleNamespace(is_valid=False),
    )
    monkeypatch.setattr(
        rl_mod, "compute_reward",
        lambda result, config=None, target_dimensions=None: 1.0,
    )


@pytest.mark.requires_torch
def test_checkpoint_roundtrip_restores_state(tmp_path, monkeypatch) -> None:
    # Trainer A: take one real step so weights + bookkeeping are non-default.
    gen_a = NeuralVAEGenerator(device="cpu")
    gen_a._init_model()
    trainer_a = RLAlignmentTrainer(
        gen_a, learning_rate=1e-2, device="cpu", output_dir=str(tmp_path / "a"), seed=0
    )
    trainer_a._init_training()
    _stub_reward(trainer_a, monkeypatch)
    trainer_a.train_step("a 20mm cube")

    ckpt = tmp_path / "ckpt.pt"
    trainer_a.save_checkpoint(ckpt)
    assert ckpt.exists()

    # Trainer B: a fresh generator (different random init) then load.
    gen_b = NeuralVAEGenerator(device="cpu")
    gen_b._init_model()
    trainer_b = RLAlignmentTrainer(
        gen_b, learning_rate=1e-2, device="cpu", output_dir=str(tmp_path / "b"), seed=1
    )
    trainer_b._init_training()

    sd_a = gen_a._model.state_dict()
    sd_b_before = gen_b._model.state_dict()
    # Sanity: independent random inits differ before loading.
    assert any(not torch.equal(sd_a[k], sd_b_before[k]) for k in sd_a)

    trainer_b.load_checkpoint(ckpt)

    sd_b_after = gen_b._model.state_dict()
    assert sd_a.keys() == sd_b_after.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b_after[key]), f"param {key} not restored"

    # Bookkeeping restored.
    assert trainer_b._step_count == trainer_a._step_count == 1
    assert trainer_b._baseline == pytest.approx(trainer_a._baseline)
