"""train_epoch smoke test (SPEC-1 M2, T2.6).

Runs one epoch over a small in-memory dataset and asserts the loop completes,
performs real per-sample steps, and reports finite aggregate metrics. The
disposal/reward oracle is stubbed (deterministic reward) so the test exercises
the full per-sample train_step path without pythonocc/cadquery.
"""
from __future__ import annotations

import math
import types

import pytest

torch = pytest.importorskip("torch")

import ll_gen.training.rl_trainer as rl_mod
from ll_gen.generators.neural_vqvae import NeuralVQVAEGenerator
from ll_gen.training.rl_trainer import RLAlignmentTrainer


@pytest.mark.requires_torch
def test_train_epoch_runs_over_small_dataset(tmp_path, monkeypatch) -> None:
    gen = NeuralVQVAEGenerator(device="cpu")
    gen._init_model()
    trainer = RLAlignmentTrainer(
        gen, learning_rate=1e-3, device="cpu", output_dir=str(tmp_path), seed=0
    )
    trainer._init_training()

    # Deterministic reward oracle (no cadquery/OCC).
    monkeypatch.setattr(
        trainer._disposal_engine,
        "dispose",
        lambda proposal, export=False: types.SimpleNamespace(is_valid=False),
    )
    monkeypatch.setattr(
        rl_mod, "compute_reward",
        lambda result, config=None, target_dimensions=None: 0.5,
    )

    dataset = [{"prompt": f"shape {i}"} for i in range(4)]
    metrics = trainer.train_epoch(dataset, shuffle=False)

    assert {"mean_reward", "mean_loss", "validity_rate", "epoch_time_ms"} <= set(metrics)
    assert metrics["mean_reward"] == pytest.approx(0.5)
    assert math.isfinite(metrics["mean_loss"])
    # One real step per sample.
    assert trainer._step_count == 4
    assert len(trainer.get_training_history()) == 4
