"""Regression: RLAlignmentTrainer.load_checkpoint must tolerate architecture drift.

A genuinely-trained checkpoint saved under an older model architecture (e.g. the
M3 VAE checkpoint, saved before STEPVAE gained the optional ``dim_encoder``
dimension-conditioning layer) must still load. Previously ``load_checkpoint``
called ``load_state_dict`` strictly and raised ``RuntimeError: Missing key(s)
... dim_encoder.weight`` -- which made the real 95%-validity checkpoint
unusable via the normal API. The loader now loads non-strictly and warns.
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# repo/ll_gen/tests/<this> -> repo/ll_gen/checkpoints/vae_rl_solid.pt
_CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "vae_rl_solid.pt"


def test_load_checkpoint_tolerates_missing_keys(tmp_path):
    if not _CKPT.exists():
        pytest.skip("trained VAE checkpoint fixture not present")

    from ll_gen.training.rl_trainer import RLAlignmentTrainer
    from ll_gen.training.run import build_generator

    generator = build_generator("vae", "cpu")
    trainer = RLAlignmentTrainer(generator, device="cpu", output_dir=str(tmp_path))
    trainer._init_training()

    # Must NOT raise even though the checkpoint predates the dim_encoder layer.
    trainer.load_checkpoint(_CKPT)
    assert generator._model is not None

    # The current model genuinely has the key the checkpoint lacks (so the test
    # is exercising the resilient path, not a no-op).
    state_keys = set(generator._model.state_dict().keys())
    assert any("dim_encoder" in k for k in state_keys)
