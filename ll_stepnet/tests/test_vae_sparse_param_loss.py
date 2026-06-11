"""Regression: STEPVAE.forward must return a finite loss when some parameter
slots are inactive across the whole batch.

The 6-command CAD schema (SOL/LINE/ARC/CIRCLE/EXTRUDE/EOS) only ever activates
parameter slots 0-7, so the param heads for slots 8-15 always receive an
all-``ignore_index`` target.  Previously the supervised forward computed
``F.cross_entropy`` per head with ``reduction='mean'``; over an all-ignored
target that averages zero elements and returns NaN, poisoning ``recon_loss`` and
``loss``.  This made the supervised reconstruction forward (used for DeepCAD
pretraining) unusable.  The forward now skips all-ignored heads and averages
over the contributing ones.

IMPORTANT: torch is imported by conftest.py BEFORE this module loads to avoid
OpenMP conflicts on macOS.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _build_vae(sample_encoder_config):
    from stepnet import STEPVAE

    model = STEPVAE(
        encoder_config=sample_encoder_config,
        latent_dim=32,
        max_seq_len=12,
    )
    model.train()
    return model


def test_forward_finite_loss_with_inactive_param_slots(sample_encoder_config, device):
    """Slots 8-15 are never active for any command -> loss must stay finite."""
    model = _build_vae(sample_encoder_config).to(device)

    batch, seq = 2, 6
    # SOL(0), LINE(1), CIRCLE(3), EXTRUDE(4), EOS(5), pad(-1)
    command_targets = torch.tensor(
        [[0, 1, 3, 4, 5, -1], [0, 1, 1, 4, 5, -1]], device=device
    )
    token_ids = command_targets.clamp(min=0)
    attention_mask = (command_targets != -1).long()

    # Only slots 0-7 ever carry a target; slots 8-15 stay -1 everywhere.
    param_targets = torch.full((batch, seq, 16), -1, dtype=torch.long, device=device)
    param_targets[:, 1, 0:4] = torch.randint(0, 256, (batch, 4), device=device)  # LINE
    param_targets[0, 2, 0:3] = torch.randint(0, 256, (3,), device=device)  # CIRCLE
    param_targets[:, 3, 0:8] = torch.randint(0, 256, (batch, 8), device=device)  # EXTRUDE

    out = model(
        token_ids,
        attention_mask=attention_mask,
        command_targets=command_targets,
        param_targets=param_targets,
    )

    assert torch.isfinite(out["loss"]).item(), "total loss must be finite"
    assert torch.isfinite(out["recon_loss"]).item(), "recon_loss must be finite"
    # The loss must be differentiable and produce real gradients.
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "backward must populate gradients"
    assert all(torch.isfinite(g).all().item() for g in grads), "gradients must be finite"


def test_forward_finite_when_only_one_command_type_present(sample_encoder_config, device):
    """Extreme case: a batch of only CIRCLE commands (slots 3-15 all-ignored)."""
    model = _build_vae(sample_encoder_config).to(device)

    command_targets = torch.tensor([[0, 3, 5], [0, 3, 5]], device=device)  # SOL, CIRCLE, EOS
    token_ids = command_targets.clamp(min=0)
    attention_mask = torch.ones_like(command_targets)

    param_targets = torch.full((2, 3, 16), -1, dtype=torch.long, device=device)
    param_targets[:, 1, 0:3] = torch.randint(0, 256, (2, 3), device=device)  # CIRCLE only

    out = model(
        token_ids,
        attention_mask=attention_mask,
        command_targets=command_targets,
        param_targets=param_targets,
    )
    assert torch.isfinite(out["loss"]).item()
    assert torch.isfinite(out["recon_loss"]).item()
