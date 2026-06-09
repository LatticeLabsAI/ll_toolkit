"""Tests for the STEPVAE supervised warm-start (SPEC-1 M3, T3.3).

``build_targets`` is pure (no torch) and tested directly; an end-to-end smoke
test runs one warm-start epoch on a tiny temp fixture (requires torch, no OCC).
"""

from __future__ import annotations

import json

import pytest

from ll_gen.training.warm_start import build_targets


def _box_command_tokens():
    # command_type ids: SOL=6, LINE=7, EXTRUDE=10, EOS=11.
    return [
        {"command_type": 6, "parameters": [], "parameter_mask": []},
        {"command_type": 7, "parameters": [10, 20, 30, 40], "parameter_mask": [1] * 4},
        {"command_type": 10, "parameters": [128], "parameter_mask": [1]},
        {"command_type": 11, "parameters": [], "parameter_mask": []},
    ]


def test_build_targets_shapes_and_padding() -> None:
    tok, amask, cmd_t, par_t = build_targets(
        _box_command_tokens(), max_commands=8, num_param_slots=16, num_command_types=6
    )
    assert len(tok) == len(amask) == len(cmd_t) == 8
    assert len(par_t) == 8 and all(len(row) == 16 for row in par_t)
    # 4 real commands, rest padding.
    assert amask == [1, 1, 1, 1, 0, 0, 0, 0]
    assert tok[:4] == [6, 7, 10, 11]
    assert tok[4:] == [0, 0, 0, 0]


def test_build_targets_command_class_offset() -> None:
    _, _, cmd_t, _ = build_targets(
        _box_command_tokens(), max_commands=8, num_param_slots=16, num_command_types=6
    )
    # class = token id - 6: SOL->0, LINE->1, EXTRUDE->4, EOS->5.
    assert cmd_t[:4] == [0, 1, 4, 5]
    assert cmd_t[4:] == [-1, -1, -1, -1]


def test_build_targets_param_levels_and_ignore() -> None:
    _, _, _, par_t = build_targets(
        _box_command_tokens(), max_commands=8, num_param_slots=16, num_command_types=6
    )
    # LINE (row 1) fills its 4 params; remaining slots are -1 (ignore).
    assert par_t[1][:4] == [10, 20, 30, 40]
    assert par_t[1][4:] == [-1] * 12
    # SOL/EOS rows have no params at all.
    assert par_t[0] == [-1] * 16
    # EXTRUDE (row 2) fills only its single depth param.
    assert par_t[2][0] == 128 and par_t[2][1:] == [-1] * 15


def test_build_targets_truncates_to_max_commands() -> None:
    tok, amask, cmd_t, _ = build_targets(
        _box_command_tokens(), max_commands=2, num_param_slots=16, num_command_types=6
    )
    assert len(tok) == 2 and amask == [1, 1]
    assert cmd_t == [0, 1]


def test_build_targets_skips_out_of_range_command_types() -> None:
    # Token id 2 (EOS special) -> class -4, out of range -> treated as padding.
    tokens = [{"command_type": 2, "parameters": [], "parameter_mask": []}]
    tok, amask, cmd_t, _ = build_targets(
        tokens, max_commands=4, num_param_slots=16, num_command_types=6
    )
    assert amask == [0, 0, 0, 0]
    assert cmd_t == [-1, -1, -1, -1]


@pytest.mark.requires_torch
def test_warm_start_one_epoch_runs_and_saves(tmp_path) -> None:
    pytest.importorskip("torch")
    from ll_gen.training.warm_start import warm_start

    # Tiny fixture: a handful of box sequences in the local-loader format.
    sequence = [
        {"type": "SOL", "params": []},
        {"type": "LINE", "params": [0.0, 0.0, 0.1, 0.0]},
        {"type": "LINE", "params": [0.1, 0.0, 0.1, 0.1]},
        {"type": "LINE", "params": [0.1, 0.1, 0.0, 0.1]},
        {"type": "LINE", "params": [0.0, 0.1, 0.0, 0.0]},
        {"type": "EXTRUDE", "params": [0.05]},
        {"type": "EOS", "params": []},
    ]
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    for i in range(8):
        (train_dir / f"{i:03d}.json").write_text(json.dumps({"sequence": sequence}))

    save_path = tmp_path / "vae_warm.pt"
    metrics = warm_start(
        data_dir=str(tmp_path),
        split="train",
        epochs=1,
        batch_size=4,
        max_samples=8,
        device="cpu",
        save_path=str(save_path),
        seed=0,
    )

    import math

    assert math.isfinite(metrics["recon_loss"])
    assert math.isfinite(metrics["total_loss"])
    assert metrics["num_samples"] == 8
    assert save_path.exists()


@pytest.mark.requires_torch
def test_warm_start_checkpoint_loads_into_generator(tmp_path) -> None:
    pytest.importorskip("torch")
    from ll_gen.training.evaluate_validity import load_generator_checkpoint
    from ll_gen.training.run import build_generator
    from ll_gen.training.warm_start import warm_start

    sequence = [
        {"type": "SOL", "params": []},
        {"type": "CIRCLE", "params": [0.0, 0.0, 0.01]},
        {"type": "EXTRUDE", "params": [0.04]},
        {"type": "EOS", "params": []},
    ]
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    for i in range(4):
        (train_dir / f"{i:03d}.json").write_text(json.dumps({"sequence": sequence}))

    save_path = tmp_path / "vae_warm.pt"
    warm_start(
        data_dir=str(tmp_path),
        split="train",
        epochs=1,
        batch_size=2,
        max_samples=4,
        device="cpu",
        save_path=str(save_path),
        seed=0,
    )
    # The saved flat state_dict must load back into a fresh generator's model.
    gen = build_generator("vae", "cpu")
    load_generator_checkpoint(gen, save_path)
