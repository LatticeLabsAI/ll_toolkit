"""Regression: integer command index -> geotoken str CommandType mapping.

The generative models' command head emits indices in stepnet's *IntEnum*
``CommandType`` space (SOL=0, LINE=1, ARC=2, CIRCLE=3, EXTRUDE=4, EOS=5).
geotoken's ``CommandType`` is a *str* Enum (ARC="ARC", ...). The decoder in
``CADGenerationPipeline._decode_to_token_sequence`` previously passed the int
index straight into geotoken's str-enum -- ``geotoken.CommandType(2)`` -- which
raises ``ValueError: 2 is not a valid CommandType``. That exception was caught
and turned into an empty "fallback" result, so **every** generated sample
(VAE / VQ-VAE / diffusion) decoded to zero commands and scored 0% valid.

The fix maps int -> stepnet member (by value) -> geotoken member (by name).
These tests pin both the root cause and the fix.

IMPORTANT: torch is imported by conftest.py BEFORE this module loads.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_int_index_is_not_a_valid_geotoken_commandtype():
    """Document the root cause: the int path raises on geotoken's str enum."""
    geo = pytest.importorskip("geotoken.tokenizer.token_types")
    from stepnet.output_heads import CommandType as StepCT

    GeoCT = geo.CommandType
    # stepnet ARC has integer value 2; geotoken's enum has no value `2`.
    assert StepCT.ARC.value == 2
    with pytest.raises(ValueError):
        GeoCT(2)
    # The fix's mapping (by name) is valid for every model command index.
    for idx in range(len(StepCT)):
        step = StepCT(idx)
        assert GeoCT[step.name].name == step.name


def test_decode_to_token_sequence_handles_all_command_indices():
    """The pipeline decode must turn each model command index into a real
    geotoken command token, not crash into an empty fallback."""
    pytest.importorskip("geotoken")
    import torch.nn as nn

    from geotoken.tokenizer.token_types import CommandType as GeoCT
    from stepnet.generation_pipeline import CADGenerationPipeline
    from stepnet.output_heads import NUM_COMMAND_TYPES, NUM_PARAM_SLOTS

    pipe = CADGenerationPipeline(model=nn.Linear(4, 4), mode="vqvae", device="cpu")

    # Command logits whose argmax selects SOL(0), ARC(2 -- the crashing index), EOS(5).
    seq_len = 3
    cmd_logits = torch.full((1, seq_len, NUM_COMMAND_TYPES), -10.0)
    for s, idx in enumerate([0, 2, 5]):
        cmd_logits[0, s, idx] = 10.0
    param_logits = [torch.zeros(1, seq_len, 256) for _ in range(NUM_PARAM_SLOTS)]

    token_seq = pipe._decode_to_token_sequence(cmd_logits, param_logits, batch_index=0)

    decoded = [t.command_type for t in token_seq.command_tokens]
    # ARC (model index 2) decoded correctly instead of raising.
    assert GeoCT.ARC in decoded
    assert GeoCT.SOL in decoded
