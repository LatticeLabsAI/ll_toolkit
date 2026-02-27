"""Shared tokenization constants and utilities for CAD command sequences.

This module centralizes the token ID definitions and quantization logic
used by both DeepCAD and Text2CAD dataset loaders, ensuring consistency
across all command-sequence-based datasets.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Special token IDs
# ---------------------------------------------------------------------------
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2

# ---------------------------------------------------------------------------
# Command type IDs
# ---------------------------------------------------------------------------
COMMAND_TYPE_IDS: Dict[str, int] = {
    "SOL": 6,
    "LINE": 7,
    "ARC": 8,
    "CIRCLE": 9,
    "EXTRUDE": 10,
    "EOS": 11,
}

# Parameter tokens start at this offset
PARAM_OFFSET = 12

# Maximum reasonable vocab size
_MAX_VOCAB_SIZE = 65536


def validate_token_space(quantization_bits: int) -> None:
    """Validate that the token space does not exceed the maximum vocab size.

    Args:
        quantization_bits: Number of bits used for parameter quantization.

    Raises:
        ValueError: If ``PARAM_OFFSET + 2**quantization_bits`` exceeds
            the maximum vocab size of 65536.
    """
    required = PARAM_OFFSET + 2**quantization_bits
    if required > _MAX_VOCAB_SIZE:
        raise ValueError(
            f"Token space overflow: PARAM_OFFSET ({PARAM_OFFSET}) + "
            f"2**{quantization_bits} ({2**quantization_bits}) = {required} "
            f"exceeds maximum vocab size ({_MAX_VOCAB_SIZE})"
        )


def quantize_parameter(
    value: Any,
    quantization_bits: int,
    normalization_range: float = 2.0,
) -> int:
    """Quantize a continuous parameter to a discrete token level.

    The parameter is first mapped from the symmetric range
    ``[-normalization_range, +normalization_range]`` to ``[0, 1]`` via::

        normalized = (value + normalization_range) / (2 * normalization_range)

    then quantized to an integer in ``[0, 2**quantization_bits - 1]``.
    This preserves negative coordinates (e.g. DeepCAD's ``[-1, 1]`` range).

    Args:
        value: Raw parameter value (numeric or convertible to float).
        quantization_bits: Number of quantization bits.
        normalization_range: Half-extent of the symmetric range (default 2.0,
            meaning values in ``[-2, 2]`` are representable).

    Returns:
        Integer quantized value in ``[0, 2**quantization_bits - 1]``.
    """
    quantization_levels = 2**quantization_bits
    try:
        normalized = (float(value) + normalization_range) / (2.0 * normalization_range)
        normalized = max(0.0, min(1.0, normalized))
    except (ValueError, TypeError):
        normalized = 0.0

    quantized = round(normalized * (quantization_levels - 1))
    quantized = max(0, min(quantization_levels - 1, quantized))
    return int(quantized)


def tokenize_command_sequence(
    commands: List[Dict[str, Any]],
    quantization_bits: int = 8,
    normalization_range: float = 2.0,
    max_commands: int = 60,
) -> Dict[str, Any]:
    """Tokenize a list of CAD commands into token IDs.

    Produces BOS, per-command type + parameter tokens, and EOS,
    then pads to ``max_commands * 10``.

    Args:
        commands: List of command dicts with ``type`` and ``params`` keys.
        quantization_bits: Bits for parameter quantization.
        normalization_range: Divisor for parameter normalization.
        max_commands: Maximum number of commands to process.

    Returns:
        Dictionary with ``token_ids``, ``command_tokens``,
        ``attention_mask``, and ``num_commands``.

    Raises:
        ValueError: If token space exceeds maximum vocab size.
    """
    validate_token_space(quantization_bits)

    token_ids: List[int] = [BOS_TOKEN_ID]
    command_tokens: List[Dict[str, Any]] = []
    attention_mask: List[int] = [1]

    for i, cmd in enumerate(commands):
        if i >= max_commands:
            break

        cmd_type = cmd.get("type", "")
        params = cmd.get("params", [])

        if cmd_type in COMMAND_TYPE_IDS:
            cmd_type_id = COMMAND_TYPE_IDS[cmd_type]
            token_ids.append(cmd_type_id)
            attention_mask.append(1)

            quantized_params: List[int] = []
            param_mask: List[int] = []

            for param in params:
                quantized = quantize_parameter(
                    param, quantization_bits, normalization_range
                )
                quantized_params.append(quantized)
                param_mask.append(1)

                param_token_id = PARAM_OFFSET + quantized
                token_ids.append(param_token_id)
                attention_mask.append(1)

            command_tokens.append({
                "command_type": cmd_type_id,
                "parameters": quantized_params,
                "parameter_mask": param_mask,
            })

    # Add EOS token
    token_ids.append(EOS_TOKEN_ID)
    attention_mask.append(1)

    num_commands = len(command_tokens)

    # Pad to max_commands * 10
    max_len = max_commands * 10
    while len(token_ids) < max_len:
        token_ids.append(PAD_TOKEN_ID)
        attention_mask.append(0)

    token_ids = token_ids[:max_len]
    attention_mask = attention_mask[:max_len]

    return {
        "token_ids": token_ids,
        "command_tokens": command_tokens,
        "attention_mask": attention_mask,
        "num_commands": num_commands,
    }
