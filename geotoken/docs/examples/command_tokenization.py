#!/usr/bin/env python3
"""Example: CAD command sequence tokenization."""
from __future__ import annotations

from geotoken import (
    CommandSequenceTokenizer,
    CADVocabulary,
    CommandType,
)


def create_square_extrude() -> list[dict]:
    """Create commands for a square extrusion."""
    return [
        {"type": "SOL", "params": [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"type": "LINE", "params": [0.0, 0.0, 0, 1.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"type": "LINE", "params": [1.0, 0.0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"type": "LINE", "params": [1.0, 1.0, 0, 0.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"type": "LINE", "params": [0.0, 1.0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"type": "EXTRUDE", "params": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0]},
        {"type": "EOS", "params": [0] * 16},
    ]


def main():
    print("=== GeoToken Command Tokenization Example ===\n")

    # Create sample commands
    commands = create_square_extrude()
    print(f"Input: {len(commands)} commands")
    for cmd in commands:
        active = [p for p in cmd["params"] if p != 0]
        print(f"  {cmd['type']}: {active if active else '(no params)'}")

    # Tokenize
    tokenizer = CommandSequenceTokenizer()
    token_seq = tokenizer.tokenize(commands)

    print(f"\nTokenization results:")
    print(f"  Command tokens: {len(token_seq.command_tokens)}")

    # Show token details
    print("\nToken breakdown:")
    for i, tok in enumerate(token_seq.command_tokens[:7]):  # Show first 7
        cmd_name = CommandType(tok.command_type).name
        active_count = sum(tok.parameter_mask)
        print(f"  [{i}] {cmd_name}: {active_count} active params")

    # Encode to integer IDs
    vocab = CADVocabulary()
    token_ids = vocab.encode(token_seq.command_tokens)

    print(f"\nVocabulary encoding:")
    print(f"  Vocab size: {vocab.vocab_size}")
    print(f"  Encoded sequence length: {len(token_ids)}")
    print(f"  Token IDs (first 10): {token_ids[:10]}")

    # Decode back
    decoded = vocab.decode(token_ids)
    print(f"\nRoundtrip verification:")
    print(f"  Decoded tokens: {len(decoded)}")

    # Dequantize parameters
    reconstructed = tokenizer.dequantize_parameters(token_seq.command_tokens[:7])
    print(f"\nDequantized commands:")
    for cmd in reconstructed:
        active = [f"{p:.3f}" for p in cmd["params"] if abs(p) > 1e-6]
        cmd_name = cmd["type"].name if hasattr(cmd["type"], "name") else str(cmd["type"])
        print(f"  {cmd_name}: {active if active else '(no params)'}")


if __name__ == "__main__":
    main()
