"""Integration tests for command tokenizer pipeline."""
from __future__ import annotations

import json
import numpy as np
import pytest

from geotoken.tokenizer.command_tokenizer import CommandSequenceTokenizer
from geotoken.tokenizer.vocabulary import CADVocabulary, BOS_TOKEN_ID, EOS_TOKEN_ID
from geotoken.tokenizer.token_types import CommandType
from geotoken.config import CommandTokenizationConfig, PrecisionTier


class TestFullPipeline:
    """Integration tests for full tokenization pipeline."""

    def test_deepcad_format_tokenize_encode_decode_dequantize(self):
        """Test full pipeline: DeepCAD JSON -> tokenize -> encode -> decode -> dequantize."""
        # Sample DeepCAD-style construction history
        construction_history = {
            "sequence": [
                {"type": "SOL", "params": [0.0, 0.0]},
                {"type": "Line", "params": [0.0, 0.0, 1.0, 0.0]},
                {"type": "Line", "params": [1.0, 0.0, 1.0, 1.0]},
                {"type": "Line", "params": [1.0, 1.0, 0.0, 1.0]},
                {"type": "Line", "params": [0.0, 1.0, 0.0, 0.0]},
                {"type": "Ext", "params": [0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]},
            ]
        }

        # Step 1: Tokenize
        tokenizer = CommandSequenceTokenizer()
        token_seq = tokenizer.tokenize(construction_history)

        assert len(token_seq.command_tokens) > 0
        assert token_seq.metadata["source_format"] == "deepcad"

        # Step 2: Encode to integer IDs
        vocab = CADVocabulary()
        ids = vocab.encode(token_seq.command_tokens)

        assert ids[0] == BOS_TOKEN_ID
        assert ids[-1] == EOS_TOKEN_ID
        assert len(ids) > 2

        # Step 3: Decode back to CommandTokens
        decoded_tokens = vocab.decode(ids)

        assert len(decoded_tokens) > 0

        # Step 4: Dequantize back to continuous parameters
        dequantized = tokenizer.dequantize_parameters(decoded_tokens)

        assert len(dequantized) > 0
        # Check that command types are preserved
        original_types = [CommandType.SOL, CommandType.LINE, CommandType.LINE,
                         CommandType.LINE, CommandType.LINE, CommandType.EXTRUDE]
        for i, cmd in enumerate(dequantized[:6]):
            if i < len(original_types):
                assert cmd["type"] == original_types[i]

    def test_pipeline_with_list_input(self):
        """Test pipeline with list input (no 'sequence' key)."""
        commands = [
            {"type": "Circle", "params": [0.5, 0.5, 0.25]},
            {"type": "Ext", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]},
        ]

        tokenizer = CommandSequenceTokenizer()
        token_seq = tokenizer.tokenize(commands)
        vocab = CADVocabulary()

        ids = vocab.encode(token_seq.command_tokens)
        decoded = vocab.decode(ids)
        dequantized = tokenizer.dequantize_parameters(decoded)

        assert len(dequantized) >= 1

    def test_pipeline_with_arc_command(self):
        """Test pipeline handles ARC commands correctly."""
        commands = [
            {"type": "SOL", "params": [0.0, 0.0]},
            {"type": "Arc", "params": [0.0, 0.0, 0.5, 0.5, 1.0, 0.0]},
            {"type": "Ext", "params": [0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]},
        ]

        tokenizer = CommandSequenceTokenizer()
        token_seq = tokenizer.tokenize(commands)
        vocab = CADVocabulary()

        ids = vocab.encode(token_seq.command_tokens)
        decoded = vocab.decode(ids)

        # Find ARC in decoded
        arc_tokens = [t for t in decoded if t.command_type == CommandType.ARC]
        assert len(arc_tokens) >= 1

    def test_pipeline_preserves_command_order(self):
        """Test that pipeline preserves command order."""
        commands = [
            {"type": "SOL", "params": [0.0, 0.0]},
            {"type": "Line", "params": [0.0, 0.0, 1.0, 0.0]},
            {"type": "Circle", "params": [0.5, 0.5, 0.1]},
            {"type": "Line", "params": [1.0, 0.0, 1.0, 1.0]},
            {"type": "Ext", "params": [0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]},
        ]

        tokenizer = CommandSequenceTokenizer()
        token_seq = tokenizer.tokenize(commands)
        vocab = CADVocabulary()

        ids = vocab.encode(token_seq.command_tokens)
        decoded = vocab.decode(ids)

        expected_order = [CommandType.SOL, CommandType.LINE, CommandType.CIRCLE,
                         CommandType.LINE, CommandType.EXTRUDE]
        for i, token in enumerate(decoded[:len(expected_order)]):
            assert token.command_type == expected_order[i], f"Order mismatch at {i}"


class TestRoundtripQuality:
    """Tests for roundtrip quality analysis."""

    def test_roundtrip_quality_low_error(self):
        """Test that roundtrip has low quantization error."""
        commands = [
            {"type": "Line", "params": [0.0, 0.0, 0.5, 0.5]},
        ]

        tokenizer = CommandSequenceTokenizer()
        quality = tokenizer.analyze_roundtrip_quality(commands)

        # With 8-bit quantization, error should be small
        assert quality["param_mse"] < 0.01
        assert quality["max_error"] < 0.1
        assert quality["command_preservation_rate"] == 1.0

    def test_roundtrip_quality_higher_precision(self):
        """Test that higher precision reduces error."""
        commands = [
            {"type": "Line", "params": [0.1, 0.2, 0.3, 0.4]},
        ]

        config_standard = CommandTokenizationConfig(
            parameter_quantization=PrecisionTier.STANDARD
        )
        config_precision = CommandTokenizationConfig(
            parameter_quantization=PrecisionTier.PRECISION
        )

        tokenizer_std = CommandSequenceTokenizer(command_config=config_standard)
        tokenizer_prec = CommandSequenceTokenizer(command_config=config_precision)

        quality_std = tokenizer_std.analyze_roundtrip_quality(commands)
        quality_prec = tokenizer_prec.analyze_roundtrip_quality(commands)

        # Higher precision should have lower or equal error
        assert quality_prec["param_mse"] <= quality_std["param_mse"] + 1e-6


class TestVocabularyIntegration:
    """Tests for vocabulary integration with tokenizer."""

    def test_vocab_size_sufficient(self):
        """Test that vocab size is sufficient for all commands."""
        vocab = CADVocabulary()
        tokenizer = CommandSequenceTokenizer()

        # Create a complex sequence
        commands = [
            {"type": "SOL", "params": [0.0, 0.0]},
            {"type": "Line", "params": [0.0, 0.0, 1.0, 0.0]},
            {"type": "Arc", "params": [0.0, 0.0, 0.5, 0.5, 1.0, 0.0]},
            {"type": "Circle", "params": [0.5, 0.5, 0.25]},
            {"type": "Ext", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]},
        ]

        token_seq = tokenizer.tokenize(commands)
        ids = vocab.encode(token_seq.command_tokens)

        # All IDs should be within vocab range
        assert all(0 <= id < vocab.vocab_size for id in ids)

    def test_consistent_encoding_across_instances(self):
        """Test that different vocab instances produce same encoding."""
        vocab1 = CADVocabulary()
        vocab2 = CADVocabulary()
        tokenizer = CommandSequenceTokenizer()

        commands = [{"type": "Line", "params": [0.0, 0.0, 1.0, 1.0]}]
        token_seq = tokenizer.tokenize(commands)

        ids1 = vocab1.encode(token_seq.command_tokens)
        ids2 = vocab2.encode(token_seq.command_tokens)

        assert ids1 == ids2


class TestEdgeCases:
    """Tests for edge cases in the pipeline."""

    def test_empty_sequence(self):
        """Test handling of empty sequence."""
        tokenizer = CommandSequenceTokenizer()
        vocab = CADVocabulary()

        token_seq = tokenizer.tokenize([])
        ids = vocab.encode(token_seq.command_tokens)
        decoded = vocab.decode(ids)

        assert ids == [BOS_TOKEN_ID, EOS_TOKEN_ID]

    def test_unknown_command_skipped(self):
        """Test that unknown commands are gracefully skipped."""
        commands = [
            {"type": "UnknownCommand", "params": [1.0]},
            {"type": "Line", "params": [0.0, 0.0, 1.0, 0.0]},
        ]

        tokenizer = CommandSequenceTokenizer()
        token_seq = tokenizer.tokenize(commands)

        # Unknown should be skipped, only LINE remains
        non_pad_tokens = [t for t in token_seq.command_tokens
                         if t.command_type != CommandType.EOS]
        assert len(non_pad_tokens) >= 1
        assert any(t.command_type == CommandType.LINE for t in non_pad_tokens)

    def test_extreme_parameter_values(self):
        """Test handling of extreme parameter values."""
        commands = [
            {"type": "Line", "params": [1e6, -1e6, 1e6, 1e6]},  # Very large
        ]

        tokenizer = CommandSequenceTokenizer()
        vocab = CADVocabulary()

        token_seq = tokenizer.tokenize(commands)
        ids = vocab.encode(token_seq.command_tokens)
        decoded = vocab.decode(ids)
        dequantized = tokenizer.dequantize_parameters(decoded)

        # Should not crash, values should be clamped
        assert len(dequantized) >= 1
