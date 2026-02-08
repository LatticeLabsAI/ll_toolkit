"""Tests for CommandSequenceTokenizer."""
from __future__ import annotations

import logging
import pytest
import numpy as np

from geotoken.tokenizer.command_tokenizer import CommandSequenceTokenizer
from geotoken.tokenizer.token_types import CommandToken, CommandType, TokenSequence
from geotoken.config import CommandTokenizationConfig, PrecisionTier


class TestCommandSequenceTokenizerInit:
    """Tests for CommandSequenceTokenizer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        tokenizer = CommandSequenceTokenizer()
        assert tokenizer.quant_config is not None
        assert tokenizer.seq_config is not None
        assert tokenizer.cmd_config is not None
        assert tokenizer._param_levels == 256  # 2^8 default

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = CommandTokenizationConfig(
            max_sequence_length=30,
            parameter_quantization=PrecisionTier.PRECISION,
        )
        tokenizer = CommandSequenceTokenizer(command_config=config)
        assert tokenizer.cmd_config.max_sequence_length == 30
        assert tokenizer._param_levels == 1024  # 2^10


class TestTokenize:
    """Tests for tokenize() method."""

    def test_tokenize_empty_list(self):
        """Test tokenizing empty sequence."""
        tokenizer = CommandSequenceTokenizer()
        result = tokenizer.tokenize([])
        assert isinstance(result, TokenSequence)
        # Should have padding if pad_to_max_length is True
        assert result.metadata["num_raw_commands"] == 0

    def test_tokenize_empty_dict(self):
        """Test tokenizing empty dict."""
        tokenizer = CommandSequenceTokenizer()
        result = tokenizer.tokenize({"sequence": []})
        assert isinstance(result, TokenSequence)
        assert result.metadata["num_raw_commands"] == 0

    def test_tokenize_invalid_type_raises(self):
        """Test that invalid input type raises TypeError."""
        tokenizer = CommandSequenceTokenizer()
        with pytest.raises(TypeError, match="must be dict or list"):
            tokenizer.tokenize("invalid")

    def test_tokenize_single_line(self):
        """Test tokenizing single LINE command."""
        tokenizer = CommandSequenceTokenizer()
        commands = [
            {"type": "Line", "params": [0.0, 0.0, 1.0, 1.0]}
        ]
        result = tokenizer.tokenize(commands)
        assert len(result.command_tokens) > 0

    def test_tokenize_with_dict_params(self):
        """Test tokenizing with dict-style params."""
        tokenizer = CommandSequenceTokenizer()
        commands = [
            {"type": "Line", "parameters": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}}
        ]
        result = tokenizer.tokenize(commands)
        assert len(result.command_tokens) > 0


class TestParseConstructionHistory:
    """Tests for parse_construction_history() method."""

    def test_parse_line_command(self):
        """Test parsing LINE command."""
        tokenizer = CommandSequenceTokenizer()
        commands = [{"type": "Line", "params": [0.0, 0.0, 1.0, 1.0]}]
        parsed = tokenizer.parse_construction_history(commands)
        assert len(parsed) == 1
        assert parsed[0]["type"] == CommandType.LINE
        assert parsed[0]["params"] == [0.0, 0.0, 1.0, 1.0]

    def test_parse_arc_command(self):
        """Test parsing ARC command."""
        tokenizer = CommandSequenceTokenizer()
        commands = [{"type": "Arc", "params": [0.0, 0.0, 0.5, 0.5, 1.0, 0.0]}]
        parsed = tokenizer.parse_construction_history(commands)
        assert len(parsed) == 1
        assert parsed[0]["type"] == CommandType.ARC
        assert len(parsed[0]["params"]) == 6

    def test_parse_circle_command(self):
        """Test parsing CIRCLE command."""
        tokenizer = CommandSequenceTokenizer()
        commands = [{"type": "Circle", "params": [0.5, 0.5, 0.25]}]
        parsed = tokenizer.parse_construction_history(commands)
        assert len(parsed) == 1
        assert parsed[0]["type"] == CommandType.CIRCLE
        assert len(parsed[0]["params"]) == 3

    def test_parse_extrude_command(self):
        """Test parsing EXTRUDE command."""
        tokenizer = CommandSequenceTokenizer()
        commands = [{"type": "Ext", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]}]
        parsed = tokenizer.parse_construction_history(commands)
        assert len(parsed) == 1
        assert parsed[0]["type"] == CommandType.EXTRUDE

    def test_parse_unknown_command_skipped(self, caplog):
        """Test that unknown commands are skipped with warning."""
        tokenizer = CommandSequenceTokenizer()
        commands = [{"type": "Unknown", "params": [1.0]}]
        with caplog.at_level(logging.WARNING):
            parsed = tokenizer.parse_construction_history(commands)
        assert len(parsed) == 0
        assert "Unknown command type" in caplog.text

    def test_parse_sol_command(self):
        """Test parsing SOL command."""
        tokenizer = CommandSequenceTokenizer()
        commands = [{"type": "SOL", "params": [0.5, 0.0]}]
        parsed = tokenizer.parse_construction_history(commands)
        assert len(parsed) == 1
        assert parsed[0]["type"] == CommandType.SOL


class TestNormalizeSketches:
    """Tests for normalize_sketches() method."""

    def test_normalize_sketches_centering(self):
        """Test that sketches are centered."""
        config = CommandTokenizationConfig(canonicalize_loops=True)
        tokenizer = CommandSequenceTokenizer(command_config=config)
        commands = [
            {"type": CommandType.LINE, "params": [10.0, 10.0, 20.0, 10.0]},
            {"type": CommandType.LINE, "params": [20.0, 10.0, 20.0, 20.0]},
        ]
        normalized = tokenizer.normalize_sketches(commands)
        # After centering, centroid should be near origin
        # Collect normalized points
        points = []
        for cmd in normalized:
            params = cmd["params"]
            points.extend([[params[0], params[1]], [params[2], params[3]]])
        points = np.array(points)
        centroid = points.mean(axis=0)
        # Centroid should be near 0 (within normalization range)
        assert abs(centroid[0]) < 2.0
        assert abs(centroid[1]) < 2.0

    def test_normalize_sketches_disabled(self):
        """Test that normalization is skipped when disabled."""
        config = CommandTokenizationConfig(canonicalize_loops=False)
        tokenizer = CommandSequenceTokenizer(command_config=config)
        commands = [
            {"type": CommandType.LINE, "params": [10.0, 10.0, 20.0, 10.0]},
        ]
        normalized = tokenizer.normalize_sketches(commands)
        # Should be unchanged
        assert normalized[0]["params"] == [10.0, 10.0, 20.0, 10.0]


class TestNormalize3D:
    """Tests for normalize_3d() method."""

    def test_normalize_3d_extrude_values(self):
        """Test 3D normalization scales extrude values."""
        tokenizer = CommandSequenceTokenizer()
        commands = [
            {"type": CommandType.SOL, "params": [0.0, 0.0]},
            {"type": CommandType.LINE, "params": [0.0, 0.0, 1.0, 0.0]},
            {"type": CommandType.EXTRUDE, "params": [10.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]},
        ]
        normalized = tokenizer.normalize_3d(commands)
        # Extrude values should be scaled to fit normalization range
        extrude_cmd = [c for c in normalized if c["type"] == CommandType.EXTRUDE][0]
        assert all(abs(p) <= tokenizer._norm_range for p in extrude_cmd["params"])

    def test_normalize_3d_includes_sol_offsets(self):
        """Test that SOL z-offset is included in 3D normalization."""
        tokenizer = CommandSequenceTokenizer()
        commands = [
            {"type": CommandType.SOL, "params": [5.0, 0.0]},  # z-offset of 5
            {"type": CommandType.LINE, "params": [0.0, 0.0, 1.0, 0.0]},
            {"type": CommandType.EXTRUDE, "params": [2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]},
        ]
        normalized = tokenizer.normalize_3d(commands)
        sol_cmd = [c for c in normalized if c["type"] == CommandType.SOL][0]
        # SOL z-offset should also be scaled
        assert abs(sol_cmd["params"][0]) <= tokenizer._norm_range

    def test_normalize_3d_no_extrude(self):
        """Test 3D normalization with no extrude commands."""
        tokenizer = CommandSequenceTokenizer()
        commands = [
            {"type": CommandType.LINE, "params": [0.0, 0.0, 1.0, 0.0]},
        ]
        normalized = tokenizer.normalize_3d(commands)
        # Should be unchanged
        assert normalized[0]["params"] == [0.0, 0.0, 1.0, 0.0]


class TestQuantizeParameters:
    """Tests for quantize_parameters() method."""

    def test_quantize_parameters_basic(self):
        """Test basic parameter quantization."""
        tokenizer = CommandSequenceTokenizer()
        commands = [
            {"type": CommandType.LINE, "params": [0.0, 0.0, 1.0, 0.0]},
        ]
        tokens = tokenizer.quantize_parameters(commands)
        assert len(tokens) == 1
        assert tokens[0].command_type == CommandType.LINE
        # Parameters should be quantized integers
        assert all(isinstance(p, int) for p in tokens[0].parameters)

    def test_quantize_dequantize_roundtrip(self):
        """Test that quantize-dequantize roundtrip is close to original."""
        tokenizer = CommandSequenceTokenizer()
        commands = [
            {"type": CommandType.LINE, "params": [0.5, -0.5, 0.8, -0.2]},
        ]
        tokens = tokenizer.quantize_parameters(commands)
        dequantized = tokenizer.dequantize_parameters(tokens)

        # Values should be close to original (within quantization error)
        for i, (orig, deq) in enumerate(zip(commands[0]["params"], dequantized[0]["params"])):
            if tokens[0].parameter_mask[i]:
                assert abs(orig - deq) < 0.02  # ~1% error for 8-bit


class TestPadOrTruncate:
    """Tests for pad_or_truncate() method."""

    def test_pad_short_sequence(self):
        """Test padding of short sequence."""
        config = CommandTokenizationConfig(max_sequence_length=10, pad_to_max_length=True)
        tokenizer = CommandSequenceTokenizer(command_config=config)
        tokens = [
            CommandToken(command_type=CommandType.LINE, parameters=[0]*16, parameter_mask=[True]*4 + [False]*12)
        ]
        padded = tokenizer.pad_or_truncate(tokens)
        assert len(padded) == 10
        # Check that padding tokens are EOS
        assert all(t.command_type == CommandType.EOS for t in padded[1:])

    def test_truncate_long_sequence(self):
        """Test truncation of long sequence."""
        config = CommandTokenizationConfig(max_sequence_length=5, pad_to_max_length=True)
        tokenizer = CommandSequenceTokenizer(command_config=config)
        tokens = [
            CommandToken(command_type=CommandType.LINE, parameters=[0]*16, parameter_mask=[True]*4 + [False]*12)
            for _ in range(10)
        ]
        truncated = tokenizer.pad_or_truncate(tokens)
        assert len(truncated) == 5

    def test_truncate_preserves_sketch_pair(self):
        """Test that truncation preserves complete sketch-extrude pairs."""
        config = CommandTokenizationConfig(max_sequence_length=5, pad_to_max_length=True)
        tokenizer = CommandSequenceTokenizer(command_config=config)
        tokens = [
            CommandToken(command_type=CommandType.SOL, parameters=[0]*16, parameter_mask=[True,True] + [False]*14),
            CommandToken(command_type=CommandType.LINE, parameters=[0]*16, parameter_mask=[True]*4 + [False]*12),
            CommandToken(command_type=CommandType.LINE, parameters=[0]*16, parameter_mask=[True]*4 + [False]*12),
            CommandToken(command_type=CommandType.EXTRUDE, parameters=[0]*16, parameter_mask=[True]*8 + [False]*8),
            CommandToken(command_type=CommandType.SOL, parameters=[0]*16, parameter_mask=[True,True] + [False]*14),
            CommandToken(command_type=CommandType.LINE, parameters=[0]*16, parameter_mask=[True]*4 + [False]*12),
            CommandToken(command_type=CommandType.LINE, parameters=[0]*16, parameter_mask=[True]*4 + [False]*12),
        ]
        truncated = tokenizer.pad_or_truncate(tokens)
        # Should cut after first EXTRUDE (index 3), then pad to 5
        assert truncated[3].command_type == CommandType.EXTRUDE


class TestAnalyzeRoundtripQuality:
    """Tests for analyze_roundtrip_quality() method."""

    def test_roundtrip_quality_basic(self):
        """Test roundtrip quality analysis."""
        tokenizer = CommandSequenceTokenizer()
        commands = [
            {"type": "Line", "params": [0.0, 0.0, 1.0, 0.0]},
            {"type": "Line", "params": [1.0, 0.0, 1.0, 1.0]},
        ]
        quality = tokenizer.analyze_roundtrip_quality(commands)
        assert "param_mse" in quality
        assert "max_error" in quality
        assert "command_preservation_rate" in quality
        # With same types, preservation should be 1.0
        assert quality["command_preservation_rate"] == 1.0

    def test_roundtrip_quality_empty(self):
        """Test roundtrip quality with empty sequence."""
        tokenizer = CommandSequenceTokenizer()
        quality = tokenizer.analyze_roundtrip_quality([])
        assert quality["param_mse"] == 0.0
        assert quality["max_error"] == 0.0


class TestDequantizeParameters:
    """Tests for dequantize_parameters() method."""

    def test_dequantize_basic(self):
        """Test basic dequantization."""
        tokenizer = CommandSequenceTokenizer()
        tokens = [
            CommandToken(
                command_type=CommandType.LINE,
                parameters=[128, 64, 192, 32] + [0]*12,
                parameter_mask=[True, True, True, True] + [False]*12,
            )
        ]
        dequantized = tokenizer.dequantize_parameters(tokens)
        assert len(dequantized) == 1
        assert dequantized[0]["type"] == CommandType.LINE
        # Values should be floats
        assert all(isinstance(p, float) for p in dequantized[0]["params"])
