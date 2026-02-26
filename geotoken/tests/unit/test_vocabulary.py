"""Tests for CADVocabulary."""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pytest

from geotoken.tokenizer.vocabulary import (
    CADVocabulary,
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    SEP_TOKEN_ID,
    UNK_TOKEN_ID,
    NUM_SPECIAL_TOKENS,
)
from geotoken.tokenizer.token_types import CommandToken, CommandType


class TestCADVocabularyInit:
    """Tests for CADVocabulary initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        vocab = CADVocabulary()
        assert vocab.num_command_types == 6
        assert vocab.num_parameters == 16
        assert vocab.num_levels == 256

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        vocab = CADVocabulary(num_command_types=4, num_parameters=8, num_levels=128)
        assert vocab.num_command_types == 4
        assert vocab.num_parameters == 8
        assert vocab.num_levels == 128


class TestVocabSize:
    """Tests for vocab_size property."""

    def test_vocab_size_calculation(self):
        """Test that vocab_size is calculated correctly with full blocks.

        Total = specials + cmd_block + constraint_block + graph_blocks
        cmd_block = 6 + (6 * 16 * 256) = 24582
        constraint_block = 9 * 60 * 60 = 32400
        graph_blocks = 6 + (48 * 256) + (16 * 256) = 6 + 12288 + 4096 = 16390
        Total = 5 + 24582 + 32400 + 16390 = 73377
        """
        vocab = CADVocabulary(num_command_types=6, num_parameters=16, num_levels=256)
        cmd_block = 6 + (6 * 16 * 256)
        constraint_block = 9 * 60 * 60
        graph_structure = 6
        node_features = 48 * 256
        edge_features = 16 * 256
        expected = (NUM_SPECIAL_TOKENS + cmd_block + constraint_block
                    + graph_structure + node_features + edge_features)
        assert vocab.vocab_size == expected

    def test_vocab_size_small(self):
        """Test vocab_size with smaller values.

        All blocks are computed from their respective parameters.
        """
        vocab = CADVocabulary(
            num_command_types=2, num_parameters=4, num_levels=8,
            num_constraint_types=2, max_constraint_index=10,
            num_graph_structure_types=6,
            node_feature_dim=4, edge_feature_dim=2, graph_feature_levels=8,
        )
        cmd_block = 2 + (2 * 4 * 8)
        constraint_block = 2 * 10 * 10
        graph_structure = 6
        node_features = 4 * 8
        edge_features = 2 * 8
        expected = (NUM_SPECIAL_TOKENS + cmd_block + constraint_block
                    + graph_structure + node_features + edge_features)
        assert vocab.vocab_size == expected


class TestSpecialTokens:
    """Tests for special token IDs."""

    def test_pad_token_id(self):
        """Test PAD token ID."""
        vocab = CADVocabulary()
        assert vocab.pad_token_id == PAD_TOKEN_ID
        assert vocab.pad_token_id == 0

    def test_bos_token_id(self):
        """Test BOS token ID."""
        vocab = CADVocabulary()
        assert vocab.bos_token_id == BOS_TOKEN_ID
        assert vocab.bos_token_id == 1

    def test_eos_token_id(self):
        """Test EOS token ID."""
        vocab = CADVocabulary()
        assert vocab.eos_token_id == EOS_TOKEN_ID
        assert vocab.eos_token_id == 2

    def test_sep_token_id(self):
        """Test SEP token ID."""
        vocab = CADVocabulary()
        assert vocab.sep_token_id == SEP_TOKEN_ID
        assert vocab.sep_token_id == 3

    def test_unk_token_id(self):
        """Test UNK token ID."""
        vocab = CADVocabulary()
        assert vocab.unk_token_id == UNK_TOKEN_ID
        assert vocab.unk_token_id == 4


class TestEncode:
    """Tests for encode() method."""

    def test_encode_empty_list(self):
        """Test encoding empty token list."""
        vocab = CADVocabulary()
        ids = vocab.encode([])
        # Should have BOS and EOS
        assert ids == [BOS_TOKEN_ID, EOS_TOKEN_ID]

    def test_encode_invalid_type_raises(self):
        """Test that invalid input type raises TypeError."""
        vocab = CADVocabulary()
        with pytest.raises(TypeError, match="must be a list"):
            vocab.encode("invalid")

    def test_encode_invalid_item_raises(self):
        """Test that non-CommandToken item raises TypeError."""
        vocab = CADVocabulary()
        with pytest.raises(TypeError, match="must be CommandToken"):
            vocab.encode([{"type": "LINE"}])

    def test_encode_single_command(self):
        """Test encoding single command."""
        vocab = CADVocabulary()
        tokens = [
            CommandToken(
                command_type=CommandType.LINE,
                parameters=[100, 50, 200, 150] + [0]*12,
                parameter_mask=[True, True, True, True] + [False]*12,
            )
        ]
        ids = vocab.encode(tokens)
        assert ids[0] == BOS_TOKEN_ID
        assert ids[-1] == EOS_TOKEN_ID
        # Should have command type token + 4 parameter tokens + BOS + EOS
        assert len(ids) >= 6

    def test_encode_multiple_commands(self):
        """Test encoding multiple commands."""
        vocab = CADVocabulary()
        tokens = [
            CommandToken(
                command_type=CommandType.LINE,
                parameters=[100, 50, 200, 150] + [0]*12,
                parameter_mask=[True, True, True, True] + [False]*12,
            ),
            CommandToken(
                command_type=CommandType.CIRCLE,
                parameters=[128, 128, 64] + [0]*13,
                parameter_mask=[True, True, True] + [False]*13,
            ),
        ]
        ids = vocab.encode(tokens)
        assert ids[0] == BOS_TOKEN_ID
        assert ids[-1] == EOS_TOKEN_ID

    def test_encode_stops_at_eos_command(self):
        """Test that encoding stops at EOS command type."""
        vocab = CADVocabulary()
        tokens = [
            CommandToken(
                command_type=CommandType.LINE,
                parameters=[100, 50, 200, 150] + [0]*12,
                parameter_mask=[True, True, True, True] + [False]*12,
            ),
            CommandToken(
                command_type=CommandType.EOS,
                parameters=[0]*16,
                parameter_mask=[False]*16,
            ),
            CommandToken(
                command_type=CommandType.CIRCLE,  # Should not be encoded
                parameters=[128, 128, 64] + [0]*13,
                parameter_mask=[True, True, True] + [False]*13,
            ),
        ]
        ids = vocab.encode(tokens)
        # The CIRCLE after EOS should not be encoded
        # Count command type tokens (after special tokens range)
        cmd_type_tokens = [i for i in ids if NUM_SPECIAL_TOKENS <= i < NUM_SPECIAL_TOKENS + 6]
        assert len(cmd_type_tokens) == 1  # Only LINE


class TestDecode:
    """Tests for decode() method."""

    def test_decode_empty_sequence(self):
        """Test decoding empty sequence."""
        vocab = CADVocabulary()
        commands = vocab.decode([])
        assert commands == []

    def test_decode_bos_eos_only(self):
        """Test decoding BOS/EOS only."""
        vocab = CADVocabulary()
        commands = vocab.decode([BOS_TOKEN_ID, EOS_TOKEN_ID])
        assert commands == []

    def test_encode_decode_roundtrip(self):
        """Test that encode-decode roundtrip preserves commands."""
        vocab = CADVocabulary()
        original_tokens = [
            CommandToken(
                command_type=CommandType.LINE,
                parameters=[100, 50, 200, 150] + [0]*12,
                parameter_mask=[True, True, True, True] + [False]*12,
            ),
        ]
        ids = vocab.encode(original_tokens)
        decoded = vocab.decode(ids)

        assert len(decoded) == 1
        assert decoded[0].command_type == CommandType.LINE
        assert decoded[0].parameters[:4] == [100, 50, 200, 150]

    def test_decode_with_misaligned_tokens_logs_warning(self, caplog):
        """Test that misaligned tokens produce warning."""
        vocab = CADVocabulary()
        # Create a sequence with a valid command type followed by
        # a parameter token that doesn't match the expected type/param
        cmd_type_token = NUM_SPECIAL_TOKENS + 1  # LINE type
        # Create a misaligned param token (for different type)
        misaligned_param = vocab._param_token_id(type_idx=0, param_idx=0, value=100)  # SOL param

        with caplog.at_level(logging.WARNING):
            commands = vocab.decode([BOS_TOKEN_ID, cmd_type_token, misaligned_param, EOS_TOKEN_ID])

        # Should log warning about misalignment
        assert "Decode misalignment" in caplog.text


class TestEncodeFlat:
    """Tests for encode_flat() method."""

    def test_encode_flat_same_as_encode(self):
        """Test that encode_flat produces same result as encode."""
        vocab = CADVocabulary()
        tokens = [
            CommandToken(
                command_type=CommandType.LINE,
                parameters=[100, 50, 200, 150] + [0]*12,
                parameter_mask=[True, True, True, True] + [False]*12,
            ),
        ]
        flat_ids = vocab.encode_flat(tokens)
        regular_ids = vocab.encode(tokens)
        assert flat_ids == regular_ids


class TestSaveLoad:
    """Tests for save/load methods."""

    def test_save_creates_file(self):
        """Test that save creates a JSON file."""
        vocab = CADVocabulary(num_command_types=4, num_parameters=8, num_levels=64)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            vocab.save(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["num_command_types"] == 4
            assert data["num_parameters"] == 8
            assert data["num_levels"] == 64
        finally:
            path.unlink()

    def test_load_restores_vocab(self):
        """Test that load restores vocabulary correctly."""
        vocab = CADVocabulary(num_command_types=4, num_parameters=8, num_levels=64)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            vocab.save(path)
            loaded = CADVocabulary.load(path)
            assert loaded.num_command_types == 4
            assert loaded.num_parameters == 8
            assert loaded.num_levels == 64
            assert loaded.vocab_size == vocab.vocab_size
        finally:
            path.unlink()

    def test_save_load_roundtrip(self):
        """Test complete save/load roundtrip preserves encoding."""
        vocab = CADVocabulary()
        tokens = [
            CommandToken(
                command_type=CommandType.LINE,
                parameters=[100, 50, 200, 150] + [0]*12,
                parameter_mask=[True, True, True, True] + [False]*12,
            ),
        ]
        original_ids = vocab.encode(tokens)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            vocab.save(path)
            loaded = CADVocabulary.load(path)
            loaded_ids = loaded.encode(tokens)
            assert original_ids == loaded_ids
        finally:
            path.unlink()


class TestParamTokenId:
    """Tests for _param_token_id() method."""

    def test_param_token_id_basic(self):
        """Test parameter token ID computation."""
        vocab = CADVocabulary(num_command_types=6, num_parameters=16, num_levels=256)
        # First parameter of first type
        token_id = vocab._param_token_id(type_idx=0, param_idx=0, value=0)
        # Should be offset + 0
        assert token_id == vocab._param_offset

    def test_param_token_id_clamps_value(self):
        """Test that values are clamped to valid range."""
        vocab = CADVocabulary(num_levels=256)
        # Test value > max
        token_id_high = vocab._param_token_id(type_idx=0, param_idx=0, value=1000)
        token_id_max = vocab._param_token_id(type_idx=0, param_idx=0, value=255)
        assert token_id_high == token_id_max

        # Test value < 0
        token_id_low = vocab._param_token_id(type_idx=0, param_idx=0, value=-100)
        token_id_zero = vocab._param_token_id(type_idx=0, param_idx=0, value=0)
        assert token_id_low == token_id_zero


class TestDecodeParamToken:
    """Tests for _decode_param_token() method."""

    def test_decode_param_token_roundtrip(self):
        """Test encode-decode roundtrip for param tokens."""
        vocab = CADVocabulary()
        type_idx = 1
        param_idx = 2
        value = 128
        token_id = vocab._param_token_id(type_idx, param_idx, value)
        decoded = vocab._decode_param_token(token_id, type_idx, param_idx)
        assert decoded == value

    def test_decode_param_token_wrong_type(self, caplog):
        """Test decoding with wrong expected type returns None and logs."""
        vocab = CADVocabulary()
        token_id = vocab._param_token_id(type_idx=0, param_idx=0, value=100)
        with caplog.at_level(logging.WARNING):
            result = vocab._decode_param_token(token_id, expected_type=1, expected_param=0)
        assert result is None
        assert "Decode misalignment" in caplog.text

    def test_decode_param_token_below_offset(self):
        """Test decoding token below param offset returns None."""
        vocab = CADVocabulary()
        result = vocab._decode_param_token(token_id=0, expected_type=0, expected_param=0)
        assert result is None
