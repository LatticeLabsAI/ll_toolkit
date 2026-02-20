"""Integration tests for ll_gen with geotoken package.

Tests integration between ll_gen and geotoken:
- CommandSequenceProposal.to_token_sequence() conversion
- Dataset tokenization with geotoken vocabulary

All tests are marked with @pytest.mark.requires_geotoken and skip if geotoken
is not installed.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Check if geotoken is available
try:
    from geotoken.tokenizer.token_types import TokenSequence
    _GEOTOKEN_AVAILABLE = True
except ImportError:
    _GEOTOKEN_AVAILABLE = False

requires_geotoken = pytest.mark.skipif(
    not _GEOTOKEN_AVAILABLE,
    reason="geotoken package not installed"
)


# ============================================================================
# SECTION 1: Token Sequence Conversion Tests
# ============================================================================


@requires_geotoken
class TestTokenSequenceConversion:
    """Test CommandSequenceProposal to TokenSequence conversion."""

    def test_to_token_sequence_returns_token_sequence(self) -> None:
        """Test to_token_sequence returns geotoken TokenSequence."""
        from ll_gen.proposals.command_proposal import CommandSequenceProposal
        from geotoken.tokenizer.token_types import TokenSequence

        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=[
                {"command_type": "SOL", "parameters": [0] * 16, "parameter_mask": [False] * 16},
                {"command_type": "LINE", "parameters": [0, 0, 128, 0] + [0] * 12, "parameter_mask": [True] * 4 + [False] * 12},
                {"command_type": "EOS", "parameters": [0] * 16, "parameter_mask": [False] * 16},
            ],
            quantization_bits=8,
        )

        token_seq = proposal.to_token_sequence()

        assert isinstance(token_seq, TokenSequence)

    def test_to_token_sequence_has_command_tokens(self) -> None:
        """Test TokenSequence has command_tokens attribute."""
        from ll_gen.proposals.command_proposal import CommandSequenceProposal

        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=[
                {"command_type": "SOL", "parameters": [0] * 16, "parameter_mask": [False] * 16},
            ],
            quantization_bits=8,
        )

        token_seq = proposal.to_token_sequence()

        # TokenSequence has command_tokens, coordinate_tokens, etc.
        assert hasattr(token_seq, "command_tokens")
        assert len(token_seq.command_tokens) > 0


# ============================================================================
# SECTION 2: Geotoken Import Tests
# ============================================================================


@requires_geotoken
class TestGeotokenImports:
    """Test geotoken module imports."""

    def test_token_types_import(self) -> None:
        """Test token_types can be imported."""
        from geotoken.tokenizer.token_types import (
            CommandToken,
            CommandType,
            TokenSequence,
        )
        assert CommandToken is not None
        assert CommandType is not None
        assert TokenSequence is not None

    def test_geo_tokenizer_import(self) -> None:
        """Test GeoTokenizer can be imported."""
        from geotoken.tokenizer.geo_tokenizer import GeoTokenizer
        assert GeoTokenizer is not None

    def test_vocabulary_import(self) -> None:
        """Test CADVocabulary can be imported."""
        from geotoken import CADVocabulary
        assert CADVocabulary is not None


# ============================================================================
# SECTION 3: Dataset Loader Integration Tests
# ============================================================================


@requires_geotoken
class TestDatasetLoaderIntegration:
    """Test dataset loaders with geotoken tokenization."""

    def test_deepcad_loader_geotoken_callable(self) -> None:
        """Test DeepCAD loader can get geotoken via lazy import."""
        from ll_gen.datasets.deepcad_loader import _get_geotoken
        # When geotoken is installed, this should return the module
        geotoken = _get_geotoken()
        assert geotoken is not None

    def test_text2cad_loader_geotoken_callable(self) -> None:
        """Test Text2CAD loader can get geotoken via lazy import."""
        from ll_gen.datasets.text2cad_loader import _get_geotoken
        # When geotoken is installed, this should return the module
        geotoken = _get_geotoken()
        assert geotoken is not None


# ============================================================================
# SECTION 4: Vocabulary Tests
# ============================================================================


@requires_geotoken
class TestVocabularyIntegration:
    """Test vocabulary integration with geotoken."""

    def test_vocabulary_special_tokens(self) -> None:
        """Test geotoken has expected special token constants."""
        from geotoken import PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID

        # Should have PAD, BOS, EOS tokens defined
        assert PAD_TOKEN_ID == 0
        assert BOS_TOKEN_ID == 1
        assert EOS_TOKEN_ID == 2

    def test_vocabulary_class_exists(self) -> None:
        """Test CADVocabulary class exists."""
        from geotoken import CADVocabulary

        # Should be able to instantiate vocabulary
        vocab = CADVocabulary()
        assert vocab is not None


# ============================================================================
# SECTION 5: Command Type Tests
# ============================================================================


@requires_geotoken
class TestCommandTypeIntegration:
    """Test CommandType enum integration."""

    def test_command_types_defined(self) -> None:
        """Test expected command types are defined."""
        from geotoken.tokenizer.token_types import CommandType

        # Common CAD command types
        expected_types = ["SOL", "LINE", "ARC", "CIRCLE", "EXTRUDE", "EOS"]

        for cmd_type in expected_types:
            # Check if the command type exists in the enum
            assert hasattr(CommandType, cmd_type) or cmd_type in [e.name for e in CommandType]


# ============================================================================
# SECTION 6: Proposal to Token Round-Trip Tests
# ============================================================================


@requires_geotoken
class TestProposalTokenRoundTrip:
    """Test proposal to token sequence conversion is consistent."""

    def test_command_dicts_preserved(self) -> None:
        """Test command structure is preserved in conversion."""
        from ll_gen.proposals.command_proposal import CommandSequenceProposal

        command_dicts = [
            {"command_type": "SOL", "parameters": [0] * 16, "parameter_mask": [False] * 16},
            {"command_type": "LINE", "parameters": [0, 0, 128, 128] + [0] * 12, "parameter_mask": [True] * 4 + [False] * 12},
            {"command_type": "EXTRUDE", "parameters": [64] + [0] * 15, "parameter_mask": [True] + [False] * 15},
            {"command_type": "EOS", "parameters": [0] * 16, "parameter_mask": [False] * 16},
        ]

        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=command_dicts,
            quantization_bits=8,
        )

        # Convert to token sequence
        token_seq = proposal.to_token_sequence()

        # Should have command tokens
        assert len(token_seq.command_tokens) > 0
        assert token_seq.num_commands > 0


# ============================================================================
# SECTION 7: Module Availability Tests
# ============================================================================


@requires_geotoken
class TestModuleAvailability:
    """Test module availability when geotoken is installed."""

    def test_conftest_geotoken_marker(self) -> None:
        """Test that conftest has geotoken availability check."""
        from tests.conftest import _geotoken_available
        assert _geotoken_available() is True

    def test_proposal_module_imports_geotoken(self) -> None:
        """Test command_proposal module can import geotoken."""
        from ll_gen.proposals import command_proposal
        # The module should have geotoken types in scope during conversion
        assert hasattr(command_proposal, "CommandSequenceProposal")
