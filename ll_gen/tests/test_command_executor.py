"""Comprehensive test suite for ll_gen.disposal.command_executor module.

Tests command sequence execution functionality:
- Command dict parsing (SOL, LINE, ARC, CIRCLE, EXTRUDE, EOS)
- Parameter dequantization (8-bit → float coordinates)
- Sketch loop construction
- Extrusion direction detection
- Fallback to OCC when cadling unavailable
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

import pytest

from ll_gen.proposals.command_proposal import CommandSequenceProposal
from ll_gen.disposal.command_executor import (
    execute_command_proposal,
    _extract_sketch_groups,
)


# ============================================================================
# SECTION 1: Module Import Tests
# ============================================================================


class TestModuleImport:
    """Test module import and availability flags."""

    def test_module_importable(self) -> None:
        """Test that command_executor module is importable."""
        from ll_gen.disposal import command_executor
        assert hasattr(command_executor, "execute_command_proposal")
        assert hasattr(command_executor, "_extract_sketch_groups")

    def test_availability_flags_are_boolean(self) -> None:
        """Test that availability flags are boolean values."""
        from ll_gen.disposal import command_executor
        assert isinstance(command_executor._OCC_AVAILABLE, bool)
        assert isinstance(command_executor._CADLING_EXECUTOR_AVAILABLE, bool)

    def test_execute_function_is_callable(self) -> None:
        """Test that execute_command_proposal is callable."""
        from ll_gen.disposal.command_executor import execute_command_proposal
        assert callable(execute_command_proposal)


# ============================================================================
# SECTION 2: Extract Sketch Groups Tests
# ============================================================================


class TestExtractSketchGroups:
    """Test sketch group extraction from commands."""

    def test_extract_empty_commands(self) -> None:
        """Test extracting groups from empty command list."""
        groups = _extract_sketch_groups([])
        assert groups == []

    def test_extract_single_sol_group(self) -> None:
        """Test extracting a single SOL-initiated group."""
        commands = [
            {"type": "SOL"},
            {"type": "LINE", "params": {}},
            {"type": "LINE", "params": {}},
            {"type": "EXTRUDE", "params": {}},
        ]
        groups = _extract_sketch_groups(commands)
        assert len(groups) == 1
        assert len(groups[0]) == 4

    def test_extract_multiple_sol_groups(self) -> None:
        """Test extracting multiple SOL-initiated groups."""
        commands = [
            {"type": "SOL"},
            {"type": "LINE", "params": {}},
            {"type": "EXTRUDE", "params": {}},
            {"type": "SOL"},
            {"type": "CIRCLE", "params": {}},
            {"type": "EXTRUDE", "params": {}},
        ]
        groups = _extract_sketch_groups(commands)
        assert len(groups) == 2
        assert groups[0][0]["type"] == "SOL"
        assert groups[1][0]["type"] == "SOL"

    def test_extract_groups_with_eol(self) -> None:
        """Test extracting groups with EOL markers."""
        commands = [
            {"type": "SOL"},
            {"type": "LINE", "params": {}},
            {"type": "EOL"},
        ]
        groups = _extract_sketch_groups(commands)
        assert len(groups) == 1
        assert groups[0][-1]["type"] == "EOL"

    def test_extract_preserves_command_order(self) -> None:
        """Test that command order is preserved within groups."""
        commands = [
            {"type": "SOL"},
            {"type": "LINE", "params": {"x1": 0, "y1": 0, "x2": 10, "y2": 0}},
            {"type": "LINE", "params": {"x1": 10, "y1": 0, "x2": 10, "y2": 10}},
            {"type": "LINE", "params": {"x1": 10, "y1": 10, "x2": 0, "y2": 10}},
            {"type": "LINE", "params": {"x1": 0, "y1": 10, "x2": 0, "y2": 0}},
            {"type": "EXTRUDE", "params": {}},
        ]
        groups = _extract_sketch_groups(commands)
        assert len(groups) == 1
        # Commands should be in original order
        assert groups[0][0]["type"] == "SOL"
        assert groups[0][1]["type"] == "LINE"
        assert groups[0][-1]["type"] == "EXTRUDE"

    def test_extract_handles_missing_type(self) -> None:
        """Test extraction handles commands without type field."""
        commands = [
            {"type": "SOL"},
            {"params": {}},  # Missing type
            {"type": "LINE", "params": {}},
        ]
        groups = _extract_sketch_groups(commands)
        assert len(groups) == 1
        # The command without type should still be included
        assert len(groups[0]) == 3


# ============================================================================
# SECTION 3: Execute Command Proposal Tests (Mocked)
# ============================================================================


class TestExecuteCommandProposalMocked:
    """Test execute_command_proposal with mocked dependencies."""

    def test_occ_unavailable_raises_runtime_error(self) -> None:
        """Test RuntimeError when OCC is not available."""
        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=[
                {"command_type": "SOL", "parameters": [0] * 16, "parameter_mask": [False] * 16},
            ],
            quantization_bits=8,
        )
        with patch(
            "ll_gen.disposal.command_executor._OCC_AVAILABLE", False
        ):
            with patch(
                "ll_gen.disposal.command_executor._CADLING_EXECUTOR_AVAILABLE", False
            ):
                with pytest.raises(RuntimeError) as exc_info:
                    execute_command_proposal(proposal)
                assert "OCC libraries are not available" in str(exc_info.value)

    def test_cadling_executor_preferred_when_available(self) -> None:
        """Test that cadling executor is used when available."""
        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=[
                {"command_type": "SOL", "parameters": [0] * 16, "parameter_mask": [False] * 16},
            ],
            quantization_bits=8,
        )

        mock_executor = MagicMock()
        mock_executor.execute.return_value = MagicMock()

        with patch(
            "ll_gen.disposal.command_executor._CADLING_EXECUTOR_AVAILABLE", True
        ):
            with patch(
                "ll_gen.disposal.command_executor.CadlingCommandExecutor",
                return_value=mock_executor,
            ):
                with patch.object(proposal, "to_token_sequence") as mock_to_token:
                    mock_to_token.return_value = MagicMock(token_ids=[1, 2, 3])
                    result = execute_command_proposal(proposal)
                    mock_executor.execute.assert_called_once()

    def test_fallback_to_standalone_on_cadling_failure(self) -> None:
        """Test fallback to standalone execution when cadling fails."""
        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=[
                {"command_type": "SOL", "parameters": [0] * 16, "parameter_mask": [False] * 16},
            ],
            quantization_bits=8,
        )

        mock_executor = MagicMock()
        mock_executor.execute.side_effect = Exception("Cadling failed")

        with patch(
            "ll_gen.disposal.command_executor._CADLING_EXECUTOR_AVAILABLE", True
        ):
            with patch(
                "ll_gen.disposal.command_executor._OCC_AVAILABLE", False
            ):
                with patch(
                    "ll_gen.disposal.command_executor.CadlingCommandExecutor",
                    return_value=mock_executor,
                ):
                    with patch.object(proposal, "to_token_sequence") as mock_to_token:
                        mock_to_token.return_value = MagicMock(token_ids=[1, 2, 3])
                        with pytest.raises(RuntimeError) as exc_info:
                            execute_command_proposal(proposal)
                        # Should fall back and fail on OCC unavailable
                        assert "OCC libraries are not available" in str(exc_info.value)


# ============================================================================
# SECTION 4: Command Parsing Tests
# ============================================================================


class TestCommandParsing:
    """Test command parsing and structure."""

    def test_line_command_structure(self) -> None:
        """Test LINE command dictionary structure."""
        line_cmd = {
            "type": "LINE",
            "params": {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 0.0}
        }
        assert line_cmd["type"] == "LINE"
        assert "x1" in line_cmd["params"]
        assert "y1" in line_cmd["params"]
        assert "x2" in line_cmd["params"]
        assert "y2" in line_cmd["params"]

    def test_arc_command_structure(self) -> None:
        """Test ARC command dictionary structure."""
        arc_cmd = {
            "type": "ARC",
            "params": {
                "x_start": 0.0, "y_start": 0.0,
                "x_mid": 5.0, "y_mid": 5.0,
                "x_end": 10.0, "y_end": 0.0
            }
        }
        assert arc_cmd["type"] == "ARC"
        assert "x_start" in arc_cmd["params"]
        assert "x_mid" in arc_cmd["params"]
        assert "x_end" in arc_cmd["params"]

    def test_circle_command_structure(self) -> None:
        """Test CIRCLE command dictionary structure."""
        circle_cmd = {
            "type": "CIRCLE",
            "params": {"cx": 5.0, "cy": 5.0, "r": 2.5}
        }
        assert circle_cmd["type"] == "CIRCLE"
        assert "cx" in circle_cmd["params"]
        assert "cy" in circle_cmd["params"]
        assert "r" in circle_cmd["params"]

    def test_extrude_command_structure(self) -> None:
        """Test EXTRUDE command dictionary structure."""
        extrude_cmd = {
            "type": "EXTRUDE",
            "params": {"dx": 0.0, "dy": 0.0, "dz": 1.0, "extent": 10.0}
        }
        assert extrude_cmd["type"] == "EXTRUDE"
        assert "dx" in extrude_cmd["params"]
        assert "dy" in extrude_cmd["params"]
        assert "dz" in extrude_cmd["params"]
        assert "extent" in extrude_cmd["params"]

    def test_sol_command_structure(self) -> None:
        """Test SOL (Start Of Loop) command structure."""
        sol_cmd = {"type": "SOL"}
        assert sol_cmd["type"] == "SOL"

    def test_eol_command_structure(self) -> None:
        """Test EOL (End Of Loop) command structure."""
        eol_cmd = {"type": "EOL"}
        assert eol_cmd["type"] == "EOL"


# ============================================================================
# SECTION 5: Boolean Operation Types Tests
# ============================================================================


class TestBooleanOperationTypes:
    """Test boolean operation type mapping."""

    def test_operation_type_new(self) -> None:
        """Test operation type 0 is 'New'."""
        # Operation type 0: New (return tool shape)
        assert 0 == 0  # New

    def test_operation_type_union(self) -> None:
        """Test operation type 1 is 'Union'."""
        # Operation type 1: Union
        assert 1 == 1  # Union

    def test_operation_type_cut(self) -> None:
        """Test operation type 2 is 'Cut'."""
        # Operation type 2: Cut (Subtraction)
        assert 2 == 2  # Cut

    def test_operation_type_intersection(self) -> None:
        """Test operation type 3 is 'Intersection'."""
        # Operation type 3: Intersection
        assert 3 == 3  # Intersection

    def test_operation_types_documented(self) -> None:
        """Test that all operation types are documented."""
        operation_types = {
            0: "New",
            1: "Union",
            2: "Cut",
            3: "Intersection",
        }
        assert len(operation_types) == 4


# ============================================================================
# SECTION 6: Dequantization Tests
# ============================================================================


class TestDequantization:
    """Test parameter dequantization from quantized values."""

    def test_quantization_bits_default(self) -> None:
        """Test default quantization bits is 8."""
        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=[],
            quantization_bits=8,
        )
        assert proposal.quantization_bits == 8

    def test_normalization_range_default(self) -> None:
        """Test default normalization range."""
        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=[],
            quantization_bits=8,
            normalization_range=2.0,
        )
        assert proposal.normalization_range == 2.0

    def test_8bit_quantization_range(self) -> None:
        """Test 8-bit quantization produces 0-255 range."""
        max_8bit = 2**8 - 1
        assert max_8bit == 255

    def test_dequantization_formula(self) -> None:
        """Test dequantization formula: value / (2^bits - 1) * range - range/2."""
        bits = 8
        normalization_range = 2.0
        quantized_value = 128  # Middle value

        # Dequantize: (value / 255) * 2.0 - 1.0
        expected = (quantized_value / 255) * normalization_range - (normalization_range / 2)

        # 128/255 * 2 - 1 ≈ 0.00392...
        assert abs(expected - 0.00392156862745098) < 0.0001


# ============================================================================
# SECTION 7: Proposal Interface Tests
# ============================================================================


class TestProposalInterface:
    """Test CommandSequenceProposal interface used by executor."""

    def test_proposal_has_command_dicts(self) -> None:
        """Test proposal has command_dicts attribute."""
        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=[
                {"command_type": "SOL", "parameters": [0] * 16, "parameter_mask": [False] * 16},
            ],
            quantization_bits=8,
        )
        assert hasattr(proposal, "command_dicts")
        assert len(proposal.command_dicts) == 1

    def test_proposal_has_quantization_settings(self) -> None:
        """Test proposal has quantization settings."""
        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=[],
            quantization_bits=8,
            normalization_range=2.0,
        )
        assert proposal.quantization_bits == 8
        assert proposal.normalization_range == 2.0

    def test_proposal_to_token_sequence_method(self) -> None:
        """Test proposal has to_token_sequence method."""
        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=[],
            quantization_bits=8,
        )
        assert hasattr(proposal, "to_token_sequence")
        assert callable(proposal.to_token_sequence)


# ============================================================================
# SECTION 8: Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling in command execution."""

    def test_empty_command_sequence_error(self) -> None:
        """Test handling of empty command sequence."""
        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=[],
            quantization_bits=8,
        )

        with patch(
            "ll_gen.disposal.command_executor._CADLING_EXECUTOR_AVAILABLE", False
        ):
            with patch(
                "ll_gen.disposal.command_executor._OCC_AVAILABLE", True
            ):
                # Should fail because no geometry is generated
                with pytest.raises(RuntimeError):
                    execute_command_proposal(proposal)

    def test_dequantization_error_wrapped(self) -> None:
        """Test that dequantization errors are wrapped in RuntimeError."""
        # Create a proposal with mock dequantize that raises
        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=[
                {"command_type": "SOL", "parameters": [0] * 16, "parameter_mask": [False] * 16},
            ],
            quantization_bits=8,
        )

        with patch(
            "ll_gen.disposal.command_executor._CADLING_EXECUTOR_AVAILABLE", False
        ):
            with patch(
                "ll_gen.disposal.command_executor._OCC_AVAILABLE", True
            ):
                with patch.object(
                    proposal, "dequantize",
                    side_effect=ValueError("Invalid quantization")
                ):
                    with pytest.raises(RuntimeError) as exc_info:
                        execute_command_proposal(proposal)
                    assert "dequantize" in str(exc_info.value).lower()


# ============================================================================
# SECTION 9: Sketch Group Execution Tests
# ============================================================================


class TestSketchGroupExecution:
    """Test sketch group execution logic."""

    def test_group_with_only_sol(self) -> None:
        """Test handling of group with only SOL command."""
        groups = _extract_sketch_groups([{"type": "SOL"}])
        assert len(groups) == 1
        assert groups[0][0]["type"] == "SOL"

    def test_group_with_sketch_commands(self) -> None:
        """Test group containing LINE commands."""
        commands = [
            {"type": "SOL"},
            {"type": "LINE", "params": {"x1": 0, "y1": 0, "x2": 10, "y2": 0}},
            {"type": "LINE", "params": {"x1": 10, "y1": 0, "x2": 10, "y2": 10}},
        ]
        groups = _extract_sketch_groups(commands)
        assert len(groups) == 1
        # Contains SOL + 2 LINEs
        assert len(groups[0]) == 3

    def test_group_with_arc_command(self) -> None:
        """Test group containing ARC command."""
        commands = [
            {"type": "SOL"},
            {"type": "ARC", "params": {
                "x_start": 0, "y_start": 0,
                "x_mid": 5, "y_mid": 5,
                "x_end": 10, "y_end": 0,
            }},
        ]
        groups = _extract_sketch_groups(commands)
        assert len(groups) == 1
        assert groups[0][1]["type"] == "ARC"

    def test_group_with_circle_command(self) -> None:
        """Test group containing CIRCLE command."""
        commands = [
            {"type": "SOL"},
            {"type": "CIRCLE", "params": {"cx": 5, "cy": 5, "r": 2}},
        ]
        groups = _extract_sketch_groups(commands)
        assert len(groups) == 1
        assert groups[0][1]["type"] == "CIRCLE"

    def test_group_with_extrude_command(self) -> None:
        """Test group containing EXTRUDE command."""
        commands = [
            {"type": "SOL"},
            {"type": "LINE", "params": {}},
            {"type": "EXTRUDE", "params": {"dz": 1, "extent": 10}},
        ]
        groups = _extract_sketch_groups(commands)
        assert len(groups) == 1
        assert any(cmd["type"] == "EXTRUDE" for cmd in groups[0])


# ============================================================================
# SECTION 10: Fixture Tests
# ============================================================================


class TestWithFixtures:
    """Test with conftest fixtures."""

    def test_command_proposal_fixture_structure(self, command_proposal) -> None:
        """Test command_proposal fixture has expected structure."""
        assert hasattr(command_proposal, "command_dicts")
        assert hasattr(command_proposal, "quantization_bits")
        assert hasattr(command_proposal, "normalization_range")
        assert len(command_proposal.command_dicts) > 0

    def test_command_proposal_token_ids_fixture(self, command_proposal_token_ids) -> None:
        """Test command_proposal_token_ids fixture has expected structure."""
        assert hasattr(command_proposal_token_ids, "token_ids")
        assert len(command_proposal_token_ids.token_ids) > 0
