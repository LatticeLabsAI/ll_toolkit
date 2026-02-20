"""Unit tests for CommandExecutor token execution.

Tests the neural generation path: token IDs → CAD geometry.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestCommandExecutor:
    """Tests for CommandExecutor class."""

    def test_execute_returns_dict(self):
        """Test execute() returns result dictionary with expected keys."""
        from cadling.generation.reconstruction.command_executor import CommandExecutor

        executor = CommandExecutor(tolerance=1e-6)

        # Empty token list should return invalid result
        result = executor.execute([])

        assert isinstance(result, dict)
        assert "valid" in result
        assert "errors" in result

    def test_execute_tokens_wrapper_exists(self):
        """Test execute_tokens() method exists with correct signature."""
        from cadling.generation.reconstruction.command_executor import CommandExecutor

        executor = CommandExecutor()

        # Check method exists
        assert hasattr(executor, "execute_tokens")
        assert callable(executor.execute_tokens)

    @patch.object(
        __import__("cadling.generation.reconstruction.command_executor", fromlist=["CommandExecutor"]).CommandExecutor,
        "execute",
    )
    @patch.object(
        __import__("cadling.generation.reconstruction.command_executor", fromlist=["CommandExecutor"]).CommandExecutor,
        "export_step",
    )
    def test_execute_tokens_calls_execute(self, mock_export, mock_execute):
        """Test execute_tokens() calls execute() internally."""
        from cadling.generation.reconstruction.command_executor import CommandExecutor

        mock_execute.return_value = {
            "valid": True,
            "shape": MagicMock(),
            "errors": [],
        }
        mock_export.return_value = True

        executor = CommandExecutor()
        result = executor.execute_tokens([1, 2, 3, 4, 5])

        mock_execute.assert_called_once_with([1, 2, 3, 4, 5])

    @patch.object(
        __import__("cadling.generation.reconstruction.command_executor", fromlist=["CommandExecutor"]).CommandExecutor,
        "execute",
    )
    @patch.object(
        __import__("cadling.generation.reconstruction.command_executor", fromlist=["CommandExecutor"]).CommandExecutor,
        "export_step",
    )
    def test_execute_tokens_exports_when_path_given(self, mock_export, mock_execute):
        """Test execute_tokens() exports when output_path provided."""
        from cadling.generation.reconstruction.command_executor import CommandExecutor

        mock_shape = MagicMock()
        mock_execute.return_value = {
            "valid": True,
            "shape": mock_shape,
            "errors": [],
        }
        mock_export.return_value = True

        executor = CommandExecutor()
        executor.execute_tokens([1, 2, 3], output_path="/tmp/output.step")

        mock_export.assert_called_once_with(mock_shape, "/tmp/output.step")

    @patch.object(
        __import__("cadling.generation.reconstruction.command_executor", fromlist=["CommandExecutor"]).CommandExecutor,
        "execute",
    )
    def test_execute_tokens_returns_none_on_invalid(self, mock_execute):
        """Test execute_tokens() returns None when execution fails."""
        from cadling.generation.reconstruction.command_executor import CommandExecutor

        mock_execute.return_value = {
            "valid": False,
            "shape": None,
            "errors": ["Token decode error"],
        }

        executor = CommandExecutor()
        result = executor.execute_tokens([1, 2, 3])

        assert result is None

    @patch.object(
        __import__("cadling.generation.reconstruction.command_executor", fromlist=["CommandExecutor"]).CommandExecutor,
        "execute",
    )
    def test_execute_tokens_returns_shape_on_valid(self, mock_execute):
        """Test execute_tokens() returns shape when execution succeeds."""
        from cadling.generation.reconstruction.command_executor import CommandExecutor

        mock_shape = MagicMock()
        mock_execute.return_value = {
            "valid": True,
            "shape": mock_shape,
            "errors": [],
        }

        executor = CommandExecutor()
        result = executor.execute_tokens([1, 2, 3])

        assert result == mock_shape

    def test_export_step_exists(self):
        """Test export_step() method exists."""
        from cadling.generation.reconstruction.command_executor import CommandExecutor

        executor = CommandExecutor()

        assert hasattr(executor, "export_step")
        assert callable(executor.export_step)


class TestCommandToken:
    """Tests for CommandToken dataclass."""

    def test_command_token_creation(self):
        """Test CommandToken dataclass can be created."""
        from cadling.generation.reconstruction.command_executor import (
            CommandToken,
            CommandType,
        )

        token = CommandToken(
            command_type=CommandType.LINE,
            raw_token_id=100,
            params=[0.1, 0.2, 0.3, 0.4],
        )

        assert token.command_type == CommandType.LINE
        assert token.raw_token_id == 100
        assert len(token.params) == 4


class TestCommandType:
    """Tests for CommandType enum."""

    def test_command_types_exist(self):
        """Test expected command types are defined."""
        from cadling.generation.reconstruction.command_executor import CommandType

        expected_types = ["SOL", "EOL", "EOS", "LINE", "ARC", "CIRCLE", "EXTRUDE"]

        for type_name in expected_types:
            assert hasattr(CommandType, type_name)
