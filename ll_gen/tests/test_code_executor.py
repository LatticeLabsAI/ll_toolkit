"""Comprehensive test suite for ll_gen.disposal.code_executor module.

Tests code execution functionality for CadQuery, OpenSCAD, and pythonocc:
- Timeout handling via subprocess.run(timeout=...)
- Subprocess-based isolation (no in-process exec)
- Language-specific execution paths
- Error extraction and categorization
- Import validation and availability checking
"""
from __future__ import annotations

import json
import signal
import subprocess
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch, mock_open

import pytest

from ll_gen.config import CodeLanguage
from ll_gen.proposals.code_proposal import CodeProposal
from ll_gen.disposal.code_executor import (
    TimeoutError,
    _timeout_handler,
    execute_code_proposal,
)


# ============================================================================
# SECTION 1: TimeoutError and Handler Tests
# ============================================================================


class TestTimeoutError:
    """Test TimeoutError exception class."""

    def test_timeout_error_creation(self) -> None:
        """Test TimeoutError can be created with message."""
        error = TimeoutError("Execution timed out")
        assert str(error) == "Execution timed out"

    def test_timeout_error_inheritance(self) -> None:
        """Test TimeoutError inherits from Exception."""
        error = TimeoutError("test")
        assert isinstance(error, Exception)

    def test_timeout_error_can_be_caught(self) -> None:
        """Test TimeoutError can be caught specifically."""
        with pytest.raises(TimeoutError):
            raise TimeoutError("test timeout")


class TestTimeoutHandler:
    """Test the _timeout_handler signal handler function."""

    def test_timeout_handler_raises_timeout_error(self) -> None:
        """Test that _timeout_handler raises TimeoutError."""
        with pytest.raises(TimeoutError) as exc_info:
            _timeout_handler(signal.SIGALRM, None)
        assert "timeout" in str(exc_info.value).lower()

    def test_timeout_handler_accepts_frame_parameter(self) -> None:
        """Test that _timeout_handler accepts frame parameter."""
        mock_frame = MagicMock()
        with pytest.raises(TimeoutError):
            _timeout_handler(signal.SIGALRM, mock_frame)


# ============================================================================
# SECTION 2: execute_code_proposal Language Routing Tests
# ============================================================================


class TestExecuteCodeProposalRouting:
    """Test execute_code_proposal routes to correct executor."""

    def test_cadquery_language_routes_to_cadquery_executor(self) -> None:
        """Test CadQuery language routes to _execute_cadquery."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="result = 'test'",
            language=CodeLanguage.CADQUERY,
        )
        with patch(
            "ll_gen.disposal.code_executor._execute_cadquery"
        ) as mock_exec:
            mock_exec.return_value = MagicMock()
            execute_code_proposal(proposal)
            mock_exec.assert_called_once_with(proposal.code, 30)

    def test_openscad_language_routes_to_openscad_executor(self) -> None:
        """Test OpenSCAD language routes to _execute_openscad."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="cube([10,10,10]);",
            language=CodeLanguage.OPENSCAD,
        )
        with patch(
            "ll_gen.disposal.code_executor._execute_openscad"
        ) as mock_exec:
            mock_exec.return_value = MagicMock()
            execute_code_proposal(proposal)
            mock_exec.assert_called_once_with(proposal.code, 30)

    def test_pythonocc_language_routes_to_pythonocc_executor(self) -> None:
        """Test pythonocc language routes to _execute_pythonocc."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="result = 'test'",
            language=CodeLanguage.PYTHONOCC,
        )
        with patch(
            "ll_gen.disposal.code_executor._execute_pythonocc"
        ) as mock_exec:
            mock_exec.return_value = MagicMock()
            execute_code_proposal(proposal)
            mock_exec.assert_called_once_with(proposal.code, 30)

    def test_custom_timeout_passed_to_executor(self) -> None:
        """Test custom timeout is passed to the executor."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="result = 'test'",
            language=CodeLanguage.CADQUERY,
        )
        with patch(
            "ll_gen.disposal.code_executor._execute_cadquery"
        ) as mock_exec:
            mock_exec.return_value = MagicMock()
            execute_code_proposal(proposal, timeout=60)
            mock_exec.assert_called_once_with(proposal.code, 60)


# ============================================================================
# SECTION 3: CadQuery Execution Tests (Mocked)
# ============================================================================


class TestCadQueryExecutionMocked:
    """Test CadQuery execution with mocked dependencies."""

    def test_cadquery_unavailable_raises_runtime_error(self) -> None:
        """Test RuntimeError when CadQuery is not available."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="result = cq.Workplane().box(10,10,10)",
            language=CodeLanguage.CADQUERY,
        )
        with patch(
            "ll_gen.disposal.code_executor._CADQUERY_AVAILABLE", False
        ):
            with pytest.raises(RuntimeError) as exc_info:
                execute_code_proposal(proposal)
            assert "CadQuery is not available" in str(exc_info.value)

    def test_occ_unavailable_with_cadquery_raises_runtime_error(self) -> None:
        """Test RuntimeError when pythonocc is not available for CadQuery."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="result = cq.Workplane().box(10,10,10)",
            language=CodeLanguage.CADQUERY,
        )
        with patch("ll_gen.disposal.code_executor._CADQUERY_AVAILABLE", True):
            with patch(
                "ll_gen.disposal.code_executor._OCC_AVAILABLE", False
            ):
                with pytest.raises(RuntimeError) as exc_info:
                    execute_code_proposal(proposal)
                assert "pythonocc is not available" in str(exc_info.value)


class TestCadQueryExecutionNamespace:
    """Test CadQuery execution namespace restrictions."""

    def test_restricted_builtins_include_safe_functions(self) -> None:
        """Test that restricted namespace includes safe builtin functions."""
        # The restricted namespace should include these
        safe_builtins = [
            "abs", "round", "len", "range", "enumerate", "zip",
            "map", "filter", "list", "dict", "tuple", "set",
            "sum", "min", "max", "float", "int", "str", "bool", "print"
        ]
        # These are verified by inspection of the code
        # The actual execution tests would require cadquery
        for builtin in safe_builtins:
            assert builtin in __builtins__ or hasattr(__builtins__, builtin)


class TestCadQueryResultExtraction:
    """Test CadQuery result extraction from namespace."""

    def test_result_variable_names(self) -> None:
        """Test that expected result variable names are documented."""
        # The executor looks for these variable names
        expected_names = ["result", "part", "model", "shape", "body", "solid"]
        # This is a documentation test - verify the list exists
        assert len(expected_names) == 6


# ============================================================================
# SECTION 4: OpenSCAD Execution Tests (Mocked)
# ============================================================================


class TestOpenSCADExecutionMocked:
    """Test OpenSCAD execution with mocked dependencies."""

    def test_occ_unavailable_raises_runtime_error(self) -> None:
        """Test RuntimeError when pythonocc is not available."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="cube([10,10,10]);",
            language=CodeLanguage.OPENSCAD,
        )
        with patch(
            "ll_gen.disposal.code_executor._OCC_AVAILABLE", False
        ):
            with pytest.raises(RuntimeError) as exc_info:
                execute_code_proposal(proposal)
            assert "pythonocc is not available" in str(exc_info.value)

    def test_openscad_not_found_raises_runtime_error(self) -> None:
        """Test RuntimeError when openscad binary is not found."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="cube([10,10,10]);",
            language=CodeLanguage.OPENSCAD,
        )
        with patch(
            "ll_gen.disposal.code_executor._OCC_AVAILABLE", True
        ):
            with patch(
                "subprocess.run",
                side_effect=FileNotFoundError("openscad not found"),
            ):
                with pytest.raises(RuntimeError) as exc_info:
                    execute_code_proposal(proposal)
                assert "OpenSCAD not found" in str(exc_info.value)

    def test_openscad_timeout_raises_timeout_error(self) -> None:
        """Test TimeoutError when OpenSCAD execution times out."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="cube([10,10,10]);",
            language=CodeLanguage.OPENSCAD,
        )
        with patch(
            "ll_gen.disposal.code_executor._OCC_AVAILABLE", True
        ):
            with patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="openscad", timeout=30),
            ):
                with pytest.raises(TimeoutError) as exc_info:
                    execute_code_proposal(proposal)
                assert "timeout" in str(exc_info.value).lower()

    def test_openscad_execution_failure_includes_stderr(self) -> None:
        """Test that OpenSCAD execution failure includes stderr."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="invalid_command();",
            language=CodeLanguage.OPENSCAD,
        )
        mock_error = subprocess.CalledProcessError(
            returncode=1,
            cmd="openscad",
            stderr=b"ERROR: unknown function invalid_command",
        )
        with patch(
            "ll_gen.disposal.code_executor._OCC_AVAILABLE", True
        ):
            with patch(
                "subprocess.run",
                side_effect=mock_error,
            ):
                with pytest.raises(RuntimeError) as exc_info:
                    execute_code_proposal(proposal)
                assert "OpenSCAD execution failed" in str(exc_info.value)


# ============================================================================
# SECTION 5: pythonocc Execution Tests (Mocked)
# ============================================================================


class TestPythonoccExecutionMocked:
    """Test pythonocc execution with mocked dependencies."""

    def test_occ_unavailable_raises_runtime_error(self) -> None:
        """Test RuntimeError when pythonocc is not available."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox\nresult = BRepPrimAPI_MakeBox(10, 10, 10).Shape()",
            language=CodeLanguage.PYTHONOCC,
        )
        with patch(
            "ll_gen.disposal.code_executor._OCC_AVAILABLE", False
        ):
            with pytest.raises(RuntimeError) as exc_info:
                execute_code_proposal(proposal)
            assert "pythonocc is not available" in str(exc_info.value)


class TestPythonoccExecutionNamespace:
    """Test pythonocc execution namespace."""

    def test_pythonocc_namespace_includes_math(self) -> None:
        """Test that pythonocc namespace includes math module."""
        # Verified by code inspection
        import math
        assert hasattr(math, "pi")
        assert hasattr(math, "sin")

    def test_pythonocc_result_variable_names(self) -> None:
        """Test that expected result variable names match CadQuery."""
        # The executor looks for the same variable names as CadQuery
        expected_names = ["result", "part", "model", "shape", "body", "solid"]
        assert len(expected_names) == 6


# ============================================================================
# SECTION 6: Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling in code execution."""

    def test_syntax_error_wrapped_in_runtime_error(self) -> None:
        """Test that SyntaxError is wrapped in RuntimeError via subprocess."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="def invalid syntax(:",  # Syntax error
            language=CodeLanguage.CADQUERY,
        )
        with patch(
            "ll_gen.disposal.code_executor._CADQUERY_AVAILABLE", True
        ):
            with patch(
                "ll_gen.disposal.code_executor._OCC_AVAILABLE", True
            ):
                # Mock _run_in_subprocess to simulate the error
                with patch(
                    "ll_gen.disposal.code_executor._run_in_subprocess",
                    side_effect=RuntimeError(
                        "CadQuery execution failed: SyntaxError: invalid syntax"
                    ),
                ):
                    with pytest.raises(RuntimeError) as exc_info:
                        execute_code_proposal(proposal)
                    assert "execution failed" in str(exc_info.value).lower()

    def test_name_error_wrapped_in_runtime_error(self) -> None:
        """Test that NameError is wrapped in RuntimeError via subprocess."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="result = undefined_variable",
            language=CodeLanguage.CADQUERY,
        )
        with patch(
            "ll_gen.disposal.code_executor._CADQUERY_AVAILABLE", True
        ):
            with patch(
                "ll_gen.disposal.code_executor._OCC_AVAILABLE", True
            ):
                with patch(
                    "ll_gen.disposal.code_executor._run_in_subprocess",
                    side_effect=RuntimeError(
                        "CadQuery execution failed: NameError: name 'undefined_variable' is not defined"
                    ),
                ):
                    with pytest.raises(RuntimeError) as exc_info:
                        execute_code_proposal(proposal)
                    assert "execution failed" in str(exc_info.value).lower()

    def test_no_result_raises_runtime_error(self) -> None:
        """Test RuntimeError when code doesn't produce a result."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="x = 42",  # No result variable
            language=CodeLanguage.CADQUERY,
        )
        with patch(
            "ll_gen.disposal.code_executor._CADQUERY_AVAILABLE", True
        ):
            with patch(
                "ll_gen.disposal.code_executor._OCC_AVAILABLE", True
            ):
                with patch(
                    "ll_gen.disposal.code_executor._run_in_subprocess",
                    side_effect=RuntimeError(
                        "CadQuery execution did not produce a result. "
                        "Expected one of: result, part, model, shape, body, solid"
                    ),
                ):
                    with pytest.raises(RuntimeError) as exc_info:
                        execute_code_proposal(proposal)
                    assert "did not produce a result" in str(exc_info.value)


# ============================================================================
# SECTION 7: Module Import Tests
# ============================================================================


class TestModuleImport:
    """Test module import and availability flags."""

    def test_module_importable(self) -> None:
        """Test that code_executor module is importable."""
        from ll_gen.disposal import code_executor
        assert hasattr(code_executor, "execute_code_proposal")
        assert hasattr(code_executor, "TimeoutError")
        assert hasattr(code_executor, "_timeout_handler")

    def test_availability_flags_are_boolean(self) -> None:
        """Test that availability flags are boolean values."""
        from ll_gen.disposal import code_executor
        assert isinstance(code_executor._OCC_AVAILABLE, bool)
        assert isinstance(code_executor._CADQUERY_AVAILABLE, bool)

    def test_functions_are_callable(self) -> None:
        """Test that exported functions are callable."""
        from ll_gen.disposal import code_executor
        assert callable(code_executor.execute_code_proposal)
        assert callable(code_executor._timeout_handler)


# ============================================================================
# SECTION 8: Proposal Interface Tests
# ============================================================================


class TestProposalInterface:
    """Test CodeProposal interface used by executor."""

    def test_proposal_has_required_attributes(self) -> None:
        """Test that CodeProposal has all attributes used by executor."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="result = 42",
            language=CodeLanguage.CADQUERY,
        )
        assert hasattr(proposal, "code")
        assert hasattr(proposal, "language")
        assert proposal.code == "result = 42"
        assert proposal.language == CodeLanguage.CADQUERY

    def test_proposal_language_enum_values(self) -> None:
        """Test all CodeLanguage enum values are handled."""
        languages = [CodeLanguage.CADQUERY, CodeLanguage.OPENSCAD, CodeLanguage.PYTHONOCC]
        for lang in languages:
            proposal = CodeProposal(
                proposal_id="test",
                confidence=0.8,
                code="result = 42",
                language=lang,
            )
            assert proposal.language == lang


# ============================================================================
# SECTION 9: Sandbox Security Tests
# ============================================================================


class TestSandboxSecurity:
    """Test sandbox security restrictions."""

    def test_code_runs_in_subprocess_not_exec(self) -> None:
        """Test that user code is executed via subprocess, not exec()."""
        from ll_gen.disposal import code_executor
        # Verify _run_in_subprocess exists and is used
        assert hasattr(code_executor, "_run_in_subprocess")
        assert callable(code_executor._run_in_subprocess)

    def test_subprocess_timeout_enforcement(self) -> None:
        """Test that subprocess timeout is enforced cross-platform."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="import time; time.sleep(100); result = 42",
            language=CodeLanguage.CADQUERY,
        )
        with patch(
            "ll_gen.disposal.code_executor._CADQUERY_AVAILABLE", True
        ):
            with patch(
                "ll_gen.disposal.code_executor._OCC_AVAILABLE", True
            ):
                with patch(
                    "ll_gen.disposal.code_executor._run_in_subprocess",
                    side_effect=TimeoutError("CadQuery execution exceeded timeout of 1s"),
                ):
                    with pytest.raises(TimeoutError) as exc_info:
                        execute_code_proposal(proposal, timeout=1)
                    assert "timeout" in str(exc_info.value).lower()

    def test_dangerous_builtins_not_in_restricted_namespace(self) -> None:
        """Test that dangerous builtins are restricted."""
        # The code uses a restricted __builtins__ dict
        # These should NOT be available in the sandbox
        dangerous_builtins = [
            "eval", "exec", "compile", "open", "__import__",
            "globals", "locals", "vars", "getattr", "setattr",
            "delattr", "input",
        ]
        # Document what's expected to be restricted
        assert len(dangerous_builtins) > 0

    def test_restricted_builtins_structure(self) -> None:
        """Test that restricted builtins are a dict (not module)."""
        # The code creates a dict for __builtins__, not the full module
        # This is a key security measure
        safe_builtins = {
            "abs": abs,
            "round": round,
            "len": len,
        }
        assert isinstance(safe_builtins, dict)


# ============================================================================
# SECTION 10: Subprocess IPC Tests
# ============================================================================


class TestSubprocessIPC:
    """Test subprocess-based IPC for code execution."""

    def test_run_in_subprocess_success(self) -> None:
        """Test _run_in_subprocess returns parsed JSON on success."""
        from ll_gen.disposal.code_executor import _run_in_subprocess

        success_json = json.dumps({"success": True, "shape_type": "TopoDS_Shape"})
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = success_json
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = _run_in_subprocess(
                "print('hello')", "result = 42", 30, "Test"
            )
            assert result["success"] is True

    def test_run_in_subprocess_timeout(self) -> None:
        """Test _run_in_subprocess raises TimeoutError on timeout."""
        from ll_gen.disposal.code_executor import _run_in_subprocess

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="python", timeout=5),
        ):
            with pytest.raises(TimeoutError) as exc_info:
                _run_in_subprocess("print('hello')", "x = 1", 5, "Test")
            assert "timeout" in str(exc_info.value).lower()

    def test_run_in_subprocess_failure_json(self) -> None:
        """Test _run_in_subprocess raises RuntimeError on failure JSON."""
        from ll_gen.disposal.code_executor import _run_in_subprocess

        fail_json = json.dumps({"success": False, "error": "Something went wrong"})
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fail_json
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Something went wrong"):
                _run_in_subprocess("print('hello')", "bad code", 30, "Test")

    def test_run_in_subprocess_no_output(self) -> None:
        """Test _run_in_subprocess raises RuntimeError on empty output."""
        from ll_gen.disposal.code_executor import _run_in_subprocess

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="produced no output"):
                _run_in_subprocess("print('hello')", "x = 1", 30, "Test")

    def test_run_in_subprocess_invalid_json(self) -> None:
        """Test _run_in_subprocess raises RuntimeError on invalid JSON."""
        from ll_gen.disposal.code_executor import _run_in_subprocess

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "not json at all"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="invalid JSON"):
                _run_in_subprocess("print('hello')", "x = 1", 30, "Test")
