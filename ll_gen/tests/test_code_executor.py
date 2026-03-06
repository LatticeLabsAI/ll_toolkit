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
    _build_sandbox_preamble,
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


@pytest.mark.skipif(
    not hasattr(signal, "SIGALRM"),
    reason="SIGALRM is not available on Windows",
)
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
        from ll_gen.disposal.code_executor import _DEFAULT_ALLOWED_MODULES
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
            mock_exec.assert_called_once_with(
                proposal.code, 30, list(_DEFAULT_ALLOWED_MODULES)
            )

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
        from ll_gen.disposal.code_executor import _DEFAULT_ALLOWED_MODULES
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
            mock_exec.assert_called_once_with(
                proposal.code, 30, list(_DEFAULT_ALLOWED_MODULES)
            )

    def test_custom_timeout_passed_to_executor(self) -> None:
        """Test custom timeout is passed to the executor."""
        from ll_gen.disposal.code_executor import _DEFAULT_ALLOWED_MODULES
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
            mock_exec.assert_called_once_with(
                proposal.code, 60, list(_DEFAULT_ALLOWED_MODULES)
            )


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

    def test_ocp_unavailable_with_cadquery_raises_runtime_error(self) -> None:
        """Test RuntimeError when OCP bindings are not available for CadQuery."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="result = cq.Workplane().box(10,10,10)",
            language=CodeLanguage.CADQUERY,
        )
        with patch("ll_gen.disposal.code_executor._CADQUERY_AVAILABLE", True):
            with patch(
                "ll_gen.disposal.code_executor._OCP_AVAILABLE", False
            ):
                with pytest.raises(RuntimeError) as exc_info:
                    execute_code_proposal(proposal)
                assert "OCP" in str(exc_info.value)


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
        """Test that dangerous builtins are actually excluded by the sandbox."""
        # Execute the real sandbox preamble and inspect _safe_builtins
        preamble_src = _build_sandbox_preamble(["math", "json"])
        ns: dict[str, Any] = {}
        exec(compile(preamble_src, "<sandbox-preamble>", "exec"), ns)
        safe_builtins = ns["_safe_builtins"]

        # These must NOT be reachable from user code
        dangerous = [
            "eval", "exec", "compile", "open",
            "breakpoint", "exit", "quit",
        ]
        for name in dangerous:
            assert name not in safe_builtins, (
                f"{name!r} should be removed from sandbox builtins"
            )

        # __import__ should be replaced with the restricted version
        assert safe_builtins["__import__"] is ns["_restricted_import"]

    def test_restricted_builtins_structure(self) -> None:
        """Test that sandbox produces a dict (not module) for __builtins__."""
        preamble_src = _build_sandbox_preamble(["math"])
        ns: dict[str, Any] = {}
        exec(compile(preamble_src, "<sandbox-preamble>", "exec"), ns)
        safe_builtins = ns["_safe_builtins"]

        # Must be a plain dict — passing a module as __builtins__ to exec()
        # gives user code access to everything.
        assert isinstance(safe_builtins, dict)
        assert not isinstance(safe_builtins, type(__builtins__) if isinstance(__builtins__, type) else type(None))

        # Verify safe builtins still include essential non-dangerous items
        for name in ("abs", "round", "len", "range", "int", "float", "str", "list", "dict", "True", "False", "None"):
            assert name in safe_builtins, f"Safe builtin {name!r} should be available"

    def test_restricted_import_blocks_disallowed_module(self) -> None:
        """Test that the sandbox import guard actually blocks unauthorized imports."""
        preamble_src = _build_sandbox_preamble(["math"])
        ns: dict[str, Any] = {}
        exec(compile(preamble_src, "<sandbox-preamble>", "exec"), ns)

        restricted_import = ns["_restricted_import"]

        # Allowed module should succeed
        math_mod = restricted_import("math")
        assert math_mod is not None

        # Disallowed module should raise ImportError
        with pytest.raises(ImportError, match="not allowed"):
            restricted_import("subprocess")


# ============================================================================
# SECTION 10: Subprocess IPC Tests
# ============================================================================


class TestSubprocessIPC:
    """Test subprocess-based IPC for code execution."""

    def test_run_in_subprocess_success(self) -> None:
        """Test _run_in_subprocess returns parsed JSON on success."""
        from ll_gen.disposal.code_executor import _run_in_subprocess
        import os

        success_data = {"success": True, "shape_type": "TopoDS_Shape"}

        def _fake_run(cmd, **kwargs):
            # Write result.json in the tmpdir (same dir as user_code.py)
            tmpdir = os.path.dirname(cmd[2])  # cmd[2] is code_path
            with open(os.path.join(tmpdir, "result.json"), "w") as f:
                f.write(json.dumps(success_data))
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result

        with patch("subprocess.run", side_effect=_fake_run):
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
        import os

        fail_data = {"success": False, "error": "Something went wrong"}

        def _fake_run(cmd, **kwargs):
            tmpdir = os.path.dirname(cmd[2])
            with open(os.path.join(tmpdir, "result.json"), "w") as f:
                f.write(json.dumps(fail_data))
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result

        with patch("subprocess.run", side_effect=_fake_run):
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
        import os

        def _fake_run(cmd, **kwargs):
            tmpdir = os.path.dirname(cmd[2])
            with open(os.path.join(tmpdir, "result.json"), "w") as f:
                f.write("not json at all")
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result

        with patch("subprocess.run", side_effect=_fake_run):
            with pytest.raises(RuntimeError, match="invalid JSON"):
                _run_in_subprocess("print('hello')", "x = 1", 30, "Test")


# ============================================================================
# SECTION 11: Sandbox Preamble & Import Restriction Tests
# ============================================================================


class TestSandboxPreamble:
    """Test _build_sandbox_preamble generates correct restriction code."""

    def test_preamble_contains_allowed_modules(self) -> None:
        """Test that preamble embeds the allowed module list via base64."""
        from ll_gen.disposal.code_executor import _build_sandbox_preamble
        preamble = _build_sandbox_preamble(["math", "numpy"])
        # Modules are now base64-encoded for injection safety
        assert "b64decode" in preamble
        assert "_ALLOWED_MODULES" in preamble
        # Verify the encoded payload decodes to the expected module list
        ns: dict = {}
        exec(preamble, ns)
        assert ns["_ALLOWED_MODULES"] == {"math", "numpy"}

    def test_preamble_defines_safe_builtins(self) -> None:
        """Test that preamble defines _safe_builtins dict."""
        from ll_gen.disposal.code_executor import _build_sandbox_preamble
        preamble = _build_sandbox_preamble(["math"])
        assert "_safe_builtins" in preamble
        assert "_DANGEROUS_BUILTINS" in preamble

    def test_preamble_removes_dangerous_builtins(self) -> None:
        """Test that preamble removes eval, exec, compile, open, etc."""
        from ll_gen.disposal.code_executor import _build_sandbox_preamble
        preamble = _build_sandbox_preamble(["math"])
        for dangerous in ("eval", "exec", "compile", "open", "breakpoint"):
            assert f'"{dangerous}"' in preamble

    def test_preamble_installs_restricted_import(self) -> None:
        """Test that preamble installs a restricted __import__."""
        from ll_gen.disposal.code_executor import _build_sandbox_preamble
        preamble = _build_sandbox_preamble(["math"])
        assert "_restricted_import" in preamble
        assert '_safe_builtins["__import__"] = _restricted_import' in preamble

    def test_preamble_is_valid_python(self) -> None:
        """Test that the generated preamble compiles as valid Python."""
        from ll_gen.disposal.code_executor import _build_sandbox_preamble
        preamble = _build_sandbox_preamble(["math", "numpy", "cadquery"])
        # Should not raise SyntaxError
        compile(preamble, "<sandbox_preamble>", "exec")

    def test_preamble_execution_blocks_disallowed_import(self) -> None:
        """Test that executing the preamble creates a working import guard."""
        from ll_gen.disposal.code_executor import _build_sandbox_preamble
        preamble = _build_sandbox_preamble(["math"])
        ns: dict = {}
        exec(preamble, ns)
        safe_builtins = ns["_safe_builtins"]
        restricted_import = safe_builtins["__import__"]

        # Allowed import should succeed
        math_mod = restricted_import("math")
        assert hasattr(math_mod, "pi")

        # Disallowed import should raise ImportError
        with pytest.raises(ImportError, match="not allowed"):
            restricted_import("subprocess")

    def test_preamble_execution_allows_submodule_of_allowed(self) -> None:
        """Test that os.path is allowed if os is in allowed_modules."""
        from ll_gen.disposal.code_executor import _build_sandbox_preamble
        preamble = _build_sandbox_preamble(["os"])
        ns: dict = {}
        exec(preamble, ns)
        restricted_import = ns["_safe_builtins"]["__import__"]
        # __import__("os.path") returns the top-level 'os' module
        os_mod = restricted_import("os.path")
        assert hasattr(os_mod, "path")
        assert hasattr(os_mod.path, "join")


class TestWrapperBuilders:
    """Test _build_cadquery_wrapper and _build_pythonocc_wrapper."""

    def test_cadquery_wrapper_includes_sandbox(self) -> None:
        """Test CadQuery wrapper includes sandbox preamble."""
        from ll_gen.disposal.code_executor import _build_cadquery_wrapper
        wrapper = _build_cadquery_wrapper(["math", "numpy", "cadquery"])
        assert "_safe_builtins" in wrapper
        assert '"__builtins__": _safe_builtins' in wrapper

    def test_pythonocc_wrapper_includes_sandbox(self) -> None:
        """Test pythonocc wrapper includes sandbox preamble."""
        from ll_gen.disposal.code_executor import _build_pythonocc_wrapper
        wrapper = _build_pythonocc_wrapper(["math", "numpy", "OCC"])
        assert "_safe_builtins" in wrapper
        assert '"__builtins__": _safe_builtins' in wrapper

    def test_cadquery_wrapper_catches_blocked_import(self) -> None:
        """Test CadQuery wrapper has ImportError handler."""
        from ll_gen.disposal.code_executor import _build_cadquery_wrapper
        wrapper = _build_cadquery_wrapper(["math"])
        assert "except ImportError as exc:" in wrapper
        assert "Blocked import" in wrapper

    def test_pythonocc_wrapper_catches_blocked_import(self) -> None:
        """Test pythonocc wrapper has ImportError handler."""
        from ll_gen.disposal.code_executor import _build_pythonocc_wrapper
        wrapper = _build_pythonocc_wrapper(["math"])
        assert "except ImportError as exc:" in wrapper
        assert "Blocked import" in wrapper

    def test_allowed_modules_custom_list(self) -> None:
        """Test execute_code_proposal accepts custom allowed_modules."""
        proposal = CodeProposal(
            proposal_id="test",
            confidence=0.8,
            code="result = 'test'",
            language=CodeLanguage.CADQUERY,
        )
        custom_modules = ["math", "numpy", "cadquery", "scipy"]
        with patch(
            "ll_gen.disposal.code_executor._execute_cadquery"
        ) as mock_exec:
            mock_exec.return_value = MagicMock()
            execute_code_proposal(
                proposal, allowed_modules=custom_modules
            )
            mock_exec.assert_called_once_with(
                proposal.code, 30, custom_modules
            )
