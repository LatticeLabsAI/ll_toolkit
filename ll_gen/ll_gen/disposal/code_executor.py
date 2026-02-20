"""Execute CodeProposal objects in a sandboxed environment.

This module provides functionality to execute code proposals (CadQuery, OpenSCAD,
or raw pythonocc Python scripts) in a sandboxed environment and extract the
resulting TopoDS_Shape geometry.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import tempfile
from typing import Any

from ll_gen.config import CodeLanguage
from ll_gen.proposals.code_proposal import CodeProposal

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_OCC_AVAILABLE = False
_CADQUERY_AVAILABLE = False

try:
    from OCC.Core.StlAPI import StlAPI_Reader
    from OCC.Core.TopoDS import TopoDS_Shape
    _OCC_AVAILABLE = True
except ImportError:
    pass

cadquery = None  # sentinel for unittest.mock.patch
try:
    import cadquery
    _CADQUERY_AVAILABLE = True
except ImportError:
    pass


class TimeoutError(Exception):
    """Raised when code execution exceeds the timeout limit."""

    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    """Handle timeout signal.

    Args:
        signum: Signal number.
        frame: Current stack frame.

    Raises:
        TimeoutError: Always raised to interrupt execution.
    """
    raise TimeoutError("Code execution exceeded timeout limit")


def execute_code_proposal(
    proposal: CodeProposal, timeout: int = 30
) -> Any:
    """Execute a CodeProposal and return the resulting geometry.

    Executes code proposals in a sandboxed environment with appropriate
    restrictions based on the code language. Supports CadQuery, OpenSCAD,
    and pythonocc code proposals.

    Args:
        proposal: The CodeProposal object containing code and language.
        timeout: Maximum execution time in seconds. Defaults to 30.

    Returns:
        The resulting TopoDS_Shape object.

    Raises:
        RuntimeError: If execution fails, including timeout, import errors,
            or missing required dependencies.
        TimeoutError: If execution exceeds the timeout limit.
    """
    if proposal.language == CodeLanguage.CADQUERY:
        return _execute_cadquery(proposal.code, timeout)
    elif proposal.language == CodeLanguage.OPENSCAD:
        return _execute_openscad(proposal.code, timeout)
    elif proposal.language == CodeLanguage.PYTHONOCC:
        return _execute_pythonocc(proposal.code, timeout)
    else:
        raise RuntimeError(
            f"Unsupported code language: {proposal.language}"
        )


def _execute_cadquery(code: str, timeout: int) -> Any:
    """Execute CadQuery code in a sandboxed environment.

    Args:
        code: The CadQuery Python code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        The resulting TopoDS_Shape object.

    Raises:
        RuntimeError: If execution fails or required dependencies are missing.
        TimeoutError: If execution exceeds the timeout limit.
    """
    if not _CADQUERY_AVAILABLE:
        raise RuntimeError(
            "CadQuery is not available. Install it with: pip install cadquery"
        )

    if not _OCC_AVAILABLE:
        raise RuntimeError(
            "pythonocc is not available. Install it with: pip install pythonocc"
        )

    # Create restricted namespace with allowed imports
    # Provide common CadQuery aliases to avoid import statements
    namespace = {
        "cadquery": cadquery,
        "cq": cadquery.Workplane,  # Common alias used in examples
        "Workplane": cadquery.Workplane,
        "math": __import__("math"),
        "numpy": __import__("numpy"),
        "__builtins__": {
            "abs": abs,
            "round": round,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "sum": sum,
            "min": min,
            "max": max,
            "float": float,
            "int": int,
            "str": str,
            "bool": bool,
            "print": print,
            "Exception": Exception,
        },
    }

    # Set up signal alarm for timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        exec(code, namespace)
    except TimeoutError as e:
        raise TimeoutError(str(e)) from e
    except Exception as e:
        raise RuntimeError(
            f"CadQuery execution failed: {type(e).__name__}: {str(e)}"
        ) from e
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    # Try to extract the result from various possible variable names
    result = None
    for var_name in ["result", "part", "model", "shape", "body", "solid"]:
        if var_name in namespace:
            result = namespace[var_name]
            break

    if result is None:
        raise RuntimeError(
            "CadQuery execution did not produce a result. "
            "Expected one of: result, part, model, shape, body, solid"
        )

    # Extract TopoDS_Shape from CadQuery Workplane if needed
    if isinstance(result, cadquery.Workplane):
        try:
            cq_shape = result.val().wrapped
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract shape from CadQuery result: {str(e)}"
            ) from e
    else:
        cq_shape = result

    # Convert from OCP (cadquery) to OCC.Core (pythonocc) via BREP serialization
    # CadQuery uses OCP SWIG bindings which are incompatible with OCC.Core
    return _convert_ocp_to_occ(cq_shape)


def _convert_ocp_to_occ(ocp_shape: Any) -> Any:
    """Convert an OCP (cadquery) shape to OCC.Core (pythonocc) shape.

    CadQuery uses OCP SWIG bindings while our validation uses OCC.Core.
    These are incompatible, so we serialize through BREP format.

    Args:
        ocp_shape: Shape from OCP (cadquery's SWIG bindings).

    Returns:
        TopoDS_Shape from OCC.Core (pythonocc's SWIG bindings).

    Raises:
        RuntimeError: If conversion fails.
    """
    import tempfile
    import os

    # Serialize to BREP using OCP's exporter
    try:
        from OCP.BRepTools import BRepTools
        from OCP.BRep import BRep_Builder
    except ImportError:
        # If OCP not available, assume the shape is already OCC.Core compatible
        return ocp_shape

    # Write to temp file using OCP
    with tempfile.NamedTemporaryFile(suffix=".brep", delete=False) as f:
        temp_path = f.name

    try:
        # Export using OCP
        BRepTools.Write_s(ocp_shape, temp_path)

        # Import using pythonocc (OCC.Core)
        from OCC.Core.BRepTools import breptools
        from OCC.Core.BRep import BRep_Builder as OCC_Builder
        from OCC.Core.TopoDS import TopoDS_Shape

        occ_shape = TopoDS_Shape()
        builder = OCC_Builder()
        success = breptools.Read(occ_shape, temp_path, builder)

        if not success:
            raise RuntimeError("Failed to read BREP file with pythonocc")

        return occ_shape

    except Exception as e:
        raise RuntimeError(
            f"Failed to convert shape from OCP to OCC.Core: {str(e)}"
        ) from e
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _execute_openscad(code: str, timeout: int) -> Any:
    """Execute OpenSCAD code and convert the result to TopoDS_Shape.

    Args:
        code: The OpenSCAD code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        The resulting TopoDS_Shape object.

    Raises:
        RuntimeError: If execution fails or required dependencies are missing.
    """
    if not _OCC_AVAILABLE:
        raise RuntimeError(
            "pythonocc is not available. Install it with: pip install pythonocc"
        )

    # Create temporary files for input and output
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "input.scad")
        output_file = os.path.join(tmpdir, "output.stl")

        # Write OpenSCAD code to temporary file
        try:
            with open(input_file, "w") as f:
                f.write(code)
        except Exception as e:
            raise RuntimeError(
                f"Failed to write OpenSCAD code to temporary file: {str(e)}"
            ) from e

        # Execute OpenSCAD
        try:
            subprocess.run(
                ["openscad", "-o", output_file, input_file],
                timeout=timeout,
                check=True,
                capture_output=True,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"OpenSCAD execution exceeded timeout of {timeout}s"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"OpenSCAD execution failed: {e.stderr.decode('utf-8', errors='replace')}"
            ) from e
        except FileNotFoundError as e:
            raise RuntimeError(
                "OpenSCAD not found. Please install OpenSCAD."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"OpenSCAD execution failed: {type(e).__name__}: {str(e)}"
            ) from e

        # Read the STL file and convert to TopoDS_Shape
        try:
            if not os.path.exists(output_file):
                raise RuntimeError("OpenSCAD did not produce output.stl")

            shape = TopoDS_Shape()
            stl_reader = StlAPI_Reader()
            stl_reader.Read(shape, output_file)

            if shape.IsNull():
                raise RuntimeError(
                    "Failed to read STL file or resulting shape is null"
                )

            return shape
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert STL to TopoDS_Shape: {str(e)}"
            ) from e


def _execute_pythonocc(code: str, timeout: int) -> Any:
    """Execute pythonocc code in a sandboxed environment.

    Args:
        code: The pythonocc Python code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        The resulting TopoDS_Shape object.

    Raises:
        RuntimeError: If execution fails or required dependencies are missing.
        TimeoutError: If execution exceeds the timeout limit.
    """
    if not _OCC_AVAILABLE:
        raise RuntimeError(
            "pythonocc is not available. Install it with: pip install pythonocc"
        )

    # Import all OCC.Core modules for the namespace
    try:
        from OCC import Core as occ_core

        # Build a namespace with all OCC.Core submodules
        occ_namespace = {}
        for attr_name in dir(occ_core):
            if not attr_name.startswith("_"):
                attr = getattr(occ_core, attr_name)
                occ_namespace[attr_name] = attr
    except Exception as e:
        raise RuntimeError(f"Failed to import OCC.Core modules: {str(e)}") from e

    # Create restricted namespace with allowed imports
    namespace = {
        "math": __import__("math"),
        "numpy": __import__("numpy"),
        "__builtins__": {
            "abs": abs,
            "round": round,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "sum": sum,
            "min": min,
            "max": max,
            "float": float,
            "int": int,
            "str": str,
            "bool": bool,
            "print": print,
            "Exception": Exception,
        },
        **occ_namespace,
    }

    # Set up signal alarm for timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        exec(code, namespace)
    except TimeoutError as e:
        raise TimeoutError(str(e)) from e
    except Exception as e:
        raise RuntimeError(
            f"pythonocc execution failed: {type(e).__name__}: {str(e)}"
        ) from e
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    # Try to extract the result from the namespace
    result = None
    for var_name in ["result", "part", "model", "shape", "body", "solid"]:
        if var_name in namespace:
            result = namespace[var_name]
            break

    if result is None:
        raise RuntimeError(
            "pythonocc execution did not produce a result. "
            "Expected one of: result, part, model, shape, body, solid"
        )

    # Verify result is a TopoDS_Shape
    if not isinstance(result, TopoDS_Shape):
        raise RuntimeError(
            f"pythonocc result is not a TopoDS_Shape, got: {type(result)}"
        )

    return result
