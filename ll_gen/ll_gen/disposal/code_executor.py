"""Execute CodeProposal objects in a sandboxed subprocess.

This module provides functionality to execute code proposals (CadQuery, OpenSCAD,
or raw pythonocc Python scripts) in an isolated subprocess and extract geometry
metadata from the resulting TopoDS_Shape.

Security: All user code runs in a separate process via ``subprocess.run()``,
never via ``exec()`` in the parent.  Timeout is enforced by
``subprocess.run(timeout=...)``, which works cross-platform.

The exec() namespace restricts ``__builtins__`` to remove dangerous functions
(``eval``, ``exec``, ``compile``, ``open``, ``breakpoint``) and installs a
restricted ``__import__`` that enforces the ``allowed_modules`` whitelist
from :class:`~ll_gen.config.CodegenConfig`.

**Known limitation (defense-in-depth only):** The exec-level sandbox is NOT
a security boundary.  Pre-injected objects (numpy, cadquery, math, OCC classes)
expose attribute chains that can reach ``ctypes``, ``importlib``, or the full
class hierarchy (e.g. ``numpy.ctypeslib.ctypes``,
``type(obj).__subclasses__()``).  The subprocess boundary is the primary
security control — the exec sandbox merely reduces the casual attack surface.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, List, Union

from ll_gen.config import CodeLanguage, CodegenConfig
from ll_gen.proposals.code_proposal import CodeProposal

_log = logging.getLogger(__name__)

# Default allowed modules when no config is provided
_DEFAULT_ALLOWED_MODULES = CodegenConfig().allowed_modules

# Lazy imports for optional dependencies
_OCC_AVAILABLE = False
_CADQUERY_AVAILABLE = False
_OCP_AVAILABLE = False

try:
    from OCC.Core.StlAPI import StlAPI_Reader
    from OCC.Core.TopoDS import TopoDS_Shape
    _OCC_AVAILABLE = True
except ImportError:
    pass

try:
    from OCP.TopoDS import TopoDS_Shape as _OCP_TopoDS_Shape  # noqa: F811
    _OCP_AVAILABLE = True
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


# ---------------------------------------------------------------------------
# Sandbox infrastructure for subprocess wrapper scripts
# ---------------------------------------------------------------------------

def _build_sandbox_preamble(allowed_modules: List[str]) -> str:
    """Build Python source that sets up restricted builtins and import guard.

    The generated code defines ``_safe_builtins`` (a dict) that should be
    assigned to ``namespace["__builtins__"]`` before calling ``exec()``.

    Args:
        allowed_modules: List of top-level module names the user code may import.

    Returns:
        A string of Python source code to be included in wrapper scripts.
    """
    modules_json = json.dumps(sorted(allowed_modules))
    modules_b64 = base64.b64encode(modules_json.encode("utf-8")).decode("ascii")
    return textwrap.dedent(f"""\
        # ---- Sandbox: restricted builtins & import guard ----
        import base64 as _b64_mod
        import builtins as _builtins_mod
        import json as _json_mod

        _ALLOWED_MODULES = set(_json_mod.loads(
            _b64_mod.b64decode({modules_b64!r}).decode("utf-8")
        ))
        _ORIGINAL_IMPORT = _builtins_mod.__import__

        def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0:
                top_level = name.split(".")[0]
                if top_level not in _ALLOWED_MODULES:
                    raise ImportError(
                        f"Import of {{name!r}} is not allowed. "
                        f"Allowed modules: {{sorted(_ALLOWED_MODULES)}}"
                    )
            return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

        _DANGEROUS_BUILTINS = frozenset({{
            "__import__", "eval", "exec", "compile",
            "open", "breakpoint", "exit", "quit",
        }})
        _safe_builtins = {{
            k: v for k, v in vars(_builtins_mod).items()
            if k not in _DANGEROUS_BUILTINS
        }}
        _safe_builtins["__import__"] = _restricted_import
        # ---- End sandbox setup ----
    """)


def _build_cadquery_wrapper(allowed_modules: List[str]) -> str:
    """Build the CadQuery subprocess wrapper script.

    Args:
        allowed_modules: Whitelist of importable top-level modules.

    Returns:
        Complete Python source for the CadQuery wrapper.
    """
    sandbox = _build_sandbox_preamble(allowed_modules)
    return textwrap.dedent("""\
        import json, sys, math, traceback
        try:
            import numpy
        except ImportError:
            numpy = None
        try:
            import cadquery
            from cadquery import Workplane
            cq = Workplane
        except ImportError:
            print(json.dumps({"success": False, "error": "CadQuery not available"}))
            sys.exit(0)

    """) + sandbox + textwrap.dedent("""\

        # --- Security boundary ---
        # Everything above this point is TRUSTED wrapper infrastructure.
        # The sandbox boundary is the exec(code, namespace) call below.
        # Only code running inside exec() is subject to the restricted
        # builtins and import guard.
        #
        # KNOWN LIMITATION: Pre-injected objects (numpy, cadquery, math)
        # expose attribute chains that can bypass the exec-level sandbox
        # (e.g. numpy.ctypeslib.ctypes, type(obj).__subclasses__()).
        # The subprocess boundary is the PRIMARY security control.
        # The exec sandbox is defense-in-depth only.

        namespace = {
            "__builtins__": _safe_builtins,
            "cadquery": cadquery,
            "cq": cq,
            "Workplane": Workplane,
            "math": math,
            "numpy": numpy,
        }

        import os as _os_init
        def _write_result_and_exit(meta_dict):
            _rp = _os_init.path.join(_os_init.path.dirname(sys.argv[1]), "result.json")
            with open(_rp, 'w') as _f:
                _f.write(json.dumps(meta_dict))
            sys.exit(0)

        code = open(sys.argv[1]).read()
        # Redirect stdout so user print() calls don't interfere with IPC
        _orig_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            exec(code, namespace)
        except ImportError as exc:
            _write_result_and_exit({
                "success": False,
                "error": f"Blocked import: {type(exc).__name__}: {exc}",
            })
        except Exception as exc:
            _write_result_and_exit({
                "success": False,
                "error": f"CadQuery execution failed: {type(exc).__name__}: {exc}",
            })
        finally:
            sys.stdout = _orig_stdout

        result_obj = None
        for var_name in ("result", "part", "model", "shape", "body", "solid"):
            if var_name in namespace:
                result_obj = namespace[var_name]
                break

        if result_obj is None:
            _write_result_and_exit({
                "success": False,
                "error": ("CadQuery execution did not produce a result. "
                          "Expected one of: result, part, model, shape, body, solid"),
            })

        # Extract metadata from the CadQuery result
        meta = {"success": True, "shape_type": type(result_obj).__name__}
        try:
            if isinstance(result_obj, cadquery.Workplane):
                cq_shape = result_obj.val().wrapped
            else:
                cq_shape = result_obj
            meta["shape_class"] = type(cq_shape).__name__
            # Try basic geometry stats via OCC
            try:
                from OCC.Core.BRepGProp import brepgprop
                from OCC.Core.GProp import GProp_GProps
                from OCC.Core.TopExp import TopExp_Explorer
                from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX

                props = GProp_GProps()
                brepgprop.VolumeProperties(cq_shape, props)
                meta["volume"] = props.Mass()

                sprops = GProp_GProps()
                brepgprop.SurfaceProperties(cq_shape, sprops)
                meta["surface_area"] = sprops.Mass()

                face_count = 0
                exp = TopExp_Explorer(cq_shape, TopAbs_FACE)
                while exp.More():
                    face_count += 1
                    exp.Next()
                meta["face_count"] = face_count

                edge_count = 0
                exp = TopExp_Explorer(cq_shape, TopAbs_EDGE)
                while exp.More():
                    edge_count += 1
                    exp.Next()
                meta["edge_count"] = edge_count

                vertex_count = 0
                exp = TopExp_Explorer(cq_shape, TopAbs_VERTEX)
                while exp.More():
                    vertex_count += 1
                    exp.Next()
                meta["vertex_count"] = vertex_count
            except Exception:
                pass
        except Exception as exc:
            meta["success"] = False
            meta["error"] = f"Failed to extract shape: {type(exc).__name__}: {exc}"

        # Serialize shape to BREP for the parent process to reload
        try:
            import os as _os
            brep_path = _os.path.join(_os.path.dirname(sys.argv[1]), "shape.brep")
            # CadQuery uses OCP bindings; write BREP via OCP.BRepTools
            from OCP.BRepTools import BRepTools as _BRepTools
            _BRepTools.Write_s(cq_shape, brep_path)
            meta["brep_path"] = brep_path
        except Exception:
            # Fallback: try OCC.Core for pythonocc environments
            try:
                from OCC.Core.BRepTools import breptools as _breptools
                _breptools.Write(cq_shape, brep_path)
                meta["brep_path"] = brep_path
            except Exception:
                pass

        import os as _os2
        _result_path = _os2.path.join(_os2.path.dirname(sys.argv[1]), "result.json")
        with open(_result_path, 'w') as _rf:
            _rf.write(json.dumps(meta))
    """)


def _build_pythonocc_wrapper(allowed_modules: List[str]) -> str:
    """Build the pythonocc subprocess wrapper script.

    Args:
        allowed_modules: Whitelist of importable top-level modules.

    Returns:
        Complete Python source for the pythonocc wrapper.
    """
    sandbox = _build_sandbox_preamble(allowed_modules)
    return textwrap.dedent("""\
        import json, sys, math, traceback
        try:
            import numpy
        except ImportError:
            numpy = None
        try:
            from OCC.Core.TopoDS import TopoDS_Shape
        except ImportError:
            print(json.dumps({"success": False, "error": "pythonocc not available"}))
            sys.exit(0)

    """) + sandbox + textwrap.dedent("""\

        # --- Security boundary ---
        # Everything above this point is TRUSTED wrapper infrastructure.
        # The sandbox applies only inside exec(code, namespace) below.
        #
        # KNOWN LIMITATION: Pre-injected objects (numpy, math, OCC classes)
        # expose attribute chains that can bypass the exec-level sandbox
        # (e.g. numpy.ctypeslib.ctypes, type(obj).__subclasses__()).
        # The subprocess boundary is the PRIMARY security control.
        # The exec sandbox is defense-in-depth only.

        namespace = {
            "__builtins__": _safe_builtins,
            "math": math,
            "numpy": numpy,
        }

        # Pre-import commonly-used OCC classes into the namespace so user
        # code can reference them directly (e.g. BRepPrimAPI_MakeBox(...)).
        # This uses the wrapper's unrestricted __import__ intentionally —
        # OCC is already in the allowed_modules whitelist, so this grants
        # no additional access beyond what the restricted import allows.
        _OCC_IMPORTS = {
            "OCC.Core.gp": [
                "gp_Pnt", "gp_Vec", "gp_Dir", "gp_Ax1", "gp_Ax2", "gp_Circ",
                "gp_Pln", "gp_Trsf", "gp_Pnt2d",
            ],
            "OCC.Core.BRepPrimAPI": [
                "BRepPrimAPI_MakeBox", "BRepPrimAPI_MakeSphere",
                "BRepPrimAPI_MakeCylinder", "BRepPrimAPI_MakeCone",
                "BRepPrimAPI_MakeTorus", "BRepPrimAPI_MakePrism",
                "BRepPrimAPI_MakeRevol",
            ],
            "OCC.Core.BRepAlgoAPI": [
                "BRepAlgoAPI_Fuse", "BRepAlgoAPI_Cut", "BRepAlgoAPI_Common",
                "BRepAlgoAPI_Section",
            ],
            "OCC.Core.BRepBuilderAPI": [
                "BRepBuilderAPI_MakeEdge", "BRepBuilderAPI_MakeWire",
                "BRepBuilderAPI_MakeFace", "BRepBuilderAPI_MakeShell",
                "BRepBuilderAPI_MakeSolid", "BRepBuilderAPI_Transform",
                "BRepBuilderAPI_Sewing",
            ],
            "OCC.Core.BRepFilletAPI": [
                "BRepFilletAPI_MakeFillet", "BRepFilletAPI_MakeChamfer",
            ],
            "OCC.Core.BRepOffsetAPI": [
                "BRepOffsetAPI_MakeThickSolid", "BRepOffsetAPI_MakePipe",
                "BRepOffsetAPI_ThruSections",
            ],
            "OCC.Core.TopoDS": ["TopoDS_Shape", "topods"],
            "OCC.Core.TopExp": ["TopExp_Explorer", "topexp"],
            "OCC.Core.TopAbs": [
                "TopAbs_FACE", "TopAbs_EDGE", "TopAbs_VERTEX",
                "TopAbs_WIRE", "TopAbs_SHELL", "TopAbs_SOLID",
            ],
            "OCC.Core.BRepGProp": ["brepgprop"],
            "OCC.Core.GProp": ["GProp_GProps"],
            "OCC.Core.BRepTools": ["breptools"],
            "OCC.Core.BRep": ["BRep_Tool", "BRep_Builder"],
            "OCC.Core.GeomAPI": [
                "GeomAPI_PointsToBSpline", "GeomAPI_Interpolate",
            ],
            "OCC.Core.TColgp": ["TColgp_Array1OfPnt", "TColgp_Array2OfPnt"],
        }
        for _mod_path, _names in _OCC_IMPORTS.items():
            try:
                _mod = __import__(_mod_path, fromlist=_names)
                for _name in _names:
                    try:
                        namespace[_name] = getattr(_mod, _name)
                    except AttributeError:
                        pass
            except ImportError:
                pass

        import os as _os_init
        def _write_result_and_exit(meta_dict):
            _rp = _os_init.path.join(_os_init.path.dirname(sys.argv[1]), "result.json")
            with open(_rp, 'w') as _f:
                _f.write(json.dumps(meta_dict))
            sys.exit(0)

        code = open(sys.argv[1]).read()
        # Redirect stdout so user print() calls don't interfere with IPC
        _orig_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            exec(code, namespace)
        except ImportError as exc:
            _write_result_and_exit({
                "success": False,
                "error": f"Blocked import: {type(exc).__name__}: {exc}",
            })
        except Exception as exc:
            _write_result_and_exit({
                "success": False,
                "error": f"pythonocc execution failed: {type(exc).__name__}: {exc}",
            })
        finally:
            sys.stdout = _orig_stdout

        result_obj = None
        for var_name in ("result", "part", "model", "shape", "body", "solid"):
            if var_name in namespace:
                result_obj = namespace[var_name]
                break

        if result_obj is None:
            _write_result_and_exit({
                "success": False,
                "error": ("pythonocc execution did not produce a result. "
                          "Expected one of: result, part, model, shape, body, solid"),
            })

        if not isinstance(result_obj, TopoDS_Shape):
            _write_result_and_exit({
                "success": False,
                "error": f"pythonocc result is not a TopoDS_Shape, got: {type(result_obj)}",
            })

        meta = {"success": True, "shape_type": type(result_obj).__name__}
        try:
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX

            props = GProp_GProps()
            brepgprop.VolumeProperties(result_obj, props)
            meta["volume"] = props.Mass()

            sprops = GProp_GProps()
            brepgprop.SurfaceProperties(result_obj, sprops)
            meta["surface_area"] = sprops.Mass()

            face_count = 0
            exp = TopExp_Explorer(result_obj, TopAbs_FACE)
            while exp.More():
                face_count += 1
                exp.Next()
            meta["face_count"] = face_count

            edge_count = 0
            exp = TopExp_Explorer(result_obj, TopAbs_EDGE)
            while exp.More():
                edge_count += 1
                exp.Next()
            meta["edge_count"] = edge_count

            vertex_count = 0
            exp = TopExp_Explorer(result_obj, TopAbs_VERTEX)
            while exp.More():
                vertex_count += 1
                exp.Next()
            meta["vertex_count"] = vertex_count
        except Exception:
            pass

        # Serialize shape to BREP for the parent process to reload
        try:
            import os
            brep_path = os.path.join(os.path.dirname(sys.argv[1]), "shape.brep")
            from OCC.Core.BRepTools import breptools
            breptools.Write(result_obj, brep_path)
            meta["brep_path"] = brep_path
        except Exception:
            pass

        import os as _os2
        _result_path = _os2.path.join(_os2.path.dirname(sys.argv[1]), "result.json")
        with open(_result_path, 'w') as _rf:
            _rf.write(json.dumps(meta))
    """)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_code_proposal(
    proposal: CodeProposal,
    timeout: int = 30,
    allowed_modules: List[str] | None = None,
) -> Any:
    """Execute a CodeProposal and return geometry metadata.

    Executes code proposals in an isolated subprocess with cross-platform
    timeout enforcement.  Supports CadQuery, OpenSCAD, and pythonocc code
    proposals.

    Args:
        proposal: The CodeProposal object containing code and language.
        timeout: Maximum execution time in seconds. Defaults to 30.
        allowed_modules: Top-level modules the user code is permitted to
            import.  Defaults to :attr:`CodegenConfig.allowed_modules`.

    Returns:
        A dict of geometry metadata (face_count, volume, etc.) on success,
        or a TopoDS_Shape when OCC is available and the language is OpenSCAD.

    Raises:
        RuntimeError: If execution fails, including timeout, import errors,
            or missing required dependencies.
        TimeoutError: If execution exceeds the timeout limit.
    """
    if allowed_modules is None:
        allowed_modules = list(_DEFAULT_ALLOWED_MODULES)

    if proposal.language == CodeLanguage.CADQUERY:
        return _execute_cadquery(proposal.code, timeout, allowed_modules)
    elif proposal.language == CodeLanguage.OPENSCAD:
        return _execute_openscad(proposal.code, timeout)
    elif proposal.language == CodeLanguage.PYTHONOCC:
        return _execute_pythonocc(proposal.code, timeout, allowed_modules)
    else:
        raise RuntimeError(
            f"Unsupported code language: {proposal.language}"
        )


def _run_in_subprocess(
    wrapper_script: str, code: str, timeout: int, language_label: str
) -> Union[dict[str, Any], Any]:
    """Write code to a temp file, run wrapper_script in a subprocess, parse JSON.

    Args:
        wrapper_script: Python source for the wrapper that executes user code.
        code: The user's code to execute.
        timeout: Timeout in seconds.
        language_label: Human label for error messages (e.g. "CadQuery").

    Returns:
        Parsed JSON dict from subprocess stdout.

    Raises:
        TimeoutError: On subprocess timeout.
        RuntimeError: On any other failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = os.path.join(tmpdir, "user_code.py")
        wrapper_path = os.path.join(tmpdir, "wrapper.py")

        with open(code_path, "w") as f:
            f.write(code)
        with open(wrapper_path, "w") as f:
            f.write(wrapper_script)

        # Install SIGALRM as defense-in-depth timeout on Unix systems.
        # The primary timeout is subprocess.run(timeout=...), but SIGALRM
        # guards the parent process against hangs in post-subprocess work
        # (e.g. BREP reload, JSON parsing) that could exceed the budget.
        _prev_handler = None
        _prev_alarm = 0
        _has_sigalrm = hasattr(signal, "SIGALRM")
        if _has_sigalrm:
            _prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            # Allow extra headroom beyond the subprocess timeout for
            # wrapper overhead (file I/O, JSON parsing, BREP reload).
            _prev_alarm = signal.alarm(timeout + 5)

        try:
            completed = subprocess.run(
                [sys.executable, wrapper_path, code_path],
                timeout=timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"{language_label} execution exceeded timeout of {timeout}s"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"{language_label} subprocess execution failed: "
                f"{type(e).__name__}: {str(e)}"
            ) from e
        finally:
            # Disarm SIGALRM and restore the previous handler
            if _has_sigalrm:
                signal.alarm(_prev_alarm)
                if _prev_handler is not None:
                    signal.signal(signal.SIGALRM, _prev_handler)

        # Read result from file-based IPC (immune to user print() pollution)
        result_path = os.path.join(tmpdir, "result.json")
        if os.path.isfile(result_path):
            with open(result_path, "r") as rf:
                result_text = rf.read()
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"{language_label} execution produced invalid JSON: {result_text[:200]}"
                ) from e
        else:
            # Fallback: no result file written (e.g. early exit or crash)
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{language_label} execution failed: {stderr or stdout}"
                )
            raise RuntimeError(
                f"{language_label} execution produced no output"
            )

        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error")
            if "did not produce a result" in error_msg:
                raise RuntimeError(error_msg)
            raise RuntimeError(error_msg)

        # If a BREP file was written, reload it as a TopoDS_Shape
        brep_path = result.get("brep_path")
        if brep_path and os.path.isfile(brep_path) and _OCC_AVAILABLE:
            try:
                from OCC.Core.BRepTools import breptools
                from OCC.Core.BRep import BRep_Builder
                from OCC.Core.TopoDS import TopoDS_Shape as _TopoDS_Shape

                shape = _TopoDS_Shape()
                builder = BRep_Builder()
                if breptools.Read(shape, brep_path, builder):
                    _log.debug("Reloaded TopoDS_Shape from BREP file")
                    return shape
                else:
                    _log.warning("Failed to read BREP file; returning metadata dict")
            except Exception as exc:
                _log.warning("BREP reload failed: %s", exc)

        return result


def _execute_cadquery(
    code: str, timeout: int, allowed_modules: List[str]
) -> Any:
    """Execute CadQuery code in a subprocess.

    Args:
        code: The CadQuery Python code to execute.
        timeout: Maximum execution time in seconds.
        allowed_modules: Whitelist of importable top-level modules.

    Returns:
        A dict of geometry metadata from the subprocess.

    Raises:
        RuntimeError: If execution fails or required dependencies are missing.
        TimeoutError: If execution exceeds the timeout limit.
    """
    if not _CADQUERY_AVAILABLE:
        raise RuntimeError(
            "CadQuery is not available. Install it with: pip install cadquery"
        )

    if not _OCP_AVAILABLE:
        raise RuntimeError(
            "OCP (CadQuery's OpenCASCADE bindings) is not available. "
            "Install CadQuery with: conda install -c cadquery cadquery"
        )

    wrapper = _build_cadquery_wrapper(allowed_modules)
    result = _run_in_subprocess(wrapper, code, timeout, "CadQuery")
    if isinstance(result, dict):
        # Note: BREP→OCC conversion is handled inside _run_in_subprocess
        # while the temp directory still exists. If we reach here with a dict,
        # the BREP reload either wasn't attempted or failed — the brep_path
        # is invalid because the TemporaryDirectory has been cleaned up.
        _log.debug("CadQuery returned metadata dict with keys: %s", list(result.keys()))
    else:
        _log.debug("CadQuery returned TopoDS_Shape object")
    return result


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


def _convert_ocp_to_occ_from_brep(brep_path: str) -> Any:
    """Read a BREP file written by OCP/CadQuery into an OCC.Core TopoDS_Shape.

    Args:
        brep_path: Path to the BREP file on disk.

    Returns:
        OCC.Core TopoDS_Shape, or None if reading fails.
    """
    from OCC.Core.BRepTools import breptools
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Shape

    shape = TopoDS_Shape()
    builder = BRep_Builder()
    if breptools.Read(shape, brep_path, builder):
        return shape
    return None


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


def _execute_pythonocc(
    code: str, timeout: int, allowed_modules: List[str]
) -> Any:
    """Execute pythonocc code in a subprocess.

    Args:
        code: The pythonocc Python code to execute.
        timeout: Maximum execution time in seconds.
        allowed_modules: Whitelist of importable top-level modules.

    Returns:
        A dict of geometry metadata from the subprocess.

    Raises:
        RuntimeError: If execution fails or required dependencies are missing.
        TimeoutError: If execution exceeds the timeout limit.
    """
    if not _OCC_AVAILABLE:
        raise RuntimeError(
            "pythonocc is not available. Install it with: pip install pythonocc"
        )

    # Ensure OCC is importable for user code via the restricted __import__
    if "OCC" not in allowed_modules:
        allowed_modules = [*allowed_modules, "OCC"]

    wrapper = _build_pythonocc_wrapper(allowed_modules)
    result = _run_in_subprocess(wrapper, code, timeout, "pythonocc")
    if isinstance(result, dict):
        _log.debug("pythonocc returned metadata dict with keys: %s", list(result.keys()))
    else:
        _log.debug("pythonocc returned TopoDS_Shape object")
    return result
