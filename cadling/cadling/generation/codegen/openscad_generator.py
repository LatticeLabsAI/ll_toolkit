"""OpenSCAD code generation backend.

This module provides LLM-driven OpenSCAD script generation and execution
via the openscad command-line tool for text-to-CAD generation.

Classes:
    OpenSCADGenerator: Generate OpenSCAD scripts from text descriptions
    OpenSCADExecutor: Execute OpenSCAD scripts via subprocess

Example:
    generator = OpenSCADGenerator(model_name="gpt-4", api_provider="openai")
    script = generator.generate("A box with rounded edges and a hole through it")

    executor = OpenSCADExecutor()
    result = executor.execute(script, "output.stl")

    if result["success"]:
        print(f"STL written to: {result['output_path']}")
    else:
        print(f"Execution failed: {result['error']}")
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger(__name__)

# Lazy import flag for pythonocc (needed for STL->STEP conversion)
_OCC_AVAILABLE = False
try:
    from OCC.Core.STEPControl import STEPControl_Writer
    from OCC.Core.StlAPI import StlAPI_Reader as _StlReader

    _OCC_AVAILABLE = True
except ImportError:
    _log.debug("pythonocc not available; STL-to-STEP conversion will not work")


def _convert_image_to_png_openscad(image_path: Path, suffix: str) -> bytes | None:
    """Convert BMP, TIFF, or SVG images to PNG bytes in memory.

    Args:
        image_path: Path to the source image file.
        suffix: File extension (without dot): 'bmp', 'tiff', 'tif', or 'svg'.

    Returns:
        PNG image bytes, or None if conversion fails.
    """
    import io

    if suffix == "svg":
        try:
            import cairosvg

            png_bytes = cairosvg.svg2png(url=str(image_path))
            return png_bytes
        except ImportError:
            _log.warning(
                "cairosvg not installed — cannot convert SVG to PNG. "
                "Install with: pip install cairosvg"
            )
            return None
        except Exception as exc:
            _log.warning("SVG to PNG conversion failed: %s", exc)
            return None
    else:
        try:
            from PIL import Image as PILImage

            img = PILImage.open(image_path)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            _log.warning(
                "Pillow not installed — cannot convert %s to PNG. "
                "Install with: pip install pillow",
                suffix,
            )
            return None
        except Exception as exc:
            _log.warning("%s to PNG conversion failed: %s", suffix.upper(), exc)
            return None


class OpenSCADGenerator:
    """Generate OpenSCAD scripts from natural language descriptions.

    Uses an LLM (via the ChatAgent interface from cadling.sdg.qa.utils) to
    convert a text description into a complete OpenSCAD script.

    Attributes:
        model_name: LLM model identifier (e.g., 'gpt-4', 'claude-3-opus').
        api_provider: LLM provider name ('openai', 'anthropic', 'vllm', 'ollama').
        _agent: Lazily-initialized ChatAgent instance.
        _system_prompt: Cached system prompt text.
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_provider: str = "openai",
    ) -> None:
        """Initialize OpenSCADGenerator.

        Args:
            model_name: LLM model identifier.
            api_provider: LLM provider name.
        """
        self.model_name = model_name
        self.api_provider = api_provider
        self._agent: Any = None
        self._system_prompt: str | None = None
        _log.info(
            "Initialized OpenSCADGenerator (model=%s, provider=%s)",
            model_name,
            api_provider,
        )

    def _get_agent(self) -> Any:
        """Lazily initialize and return the ChatAgent.

        Returns:
            Configured ChatAgent instance.

        Raises:
            ImportError: If required LLM library is not installed.
        """
        if self._agent is not None:
            return self._agent

        from cadling.sdg.qa.utils import ChatAgent
        from cadling.sdg.qa.base import LlmOptions, LlmProvider

        provider_map = {
            "openai": LlmProvider.OPENAI,
            "anthropic": LlmProvider.ANTHROPIC,
            "vllm": LlmProvider.VLLM,
            "ollama": LlmProvider.OLLAMA,
            "openai_compatible": LlmProvider.OPENAI_COMPATIBLE,
            "mlx": LlmProvider.MLX,
        }

        provider = provider_map.get(self.api_provider.lower())
        if provider is None:
            raise ValueError(
                f"Unsupported API provider: {self.api_provider}. "
                f"Supported: {list(provider_map.keys())}"
            )

        options = LlmOptions(
            provider=provider,
            model_id=self.model_name,
            temperature=0.4,
            max_tokens=4096,
        )
        self._agent = ChatAgent.from_options(options)
        _log.debug("ChatAgent initialized for OpenSCAD generation")
        return self._agent

    def _build_system_prompt(self) -> str:
        """Build or return cached OpenSCAD system prompt.

        Loads the prompt from the prompts directory and caches it.

        Returns:
            System prompt text with OpenSCAD syntax reference.
        """
        if self._system_prompt is not None:
            return self._system_prompt

        from cadling.generation.codegen.prompts import load_openscad_system_prompt

        self._system_prompt = load_openscad_system_prompt()
        return self._system_prompt

    @staticmethod
    def _encode_image(image_path: str | Path) -> tuple[str, str, int] | None:
        """Encode an image file to a base64 data-URI string.

        Supports PNG, JPEG, WebP, GIF natively, and additionally BMP, TIFF,
        and SVG by converting to PNG in-memory before encoding.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (data_uri, mime_type, raw_byte_count), or *None* if the
            file does not exist.
        """
        import base64 as _b64

        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            _log.warning("Reference image not found: %s", image_path)
            return None

        suffix = image_path_obj.suffix.lower().lstrip(".")

        # Formats that need conversion to PNG before encoding
        convert_formats = {"bmp", "tiff", "tif", "svg"}

        if suffix in convert_formats:
            image_data = _convert_image_to_png_openscad(image_path_obj, suffix)
            if image_data is None:
                _log.warning(
                    "Failed to convert %s image to PNG: %s", suffix, image_path
                )
                return None
            mime_type = "image/png"
        else:
            image_data = image_path_obj.read_bytes()
            mime_map = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "webp": "image/webp",
                "gif": "image/gif",
            }
            mime_type = mime_map.get(suffix, "image/png")

        b64_data = _b64.b64encode(image_data).decode("ascii")
        data_uri = f"data:{mime_type};base64,{b64_data}"
        return data_uri, mime_type, len(image_data)

    def _build_user_prompt(
        self,
        description: str,
        image_path: str | None = None,
    ) -> str:
        """Build user prompt for OpenSCAD code generation.

        Selects the appropriate prompt template based on input modality:

        - **Text-only** (no image): uses ``text_to_code.txt``.
        - **Image-primary** (image provided, description absent or
          auto-generated): uses ``image_to_code.txt`` with structured
          image-analysis instructions.
        - **Text + image**: uses ``text_to_code.txt`` with the image
          injected into the ``{IMAGE_SECTION}`` placeholder.

        Args:
            description: Natural language description of the desired part.
            image_path: Optional path to a reference image.

        Returns:
            Complete user prompt string.
        """
        from cadling.generation.codegen.prompts import (
            load_image_to_code_prompt,
            load_text_to_code_prompt,
        )

        encoded = None
        if image_path is not None:
            encoded = self._encode_image(image_path)

        # Image-primary route
        _AUTO_DESCRIPTIONS = {
            "",
            "Generate a 3D CAD model that matches the provided reference image. "
            "Infer the geometry, dimensions, and features from the image.",
        }
        image_is_primary = (
            encoded is not None
            and description.strip() in _AUTO_DESCRIPTIONS
        )

        if image_is_primary:
            template = load_image_to_code_prompt()
            data_uri, mime_type, raw_size = encoded
            image_block = (
                f"A reference image has been provided as base64 ({mime_type}).\n"
                f"Analyze the geometry, proportions, and features below:\n\n"
                f"{data_uri}\n"
            )
            prompt = template.replace("{IMAGE_DATA}", image_block)
            prompt = prompt.replace(
                "{DESCRIPTION}",
                description if description.strip() else "(none provided)",
            )
            _log.debug(
                "Using image-to-code template for OpenSCAD: %s (%d bytes)",
                image_path,
                raw_size,
            )
            return prompt

        # Text-primary route (with optional supplementary image)
        template = load_text_to_code_prompt()
        prompt = template.replace("{DESCRIPTION}", description)

        if encoded is not None:
            data_uri, mime_type, raw_size = encoded
            image_section = (
                "\n\n## Reference Image\n"
                f"A reference image has been provided as base64 ({mime_type}).\n"
                "Use this image to inform the geometry, proportions, and features "
                "of the OpenSCAD model. The image is encoded below:\n\n"
                f"{data_uri}\n\n"
                "Interpret the visual shape, features, and proportions from this "
                "image and translate them into accurate OpenSCAD geometry."
            )
            prompt = prompt.replace("{IMAGE_SECTION}", image_section)
            _log.debug(
                "Included reference image in text-to-code: %s (%d bytes)",
                image_path,
                raw_size,
            )
        else:
            prompt = prompt.replace("{IMAGE_SECTION}", "")

        return prompt

    def generate(
        self,
        description: str,
        reference_image_path: str | None = None,
    ) -> str:
        """Generate an OpenSCAD script from a text description and/or image.

        Sends the description (and optional reference image) to the LLM
        with an OpenSCAD-specific system prompt, then extracts the OpenSCAD
        code block from the response.

        Args:
            description: Natural language description of the desired part.
            reference_image_path: Optional path to a reference image.

        Returns:
            OpenSCAD script as a string.

        Raises:
            RuntimeError: If LLM call fails.
            ValueError: If no valid OpenSCAD code can be extracted.
        """
        agent = self._get_agent()
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(description, reference_image_path)

        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        _log.info("OpenSCAD generation for: %s", description[:80])
        try:
            response = agent.ask(full_prompt, max_tokens=4096)
        except Exception as e:
            raise RuntimeError(f"OpenSCAD generation LLM call failed: {e}") from e

        script = self._extract_code(response)
        _log.info("Generated OpenSCAD script (%d chars)", len(script))
        return script

    def repair(
        self,
        code: str,
        error: str,
        description: str,
    ) -> str:
        """Repair OpenSCAD code based on error message using LLM.

        Sends the failing code, error message, and original description to the
        LLM with a repair-specific prompt to generate a fixed version.

        Args:
            code: The failing OpenSCAD script.
            error: Error message from execution or validation.
            description: Original text description of the desired part.

        Returns:
            Repaired OpenSCAD script.

        Raises:
            RuntimeError: If repair LLM call fails.
        """
        from cadling.generation.codegen.prompts import load_repair_prompt

        agent = self._get_agent()
        system_prompt = self._build_system_prompt()

        repair_template = load_repair_prompt()
        user_prompt = repair_template.replace("{DESCRIPTION}", description)
        user_prompt = user_prompt.replace("{PREVIOUS_CODE}", code)
        user_prompt = user_prompt.replace("{ERRORS}", error)

        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        _log.info("OpenSCAD repair for error: %s", error[:80])
        try:
            response = agent.ask(full_prompt, max_tokens=4096)
        except Exception as e:
            raise RuntimeError(f"OpenSCAD repair LLM call failed: {e}") from e

        script = self._extract_code(response)
        _log.info("Repaired OpenSCAD script (%d chars)", len(script))
        return script

    def _extract_code(self, response: str) -> str:
        """Extract OpenSCAD code block from LLM response.

        Looks for fenced code blocks (```openscad ... ``` or ```scad ... ```
        or plain ```) and returns the content.

        Args:
            response: Raw LLM response text.

        Returns:
            Extracted OpenSCAD code.

        Raises:
            ValueError: If no valid OpenSCAD code can be extracted.
        """
        # Try fenced code blocks with various language tags
        patterns = [
            r"```(?:openscad|scad)\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                code = max(matches, key=len).strip()
                if code:
                    return code

        # Fallback: look for OpenSCAD-like content
        lines = response.strip().split("\n")
        openscad_indicators = [
            "module ",
            "difference()",
            "union()",
            "intersection()",
            "cube(",
            "cylinder(",
            "sphere(",
            "translate(",
            "rotate(",
            "scale(",
            "linear_extrude(",
            "rotate_extrude(",
            "polyhedron(",
            "$fn",
            "$fa",
            "$fs",
        ]
        code_lines = []
        in_code = False

        for line in lines:
            stripped = line.strip()
            if any(ind in stripped for ind in openscad_indicators):
                in_code = True
            if in_code:
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()

        raise ValueError(
            "Could not extract valid OpenSCAD code from LLM response. "
            f"Response preview: {response[:200]}"
        )


class OpenSCADExecutor:
    """Execute OpenSCAD scripts via the command-line tool.

    Writes scripts to temporary .scad files and invokes the openscad
    command to produce STL (or other format) output. Also provides
    STL-to-STEP conversion using pythonocc when available.

    Attributes:
        openscad_path: Path to the openscad executable.
        _openscad_available: Whether the openscad binary was found.
    """

    def __init__(self, openscad_path: str = "openscad") -> None:
        """Initialize OpenSCADExecutor.

        Args:
            openscad_path: Path to the openscad executable. Defaults to
                searching PATH.
        """
        self.openscad_path = openscad_path
        self._openscad_available = shutil.which(openscad_path) is not None

        if not self._openscad_available:
            _log.warning(
                "OpenSCAD executable not found at '%s'. "
                "Install OpenSCAD or provide the correct path.",
                openscad_path,
            )
        else:
            resolved = shutil.which(openscad_path)
            _log.info("Initialized OpenSCADExecutor (path=%s)", resolved)

    def execute(self, script: str, output_path: str) -> dict:
        """Execute an OpenSCAD script and produce output geometry.

        Writes the script to a temporary .scad file, then runs openscad
        to compile it into the specified output format (determined by
        the output_path file extension: .stl, .off, .amf, .3mf, .csg,
        .dxf, .svg, .png).

        Args:
            script: OpenSCAD script string.
            output_path: Path for the output file (e.g., 'output.stl').

        Returns:
            Dictionary with keys:
                - 'output_path': Path to the generated file (or None)
                - 'success': Whether execution succeeded
                - 'error': Error message string (or None)
                - 'execution_time': Time taken in seconds
                - 'stdout': OpenSCAD stdout output
                - 'stderr': OpenSCAD stderr output
        """
        if not self._openscad_available:
            return {
                "output_path": None,
                "success": False,
                "error": (
                    f"OpenSCAD executable not found at '{self.openscad_path}'. "
                    "Install OpenSCAD: https://openscad.org/downloads.html"
                ),
                "execution_time": 0.0,
                "stdout": "",
                "stderr": "",
            }

        start_time = time.time()

        # Write script to temp file
        try:
            scad_tmpfile = tempfile.NamedTemporaryFile(
                suffix=".scad", mode="w", delete=False, encoding="utf-8"
            )
            scad_tmpfile.write(script)
            scad_tmpfile.flush()
            scad_path = scad_tmpfile.name
            scad_tmpfile.close()
        except Exception as e:
            return {
                "output_path": None,
                "success": False,
                "error": f"Failed to write temp .scad file: {e}",
                "execution_time": time.time() - start_time,
                "stdout": "",
                "stderr": "",
            }

        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Run openscad
        cmd = [self.openscad_path, "-o", str(output_file), scad_path]
        _log.info("Running OpenSCAD: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            execution_time = time.time() - start_time

            if proc.returncode != 0:
                error_msg = proc.stderr.strip() if proc.stderr else (
                    f"OpenSCAD exited with code {proc.returncode}"
                )
                _log.error("OpenSCAD failed: %s", error_msg)
                return {
                    "output_path": None,
                    "success": False,
                    "error": error_msg,
                    "execution_time": execution_time,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }

            # Verify output file was created
            if not output_file.exists():
                return {
                    "output_path": None,
                    "success": False,
                    "error": "OpenSCAD completed but output file not created",
                    "execution_time": execution_time,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }

            output_size = output_file.stat().st_size
            _log.info(
                "OpenSCAD produced %s (%d bytes) in %.1fs",
                output_file.name,
                output_size,
                execution_time,
            )

            return {
                "output_path": str(output_file),
                "success": True,
                "error": None,
                "execution_time": execution_time,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            _log.error("OpenSCAD timed out after 120s")
            return {
                "output_path": None,
                "success": False,
                "error": "OpenSCAD execution timed out (120s limit)",
                "execution_time": execution_time,
                "stdout": "",
                "stderr": "",
            }
        except Exception as e:
            execution_time = time.time() - start_time
            _log.error("OpenSCAD execution error: %s", e)
            return {
                "output_path": None,
                "success": False,
                "error": f"OpenSCAD execution error: {e}",
                "execution_time": execution_time,
                "stdout": "",
                "stderr": "",
            }
        finally:
            # Clean up temp scad file
            try:
                Path(scad_path).unlink(missing_ok=True)
            except Exception:
                pass

    def convert_to_step(self, stl_path: str, step_path: str) -> bool:
        """Convert STL file to STEP format via pythonocc.

        Uses pythonocc's STL reader and STEP writer to convert mesh
        geometry to a B-Rep STEP representation. Note that STL-to-STEP
        conversion produces tessellated (triangulated) B-Rep faces, not
        analytical surfaces.

        Args:
            stl_path: Path to input STL file.
            step_path: Path to output STEP file.

        Returns:
            True if conversion succeeded, False otherwise.
        """
        if not _OCC_AVAILABLE:
            _log.error(
                "pythonocc not available for STL-to-STEP conversion. "
                "Install with: conda install -c conda-forge pythonocc-core"
            )
            return False

        stl_file = Path(stl_path)
        if not stl_file.exists():
            _log.error("STL file not found: %s", stl_path)
            return False

        try:
            from OCC.Core.StlAPI import StlAPI_Reader
            from OCC.Core.STEPControl import (
                STEPControl_Writer,
                STEPControl_AsIs,
            )
            from OCC.Core.IFSelect import IFSelect_RetDone
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.TopoDS import TopoDS_Shape

            # Read STL
            stl_reader = StlAPI_Reader()
            shape = TopoDS_Shape()
            stl_reader.Read(shape, str(stl_file))

            if shape.IsNull():
                _log.error("STL reader produced null shape")
                return False

            # Write STEP
            step_output = Path(step_path)
            step_output.parent.mkdir(parents=True, exist_ok=True)

            step_writer = STEPControl_Writer()
            step_writer.Transfer(shape, STEPControl_AsIs)
            status = step_writer.Write(str(step_output))

            if status != IFSelect_RetDone:
                _log.error("STEP writer failed with status: %s", status)
                return False

            _log.info(
                "Converted STL -> STEP: %s -> %s", stl_path, step_path
            )
            return True

        except Exception as e:
            _log.error("STL-to-STEP conversion failed: %s", e)
            return False
