"""CadQuery code generation backend.

This module provides LLM-driven CadQuery script generation, sandboxed execution,
and validation with retry loops for text-to-CAD generation.

Classes:
    CadQueryGenerator: Generate CadQuery Python scripts from text descriptions
    CadQueryExecutor: Execute CadQuery scripts in a sandboxed environment
    CadQueryValidator: Validate generated STEP output and retry on failure

Example:
    generator = CadQueryGenerator(model_name="gpt-4", api_provider="openai")
    script = generator.generate("A flanged bearing housing with 4 bolt holes")

    executor = CadQueryExecutor(timeout=30)
    result = executor.execute(script)

    if result["success"]:
        executor.export_step(result, "output.step")
    else:
        print(f"Execution failed: {result['error']}")

    # Full validate-and-retry loop
    validator = CadQueryValidator()
    final_result = validator.validate_and_retry(
        generator, executor,
        "A flanged bearing housing with 4 bolt holes",
        max_retries=3,
    )
"""

from __future__ import annotations

import base64
import logging
import re
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

# Lazy import flags
_CADQUERY_AVAILABLE = False
_OCC_AVAILABLE = False

try:
    import cadquery as cq

    _CADQUERY_AVAILABLE = True
except ImportError:
    _log.debug("cadquery not available; CadQueryExecutor will not work")

try:
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer

    _OCC_AVAILABLE = True
except ImportError:
    _log.debug("pythonocc not available; topology validation will be limited")


def _convert_image_to_png(image_path: Path, suffix: str) -> bytes | None:
    """Convert BMP, TIFF, or SVG images to PNG bytes in memory.

    Args:
        image_path: Path to the source image file.
        suffix: File extension (without dot): 'bmp', 'tiff', 'tif', or 'svg'.

    Returns:
        PNG image bytes, or None if conversion fails.
    """
    import io

    if suffix == "svg":
        # SVG → PNG via cairosvg (optional dependency)
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
        # BMP, TIFF → PNG via Pillow
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


class CadQueryGenerator:
    """Generate CadQuery Python scripts from natural language descriptions.

    Uses an LLM (via the ChatAgent interface from cadling.sdg.qa.utils) to
    convert a text description -- and optionally a reference image -- into a
    complete CadQuery Python script that produces a Workplane result.

    Attributes:
        model_name: LLM model identifier (e.g., 'gpt-4', 'claude-3-opus').
        api_provider: LLM provider name ('openai', 'anthropic', 'vllm', 'ollama').
        max_retries: Maximum retry attempts for LLM calls on transient errors.
        _agent: Lazily-initialized ChatAgent instance.
        _system_prompt: Cached system prompt text.
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_provider: str = "openai",
        max_retries: int = 3,
    ) -> None:
        """Initialize CadQueryGenerator.

        Args:
            model_name: LLM model identifier.
            api_provider: LLM provider name.
            max_retries: Maximum retries for transient LLM errors.
        """
        self.model_name = model_name
        self.api_provider = api_provider
        self.max_retries = max_retries
        self._agent: Any = None
        self._system_prompt: str | None = None
        _log.info(
            "Initialized CadQueryGenerator (model=%s, provider=%s)",
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

        from cadling.sdg.qa.utils import ChatAgent, initialize_llm
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
        _log.debug("ChatAgent initialized for CadQuery generation")
        return self._agent

    def _build_system_prompt(self) -> str:
        """Build or return cached CadQuery system prompt.

        Loads the prompt from the prompts directory and caches it.

        Returns:
            System prompt text with CadQuery API reference.
        """
        if self._system_prompt is not None:
            return self._system_prompt

        from cadling.generation.codegen.prompts import load_cadquery_system_prompt

        self._system_prompt = load_cadquery_system_prompt()
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
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            _log.warning("Reference image not found: %s", image_path)
            return None

        suffix = image_path_obj.suffix.lower().lstrip(".")

        # Formats that need conversion to PNG before encoding
        convert_formats = {"bmp", "tiff", "tif", "svg"}

        if suffix in convert_formats:
            image_data = _convert_image_to_png(image_path_obj, suffix)
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

        b64_data = base64.b64encode(image_data).decode("ascii")
        data_uri = f"data:{mime_type};base64,{b64_data}"
        return data_uri, mime_type, len(image_data)

    def _build_user_prompt(
        self,
        description: str,
        image_path: str | None = None,
    ) -> str:
        """Build user prompt for code generation.

        Selects the appropriate prompt template based on input modality:

        - **Text-only** (no image): uses ``text_to_code.txt``.
        - **Text + image**: uses ``text_to_code.txt`` with the image injected
          into the ``{IMAGE_SECTION}`` placeholder.
        - **Image-only** (description is auto-generated or empty): uses the
          dedicated ``image_to_code.txt`` template which contains structured
          image-analysis instructions (form ID, feature decomposition,
          dimension estimation, multi-view guidance).

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

        # ------------------------------------------------------------------
        # Route: image-primary  (image provided, description is absent or
        # was auto-generated by the CLI as a generic placeholder)
        # ------------------------------------------------------------------
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
                "Using image-to-code template with image: %s (%d bytes)",
                image_path,
                raw_size,
            )
            return prompt

        # ------------------------------------------------------------------
        # Route: text-primary  (text provided, image is optional supplement)
        # ------------------------------------------------------------------
        template = load_text_to_code_prompt()
        prompt = template.replace("{DESCRIPTION}", description)

        if encoded is not None:
            data_uri, mime_type, raw_size = encoded
            image_section = (
                "\n\n## Reference Image\n"
                f"A reference image has been provided as base64 ({mime_type}).\n"
                "Use this image to inform the geometry, proportions, and features "
                "of the CadQuery model. The image is encoded below:\n\n"
                f"{data_uri}\n\n"
                "Interpret the visual shape, features, and proportions from this "
                "image and translate them into accurate CadQuery geometry."
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
        """Generate a CadQuery Python script from a text description.

        Sends the description (and optional reference image) to the LLM
        with a CadQuery-specific system prompt, then extracts the Python
        code block from the LLM response.

        Args:
            description: Natural language description of the desired part.
            reference_image_path: Optional path to a reference image.

        Returns:
            CadQuery Python script as a string.

        Raises:
            RuntimeError: If LLM fails after max_retries attempts.
            ValueError: If no valid Python code block is found in response.
        """
        agent = self._get_agent()
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(description, reference_image_path)

        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                _log.info(
                    "CadQuery generation attempt %d/%d for: %s",
                    attempt,
                    self.max_retries,
                    description[:80],
                )
                response = agent.ask(full_prompt, max_tokens=4096)
                script = self._extract_code(response)
                _log.info(
                    "Generated CadQuery script (%d chars) on attempt %d",
                    len(script),
                    attempt,
                )
                return script
            except Exception as e:
                last_error = e
                _log.warning(
                    "CadQuery generation attempt %d failed: %s", attempt, e
                )
                if attempt < self.max_retries:
                    time.sleep(1.0 * attempt)

        raise RuntimeError(
            f"CadQuery generation failed after {self.max_retries} attempts: "
            f"{last_error}"
        )

    def repair(
        self,
        code: str,
        error: str,
        description: str,
    ) -> str:
        """Repair CadQuery code based on error message using LLM.

        Sends the failing code, error message, and original description to the
        LLM with a repair-specific prompt to generate a fixed version.

        Args:
            code: The failing CadQuery script.
            error: Error message from execution or validation.
            description: Original text description of the desired part.

        Returns:
            Repaired CadQuery Python script.

        Raises:
            RuntimeError: If repair fails after max_retries attempts.
        """
        from cadling.generation.codegen.prompts import load_repair_prompt

        agent = self._get_agent()
        system_prompt = self._build_system_prompt()

        repair_template = load_repair_prompt()
        user_prompt = repair_template.replace("{DESCRIPTION}", description)
        user_prompt = user_prompt.replace("{PREVIOUS_CODE}", code)
        user_prompt = user_prompt.replace("{ERRORS}", error)

        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        last_error_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                _log.info(
                    "CadQuery repair attempt %d/%d for error: %s",
                    attempt,
                    self.max_retries,
                    error[:80],
                )
                response = agent.ask(full_prompt, max_tokens=4096)
                script = self._extract_code(response)
                _log.info(
                    "Repaired CadQuery script (%d chars) on attempt %d",
                    len(script),
                    attempt,
                )
                return script
            except Exception as e:
                last_error_exc = e
                _log.warning(
                    "CadQuery repair attempt %d failed: %s", attempt, e
                )
                if attempt < self.max_retries:
                    time.sleep(1.0 * attempt)

        raise RuntimeError(
            f"CadQuery repair failed after {self.max_retries} attempts: "
            f"{last_error_exc}"
        )

    def _extract_code(self, response: str) -> str:
        """Extract Python code block from LLM response.

        Looks for fenced code blocks (```python ... ```) and returns the
        content. Falls back to the entire response if no code block found
        but the response appears to be valid Python.

        Handles MLX-specific artifacts like special tokens and prompt echoes.

        Args:
            response: Raw LLM response text.

        Returns:
            Extracted Python code.

        Raises:
            ValueError: If no valid Python code can be extracted.
        """
        # Step 1: Clean common MLX/LLM artifacts from response
        cleaned = self._clean_llm_response(response)

        # Step 2: Try fenced code blocks first
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, cleaned, re.DOTALL)
        if matches:
            # Return the longest code block (most likely to be the full script)
            code = max(matches, key=len).strip()
            if code:
                return self._clean_extracted_code(code)

        # Step 3: Try to find code between import and result assignment
        import_match = re.search(
            r"(import cadquery as cq[\s\S]*?result\s*=[\s\S]*?)(?:\n\n[A-Z]|\Z)",
            cleaned,
            re.MULTILINE,
        )
        if import_match:
            code = import_match.group(1).strip()
            if code:
                return self._clean_extracted_code(code)

        # Step 4: Fallback - find lines starting with Python indicators
        lines = cleaned.split("\n")
        python_indicators = [
            "import ",
            "from ",
            "result ",
            "result=",
            "cq.",
            "Workplane",
            "def ",
            "#",  # Comments
        ]
        code_lines = []
        in_code = False

        for line in lines:
            stripped = line.strip()
            # Start capturing when we see a Python indicator
            if any(stripped.startswith(ind) for ind in python_indicators):
                in_code = True
            # Stop capturing at clear non-code markers (explanation text)
            if in_code and stripped and re.match(r"^[A-Z][a-z]+.*[.:!]$", stripped):
                # Looks like an English sentence, might be post-code explanation
                # Only break if we already have substantial code
                if len(code_lines) > 5:
                    break
            if in_code:
                code_lines.append(line)

        if code_lines:
            return self._clean_extracted_code("\n".join(code_lines))

        raise ValueError(
            "Could not extract valid Python code from LLM response. "
            f"Response preview: {cleaned[:200]}"
        )

    def _clean_llm_response(self, response: str) -> str:
        """Clean common LLM artifacts from response text.

        Removes special tokens, JSON wrappers, and other artifacts that
        some models (especially MLX-served local models) may produce.

        Args:
            response: Raw LLM response text.

        Returns:
            Cleaned response text.
        """
        cleaned = response

        # Remove common end-of-text tokens from various models
        end_tokens = [
            "<|endoftext|>",
            "<|im_end|>",
            "<|eot_id|>",
            "</s>",
            "<|end|>",
            "<|assistant|>",
            "[/INST]",
            "```\n\n---",  # Common markdown ending
        ]
        for token in end_tokens:
            if token in cleaned:
                cleaned = cleaned.split(token)[0]

        # Remove thinking/reasoning blocks some models produce
        # E.g., <thinking>...</thinking> or <|think|>...</|think|>
        cleaned = re.sub(r"<\|?thinking\|?>.*?</?\|?thinking\|?>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<\|?think\|?>.*?</?\|?think\|?>", "", cleaned, flags=re.DOTALL)

        # Remove JSON wrapper if model wrapped response in JSON
        # E.g., {"code": "import cadquery..."}
        json_match = re.search(r'\{\s*"code"\s*:\s*"(.*?)"\s*\}', cleaned, re.DOTALL)
        if json_match:
            # Unescape JSON string
            cleaned = json_match.group(1).replace("\\n", "\n").replace('\\"', '"')

        return cleaned.strip()

    def _clean_extracted_code(self, code: str) -> str:
        """Clean extracted code of remaining artifacts.

        Args:
            code: Extracted Python code.

        Returns:
            Cleaned Python code.
        """
        lines = code.split("\n")
        cleaned_lines = []

        for line in lines:
            # Skip lines that are clearly not Python
            stripped = line.strip()

            # Skip empty markdown artifacts
            if stripped in ("```", "```python", "```py"):
                continue

            # Skip lines that look like model commentary
            if stripped.startswith("This ") and stripped.endswith(":"):
                continue
            if stripped.startswith("Here ") and "code" in stripped.lower():
                continue

            cleaned_lines.append(line)

        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()

        # Ensure we have a result assignment
        code_text = "\n".join(cleaned_lines)

        # If code doesn't have 'result =' anywhere, try to find the last
        # workplane assignment and alias it to 'result'
        if "result" not in code_text and "=" in code_text:
            # Find the last variable assignment that looks like a workplane
            last_assignment_match = re.search(
                r"^(\w+)\s*=\s*\(?.*(?:Workplane|\.extrude|\.box|\.cylinder)",
                code_text,
                re.MULTILINE,
            )
            if last_assignment_match:
                var_name = last_assignment_match.group(1)
                if var_name != "cq":
                    code_text += f"\nresult = {var_name}"
                    _log.debug("Added 'result = %s' to extracted code", var_name)

        return code_text.strip()

    def generate_from_graph(
        self,
        node_features: "np.ndarray",
        edge_index: Optional["np.ndarray"] = None,
        adjacency: Optional[Dict[int, List[int]]] = None,
        edge_features: Optional["np.ndarray"] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """Generate CadQuery code from decoded graph features.

        Converts graph node/edge features into a structured prompt and uses
        the LLM to generate CadQuery code. Includes validation and retry loop.

        Args:
            node_features: Face features [N, feature_dim] from decoder
            edge_index: Adjacency in COO format [2, M]
            adjacency: Adjacency as dict (alternative to edge_index)
            edge_features: Optional edge features [M, edge_dim]
            metadata: Optional metadata (volume, bbox, etc.)
            max_iterations: Maximum generate-validate iterations

        Returns:
            Dictionary with:
                - success: Whether generation succeeded
                - script: Generated CadQuery script
                - prompt: The prompt used for generation
                - iterations: Number of iterations used
                - errors: List of errors encountered

        Example:
            generator = CadQueryGenerator(model_name="gpt-4")
            result = generator.generate_from_graph(
                node_features=decoded_features,
                edge_index=adjacency,
            )
            if result["success"]:
                print(result["script"])
        """
        import numpy as np

        from cadling.generation.codegen.graph_prompt_formatter import GraphToPromptFormatter

        # Format graph as prompt
        formatter = GraphToPromptFormatter()
        prompt = formatter.format_for_cadquery(
            node_features=node_features,
            edge_index=edge_index,
            adjacency=adjacency,
            edge_features=edge_features,
            metadata=metadata,
        )

        errors: List[str] = []
        final_script = ""

        for iteration in range(1, max_iterations + 1):
            try:
                _log.info(
                    "Graph-to-code generation attempt %d/%d",
                    iteration,
                    max_iterations,
                )

                if iteration == 1:
                    script = self.generate(prompt)
                else:
                    # Use repair with accumulated errors
                    error_context = "\n".join(f"- {err}" for err in errors[-3:])
                    script = self.repair(
                        code=final_script,
                        error=error_context,
                        description=prompt,
                    )

                final_script = script

                # Basic syntax validation
                try:
                    compile(script, "<graph_generated>", "exec")
                except SyntaxError as e:
                    errors.append(f"Syntax error: {e}")
                    continue

                # Check for required result assignment
                if "result" not in script:
                    errors.append("Script missing 'result' variable assignment")
                    continue

                _log.info(
                    "Generated CadQuery script from graph (%d chars) on iteration %d",
                    len(script),
                    iteration,
                )

                return {
                    "success": True,
                    "script": script,
                    "prompt": prompt,
                    "iterations": iteration,
                    "errors": errors,
                }

            except Exception as e:
                errors.append(f"Generation error: {e}")
                _log.warning(
                    "Graph-to-code generation attempt %d failed: %s",
                    iteration,
                    e,
                )

        return {
            "success": False,
            "script": final_script,
            "prompt": prompt,
            "iterations": max_iterations,
            "errors": errors,
        }


class CadQueryExecutor:
    """Execute CadQuery scripts in a sandboxed environment.

    Runs generated CadQuery Python scripts with restricted builtins and
    a timeout, captures the resulting Workplane or Shape object, and
    provides STEP export functionality.

    Attributes:
        timeout: Maximum execution time in seconds.
    """

    # Allowed builtins for sandboxed execution
    _SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "hasattr",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "list",
        "map",
        "max",
        "min",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "True",
        "False",
        "None",
        "__import__",
    }

    # Allowed modules for import within sandbox
    _ALLOWED_MODULES = {
        "cadquery",
        "math",
        "cmath",
        "numpy",
        "functools",
        "itertools",
        "collections",
    }

    def __init__(self, timeout: int = 30) -> None:
        """Initialize CadQueryExecutor.

        Args:
            timeout: Maximum execution time in seconds.
        """
        self.timeout = timeout
        _log.info("Initialized CadQueryExecutor (timeout=%ds)", timeout)

    def execute(self, script: str) -> dict:
        """Execute a CadQuery script in a sandboxed environment.

        Runs the script with restricted builtins and a timeout. Captures
        the 'result' variable from the script namespace, which should be
        a CadQuery Workplane or Shape object.

        Args:
            script: CadQuery Python script string.

        Returns:
            Dictionary with keys:
                - 'shape': The CadQuery shape/workplane object (or None)
                - 'success': Whether execution succeeded
                - 'error': Error message string (or None)
                - 'execution_time': Time taken in seconds
        """
        if not _CADQUERY_AVAILABLE:
            return {
                "shape": None,
                "success": False,
                "error": (
                    "cadquery is not installed. "
                    "Install with: conda install -c cadquery cadquery"
                ),
                "execution_time": 0.0,
            }

        start_time = time.time()
        result = self._sandbox_execute(script)
        result["execution_time"] = time.time() - start_time
        return result

    def export_step(self, result: dict, output_path: str) -> bool:
        """Export execution result to STEP file.

        Args:
            result: Execution result dictionary from execute().
            output_path: Path to write the STEP file.

        Returns:
            True if export succeeded, False otherwise.
        """
        if not result.get("success") or result.get("shape") is None:
            _log.warning("Cannot export: no valid shape in result")
            return False

        if not _CADQUERY_AVAILABLE:
            _log.error("cadquery not available for STEP export")
            return False

        try:
            import cadquery as cq

            shape = result["shape"]
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(shape, cq.Workplane):
                cq.exporters.export(shape, str(output), exportType="STEP")
            elif hasattr(shape, "exportStep"):
                shape.exportStep(str(output))
            else:
                # Try wrapping in Assembly for export
                assy = cq.Assembly()
                assy.add(shape)
                assy.save(str(output), exportType="STEP")

            _log.info("Exported STEP file: %s", output_path)
            return True

        except Exception as e:
            _log.error("STEP export failed: %s", e)
            return False

    def _sandbox_execute(self, script: str) -> dict:
        """Execute CadQuery script in a subprocess sandbox.

        Validates the script AST for safety, then executes it in an isolated
        subprocess with a timeout and restricted permissions. Falls back to
        in-process restricted execution if subprocess isolation is unavailable.

        Args:
            script: CadQuery Python script string.

        Returns:
            Dictionary with 'shape', 'success', and 'error' keys.
        """
        import ast

        # --- Step 1: AST validation before any execution ---
        try:
            tree = ast.parse(script, filename="<cadquery_script>")
        except SyntaxError as e:
            _log.error("Script has syntax errors: %s", e)
            return {"shape": None, "success": False, "error": f"Syntax error: {e}"}

        # Walk the AST to detect dangerous constructs
        _FORBIDDEN_NAMES = {
            "eval", "exec", "compile", "globals", "locals", "vars",
            "__import__", "getattr", "setattr", "delattr",
            "breakpoint", "exit", "quit",
            "open",  # file I/O
        }
        _FORBIDDEN_ATTRIBUTES = {
            "__subclasses__", "__bases__", "__class__", "__mro__",
            "__globals__", "__code__", "__builtins__",
            "system", "popen", "subprocess", "Popen",
        }

        for node in ast.walk(tree):
            # Check function calls to forbidden names
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in _FORBIDDEN_NAMES:
                    return {
                        "shape": None,
                        "success": False,
                        "error": f"Forbidden function call: {func.id}()",
                    }
            # Check forbidden attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in _FORBIDDEN_ATTRIBUTES:
                    return {
                        "shape": None,
                        "success": False,
                        "error": f"Forbidden attribute access: .{node.attr}",
                    }
            # Check forbidden Name references (e.g., os, sys, subprocess)
            if isinstance(node, ast.Name) and node.id in {"os", "sys", "subprocess", "shutil"}:
                return {
                    "shape": None,
                    "success": False,
                    "error": f"Forbidden module reference: {node.id}",
                }
            # Disallow Import/ImportFrom of non-allowed modules
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level = alias.name.split(".")[0]
                    if top_level not in self._ALLOWED_MODULES:
                        return {
                            "shape": None,
                            "success": False,
                            "error": (
                                f"Import of '{alias.name}' is not allowed in sandbox. "
                                f"Allowed modules: {sorted(self._ALLOWED_MODULES)}"
                            ),
                        }
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    top_level = node.module.split(".")[0]
                    if top_level not in self._ALLOWED_MODULES:
                        return {
                            "shape": None,
                            "success": False,
                            "error": (
                                f"Import from '{node.module}' is not allowed in sandbox. "
                                f"Allowed modules: {sorted(self._ALLOWED_MODULES)}"
                            ),
                        }

        _log.debug("AST validation passed for CadQuery script")

        # --- Step 2: Try subprocess-based execution for full isolation ---
        subprocess_result = self._subprocess_execute(script)
        if subprocess_result is not None:
            return subprocess_result

        # --- Step 3: Fallback to in-process restricted execution ---
        _log.debug("Subprocess execution unavailable, using in-process sandbox")
        return self._inprocess_execute(script)

    def _subprocess_execute(self, script: str) -> dict | None:
        """Execute CadQuery script in an isolated subprocess.

        Writes the script to a temporary file and runs it in a subprocess
        with a timeout. Returns None if subprocess execution is not feasible
        (e.g., cadquery not importable in subprocess).

        Args:
            script: CadQuery Python script string.

        Returns:
            Result dict, or None if subprocess execution is not feasible.
        """
        import subprocess
        import json
        import textwrap

        # Build a wrapper script that executes the user code and serializes
        # the result to a temp STEP file so the parent can load it
        wrapper = textwrap.dedent("""\
            import sys
            import json
            import tempfile

            try:
                import cadquery as cq
            except ImportError:
                print(json.dumps({{"success": False, "error": "cadquery not available"}}))
                sys.exit(1)

            import math
            import os

            tmp_path = None
            try:
                exec_globals = {{"cq": cq, "cadquery": cq, "math": math}}
                exec(compile({script_repr}, "<cadquery_script>", "exec"), exec_globals)

                result = exec_globals.get("result")
                if result is None:
                    for var_name in ("part", "model", "shape", "body", "solid"):
                        result = exec_globals.get(var_name)
                        if result is not None:
                            break

                if result is None:
                    print(json.dumps({{"success": False, "error": "No result variable"}}))
                    sys.exit(0)

                # Export to temp STEP for parent to load
                tmp = tempfile.NamedTemporaryFile(suffix=".step", delete=False)
                tmp_path = tmp.name
                tmp.close()

                if isinstance(result, cq.Workplane):
                    cq.exporters.export(result, tmp_path, exportType="STEP")
                elif hasattr(result, "exportStep"):
                    result.exportStep(tmp_path)
                else:
                    assy = cq.Assembly()
                    assy.add(result)
                    assy.save(tmp_path, exportType="STEP")

                print(json.dumps({{"success": True, "step_path": tmp_path}}))

            except Exception as e:
                # Clean up temp file on failure so it doesn't leak
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                print(json.dumps({{"success": False, "error": str(e)}}))
                sys.exit(0)
        """).format(script_repr=repr(script))

        try:
            proc = subprocess.run(
                [sys.executable, "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if proc.returncode != 0 and not proc.stdout.strip():
                _log.debug(
                    "Subprocess failed (rc=%d): %s",
                    proc.returncode,
                    proc.stderr[:200],
                )
                return None

            output = proc.stdout.strip()
            if not output:
                return None

            result_data = json.loads(output)

            if result_data.get("success") and result_data.get("step_path"):
                # Load the STEP file back into a CadQuery shape
                step_path = result_data["step_path"]
                try:
                    import cadquery as cq

                    shape = cq.importers.importStep(step_path)
                    _log.info("Subprocess execution succeeded, loaded STEP result")
                    return {"shape": shape, "success": True, "error": None}
                except Exception as e:
                    _log.warning("Failed to load subprocess STEP result: %s", e)
                    return {"shape": None, "success": False, "error": str(e)}
                finally:
                    import os
                    try:
                        os.unlink(step_path)
                    except OSError:
                        pass
            else:
                return {
                    "shape": None,
                    "success": False,
                    "error": result_data.get("error", "Unknown subprocess error"),
                }

        except subprocess.TimeoutExpired:
            _log.error("Subprocess execution timed out after %ds", self.timeout)
            return {
                "shape": None,
                "success": False,
                "error": f"Script execution exceeded {self.timeout}s timeout",
            }
        except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
            _log.debug("Subprocess execution not feasible: %s", e)
            return None

    def _inprocess_execute(self, script: str) -> dict:
        """Fallback in-process restricted execution.

        Uses restricted builtins and signal-based timeout. Only used when
        subprocess execution is unavailable.

        Args:
            script: CadQuery Python script string (already AST-validated).

        Returns:
            Dictionary with 'shape', 'success', and 'error' keys.
        """
        import builtins as _builtins_module

        # Build restricted builtins
        safe_builtins = {}
        for name in self._SAFE_BUILTINS:
            if hasattr(_builtins_module, name):
                safe_builtins[name] = getattr(_builtins_module, name)

        # Create a guarded __import__ that only allows safe modules
        original_import = _builtins_module.__import__

        def guarded_import(name, *args, **kwargs):
            top_level = name.split(".")[0]
            if top_level not in self._ALLOWED_MODULES:
                raise ImportError(
                    f"Import of '{name}' is not allowed in sandbox. "
                    f"Allowed modules: {sorted(self._ALLOWED_MODULES)}"
                )
            return original_import(name, *args, **kwargs)

        safe_builtins["__import__"] = guarded_import

        # Pre-import cadquery into the namespace
        import cadquery as cq
        import math

        sandbox_globals: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "cq": cq,
            "cadquery": cq,
            "math": math,
        }

        # Set up timeout handler
        def timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError(
                f"CadQuery script execution exceeded {self.timeout}s timeout"
            )

        # Execute with timeout
        old_handler = None
        try:
            # signal.alarm only works on Unix
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)

            exec(compile(script, "<cadquery_script>", "exec"), sandbox_globals)

            signal.alarm(0)  # Cancel the alarm

        except TimeoutError as e:
            _log.error("Script execution timed out: %s", e)
            return {"shape": None, "success": False, "error": str(e)}
        except Exception as e:
            signal.alarm(0)
            _log.error("Script execution failed: %s", e)
            return {"shape": None, "success": False, "error": str(e)}
        finally:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

        # Extract result from the namespace
        shape = sandbox_globals.get("result")
        if shape is None:
            # Try common alternative variable names
            for var_name in ("part", "model", "shape", "body", "solid"):
                shape = sandbox_globals.get(var_name)
                if shape is not None:
                    _log.debug("Found result in variable '%s'", var_name)
                    break

        if shape is None:
            return {
                "shape": None,
                "success": False,
                "error": (
                    "Script executed but no 'result' variable was set. "
                    "The CadQuery script must assign the final Workplane to "
                    "a variable named 'result'."
                ),
            }

        _log.info("Script executed successfully, captured shape object")
        return {"shape": shape, "success": True, "error": None}


class CadQueryValidator:
    """Validate generated STEP files and orchestrate retry loops.

    Runs basic topology checks on STEP output (solid validity, face/edge
    counts) and provides a validate-and-retry workflow that feeds errors
    back to the LLM for correction.

    Attributes:
        _topology_checks_available: Whether pythonocc topology checks work.
    """

    def __init__(self) -> None:
        """Initialize CadQueryValidator."""
        self._topology_checks_available = _OCC_AVAILABLE
        _log.info(
            "Initialized CadQueryValidator (occ_available=%s)",
            self._topology_checks_available,
        )

    def validate(self, step_path: str) -> dict:
        """Validate a STEP file with basic topology checks.

        Reads the STEP file, checks that it contains valid solids, and
        runs BRepCheck_Analyzer for structural validity.

        Args:
            step_path: Path to STEP file to validate.

        Returns:
            Dictionary with keys:
                - 'valid': Overall validity flag
                - 'compile_rate': Success fraction (0.0 or 1.0)
                - 'errors': List of error message strings
                - 'num_faces': Number of faces found
                - 'num_edges': Number of edges found
                - 'num_vertices': Number of vertices found
        """
        errors: List[str] = []
        num_faces = 0
        num_edges = 0
        num_vertices = 0

        step_file = Path(step_path)
        if not step_file.exists():
            return {
                "valid": False,
                "compile_rate": 0.0,
                "errors": [f"STEP file not found: {step_path}"],
                "num_faces": 0,
                "num_edges": 0,
                "num_vertices": 0,
            }

        if not self._topology_checks_available:
            # Without pythonocc we can only do basic file-level checks
            file_size = step_file.stat().st_size
            if file_size < 100:
                errors.append(
                    f"STEP file suspiciously small ({file_size} bytes)"
                )
                return {
                    "valid": False,
                    "compile_rate": 0.0,
                    "errors": errors,
                    "num_faces": 0,
                    "num_edges": 0,
                    "num_vertices": 0,
                }

            # Check for STEP header markers
            content = step_file.read_text(encoding="utf-8", errors="replace")
            has_header = "HEADER;" in content
            has_data = "DATA;" in content
            has_end = "END-ISO" in content or "ENDSEC;" in content

            if not (has_header and has_data and has_end):
                missing = []
                if not has_header:
                    missing.append("HEADER")
                if not has_data:
                    missing.append("DATA")
                if not has_end:
                    missing.append("END section")
                errors.append(
                    f"STEP file missing required sections: {', '.join(missing)}"
                )
                return {
                    "valid": False,
                    "compile_rate": 0.0,
                    "errors": errors,
                    "num_faces": 0,
                    "num_edges": 0,
                    "num_vertices": 0,
                }

            _log.info(
                "Basic STEP validation passed (no topology checks without pythonocc)"
            )
            return {
                "valid": True,
                "compile_rate": 1.0,
                "errors": [],
                "num_faces": 0,
                "num_edges": 0,
                "num_vertices": 0,
            }

        # Full topology validation with pythonocc
        try:
            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.IFSelect import IFSelect_RetDone
            from OCC.Core.BRepCheck import BRepCheck_Analyzer
            from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
            from OCC.Core.TopExp import TopExp_Explorer

            reader = STEPControl_Reader()
            status = reader.ReadFile(str(step_file))

            if status != IFSelect_RetDone:
                errors.append(f"Failed to read STEP file (status: {status})")
                return {
                    "valid": False,
                    "compile_rate": 0.0,
                    "errors": errors,
                    "num_faces": 0,
                    "num_edges": 0,
                    "num_vertices": 0,
                }

            reader.TransferRoots()
            shape = reader.OneShape()

            # Count topology elements
            face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while face_explorer.More():
                num_faces += 1
                face_explorer.Next()

            edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            while edge_explorer.More():
                num_edges += 1
                edge_explorer.Next()

            vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
            while vertex_explorer.More():
                num_vertices += 1
                vertex_explorer.Next()

            if num_faces == 0:
                errors.append("STEP file contains no faces")

            # Run BRepCheck_Analyzer
            analyzer = BRepCheck_Analyzer(shape)
            if not analyzer.IsValid():
                errors.append("BRepCheck_Analyzer reports invalid shape")

            is_valid = len(errors) == 0
            compile_rate = 1.0 if is_valid else 0.0

            _log.info(
                "STEP validation: valid=%s, faces=%d, edges=%d, vertices=%d",
                is_valid,
                num_faces,
                num_edges,
                num_vertices,
            )

            return {
                "valid": is_valid,
                "compile_rate": compile_rate,
                "errors": errors,
                "num_faces": num_faces,
                "num_edges": num_edges,
                "num_vertices": num_vertices,
            }

        except Exception as e:
            _log.error("Topology validation failed with exception: %s", e)
            errors.append(f"Validation exception: {e}")
            return {
                "valid": False,
                "compile_rate": 0.0,
                "errors": errors,
                "num_faces": num_faces,
                "num_edges": num_edges,
                "num_vertices": num_vertices,
            }

    def validate_and_retry(
        self,
        generator: CadQueryGenerator,
        executor: CadQueryExecutor,
        description: str,
        max_retries: int = 3,
    ) -> dict:
        """Generate, execute, validate, and retry on failure.

        Implements a closed-loop workflow:
        1. Generate CadQuery script from description
        2. Execute the script
        3. Export to STEP
        4. Validate the STEP file
        5. If validation fails, feed the error back to the LLM and retry

        Args:
            generator: CadQueryGenerator instance for script generation.
            executor: CadQueryExecutor instance for script execution.
            description: Natural language description of the desired part.
            max_retries: Maximum number of generate-validate cycles.

        Returns:
            Dictionary with keys:
                - 'success': Whether a valid model was produced
                - 'step_path': Path to the final STEP file (or None)
                - 'script': The final CadQuery script
                - 'validation': Validation result dict
                - 'attempts': Number of attempts made
                - 'errors': List of error messages from all attempts
        """
        import os

        from cadling.generation.codegen.prompts import load_repair_prompt

        repair_template = load_repair_prompt()
        all_errors: List[str] = []
        final_script = ""
        final_validation: dict = {}
        # Track temp STEP files so we can clean up intermediates on failure
        temp_step_paths: List[str] = []

        try:
            for attempt in range(1, max_retries + 1):
                _log.info(
                    "Validate-and-retry attempt %d/%d for: %s",
                    attempt,
                    max_retries,
                    description[:80],
                )

                # Step 1: Generate script
                try:
                    if attempt == 1:
                        script = generator.generate(description)
                    else:
                        # Build repair prompt with previous errors
                        error_context = "\n".join(
                            f"- {err}" for err in all_errors[-5:]
                        )
                        repair_prompt = repair_template.replace(
                            "{DESCRIPTION}", description
                        )
                        repair_prompt = repair_prompt.replace(
                            "{PREVIOUS_CODE}", final_script
                        )
                        repair_prompt = repair_prompt.replace(
                            "{ERRORS}", error_context
                        )

                        system_prompt = generator._build_system_prompt()
                        full_prompt = f"{system_prompt}\n\n---\n\n{repair_prompt}"
                        agent = generator._get_agent()
                        response = agent.ask(full_prompt, max_tokens=4096)
                        script = generator._extract_code(response)

                    final_script = script

                except Exception as e:
                    error_msg = f"Attempt {attempt}: Generation failed - {e}"
                    _log.error(error_msg)
                    all_errors.append(error_msg)
                    continue

                # Step 2: Execute
                exec_result = executor.execute(script)
                if not exec_result["success"]:
                    error_msg = (
                        f"Attempt {attempt}: Execution failed - "
                        f"{exec_result['error']}"
                    )
                    _log.warning(error_msg)
                    all_errors.append(error_msg)
                    continue

                # Step 3: Export to temp STEP file
                with tempfile.NamedTemporaryFile(
                    suffix=".step", delete=False
                ) as tmp:
                    step_path = tmp.name
                temp_step_paths.append(step_path)

                export_ok = executor.export_step(exec_result, step_path)
                if not export_ok:
                    error_msg = f"Attempt {attempt}: STEP export failed"
                    _log.warning(error_msg)
                    all_errors.append(error_msg)
                    continue

                # Step 4: Validate
                validation = self.validate(step_path)
                final_validation = validation

                if validation["valid"]:
                    _log.info(
                        "Valid model produced on attempt %d/%d",
                        attempt,
                        max_retries,
                    )
                    # Clean up intermediate temp files (not the successful one)
                    for tmp_path in temp_step_paths:
                        if tmp_path != step_path:
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                    return {
                        "success": True,
                        "step_path": step_path,
                        "script": final_script,
                        "validation": validation,
                        "attempts": attempt,
                        "errors": all_errors,
                    }

                # Validation failed -- collect errors for next attempt
                for err in validation["errors"]:
                    error_msg = f"Attempt {attempt}: Validation - {err}"
                    all_errors.append(error_msg)
                _log.warning(
                    "Validation failed on attempt %d: %s",
                    attempt,
                    validation["errors"],
                )

            _log.error(
                "All %d attempts failed for: %s", max_retries, description[:80]
            )
            return {
                "success": False,
                "step_path": None,
                "script": final_script,
                "validation": final_validation,
                "attempts": max_retries,
                "errors": all_errors,
            }
        finally:
            # Clean up all temp STEP files on failure (no valid result returned)
            # On success, the winning file is preserved and intermediates
            # were already cleaned above.
            # We only reach the finally after the for-loop exhaustion (failure)
            # or an unexpected exception, so clean everything.
            if not final_validation.get("valid", False):
                for tmp_path in temp_step_paths:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
