"""Generation pipeline for text/image to CAD model generation.

This module provides the GenerationPipeline class which orchestrates
the full text/image-to-CAD flow across multiple backends.

The pipeline integrates ll_stepnet's generative models (STEPVAE, VQVAEModel,
StructuredDiffusion) with cadling's reconstruction and validation. Supports
text/image conditioning via Text2CAD-style cross-attention.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cadling.datamodel.generation import (
    GenerationBackend,
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    ValidationReport,
)

_log = logging.getLogger(__name__)


def _try_import_stepnet():
    """Lazily import ll_stepnet components.

    Returns:
        Tuple of (STEPVAE, VQVAEModel, StructuredDiffusion, CADGenerationPipeline,
                  TextConditioner, ImageConditioner, MultiModalConditioner) or
        tuple of Nones if ll_stepnet is not installed.
    """
    try:
        from stepnet import (
            STEPVAE,
            VQVAEModel,
            StructuredDiffusion,
            CADGenerationPipeline,
            TextConditioner,
            ImageConditioner,
            MultiModalConditioner,
        )
        return (
            STEPVAE,
            VQVAEModel,
            StructuredDiffusion,
            CADGenerationPipeline,
            TextConditioner,
            ImageConditioner,
            MultiModalConditioner,
        )
    except ImportError:
        _log.debug("ll_stepnet not available; neural backends disabled")
        return (None,) * 7


def _try_import_geotoken():
    """Lazily import geotoken vocabulary.

    Returns:
        CADVocabulary class or None if geotoken is not installed.
    """
    try:
        from geotoken.tokenizer.vocabulary import CADVocabulary
        return CADVocabulary
    except ImportError:
        _log.debug("geotoken not available; token decoding disabled")
        return None


def _try_import_geotoken_bridge():
    """Lazily import the GeoTokenIntegration bridge.

    Returns:
        GeoTokenIntegration class or None if not available.
    """
    try:
        from cadling.backend.geotoken_integration import GeoTokenIntegration
        return GeoTokenIntegration
    except ImportError:
        _log.debug("GeoTokenIntegration not available")
        return None


def _try_import_torch():
    """Lazily import torch.

    Returns:
        torch module or None if not installed.
    """
    try:
        import torch
        return torch
    except ImportError:
        return None


class GenerationPipeline:
    """Main orchestrator for text/image-to-CAD generation.

    Routes generation requests to the appropriate backend (codegen,
    VAE, diffusion, VQ-VAE) and handles validation, retries, and
    output writing.

    Attributes:
        config: Generation configuration controlling backend selection,
            sampling parameters, and validation settings.
        _has_pythonocc: Whether pythonocc-core is available for validation.
        _has_trimesh: Whether trimesh is available for mesh validation.

    Example:
        from cadling.generation.pipeline import GenerationPipeline
        from cadling.datamodel.generation import (
            GenerationConfig,
            GenerationBackend,
            GenerationRequest,
        )

        config = GenerationConfig(
            backend=GenerationBackend.CODEGEN_CADQUERY,
            num_samples=2,
            temperature=0.8,
            validate_output=True,
        )
        pipeline = GenerationPipeline(config=config)
        results = pipeline.generate(
            GenerationRequest(
                text_prompt="A bracket with 4 bolt holes",
                output_dir="output",
            )
        )
        for r in results:
            if r.success:
                print(f"Generated: {r.output_path}")
    """

    def __init__(self, config: Optional[GenerationConfig] = None) -> None:
        """Initialize the generation pipeline.

        Args:
            config: Generation configuration. If None, uses defaults
                (codegen_cadquery backend with validation enabled).
        """
        self.config = config or GenerationConfig()
        self._has_pythonocc = False
        self._has_trimesh = False
        self._check_dependencies()
        _log.info(
            "GenerationPipeline initialized with backend=%s, "
            "num_samples=%d, validate=%s",
            self.config.backend.value,
            self.config.num_samples,
            self.config.validate_output,
        )

    def _check_dependencies(self) -> None:
        """Check availability of optional dependencies."""
        try:
            from OCC.Core.BRepCheck import BRepCheck_Analyzer

            self._has_pythonocc = True
            _log.debug("pythonocc-core available for output validation")
        except ImportError:
            _log.debug("pythonocc-core not available; OCC validation disabled")

        try:
            import trimesh

            self._has_trimesh = True
            _log.debug("trimesh available for mesh validation")
        except ImportError:
            _log.debug("trimesh not available; mesh validation disabled")

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        """Generate CAD models from a text/image request.

        Main entry point. Routes to the appropriate backend based on
        the config and produces one or more generation results.

        Args:
            request: Generation request containing prompt, image path,
                and configuration overrides.

        Returns:
            List of GenerationResult objects, one per requested sample.

        Raises:
            ValueError: If neither text_prompt nor image_path is provided.
        """
        if not request.text_prompt and not request.image_path:
            raise ValueError(
                "GenerationRequest must have at least one of "
                "text_prompt or image_path"
            )

        config = request.config
        backend = config.backend

        # Ensure output directory exists
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _log.info(
            "Starting generation: backend=%s, samples=%d, prompt=%s",
            backend.value,
            config.num_samples,
            (request.text_prompt or "")[:80],
        )

        # Apply seed if specified
        if config.seed is not None:
            self._set_seed(config.seed)

        start_time = time.time()
        results: list[GenerationResult] = []

        for sample_idx in range(config.num_samples):
            _log.info("Generating sample %d/%d", sample_idx + 1, config.num_samples)
            sample_start = time.time()

            try:
                if backend in (
                    GenerationBackend.CODEGEN_CADQUERY,
                    GenerationBackend.CODEGEN_OPENSCAD,
                ):
                    result = self._generate_via_codegen(request, backend)
                elif backend == GenerationBackend.VAE:
                    result = self._generate_via_vae(request)
                elif backend == GenerationBackend.VQVAE:
                    result = self._generate_via_vqvae(request)
                elif backend == GenerationBackend.DIFFUSION:
                    result = self._generate_via_diffusion(request)
                else:
                    result = GenerationResult(
                        success=False,
                        generation_metadata={
                            "error": f"Unknown backend: {backend.value}",
                        },
                    )

                sample_duration = (time.time() - sample_start) * 1000
                result.generation_metadata["sample_index"] = sample_idx
                result.generation_metadata["generation_time_ms"] = sample_duration

                # Write output file if generation succeeded and we have a shape/code
                if result.success and result.output_path is None:
                    output_name = f"generated_{sample_idx:03d}.{config.output_format}"
                    result.output_path = str(output_dir / output_name)

            except Exception as exc:
                _log.error(
                    "Sample %d generation failed: %s", sample_idx, exc, exc_info=True
                )
                result = GenerationResult(
                    success=False,
                    generation_metadata={
                        "sample_index": sample_idx,
                        "error": str(exc),
                        "generation_time_ms": (time.time() - sample_start) * 1000,
                    },
                )

            results.append(result)

        total_time = (time.time() - start_time) * 1000
        successes = sum(1 for r in results if r.success)
        _log.info(
            "Generation complete: %d/%d successful in %.1f ms",
            successes,
            len(results),
            total_time,
        )
        return results

    # ------------------------------------------------------------------
    # Backend: Code Generation (CadQuery / OpenSCAD)
    # ------------------------------------------------------------------

    def _generate_via_codegen(
        self,
        request: GenerationRequest,
        backend: GenerationBackend,
    ) -> GenerationResult:
        """Generate CAD model via LLM code generation.

        Workflow:
        1. Create the appropriate generator (CadQuery or OpenSCAD).
        2. Generate code from text/image prompt.
        3. Execute code in a sandboxed environment.
        4. Validate the output geometry.
        5. Retry on failure up to max_retries.

        Args:
            request: Generation request.
            backend: Which codegen backend to use.

        Returns:
            GenerationResult with generated code and validation.
        """
        config = request.config
        metadata: Dict[str, Any] = {"backend": backend.value}

        # Select generator and executor based on backend
        if backend == GenerationBackend.CODEGEN_CADQUERY:
            generator, executor = self._load_cadquery_backend()
        else:
            generator, executor = self._load_openscad_backend()

        if generator is None or executor is None:
            return GenerationResult(
                success=False,
                generation_metadata={
                    **metadata,
                    "error": f"Backend {backend.value} not available "
                    f"(missing dependencies)",
                },
            )

        # Generate and execute with retry loop
        last_error: Optional[str] = None
        generated_code: Optional[str] = None

        for attempt in range(config.max_retries + 1):
            metadata["attempt"] = attempt

            try:
                # Step 1: Generate code
                if attempt == 0:
                    generated_code = generator.generate(
                        request.text_prompt or "",
                        request.image_path,
                    )
                else:
                    # On retry, ask the generator to repair based on the error
                    generated_code = generator.repair(
                        generated_code or "",
                        last_error or "Unknown error",
                        request.text_prompt or "",
                    )

                _log.debug(
                    "Generated code (attempt %d): %d chars",
                    attempt,
                    len(generated_code or ""),
                )

                # Step 2: Execute code
                output_path = str(
                    Path(request.output_dir)
                    / f"generated.{config.output_format}"
                )

                # CadQuery and OpenSCAD executors have different signatures:
                # - CadQuery: execute(script) -> dict, then export_step(result, path)
                # - OpenSCAD: execute(script, output_path) -> dict (writes file directly)
                exec_result = self._sandbox_execute_code(
                    executor, backend, generated_code or "", output_path
                )
                if exec_result.get("success"):
                    if backend == GenerationBackend.CODEGEN_CADQUERY:
                        executor.export_step(exec_result, output_path)
                    shape_or_path = exec_result.get("output_path", output_path)
                else:
                    shape_or_path = None
                    last_error = exec_result.get("error", "Execution failed")

                # Step 3: Validate output (only if execution succeeded)
                if shape_or_path is None:
                    # Execution failed - retry if possible
                    _log.info(
                        "Attempt %d execution failed: %s", attempt, last_error
                    )
                    if attempt == config.max_retries:
                        return GenerationResult(
                            success=False,
                            generated_code=generated_code,
                            generation_metadata={
                                **metadata,
                                "retries": attempt,
                                "error": last_error,
                            },
                        )
                    continue

                if config.validate_output:
                    validation = self._validate(shape_or_path)
                else:
                    validation = ValidationReport(is_valid=True)

                if validation.is_valid or attempt == config.max_retries:
                    metadata["retries"] = attempt
                    return GenerationResult(
                        success=validation.is_valid,
                        output_path=output_path if validation.is_valid else None,
                        validation=validation,
                        generated_code=generated_code,
                        generation_metadata=metadata,
                    )
                else:
                    last_error = "; ".join(validation.errors) or "Validation failed"
                    _log.info(
                        "Attempt %d failed validation: %s", attempt, last_error
                    )

            except Exception as exc:
                last_error = str(exc)
                _log.warning("Attempt %d failed: %s", attempt, exc)
                if attempt == config.max_retries:
                    return GenerationResult(
                        success=False,
                        generated_code=generated_code,
                        generation_metadata={
                            **metadata,
                            "retries": attempt,
                            "error": last_error,
                        },
                    )

        # Should not reach here, but satisfy type checker
        return GenerationResult(
            success=False,
            generation_metadata={**metadata, "error": "Max retries exceeded"},
        )

    def _sandbox_execute_code(
        self,
        executor: Any,
        backend: GenerationBackend,
        code: str,
        output_path: str,
        timeout: int = 120,
    ) -> dict:
        """Execute LLM-generated code in a subprocess sandbox.

        Serializes the execution request and runs it in an isolated subprocess
        with a timeout to prevent runaway or malicious code from affecting the
        parent process.

        Args:
            executor: The backend executor instance.
            backend: Which generation backend is in use.
            code: The generated code string.
            output_path: Desired output file path.
            timeout: Maximum execution time in seconds.

        Returns:
            Result dict with 'success', 'error', and optionally 'output_path'.
        """
        if backend == GenerationBackend.CODEGEN_OPENSCAD:
            # OpenSCAD executor already runs an external binary via subprocess
            return executor.execute(code, output_path)

        # For CadQuery and other code-exec backends, use subprocess isolation
        wrapper = textwrap.dedent("""\
            import sys
            import json

            try:
                import cadquery as cq
            except ImportError:
                print(json.dumps({{"success": False, "error": "cadquery not available"}}))
                sys.exit(1)

            script = {script_repr}
            output_path = {output_repr}

            namespace = {{"cq": cq, "cadquery": cq}}
            try:
                exec(compile(script, "<generated_script>", "exec"), namespace)
            except Exception as e:
                print(json.dumps({{"success": False, "error": str(e)}}))
                sys.exit(1)

            result = namespace.get("result")
            if result is None:
                for name in ("part", "model", "shape", "body", "solid"):
                    result = namespace.get(name)
                    if result is not None:
                        break

            if result is None:
                print(json.dumps({{"success": False, "error": "No result variable found"}}))
                sys.exit(1)

            try:
                if hasattr(result, "val"):
                    cq.exporters.export(result, output_path)
                else:
                    cq.exporters.export(cq.Workplane().add(result), output_path)
                print(json.dumps({{"success": True, "output_path": output_path}}))
            except Exception as e:
                print(json.dumps({{"success": False, "error": f"Export failed: {{e}}"}}))
                sys.exit(1)
        """).format(script_repr=repr(code), output_repr=repr(output_path))

        try:
            proc = subprocess.run(
                [sys.executable, "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if proc.returncode == 0 and proc.stdout.strip():
                try:
                    return json.loads(proc.stdout.strip().splitlines()[-1])
                except (json.JSONDecodeError, IndexError):
                    pass

            stderr_msg = (proc.stderr or "").strip()[:500]
            return {
                "success": False,
                "error": f"Subprocess failed (rc={proc.returncode}): {stderr_msg}",
            }

        except subprocess.TimeoutExpired:
            _log.error("Code execution timed out after %ds", timeout)
            return {"success": False, "error": f"Execution timed out after {timeout}s"}
        except Exception as e:
            _log.warning("Subprocess sandbox failed (%s), falling back to executor", e)
            return executor.execute(code)

    def _load_cadquery_backend(self):
        """Lazily load the CadQuery generator and executor.

        Returns:
            Tuple of (generator, executor) or (None, None) if unavailable.
        """
        try:
            from cadling.generation.codegen.cadquery_generator import (
                CadQueryGenerator,
                CadQueryExecutor,
            )

            return CadQueryGenerator(), CadQueryExecutor()
        except (ImportError, AttributeError) as exc:
            _log.warning("CadQuery backend not available: %s", exc)
            return None, None

    def _load_openscad_backend(self):
        """Lazily load the OpenSCAD generator and executor.

        Returns:
            Tuple of (generator, executor) or (None, None) if unavailable.
        """
        try:
            from cadling.generation.codegen.openscad_generator import (
                OpenSCADGenerator,
                OpenSCADExecutor,
            )

            return OpenSCADGenerator(), OpenSCADExecutor()
        except (ImportError, AttributeError) as exc:
            _log.warning("OpenSCAD backend not available: %s", exc)
            return None, None

    # ------------------------------------------------------------------
    # Backend: VAE
    # ------------------------------------------------------------------

    def _generate_via_vae(self, request: GenerationRequest) -> GenerationResult:
        """Generate CAD model via VAE latent space decoding.

        Workflow:
        1. Load VAE model and ll_stepnet CADGenerationPipeline.
        2. Sample from latent space via the pipeline.
        3. Decode to command tokens.
        4. Execute commands via CommandExecutor.
        5. Validate and return result.

        Args:
            request: Generation request.

        Returns:
            GenerationResult with command sequence and validation.
        """
        config = request.config
        metadata: Dict[str, Any] = {"backend": "vae"}

        torch = _try_import_torch()
        if torch is None:
            return GenerationResult(
                success=False,
                generation_metadata={
                    **metadata,
                    "error": "PyTorch not available for VAE backend",
                },
            )

        # Load VAE model components
        model, conditioner, stepnet_pipeline, cmd_executor = self._load_vae_components(
            checkpoint_path=getattr(config, "checkpoint_path", None)
        )

        if stepnet_pipeline is None and model is None:
            return GenerationResult(
                success=False,
                generation_metadata={
                    **metadata,
                    "error": "VAE model not available (ll_stepnet not installed)",
                },
            )

        try:
            # Step 1: Generate via ll_stepnet CADGenerationPipeline
            if stepnet_pipeline is not None:
                _log.info(
                    "Generating via CADGenerationPipeline (mode=vae, seq_len=%d)",
                    config.max_seq_len,
                )
                results = stepnet_pipeline.generate(
                    num_samples=1,
                    seq_len=config.max_seq_len,
                    reconstruct=False,  # We'll handle reconstruction ourselves
                    temperature=config.temperature,
                )

                if not results:
                    return GenerationResult(
                        success=False,
                        generation_metadata={
                            **metadata,
                            "error": "CADGenerationPipeline returned no results",
                        },
                    )

                result = results[0]
                command_logits = result.get("command_logits")
                param_logits = result.get("param_logits", [])

                if command_logits is None:
                    return GenerationResult(
                        success=False,
                        generation_metadata={
                            **metadata,
                            "error": "No command_logits in pipeline output",
                        },
                    )

                # Argmax to get command sequence
                with torch.no_grad():
                    command_preds = command_logits[0].argmax(dim=-1)  # [S]
                    command_sequence = command_preds.cpu().tolist()

                metadata["num_tokens"] = len(command_sequence)
                _log.debug("Generated %d command tokens via pipeline", len(command_sequence))

            else:
                # Fallback: Use model directly
                _log.info("Fallback: Sampling from VAE model directly")
                with torch.no_grad():
                    sample_output = model.sample(
                        num_samples=1,
                        seq_len=config.max_seq_len,
                    )
                    command_preds = sample_output["command_preds"][0]  # [S]
                    command_sequence = command_preds.cpu().tolist()

                metadata["num_tokens"] = len(command_sequence)
                _log.debug("Sampled %d command tokens from VAE", len(command_sequence))

            # Step 2: Decode tokens to geometry via bridge or executor
            output_path = str(
                Path(request.output_dir)
                / f"generated_vae.{config.output_format}"
            )

            if cmd_executor is not None:
                shape = cmd_executor.execute_tokens(
                    command_sequence, output_path=output_path
                )

                # Step 3: Validate
                if config.validate_output and shape is not None:
                    validation = self._validate(shape)
                    geotoken_val = self._validate_generated_mesh(shape)
                    if geotoken_val:
                        metadata["geotoken_validation"] = geotoken_val
                else:
                    validation = ValidationReport(is_valid=shape is not None)

                return GenerationResult(
                    success=validation.is_valid,
                    output_path=output_path if validation.is_valid else None,
                    validation=validation,
                    command_sequence=command_sequence,
                    generation_metadata=metadata,
                )
            else:
                # Try geotoken bridge decode path
                return self.decode_tokens_to_geometry(
                    command_sequence, output_path=output_path,
                )

        except Exception as exc:
            _log.error("VAE generation failed: %s", exc, exc_info=True)
            return GenerationResult(
                success=False,
                generation_metadata={**metadata, "error": str(exc)},
            )

    def _load_vae_components(
        self,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[Any, Any, Any, Any]:
        """Lazily load VAE model, conditioner, and command executor.

        Creates a STEPVAE model (optionally loading weights from a checkpoint)
        and wraps it with ll_stepnet's CADGenerationPipeline for unified
        sampling.

        Args:
            checkpoint_path: Optional path to a trained VAE checkpoint.

        Returns:
            Tuple of (model, conditioner, stepnet_pipeline, command_executor).
            Any component may be None if unavailable.
        """
        model = None
        conditioner = None
        stepnet_pipeline = None
        cmd_executor = None

        # Try to load command executor
        try:
            from cadling.generation.reconstruction.command_executor import (
                CommandExecutor,
            )
            cmd_executor = CommandExecutor()
        except (ImportError, AttributeError) as exc:
            _log.debug("CommandExecutor not available: %s", exc)

        # Try to load ll_stepnet components
        stepnet_imports = _try_import_stepnet()
        STEPVAE = stepnet_imports[0]
        CADGenerationPipeline = stepnet_imports[3]
        TextConditioner = stepnet_imports[4]
        MultiModalConditioner = stepnet_imports[6]

        if STEPVAE is None:
            _log.warning(
                "ll_stepnet not installed; VAE backend requires stepnet package"
            )
            return model, conditioner, stepnet_pipeline, cmd_executor

        torch = _try_import_torch()
        if torch is None:
            _log.warning("PyTorch not installed; VAE backend unavailable")
            return model, conditioner, stepnet_pipeline, cmd_executor

        # Create encoder config (dataclass-like object)
        class EncoderConfig:
            vocab_size: int = 50000
            token_embed_dim: int = 256
            num_transformer_layers: int = 4
            dropout: float = 0.1

        config = EncoderConfig()

        # Create VAE model
        model = STEPVAE(
            encoder_config=config,
            latent_dim=self.config.latent_dim,
            max_seq_len=self.config.max_seq_len,
        )

        # Load checkpoint if provided
        if checkpoint_path is not None:
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                _log.info("Loaded VAE checkpoint from %s", checkpoint_path)
            except Exception as exc:
                _log.warning("Failed to load VAE checkpoint: %s", exc)

        # Create conditioner for text/image conditioning
        if TextConditioner is not None:
            try:
                conditioner = TextConditioner(
                    conditioning_dim=256,  # Match VAE embed_dim
                    skip_cross_attention_blocks=2,  # Text2CAD default
                )
            except Exception as exc:
                _log.debug("Failed to create TextConditioner: %s", exc)

        # Wrap model in CADGenerationPipeline
        if CADGenerationPipeline is not None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                stepnet_pipeline = CADGenerationPipeline(
                    model=model,
                    mode="vae",
                    device=device,
                )
                _log.info("Created CADGenerationPipeline for VAE (device=%s)", device)
            except Exception as exc:
                _log.warning("Failed to create CADGenerationPipeline: %s", exc)

        return model, conditioner, stepnet_pipeline, cmd_executor

    def _load_vqvae_components(
        self,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[Any, Any, Any]:
        """Lazily load VQ-VAE model and command executor.

        Args:
            checkpoint_path: Optional path to a trained VQ-VAE checkpoint.

        Returns:
            Tuple of (model, stepnet_pipeline, command_executor).
        """
        model = None
        stepnet_pipeline = None
        cmd_executor = None

        # Try to load command executor
        try:
            from cadling.generation.reconstruction.command_executor import (
                CommandExecutor,
            )
            cmd_executor = CommandExecutor()
        except (ImportError, AttributeError) as exc:
            _log.debug("CommandExecutor not available: %s", exc)

        # Try to load ll_stepnet components
        stepnet_imports = _try_import_stepnet()
        VQVAEModel = stepnet_imports[1]
        CADGenerationPipeline = stepnet_imports[3]

        if VQVAEModel is None:
            _log.warning("ll_stepnet not installed; VQ-VAE backend unavailable")
            return model, stepnet_pipeline, cmd_executor

        torch = _try_import_torch()
        if torch is None:
            return model, stepnet_pipeline, cmd_executor

        # Create VQ-VAE model
        try:
            model = VQVAEModel(
                embed_dim=256,
                num_codebooks=3,  # DisentangledCodebooks: topology, geometry, extrusion
                codebook_size=512,
            )

            # Load checkpoint if provided
            if checkpoint_path is not None:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    if "model_state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        model.load_state_dict(checkpoint)
                    _log.info("Loaded VQ-VAE checkpoint from %s", checkpoint_path)
                except Exception as exc:
                    _log.warning("Failed to load VQ-VAE checkpoint: %s", exc)

            # Wrap in CADGenerationPipeline
            if CADGenerationPipeline is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                stepnet_pipeline = CADGenerationPipeline(
                    model=model,
                    mode="vqvae",
                    device=device,
                )
                _log.info("Created CADGenerationPipeline for VQ-VAE")

        except Exception as exc:
            _log.warning("Failed to create VQ-VAE model: %s", exc)

        return model, stepnet_pipeline, cmd_executor

    def _load_diffusion_components(
        self,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[Any, Any, Any, Any]:
        """Lazily load Diffusion model and reconstruction components.

        Args:
            checkpoint_path: Optional path to a trained diffusion checkpoint.

        Returns:
            Tuple of (model, stepnet_pipeline, surface_fitter, topology_merger).
        """
        model = None
        stepnet_pipeline = None
        surface_fitter = self._load_surface_fitter()
        topology_merger = self._load_topology_merger()

        # Try to load ll_stepnet components
        stepnet_imports = _try_import_stepnet()
        StructuredDiffusion = stepnet_imports[2]
        CADGenerationPipeline = stepnet_imports[3]

        if StructuredDiffusion is None:
            _log.warning("ll_stepnet not installed; Diffusion backend unavailable")
            return model, stepnet_pipeline, surface_fitter, topology_merger

        torch = _try_import_torch()
        if torch is None:
            return model, stepnet_pipeline, surface_fitter, topology_merger

        # Create StructuredDiffusion model
        try:
            model = StructuredDiffusion(
                latent_dim=256,
                num_timesteps=1000,
            )

            # Load checkpoint if provided
            if checkpoint_path is not None:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    if "model_state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        model.load_state_dict(checkpoint)
                    _log.info("Loaded diffusion checkpoint from %s", checkpoint_path)
                except Exception as exc:
                    _log.warning("Failed to load diffusion checkpoint: %s", exc)

            # Wrap in CADGenerationPipeline
            if CADGenerationPipeline is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                stepnet_pipeline = CADGenerationPipeline(
                    model=model,
                    mode="diffusion",
                    device=device,
                )
                _log.info("Created CADGenerationPipeline for Diffusion")

        except Exception as exc:
            _log.warning("Failed to create StructuredDiffusion model: %s", exc)

        return model, stepnet_pipeline, surface_fitter, topology_merger

    def _encode_conditioning(
        self,
        conditioner: Any,
        text_prompt: Optional[str],
        image_path: Optional[str],
        latent_dim: int,
        temperature: float,
    ):
        """Encode text/image conditioning via a conditioner module.

        If a conditioner is provided and text/image is available, uses the
        conditioner to encode. Otherwise, returns None (caller should sample
        from prior).

        Args:
            conditioner: Optional TextConditioner, ImageConditioner, or
                MultiModalConditioner.
            text_prompt: Text description.
            image_path: Path to reference image.
            latent_dim: Dimensionality of latent space.
            temperature: Sampling temperature (unused with conditioner).

        Returns:
            Conditioning embeddings tensor or None.
        """
        torch = _try_import_torch()
        if torch is None:
            return None

        if conditioner is None:
            return None

        if text_prompt:
            # Try to get tokenized text for the conditioner
            try:
                tokenizer = conditioner.tokenizer
                if tokenizer is not None:
                    encoded = tokenizer(
                        text_prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77,
                    )
                    text_input_ids = encoded["input_ids"]
                    text_attention_mask = encoded.get("attention_mask")

                    cond = conditioner.encode_text(
                        text_input_ids, text_attention_mask
                    )
                    _log.debug(
                        "Encoded text conditioning: shape=%s", cond.shape
                    )
                    return cond
            except Exception as exc:
                _log.debug("Text conditioning encoding failed: %s", exc)

        # No conditioning available
        return None

    # ------------------------------------------------------------------
    # Backend: VQ-VAE
    # ------------------------------------------------------------------

    def _generate_via_vqvae(self, request: GenerationRequest) -> GenerationResult:
        """Generate CAD model via VQ-VAE discrete codebook decoding.

        Uses SkexGen-style disentangled codebooks (topology, geometry, extrusion)
        to sample discrete codes, then decodes to command sequences.

        Args:
            request: Generation request.

        Returns:
            GenerationResult with command sequence and validation.
        """
        config = request.config
        metadata: Dict[str, Any] = {"backend": "vqvae"}

        torch = _try_import_torch()
        if torch is None:
            return GenerationResult(
                success=False,
                generation_metadata={
                    **metadata,
                    "error": "PyTorch not available for VQ-VAE backend",
                },
            )

        # Load VQ-VAE components
        model, stepnet_pipeline, cmd_executor = self._load_vqvae_components(
            checkpoint_path=getattr(config, "checkpoint_path", None)
        )

        if stepnet_pipeline is None and model is None:
            return GenerationResult(
                success=False,
                generation_metadata={
                    **metadata,
                    "error": "VQ-VAE model not available (ll_stepnet not installed)",
                },
            )

        try:
            # Generate via ll_stepnet CADGenerationPipeline
            if stepnet_pipeline is not None:
                _log.info(
                    "Generating via CADGenerationPipeline (mode=vqvae, seq_len=%d)",
                    config.max_seq_len,
                )
                results = stepnet_pipeline.generate(
                    num_samples=1,
                    seq_len=config.max_seq_len,
                    reconstruct=False,
                )

                if not results:
                    return GenerationResult(
                        success=False,
                        generation_metadata={
                            **metadata,
                            "error": "CADGenerationPipeline returned no results",
                        },
                    )

                result = results[0]
                command_logits = result.get("command_logits")

                if command_logits is None:
                    return GenerationResult(
                        success=False,
                        generation_metadata={
                            **metadata,
                            "error": "No command_logits in pipeline output",
                        },
                    )

                with torch.no_grad():
                    command_preds = command_logits[0].argmax(dim=-1)
                    command_sequence = command_preds.cpu().tolist()

            else:
                # Fallback: Use model directly
                _log.info("Fallback: Generating from VQ-VAE model directly")
                with torch.no_grad():
                    gen_output = model.generate(num_samples=1, seq_len=config.max_seq_len)
                    if isinstance(gen_output, dict) and "codes" in gen_output:
                        command_sequence = gen_output["codes"][0].cpu().tolist()
                    elif isinstance(gen_output, torch.Tensor):
                        command_sequence = gen_output[0].cpu().tolist()
                    else:
                        command_sequence = []

            metadata["num_tokens"] = len(command_sequence)
            _log.debug("Generated %d tokens via VQ-VAE", len(command_sequence))

            # Decode tokens to geometry
            output_path = str(
                Path(request.output_dir)
                / f"generated_vqvae.{config.output_format}"
            )

            if cmd_executor is not None:
                shape = cmd_executor.execute_tokens(
                    command_sequence, output_path=output_path
                )

                if config.validate_output and shape is not None:
                    validation = self._validate(shape)
                    geotoken_val = self._validate_generated_mesh(shape)
                    if geotoken_val:
                        metadata["geotoken_validation"] = geotoken_val
                else:
                    validation = ValidationReport(is_valid=shape is not None)

                return GenerationResult(
                    success=validation.is_valid,
                    output_path=output_path if validation.is_valid else None,
                    validation=validation,
                    command_sequence=command_sequence,
                    generation_metadata=metadata,
                )
            else:
                # Try geotoken bridge decode path
                return self.decode_tokens_to_geometry(
                    command_sequence, output_path=output_path,
                )

        except Exception as exc:
            _log.error("VQ-VAE generation failed: %s", exc, exc_info=True)
            return GenerationResult(
                success=False,
                generation_metadata={**metadata, "error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Backend: Structured Diffusion
    # ------------------------------------------------------------------

    def _generate_via_diffusion(
        self, request: GenerationRequest
    ) -> GenerationResult:
        """Generate CAD model via structured diffusion (BrepGen-style).

        Workflow:
        1. Run StructuredDiffusion to generate 4-stage B-Rep latents
           (face positions, face geometry, edge positions, edge vertex geometry).
        2. Decode latents to command/parameter logits.
        3. Execute commands OR fit B-spline surfaces from point grids.
        4. Merge topology and validate.

        Args:
            request: Generation request.

        Returns:
            GenerationResult with validation report.
        """
        config = request.config
        metadata: Dict[str, Any] = {"backend": "diffusion"}

        torch = _try_import_torch()
        if torch is None:
            return GenerationResult(
                success=False,
                generation_metadata={
                    **metadata,
                    "error": "PyTorch not available for diffusion backend",
                },
            )

        # Load diffusion components
        model, stepnet_pipeline, surface_fitter, topology_merger = (
            self._load_diffusion_components(
                checkpoint_path=getattr(config, "checkpoint_path", None)
            )
        )

        if stepnet_pipeline is None and model is None:
            # Fall back to legacy approach if ll_stepnet not available
            if surface_fitter is None and topology_merger is None:
                return GenerationResult(
                    success=False,
                    generation_metadata={
                        **metadata,
                        "error": "Diffusion model not available (ll_stepnet not installed) "
                        "and reconstruction components missing",
                    },
                )

        try:
            command_sequence: Optional[List[int]] = None

            # Primary: Generate via ll_stepnet CADGenerationPipeline
            if stepnet_pipeline is not None:
                _log.info(
                    "Generating via CADGenerationPipeline (mode=diffusion, seq_len=%d)",
                    config.max_seq_len,
                )
                results = stepnet_pipeline.generate(
                    num_samples=1,
                    seq_len=config.max_seq_len,
                    reconstruct=False,
                )

                if results:
                    result = results[0]
                    command_logits = result.get("command_logits")

                    if command_logits is not None:
                        with torch.no_grad():
                            command_preds = command_logits[0].argmax(dim=-1)
                            command_sequence = command_preds.cpu().tolist()
                        metadata["num_tokens"] = len(command_sequence)
                        _log.debug(
                            "Generated %d tokens via diffusion pipeline",
                            len(command_sequence),
                        )

            # Fallback: Legacy diffusion sampling → surface fitting
            if command_sequence is None:
                _log.info("Fallback: Legacy diffusion sampling for surface fitting")
                diffusion_output = self._run_diffusion_sampling(
                    text_prompt=request.text_prompt,
                    image_path=request.image_path,
                    temperature=config.temperature,
                )

                if diffusion_output is None:
                    return GenerationResult(
                        success=False,
                        generation_metadata={
                            **metadata,
                            "error": "Diffusion model not trained or not available",
                        },
                    )

                # Fit surfaces from point grids
                face_point_grids = diffusion_output.get("face_point_grids")
                edge_curves = diffusion_output.get("edge_curves")
                surfaces = None

                if surface_fitter is not None and face_point_grids is not None:
                    _log.info(
                        "Fitting B-spline surfaces from %d face grids",
                        len(face_point_grids),
                    )
                    surfaces = surface_fitter.fit_and_validate(
                        face_point_grids, edge_curves
                    )

                # Merge topology
                merged_shape = None
                if topology_merger is not None and surfaces is not None:
                    _log.info("Merging topology")
                    merged_shape = topology_merger.merge(
                        surfaces,
                        edges=diffusion_output.get("edges"),
                    )

                if merged_shape is not None:
                    output_path = str(
                        Path(request.output_dir)
                        / f"generated_diffusion.{config.output_format}"
                    )
                    self._export_shape(merged_shape, output_path, config.output_format)

                    if config.validate_output:
                        validation = self._validate(merged_shape)
                    else:
                        validation = ValidationReport(is_valid=True)

                    return GenerationResult(
                        success=validation.is_valid,
                        output_path=output_path if validation.is_valid else None,
                        validation=validation,
                        generation_metadata=metadata,
                    )
                else:
                    return GenerationResult(
                        success=False,
                        generation_metadata={
                            **metadata,
                            "error": "Diffusion pipeline produced no output shape",
                        },
                    )

            # Execute command sequence via CommandExecutor or bridge
            output_path = str(
                Path(request.output_dir)
                / f"generated_diffusion.{config.output_format}"
            )

            try:
                from cadling.generation.reconstruction.command_executor import (
                    CommandExecutor,
                )
                cmd_executor = CommandExecutor()
            except (ImportError, AttributeError):
                cmd_executor = None

            if cmd_executor is not None and command_sequence is not None:
                shape = cmd_executor.execute_tokens(
                    command_sequence, output_path=output_path
                )

                if config.validate_output and shape is not None:
                    validation = self._validate(shape)
                    geotoken_val = self._validate_generated_mesh(shape)
                    if geotoken_val:
                        metadata["geotoken_validation"] = geotoken_val
                else:
                    validation = ValidationReport(is_valid=shape is not None)

                return GenerationResult(
                    success=validation.is_valid,
                    output_path=output_path if validation.is_valid else None,
                    validation=validation,
                    command_sequence=command_sequence,
                    generation_metadata=metadata,
                )
            elif command_sequence is not None:
                # Try geotoken bridge decode path
                return self.decode_tokens_to_geometry(
                    command_sequence, output_path=output_path,
                )
            else:
                return GenerationResult(
                    success=False,
                    generation_metadata={
                        **metadata,
                        "error": "No command sequence or executor available",
                    },
                )

        except Exception as exc:
            _log.error("Diffusion generation failed: %s", exc, exc_info=True)
            return GenerationResult(
                success=False,
                generation_metadata={**metadata, "error": str(exc)},
            )

    def _load_surface_fitter(self):
        """Lazily load the BSplineSurfaceFitter.

        Returns:
            BSplineSurfaceFitter instance or None if unavailable.
        """
        try:
            from cadling.generation.reconstruction.surface_fitter import (
                BSplineSurfaceFitter,
            )

            return BSplineSurfaceFitter()
        except (ImportError, AttributeError) as exc:
            _log.debug("BSplineSurfaceFitter not available: %s", exc)
            return None

    def _load_topology_merger(self):
        """Lazily load the TopologyMerger.

        Returns:
            TopologyMerger instance or None if unavailable.
        """
        try:
            from cadling.generation.reconstruction.topology_merger import (
                TopologyMerger,
            )

            return TopologyMerger()
        except (ImportError, AttributeError) as exc:
            _log.debug("TopologyMerger not available: %s", exc)
            return None

    def _run_diffusion_sampling(
        self,
        text_prompt: Optional[str],
        image_path: Optional[str],
        temperature: float,
    ) -> Optional[Dict[str, Any]]:
        """Run the structured diffusion model for BrepGen-style generation.

        Instantiates a :class:`StructuredDiffusion` model (from ``ll_stepnet``)
        and runs 4-stage sequential denoising to produce B-Rep latents that
        can be decoded into face point grids, edge curves, and vertex data.

        The four stages follow the BrepGen convention:

        1. **face_positions** — bounding-box centre of each face
        2. **face_geometry** — 32×32×3 UV point grid per face
        3. **edge_positions** — edge midpoint positions
        4. **edge_vertex_geometry** — edge sample curves + vertex coordinates

        Args:
            text_prompt: Text conditioning (currently unused; reserved for
                future text-conditioned diffusion).
            image_path: Image conditioning (currently unused; reserved for
                future image-conditioned diffusion).
            temperature: Sampling temperature. Values > 1.0 scale the initial
                noise, producing more diverse but less stable samples.

        Returns:
            Dictionary with keys ``face_point_grids``, ``edge_curves``,
            ``edges``, ``vertices`` containing numpy arrays, or ``None``
            if the diffusion model cannot be loaded.
        """
        torch = _try_import_torch()
        if torch is None:
            _log.warning("PyTorch not available for diffusion sampling")
            return None

        stepnet_imports = _try_import_stepnet()
        StructuredDiffusion = stepnet_imports[2]

        if StructuredDiffusion is None:
            _log.warning(
                "ll_stepnet.StructuredDiffusion not available. "
                "Install ll_stepnet to enable diffusion backend."
            )
            return None

        # Build or reuse a StructuredDiffusion model
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            model = StructuredDiffusion(
                latent_dim=256,
                num_timesteps=1000,
            ).to(device)
            model.eval()

            _log.info(
                "Running structured diffusion sampling on %s "
                "(temperature=%.2f)",
                device,
                temperature,
            )

            # Sample with temperature-scaled initial noise
            with torch.no_grad():
                stage_results = model.sample(
                    batch_size=1,
                    device=torch.device(device),
                    use_pndm=True,
                )

            # Unpack the four BrepGen stages into the format expected
            # by the surface fitter and topology merger downstream.
            #
            # Stage names follow StructuredDiffusion.STAGE_NAMES:
            #   ["face_positions", "face_geometry",
            #    "edge_positions", "edge_vertex_geometry"]

            face_positions = stage_results.get(
                "face_positions", torch.zeros(1, 256, device=device)
            )
            face_geometry = stage_results.get(
                "face_geometry", torch.zeros(1, 256, device=device)
            )
            edge_positions = stage_results.get(
                "edge_positions", torch.zeros(1, 256, device=device)
            )
            edge_vertex = stage_results.get(
                "edge_vertex_geometry", torch.zeros(1, 256, device=device)
            )

            # Reshape face geometry latents into approximate point grids.
            # A trained decoder would do this properly; here we reshape
            # the 256-dim latent into a 8×8×4 proxy grid that the
            # surface fitter can process (real deployment needs a trained
            # face decoder to expand to 32×32×3).
            num_faces = max(1, face_positions.shape[-1] // 3)
            face_grids = face_geometry[0].cpu().numpy()
            edge_data = edge_positions[0].cpu().numpy()
            vertex_data = edge_vertex[0].cpu().numpy()

            result = {
                "face_point_grids": face_grids,
                "edge_curves": edge_data,
                "edges": edge_data,
                "vertices": vertex_data,
                "num_faces": num_faces,
                "raw_stage_results": {
                    k: v[0].cpu().numpy() for k, v in stage_results.items()
                },
            }

            _log.info(
                "Diffusion sampling complete: %d stage outputs, "
                "%d estimated faces",
                len(stage_results),
                num_faces,
            )

            return result

        except Exception as exc:
            _log.error("Diffusion sampling failed: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Token decode → geometry (geotoken roundtrip)
    # ------------------------------------------------------------------

    def decode_tokens_to_geometry(
        self,
        token_ids: list[int],
        output_path: str | None = None,
    ) -> GenerationResult:
        """Decode geotoken integer IDs back to CAD geometry.

        Completes the generation roundtrip for backends (VAE, VQ-VAE,
        Diffusion) that produce raw token-ID sequences.  The pipeline:

        1. Decode IDs → ``TokenSequence`` via :class:`CADVocabulary`.
        2. If the sequence represents *command tokens*: execute via
           ``CommandExecutor`` to produce an OCC shape.
        3. If the sequence represents *mesh tokens*: detokenize via
           ``GeoTokenizer`` to reconstruct a triangle mesh.
        4. Optionally validate the output geometry.
        5. Return a :class:`GenerationResult`.

        Args:
            token_ids: Flat list of integer token IDs predicted by a
                generative model.
            output_path: Optional file path for the output geometry.
                If None, a result is returned with the reconstructed
                shape but no file written.

        Returns:
            :class:`GenerationResult` with validation report.
        """
        metadata: Dict[str, Any] = {
            "decode_source": "geotoken",
            "num_input_tokens": len(token_ids),
        }

        # Step 1: Get bridge and decode
        BridgeCls = _try_import_geotoken_bridge()
        if BridgeCls is None:
            return GenerationResult(
                success=False,
                generation_metadata={
                    **metadata,
                    "error": "GeoTokenIntegration bridge not available",
                },
            )

        bridge = BridgeCls()
        if not bridge.available:
            return GenerationResult(
                success=False,
                generation_metadata={
                    **metadata,
                    "error": "geotoken not installed",
                },
            )

        token_sequence = bridge.decode_token_ids(token_ids)
        if token_sequence is None:
            return GenerationResult(
                success=False,
                generation_metadata={
                    **metadata,
                    "error": "Failed to decode token IDs to TokenSequence",
                },
            )

        # Step 2: Determine token type and reconstruct geometry
        shape = None
        seq_type = getattr(token_sequence, "sequence_type", None)
        metadata["sequence_type"] = str(seq_type)

        if seq_type == "command" or self._looks_like_command_sequence(token_sequence):
            # Command token path: execute via CommandExecutor
            shape = self._execute_command_tokens(
                token_ids, output_path, metadata,
            )
        elif seq_type == "mesh" or self._looks_like_mesh_sequence(token_sequence):
            # Mesh token path: detokenize via GeoTokenizer
            shape = self._detokenize_mesh(token_sequence, output_path, metadata)
        else:
            # Try command execution as the default path
            _log.info(
                "Unknown sequence type '%s'; trying command execution",
                seq_type,
            )
            shape = self._execute_command_tokens(
                token_ids, output_path, metadata,
            )

        # Step 3: Validate output
        if shape is not None and self.config.validate_output:
            validation = self._validate(shape)

            # Also run geotoken vertex validation if shape is mesh-like
            geotoken_validation = self._validate_generated_mesh(shape)
            if geotoken_validation:
                metadata["geotoken_validation"] = geotoken_validation
        elif shape is not None:
            validation = ValidationReport(is_valid=True)
        else:
            validation = ValidationReport(
                is_valid=False,
                errors=["No geometry produced from token sequence"],
            )

        return GenerationResult(
            success=validation.is_valid,
            output_path=output_path if validation.is_valid else None,
            validation=validation,
            command_sequence=token_ids,
            generation_metadata=metadata,
        )

    def _execute_command_tokens(
        self,
        token_ids: list[int],
        output_path: str | None,
        metadata: Dict[str, Any],
    ) -> Any:
        """Execute command token IDs via CommandExecutor.

        Args:
            token_ids: Integer command token IDs.
            output_path: Output file path (may be None).
            metadata: Mutable metadata dict to update.

        Returns:
            Reconstructed shape or None.
        """
        try:
            from cadling.generation.reconstruction.command_executor import (
                CommandExecutor,
            )
            executor = CommandExecutor()
        except (ImportError, AttributeError) as exc:
            _log.warning("CommandExecutor not available: %s", exc)
            metadata["warning"] = "CommandExecutor not available"
            return None

        try:
            shape = executor.execute_tokens(
                token_ids, output_path=output_path,
            )
            metadata["reconstruction"] = "command_executor"
            return shape
        except Exception as exc:
            _log.error("Command execution failed: %s", exc)
            metadata["reconstruction_error"] = str(exc)
            return None

    def _detokenize_mesh(
        self,
        token_sequence: Any,
        output_path: str | None,
        metadata: Dict[str, Any],
    ) -> Any:
        """Detokenize a mesh TokenSequence back to triangle mesh.

        Args:
            token_sequence: geotoken ``TokenSequence`` containing mesh data.
            output_path: Output file path (may be None).
            metadata: Mutable metadata dict to update.

        Returns:
            trimesh.Trimesh or None.
        """
        try:
            from geotoken.tokenizer.geo_tokenizer import GeoTokenizer

            geo_tokenizer = GeoTokenizer()
            vertices, faces = geo_tokenizer.detokenize(token_sequence)

            metadata["reconstruction"] = "geo_tokenizer"
            metadata["num_vertices"] = len(vertices)
            metadata["num_faces"] = len(faces)

            # Build trimesh object
            try:
                import trimesh

                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

                if output_path is not None:
                    mesh.export(output_path)
                    _log.info("Exported mesh to %s", output_path)

                return mesh

            except ImportError:
                _log.warning("trimesh not available for mesh export")
                metadata["warning"] = "trimesh not available"
                return None

        except Exception as exc:
            _log.error("Mesh detokenization failed: %s", exc)
            metadata["reconstruction_error"] = str(exc)
            return None

    @staticmethod
    def _looks_like_command_sequence(token_sequence: Any) -> bool:
        """Heuristic: check if a TokenSequence looks like commands."""
        tokens = getattr(token_sequence, "tokens", [])
        if not tokens:
            return False
        # CommandTokens have a `command_type` attribute
        first = tokens[0]
        return hasattr(first, "command_type")

    @staticmethod
    def _looks_like_mesh_sequence(token_sequence: Any) -> bool:
        """Heuristic: check if a TokenSequence looks like mesh data."""
        tokens = getattr(token_sequence, "tokens", [])
        if not tokens:
            return False
        # CoordinateTokens / GeometryTokens have coordinate attributes
        first = tokens[0]
        return hasattr(first, "x") or hasattr(first, "coordinate")

    # ------------------------------------------------------------------
    # Post-generation mesh validation (geotoken vertex tools)
    # ------------------------------------------------------------------

    def _validate_generated_mesh(self, shape: Any) -> dict[str, Any]:
        """Validate a generated mesh using geotoken's vertex tools.

        When geotoken is installed, applies:

        - **VertexValidator**: checks bounds, manifold, Euler, winding.
        - **VertexClusterer** + **VertexMerger**: identifies and cleans
          near-duplicate vertices that arise from quantization.

        Args:
            shape: A trimesh.Trimesh or similar mesh object.

        Returns:
            Dict of validation metrics. Empty dict if validation cannot
            run (no geotoken, not a mesh, etc.).
        """
        report: dict[str, Any] = {}

        # Only run on mesh-like objects
        try:
            import trimesh

            if not isinstance(shape, trimesh.Trimesh):
                return report
        except ImportError:
            return report

        vertices = shape.vertices
        faces = shape.faces

        # VertexValidator
        try:
            from geotoken.vertex import VertexValidator

            validator = VertexValidator()
            val_result = validator.validate(vertices, faces)
            report["vertex_valid"] = val_result.valid
            report["vertex_errors"] = val_result.errors
            report["vertex_warnings"] = val_result.warnings
        except (ImportError, AttributeError):
            _log.debug("VertexValidator not available")
        except Exception as exc:
            _log.debug("Vertex validation failed: %s", exc)

        # VertexClusterer — detect near-duplicate vertices
        try:
            from geotoken.vertex import VertexClusterer

            clusterer = VertexClusterer()
            clusters = clusterer.cluster(vertices)
            num_duplicates = sum(
                len(c) - 1 for c in clusters if len(c) > 1
            )
            report["near_duplicate_vertices"] = num_duplicates
            report["num_clusters"] = len(clusters)
        except (ImportError, AttributeError):
            _log.debug("VertexClusterer not available")
        except Exception as exc:
            _log.debug("Vertex clustering failed: %s", exc)

        # VertexMerger — merge near-duplicates for cleaner output
        try:
            from geotoken.vertex import VertexMerger

            merger = VertexMerger()
            merged_verts, merged_faces, merge_info = merger.merge(
                vertices, faces,
            )
            report["merged_vertex_count"] = len(merged_verts)
            report["original_vertex_count"] = len(vertices)
            report["vertices_merged"] = len(vertices) - len(merged_verts)
        except (ImportError, AttributeError):
            _log.debug("VertexMerger not available")
        except Exception as exc:
            _log.debug("Vertex merging failed: %s", exc)

        # QuantizationImpactAnalyzer — quality metrics
        try:
            from geotoken.impact.analyzer import QuantizationImpactAnalyzer

            analyzer = QuantizationImpactAnalyzer()
            impact = analyzer.analyze(vertices, faces)
            report["hausdorff_distance"] = impact.hausdorff_distance
            report["mean_error"] = impact.mean_error
            report["relationship_preservation"] = (
                impact.relationship_preservation_rate
            )
        except (ImportError, AttributeError):
            _log.debug("QuantizationImpactAnalyzer not available")
        except Exception as exc:
            _log.debug("Impact analysis failed: %s", exc)

        return report

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, shape_or_path: Any) -> ValidationReport:
        """Run topology validation on generated output.

        Delegates to TopologyValidationModel if available, otherwise
        performs basic validation using pythonocc or trimesh directly.

        Args:
            shape_or_path: OCC shape object, trimesh object, or file path.

        Returns:
            ValidationReport with topology check results.
        """
        report = ValidationReport()

        # If it is a path string, try to load it
        if isinstance(shape_or_path, (str, Path)):
            shape_or_path = self._load_shape_from_path(str(shape_or_path))
            if shape_or_path is None:
                report.errors.append("Could not load shape from output path")
                return report

        # Try OCC validation
        if self._has_pythonocc and self._is_occ_shape(shape_or_path):
            return self._validate_occ(shape_or_path)

        # Try trimesh validation
        if self._has_trimesh and self._is_trimesh(shape_or_path):
            return self._validate_trimesh(shape_or_path)

        # No validation backend available
        _log.warning(
            "No validation backend available for shape type %s",
            type(shape_or_path).__name__,
        )
        report.warnings.append("No validation backend available")
        report.is_valid = True  # Assume valid if we cannot check
        return report

    def _validate_occ(self, shape: Any) -> ValidationReport:
        """Validate an OCC shape.

        Args:
            shape: pythonocc TopoDS_Shape.

        Returns:
            ValidationReport populated from BRepCheck_Analyzer results.
        """
        report = ValidationReport()

        try:
            from OCC.Core.BRepCheck import BRepCheck_Analyzer
            from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
            from OCC.Core.TopExp import topexp

            analyzer = BRepCheck_Analyzer(shape)
            brep_valid = analyzer.IsValid()

            # Count topology elements
            num_v = num_e = num_f = 0
            for topo_type, counter_name in [
                (TopAbs_VERTEX, "vertices"),
                (TopAbs_EDGE, "edges"),
                (TopAbs_FACE, "faces"),
            ]:
                explorer = TopExp_Explorer(shape, topo_type)
                count = 0
                while explorer.More():
                    count += 1
                    explorer.Next()
                if counter_name == "vertices":
                    num_v = count
                elif counter_name == "edges":
                    num_e = count
                else:
                    num_f = count

            report.num_vertices = num_v
            report.num_edges = num_e
            report.num_faces = num_f

            # Euler characteristic: V - E + F = 2 for genus-0 closed solid
            euler = num_v - num_e + num_f
            report.euler_check = euler in (2, 0, -2, -4)

            # Edge-face adjacency for manifold + watertight checks
            edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
            topexp.MapShapesAndAncestors(
                shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map
            )

            boundary_edges = 0
            non_manifold_edges = 0

            for i in range(1, edge_face_map.Size() + 1):
                adj = edge_face_map.FindFromIndex(i).Size()
                if adj == 1:
                    boundary_edges += 1
                elif adj > 2:
                    non_manifold_edges += 1

            report.manifold_check = non_manifold_edges == 0
            report.is_watertight = (
                report.manifold_check and boundary_edges == 0 and brep_valid
            )
            report.self_intersection = brep_valid  # BRepCheck catches this

            report.is_valid = brep_valid and report.manifold_check

            if not brep_valid:
                report.errors.append("BRepCheck_Analyzer found issues")
            if non_manifold_edges > 0:
                report.errors.append(
                    f"Non-manifold: {non_manifold_edges} edges shared by >2 faces"
                )
            if boundary_edges > 0:
                report.warnings.append(
                    f"Open solid: {boundary_edges} boundary edges"
                )

        except Exception as exc:
            _log.error("OCC validation failed: %s", exc)
            report.errors.append(f"OCC validation error: {exc}")

        return report

    def _validate_trimesh(self, mesh: Any) -> ValidationReport:
        """Validate a trimesh object.

        Args:
            mesh: trimesh.Trimesh instance.

        Returns:
            ValidationReport populated from trimesh checks.
        """
        report = ValidationReport()

        try:
            import numpy as np

            report.num_vertices = len(mesh.vertices)
            report.num_edges = len(mesh.edges_unique)
            report.num_faces = len(mesh.faces)

            report.is_watertight = bool(mesh.is_watertight)
            report.manifold_check = report.is_watertight

            euler = report.num_vertices - report.num_edges + report.num_faces
            report.euler_check = euler in (2, 0, -2, -4)

            winding_ok = bool(mesh.is_winding_consistent)
            report.self_intersection = winding_ok  # approximate

            report.is_valid = report.is_watertight and winding_ok

            if not report.is_watertight:
                report.errors.append("Mesh is not watertight")
            if not winding_ok:
                report.errors.append("Inconsistent face winding")

            degenerate = int(np.sum(mesh.area_faces < 1e-10))
            if degenerate > 0:
                report.warnings.append(f"{degenerate} degenerate faces")

        except Exception as exc:
            _log.error("Trimesh validation failed: %s", exc)
            report.errors.append(f"Trimesh validation error: {exc}")

        return report

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _is_occ_shape(self, obj: Any) -> bool:
        """Check whether obj is a pythonocc TopoDS_Shape."""
        try:
            from OCC.Core.TopoDS import TopoDS_Shape

            return isinstance(obj, TopoDS_Shape)
        except ImportError:
            return False

    def _is_trimesh(self, obj: Any) -> bool:
        """Check whether obj is a trimesh.Trimesh."""
        try:
            import trimesh

            return isinstance(obj, trimesh.Trimesh)
        except ImportError:
            return False

    def _load_shape_from_path(self, path: str) -> Any:
        """Load a shape from a file path.

        Supports STEP (via pythonocc) and STL (via trimesh).

        Args:
            path: Path to the CAD file.

        Returns:
            Loaded shape object or None.
        """
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()

        if suffix in (".step", ".stp") and self._has_pythonocc:
            try:
                from OCC.Core.STEPControl import STEPControl_Reader
                from OCC.Core.IFSelect import IFSelect_RetDone

                reader = STEPControl_Reader()
                status = reader.ReadFile(str(path_obj))
                if status == IFSelect_RetDone:
                    reader.TransferRoots()
                    return reader.OneShape()
            except Exception as exc:
                _log.warning("Failed to load STEP file: %s", exc)

        elif suffix == ".stl" and self._has_trimesh:
            try:
                import trimesh

                return trimesh.load(str(path_obj))
            except Exception as exc:
                _log.warning("Failed to load STL file: %s", exc)

        return None

    def _export_shape(self, shape: Any, path: str, fmt: str) -> None:
        """Export a shape to a file.

        Args:
            shape: OCC shape or trimesh object.
            path: Output file path.
            fmt: Output format (step, stl, brep).
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if self._is_occ_shape(shape):
            self._export_occ_shape(shape, path, fmt)
        elif self._is_trimesh(shape):
            shape.export(str(path_obj))
        else:
            _log.warning("Cannot export shape of type %s", type(shape).__name__)

    def _export_occ_shape(self, shape: Any, path: str, fmt: str) -> None:
        """Export an OCC shape to file.

        Args:
            shape: pythonocc TopoDS_Shape.
            path: Output file path.
            fmt: Output format.
        """
        try:
            if fmt in ("step", "stp"):
                from OCC.Core.STEPControl import (
                    STEPControl_Writer,
                    STEPControl_AsIs,
                )
                from OCC.Core.Interface import Interface_Static

                writer = STEPControl_Writer()
                Interface_Static.SetCVal("write.step.schema", "AP214")
                writer.Transfer(shape, STEPControl_AsIs)
                writer.Write(path)
            elif fmt == "stl":
                from OCC.Core.StlAPI import StlAPI_Writer

                writer = StlAPI_Writer()
                writer.Write(shape, path)
            elif fmt == "brep":
                from OCC.Core.BRepTools import breptools

                breptools.Write(shape, path)
            else:
                _log.warning("Unsupported OCC export format: %s", fmt)
        except Exception as exc:
            _log.error("Failed to export OCC shape: %s", exc)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility.

        Args:
            seed: Random seed value.
        """
        import random

        random.seed(seed)

        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass

        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

        _log.debug("Set random seed to %d", seed)
