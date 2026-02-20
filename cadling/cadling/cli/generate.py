"""CLI commands for CAD model generation.

This module provides Click-based CLI commands for generating CAD models
from text descriptions or reference images using the code generation
backend (CadQuery or OpenSCAD).

Commands:
    generate: Generate a CAD model from text and/or image input

Usage:
    cadling generate --from-text "description" --output part.step
    cadling generate --from-text "..." --backend cadquery --output part.step
    cadling generate --from-text "..." --backend openscad --output part.stl
    cadling generate --from-image render.png --output part.step
    cadling generate --from-text "..." --validate --max-retries 3
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click

_log = logging.getLogger(__name__)


@click.group()
def generate_group():
    """CAD model generation commands.

    Generate 3D CAD models from text descriptions or reference images
    using LLM-powered code generation backends.
    """
    pass


@generate_group.command("generate")
@click.option(
    "--from-text",
    "text_prompt",
    type=str,
    default=None,
    help="Natural language description of the desired CAD model.",
)
@click.option(
    "--from-image",
    "image_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to a reference image (sketch, photo, rendering).",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=True,
    help="Output file path (e.g., part.step, part.stl).",
)
@click.option(
    "--backend",
    type=click.Choice(["cadquery", "openscad"], case_sensitive=False),
    default="cadquery",
    help="Code generation backend (default: cadquery).",
)
@click.option(
    "--model",
    "-m",
    "model_name",
    type=str,
    default="gpt-4",
    help="LLM model name for code generation (default: gpt-4).",
)
@click.option(
    "--provider",
    type=click.Choice(
        ["openai", "anthropic", "vllm", "ollama", "openai_compatible", "mlx"],
        case_sensitive=False,
    ),
    default="openai",
    help="LLM API provider (default: openai). Use 'mlx' for Apple Silicon native.",
)
@click.option(
    "--validate/--no-validate",
    default=False,
    help="Run topology validation on generated output.",
)
@click.option(
    "--max-retries",
    type=int,
    default=3,
    help="Maximum retry attempts on validation failure (default: 3).",
)
@click.option(
    "--timeout",
    type=int,
    default=30,
    help="Script execution timeout in seconds (default: 30).",
)
@click.option(
    "--save-script",
    type=click.Path(),
    default=None,
    help="Save the generated script to this path.",
)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    default=None,
    help="API key (default: from OPENAI_API_KEY env var).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging output.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Debug mode: dump full LLM response and auto-save scripts on failure.",
)
def generate_cmd(
    text_prompt: Optional[str],
    image_path: Optional[str],
    output_path: str,
    backend: str,
    model_name: str,
    provider: str,
    validate: bool,
    max_retries: int,
    timeout: int,
    save_script: Optional[str],
    api_key: Optional[str],
    verbose: bool,
    debug: bool,
) -> None:
    """Generate a CAD model from text description or reference image.

    Uses LLM-powered code generation to create CadQuery or OpenSCAD scripts,
    then executes them to produce STEP or STL output files.

    At least one of --from-text or --from-image must be provided.

    Examples:

        \b
        # Generate from text description
        cadling generate --from-text "A flanged bearing housing" -o bearing.step

        \b
        # Generate with OpenSCAD backend
        cadling generate --from-text "A box with holes" --backend openscad -o box.stl

        \b
        # Generate from text + reference image
        cadling generate --from-text "A bracket" --from-image sketch.png -o bracket.step

        \b
        # Generate with validation and retries
        cadling generate --from-text "A gear" --validate --max-retries 5 -o gear.step

        \b
        # Use Anthropic Claude model
        cadling generate --from-text "A pipe flange" --provider anthropic --model claude-3-opus -o flange.step
    """
    # Configure logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Validate inputs
    if text_prompt is None and image_path is None:
        click.echo(
            "Error: At least one of --from-text or --from-image must be provided.",
            err=True,
        )
        sys.exit(1)

    # Build description from text and/or image context
    description = text_prompt or ""
    if text_prompt is None and image_path is not None:
        description = (
            "Generate a 3D CAD model that matches the provided reference image. "
            "Infer the geometry, dimensions, and features from the image."
        )

    output_file = Path(output_path)
    output_suffix = output_file.suffix.lower()

    click.echo(f"Backend: {backend}", err=True)
    click.echo(f"Model: {model_name} ({provider})", err=True)
    click.echo(f"Output: {output_path}", err=True)
    if text_prompt:
        click.echo(f"Description: {text_prompt[:100]}...", err=True)
    if image_path:
        click.echo(f"Reference image: {image_path}", err=True)

    start_time = time.time()

    try:
        if backend.lower() == "cadquery":
            _generate_cadquery(
                description=description,
                image_path=image_path,
                output_path=output_path,
                model_name=model_name,
                provider=provider,
                validate=validate,
                max_retries=max_retries,
                timeout=timeout,
                save_script=save_script,
                debug=debug,
            )
        elif backend.lower() == "openscad":
            _generate_openscad(
                description=description,
                output_path=output_path,
                model_name=model_name,
                provider=provider,
                save_script=save_script,
            )
        else:
            click.echo(f"Error: Unknown backend '{backend}'", err=True)
            sys.exit(1)

        elapsed = time.time() - start_time
        click.echo(f"\nGeneration completed in {elapsed:.1f}s", err=True)

    except Exception as e:
        elapsed = time.time() - start_time
        click.echo(f"\nError after {elapsed:.1f}s: {e}", err=True)
        _log.exception("Generation failed")
        sys.exit(1)


def _generate_cadquery(
    description: str,
    image_path: Optional[str],
    output_path: str,
    model_name: str,
    provider: str,
    validate: bool,
    max_retries: int,
    timeout: int,
    save_script: Optional[str],
    debug: bool = False,
) -> None:
    """Run CadQuery generation pipeline.

    Args:
        description: Part description text.
        image_path: Optional reference image path.
        output_path: Output STEP file path.
        model_name: LLM model name.
        provider: LLM API provider.
        validate: Whether to run validation loop.
        max_retries: Max retry attempts.
        timeout: Script execution timeout.
        save_script: Optional path to save generated script.
        debug: Enable debug mode for verbose LLM response output.
    """
    from cadling.generation.codegen.cadquery_generator import (
        CadQueryGenerator,
        CadQueryExecutor,
        CadQueryValidator,
    )

    generator = CadQueryGenerator(
        model_name=model_name,
        api_provider=provider,
        max_retries=max_retries,
    )
    executor = CadQueryExecutor(timeout=timeout)

    if validate:
        # Use validate-and-retry loop
        click.echo("\nRunning validate-and-retry loop...", err=True)
        validator = CadQueryValidator()
        result = validator.validate_and_retry(
            generator=generator,
            executor=executor,
            description=description,
            max_retries=max_retries,
        )

        if result["success"]:
            # Move the validated STEP file to the output path
            import shutil

            step_path = result["step_path"]
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(step_path, str(output_file))

            click.echo(
                f"\nValid model generated after {result['attempts']} attempt(s).",
                err=True,
            )
            click.echo(f"STEP file: {output_path}", err=True)

            validation = result["validation"]
            click.echo(
                f"Topology: {validation.get('num_faces', '?')} faces, "
                f"{validation.get('num_edges', '?')} edges, "
                f"{validation.get('num_vertices', '?')} vertices",
                err=True,
            )

            # Save the final script if requested
            if save_script:
                script_path = Path(save_script)
                script_path.parent.mkdir(parents=True, exist_ok=True)
                script_path.write_text(result["script"], encoding="utf-8")
                click.echo(f"Script saved: {save_script}", err=True)
        else:
            click.echo(
                f"\nFailed to generate valid model after {result['attempts']} "
                f"attempt(s).",
                err=True,
            )
            if result["errors"]:
                click.echo("Errors:", err=True)
                for err in result["errors"][-5:]:
                    click.echo(f"  - {err}", err=True)

            # Save the last script even on failure for debugging
            if save_script and result.get("script"):
                script_path = Path(save_script)
                script_path.parent.mkdir(parents=True, exist_ok=True)
                script_path.write_text(result["script"], encoding="utf-8")
                click.echo(
                    f"Last attempted script saved: {save_script}", err=True
                )

            sys.exit(1)

    else:
        # Single attempt without validation
        click.echo("\nGenerating CadQuery script...", err=True)
        script = generator.generate(
            description=description,
            reference_image_path=image_path,
        )
        click.echo(f"Script generated ({len(script)} chars)", err=True)

        # Debug: show raw script content
        if debug:
            click.echo("\n--- DEBUG: Generated Script ---", err=True)
            click.echo(script, err=True)
            click.echo("--- END DEBUG ---\n", err=True)

        # Save script if requested
        if save_script:
            script_path = Path(save_script)
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script, encoding="utf-8")
            click.echo(f"Script saved: {save_script}", err=True)

        # Execute
        click.echo("Executing script...", err=True)
        exec_result = executor.execute(script)

        if not exec_result["success"]:
            click.echo(
                f"Execution failed: {exec_result['error']}", err=True
            )
            # Auto-save script on failure for debugging (if not already saved)
            if not save_script:
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".py",
                    prefix="cadquery_failed_",
                    delete=False,
                ) as tmp:
                    tmp.write(script)
                    failed_script_path = tmp.name
                click.echo(
                    f"Failed script auto-saved to: {failed_script_path}",
                    err=True,
                )
            sys.exit(1)

        click.echo(
            f"Script executed in {exec_result.get('execution_time', 0):.1f}s",
            err=True,
        )

        # Export
        click.echo("Exporting STEP file...", err=True)
        export_ok = executor.export_step(exec_result, output_path)

        if not export_ok:
            click.echo("STEP export failed.", err=True)
            # Auto-save script on export failure
            if not save_script:
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".py",
                    prefix="cadquery_export_failed_",
                    delete=False,
                ) as tmp:
                    tmp.write(script)
                    failed_script_path = tmp.name
                click.echo(
                    f"Script auto-saved to: {failed_script_path}",
                    err=True,
                )
            sys.exit(1)

        click.echo(f"STEP file: {output_path}", err=True)


def _generate_openscad(
    description: str,
    output_path: str,
    model_name: str,
    provider: str,
    save_script: Optional[str],
) -> None:
    """Run OpenSCAD generation pipeline.

    Args:
        description: Part description text.
        output_path: Output file path (typically .stl).
        model_name: LLM model name.
        provider: LLM API provider.
        save_script: Optional path to save generated script.
    """
    from cadling.generation.codegen.openscad_generator import (
        OpenSCADGenerator,
        OpenSCADExecutor,
    )

    generator = OpenSCADGenerator(
        model_name=model_name,
        api_provider=provider,
    )
    executor = OpenSCADExecutor()

    # Generate script
    click.echo("\nGenerating OpenSCAD script...", err=True)
    script = generator.generate(description)
    click.echo(f"Script generated ({len(script)} chars)", err=True)

    # Save script if requested
    if save_script:
        script_path = Path(save_script)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script, encoding="utf-8")
        click.echo(f"Script saved: {save_script}", err=True)

    # Determine output format
    output_file = Path(output_path)
    output_suffix = output_file.suffix.lower()

    if output_suffix == ".step" or output_suffix == ".stp":
        # Need to generate STL first, then convert
        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=".stl", delete=False
        ) as tmp:
            stl_path = tmp.name

        click.echo("Executing OpenSCAD -> STL...", err=True)
        exec_result = executor.execute(script, stl_path)

        if not exec_result["success"]:
            click.echo(
                f"OpenSCAD execution failed: {exec_result['error']}", err=True
            )
            sys.exit(1)

        click.echo(
            f"STL generated in {exec_result.get('execution_time', 0):.1f}s",
            err=True,
        )

        # Convert STL to STEP
        click.echo("Converting STL -> STEP...", err=True)
        convert_ok = executor.convert_to_step(stl_path, output_path)

        if not convert_ok:
            click.echo(
                "STL-to-STEP conversion failed. "
                "Ensure pythonocc-core is installed.",
                err=True,
            )
            # Save the STL as fallback
            stl_fallback = output_file.with_suffix(".stl")
            import shutil

            shutil.move(stl_path, str(stl_fallback))
            click.echo(
                f"STL saved as fallback: {stl_fallback}", err=True
            )
            sys.exit(1)

        click.echo(f"STEP file: {output_path}", err=True)

        # Clean up temp STL
        try:
            Path(stl_path).unlink(missing_ok=True)
        except Exception:
            pass

    else:
        # Direct output (STL, OFF, AMF, 3MF, etc.)
        click.echo(f"Executing OpenSCAD -> {output_suffix}...", err=True)
        exec_result = executor.execute(script, output_path)

        if not exec_result["success"]:
            click.echo(
                f"OpenSCAD execution failed: {exec_result['error']}", err=True
            )
            sys.exit(1)

        click.echo(
            f"Output generated in {exec_result.get('execution_time', 0):.1f}s",
            err=True,
        )
        click.echo(f"Output file: {output_path}", err=True)
