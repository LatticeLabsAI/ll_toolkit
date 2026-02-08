"""CAD generation pipeline - text/image to CAD model generation.

This package provides a unified pipeline for generating CAD models from
text descriptions or reference images. Multiple generation backends are
supported including code generation (CadQuery, OpenSCAD) and neural
approaches (VAE, diffusion, VQ-VAE).

Subpackages:
    - reconstruction: Output reconstruction for generated CAD models
    - codegen: Code generation backends (CadQuery, OpenSCAD)

Classes:
    GenerationPipeline: Main orchestrator for text/image-to-CAD generation.

Example:
    from cadling.generation import GenerationPipeline
    from cadling.datamodel.generation import (
        GenerationConfig,
        GenerationBackend,
        GenerationRequest,
    )

    config = GenerationConfig(
        backend=GenerationBackend.CODEGEN_CADQUERY,
        num_samples=1,
        validate_output=True,
    )
    request = GenerationRequest(
        text_prompt="A bracket with two mounting holes",
        config=config,
    )
    pipeline = GenerationPipeline(config=config)
    results = pipeline.generate(request)
"""

from __future__ import annotations

from cadling.generation.pipeline import GenerationPipeline

__all__ = [
    "GenerationPipeline",
]
