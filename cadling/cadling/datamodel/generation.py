"""Data models for CAD generation pipeline.

This module provides Pydantic v2 data models for the text/image-to-CAD generation
pipeline, including configuration, request/response types, and validation reports.

Classes:
    GenerationBackend: Enum of supported generation backends.
    GenerationConfig: Configuration for generation runs.
    GenerationRequest: Input request for CAD generation.
    ValidationReport: Topology/geometry validation results for generated output.
    GenerationResult: Result of a single generation attempt.

Example:
    from cadling.datamodel.generation import (
        GenerationBackend,
        GenerationConfig,
        GenerationRequest,
        GenerationResult,
    )

    config = GenerationConfig(
        backend=GenerationBackend.CODEGEN_CADQUERY,
        num_samples=3,
        temperature=0.8,
        validate_output=True,
    )
    request = GenerationRequest(
        text_prompt="Create a flanged bearing housing with 4 bolt holes",
        config=config,
        output_dir="generated_output",
    )
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)


class GenerationBackend(str, Enum):
    """Supported generation backends.

    Each backend represents a different approach to CAD model generation:

    - VAE: Variational autoencoder decoding latent vectors to command sequences.
    - DIFFUSION: Structured diffusion over B-Rep primitives with surface fitting.
    - CODEGEN_CADQUERY: LLM-based CadQuery Python code generation and execution.
    - CODEGEN_OPENSCAD: LLM-based OpenSCAD script generation and compilation.
    - VQVAE: Vector-quantized VAE with discrete codebook for CAD commands.
    """

    VAE = "vae"
    DIFFUSION = "diffusion"
    CODEGEN_CADQUERY = "codegen_cadquery"
    CODEGEN_OPENSCAD = "codegen_openscad"
    VQVAE = "vqvae"


class GenerationConfig(BaseModel):
    """Configuration for a CAD generation run.

    Controls which backend to use, sampling parameters, and post-generation
    validation settings.

    Attributes:
        backend: Generation backend to use.
        num_samples: Number of candidate samples to generate.
        temperature: Sampling temperature (higher = more diverse, lower = more
            deterministic). Applies to both code generation LLMs and latent
            sampling in VAE/diffusion backends.
        max_retries: Maximum retry attempts per sample on validation failure.
        validate_output: Whether to run topology validation on generated output.
        output_format: Desired output file format (step, stl, brep).
        latent_dim: Dimensionality of the latent space for VAE/VQVAE backends.
        max_seq_len: Maximum command sequence length for autoregressive decoding.
        seed: Optional random seed for reproducibility.
    """

    backend: GenerationBackend = GenerationBackend.CODEGEN_CADQUERY
    num_samples: int = 1
    temperature: float = 1.0
    max_retries: int = 3
    validate_output: bool = True
    output_format: str = "step"
    latent_dim: int = 256
    max_seq_len: int = 60
    seed: Optional[int] = None


class GenerationRequest(BaseModel):
    """Input request for CAD model generation.

    At least one of text_prompt or image_path must be provided. When both
    are provided the pipeline fuses text and image conditioning (e.g., image
    provides shape reference while text adds constraints).

    Attributes:
        text_prompt: Natural-language description of the desired CAD model.
        image_path: Path to a reference image (sketch, photo, rendering).
        reference_step_path: Optional path to a reference STEP file for
            style transfer or variation generation.
        config: Generation configuration.
        output_dir: Directory to write generated files to.
    """

    text_prompt: Optional[str] = None
    image_path: Optional[str] = None
    reference_step_path: Optional[str] = None
    config: GenerationConfig = Field(default_factory=GenerationConfig)
    output_dir: str = "generated_output"


class ValidationReport(BaseModel):
    """Topology and geometry validation report for a generated CAD model.

    Captures the results of post-generation validation checks including
    manifoldness, watertightness, Euler characteristic, and self-intersection
    detection.

    Attributes:
        is_valid: Overall validity flag (True if no critical errors).
        is_watertight: Whether the solid is closed (no boundary edges).
        euler_check: Whether the Euler characteristic matches expectations.
        manifold_check: Whether the model is manifold (each edge shared by
            at most 2 faces).
        self_intersection: Whether self-intersection detection passed (True
            means no self-intersections found).
        num_faces: Number of faces in the generated model.
        num_edges: Number of edges in the generated model.
        num_vertices: Number of vertices in the generated model.
        errors: List of critical error messages.
        warnings: List of non-critical warning messages.
    """

    is_valid: bool = False
    is_watertight: bool = False
    euler_check: bool = False
    manifold_check: bool = False
    self_intersection: bool = False
    num_faces: int = 0
    num_edges: int = 0
    num_vertices: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GenerationResult(BaseModel):
    """Result of a single CAD model generation attempt.

    Contains the generated output (file path, code, command sequence) along
    with validation results and metadata about the generation process.

    Attributes:
        success: Whether the generation completed successfully.
        output_path: Path to the generated CAD file on disk (if written).
        validation: Topology/geometry validation report.
        command_sequence: Decoded command token sequence (for VAE/VQVAE backends).
        generated_code: Generated source code (for codegen backends).
        generation_metadata: Additional metadata (timings, model info, etc.).
    """

    model_config = {"arbitrary_types_allowed": True}

    success: bool = False
    output_path: Optional[str] = None
    validation: ValidationReport = Field(default_factory=ValidationReport)
    command_sequence: Optional[list[int]] = None
    generated_code: Optional[str] = None
    generation_metadata: dict = Field(default_factory=dict)
