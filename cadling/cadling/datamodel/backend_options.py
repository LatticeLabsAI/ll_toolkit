"""Backend configuration options.

This module provides configuration options for CAD backends, adapted from
docling's backend options but extended for CAD-specific features like
rendering and ll_stepnet integration.

Classes:
    BackendOptions: Base backend options.
    STEPBackendOptions: Options for STEP backend.
    STLBackendOptions: Options for STL backend.
    BRepBackendOptions: Options for BRep backend.
    DXFBackendOptions: Options for DXF 2D drawing backend.
    PDFBackendOptions: Options for PDF engineering drawing backend.

Example:
    options = STEPBackendOptions(
        enable_rendering=True,
        enable_ll_stepnet=True,
        render_resolution=2048
    )
"""

from __future__ import annotations

from typing import Literal, Optional, List

from pydantic import BaseModel


class BackendOptions(BaseModel):
    """Base configuration options for CAD backends.

    Attributes:
        verbose: Enable verbose logging.
        strict: Strict parsing mode (fail on errors vs. best-effort).
    """

    verbose: bool = False
    strict: bool = True


class STEPBackendOptions(BackendOptions):
    """Options for STEP backend.

    Attributes:
        enable_rendering: Whether to enable rendering support.
        enable_ll_stepnet: Whether to use ll_stepnet for parsing.
        render_resolution: Default resolution for rendering.
        parse_header: Whether to parse STEP header information.
        extract_features: Whether to extract geometric features.
        build_topology: Whether to build topology graph.
    """

    enable_rendering: bool = False
    enable_ll_stepnet: bool = True
    render_resolution: int = 1024
    parse_header: bool = True
    extract_features: bool = True
    build_topology: bool = True


class STLBackendOptions(BackendOptions):
    """Options for STL backend.

    Attributes:
        enable_rendering: Whether to enable rendering support.
        validate_mesh: Whether to validate mesh (manifold, watertight).
        compute_properties: Whether to compute mesh properties (volume, area).
        render_resolution: Default resolution for rendering.
    """

    enable_rendering: bool = False
    validate_mesh: bool = True
    compute_properties: bool = True
    render_resolution: int = 1024


class BRepBackendOptions(BackendOptions):
    """Options for BRep backend.

    Attributes:
        enable_rendering: Whether to enable rendering support.
        render_resolution: Default resolution for rendering.
        extract_topology: Whether to extract topological information.
    """

    enable_rendering: bool = True
    render_resolution: int = 1024
    extract_topology: bool = True


class DXFBackendOptions(BackendOptions):
    """Options for DXF 2D drawing backend.

    Controls how DXF files are parsed, including which layers to extract,
    whether to extract dimension annotations, and tolerance for closing
    gaps in polylines.

    Attributes:
        extract_dimensions: Whether to extract dimension annotations.
        merge_layers: Whether to merge all layers into a single profile.
        tolerance: Distance tolerance (mm) for closing polyline gaps.
        target_layers: Specific layers to extract (None = all layers).
        inline_blocks: Whether to inline block references into modelspace.
    """

    extract_dimensions: bool = True
    merge_layers: bool = False
    tolerance: float = 0.01  # mm, for closing gaps in polylines
    target_layers: Optional[List[str]] = None  # None = all layers
    inline_blocks: bool = True


class PDFBackendOptions(BackendOptions):
    """Options for PDF engineering drawing backend.

    Controls how PDF files are processed, with support for auto-detecting
    whether a PDF contains vector geometry (extractable) or raster images
    (requires VLM/OCR fallback).

    Attributes:
        extraction_mode: Strategy for extracting geometry from PDF pages.
            - "auto": Detect vector vs raster per page (default).
            - "vector": Force vector path extraction.
            - "raster": Force raster rendering + VLM/OCR.
        dpi: Resolution for raster fallback rendering.
        page_range: Pages to process as (start, end) tuple (None = all pages).
        extract_dimensions: Whether to extract dimension text annotations.
        ocr_language: Language for OCR text extraction (default: English).
    """

    extraction_mode: Literal["auto", "vector", "raster"] = "auto"
    dpi: int = 300
    page_range: Optional[tuple] = None  # None = all pages
    extract_dimensions: bool = True
    ocr_language: str = "en"
