"""Backend configuration options.

This module provides configuration options for CAD backends, adapted from
docling's backend options but extended for CAD-specific features like
rendering and ll_stepnet integration.

Classes:
    BackendOptions: Base backend options.
    STEPBackendOptions: Options for STEP backend.
    STLBackendOptions: Options for STL backend.

Example:
    options = STEPBackendOptions(
        enable_rendering=True,
        enable_ll_stepnet=True,
        render_resolution=2048
    )
"""

from __future__ import annotations

from typing import Optional

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
