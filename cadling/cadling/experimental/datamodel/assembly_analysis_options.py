"""Assembly analysis and processing options.

This module provides configuration options for processing multi-part CAD
assemblies, including component detection, mate extraction, and BOM generation.

Classes:
    MateType: Enumeration of assembly mate types.
    AssemblyAnalysisOptions: Options for assembly processing.

Example:
    options = AssemblyAnalysisOptions(
        detect_components=True,
        extract_mates=True,
        generate_bom=True
    )
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, List, Optional

from pydantic import Field

from cadling.datamodel.pipeline_options import PipelineOptions


class MateType(str, Enum):
    """Enumeration of assembly mate/constraint types."""

    COINCIDENT = "coincident"
    CONCENTRIC = "concentric"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    TANGENT = "tangent"
    DISTANCE = "distance"
    ANGLE = "angle"
    FASTENER = "fastener"
    WELD = "weld"
    UNKNOWN = "unknown"


class AssemblyAnalysisOptions(PipelineOptions):
    """Configuration options for assembly analysis and processing.

    This options class configures the processing of multi-part CAD assemblies,
    including component detection, mate/constraint extraction, Bill of Materials
    (BOM) generation, and interference checking.

    Attributes:
        kind: Discriminator for option type.
        detect_components: Whether to detect individual components in assembly.
        extract_mates: Whether to extract mate/constraint relationships.
        generate_bom: Whether to generate Bill of Materials.
        check_interference: Whether to check for component interference.
        max_components: Maximum number of components to process.
        mate_types_to_extract: List of mate types to extract.
        bom_include_metadata: Whether to include component metadata in BOM.
        bom_include_properties: Whether to include physical properties in BOM.
        interference_tolerance: Tolerance for interference detection (mm).
        process_subassemblies: Whether to recursively process subassemblies.
        component_naming_strategy: Strategy for naming components.
    """

    kind: ClassVar[str] = "cadling_experimental_assembly"

    detect_components: bool = Field(
        default=True,
        description="Whether to detect individual components in assembly"
    )
    extract_mates: bool = Field(
        default=True,
        description="Whether to extract mate/constraint relationships between components"
    )
    generate_bom: bool = Field(
        default=True,
        description="Whether to generate Bill of Materials (BOM)"
    )
    check_interference: bool = Field(
        default=False,
        description="Whether to check for component interference (computationally expensive)"
    )
    max_components: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum number of components to process"
    )
    mate_types_to_extract: List[MateType] = Field(
        default_factory=lambda: [
            MateType.COINCIDENT,
            MateType.CONCENTRIC,
            MateType.PARALLEL,
            MateType.PERPENDICULAR,
            MateType.FASTENER,
        ],
        description="List of mate types to extract"
    )
    bom_include_metadata: bool = Field(
        default=True,
        description="Whether to include component metadata (name, ID, material) in BOM"
    )
    bom_include_properties: bool = Field(
        default=True,
        description="Whether to include physical properties (volume, mass, bbox) in BOM"
    )
    interference_tolerance: float = Field(
        default=0.01,
        ge=0.0,
        description="Tolerance for interference detection in millimeters"
    )
    process_subassemblies: bool = Field(
        default=True,
        description="Whether to recursively process subassemblies"
    )
    component_naming_strategy: str = Field(
        default="hierarchical",
        description="Strategy for naming components (hierarchical, flat, preserve_original)"
    )
    extract_fasteners: bool = Field(
        default=True,
        description="Whether to identify and extract fastener components (bolts, screws, etc.)"
    )
    group_identical_parts: bool = Field(
        default=True,
        description="Whether to group identical parts in BOM for quantity tracking"
    )
