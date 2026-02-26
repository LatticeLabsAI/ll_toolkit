"""Configuration for geometric tokenization and quantization."""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Literal, Optional


_PRECISION_BITS_MAP: dict[str, int] = {"draft": 6, "standard": 8, "precision": 10}


class PrecisionTier(str, Enum):
    """Quantization precision tiers.

    | Tier      | Bits | Levels | Use case           |
    |-----------|------|--------|--------------------|
    | DRAFT     | 6    | 64     | Fast preview       |
    | STANDARD  | 8    | 256    | Balanced (default) |
    | PRECISION | 10   | 1024   | High fidelity      |
    """
    DRAFT = "draft"
    STANDARD = "standard"
    PRECISION = "precision"

    @property
    def bits(self) -> int:
        return _PRECISION_BITS_MAP[self.value]

    @property
    def levels(self) -> int:
        return 2 ** self.bits


@dataclass
class NormalizationConfig:
    """Configuration for geometry normalization."""
    center: bool = True                    # Center to origin
    uniform_scale: bool = True             # Scale to unit bounding cube
    preserve_aspect_ratio: bool = True     # Don't distort proportions
    target_range: tuple[float, float] = (0.0, 1.0)  # Output coordinate range


@dataclass
class AdaptiveBitAllocationConfig:
    """Configuration for adaptive bit allocation.

    Bit semantics:
        min_bits: Absolute floor — no vertex ever gets fewer bits than this.
        base_bits: Starting allocation for flat / low-complexity regions.
        max_bits: Absolute ceiling — no vertex ever gets more bits than this.

    The invariant ``min_bits <= base_bits <= max_bits`` must always hold.
    """
    base_bits: int = 8                     # Starting bits for flat regions
    max_additional_bits: int = 4           # Max extra bits for complex regions
    min_bits: int = 4                      # Absolute minimum
    max_bits: int = 16                     # Absolute maximum
    curvature_weight: float = 0.7          # Weight for curvature in complexity
    density_weight: float = 0.3            # Weight for feature density
    percentile_low: float = 10.0           # Below this percentile -> base_bits
    percentile_high: float = 90.0          # Above this percentile -> base + max_additional

    def __post_init__(self) -> None:
        """Validate that min_bits <= base_bits <= max_bits."""
        if not (self.min_bits <= self.base_bits <= self.max_bits):
            raise ValueError(
                f"Bit allocation invariant violated: min_bits ({self.min_bits}) "
                f"<= base_bits ({self.base_bits}) <= max_bits ({self.max_bits}) "
                f"must hold."
            )
        if self.percentile_low >= self.percentile_high:
            raise ValueError(
                f"percentile_low ({self.percentile_low}) must be strictly less "
                f"than percentile_high ({self.percentile_high})."
            )


@dataclass
class QuantizationConfig:
    """Main quantization configuration."""
    tier: PrecisionTier = PrecisionTier.STANDARD
    adaptive: bool = True                  # Use adaptive vs uniform
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    bit_allocation: AdaptiveBitAllocationConfig = field(default_factory=AdaptiveBitAllocationConfig)
    minimum_feature_threshold: float = 0.05  # Min distance to prevent collapse
    float_tolerance: float = 1e-10         # Numerical tolerance


@dataclass
class CommandTokenizationConfig:
    """Configuration for CAD command sequence tokenization.

    Controls how parametric CAD construction history is converted to
    fixed-length command token sequences for transformer consumption.

    Args:
        max_sequence_length: Target command count (DeepCAD uses 60).
        coordinate_quantization: Precision for coordinate values.
        parameter_quantization: Precision for command parameters.
        normalization_range: Bounding cube size (2.0 = 2x2x2 cube).
        canonicalize_loops: Reorder sketch loops to canonical form.
        include_constraints: Include SketchGraphs-style constraint tokens.
        pad_to_max_length: Pad shorter sequences to max_sequence_length.
        source_format: Source data format. "deepcad" expects compact params
            matching masks directly. "cadling" will auto-strip z-interleaved
            and padded params to compact form for backward compatibility with
            older cadling output. Default "auto" detects format heuristically.
    """
    max_sequence_length: int = 60
    coordinate_quantization: PrecisionTier = PrecisionTier.STANDARD
    parameter_quantization: PrecisionTier = PrecisionTier.STANDARD
    normalization_range: float = 2.0
    canonicalize_loops: bool = True
    include_constraints: bool = False
    pad_to_max_length: bool = True
    source_format: Literal["deepcad", "cadling", "auto"] = "auto"


@dataclass
class GraphTokenizationConfig:
    """Configuration for B-Rep topology graph tokenization.

    Controls how enriched B-Rep topology graphs (nodes with feature
    vectors, edges with feature vectors, adjacency structure) are
    converted to flat token sequences for transformer consumption.

    Args:
        node_bits: Quantization bits per dimension for node features.
        edge_bits: Quantization bits per dimension for edge features.
        max_nodes: Maximum node count (truncate or pad).
        max_edges: Maximum edge count (truncate or pad).
        include_uv_grids: Include UV-grid summary tokens per face/edge.
        uv_grid_summary_dim: Number of summary dimensions per UV grid.
        node_feature_dim: Expected node feature dimensionality (48 for cadling).
        edge_feature_dim: Expected edge feature dimensionality.  Set to 12
            for BrepGen-aligned configurations, or 16 for cadling's default
            enhanced edge features (which include 4 extra topology flags).
        adjacency_encoding: How adjacency is serialized.
            "explicit" lists neighbor indices per node.
            "implicit" relies on sorted edge list.
        pad_to_max: Pad shorter graphs to max_nodes/max_edges.
    """
    node_bits: int = 8
    edge_bits: int = 8
    max_nodes: int = 256
    max_edges: int = 1024
    include_uv_grids: bool = False
    uv_grid_summary_dim: int = 6
    node_feature_dim: int = 48
    edge_feature_dim: int = 16  # 12 for BrepGen compat, 16 for cadling extended
    adjacency_encoding: Literal["implicit", "explicit"] = "explicit"
    pad_to_max: bool = True

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        if self.max_nodes > 65535:
            raise ValueError(
                f"max_nodes must be <= 65535 (edge encoding uses 16-bit packing), "
                f"got {self.max_nodes}"
            )
