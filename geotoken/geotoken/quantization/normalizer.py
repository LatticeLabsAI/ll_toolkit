"""Relationship-preserving normalization for geometric data.

Normalizes geometry to a unit bounding cube while preserving
geometric relationships (parallel, perpendicular, symmetric).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import NormalizationConfig

_log = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """Result of normalization with transform parameters."""
    normalized_vertices: np.ndarray  # Normalized vertex positions
    center: np.ndarray               # Translation applied
    scale: np.ndarray | float        # Scale factor(s): float if uniform, (3,) array if per-axis
    original_bbox_min: np.ndarray    # Original bounding box min
    original_bbox_max: np.ndarray    # Original bounding box max

    def to_dict(self) -> dict:
        """Serialize normalization result to dictionary.

        Returns:
            Dictionary with all normalization parameters.
        """
        scale_val = self.scale.tolist() if isinstance(self.scale, np.ndarray) else self.scale
        return {
            "center": self.center.tolist(),
            "scale": scale_val,
            "original_bbox_min": self.original_bbox_min.tolist(),
            "original_bbox_max": self.original_bbox_max.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NormalizationResult":
        """Deserialize normalization result from dictionary.

        Args:
            data: Dictionary with normalization parameters.

        Returns:
            NormalizationResult instance.
        """
        scale_raw = data["scale"]
        scale = np.array(scale_raw) if isinstance(scale_raw, list) else scale_raw
        return cls(
            normalized_vertices=np.array(data.get("normalized_vertices", np.empty((0, 3)))),
            center=np.array(data["center"]),
            scale=scale,
            original_bbox_min=np.array(data["original_bbox_min"]),
            original_bbox_max=np.array(data["original_bbox_max"]),
        )


class RelationshipPreservingNormalizer:
    """Normalizes geometry to unit cube while preserving relationships.

    Uses uniform scaling (same factor for all axes) to preserve
    angles and proportions.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()

    def normalize(self, vertices: np.ndarray) -> NormalizationResult:
        """Normalize vertices to target range.

        Args:
            vertices: (N, 3) vertex positions

        Returns:
            NormalizationResult with normalized vertices and transform params
        """
        if len(vertices) == 0:
            return NormalizationResult(
                normalized_vertices=vertices.copy(),
                center=np.zeros(3),
                scale=1.0,
                original_bbox_min=np.zeros(3),
                original_bbox_max=np.zeros(3),
            )

        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)

        # Center
        center = (bbox_min + bbox_max) / 2.0
        centered = vertices - center if self.config.center else vertices.copy()

        # Scale to fit in unit cube
        extent = bbox_max - bbox_min
        target_range = self.config.target_range[1] - self.config.target_range[0]

        if self.config.preserve_aspect_ratio:
            # Uniform scale: same factor for all axes
            max_extent = np.max(extent)
            if max_extent < 1e-12:
                scale = 1.0
            else:
                scale = target_range / max_extent
        else:
            # Non-uniform scale: independent factor per axis
            safe_extent = np.where(extent < 1e-12, 1.0, extent)
            scale = np.where(extent < 1e-12, 1.0, target_range / safe_extent)

        normalized = centered * scale

        # Shift to target range
        if self.config.center:
            offset = (self.config.target_range[0] + self.config.target_range[1]) / 2.0
            normalized = normalized + offset

        return NormalizationResult(
            normalized_vertices=normalized,
            center=center,
            scale=scale,
            original_bbox_min=bbox_min,
            original_bbox_max=bbox_max,
        )

    def denormalize(
        self,
        normalized_vertices: np.ndarray,
        result: NormalizationResult,
    ) -> np.ndarray:
        """Reverse normalization to recover original coordinates.

        Args:
            normalized_vertices: Normalized vertex positions
            result: NormalizationResult from normalize()

        Returns:
            Denormalized vertex positions
        """
        if self.config.center:
            offset = (self.config.target_range[0] + self.config.target_range[1]) / 2.0
            centered = normalized_vertices - offset
        else:
            centered = normalized_vertices.copy()

        # Check for degenerate scale
        scale_is_degenerate = (
            np.all(np.abs(result.scale) < 1e-12)
            if isinstance(result.scale, np.ndarray)
            else abs(result.scale) < 1e-12
        )
        if scale_is_degenerate:
            if self.config.center:
                return centered + result.center
            return centered

        unscaled = centered / result.scale

        if self.config.center:
            return unscaled + result.center
        return unscaled
