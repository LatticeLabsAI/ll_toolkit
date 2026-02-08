"""Quantization quality metrics."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

_log = logging.getLogger(__name__)


@dataclass
class FeatureLossMetric:
    """Metrics for feature loss due to quantization."""
    collapsed_vertex_pairs: int = 0     # Pairs that mapped to same value
    total_vertex_pairs: int = 0
    collapse_rate: float = 0.0          # collapsed / total
    mean_displacement: float = 0.0       # Mean per-vertex error
    max_displacement: float = 0.0        # Max per-vertex error

    @classmethod
    def compute(
        cls,
        original: np.ndarray,
        reconstructed: np.ndarray,
        quantization_epsilon: float = 1e-10,
    ) -> "FeatureLossMetric":
        """Compute feature loss between original and reconstructed vertices.

        Uses spatial hashing for O(n) collision detection instead of O(n²)
        pairwise comparison.

        Args:
            original: (N, 3) original vertex positions
            reconstructed: (N, 3) reconstructed vertex positions
            quantization_epsilon: Threshold for considering vertices collapsed

        Returns:
            FeatureLossMetric with collapse statistics
        """
        if len(original) == 0:
            return cls()

        displacements = np.linalg.norm(original - reconstructed, axis=1)
        n = len(original)

        # Use spatial hashing to find collapsed vertices efficiently
        # Round reconstructed coords to detect exact matches
        # Scale to integer grid for hashing
        scale = 1.0 / max(quantization_epsilon, 1e-15)
        int_coords = np.round(reconstructed * scale).astype(np.int64)

        # Build hash: quantized coords → list of original vertex indices
        coord_to_indices: dict[tuple[int, int, int], list[int]] = {}
        for i in range(n):
            key = (int(int_coords[i, 0]), int(int_coords[i, 1]), int(int_coords[i, 2]))
            if key not in coord_to_indices:
                coord_to_indices[key] = []
            coord_to_indices[key].append(i)

        # Count collapsed pairs: buckets with multiple vertices where
        # original distance was > threshold
        collapsed = 0
        total_pairs = 0
        orig_distance_threshold = 1e-6

        for indices in coord_to_indices.values():
            if len(indices) <= 1:
                continue

            # All pairs within this bucket are collapsed in reconstructed
            for idx_a in range(len(indices)):
                i = indices[idx_a]
                for idx_b in range(idx_a + 1, len(indices)):
                    j = indices[idx_b]
                    total_pairs += 1
                    orig_dist = np.linalg.norm(original[i] - original[j])
                    if orig_dist > orig_distance_threshold:
                        collapsed += 1

        return cls(
            collapsed_vertex_pairs=collapsed,
            total_vertex_pairs=max(total_pairs, 1),
            collapse_rate=collapsed / max(total_pairs, 1),
            mean_displacement=float(np.mean(displacements)),
            max_displacement=float(np.max(displacements)),
        )
