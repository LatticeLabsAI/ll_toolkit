"""Adaptive quantization with per-vertex precision.

Allocates more bits to geometrically complex regions and fewer
to flat/simple regions, reducing token count while preserving
important features.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import QuantizationConfig
from ..analysis.curvature import CurvatureAnalyzer
from ..analysis.feature_density import FeatureDensityAnalyzer
from .bit_allocator import BitAllocator
from .normalizer import RelationshipPreservingNormalizer, NormalizationResult

_log = logging.getLogger(__name__)


@dataclass
class AdaptiveQuantizationResult:
    """Result of adaptive quantization."""
    quantized_vertices: np.ndarray    # (N, 3) quantized integer values
    bits_per_vertex: np.ndarray       # (N,) bit width per vertex
    normalization: NormalizationResult
    total_bits: int                    # Sum of bits used
    mean_bits: float


class AdaptiveQuantizer:
    """Adaptive precision quantizer for geometric data.

    Pipeline:
    1. Normalize to unit bounding cube
    2. Analyze curvature
    3. Analyze feature density
    4. Allocate bits per vertex
    5. Quantize with per-vertex precision
    6. Prevent feature collapse
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()
        self.normalizer = RelationshipPreservingNormalizer(self.config.normalization)
        self.curvature_analyzer = CurvatureAnalyzer()
        self.density_analyzer = FeatureDensityAnalyzer()
        self.bit_allocator = BitAllocator(self.config.bit_allocation)

    def quantize(
        self,
        vertices: np.ndarray,
        faces: Optional[np.ndarray] = None,
    ) -> AdaptiveQuantizationResult:
        """Quantize vertices with adaptive precision.

        Args:
            vertices: (N, 3) vertex positions
            faces: (F, 3) face indices (optional, improves analysis)

        Returns:
            AdaptiveQuantizationResult

        Raises:
            TypeError: If vertices is not a numpy array.
            ValueError: If vertices is not a 2D array with 3 columns.
        """
        if not isinstance(vertices, np.ndarray):
            raise TypeError(
                f"vertices must be numpy array, got {type(vertices).__name__}"
            )
        if vertices.ndim != 2 or (len(vertices) > 0 and vertices.shape[1] != 3):
            raise ValueError(
                f"vertices must be (N, 3) array, got shape {vertices.shape}"
            )

        if len(vertices) == 0:
            return AdaptiveQuantizationResult(
                quantized_vertices=np.array([], dtype=int).reshape(0, 3),
                bits_per_vertex=np.array([], dtype=int),
                normalization=self.normalizer.normalize(vertices),
                total_bits=0,
                mean_bits=0.0,
            )

        # Step 1: Normalize
        norm_result = self.normalizer.normalize(vertices)
        normalized = norm_result.normalized_vertices

        # Step 2-3: Analyze complexity
        if faces is not None and len(faces) > 0:
            curvature = self.curvature_analyzer.analyze_mesh(vertices, faces)
            density = self.density_analyzer.analyze(vertices, faces)
        else:
            curvature = self.curvature_analyzer.analyze_point_cloud(vertices)
            # For point clouds without faces, density is uniform
            density_vals = np.zeros(len(vertices))
            from ..analysis.feature_density import FeatureDensityResult
            density = FeatureDensityResult(
                density=density_vals,
                edge_length_variance=density_vals,
                face_area_variance=density_vals,
            )

        # Step 4: Compute complexity and allocate bits
        w_c = self.config.bit_allocation.curvature_weight
        w_f = self.config.bit_allocation.density_weight
        complexity = w_c * curvature.combined_magnitude + w_f * density.density

        # Normalize complexity to prevent numerical issues
        c_max = np.max(complexity)
        if c_max > 0:
            complexity = complexity / c_max

        allocation = self.bit_allocator.allocate(complexity)

        # Step 5: Quantize per vertex
        quantized = np.zeros_like(normalized, dtype=int)
        for i in range(len(normalized)):
            bits = int(allocation.bits_per_vertex[i])
            levels = 2 ** bits
            for d in range(3):
                val = np.clip(normalized[i, d], 0.0, 1.0)
                quantized[i, d] = int(np.round(val * (levels - 1)))
                quantized[i, d] = min(max(quantized[i, d], 0), levels - 1)

        # Step 6: Feature collapse prevention
        quantized = self._prevent_feature_collapse(
            quantized, vertices, allocation.bits_per_vertex
        )

        total_bits = int(np.sum(allocation.bits_per_vertex) * 3)

        return AdaptiveQuantizationResult(
            quantized_vertices=quantized,
            bits_per_vertex=allocation.bits_per_vertex,
            normalization=norm_result,
            total_bits=total_bits,
            mean_bits=float(allocation.mean_bits),
        )

    def dequantize(self, result: AdaptiveQuantizationResult) -> np.ndarray:
        """Dequantize back to original coordinate space.

        Args:
            result: Quantization result

        Returns:
            (N, 3) reconstructed vertex positions
        """
        n = len(result.quantized_vertices)
        if n == 0:
            return np.array([]).reshape(0, 3)

        reconstructed_norm = np.zeros((n, 3))
        for i in range(n):
            bits = int(result.bits_per_vertex[i])
            levels = 2 ** bits
            for d in range(3):
                reconstructed_norm[i, d] = result.quantized_vertices[i, d] / (levels - 1)

        # Denormalize
        return self.normalizer.denormalize(reconstructed_norm, result.normalization)

    def _prevent_feature_collapse(
        self,
        quantized: np.ndarray,
        original_vertices: np.ndarray,
        bits_per_vertex: np.ndarray,
    ) -> np.ndarray:
        """Prevent distinct vertices from collapsing to same quantized value.

        Uses spatial hashing for O(n) collision detection instead of O(n²)
        pairwise comparison. If two originally-distinct vertices map to the
        same quantized value and their original distance exceeds
        minimum_feature_threshold, nudge one by 1 quantization level.

        Args:
            quantized: (N, 3) quantized values
            original_vertices: (N, 3) original positions
            bits_per_vertex: (N,) bit widths

        Returns:
            Adjusted quantized values
        """
        threshold = self.config.minimum_feature_threshold
        result = quantized.copy()
        n = len(quantized)

        if n == 0:
            return result

        # Build spatial hash: map quantized coords tuple → list of vertex indices
        coord_to_indices: dict[tuple[int, int, int], list[int]] = {}
        for i in range(n):
            key = (int(result[i, 0]), int(result[i, 1]), int(result[i, 2]))
            if key not in coord_to_indices:
                coord_to_indices[key] = []
            coord_to_indices[key].append(i)

        # Only process buckets with collisions (more than one vertex)
        for key, indices in coord_to_indices.items():
            if len(indices) <= 1:
                continue

            # Check all pairs within this collision bucket
            for idx_a in range(len(indices)):
                i = indices[idx_a]
                for idx_b in range(idx_a + 1, len(indices)):
                    j = indices[idx_b]

                    # Skip if already different (from previous nudge)
                    if not np.array_equal(result[i], result[j]):
                        continue

                    orig_dist = np.linalg.norm(
                        original_vertices[i] - original_vertices[j]
                    )
                    if orig_dist > threshold:
                        # Nudge vertex j by 1 level in the dimension
                        # with largest original difference
                        diff = np.abs(original_vertices[i] - original_vertices[j])
                        dim = np.argmax(diff)
                        bits = int(bits_per_vertex[j])
                        max_level = 2 ** bits - 1

                        if result[j, dim] < max_level:
                            result[j, dim] += 1
                        elif result[j, dim] > 0:
                            result[j, dim] -= 1

        return result
