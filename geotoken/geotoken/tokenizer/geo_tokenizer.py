"""Main entry point for geometric tokenization.

GeoTokenizer orchestrates the full pipeline:
normalize -> analyze -> allocate -> quantize -> tokenize
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import QuantizationConfig, PrecisionTier
from ..quantization.adaptive import AdaptiveQuantizer
from ..quantization.uniform import UniformQuantizer
from ..quantization.normalizer import RelationshipPreservingNormalizer
from ..impact.analyzer import QuantizationImpactAnalyzer, ImpactReport
from .token_types import CoordinateToken, GeometryToken, TokenSequence

_log = logging.getLogger(__name__)


class GeoTokenizer:
    """Main geometric tokenizer.

    Converts 3D mesh/point cloud data into discrete token sequences
    using adaptive or uniform quantization.

    Example:
        tokenizer = GeoTokenizer()
        tokens = tokenizer.tokenize(vertices, faces)
        reconstructed = tokenizer.detokenize(tokens)
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        """Initialize tokenizer.

        Args:
            config: Quantization configuration. Defaults to STANDARD tier
                   with adaptive quantization.
        """
        self.config = config or QuantizationConfig()
        self.adaptive_quantizer = AdaptiveQuantizer(self.config)
        self.uniform_quantizer = UniformQuantizer(bits=self.config.tier.bits)
        self.normalizer = RelationshipPreservingNormalizer(self.config.normalization)

    def tokenize(
        self,
        vertices: np.ndarray,
        faces: Optional[np.ndarray] = None,
    ) -> TokenSequence:
        """Tokenize 3D geometry into a token sequence.

        Args:
            vertices: (N, 3) vertex positions
            faces: (F, 3) face indices (optional, improves adaptive quality)

        Returns:
            TokenSequence with coordinate and geometry tokens

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
            return TokenSequence(metadata={"config": self.config.tier.value})

        if self.config.adaptive and faces is not None:
            return self._tokenize_adaptive(vertices, faces)
        else:
            return self._tokenize_uniform(vertices, faces)

    def detokenize(self, tokens: TokenSequence) -> np.ndarray:
        """Reconstruct 3D coordinates from token sequence.

        Args:
            tokens: TokenSequence from tokenize()

        Returns:
            (N, 3) reconstructed vertex positions
        """
        if not tokens.coordinate_tokens:
            return np.array([]).reshape(0, 3)

        # Check if we have adaptive (variable bits) or uniform
        bits_array = tokens.bits_per_vertex
        if len(set(bits_array)) <= 1:
            # Uniform dequantization
            bits = int(bits_array[0]) if len(bits_array) > 0 else self.config.tier.bits
            quantized = tokens.to_array().astype(float)
            levels = 2 ** bits
            normalized = quantized / (levels - 1)
        else:
            # Adaptive dequantization
            n = len(tokens.coordinate_tokens)
            normalized = np.zeros((n, 3))
            for i, token in enumerate(tokens.coordinate_tokens):
                levels = token.levels
                normalized[i, 0] = token.x / (levels - 1)
                normalized[i, 1] = token.y / (levels - 1)
                normalized[i, 2] = token.z / (levels - 1)

        # Denormalize using stored parameters
        norm_center = np.array(tokens.metadata.get("norm_center", [0, 0, 0]))
        norm_scale = tokens.metadata.get("norm_scale", 1.0)

        from ..quantization.normalizer import NormalizationResult
        norm_result = NormalizationResult(
            normalized_vertices=normalized,
            center=norm_center,
            scale=norm_scale,
            original_bbox_min=np.array(tokens.metadata.get("bbox_min", [0, 0, 0])),
            original_bbox_max=np.array(tokens.metadata.get("bbox_max", [0, 0, 0])),
        )

        # Pass the dequantized floats as both the coordinate data and
        # inside the NormalizationResult so denormalize uses the correct values.
        norm_result.normalized_vertices = normalized
        return self.normalizer.denormalize(norm_result.normalized_vertices, norm_result)

    def analyze_impact(
        self,
        vertices: np.ndarray,
        faces: Optional[np.ndarray] = None,
    ) -> ImpactReport:
        """Analyze quantization impact on geometry quality.

        Args:
            vertices: Original vertex positions
            faces: Face indices

        Returns:
            ImpactReport with quality metrics
        """
        analyzer = QuantizationImpactAnalyzer()
        return analyzer.analyze(vertices, faces, self.config)

    def _tokenize_adaptive(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> TokenSequence:
        """Tokenize with adaptive precision."""
        result = self.adaptive_quantizer.quantize(vertices, faces)

        # Build coordinate tokens
        coord_tokens = []
        for i in range(len(result.quantized_vertices)):
            q = result.quantized_vertices[i]
            bits = int(result.bits_per_vertex[i])
            coord_tokens.append(CoordinateToken(
                x=int(q[0]),
                y=int(q[1]),
                z=int(q[2]),
                bits=bits,
                vertex_index=i,
            ))

        # Build geometry tokens for faces
        geo_tokens = []
        if faces is not None:
            for fi, face in enumerate(faces):
                geo_tokens.append(GeometryToken(
                    token_type="face",
                    indices=list(face),
                ))

        metadata = {
            "config": self.config.tier.value,
            "adaptive": True,
            "total_bits": result.total_bits,
            "mean_bits": result.mean_bits,
            "norm_center": result.normalization.center.tolist(),
            "norm_scale": result.normalization.scale,
            "bbox_min": result.normalization.original_bbox_min.tolist(),
            "bbox_max": result.normalization.original_bbox_max.tolist(),
        }

        return TokenSequence(
            coordinate_tokens=coord_tokens,
            geometry_tokens=geo_tokens,
            metadata=metadata,
        )

    def _tokenize_uniform(
        self,
        vertices: np.ndarray,
        faces: Optional[np.ndarray],
    ) -> TokenSequence:
        """Tokenize with uniform precision."""
        norm_result = self.normalizer.normalize(vertices)
        normalized = norm_result.normalized_vertices

        quantized = self.uniform_quantizer.quantize(normalized)

        coord_tokens = []
        for i in range(len(quantized)):
            q = quantized[i]
            coord_tokens.append(CoordinateToken(
                x=int(q[0]),
                y=int(q[1]),
                z=int(q[2]),
                bits=self.config.tier.bits,
                vertex_index=i,
            ))

        geo_tokens = []
        if faces is not None:
            for face in faces:
                geo_tokens.append(GeometryToken(
                    token_type="face",
                    indices=list(face),
                ))

        metadata = {
            "config": self.config.tier.value,
            "adaptive": False,
            "norm_center": norm_result.center.tolist(),
            "norm_scale": norm_result.scale,
            "bbox_min": norm_result.original_bbox_min.tolist(),
            "bbox_max": norm_result.original_bbox_max.tolist(),
        }

        return TokenSequence(
            coordinate_tokens=coord_tokens,
            geometry_tokens=geo_tokens,
            metadata=metadata,
        )
