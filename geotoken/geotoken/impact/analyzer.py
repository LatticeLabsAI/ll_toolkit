"""Quantization impact analysis.

Compares original geometry with reconstructed geometry after
quantize-dequantize roundtrip to assess quality.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from ..config import QuantizationConfig, PrecisionTier
from ..quantization.adaptive import AdaptiveQuantizer
from ..quantization.uniform import UniformQuantizer
from ..quantization.normalizer import RelationshipPreservingNormalizer
from ..analysis.geometric_relationships import RelationshipDetector
from .metrics import FeatureLossMetric

_log = logging.getLogger(__name__)


@dataclass
class ImpactReport:
    """Comprehensive quantization impact report."""
    tier: str
    hausdorff_distance: float = 0.0       # Max distance from any original to closest reconstructed
    mean_error: float = 0.0
    max_error: float = 0.0
    relationship_preservation_rate: float = 1.0
    feature_loss: Optional[FeatureLossMetric] = None
    total_bits_used: int = 0
    mean_bits_per_vertex: float = 0.0


class QuantizationImpactAnalyzer:
    """Analyzes the impact of quantization on geometry quality."""

    def __init__(self):
        self.relationship_detector = RelationshipDetector()

    def analyze(
        self,
        vertices: np.ndarray,
        faces: Optional[np.ndarray] = None,
        config: Optional[QuantizationConfig] = None,
    ) -> ImpactReport:
        """Analyze quantization impact for a given configuration.

        Args:
            vertices: Original vertex positions
            faces: Face indices (optional)
            config: Quantization config (default: STANDARD tier)

        Returns:
            ImpactReport with quality metrics
        """
        config = config or QuantizationConfig()

        if len(vertices) == 0:
            return ImpactReport(tier=config.tier.value)

        # Detect original relationships
        original_relationships = []
        if faces is not None and len(faces) > 0:
            original_relationships = self.relationship_detector.detect_face_relationships(
                vertices, faces
            )

        # Quantize and dequantize
        quantizer = AdaptiveQuantizer(config)
        quant_result = quantizer.quantize(vertices, faces)
        reconstructed = quantizer.dequantize(quant_result)

        # Compute errors (forward direction: original → reconstructed)
        errors_forward = np.linalg.norm(vertices - reconstructed, axis=1)

        # Compute bidirectional Hausdorff distance
        # Forward: max distance from any original vertex to its corresponding reconstructed
        max_forward = float(np.max(errors_forward))

        # Reverse: max distance from any reconstructed vertex to nearest original
        tree = cKDTree(vertices)
        distances, _ = tree.query(reconstructed, k=1)
        max_reverse = float(np.max(distances))

        # Hausdorff distance is the max of both directions
        hausdorff = max(max_forward, max_reverse)
        errors = errors_forward

        # Relationship preservation
        preservation_rate = None
        if original_relationships and faces is not None:
            preservation_rate = self.relationship_detector.verify_relationships(
                original_relationships, reconstructed, faces
            )
        # If verification was not possible (None), default to 1.0
        # to avoid penalizing cases where no relationships exist
        if preservation_rate is None:
            preservation_rate = 1.0

        # Feature loss
        feature_loss = FeatureLossMetric.compute(vertices, reconstructed)

        return ImpactReport(
            tier=config.tier.value,
            hausdorff_distance=hausdorff,
            mean_error=float(np.mean(errors)),
            max_error=float(np.max(errors)),
            relationship_preservation_rate=preservation_rate,
            feature_loss=feature_loss,
            total_bits_used=quant_result.total_bits,
            mean_bits_per_vertex=quant_result.mean_bits,
        )

    def compare_tiers(
        self,
        vertices: np.ndarray,
        faces: Optional[np.ndarray] = None,
    ) -> dict[str, ImpactReport]:
        """Compare all precision tiers.

        Args:
            vertices: Original vertex positions
            faces: Face indices

        Returns:
            Dict mapping tier name to ImpactReport
        """
        reports = {}
        for tier in PrecisionTier:
            config = QuantizationConfig(tier=tier)
            reports[tier.value] = self.analyze(vertices, faces, config)
        return reports
