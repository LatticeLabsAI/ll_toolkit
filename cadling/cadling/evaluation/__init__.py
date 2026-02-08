"""Evaluation metrics for CAD generation.

Provides comprehensive metrics for assessing the quality of generated
CAD shapes including validity, coverage, novelty, and distribution similarity.

Also includes benchmark comparison utilities for comparing against published
results from DeepCAD, SkexGen, BrepGen, and Text2CAD.
"""

from __future__ import annotations

from .generation_metrics import (
    BenchmarkComparison,
    GenerationMetrics,
    PUBLISHED_BENCHMARKS,
)

__all__ = [
    "BenchmarkComparison",
    "GenerationMetrics",
    "PUBLISHED_BENCHMARKS",
]
