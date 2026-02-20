"""Cadling utility libraries.

This package contains reusable utility modules for CAD data processing.

Subpackages:
    graph: Graph construction and feature extraction utilities
    geometry: Geometric analysis and UV grid extraction
    cache: Persistent caching for expensive computations
    topology: B-Rep topology analysis (face identity, coedge extraction)

Modules:
    occ_wrapper: Unified OCC/OCCWL wrapper for AI-friendly CAD processing
"""

__all__ = ["graph", "geometry", "cache", "topology", "occ_wrapper"]
