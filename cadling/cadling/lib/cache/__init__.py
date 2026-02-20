"""Cache utilities for cadling.

This package provides caching functionality for expensive computations
like geometric feature extraction.

Classes:
    FeatureCache: Persistent cache for extracted geometric features
"""

from cadling.lib.cache.feature_cache import FeatureCache

__all__ = ["FeatureCache"]
