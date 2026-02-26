"""Persistent feature cache for geometric feature extraction.

This module provides a disk-based cache for extracted geometric features
to avoid recomputing expensive operations on repeated loads of the same CAD files.

Features are cached using a SHA256 hash of the source file content plus
extraction parameters as the cache key. The cache is stored as JSON files
in ~/.cache/cadling/features/{hash}.json.

Example:
    from cadling.lib.cache.feature_cache import FeatureCache

    cache = FeatureCache()

    # Check cache
    key = cache.compute_key(file_path, {"extract_uv_grids": True})
    cached = cache.get(key)

    if cached is not None:
        features = cached
    else:
        features = expensive_extraction(shape)
        cache.set(key, features)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

_log = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and types."""

    def default(self, obj: Any) -> Any:
        """Encode numpy types to JSON-serializable formats."""
        if isinstance(obj, np.ndarray):
            return {
                "_type": "ndarray",
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
                "data": obj.tolist(),
            }
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def numpy_decoder(obj: Dict[str, Any]) -> Any:
    """Decode JSON objects to numpy arrays where appropriate."""
    if "_type" in obj and obj["_type"] == "ndarray":
        return np.array(obj["data"], dtype=obj["dtype"])
    return obj


class FeatureCache:
    """Persistent cache for extracted geometric features.

    Uses SHA256 hash of source file + extraction params as cache key.
    Features stored as JSON in ~/.cache/cadling/features/{hash}.json

    Attributes:
        cache_dir: Directory for cache files
        ttl_seconds: Time-to-live for cached entries (0 = no expiration)
        enabled: Whether caching is enabled
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        ttl_seconds: Optional[int] = None,
        enabled: bool = True,
    ):
        """Initialize feature cache.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.cache/cadling/features
            ttl_seconds: Time-to-live in seconds. Defaults to settings value or 3600
            enabled: Whether caching is enabled. Defaults to True
        """
        # Get defaults from settings if available
        default_cache_dir = Path.home() / ".cache" / "cadling" / "features"
        default_ttl = 3600

        try:
            from cadling.datamodel.settings import get_settings

            settings = get_settings()
            default_cache_dir = settings.paths.cache_dir / "features"
            default_ttl = settings.processing.cache_ttl_seconds
            enabled = enabled and settings.processing.enable_caching
        except Exception:
            pass

        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else default_ttl
        self.enabled = enabled

        # Create cache directory if it doesn't exist
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            _log.debug(f"FeatureCache initialized: dir={self.cache_dir}, ttl={self.ttl_seconds}s")

    def compute_key(
        self,
        file_path: Union[str, Path],
        params: Optional[Dict[str, Any]] = None,
        content_hash: Optional[str] = None,
    ) -> str:
        """Compute cache key for a file and extraction parameters.

        Args:
            file_path: Path to the source CAD file
            params: Extraction parameters dictionary
            content_hash: Pre-computed content hash (optional, computed if not provided)

        Returns:
            SHA256 hash string as cache key
        """
        file_path = Path(file_path)

        # Compute content hash if not provided
        if content_hash is None:
            if file_path.exists():
                content_hash = self._hash_file(file_path)
            else:
                # Use path as fallback if file doesn't exist
                content_hash = hashlib.sha256(str(file_path).encode()).hexdigest()

        # Include params in hash
        params_str = ""
        if params:
            # Sort params for consistent hashing
            sorted_params = sorted(params.items())
            params_str = json.dumps(sorted_params, cls=NumpyEncoder)

        combined = f"{content_hash}:{params_str}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _hash_file(self, file_path: Path) -> str:
        """Compute SHA256 hash of file contents.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hex digest
        """
        hasher = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                # Read in chunks for large files
                while chunk := f.read(8192):
                    hasher.update(chunk)
        except Exception as e:
            _log.warning(f"Failed to hash file {file_path}: {e}")
            hasher.update(str(file_path).encode())

        return hasher.hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached features by key.

        Args:
            key: Cache key (from compute_key)

        Returns:
            Cached features dictionary, or None if not cached or expired
        """
        if not self.enabled:
            return None

        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            _log.debug(f"Cache miss: {key[:16]}...")
            return None

        try:
            # Check expiration
            if self.ttl_seconds > 0:
                mtime = cache_file.stat().st_mtime
                age = time.time() - mtime
                if age > self.ttl_seconds:
                    _log.debug(f"Cache expired: {key[:16]}... (age={age:.0f}s)")
                    self._delete(key)
                    return None

            # Load cached features
            with open(cache_file, "r") as f:
                data = json.load(f, object_hook=numpy_decoder)

            _log.debug(f"Cache hit: {key[:16]}...")
            # Unwrap: set() stores as {"_cached_at": ..., "features": ...}
            if isinstance(data, dict) and "features" in data:
                return data["features"]
            return data

        except Exception as e:
            _log.warning(f"Failed to load cache {key[:16]}...: {e}")
            return None

    def set(self, key: str, features: Dict[str, Any]) -> bool:
        """Store features in cache.

        Args:
            key: Cache key (from compute_key)
            features: Features dictionary to cache

        Returns:
            True if caching succeeded
        """
        if not self.enabled:
            return False

        cache_file = self.cache_dir / f"{key}.json"

        try:
            # Add metadata
            data = {
                "_cached_at": time.time(),
                "_version": "1.0",
                "features": features,
            }

            # Atomic write: write to temp file then rename to prevent corruption
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.cache_dir), suffix=".json.tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(data, f, cls=NumpyEncoder, indent=2)
                os.replace(tmp_path, str(cache_file))
            except BaseException:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            _log.debug(f"Cached features: {key[:16]}...")
            return True

        except Exception as e:
            _log.warning(f"Failed to cache features {key[:16]}...: {e}")
            return False

    def _delete(self, key: str) -> bool:
        """Delete cached entry.

        Args:
            key: Cache key

        Returns:
            True if deletion succeeded
        """
        cache_file = self.cache_dir / f"{key}.json"

        try:
            if cache_file.exists():
                cache_file.unlink()
            return True
        except Exception as e:
            _log.warning(f"Failed to delete cache {key[:16]}...: {e}")
            return False

    def clear(self, max_age_seconds: Optional[int] = None) -> int:
        """Clear cache entries.

        Args:
            max_age_seconds: Only clear entries older than this. If None, clear all.

        Returns:
            Number of entries cleared
        """
        if not self.cache_dir.exists():
            return 0

        cleared = 0
        now = time.time()

        # Clean both completed cache files and orphaned temp files
        import itertools
        cache_files = itertools.chain(
            self.cache_dir.glob("*.json"),
            self.cache_dir.glob("*.json.tmp"),
        )
        for cache_file in cache_files:
            try:
                if max_age_seconds is not None:
                    age = now - cache_file.stat().st_mtime
                    if age < max_age_seconds:
                        continue

                cache_file.unlink()
                cleared += 1
            except Exception as e:
                _log.warning(f"Failed to clear cache file {cache_file}: {e}")

        _log.info(f"Cleared {cleared} cache entries")
        return cleared

    def size(self) -> int:
        """Get number of cached entries.

        Returns:
            Number of cached entries
        """
        if not self.cache_dir.exists():
            return 0

        return len(list(self.cache_dir.glob("*.json")))

    def disk_usage(self) -> int:
        """Get total disk usage of cache in bytes.

        Returns:
            Total size in bytes
        """
        if not self.cache_dir.exists():
            return 0

        total = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                total += cache_file.stat().st_size
            except Exception:
                pass

        return total

    def get_with_fallback(
        self,
        key: str,
        compute_fn: callable,
        file_path: Optional[Union[str, Path]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get from cache or compute and cache result.

        Args:
            key: Cache key (use compute_key to generate)
            compute_fn: Function to call if cache miss (returns features dict)
            file_path: Optional file path for cache key computation
            params: Optional params for cache key computation

        Returns:
            Features dictionary (cached or freshly computed)
        """
        # Try cache first
        cached = self.get(key)
        if cached is not None:
            return cached

        # Compute features
        features = compute_fn()

        # Cache the result
        self.set(key, features)

        return features


# Convenience function for quick access
import threading

_global_cache: Optional[FeatureCache] = None
_global_cache_lock = threading.Lock()


def get_feature_cache() -> FeatureCache:
    """Get global feature cache instance (thread-safe).

    Returns:
        Global FeatureCache instance
    """
    global _global_cache
    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                _global_cache = FeatureCache()
    return _global_cache


def reset_feature_cache() -> None:
    """Reset global feature cache instance."""
    global _global_cache
    _global_cache = None
