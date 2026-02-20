"""Unit tests for feature cache module.

Tests the FeatureCache class that provides persistent caching of
extracted geometric features.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestFeatureCacheImports:
    """Test module imports."""

    def test_module_imports(self):
        """Test that the module imports successfully."""
        from cadling.lib.cache import FeatureCache
        from cadling.lib.cache.feature_cache import (
            NumpyEncoder,
            numpy_decoder,
            get_feature_cache,
            reset_feature_cache,
        )

        assert FeatureCache is not None
        assert NumpyEncoder is not None
        assert numpy_decoder is not None


class TestNumpyEncoder:
    """Test NumpyEncoder JSON encoder."""

    def test_encode_ndarray(self):
        """Test encoding numpy array."""
        from cadling.lib.cache.feature_cache import NumpyEncoder

        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = json.dumps(arr, cls=NumpyEncoder)
        parsed = json.loads(result)

        assert parsed["_type"] == "ndarray"
        assert parsed["dtype"] == "float32"
        assert parsed["shape"] == [2, 2]
        assert parsed["data"] == [[1.0, 2.0], [3.0, 4.0]]

    def test_encode_numpy_int(self):
        """Test encoding numpy integer."""
        from cadling.lib.cache.feature_cache import NumpyEncoder

        val = np.int64(42)
        result = json.dumps({"value": val}, cls=NumpyEncoder)
        parsed = json.loads(result)

        assert parsed["value"] == 42

    def test_encode_numpy_float(self):
        """Test encoding numpy float."""
        from cadling.lib.cache.feature_cache import NumpyEncoder

        val = np.float32(3.14)
        result = json.dumps({"value": val}, cls=NumpyEncoder)
        parsed = json.loads(result)

        assert abs(parsed["value"] - 3.14) < 0.01

    def test_encode_numpy_bool(self):
        """Test encoding numpy bool."""
        from cadling.lib.cache.feature_cache import NumpyEncoder

        val = np.bool_(True)
        result = json.dumps({"value": val}, cls=NumpyEncoder)
        parsed = json.loads(result)

        assert parsed["value"] is True

    def test_encode_path(self):
        """Test encoding Path object."""
        from cadling.lib.cache.feature_cache import NumpyEncoder

        path = Path("/tmp/test.step")
        result = json.dumps({"path": path}, cls=NumpyEncoder)
        parsed = json.loads(result)

        assert parsed["path"] == "/tmp/test.step"


class TestNumpyDecoder:
    """Test numpy_decoder function."""

    def test_decode_ndarray(self):
        """Test decoding numpy array."""
        from cadling.lib.cache.feature_cache import numpy_decoder

        obj = {
            "_type": "ndarray",
            "dtype": "float32",
            "shape": [2, 2],
            "data": [[1.0, 2.0], [3.0, 4.0]],
        }

        result = numpy_decoder(obj)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[1, 2], [3, 4]])

    def test_decode_regular_dict(self):
        """Test decoding regular dict passes through."""
        from cadling.lib.cache.feature_cache import numpy_decoder

        obj = {"key": "value", "number": 42}
        result = numpy_decoder(obj)

        assert result == obj


class TestFeatureCache:
    """Test FeatureCache class."""

    def test_init_default(self):
        """Test default initialization."""
        from cadling.lib.cache.feature_cache import FeatureCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)

            assert cache.cache_dir == Path(tmpdir)
            assert cache.enabled is True
            assert cache.ttl_seconds > 0

    def test_init_disabled(self):
        """Test initialization with caching disabled."""
        from cadling.lib.cache.feature_cache import FeatureCache

        cache = FeatureCache(enabled=False)
        assert cache.enabled is False

    def test_compute_key_with_file(self):
        """Test computing cache key for a file."""
        from cadling.lib.cache.feature_cache import FeatureCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)

            # Create a temp file
            test_file = Path(tmpdir) / "test.step"
            test_file.write_text("STEP file content")

            key1 = cache.compute_key(test_file, {"param": 1})
            key2 = cache.compute_key(test_file, {"param": 1})
            key3 = cache.compute_key(test_file, {"param": 2})

            # Same file + params = same key
            assert key1 == key2
            # Different params = different key
            assert key1 != key3

    def test_compute_key_nonexistent_file(self):
        """Test computing cache key for nonexistent file."""
        from cadling.lib.cache.feature_cache import FeatureCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)

            # Should not raise
            key = cache.compute_key("/nonexistent/file.step")
            assert isinstance(key, str)
            assert len(key) == 64  # SHA256 hex digest

    def test_get_miss(self):
        """Test cache miss returns None."""
        from cadling.lib.cache.feature_cache import FeatureCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)

            result = cache.get("nonexistent_key")
            assert result is None

    def test_set_and_get(self):
        """Test setting and getting cache entry."""
        from cadling.lib.cache.feature_cache import FeatureCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)

            features = {
                "num_faces": 6,
                "volume": 1.0,
                "faces": [{"type": "PLANE"}],
            }

            success = cache.set("test_key", features)
            assert success is True

            result = cache.get("test_key")
            assert result is not None
            assert result["features"] == features

    def test_set_and_get_with_numpy(self):
        """Test caching numpy arrays."""
        from cadling.lib.cache.feature_cache import FeatureCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)

            features = {
                "matrix": np.array([[1, 2], [3, 4]], dtype=np.float32),
                "vector": np.array([0.1, 0.2, 0.3]),
            }

            cache.set("numpy_key", features)
            result = cache.get("numpy_key")

            assert result is not None
            np.testing.assert_array_equal(
                result["features"]["matrix"],
                features["matrix"]
            )
            np.testing.assert_array_almost_equal(
                result["features"]["vector"],
                features["vector"]
            )

    def test_get_disabled(self):
        """Test get returns None when disabled."""
        from cadling.lib.cache.feature_cache import FeatureCache

        cache = FeatureCache(enabled=False)
        result = cache.get("any_key")
        assert result is None

    def test_set_disabled(self):
        """Test set returns False when disabled."""
        from cadling.lib.cache.feature_cache import FeatureCache

        cache = FeatureCache(enabled=False)
        result = cache.set("any_key", {"data": 1})
        assert result is False

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        from cadling.lib.cache.feature_cache import FeatureCache

        with tempfile.TemporaryDirectory() as tmpdir:
            # Very short TTL
            cache = FeatureCache(cache_dir=tmpdir, ttl_seconds=1)

            cache.set("expiring_key", {"data": 1})

            # Should be available immediately
            result = cache.get("expiring_key")
            assert result is not None

            # Wait for expiration
            time.sleep(1.5)

            # Should be expired
            result = cache.get("expiring_key")
            assert result is None

    def test_clear(self):
        """Test clearing cache."""
        from cadling.lib.cache.feature_cache import FeatureCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)

            cache.set("key1", {"data": 1})
            cache.set("key2", {"data": 2})

            assert cache.size() == 2

            cleared = cache.clear()
            assert cleared == 2
            assert cache.size() == 0

    def test_clear_by_age(self):
        """Test clearing cache entries by age."""
        from cadling.lib.cache.feature_cache import FeatureCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)

            cache.set("old_key", {"data": 1})
            time.sleep(0.1)

            # Clear entries older than 0.05 seconds
            cleared = cache.clear(max_age_seconds=0.05)
            assert cleared == 1

    def test_size(self):
        """Test cache size."""
        from cadling.lib.cache.feature_cache import FeatureCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)

            assert cache.size() == 0

            cache.set("key1", {"data": 1})
            assert cache.size() == 1

            cache.set("key2", {"data": 2})
            assert cache.size() == 2

    def test_disk_usage(self):
        """Test disk usage calculation."""
        from cadling.lib.cache.feature_cache import FeatureCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)

            assert cache.disk_usage() == 0

            cache.set("key1", {"data": "x" * 1000})
            usage = cache.disk_usage()
            assert usage > 1000

    def test_get_with_fallback(self):
        """Test get_with_fallback pattern."""
        from cadling.lib.cache.feature_cache import FeatureCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)

            call_count = 0

            def compute_fn():
                nonlocal call_count
                call_count += 1
                return {"computed": True}

            # First call should compute
            result1 = cache.get_with_fallback("test_key", compute_fn)
            assert result1 == {"computed": True}
            assert call_count == 1

            # Second call should use cache
            result2 = cache.get_with_fallback("test_key", compute_fn)
            assert result2 == {"computed": True}
            assert call_count == 1  # Not incremented


class TestGlobalCache:
    """Test global cache functions."""

    def test_get_feature_cache(self):
        """Test getting global cache instance."""
        from cadling.lib.cache.feature_cache import (
            get_feature_cache,
            reset_feature_cache,
        )

        reset_feature_cache()

        cache1 = get_feature_cache()
        cache2 = get_feature_cache()

        # Should return same instance
        assert cache1 is cache2

    def test_reset_feature_cache(self):
        """Test resetting global cache."""
        from cadling.lib.cache.feature_cache import (
            get_feature_cache,
            reset_feature_cache,
        )

        cache1 = get_feature_cache()
        reset_feature_cache()
        cache2 = get_feature_cache()

        # Should be different instances after reset
        assert cache1 is not cache2
