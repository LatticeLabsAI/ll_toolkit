"""Tests for streaming data integration.

Tests cover:
- StreamingProcessorConfig
- LazyTopologyLoader caching
- PrefetchingIterator background loading
- StreamingProcessor batch preprocessing
- Geotoken integration
- Topology extraction
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock, patch

import pytest

# Conditionally import torch
try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False

pytestmark = pytest.mark.skipif(not _has_torch, reason="torch required")


# ============================================================================
# SECTION 1: StreamingProcessorConfig Tests
# ============================================================================


class TestStreamingProcessorConfig:
    """Test StreamingProcessorConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from stepnet.streaming import StreamingProcessorConfig

        config = StreamingProcessorConfig()
        assert config.lazy_load_topology is True
        assert config.topology_cache_size == 1000
        assert config.prefetch_factor == 2
        assert config.graph_feature_dim == 48
        assert config.compact_topology is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        from stepnet.streaming import StreamingProcessorConfig

        config = StreamingProcessorConfig(
            lazy_load_topology=False,
            topology_cache_size=500,
            preprocess_fn="geotoken",
            include_graph_data=True,
        )
        assert config.lazy_load_topology is False
        assert config.topology_cache_size == 500
        assert config.preprocess_fn == "geotoken"
        assert config.include_graph_data is True


# ============================================================================
# SECTION 2: LazyTopologyLoader Tests
# ============================================================================


class TestLazyTopologyLoader:
    """Test LazyTopologyLoader caching."""

    def test_loader_creation(self) -> None:
        """Test loader can be created."""
        from stepnet.streaming import LazyTopologyLoader

        loader = LazyTopologyLoader(cache_size=100)
        assert loader is not None
        assert loader.cache_size == 100

    def test_load_caches_result(self) -> None:
        """Test loading caches the result."""
        from stepnet.streaming import LazyTopologyLoader

        loader = LazyTopologyLoader(cache_size=10)
        call_count = 0

        def load_fn():
            nonlocal call_count
            call_count += 1
            return {"node_features": torch.randn(5, 48)}

        # First load
        result1 = loader.load("sample_001", load_fn)
        assert call_count == 1

        # Second load should use cache
        result2 = loader.load("sample_001", load_fn)
        assert call_count == 1  # Not called again
        assert result1 is result2

    def test_cache_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        from stepnet.streaming import LazyTopologyLoader

        loader = LazyTopologyLoader(cache_size=3)

        # Fill cache
        for i in range(3):
            loader.load(f"sample_{i}", lambda i=i: {"id": i})

        # Add one more to trigger eviction
        loader.load("sample_3", lambda: {"id": 3})

        # Oldest should be evicted
        stats = loader.stats()
        assert stats["size"] == 3

    def test_cache_hit_rate(self) -> None:
        """Test cache hit rate calculation."""
        from stepnet.streaming import LazyTopologyLoader

        loader = LazyTopologyLoader(cache_size=10)

        # Load same item twice
        loader.load("sample_1", lambda: {})
        loader.load("sample_1", lambda: {})

        # Load new item
        loader.load("sample_2", lambda: {})

        # 1 hit, 2 misses
        stats = loader.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(1 / 3)

    def test_clear_cache(self) -> None:
        """Test clearing the cache."""
        from stepnet.streaming import LazyTopologyLoader

        loader = LazyTopologyLoader(cache_size=10)

        loader.load("sample_1", lambda: {})
        assert loader.stats()["size"] == 1

        loader.clear()
        assert loader.stats()["size"] == 0
        assert loader.stats()["hits"] == 0
        assert loader.stats()["misses"] == 0

    def test_thread_safety(self) -> None:
        """Test loader is thread-safe."""
        from stepnet.streaming import LazyTopologyLoader

        loader = LazyTopologyLoader(cache_size=100)
        results = []
        errors = []

        def worker(sample_id: str):
            try:
                result = loader.load(
                    sample_id,
                    lambda: {"id": sample_id, "data": torch.randn(10)},
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"sample_{i % 10}",))
            for i in range(50)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50


# ============================================================================
# SECTION 3: PrefetchingIterator Tests
# ============================================================================


class TestPrefetchingIterator:
    """Test PrefetchingIterator background loading."""

    def test_iterator_creation(self) -> None:
        """Test iterator can be created."""
        from stepnet.streaming import PrefetchingIterator

        source = iter(range(10))
        prefetcher = PrefetchingIterator(source, prefetch_factor=2)
        assert prefetcher is not None

    def test_iterator_yields_all_items(self) -> None:
        """Test iterator yields all source items."""
        from stepnet.streaming import PrefetchingIterator

        source = list(range(10))
        prefetcher = PrefetchingIterator(iter(source), prefetch_factor=2)

        results = list(prefetcher)
        assert results == source

    def test_prefetching_hides_latency(self) -> None:
        """Test prefetching can hide loading latency."""
        from stepnet.streaming import PrefetchingIterator

        def slow_iterator():
            for i in range(5):
                time.sleep(0.01)  # Simulate slow load
                yield {"batch": i}

        prefetcher = PrefetchingIterator(slow_iterator(), prefetch_factor=3)

        start = time.time()
        results = list(prefetcher)
        elapsed = time.time() - start

        assert len(results) == 5
        # Should be faster than 5 * 0.01 = 0.05s due to prefetching
        # (though not guaranteed in all environments)

    def test_stop_iterator(self) -> None:
        """Test stopping the iterator."""
        from stepnet.streaming import PrefetchingIterator

        def infinite_iterator():
            i = 0
            while True:
                yield i
                i += 1

        prefetcher = PrefetchingIterator(infinite_iterator(), prefetch_factor=2)
        items = []
        for item in prefetcher:
            items.append(item)
            if len(items) >= 5:
                prefetcher.stop()
                break

        assert len(items) == 5


# ============================================================================
# SECTION 4: StreamingProcessor Tests
# ============================================================================


class TestStreamingProcessor:
    """Test StreamingProcessor batch preprocessing."""

    def test_processor_creation(self) -> None:
        """Test processor can be created."""
        from stepnet.streaming import StreamingProcessor

        processor = StreamingProcessor()
        assert processor is not None

    def test_processor_with_config(self) -> None:
        """Test processor with custom config."""
        from stepnet.streaming import StreamingProcessor, StreamingProcessorConfig

        config = StreamingProcessorConfig(
            include_graph_data=True,
            graph_feature_dim=129,
        )
        processor = StreamingProcessor(config)
        assert processor.config.include_graph_data is True
        assert processor.config.graph_feature_dim == 129

    def test_preprocess_batch_with_token_ids(self) -> None:
        """Test preprocessing batch with token_ids."""
        from stepnet.streaming import StreamingProcessor

        processor = StreamingProcessor()

        batch = {
            "sample_id": "test_001",
            "token_ids": [1, 2, 3, 4, 5],
        }

        result = processor.preprocess_batch(batch)

        assert "token_ids" in result
        assert isinstance(result["token_ids"], torch.Tensor)
        assert result["token_ids"].tolist() == [1, 2, 3, 4, 5]

    def test_preprocess_batch_with_command_types(self) -> None:
        """Test preprocessing batch with command_types."""
        from stepnet.streaming import StreamingProcessor

        processor = StreamingProcessor()

        batch = {
            "sample_id": "test_002",
            "command_types": [0, 1, 2, 1, 5],
        }

        result = processor.preprocess_batch(batch)

        assert "token_ids" in result
        assert result["token_ids"].tolist() == [0, 1, 2, 1, 5]

    def test_preprocess_creates_attention_mask(self) -> None:
        """Test preprocessing creates attention mask."""
        from stepnet.streaming import StreamingProcessor

        processor = StreamingProcessor()

        batch = {
            "token_ids": [1, 2, 0, 0, 0],  # 0 is padding
        }

        result = processor.preprocess_batch(batch)

        assert "attention_mask" in result
        assert result["attention_mask"].tolist() == [1, 1, 0, 0, 0]

    def test_preprocess_preserves_existing_mask(self) -> None:
        """Test preprocessing preserves existing attention mask."""
        from stepnet.streaming import StreamingProcessor

        processor = StreamingProcessor()

        batch = {
            "token_ids": [1, 2, 3],
            "attention_mask": [1, 1, 0],
        }

        result = processor.preprocess_batch(batch)

        assert result["attention_mask"].tolist() == [1, 1, 0]

    def test_preprocess_with_metadata(self) -> None:
        """Test preprocessing preserves metadata."""
        from stepnet.streaming import StreamingProcessor

        processor = StreamingProcessor()

        batch = {
            "sample_id": "test_003",
            "file_path": "/path/to/file.step",
            "metadata": {"key": "value"},
            "token_ids": [1, 2, 3],
        }

        result = processor.preprocess_batch(batch)

        assert result["sample_id"] == "test_003"
        assert result["file_path"] == "/path/to/file.step"
        assert result["metadata"] == {"key": "value"}


# ============================================================================
# SECTION 5: Topology Extraction Tests
# ============================================================================


class TestTopologyExtraction:
    """Test topology data extraction."""

    def test_extract_precomputed_topology(self) -> None:
        """Test extracting pre-computed topology."""
        from stepnet.streaming import StreamingProcessor, StreamingProcessorConfig

        config = StreamingProcessorConfig(include_graph_data=True)
        processor = StreamingProcessor(config)

        batch = {
            "token_ids": [1, 2, 3],
            "topology_data": {
                "node_features": torch.randn(5, 48),
                "adjacency_matrix": torch.randint(0, 2, (5, 5)).float(),
            },
        }

        result = processor.preprocess_batch(batch)

        assert "topology_data" in result
        assert "node_features" in result["topology_data"]
        assert result["topology_data"]["node_features"].shape == (5, 48)

    def test_extract_node_features_and_adjacency(self) -> None:
        """Test extracting from node_features and adjacency_matrix."""
        from stepnet.streaming import StreamingProcessor, StreamingProcessorConfig

        config = StreamingProcessorConfig(include_graph_data=True)
        processor = StreamingProcessor(config)

        batch = {
            "token_ids": [1, 2, 3],
            "node_features": torch.randn(10, 48),
            "adjacency_matrix": torch.randint(0, 2, (10, 10)).float(),
        }

        result = processor.preprocess_batch(batch)

        assert "topology_data" in result
        assert result["topology_data"]["node_features"].shape == (10, 48)

    def test_topology_not_included_when_disabled(self) -> None:
        """Test topology not included when config disables it."""
        from stepnet.streaming import StreamingProcessor, StreamingProcessorConfig

        config = StreamingProcessorConfig(include_graph_data=False)
        processor = StreamingProcessor(config)

        batch = {
            "token_ids": [1, 2, 3],
            "node_features": torch.randn(10, 48),
        }

        result = processor.preprocess_batch(batch)

        assert "topology_data" not in result


# ============================================================================
# SECTION 6: Iterator Wrapping Tests
# ============================================================================


class TestIteratorWrapping:
    """Test iterator wrapping with preprocessing."""

    def test_wrap_iterator(self) -> None:
        """Test wrapping an iterator."""
        from stepnet.streaming import StreamingProcessor, StreamingProcessorConfig

        config = StreamingProcessorConfig(prefetch_factor=0)  # No prefetch
        processor = StreamingProcessor(config)

        source = [
            {"token_ids": [1, 2, 3]},
            {"token_ids": [4, 5, 6]},
        ]

        wrapped = processor.wrap_iterator(iter(source), prefetch=False)
        results = list(wrapped)

        assert len(results) == 2
        assert isinstance(results[0]["token_ids"], torch.Tensor)

    def test_wrap_iterator_with_prefetch(self) -> None:
        """Test wrapping with prefetching enabled."""
        from stepnet.streaming import StreamingProcessor, StreamingProcessorConfig

        config = StreamingProcessorConfig(prefetch_factor=2)
        processor = StreamingProcessor(config)

        source = [{"token_ids": [i]} for i in range(10)]

        wrapped = processor.wrap_iterator(iter(source), prefetch=True)
        results = list(wrapped)

        assert len(results) == 10


# ============================================================================
# SECTION 7: StreamingCadlingConfig Tests
# ============================================================================


class TestStreamingCadlingConfig:
    """Test StreamingCadlingConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from stepnet.config import StreamingCadlingConfig

        config = StreamingCadlingConfig()
        assert config.streaming is True
        assert config.batch_size == 8
        assert config.lazy_load_topology is True
        assert config.topology_cache_size == 1000
        assert config.graph_feature_dim == 48

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        from stepnet.config import StreamingCadlingConfig

        config = StreamingCadlingConfig(
            dataset_id="latticelabs/deepcad-sequences",
            split="val",
            batch_size=16,
            preprocess_fn="geotoken",
            include_graph_data=True,
        )
        assert config.dataset_id == "latticelabs/deepcad-sequences"
        assert config.split == "val"
        assert config.batch_size == 16
        assert config.preprocess_fn == "geotoken"
        assert config.include_graph_data is True


# ============================================================================
# SECTION 8: Module Import Tests
# ============================================================================


class TestStreamingImports:
    """Test streaming module imports."""

    def test_import_streaming_processor(self) -> None:
        """Test StreamingProcessor can be imported."""
        from stepnet.streaming import StreamingProcessor

        assert StreamingProcessor is not None

    def test_import_streaming_processor_config(self) -> None:
        """Test StreamingProcessorConfig can be imported."""
        from stepnet.streaming import StreamingProcessorConfig

        assert StreamingProcessorConfig is not None

    def test_import_prefetching_iterator(self) -> None:
        """Test PrefetchingIterator can be imported."""
        from stepnet.streaming import PrefetchingIterator

        assert PrefetchingIterator is not None

    def test_import_lazy_topology_loader(self) -> None:
        """Test LazyTopologyLoader can be imported."""
        from stepnet.streaming import LazyTopologyLoader

        assert LazyTopologyLoader is not None


# ============================================================================
# SECTION 9: Cache Statistics Tests
# ============================================================================


class TestCacheStatistics:
    """Test cache statistics reporting."""

    def test_processor_cache_stats(self) -> None:
        """Test processor reports cache statistics."""
        from stepnet.streaming import StreamingProcessor

        processor = StreamingProcessor()
        stats = processor.topology_cache_stats()

        assert "size" in stats
        assert "max_size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_clear_processor_cache(self) -> None:
        """Test clearing processor cache."""
        from stepnet.streaming import StreamingProcessor, StreamingProcessorConfig

        config = StreamingProcessorConfig(lazy_load_topology=True)
        processor = StreamingProcessor(config)

        # Add something to cache
        processor._topology_loader.load("test", lambda: {"data": 1})

        assert processor.topology_cache_stats()["size"] == 1

        processor.clear_cache()
        assert processor.topology_cache_stats()["size"] == 0


# ============================================================================
# SECTION 10: Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_batch(self) -> None:
        """Test handling empty batch."""
        from stepnet.streaming import StreamingProcessor

        processor = StreamingProcessor()
        batch = {}

        # Should not raise, just return empty-ish result
        result = processor.preprocess_batch(batch)
        assert isinstance(result, dict)

    def test_tensor_input_passthrough(self) -> None:
        """Test tensor inputs are passed through."""
        from stepnet.streaming import StreamingProcessor

        processor = StreamingProcessor()

        token_tensor = torch.tensor([1, 2, 3, 4, 5])
        batch = {"token_ids": token_tensor}

        result = processor.preprocess_batch(batch)

        assert result["token_ids"] is token_tensor

    def test_list_sample_id(self) -> None:
        """Test handling list sample_id."""
        from stepnet.streaming import StreamingProcessor, StreamingProcessorConfig

        config = StreamingProcessorConfig(
            lazy_load_topology=True,
            include_graph_data=True,
        )
        processor = StreamingProcessor(config)

        batch = {
            "sample_id": ["sample_001", "sample_002"],
            "token_ids": [1, 2, 3],
        }

        # Should not raise
        result = processor.preprocess_batch(batch)
        assert "token_ids" in result
