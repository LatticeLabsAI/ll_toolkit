"""Streaming data processor for cadling integration.

This module provides utilities for processing streaming CAD data including:
- Batch preprocessing with configurable pipelines
- Lazy topology loading with LRU caching
- Geotoken vocabulary integration
- Background prefetching for latency hiding
"""
from __future__ import annotations

import logging
import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

_log = logging.getLogger(__name__)

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False


@dataclass
class StreamingProcessorConfig:
    """Configuration for streaming data processor.

    Attributes:
        lazy_load_topology: Whether to load topology on-demand.
        topology_cache_size: Maximum topologies to cache in memory.
        preprocess_fn: Preprocessing function name ('geotoken', 'tokenize', None).
        prefetch_factor: Number of batches to prefetch in background.
        max_memory_mb: Maximum memory for cached data.
        chunk_size: Number of samples per processing chunk.
        include_graph_data: Whether to include graph/topology data.
        graph_feature_dim: Expected node feature dimension (48 cadling, 129 legacy).
        compact_topology: Use 48-dim compact topology features.
        num_workers: Number of worker threads for prefetching.
    """

    lazy_load_topology: bool = True
    topology_cache_size: int = 1000
    preprocess_fn: Optional[str] = None
    prefetch_factor: int = 2
    max_memory_mb: int = 4096
    chunk_size: int = 1000
    include_graph_data: bool = False
    graph_feature_dim: int = 48
    compact_topology: bool = True
    num_workers: int = 2


class LazyTopologyLoader:
    """Lazy loader for topology data with LRU caching.

    Loads topology data on-demand and caches frequently accessed
    topologies to reduce redundant computation.

    Example:
        >>> loader = LazyTopologyLoader(cache_size=1000)
        >>> topology = loader.load("sample_001", lambda: extract_topology(data))
        >>> # Second call uses cache
        >>> topology = loader.load("sample_001", lambda: extract_topology(data))
    """

    def __init__(self, cache_size: int = 1000) -> None:
        """Initialize the lazy loader.

        Args:
            cache_size: Maximum number of topologies to cache.
        """
        self.cache_size = cache_size
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def load(
        self,
        sample_id: str,
        load_fn: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Load topology, using cache if available.

        Args:
            sample_id: Unique identifier for the sample.
            load_fn: Function to call if not cached.

        Returns:
            Topology data dictionary.
        """
        with self._lock:
            if sample_id in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(sample_id)
                self._hits += 1
                return self._cache[sample_id]

        # Load outside lock to avoid blocking
        topology = load_fn()

        with self._lock:
            # Check again in case another thread loaded it
            if sample_id in self._cache:
                self._hits += 1
                return self._cache[sample_id]

            self._misses += 1

            # Add to cache
            self._cache[sample_id] = topology

            # Evict oldest if at capacity
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return topology

    def clear(self) -> None:
        """Clear the topology cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.cache_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


class PrefetchingIterator:
    """Iterator that prefetches batches in a background thread.

    Hides data loading latency by prefetching the next N batches
    while the current batch is being processed.

    Example:
        >>> dataset = iter(streaming_dataset)
        >>> prefetcher = PrefetchingIterator(dataset, prefetch_factor=2)
        >>> for batch in prefetcher:
        ...     train_step(batch)  # Next batches loading in background
    """

    # Sentinel object to signal end of iteration
    _STOP_SENTINEL = object()

    def __init__(
        self,
        iterator: Iterator,
        prefetch_factor: int = 2,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the prefetching iterator.

        Args:
            iterator: Source iterator to wrap.
            prefetch_factor: Number of items to prefetch.
            timeout: Timeout in seconds for getting items.
        """
        self._iterator = iterator
        self._prefetch_factor = max(1, prefetch_factor)
        self._timeout = timeout
        # +1 to make room for sentinel
        self._queue: queue.Queue = queue.Queue(maxsize=self._prefetch_factor + 1)
        self._exhausted = threading.Event()
        self._error: Optional[Exception] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._stopped = False

    def _prefetch_worker(self) -> None:
        """Background worker that prefetches items."""
        try:
            for item in self._iterator:
                if self._exhausted.is_set():
                    break
                self._queue.put(item, timeout=self._timeout)
        except Exception as e:
            self._error = e
        finally:
            # Put sentinel to signal end - this unblocks the main thread immediately
            while not self._exhausted.is_set():
                try:
                    self._queue.put(self._STOP_SENTINEL, timeout=1.0)
                    break
                except queue.Full:
                    continue
            self._exhausted.set()

    def __iter__(self) -> "PrefetchingIterator":
        """Start the prefetching thread."""
        if not self._started:
            self._thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self._thread.start()
            self._started = True
        return self

    def __next__(self) -> Any:
        """Get the next prefetched item."""
        if self._stopped:
            raise StopIteration

        if self._error is not None:
            raise self._error

        try:
            # Try to get with timeout
            item = self._queue.get(timeout=self._timeout)

            # Check for sentinel
            if item is self._STOP_SENTINEL:
                self._stopped = True
                raise StopIteration

            return item
        except queue.Empty:
            if self._exhausted.is_set():
                self._stopped = True
                raise StopIteration
            raise

    def stop(self) -> None:
        """Stop the prefetching thread."""
        self._exhausted.set()
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class StreamingProcessor:
    """Processor for streaming CAD data batches.

    Applies preprocessing, tokenization, and topology extraction
    to streaming batches from cadling datasets.

    Example:
        >>> config = StreamingProcessorConfig(
        ...     preprocess_fn="geotoken",
        ...     include_graph_data=True,
        ... )
        >>> processor = StreamingProcessor(config)
        >>> for batch in streaming_dataset:
        ...     processed = processor.preprocess_batch(batch)
        ...     train_step(processed)
    """

    def __init__(
        self,
        config: Optional[StreamingProcessorConfig] = None,
    ) -> None:
        """Initialize the streaming processor.

        Args:
            config: Processing configuration. Uses defaults if not provided.
        """
        self.config = config or StreamingProcessorConfig()
        self._topology_loader = LazyTopologyLoader(
            cache_size=self.config.topology_cache_size
        )
        self._geotoken_vocab = None
        self._geotoken_tokenizer = None

        _log.info(
            "StreamingProcessor initialized: lazy_load=%s, cache_size=%d, "
            "preprocess_fn=%s",
            self.config.lazy_load_topology,
            self.config.topology_cache_size,
            self.config.preprocess_fn,
        )

    def preprocess_batch(
        self,
        batch: Dict[str, Any],
        include_topology: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Apply preprocessing to a batch.

        Args:
            batch: Raw batch from streaming dataset.
            include_topology: Override config.include_graph_data.

        Returns:
            Preprocessed batch ready for model input.
        """
        result = {}

        # Copy existing fields
        for key in ("sample_id", "file_path", "metadata"):
            if key in batch:
                result[key] = batch[key]

        # Process token sequences
        if "token_ids" in batch:
            result["token_ids"] = self._ensure_tensor(batch["token_ids"])
        elif "command_types" in batch:
            result["token_ids"] = self._ensure_tensor(batch["command_types"])

        # Process attention mask
        if "attention_mask" in batch:
            result["attention_mask"] = self._ensure_tensor(batch["attention_mask"])
        elif "token_ids" in result:
            # Create attention mask (1 for non-pad, 0 for pad)
            token_ids = result["token_ids"]
            result["attention_mask"] = (token_ids != 0).long()

        # Apply geotoken preprocessing if configured
        if self.config.preprocess_fn == "geotoken":
            result = self._apply_geotoken_preprocessing(result, batch)

        # Include topology/graph data if requested
        include_topo = (
            include_topology
            if include_topology is not None
            else self.config.include_graph_data
        )
        if include_topo:
            result["topology_data"] = self._extract_topology(batch)

        return result

    def _ensure_tensor(self, data: Any) -> "torch.Tensor":
        """Ensure data is a tensor."""
        if not _has_torch:
            return data

        if isinstance(data, torch.Tensor):
            return data
        return torch.tensor(data)

    def _apply_geotoken_preprocessing(
        self,
        result: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply geotoken vocabulary and tokenization.

        Args:
            result: Current result dict.
            batch: Original batch.

        Returns:
            Updated result with geotoken features.
        """
        # Lazy-load geotoken
        if self._geotoken_vocab is None:
            try:
                from geotoken import CADVocabulary
                self._geotoken_vocab = CADVocabulary()
                _log.info("Loaded geotoken CADVocabulary")
            except ImportError:
                _log.warning(
                    "geotoken not available; skipping geotoken preprocessing"
                )
                return result

        # If batch has raw command sequences, tokenize with geotoken
        if "commands" in batch and self._geotoken_tokenizer is None:
            try:
                from geotoken import CommandSequenceTokenizer
                self._geotoken_tokenizer = CommandSequenceTokenizer(
                    vocab=self._geotoken_vocab
                )
            except ImportError:
                pass

        if self._geotoken_tokenizer is not None and "commands" in batch:
            commands = batch["commands"]
            if isinstance(commands, list) and len(commands) > 0:
                try:
                    token_seq = self._geotoken_tokenizer.tokenize(commands)
                    if hasattr(token_seq, "command_tokens"):
                        result["geotoken_ids"] = self._ensure_tensor(
                            token_seq.command_tokens
                        )
                    if hasattr(token_seq, "param_tokens"):
                        result["geotoken_params"] = self._ensure_tensor(
                            token_seq.param_tokens
                        )
                except Exception as e:
                    _log.debug("Geotoken tokenization failed: %s", e)

        return result

    def _extract_topology(self, batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract topology data from batch.

        Args:
            batch: Raw batch data.

        Returns:
            Topology data dict or None.
        """
        # Check for pre-computed topology
        if "topology_data" in batch:
            return self._process_topology(batch["topology_data"])

        # Check for node features and adjacency
        if "node_features" in batch and "adjacency_matrix" in batch:
            return {
                "node_features": self._ensure_tensor(batch["node_features"]),
                "adjacency_matrix": self._ensure_tensor(batch["adjacency_matrix"]),
            }

        # Lazy load from sample data if available
        if self.config.lazy_load_topology and "sample_id" in batch:
            sample_id = batch["sample_id"]
            if isinstance(sample_id, (list, tuple)):
                sample_id = sample_id[0] if len(sample_id) > 0 else "unknown"

            return self._topology_loader.load(
                sample_id,
                lambda: self._build_topology_from_batch(batch),
            )

        return None

    def _process_topology(
        self, topology: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process topology data to ensure correct format."""
        result = {}

        if "node_features" in topology:
            node_feats = self._ensure_tensor(topology["node_features"])
            # Ensure correct feature dimension
            if (
                node_feats.dim() >= 2
                and node_feats.size(-1) != self.config.graph_feature_dim
            ):
                _log.debug(
                    "Topology feature dim %d != expected %d",
                    node_feats.size(-1),
                    self.config.graph_feature_dim,
                )
            result["node_features"] = node_feats

        if "adjacency_matrix" in topology:
            result["adjacency_matrix"] = self._ensure_tensor(
                topology["adjacency_matrix"]
            )

        if "edge_index" in topology:
            result["edge_index"] = self._ensure_tensor(topology["edge_index"])

        return result

    def _build_topology_from_batch(
        self, batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build topology data from batch features.

        Args:
            batch: Raw batch data.

        Returns:
            Constructed topology dict.
        """
        from stepnet.topology import STEPTopologyBuilder

        # If batch has STEP features, use topology builder
        if "features_list" in batch:
            builder = STEPTopologyBuilder()
            topology = builder.build_complete_topology(
                batch["features_list"],
                compact=self.config.compact_topology,
            )
            return {
                "node_features": topology["node_features"],
                "adjacency_matrix": topology["adjacency_matrix"],
            }

        # If batch has cadling topology graph
        if "topology_graph" in batch:
            topo_graph = batch["topology_graph"]
            if hasattr(topo_graph, "to_numpy_node_features"):
                import numpy as np

                node_feats = topo_graph.to_numpy_node_features()
                edge_index = topo_graph.to_edge_index()

                num_nodes = node_feats.shape[0]
                adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                if edge_index.shape[1] > 0:
                    adj_matrix[edge_index[0], edge_index[1]] = 1.0

                return {
                    "node_features": self._ensure_tensor(node_feats),
                    "adjacency_matrix": self._ensure_tensor(adj_matrix),
                }

        _log.debug("Could not build topology from batch")
        return {}

    def wrap_iterator(
        self,
        iterator: Iterator,
        prefetch: bool = True,
    ) -> Iterator:
        """Wrap an iterator with preprocessing and optional prefetching.

        Args:
            iterator: Source iterator.
            prefetch: Whether to use prefetching.

        Returns:
            Wrapped iterator that yields preprocessed batches.
        """

        def preprocessing_generator():
            for batch in iterator:
                yield self.preprocess_batch(batch)

        gen = preprocessing_generator()

        if prefetch and self.config.prefetch_factor > 0:
            return PrefetchingIterator(gen, self.config.prefetch_factor)

        return gen

    def topology_cache_stats(self) -> Dict[str, Any]:
        """Return topology cache statistics."""
        return self._topology_loader.stats()

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._topology_loader.clear()
        _log.info("Cleared topology cache")
