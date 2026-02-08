"""Streaming data pipeline for HuggingFace Hub CAD datasets.

Provides streaming wrappers for petabyte-scale CAD datasets with:
- Column projection for bandwidth efficiency
- Distributed shard splitting for multi-GPU training
- Shuffle buffers for streaming randomization
- PyTorch DataLoader integration
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

_log = logging.getLogger(__name__)

# Lazy imports
_datasets = None
_torch = None


def _ensure_datasets():
    """Lazily import datasets library."""
    global _datasets
    if _datasets is None:
        try:
            import datasets
            _datasets = datasets
        except ImportError:
            raise ImportError(
                "datasets is required for streaming. "
                "Install via: pip install datasets>=2.16.0"
            )
    return _datasets


def _ensure_torch():
    """Lazily import torch."""
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for streaming datasets. "
                "Install via: conda install pytorch -c conda-forge"
            )
    return _torch


@dataclass
class CADStreamingConfig:
    """Configuration for CAD streaming datasets.

    Attributes:
        dataset_id: HuggingFace dataset ID (e.g., 'latticelabs/deepcad-sequences').
        split: Dataset split to load ('train', 'validation', 'test').
        streaming: Whether to use streaming mode (default True for large datasets).
        columns: List of columns to load (None = all). Use for bandwidth efficiency.
        batch_size: Batch size for iteration.
        shuffle: Whether to shuffle the data.
        shuffle_buffer_size: Size of the shuffle buffer (only for streaming).
        seed: Random seed for reproducibility.
        num_shards: Total number of shards for distributed training.
        shard_index: Index of the shard for this process.
        max_samples: Maximum number of samples to load (None = all).
        trust_remote_code: Whether to trust remote code in the dataset.
        token: HuggingFace API token for private datasets.
    """

    dataset_id: str
    split: str = "train"
    streaming: bool = True
    columns: Optional[List[str]] = None
    batch_size: int = 8
    shuffle: bool = True
    shuffle_buffer_size: int = 10000
    seed: int = 42
    num_shards: int = 1
    shard_index: int = 0
    max_samples: Optional[int] = None
    trust_remote_code: bool = False
    token: Optional[str] = None
    # Cache configuration
    cache_dir: Optional[str] = None
    # Preprocessing
    remove_columns: Optional[List[str]] = None
    # Data format
    max_seq_len: int = 60
    num_params: int = 16
    # Filters for dataset filtering (applied post-load for streaming datasets)
    # Each filter is a tuple of (column_name, operator, value)
    # e.g., [("num_commands", ">=", 10), ("source", "==", "deepcad")]
    # Supported operators: "==", "!=", ">", ">=", "<", "<="
    filters: Optional[List[Tuple[str, str, Any]]] = None

    @property
    def world_size(self) -> int:
        """Alias for num_shards for DDP consistency."""
        return self.num_shards

    @property
    def rank(self) -> int:
        """Alias for shard_index for DDP consistency."""
        return self.shard_index


class CADStreamingDataset:
    """Streaming dataset wrapper for HuggingFace CAD datasets.

    Provides PyTorch IterableDataset-compatible interface with:
    - Automatic column projection
    - Distributed shard splitting
    - Epoch-based shuffle seeding
    - Lazy loading for large datasets

    Usage:
        >>> config = CADStreamingConfig(
        ...     dataset_id="latticelabs/deepcad-sequences",
        ...     batch_size=8,
        ...     shuffle=True,
        ... )
        >>> dataset = CADStreamingDataset(config)
        >>> for batch in dataset:
        ...     print(batch["command_types"].shape)
        ...     break

    Args:
        config: Streaming configuration.
        transform: Optional transform applied to each sample.
    """

    def __init__(
        self,
        config: CADStreamingConfig,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.config = config
        self.transform = transform
        self._epoch = 0
        self._dataset = None

        _log.info(
            "Initializing CADStreamingDataset: %s (streaming=%s, batch_size=%d)",
            config.dataset_id,
            config.streaming,
            config.batch_size,
        )

    def _load_dataset(self) -> "datasets.Dataset":
        """Load the HuggingFace dataset."""
        datasets = _ensure_datasets()

        load_kwargs = {
            "path": self.config.dataset_id,
            "split": self.config.split,
            "streaming": self.config.streaming,
            "trust_remote_code": self.config.trust_remote_code,
        }

        if self.config.token:
            load_kwargs["token"] = self.config.token

        if self.config.cache_dir:
            load_kwargs["cache_dir"] = self.config.cache_dir

        dataset = datasets.load_dataset(**load_kwargs)

        # Column selection
        if self.config.columns:
            if hasattr(dataset, "select_columns"):
                dataset = dataset.select_columns(self.config.columns)
            elif hasattr(dataset, "remove_columns"):
                # Get all columns and remove those not in selection
                all_cols = dataset.column_names
                if isinstance(all_cols, list):
                    remove_cols = [c for c in all_cols if c not in self.config.columns]
                    if remove_cols:
                        dataset = dataset.remove_columns(remove_cols)

        # Remove specified columns
        if self.config.remove_columns:
            dataset = dataset.remove_columns(self.config.remove_columns)

        # Apply filters (post-load for streaming datasets)
        if self.config.filters:
            dataset = self._apply_filters(dataset)

        # Shuffle with epoch-based seed
        if self.config.shuffle:
            seed = self.config.seed + self._epoch
            dataset = dataset.shuffle(seed=seed, buffer_size=self.config.shuffle_buffer_size)

        # Shard for distributed training
        if self.config.num_shards > 1:
            # Use split_dataset_by_node pattern for distributed training
            try:
                from datasets.distributed import split_dataset_by_node
                dataset = split_dataset_by_node(
                    dataset,
                    rank=self.config.shard_index,
                    world_size=self.config.num_shards,
                )
            except ImportError:
                # Fallback: manual sharding (less efficient)
                _log.warning(
                    "datasets.distributed not available, using manual sharding"
                )
                dataset = dataset.shard(
                    num_shards=self.config.num_shards,
                    index=self.config.shard_index,
                )

        # Limit samples
        if self.config.max_samples:
            dataset = dataset.take(self.config.max_samples)

        return dataset

    def _apply_filters(self, dataset: Any) -> Any:
        """Apply filter conditions to the dataset.

        Args:
            dataset: HuggingFace dataset to filter.

        Returns:
            Filtered dataset.
        """
        if not self.config.filters:
            return dataset

        def filter_fn(example: Dict[str, Any]) -> bool:
            """Check if example passes all filter conditions."""
            for column, op, value in self.config.filters:
                if column not in example:
                    continue

                example_val = example[column]

                # Apply operator
                if op == "==":
                    if example_val != value:
                        return False
                elif op == "!=":
                    if example_val == value:
                        return False
                elif op == ">":
                    if not (example_val > value):
                        return False
                elif op == ">=":
                    if not (example_val >= value):
                        return False
                elif op == "<":
                    if not (example_val < value):
                        return False
                elif op == "<=":
                    if not (example_val <= value):
                        return False
                else:
                    _log.warning("Unknown filter operator: %s", op)

            return True

        return dataset.filter(filter_fn)

    @property
    def dataset(self) -> "datasets.Dataset":
        """Lazily load and cache the dataset."""
        if self._dataset is None:
            self._dataset = self._load_dataset()
        return self._dataset

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for shuffle seed.

        Call this at the start of each epoch to ensure different
        shuffling across epochs while maintaining reproducibility.

        Args:
            epoch: Current epoch number.
        """
        if epoch != self._epoch:
            self._epoch = epoch
            self._dataset = None  # Force reload with new shuffle seed
            _log.debug("Set epoch to %d, will reshuffle on next iteration", epoch)

    def _process_sample(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single sample for training.

        Handles:
        - Numpy array conversion to tensors
        - Parameter reshaping from flat to [seq_len, num_params]
        - Attention mask creation
        """
        torch = _ensure_torch()

        processed = {}

        # Convert arrays to tensors
        for key, value in sample.items():
            if value is None:
                continue

            if isinstance(value, (list, tuple)):
                try:
                    processed[key] = torch.tensor(value)
                except (ValueError, TypeError):
                    processed[key] = value
            elif hasattr(value, "numpy"):
                # Already a tensor-like
                processed[key] = torch.from_numpy(value.numpy())
            else:
                processed[key] = value

        # Reshape parameters if present
        if "parameters" in processed:
            params = processed["parameters"]
            if params.dim() == 1:
                # Reshape from flat to [seq_len, num_params]
                seq_len = self.config.max_seq_len
                num_params = self.config.num_params
                expected_size = seq_len * num_params
                if params.numel() == expected_size:
                    processed["parameters"] = params.view(seq_len, num_params)

        # Create attention mask from mask field
        if "mask" in processed and "attention_mask" not in processed:
            processed["attention_mask"] = (processed["mask"] > 0).long()

        # Apply custom transform
        if self.transform:
            processed = self.transform(processed)

        return processed

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples.

        Yields:
            Processed sample dictionaries ready for training.
        """
        for sample in self.dataset:
            yield self._process_sample(sample)

    def __len__(self) -> int:
        """Return dataset length (may be approximate for streaming)."""
        try:
            if hasattr(self.dataset, "__len__"):
                return len(self.dataset)
            if hasattr(self.dataset, "num_rows"):
                return self.dataset.num_rows
        except Exception:
            pass
        return 0

    def batch_iter(self) -> Iterator[Dict[str, Any]]:
        """Iterate over batched samples.

        Yields:
            Batched dictionaries with stacked tensors.
        """
        torch = _ensure_torch()

        batch = []
        for sample in self:
            batch.append(sample)
            if len(batch) >= self.config.batch_size:
                yield self._collate_batch(batch)
                batch = []

        # Yield remaining samples
        if batch:
            yield self._collate_batch(batch)

    def _collate_batch(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Collate a list of samples into a batch."""
        torch = _ensure_torch()

        if not batch:
            return {}

        collated = {}
        keys = batch[0].keys()

        for key in keys:
            values = [sample[key] for sample in batch if key in sample]
            if not values:
                continue

            # Check if all values are tensors
            if all(isinstance(v, torch.Tensor) for v in values):
                try:
                    collated[key] = torch.stack(values)
                except RuntimeError:
                    # Different shapes, keep as list
                    collated[key] = values
            else:
                collated[key] = values

        return collated

    def to_torch_dataloader(
        self,
        collate_fn: Optional[Callable] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> "torch.utils.data.DataLoader":
        """Create a PyTorch DataLoader from this streaming dataset.

        Args:
            collate_fn: Custom collate function (uses default if None).
            num_workers: Number of data loading workers.
            pin_memory: Whether to pin memory for GPU transfer.

        Returns:
            PyTorch DataLoader wrapping this dataset.
        """
        torch = _ensure_torch()
        from torch.utils.data import IterableDataset, DataLoader

        # Create an IterableDataset wrapper
        class _IterableWrapper(IterableDataset):
            def __init__(wrapper_self):
                wrapper_self.streaming_dataset = self

            def __iter__(wrapper_self):
                return iter(wrapper_self.streaming_dataset)

        wrapper = _IterableWrapper()

        return DataLoader(
            wrapper,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn or self._collate_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class CADGraphStreamingDataset(CADStreamingDataset):
    """Streaming dataset for B-Rep graph data with PyG batching.

    Extends CADStreamingDataset with PyTorch Geometric-compatible
    graph batching using Batch.from_data_list().

    Usage:
        >>> config = CADStreamingConfig(
        ...     dataset_id="latticelabs/brep-graphs",
        ...     batch_size=32,
        ... )
        >>> dataset = CADGraphStreamingDataset(config)
        >>> for batch in dataset.batch_iter():
        ...     print(batch.x.shape)  # Node features
        ...     print(batch.edge_index.shape)  # Edge indices
        ...     break

    Args:
        config: Streaming configuration.
        transform: Optional transform applied to each sample.
    """

    def __init__(
        self,
        config: CADStreamingConfig,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(config, transform)
        self._has_pyg = False

        try:
            from torch_geometric.data import Data, Batch
            self._has_pyg = True
        except ImportError:
            _log.warning(
                "torch_geometric not available. "
                "Graph batching will use simple stacking."
            )

    def _process_sample(
        self, sample: Dict[str, Any]
    ) -> Union[Dict[str, Any], "Data"]:
        """Process a sample, optionally converting to PyG Data."""
        torch = _ensure_torch()

        # First do standard processing
        processed = super()._process_sample(sample)

        if not self._has_pyg:
            return processed

        # Convert to PyG Data object
        from torch_geometric.data import Data

        num_faces = processed.get("num_faces", 0)
        num_edges = processed.get("num_edges", 0)

        # Get face features
        face_features = processed.get("face_features")
        face_feat_dim = processed.get("face_feature_dim", 10)

        if face_features is not None and isinstance(face_features, torch.Tensor):
            if face_features.dim() == 1 and num_faces > 0:
                # Reshape to [num_faces, feat_dim]
                face_features = face_features[:num_faces * face_feat_dim]
                face_features = face_features.view(num_faces, face_feat_dim)

        # Get edge index
        edge_index = processed.get("edge_index")
        if edge_index is not None and isinstance(edge_index, torch.Tensor):
            if edge_index.dim() == 1:
                # Reshape from flat to [2, num_edges]
                num_edge_pairs = edge_index.numel() // 2
                edge_index = edge_index[:num_edge_pairs * 2].view(2, num_edge_pairs)

        # Get edge features
        edge_features = processed.get("edge_features")
        edge_feat_dim = processed.get("edge_feature_dim", 4)

        if edge_features is not None and isinstance(edge_features, torch.Tensor):
            if edge_features.dim() == 1 and num_edges > 0:
                edge_features = edge_features[:num_edges * edge_feat_dim]
                edge_features = edge_features.view(num_edges, edge_feat_dim)

        # Create Data object
        data = Data(
            x=face_features.float() if face_features is not None else None,
            edge_index=edge_index.long() if edge_index is not None else None,
            edge_attr=edge_features.float() if edge_features is not None else None,
            num_nodes=num_faces,
        )

        # Add additional fields
        if "face_labels" in processed and processed["face_labels"] is not None:
            data.y = processed["face_labels"]

        if "sample_id" in processed:
            data.sample_id = processed["sample_id"]

        return data

    def _collate_batch(
        self, batch: List[Any]
    ) -> Any:
        """Collate a list of PyG Data objects into a Batch."""
        if not self._has_pyg:
            return super()._collate_batch(batch)

        from torch_geometric.data import Data, Batch

        # Filter to only Data objects
        data_list = [b for b in batch if isinstance(b, Data)]

        if not data_list:
            return super()._collate_batch(batch)

        return Batch.from_data_list(data_list)


def create_streaming_dataloader(
    dataset_id: str,
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    columns: Optional[List[str]] = None,
    **kwargs: Any,
) -> "torch.utils.data.DataLoader":
    """Convenience function to create a streaming DataLoader.

    Args:
        dataset_id: HuggingFace dataset ID.
        split: Dataset split.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: DataLoader workers.
        columns: Columns to load.
        **kwargs: Additional CADStreamingConfig arguments.

    Returns:
        PyTorch DataLoader.
    """
    config = CADStreamingConfig(
        dataset_id=dataset_id,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        columns=columns,
        **kwargs,
    )

    dataset = CADStreamingDataset(config)

    return dataset.to_torch_dataloader(num_workers=num_workers)


def create_graph_streaming_dataloader(
    dataset_id: str,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs: Any,
) -> "torch.utils.data.DataLoader":
    """Convenience function to create a graph streaming DataLoader.

    Args:
        dataset_id: HuggingFace dataset ID for graph data.
        split: Dataset split.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: DataLoader workers.
        **kwargs: Additional CADStreamingConfig arguments.

    Returns:
        PyTorch DataLoader with PyG batching.
    """
    config = CADStreamingConfig(
        dataset_id=dataset_id,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs,
    )

    dataset = CADGraphStreamingDataset(config)

    return dataset.to_torch_dataloader(num_workers=num_workers)


__all__ = [
    "CADStreamingConfig",
    "CADStreamingDataset",
    "CADGraphStreamingDataset",
    "create_streaming_dataloader",
    "create_graph_streaming_dataloader",
]
