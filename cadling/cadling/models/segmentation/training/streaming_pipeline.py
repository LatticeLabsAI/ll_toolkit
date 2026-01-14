"""Streaming data pipeline for efficient CAD model training.

Provides efficient data streaming with:
- On-the-fly graph construction from STEP/mesh data
- Batching with PyTorch Geometric DataLoader
- Caching and prefetching for performance
- Multi-process data loading
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import IterableDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("PyTorch Geometric not available. Install with: pip install torch-geometric")


class StreamingCADDataset(IterableDataset):
    """Streaming dataset for CAD data with on-the-fly graph construction.

    Wraps a HuggingFace streaming dataset and converts samples to PyTorch
    Geometric graphs on-the-fly.

    Attributes:
        data_loader: Base data loader (MFCADDataLoader, etc.)
        graph_builder: Function to convert sample to PyG graph
        cache_graphs: Whether to cache constructed graphs
        max_cache_size: Maximum number of graphs to cache
    """

    def __init__(
        self,
        data_loader: Any,  # BaseDataLoader
        graph_builder: Callable[[dict[str, Any]], Data],
        cache_graphs: bool = True,
        max_cache_size: int = 1000,
    ):
        if not TORCH_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch and PyTorch Geometric required. "
                "Install: pip install torch torch-geometric"
            )

        self.data_loader = data_loader
        self.graph_builder = graph_builder
        self.cache_graphs = cache_graphs
        self.max_cache_size = max_cache_size

        # Simple cache (file_name -> graph)
        self._graph_cache: dict[str, Data] = {}

    def __iter__(self) -> Iterator[Data]:
        """Iterate over dataset, yielding PyG graphs."""
        for sample in self.data_loader:
            # Check cache first
            file_name = sample.get("file_name", "")
            if self.cache_graphs and file_name in self._graph_cache:
                yield self._graph_cache[file_name]
                continue

            # Build graph from sample
            try:
                graph = self.graph_builder(sample)

                # Cache if enabled
                if self.cache_graphs and len(self._graph_cache) < self.max_cache_size:
                    self._graph_cache[file_name] = graph

                yield graph

            except Exception as e:
                logger.warning(f"Failed to build graph for {file_name}: {e}")
                continue


class StreamingDataPipeline:
    """Complete streaming data pipeline for CAD segmentation training.

    Provides:
    - Streaming dataset loading (no full download)
    - On-the-fly graph construction
    - Efficient batching with PyG DataLoader
    - Multi-process data loading
    - Caching and prefetching

    Example:
        >>> pipeline = StreamingDataPipeline(
        ...     dataset_name="path/to/mfcad",
        ...     dataset_type="mfcad",
        ...     graph_builder=build_brep_graph,
        ...     batch_size=16,
        ...     num_workers=4
        ... )
        >>> for batch in pipeline:
        ...     logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        ...     loss = criterion(logits, batch.y)
    """

    def __init__(
        self,
        dataset_name: str,
        graph_builder: Callable[[dict[str, Any]], Data],
        dataset_type: str = "auto",
        split: str = "train",
        streaming: bool = True,
        batch_size: int = 16,
        num_workers: int = 0,
        cache_graphs: bool = True,
        max_cache_size: int = 1000,
        cache_dir: Optional[Path] = None,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000,
        **loader_kwargs,
    ):
        """Initialize streaming pipeline.

        Args:
            dataset_name: HuggingFace dataset name or local path
            graph_builder: Function to convert sample dict to PyG Data object
            dataset_type: Dataset type ('mfcad', 'mfinstseg', 'abc', 'fusion360', 'auto')
            split: Dataset split ('train', 'val', 'test')
            streaming: Whether to use streaming (avoid full download)
            batch_size: Batch size
            num_workers: Number of worker processes for data loading
            cache_graphs: Whether to cache constructed graphs in memory
            max_cache_size: Maximum number of graphs to cache
            cache_dir: Directory for HF dataset cache
            shuffle: Whether to shuffle dataset
            shuffle_buffer_size: Buffer size for shuffling (streaming only)
            **loader_kwargs: Additional arguments for data loader
        """
        # Import data loader
        from .data_loaders import get_data_loader

        # Create base data loader
        self.data_loader = get_data_loader(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            streaming=streaming,
            split=split,
            cache_dir=cache_dir,
            batch_size=1,  # StreamingCADDataset handles individual samples
            **loader_kwargs,
        )

        # Shuffle if requested
        if shuffle:
            if streaming:
                logger.info(f"Shuffling streaming dataset (buffer_size={shuffle_buffer_size})")
                self.data_loader.dataset = self.data_loader.dataset.shuffle(
                    seed=42, buffer_size=shuffle_buffer_size
                )
            else:
                logger.info("Shuffling non-streaming dataset")
                self.data_loader.shuffle(seed=42)

        # Create streaming CAD dataset
        self.cad_dataset = StreamingCADDataset(
            data_loader=self.data_loader,
            graph_builder=graph_builder,
            cache_graphs=cache_graphs,
            max_cache_size=max_cache_size,
        )

        # Create PyG DataLoader for batching
        # Note: PyG DataLoader doesn't work well with IterableDataset + num_workers > 0
        # We use custom batching instead
        self.batch_size = batch_size
        self.num_workers = num_workers

        logger.info(
            f"Streaming pipeline created: "
            f"dataset={dataset_name}, "
            f"streaming={streaming}, "
            f"batch_size={batch_size}, "
            f"cache={cache_graphs}"
        )

    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batched graphs."""
        batch_list = []

        for graph in self.cad_dataset:
            batch_list.append(graph)

            if len(batch_list) >= self.batch_size:
                # Create batch using PyG Batch
                batch = Batch.from_data_list(batch_list)
                yield batch
                batch_list = []

        # Yield final partial batch
        if batch_list:
            batch = Batch.from_data_list(batch_list)
            yield batch

    def take(self, n_batches: int) -> Iterator[Batch]:
        """Take first n batches from pipeline."""
        for i, batch in enumerate(self):
            if i >= n_batches:
                break
            yield batch


def create_streaming_pipeline(
    dataset_name: str,
    graph_builder: Callable[[dict[str, Any]], Data],
    dataset_type: str = "auto",
    split: str = "train",
    batch_size: int = 16,
    streaming: bool = True,
    **kwargs,
) -> StreamingDataPipeline:
    """Create streaming data pipeline (convenience function).

    Args:
        dataset_name: HuggingFace dataset name or local path
        graph_builder: Function to convert sample dict to PyG Data object
        dataset_type: Dataset type ('mfcad', 'mfinstseg', 'abc', 'auto')
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size
        streaming: Whether to use streaming mode
        **kwargs: Additional arguments for StreamingDataPipeline

    Returns:
        StreamingDataPipeline instance

    Example:
        >>> def build_graph(sample):
        ...     # Build PyG graph from sample
        ...     return Data(x=..., edge_index=..., y=...)
        ...
        >>> pipeline = create_streaming_pipeline(
        ...     dataset_name="path/to/mfcad",
        ...     graph_builder=build_graph,
        ...     batch_size=16
        ... )
        >>> for batch in pipeline.take(10):
        ...     print(f"Batch: {batch.num_graphs} graphs, {batch.num_nodes} nodes")
    """
    return StreamingDataPipeline(
        dataset_name=dataset_name,
        graph_builder=graph_builder,
        dataset_type=dataset_type,
        split=split,
        batch_size=batch_size,
        streaming=streaming,
        **kwargs,
    )


# Example graph builders for different tasks

def build_mesh_graph(sample: dict[str, Any]) -> Data:
    """Build PyG graph from mesh sample (ABC dataset).

    Args:
        sample: Sample from ABCDataLoader with mesh_vertices and mesh_faces

    Returns:
        PyG Data object with:
        - x: Node features [N, 7] (centroid + normal + area)
        - edge_index: Face adjacency [2, E]
        - edge_attr: Edge features [E, 2] (dihedral angle + length)
        - y: Labels (if available)
    """
    import torch
    from cadling.lib.graph import mesh_to_pyg_graph
    import trimesh

    # Create trimesh from vertices and faces
    mesh = trimesh.Trimesh(
        vertices=sample["mesh_vertices"],
        faces=sample["mesh_faces"],
    )

    # Convert to PyG graph using REAL geometric features
    graph = mesh_to_pyg_graph(mesh, use_face_graph=True)

    return graph


def build_brep_graph(sample: dict[str, Any]) -> Data:
    """Build PyG graph from B-Rep sample (MFCAD/MFInstSeg).

    Args:
        sample: Sample from MFCADDataLoader with step_content and face_labels

    Returns:
        PyG Data object with:
        - x: Node features [N, 24] (surface type + geometry)
        - edge_index: Face adjacency [2, E]
        - edge_attr: Edge features [E, 8]
        - y: Face labels [N]
    """
    import torch
    from cadling.lib.graph import brep_to_pyg_graph

    # Get STEP entities from sample
    # Data loader ensures entities is never None (skips bad samples)
    entities = sample.get("entities")

    # Get face labels if available
    face_labels = sample.get("face_labels")

    # Build graph using REAL geometric features from STEP entities
    graph_data = brep_to_pyg_graph(entities=entities, face_labels=face_labels)

    return graph_data


def build_instance_seg_graph(sample: dict[str, Any]) -> Data:
    """Build PyG graph for instance segmentation (MFInstSeg).

    Args:
        sample: Sample from MFInstSegDataLoader

    Returns:
        PyG Data object with:
        - x: Node features
        - edge_index: Face adjacency
        - y: Semantic labels
        - instance_labels: Instance IDs
        - boundary_labels: Boundary edge flags
    """
    import torch
    from cadling.lib.graph import brep_to_pyg_graph

    # Get STEP entities from sample
    # Data loader ensures entities is never None (skips bad samples)
    entities = sample.get("entities")

    # Build graph using REAL geometric features from STEP entities
    graph_data = brep_to_pyg_graph(entities=entities)

    # Add instance segmentation specific labels
    instance_masks = sample.get("instance_masks")
    if instance_masks is not None:
        graph_data.instance_labels = torch.tensor(instance_masks, dtype=torch.long)

    # Add boundary labels if available
    boundary_masks = sample.get("boundary_masks")
    if boundary_masks is not None:
        graph_data.boundary_labels = torch.tensor(boundary_masks, dtype=torch.long)

    return graph_data


# Example usage
if __name__ == "__main__":
    # Example: Create streaming pipeline for MFCAD++
    print("Creating streaming pipeline for MFCAD++ dataset...")

    pipeline = create_streaming_pipeline(
        dataset_name="path/to/mfcad",
        graph_builder=build_brep_graph,
        dataset_type="mfcad",
        split="train",
        batch_size=16,
        streaming=True,
        cache_graphs=True,
    )

    print("Streaming batches:")
    for i, batch in enumerate(pipeline.take(3)):
        print(f"\nBatch {i+1}:")
        print(f"  Graphs: {batch.num_graphs}")
        print(f"  Total nodes: {batch.num_nodes}")
        print(f"  Total edges: {batch.num_edges}")
        print(f"  Node features shape: {batch.x.shape}")
        print(f"  Labels shape: {batch.y.shape}")
