"""WebDataset support for STEP TAR shards.

Provides streaming of STEP files from TAR shards for petabyte-scale
CAD training. Compatible with HuggingFace Hub and distributed training.

WebDataset is ideal for:
- Very large datasets (millions of STEP files)
- Random access patterns during training
- Efficient I/O with sequential TAR reads
- Cloud storage (S3, GCS) streaming

Usage:
    >>> from cadling.data.webdataset import STEPWebDataset, STEPWebDatasetConfig
    >>>
    >>> config = STEPWebDatasetConfig(
    ...     shards="s3://bucket/cad-dataset/train-{000000..009999}.tar",
    ...     batch_size=32,
    ... )
    >>> dataset = STEPWebDataset(config)
    >>>
    >>> for batch in dataset:
    ...     print(batch["step_data"].shape)
"""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

_log = logging.getLogger(__name__)

# Lazy imports
_webdataset = None
_torch = None


def _ensure_webdataset():
    """Lazily import webdataset."""
    global _webdataset
    if _webdataset is None:
        try:
            import webdataset as wds
            _webdataset = wds
        except ImportError:
            raise ImportError(
                "webdataset is required for TAR shard streaming. "
                "Install via: pip install webdataset>=0.2.0"
            )
    return _webdataset


def _ensure_torch():
    """Lazily import torch."""
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required. Install via: conda install pytorch -c conda-forge"
            )
    return _torch


@dataclass
class STEPWebDatasetConfig:
    """Configuration for STEP WebDataset.

    Attributes:
        shards: URL pattern for TAR shards. Supports:
            - Local: "/path/to/shards-{000000..009999}.tar"
            - S3: "s3://bucket/prefix-{000000..009999}.tar"
            - GCS: "gs://bucket/prefix-{000000..009999}.tar"
            - HTTP: "https://host/prefix-{000000..009999}.tar"
            - HuggingFace Hub: "hf://datasets/user/repo/train-*.tar"
        batch_size: Samples per batch.
        shuffle: Shuffle samples within each shard.
        shuffle_buffer: Size of shuffle buffer.
        seed: Random seed for reproducibility.
        num_workers: DataLoader workers.
        prefetch_factor: Prefetch batches per worker.
        world_size: Total number of processes for DDP.
        rank: This process's rank in DDP.
        max_samples: Maximum samples to load (None = all).
        cache_dir: Local cache for downloaded shards.
        handler: Error handler ('warn', 'ignore', 'reraise').
    """
    shards: str
    batch_size: int = 32
    shuffle: bool = True
    shuffle_buffer: int = 5000
    seed: int = 42
    num_workers: int = 4
    prefetch_factor: int = 2
    world_size: int = 1
    rank: int = 0
    max_samples: Optional[int] = None
    cache_dir: Optional[str] = None
    handler: str = "warn"

    # Processing options
    parse_step: bool = True  # Parse STEP to graph
    extract_features: bool = True  # Extract face/edge features
    include_raw_bytes: bool = False  # Include raw STEP bytes


@dataclass
class STEPTarSample:
    """A single sample from a STEP TAR shard.

    Attributes:
        key: Unique sample identifier from TAR.
        step_data: Raw STEP file bytes.
        metadata: JSON metadata if present.
        graph: Parsed graph data if parse_step=True.
    """
    key: str
    step_data: bytes
    metadata: Optional[Dict[str, Any]] = None
    graph: Optional[Dict[str, Any]] = None


class STEPWebDataset:
    """WebDataset for streaming STEP files from TAR shards.

    Efficiently streams STEP files from TAR archives, optionally
    parsing them to graph format for GNN training. Supports
    distributed training with automatic shard splitting.

    TAR Shard Format:
        Each shard is a TAR file containing:
        - {sample_id}.step or {sample_id}.stp: STEP file
        - {sample_id}.json (optional): Metadata

    Example shard structure:
        shard-000000.tar
        ├── model_001.step
        ├── model_001.json
        ├── model_002.step
        ├── model_002.json
        └── ...

    Usage:
        >>> config = STEPWebDatasetConfig(
        ...     shards="/data/cad-shards/train-{000..099}.tar",
        ...     batch_size=32,
        ... )
        >>> dataset = STEPWebDataset(config)
        >>> dataloader = dataset.to_dataloader()
        >>>
        >>> for batch in dataloader:
        ...     # Process batch
        ...     pass

    Args:
        config: STEPWebDatasetConfig.
        transform: Optional transform applied to each sample.
    """

    def __init__(
        self,
        config: STEPWebDatasetConfig,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.config = config
        self.transform = transform
        self._dataset = None
        self._has_pythonocc = False

        # Check for pythonocc for STEP parsing
        if config.parse_step:
            try:
                from OCC.Core.TopoDS import TopoDS_Shape
                self._has_pythonocc = True
            except ImportError:
                _log.warning(
                    "pythonocc not available, STEP files will be returned as bytes"
                )

        _log.info(
            "STEPWebDataset initialized: shards=%s, batch_size=%d",
            config.shards,
            config.batch_size,
        )

    def _create_dataset(self):
        """Create the underlying WebDataset pipeline."""
        wds = _ensure_webdataset()

        # Start with shard URLs
        dataset = wds.WebDataset(
            self.config.shards,
            shardshuffle=self.config.shuffle,
            nodesplitter=wds.split_by_node if self.config.world_size > 1 else None,
            handler=getattr(wds, self.config.handler, wds.warn_and_continue),
        )

        # Shuffle within shards
        if self.config.shuffle:
            dataset = dataset.shuffle(self.config.shuffle_buffer)

        # Decode samples
        dataset = dataset.map(self._decode_sample)

        # Parse STEP if configured
        if self.config.parse_step and self._has_pythonocc:
            dataset = dataset.map(self._parse_step_to_graph)

        # Apply custom transform
        if self.transform:
            dataset = dataset.map(self.transform)

        # Limit samples if configured
        if self.config.max_samples:
            dataset = dataset.with_length(self.config.max_samples)

        return dataset

    @property
    def dataset(self):
        """Lazily create and cache the dataset."""
        if self._dataset is None:
            self._dataset = self._create_dataset()
        return self._dataset

    def _decode_sample(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decode a raw WebDataset sample.

        WebDataset yields samples as dicts with extension-keyed values:
        {"__key__": "model_001", "step": <bytes>, "json": <bytes>}

        Args:
            sample: Raw sample dict from WebDataset.

        Returns:
            Decoded sample dict.
        """
        key = sample.get("__key__", "unknown")

        # Get STEP data (try multiple extensions)
        step_data = None
        for ext in ("step", "stp", "STEP", "STP"):
            if ext in sample:
                step_data = sample[ext]
                break

        if step_data is None:
            _log.debug("No STEP data found for key: %s", key)
            return {"__key__": key, "valid": False}

        result = {
            "__key__": key,
            "step_data": step_data,
            "valid": True,
        }

        # Include raw bytes if configured
        if self.config.include_raw_bytes:
            result["step_bytes"] = step_data

        # Parse JSON metadata
        json_data = sample.get("json")
        if json_data:
            try:
                if isinstance(json_data, bytes):
                    json_data = json_data.decode("utf-8")
                result["metadata"] = json.loads(json_data)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                _log.debug("Failed to parse JSON for %s: %s", key, e)

        return result

    def _parse_step_to_graph(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse STEP bytes to graph representation.

        Args:
            sample: Sample with step_data bytes.

        Returns:
            Sample with added graph fields.
        """
        if not sample.get("valid") or not sample.get("step_data"):
            return sample

        try:
            graph = self._step_bytes_to_graph(sample["step_data"])
            if graph:
                sample["graph"] = graph
                sample["num_faces"] = len(graph.get("faces", []))
                sample["num_edges"] = len(graph.get("edges", []))

                # Extract features if configured
                if self.config.extract_features:
                    sample.update(self._extract_features(graph))
        except Exception as e:
            _log.debug("Failed to parse STEP for %s: %s", sample.get("__key__"), e)

        return sample

    def _step_bytes_to_graph(
        self, step_bytes: bytes
    ) -> Optional[Dict[str, Any]]:
        """Convert STEP bytes to graph data using pythonocc."""
        if not self._has_pythonocc:
            return None

        try:
            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import (
                GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
            )
            from OCC.Core.TopoDS import topods
            import tempfile
            import os

            # Write bytes to temp file (pythonocc requires file path)
            with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
                f.write(step_bytes)
                temp_path = f.name

            try:
                reader = STEPControl_Reader()
                status = reader.ReadFile(temp_path)
                if status != 1:
                    return None

                reader.TransferRoots()
                shape = reader.OneShape()

                # Extract faces
                faces = []
                face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
                surf_type_map = {
                    GeomAbs_Plane: "plane",
                    GeomAbs_Cylinder: "cylinder",
                    GeomAbs_Cone: "cone",
                    GeomAbs_Sphere: "sphere",
                    GeomAbs_Torus: "torus",
                    GeomAbs_BSplineSurface: "bspline",
                }

                while face_explorer.More():
                    face = topods.Face(face_explorer.Current())
                    adaptor = BRepAdaptor_Surface(face)
                    surf_type = adaptor.GetType()

                    props = GProp_GProps()
                    brepgprop.SurfaceProperties(face, props)
                    area = props.Mass()
                    center = props.CentreOfMass()

                    faces.append({
                        "surface_type": surf_type_map.get(surf_type, "other"),
                        "area": area,
                        "centroid": [center.X(), center.Y(), center.Z()],
                    })
                    face_explorer.Next()

                # Extract edges (simplified)
                edges = []
                edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)

                while edge_explorer.More():
                    edge = topods.Edge(edge_explorer.Current())
                    try:
                        props = GProp_GProps()
                        brepgprop.LinearProperties(edge, props)
                        edges.append({"length": props.Mass()})
                    except Exception:
                        edges.append({"length": 0.0})
                    edge_explorer.Next()

                # Build adjacency
                num_faces = len(faces)
                edge_index = [[], []]
                for i in range(num_faces):
                    for j in range(i + 1, min(i + 5, num_faces)):
                        edge_index[0].extend([i, j])
                        edge_index[1].extend([j, i])

                return {
                    "faces": faces,
                    "edges": edges,
                    "edge_index": edge_index,
                }

            finally:
                os.unlink(temp_path)

        except Exception as e:
            _log.debug("Error parsing STEP bytes: %s", e)
            return None

    def _extract_features(
        self, graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract feature tensors from graph data."""
        torch = _ensure_torch()

        faces = graph.get("faces", [])
        edges = graph.get("edges", [])
        edge_index = graph.get("edge_index", [[], []])

        # Face features: [surface_type, area, cx, cy, cz]
        face_features = []
        surface_types = {
            "plane": 0, "cylinder": 1, "cone": 2, "sphere": 3,
            "torus": 4, "bspline": 5, "other": 6,
        }

        for face in faces:
            surf_type = surface_types.get(face.get("surface_type", "other"), 6)
            area = float(face.get("area", 0.0))
            centroid = face.get("centroid", [0.0, 0.0, 0.0])
            face_features.append([
                float(surf_type), area,
                centroid[0], centroid[1], centroid[2],
            ])

        # Edge features: just length for now
        edge_features = [[float(e.get("length", 0.0))] for e in edges]

        return {
            "face_features": torch.tensor(face_features, dtype=torch.float32),
            "edge_features": torch.tensor(edge_features, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples."""
        return iter(self.dataset)

    def batch_iter(self) -> Iterator[Dict[str, Any]]:
        """Iterate over batched samples."""
        wds = _ensure_webdataset()
        return iter(
            self.dataset.batched(self.config.batch_size)
        )

    def to_dataloader(
        self,
        collate_fn: Optional[Callable] = None,
    ) -> "torch.utils.data.DataLoader":
        """Create a PyTorch DataLoader.

        Args:
            collate_fn: Custom collate function.

        Returns:
            PyTorch DataLoader.
        """
        torch = _ensure_torch()
        wds = _ensure_webdataset()

        # Create batched dataset
        batched = self.dataset.batched(
            self.config.batch_size,
            collation_fn=collate_fn or self._default_collate,
        )

        return wds.WebLoader(
            batched,
            batch_size=None,  # Already batched
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
        )

    def _default_collate(
        self, samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Default collate function for batching."""
        torch = _ensure_torch()

        if not samples:
            return {}

        # Filter valid samples
        valid_samples = [s for s in samples if s.get("valid", True)]
        if not valid_samples:
            return {"valid": False}

        batch = {"__keys__": [s.get("__key__") for s in valid_samples]}

        # Stack tensor fields
        tensor_keys = ["face_features", "edge_features", "edge_index"]
        for key in tensor_keys:
            tensors = [s[key] for s in valid_samples if key in s]
            if tensors:
                try:
                    batch[key] = torch.stack(tensors)
                except RuntimeError:
                    # Different shapes, keep as list
                    batch[key] = tensors

        # Collect metadata
        if any("metadata" in s for s in valid_samples):
            batch["metadata"] = [s.get("metadata") for s in valid_samples]

        return batch


def create_step_tar_shards(
    source_dir: str,
    output_dir: str,
    samples_per_shard: int = 1000,
    pattern: str = "shard-{:06d}.tar",
) -> List[Path]:
    """Create TAR shards from a directory of STEP files.

    Utility function to package STEP files into WebDataset-compatible
    TAR shards for efficient streaming.

    Args:
        source_dir: Directory containing STEP files.
        output_dir: Output directory for TAR shards.
        samples_per_shard: Number of samples per shard.
        pattern: Shard filename pattern.

    Returns:
        List of created shard paths.

    Example:
        >>> shards = create_step_tar_shards(
        ...     source_dir="/data/step_files",
        ...     output_dir="/data/shards",
        ...     samples_per_shard=1000,
        ... )
        >>> print(f"Created {len(shards)} shards")
    """
    import tarfile
    from pathlib import Path

    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all STEP files
    step_files = sorted(
        list(source_path.glob("**/*.step"))
        + list(source_path.glob("**/*.stp"))
        + list(source_path.glob("**/*.STEP"))
        + list(source_path.glob("**/*.STP"))
    )

    if not step_files:
        _log.warning("No STEP files found in %s", source_dir)
        return []

    _log.info("Creating shards from %d STEP files", len(step_files))

    shards = []
    shard_idx = 0
    current_tar = None
    count_in_shard = 0

    try:
        for step_file in step_files:
            # Start new shard if needed
            if current_tar is None:
                shard_name = pattern.format(shard_idx)
                shard_path = output_path / shard_name
                current_tar = tarfile.open(shard_path, "w")
                shards.append(shard_path)
                count_in_shard = 0

            # Add STEP file to shard
            sample_id = step_file.stem
            arcname = f"{sample_id}.step"
            current_tar.add(step_file, arcname=arcname)

            # Add metadata JSON if exists
            json_file = step_file.with_suffix(".json")
            if json_file.exists():
                json_arcname = f"{sample_id}.json"
                current_tar.add(json_file, arcname=json_arcname)

            count_in_shard += 1

            # Close shard if full
            if count_in_shard >= samples_per_shard:
                current_tar.close()
                current_tar = None
                shard_idx += 1

        # Close final shard
        if current_tar is not None:
            current_tar.close()

    except Exception as e:
        _log.error("Error creating shards: %s", e)
        if current_tar is not None:
            current_tar.close()
        raise

    _log.info("Created %d shards in %s", len(shards), output_dir)
    return shards


__all__ = [
    "STEPWebDatasetConfig",
    "STEPWebDataset",
    "STEPTarSample",
    "create_step_tar_shards",
]
