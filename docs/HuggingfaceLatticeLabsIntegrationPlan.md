# Hugging Face + LatticeLabs CAD Integration Plan

A comprehensive blueprint for making the LatticeLabs toolkit fully streaming-compliant and unified with Hugging Face's dataset ecosystem — enabling petabyte-scale CAD model training without local storage constraints.

---

## Executive Summary

**The Goal**: Transform LatticeLabs from a local-first CAD processing toolkit into a Hub-native system where:

- All CAD datasets stream directly from Hugging Face Hub
- Training runs on any scale without downloading entire datasets
- Generated datasets publish back to Hub with one command
- The existing training infrastructure works seamlessly with `datasets` library

**The Architecture**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Hugging Face Hub                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ deepcad-hf │  │  abc-hf    │  │text2cad-hf │  │ custom-cad │        │
│  │  (Parquet) │  │ (Parquet)  │  │ (Parquet)  │  │  (Parquet) │        │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘        │
│        │               │               │               │                │
│        └───────────────┴───────────────┴───────────────┘                │
│                                │                                        │
│                    HTTP Range Requests (streaming)                      │
└────────────────────────────────┼────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LatticeLabs Unified Data Layer                        │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    CADDatasetBuilder                              │   │
│  │  • Parquet schema for command sequences                          │   │
│  │  • Parquet schema for B-Rep graphs                               │   │
│  │  • Parquet schema for text-CAD pairs                             │   │
│  │  • WebDataset schema for STEP files + renders                    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│           ┌────────────────────┼────────────────────┐                   │
│           ▼                    ▼                    ▼                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │ IterableDataset │  │ IterableDataset │  │ IterableDataset │         │
│  │  (streaming)    │  │  (streaming)    │  │  (streaming)    │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                   │
│           └────────────────────┼────────────────────┘                   │
│                                ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    CADDataCollator                                │   │
│  │  • Pads command sequences to batch max                           │   │
│  │  • Batches graph structures (PyG batching)                       │   │
│  │  • Handles multi-modal (text + geometry)                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
└────────────────────────────────┼────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LatticeLabs Training Infrastructure                   │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │  VAETrainer  │  │ GANTrainer   │  │DiffusionTrain│                   │
│  │  (stepnet)   │  │  (stepnet)   │  │   (stepnet)  │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
│                                                                          │
│              All trainers accept IterableDataset directly                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Parquet Schema Design for CAD Data

The foundation is defining how CAD-specific data types serialize to Parquet columns. Each schema is designed for:

- **Streaming efficiency**: Column projection reduces bandwidth (only load needed columns)
- **Type safety**: PyArrow types preserve semantics
- **Compression**: Nested structures use efficient encodings

### 1.1 Command Sequence Schema (DeepCAD-style)

For sketch-and-extrude generation models:

```python
# Schema: latticelabs/cad-command-sequences
COMMAND_SEQUENCE_SCHEMA = {
    # Identifiers
    "model_id": pa.string(),           # Unique identifier
    "source": pa.string(),             # "deepcad", "fusion360", "onshape"
    
    # Command sequence (fixed-length padded)
    "command_types": pa.list_(pa.int8(), 60),      # [60] command type indices
    "parameters": pa.list_(pa.list_(pa.int16(), 16), 60),  # [60, 16] quantized params
    "mask": pa.list_(pa.float32(), 60),            # [60] validity mask
    "num_commands": pa.int32(),                     # Actual command count
    
    # Normalization metadata (for dequantization)
    "normalization_scale": pa.float32(),
    "normalization_offset": pa.list_(pa.float32(), 3),
    "quantization_bits": pa.int8(),
    
    # Optional: text annotations (for conditioned generation)
    "text_abstract": pa.string(),       # "two concentric cylinders"
    "text_detailed": pa.string(),       # Full description with dimensions
    
    # Optional: rendered views for image conditioning
    "render_front": pa.binary(),        # PNG bytes
    "render_side": pa.binary(),
    "render_iso": pa.binary(),
}
```

**Row Group Configuration**:

- Target 100 MB per row group (~50K sequences)
- Zstd compression level 3 (good balance)
- Dictionary encoding for `source` column

### 1.2 B-Rep Graph Schema (for GNN training)

For UV-Net, BRepNet, and graph-based models:

```python
# Schema: latticelabs/cad-brep-graphs
BREP_GRAPH_SCHEMA = {
    # Identifiers
    "model_id": pa.string(),
    "source": pa.string(),
    
    # Node features (faces)
    "num_faces": pa.int32(),
    "face_types": pa.list_(pa.int8()),              # Surface type indices
    "face_areas": pa.list_(pa.float32()),           # Normalized areas
    "face_centroids": pa.list_(pa.list_(pa.float32(), 3)),  # [N, 3]
    "face_normals": pa.list_(pa.list_(pa.float32(), 3)),    # [N, 3]
    
    # UV-Net style: sampled point grids per face
    "face_uv_grids": pa.list_(pa.list_(pa.float32(), 700)),  # [N, 10*10*7] flattened
    
    # Edge features
    "num_edges": pa.int32(),
    "edge_types": pa.list_(pa.int8()),
    "edge_lengths": pa.list_(pa.float32()),
    "dihedral_angles": pa.list_(pa.float32()),
    
    # Adjacency (COO format for PyG compatibility)
    "edge_index_row": pa.list_(pa.int32()),         # Source face indices
    "edge_index_col": pa.list_(pa.int32()),         # Target face indices
    
    # Optional: coedge structure for BRepNet
    "coedge_next": pa.list_(pa.int32()),
    "coedge_prev": pa.list_(pa.int32()),
    "coedge_mate": pa.list_(pa.int32()),
    
    # Labels (for segmentation tasks)
    "face_labels": pa.list_(pa.int8()),             # Manufacturing feature labels
}
```

**Streaming Optimization**:

- Store `face_uv_grids` separately for column projection (most expensive)
- Predicate pushdown on `num_faces` to filter by complexity

### 1.3 Text-CAD Pairs Schema (for conditioned generation)

```python
# Schema: latticelabs/text-to-cad
TEXT_CAD_SCHEMA = {
    "model_id": pa.string(),
    
    # Multi-level text annotations (Text2CAD style)
    "text_abstract": pa.string(),
    "text_intermediate": pa.string(),
    "text_detailed": pa.string(),
    "text_expert": pa.string(),
    
    # Command sequence (same as 1.1)
    "command_types": pa.list_(pa.int8(), 60),
    "parameters": pa.list_(pa.list_(pa.int16(), 16), 60),
    "mask": pa.list_(pa.float32(), 60),
    
    # CLIP/BERT embeddings (precomputed for efficiency)
    "text_embedding": pa.list_(pa.float32(), 768),  # BERT-base dim
    
    # Multi-view renders
    "renders": pa.list_(pa.binary(), 4),            # 4 views as PNG bytes
    "image_embedding": pa.list_(pa.float32(), 768), # DINOv2/CLIP embedding
}
```

### 1.4 WebDataset Schema for Raw STEP + Renders

For datasets with large STEP files and multiple renders per model:

```
shard-00000.tar
├── model_001.step          # Raw STEP file
├── model_001.json          # Metadata + command sequence
├── model_001.front.png     # Rendered views
├── model_001.side.png
├── model_001.iso.png
├── model_002.step
├── model_002.json
...
```

**Shard Configuration**:

- Target 1-2 GB per TAR shard
- ~500-1000 models per shard (depending on STEP complexity)
- Use for training image-to-CAD models

---

## Part 2: HF-Native Dataset Builders

Replace the current `BaseCADDataset` with HF `datasets` builders that support both local and streaming modes.

### 2.1 Base CAD Dataset Builder

**File**: `cadling/data/hf_builders/cad_dataset_builder.py`

```python
"""Base dataset builder for CAD data with HF datasets integration."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import (
    ArrowBasedBuilder,
    BuilderConfig,
    DatasetInfo,
    Features,
    Sequence,
    SplitGenerator,
    Value,
)
from datasets.download import DownloadManager

_log = logging.getLogger(__name__)


class CADCommandSequenceConfig(BuilderConfig):
    """Config for command sequence datasets."""
    
    def __init__(
        self,
        max_seq_len: int = 60,
        quantization_bits: int = 8,
        include_text: bool = False,
        include_renders: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.quantization_bits = quantization_bits
        self.include_text = include_text
        self.include_renders = include_renders


class CADCommandSequenceBuilder(ArrowBasedBuilder):
    """HF datasets builder for CAD command sequences.
    
    Supports:
    - Local JSON/Parquet files
    - Streaming from HF Hub
    - Automatic sharding on push_to_hub()
    
    Usage:
        # Local loading
        ds = load_dataset("latticelabs/deepcad-sequences", split="train")
        
        # Streaming
        ds = load_dataset(
            "latticelabs/deepcad-sequences",
            split="train",
            streaming=True,
        )
        for batch in ds.iter(batch_size=32):
            ...
    """
    
    BUILDER_CONFIGS = [
        CADCommandSequenceConfig(
            name="default",
            description="Standard 60-command sequences with 8-bit quantization",
        ),
        CADCommandSequenceConfig(
            name="with_text",
            include_text=True,
            description="Command sequences with text annotations",
        ),
        CADCommandSequenceConfig(
            name="with_renders",
            include_text=True,
            include_renders=True,
            description="Full multimodal: commands + text + renders",
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "default"
    
    def _info(self) -> DatasetInfo:
        """Define dataset features schema."""
        features = {
            "model_id": Value("string"),
            "command_types": Sequence(Value("int8"), length=self.config.max_seq_len),
            "parameters": Sequence(
                Sequence(Value("int16"), length=16),
                length=self.config.max_seq_len,
            ),
            "mask": Sequence(Value("float32"), length=self.config.max_seq_len),
            "num_commands": Value("int32"),
            "normalization_scale": Value("float32"),
            "quantization_bits": Value("int8"),
        }
        
        if self.config.include_text:
            features.update({
                "text_abstract": Value("string"),
                "text_detailed": Value("string"),
            })
        
        if self.config.include_renders:
            features.update({
                "render_front": Value("binary"),
                "render_side": Value("binary"),
                "render_iso": Value("binary"),
            })
        
        return DatasetInfo(
            description="CAD command sequences for generative modeling",
            features=Features(features),
            supervised_keys=None,
            homepage="https://github.com/latticelabs/toolkit",
            license="apache-2.0",
        )
    
    def _split_generators(
        self, dl_manager: DownloadManager
    ) -> List[SplitGenerator]:
        """Define data splits and their sources."""
        # This method handles both local paths and Hub URLs
        data_files = self.config.data_files or {}
        
        splits = []
        for split_name, files in data_files.items():
            downloaded = dl_manager.download_and_extract(files)
            splits.append(
                SplitGenerator(
                    name=split_name,
                    gen_kwargs={"files": downloaded},
                )
            )
        
        return splits
    
    def _generate_tables(
        self, files: List[str]
    ) -> Iterator[tuple[int, pa.Table]]:
        """Generate Arrow tables from source files.
        
        For ArrowBasedBuilder, this yields (key, pa.Table) tuples
        which are more efficient than row-by-row generation.
        """
        from geotoken.tokenizer.command_tokenizer import CommandSequenceTokenizer
        from geotoken.tokenizer.vocabulary import CADVocabulary
        
        tokenizer = CommandSequenceTokenizer()
        
        for file_idx, filepath in enumerate(files):
            if filepath.endswith(".parquet"):
                # Direct Parquet passthrough (already in correct format)
                table = pq.read_table(filepath)
                yield file_idx, table
                
            elif filepath.endswith(".json"):
                # Convert JSON to Arrow table
                records = self._process_json_file(filepath, tokenizer)
                if records:
                    table = pa.Table.from_pylist(records)
                    yield file_idx, table
    
    def _process_json_file(
        self, filepath: str, tokenizer
    ) -> List[Dict[str, Any]]:
        """Process DeepCAD-format JSON into records."""
        import json
        from pathlib import Path
        
        records = []
        path = Path(filepath)
        
        # Handle both single-file and directory structures
        if path.is_dir():
            json_files = list(path.glob("*.json"))
        else:
            json_files = [path]
        
        for json_path in json_files:
            try:
                with open(json_path) as f:
                    data = json.load(f)
                
                # Use geotoken's tokenizer for normalization + quantization
                token_seq = tokenizer.tokenize(data)
                
                record = {
                    "model_id": json_path.stem,
                    "command_types": [
                        ct.command_type.value for ct in token_seq.command_tokens
                    ],
                    "parameters": [
                        ct.parameters for ct in token_seq.command_tokens
                    ],
                    "mask": [
                        1.0 if ct.command_type.value != "EOS" else 0.0
                        for ct in token_seq.command_tokens
                    ],
                    "num_commands": token_seq.num_commands,
                    "normalization_scale": token_seq.metadata.get(
                        "normalization_range", 2.0
                    ),
                    "quantization_bits": self.config.quantization_bits,
                }
                records.append(record)
                
            except Exception as e:
                _log.debug("Skipping %s: %s", json_path, e)
        
        return records
```

### 2.2 B-Rep Graph Builder

**File**: `cadling/data/hf_builders/brep_graph_builder.py`

```python
"""HF datasets builder for B-Rep graph data."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List

import numpy as np
import pyarrow as pa
from datasets import (
    ArrowBasedBuilder,
    BuilderConfig,
    DatasetInfo,
    Features,
    Sequence,
    SplitGenerator,
    Value,
)

_log = logging.getLogger(__name__)


class BRepGraphConfig(BuilderConfig):
    """Config for B-Rep graph datasets."""
    
    def __init__(
        self,
        include_uv_grids: bool = True,
        include_coedges: bool = False,
        max_faces: int = 100,
        uv_grid_size: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_uv_grids = include_uv_grids
        self.include_coedges = include_coedges
        self.max_faces = max_faces
        self.uv_grid_size = uv_grid_size


class BRepGraphBuilder(ArrowBasedBuilder):
    """HF datasets builder for B-Rep face adjacency graphs.
    
    Processes STEP files into graph structures suitable for
    GNN training (UV-Net, BRepNet, GAT segmentation).
    
    Usage:
        ds = load_dataset(
            "latticelabs/abc-brep-graphs",
            split="train",
            streaming=True,
        )
        
        for example in ds:
            # Reconstruct PyG Data object
            data = pyg.data.Data(
                x=torch.tensor(example["face_features"]),
                edge_index=torch.tensor([
                    example["edge_index_row"],
                    example["edge_index_col"],
                ]),
            )
    """
    
    BUILDER_CONFIGS = [
        BRepGraphConfig(
            name="faces_only",
            include_uv_grids=False,
            description="Face adjacency graphs without UV sampling",
        ),
        BRepGraphConfig(
            name="with_uv",
            include_uv_grids=True,
            description="Full UV-Net style with sampled grids",
        ),
        BRepGraphConfig(
            name="with_coedges",
            include_uv_grids=True,
            include_coedges=True,
            description="BRepNet style with coedge pointers",
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "faces_only"
    
    def _info(self) -> DatasetInfo:
        uv_dim = self.config.uv_grid_size ** 2 * 7  # 10x10x7 = 700
        
        features = {
            "model_id": Value("string"),
            "num_faces": Value("int32"),
            "num_edges": Value("int32"),
            
            # Node features
            "face_types": Sequence(Value("int8")),
            "face_areas": Sequence(Value("float32")),
            "face_centroids": Sequence(Sequence(Value("float32"), length=3)),
            "face_normals": Sequence(Sequence(Value("float32"), length=3)),
            
            # Edge features
            "edge_types": Sequence(Value("int8")),
            "dihedral_angles": Sequence(Value("float32")),
            
            # Adjacency (COO format)
            "edge_index_row": Sequence(Value("int32")),
            "edge_index_col": Sequence(Value("int32")),
            
            # Labels
            "face_labels": Sequence(Value("int8")),
        }
        
        if self.config.include_uv_grids:
            features["face_uv_grids"] = Sequence(
                Sequence(Value("float32"), length=uv_dim)
            )
        
        if self.config.include_coedges:
            features.update({
                "coedge_next": Sequence(Value("int32")),
                "coedge_prev": Sequence(Value("int32")),
                "coedge_mate": Sequence(Value("int32")),
            })
        
        return DatasetInfo(
            description="B-Rep face adjacency graphs for GNN training",
            features=Features(features),
        )
    
    def _split_generators(self, dl_manager) -> List[SplitGenerator]:
        data_files = self.config.data_files or {}
        splits = []
        
        for split_name, files in data_files.items():
            downloaded = dl_manager.download_and_extract(files)
            splits.append(
                SplitGenerator(
                    name=split_name,
                    gen_kwargs={"files": downloaded},
                )
            )
        
        return splits
    
    def _generate_tables(
        self, files: List[str]
    ) -> Iterator[tuple[int, pa.Table]]:
        """Generate Arrow tables from STEP files."""
        from cadling.models.segmentation.brep_graph_builder import (
            BRepFaceGraphBuilder,
        )
        
        graph_builder = BRepFaceGraphBuilder(
            include_uv_samples=self.config.include_uv_grids,
            uv_grid_size=self.config.uv_grid_size,
        )
        
        for file_idx, filepath in enumerate(files):
            if filepath.endswith(".parquet"):
                table = pa.parquet.read_table(filepath)
                yield file_idx, table
                
            elif filepath.endswith((".step", ".stp", ".STEP", ".STP")):
                record = self._process_step_file(filepath, graph_builder)
                if record:
                    table = pa.Table.from_pylist([record])
                    yield file_idx, table
    
    def _process_step_file(
        self, filepath: str, graph_builder
    ) -> Dict[str, Any] | None:
        """Convert STEP file to graph record."""
        from pathlib import Path
        
        try:
            graph_data = graph_builder.build_from_step(filepath)
            
            if graph_data is None or graph_data.num_nodes == 0:
                return None
            
            if graph_data.num_nodes > self.config.max_faces:
                _log.debug(
                    "Skipping %s: too many faces (%d > %d)",
                    filepath, graph_data.num_nodes, self.config.max_faces,
                )
                return None
            
            record = {
                "model_id": Path(filepath).stem,
                "num_faces": int(graph_data.num_nodes),
                "num_edges": int(graph_data.num_edges),
                "face_types": graph_data.x[:, 0].tolist(),
                "face_areas": graph_data.x[:, 1].tolist(),
                "face_centroids": graph_data.x[:, 2:5].tolist(),
                "face_normals": graph_data.x[:, 5:8].tolist(),
                "edge_types": graph_data.edge_attr[:, 0].tolist(),
                "dihedral_angles": graph_data.edge_attr[:, 1].tolist(),
                "edge_index_row": graph_data.edge_index[0].tolist(),
                "edge_index_col": graph_data.edge_index[1].tolist(),
                "face_labels": (
                    graph_data.y.tolist()
                    if graph_data.y is not None
                    else [0] * graph_data.num_nodes
                ),
            }
            
            if self.config.include_uv_grids and hasattr(graph_data, "uv_grids"):
                record["face_uv_grids"] = graph_data.uv_grids.reshape(
                    graph_data.num_nodes, -1
                ).tolist()
            
            return record
            
        except Exception as e:
            _log.debug("Failed to process %s: %s", filepath, e)
            return None
```

---

## Part 3: Streaming-First Data Pipeline

### 3.1 Unified CAD Data Module

**File**: `cadling/data/streaming.py`

```python
"""Streaming-first data loading for CAD ML training.

This module provides a unified interface that wraps HF datasets
with CAD-specific preprocessing, batching, and distributed training support.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import DataLoader

_log = logging.getLogger(__name__)


@dataclass
class CADStreamingConfig:
    """Configuration for streaming CAD datasets.
    
    Attributes:
        dataset_id: HF Hub dataset ID or local path.
        split: Dataset split to load.
        streaming: Enable streaming mode (no local download).
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        shuffle_buffer_size: Buffer size for streaming shuffle.
        prefetch_factor: Batches to prefetch per worker.
        columns: Columns to load (None = all).
        filters: Parquet predicate pushdown filters.
    """
    dataset_id: str
    split: str = "train"
    streaming: bool = True
    batch_size: int = 32
    num_workers: int = 4
    shuffle_buffer_size: int = 10_000
    prefetch_factor: int = 2
    columns: Optional[List[str]] = None
    filters: Optional[List[tuple]] = None
    
    # Distributed training
    world_size: int = 1
    rank: int = 0
    
    # Preprocessing
    max_seq_len: int = 60
    pad_token_id: int = 5  # EOS token


class CADStreamingDataset:
    """Streaming dataset wrapper for CAD data.
    
    Wraps HF IterableDataset with:
    - Automatic column projection (bandwidth optimization)
    - Predicate pushdown for filtering
    - Distributed shard splitting
    - CAD-specific preprocessing
    
    Example:
        config = CADStreamingConfig(
            dataset_id="latticelabs/deepcad-sequences",
            batch_size=32,
            streaming=True,
        )
        dataset = CADStreamingDataset(config)
        
        for batch in dataset:
            command_types = batch["command_types"]  # [B, 60]
            parameters = batch["parameters"]        # [B, 60, 16]
            ...
    """
    
    def __init__(self, config: CADStreamingConfig):
        self.config = config
        self._dataset = None
        self._dataloader = None
        self._epoch = 0
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from HF Hub or local path."""
        from datasets import load_dataset
        
        load_kwargs = {
            "path": self.config.dataset_id,
            "split": self.config.split,
            "streaming": self.config.streaming,
        }
        
        # Column projection for bandwidth efficiency
        if self.config.columns:
            load_kwargs["columns"] = self.config.columns
        
        self._dataset = load_dataset(**load_kwargs)
        
        if self.config.streaming:
            self._setup_streaming()
        else:
            self._setup_mapped()
    
    def _setup_streaming(self):
        """Configure streaming mode with shuffle and sharding."""
        ds = self._dataset
        
        # Distributed: split shards across GPUs
        if self.config.world_size > 1:
            from datasets.distributed import split_dataset_by_node
            ds = split_dataset_by_node(
                ds,
                rank=self.config.rank,
                world_size=self.config.world_size,
            )
        
        # Shuffle with buffer
        if self.config.shuffle_buffer_size > 0:
            ds = ds.shuffle(
                seed=42 + self._epoch,
                buffer_size=self.config.shuffle_buffer_size,
            )
        
        # Apply preprocessing
        ds = ds.map(self._preprocess_example)
        
        self._dataset = ds
    
    def _setup_mapped(self):
        """Configure map-style dataset."""
        ds = self._dataset
        
        # Shuffle
        ds = ds.shuffle(seed=42 + self._epoch)
        
        # Preprocess
        ds = ds.map(
            self._preprocess_example,
            batched=False,
            num_proc=self.config.num_workers,
        )
        
        # Set PyTorch format
        ds = ds.with_format("torch")
        
        self._dataset = ds
    
    def _preprocess_example(
        self, example: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Preprocess a single example.
        
        Converts HF dataset columns to training-ready tensors.
        """
        import numpy as np
        
        # Ensure fixed sequence length
        cmd_types = example["command_types"]
        params = example["parameters"]
        mask = example["mask"]
        
        # Pad/truncate to max_seq_len
        if len(cmd_types) < self.config.max_seq_len:
            pad_len = self.config.max_seq_len - len(cmd_types)
            cmd_types = cmd_types + [self.config.pad_token_id] * pad_len
            params = params + [[0] * 16] * pad_len
            mask = mask + [0.0] * pad_len
        elif len(cmd_types) > self.config.max_seq_len:
            cmd_types = cmd_types[:self.config.max_seq_len]
            params = params[:self.config.max_seq_len]
            mask = mask[:self.config.max_seq_len]
        
        return {
            "command_types": np.array(cmd_types, dtype=np.int64),
            "parameters": np.array(params, dtype=np.int64),
            "mask": np.array(mask, dtype=np.float32),
            "num_commands": example.get("num_commands", len(cmd_types)),
        }
    
    def get_dataloader(self) -> DataLoader:
        """Get PyTorch DataLoader for the dataset."""
        if self._dataloader is not None:
            return self._dataloader
        
        if self.config.streaming:
            # IterableDataset: workers iterate different shards
            self._dataloader = DataLoader(
                self._dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                prefetch_factor=self.config.prefetch_factor,
                collate_fn=self._collate_fn,
            )
        else:
            # Map-style: standard DataLoader with shuffling
            self._dataloader = DataLoader(
                self._dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                prefetch_factor=self.config.prefetch_factor,
                collate_fn=self._collate_fn,
            )
        
        return self._dataloader
    
    def _collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Collate batch of examples into tensors."""
        return {
            "command_types": torch.stack([
                torch.tensor(ex["command_types"]) for ex in batch
            ]),
            "parameters": torch.stack([
                torch.tensor(ex["parameters"]) for ex in batch
            ]),
            "mask": torch.stack([
                torch.tensor(ex["mask"]) for ex in batch
            ]),
        }
    
    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling."""
        self._epoch = epoch
        if self.config.streaming and hasattr(self._dataset, "set_epoch"):
            self._dataset.set_epoch(epoch)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over batches."""
        return iter(self.get_dataloader())
    
    def __len__(self) -> int:
        """Approximate length (may be unknown for streaming)."""
        if self.config.streaming:
            # Streaming datasets don't have known length
            raise TypeError(
                "Streaming datasets don't support len(). "
                "Use dataset.info.num_examples if available."
            )
        return len(self._dataset) // self.config.batch_size


class CADGraphStreamingDataset(CADStreamingDataset):
    """Streaming dataset for B-Rep graph data.
    
    Extends CADStreamingDataset with PyTorch Geometric batching.
    """
    
    def _preprocess_example(
        self, example: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert to PyG-compatible format."""
        import numpy as np
        
        num_faces = example["num_faces"]
        
        # Stack node features
        face_features = np.column_stack([
            example["face_types"],
            example["face_areas"],
            np.array(example["face_centroids"]),
            np.array(example["face_normals"]),
        ]).astype(np.float32)
        
        # Edge index
        edge_index = np.array([
            example["edge_index_row"],
            example["edge_index_col"],
        ], dtype=np.int64)
        
        # Edge features
        edge_attr = np.column_stack([
            example["edge_types"],
            example["dihedral_angles"],
        ]).astype(np.float32)
        
        return {
            "x": face_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": np.array(example["face_labels"], dtype=np.int64),
            "num_nodes": num_faces,
        }
    
    def _collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> "torch_geometric.data.Batch":
        """Collate graphs into PyG Batch."""
        from torch_geometric.data import Batch, Data
        
        graphs = []
        for ex in batch:
            data = Data(
                x=torch.tensor(ex["x"]),
                edge_index=torch.tensor(ex["edge_index"]),
                edge_attr=torch.tensor(ex["edge_attr"]),
                y=torch.tensor(ex["y"]),
            )
            graphs.append(data)
        
        return Batch.from_data_list(graphs)
```

### 3.2 Data Collators for CAD

**File**: `cadling/data/collators.py`

```python
"""Data collators for CAD training batches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class CADCommandCollator:
    """Collator for command sequence batches.
    
    Handles:
    - Dynamic padding to batch max length
    - Attention mask creation
    - Label shifting for autoregressive training
    """
    
    pad_token_id: int = 5  # EOS
    max_length: Optional[int] = 60
    return_tensors: str = "pt"
    
    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        # Find max length in batch
        lengths = [len(f["command_types"]) for f in features]
        max_len = min(max(lengths), self.max_length or float("inf"))
        
        batch_size = len(features)
        
        # Initialize padded tensors
        command_types = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        parameters = torch.zeros(
            (batch_size, max_len, 16),
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.float,
        )
        
        for i, feat in enumerate(features):
            seq_len = min(len(feat["command_types"]), max_len)
            command_types[i, :seq_len] = torch.tensor(
                feat["command_types"][:seq_len]
            )
            parameters[i, :seq_len] = torch.tensor(
                feat["parameters"][:seq_len]
            )
            attention_mask[i, :seq_len] = 1.0
        
        # Labels for autoregressive: shift by 1
        labels = command_types.clone()
        labels[:, :-1] = command_types[:, 1:]
        labels[:, -1] = self.pad_token_id
        
        return {
            "command_types": command_types,
            "parameters": parameters,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class CADMultiModalCollator:
    """Collator for text + command sequence batches."""
    
    tokenizer: Any  # HF tokenizer for text
    pad_token_id: int = 5
    max_text_length: int = 128
    max_seq_length: int = 60
    
    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        # Collate text
        texts = [f.get("text_detailed", "") for f in features]
        text_encoding = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        
        # Collate commands
        cmd_collator = CADCommandCollator(
            pad_token_id=self.pad_token_id,
            max_length=self.max_seq_length,
        )
        cmd_batch = cmd_collator(features)
        
        return {
            **cmd_batch,
            "text_input_ids": text_encoding["input_ids"],
            "text_attention_mask": text_encoding["attention_mask"],
        }
```

---

## Part 4: Hub Publishing Pipeline

### 4.1 Dataset Publisher

**File**: `cadling/data/hub_publisher.py`

```python
"""Publish CAD datasets to Hugging Face Hub."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo, upload_file

_log = logging.getLogger(__name__)


@dataclass
class CADDatasetPublisher:
    """Publish CAD datasets to HF Hub with proper sharding.
    
    Handles:
    - Automatic sharding to ~500MB per file
    - Parquet schema enforcement
    - Dataset card generation
    - Incremental updates
    
    Example:
        publisher = CADDatasetPublisher(
            repo_id="latticelabs/deepcad-sequences",
            schema_type="command_sequences",
        )
        
        # Publish from local files
        publisher.publish_from_directory(
            source_dir="./deepcad_json",
            split="train",
        )
        
        # Or publish from generator
        publisher.publish_from_generator(
            generator=my_generator(),
            split="train",
            total_examples=178000,
        )
    """
    
    repo_id: str
    schema_type: str = "command_sequences"  # or "brep_graphs", "text_cad"
    shard_size_mb: int = 500
    private: bool = False
    token: Optional[str] = None
    
    def __post_init__(self):
        self.api = HfApi(token=self.token)
        self._schema = self._get_schema()
        self._shard_size_bytes = self.shard_size_mb * 1024 * 1024
    
    def _get_schema(self) -> pa.Schema:
        """Get PyArrow schema for the dataset type."""
        if self.schema_type == "command_sequences":
            return pa.schema([
                ("model_id", pa.string()),
                ("command_types", pa.list_(pa.int8(), 60)),
                ("parameters", pa.list_(pa.list_(pa.int16(), 16), 60)),
                ("mask", pa.list_(pa.float32(), 60)),
                ("num_commands", pa.int32()),
                ("normalization_scale", pa.float32()),
                ("quantization_bits", pa.int8()),
            ])
        elif self.schema_type == "brep_graphs":
            return pa.schema([
                ("model_id", pa.string()),
                ("num_faces", pa.int32()),
                ("num_edges", pa.int32()),
                ("face_types", pa.list_(pa.int8())),
                ("face_areas", pa.list_(pa.float32())),
                ("face_centroids", pa.list_(pa.list_(pa.float32(), 3))),
                ("face_normals", pa.list_(pa.list_(pa.float32(), 3))),
                ("edge_index_row", pa.list_(pa.int32())),
                ("edge_index_col", pa.list_(pa.int32())),
                ("face_labels", pa.list_(pa.int8())),
            ])
        else:
            raise ValueError(f"Unknown schema type: {self.schema_type}")
    
    def create_repo(self):
        """Create the dataset repository on Hub."""
        create_repo(
            repo_id=self.repo_id,
            repo_type="dataset",
            private=self.private,
            exist_ok=True,
        )
        _log.info("Created/verified repo: %s", self.repo_id)
    
    def publish_from_generator(
        self,
        generator: Iterator[Dict[str, Any]],
        split: str = "train",
        total_examples: Optional[int] = None,
    ):
        """Publish dataset from a generator with automatic sharding.
        
        Args:
            generator: Yields dicts matching the schema.
            split: Dataset split name.
            total_examples: Total count for progress (optional).
        """
        self.create_repo()
        
        shard_idx = 0
        current_batch: List[Dict[str, Any]] = []
        current_size = 0
        
        for i, record in enumerate(generator):
            current_batch.append(record)
            # Rough size estimate
            current_size += self._estimate_record_size(record)
            
            if current_size >= self._shard_size_bytes:
                self._write_and_upload_shard(
                    current_batch, split, shard_idx
                )
                shard_idx += 1
                current_batch = []
                current_size = 0
                
                if total_examples:
                    _log.info(
                        "Progress: %d/%d (%.1f%%)",
                        i + 1, total_examples, 100 * (i + 1) / total_examples,
                    )
        
        # Write final shard
        if current_batch:
            self._write_and_upload_shard(
                current_batch, split, shard_idx, is_final=True
            )
        
        # Update total shard count in filenames (cosmetic)
        total_shards = shard_idx + 1
        _log.info(
            "Published %s/%s: %d shards",
            self.repo_id, split, total_shards,
        )
    
    def _write_and_upload_shard(
        self,
        records: List[Dict[str, Any]],
        split: str,
        shard_idx: int,
        is_final: bool = False,
    ):
        """Write records to Parquet and upload to Hub."""
        import tempfile
        
        table = pa.Table.from_pylist(records, schema=self._schema)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            pq.write_table(
                table,
                f.name,
                compression="zstd",
                compression_level=3,
            )
            temp_path = f.name
        
        # Upload
        # Note: XXXXX-of-YYYYY is cosmetic; Hub loads all matching files
        filename = f"data/{split}-{shard_idx:05d}.parquet"
        
        self.api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo=filename,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Add {split} shard {shard_idx}",
        )
        
        Path(temp_path).unlink()
        _log.debug("Uploaded shard: %s", filename)
    
    def _estimate_record_size(self, record: Dict[str, Any]) -> int:
        """Rough byte size estimate for a record."""
        import json
        return len(json.dumps(record, default=str))
    
    def publish_dataset_card(
        self,
        description: str,
        citation: Optional[str] = None,
        license: str = "apache-2.0",
    ):
        """Generate and upload dataset card (README.md)."""
        card_content = f"""---
license: {license}
task_categories:
  - other
tags:
  - cad
  - 3d
  - generative
  - engineering
size_categories:
  - 100K<n<1M
---

# {self.repo_id.split("/")[-1]}

{description}

## Usage

```python
from datasets import load_dataset

# Load full dataset
ds = load_dataset("{self.repo_id}")

# Stream without downloading
ds = load_dataset("{self.repo_id}", streaming=True)
for example in ds["train"]:
    ...
```

## Schema

This dataset uses the `{self.schema_type}` schema from the LatticeLabs toolkit.

## Citation

{citation or "Citation information not provided."}
"""

        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(card_content)
            temp_path = f.name
        
        self.api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        
        Path(temp_path).unlink()
        _log.info("Uploaded dataset card")

```

---

## Part 5: Trainer Integration

### 5.1 Updated VAE Trainer with Streaming Support

**File**: `ll_stepnet/stepnet/training/streaming_vae_trainer.py`

```python
"""VAE trainer with HF datasets streaming support."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_log = logging.getLogger(__name__)


class StreamingVAETrainer:
    """VAE trainer that works with HF IterableDataset.
    
    Key differences from standard training:
    - No dataset length (can't compute steps per epoch)
    - Uses step-based rather than epoch-based scheduling
    - Supports checkpointing mid-stream
    - Handles distributed streaming
    
    Example:
        from cadling.data.streaming import CADStreamingDataset, CADStreamingConfig
        
        config = CADStreamingConfig(
            dataset_id="latticelabs/deepcad-sequences",
            streaming=True,
            batch_size=32,
        )
        dataset = CADStreamingDataset(config)
        
        trainer = StreamingVAETrainer(
            model=vae_model,
            dataset=dataset,
            total_steps=100_000,
            checkpoint_every=5000,
        )
        trainer.train()
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: "CADStreamingDataset",
        total_steps: int = 100_000,
        learning_rate: float = 1e-4,
        kl_weight: float = 1.0,
        kl_warmup_steps: int = 5000,
        checkpoint_every: int = 5000,
        checkpoint_dir: str = "./checkpoints",
        device: str = "cuda",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_every: int = 100,
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.kl_warmup_steps = kl_warmup_steps
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_every = log_every
        
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        
        self.global_step = 0
        self.epoch = 0
    
    def train(self):
        """Main training loop."""
        from pathlib import Path
        
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        self.model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        
        while self.global_step < self.total_steps:
            # Set epoch for shuffle reproducibility
            self.dataset.set_epoch(self.epoch)
            
            for batch in self.dataset:
                # Move to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                outputs = self.model(
                    command_types=batch["command_types"],
                    parameters=batch["parameters"],
                    mask=batch["mask"],
                )
                
                # Compute loss with KL warmup
                kl_weight = self._get_kl_weight()
                loss = outputs["recon_loss"] + kl_weight * outputs["kl_loss"]
                loss = loss / self.gradient_accumulation_steps
                
                # Backward
                loss.backward()
                
                # Accumulate
                running_loss += loss.item() * self.gradient_accumulation_steps
                running_recon += outputs["recon_loss"].item()
                running_kl += outputs["kl_loss"].item()
                
                # Step
                if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.log_every == 0:
                    avg_loss = running_loss / self.log_every
                    avg_recon = running_recon / self.log_every
                    avg_kl = running_kl / self.log_every
                    
                    _log.info(
                        "Step %d/%d | Loss: %.4f | Recon: %.4f | KL: %.4f | "
                        "KL_w: %.4f | LR: %.2e",
                        self.global_step, self.total_steps,
                        avg_loss, avg_recon, avg_kl,
                        kl_weight, self.scheduler.get_last_lr()[0],
                    )
                    
                    running_loss = 0.0
                    running_recon = 0.0
                    running_kl = 0.0
                
                # Checkpoint
                if self.global_step % self.checkpoint_every == 0:
                    self._save_checkpoint()
                
                if self.global_step >= self.total_steps:
                    break
            
            self.epoch += 1
        
        # Final checkpoint
        self._save_checkpoint(final=True)
        _log.info("Training complete: %d steps", self.global_step)
    
    def _get_kl_weight(self) -> float:
        """Compute KL weight with warmup."""
        if self.global_step >= self.kl_warmup_steps:
            return self.kl_weight
        return self.kl_weight * (self.global_step / self.kl_warmup_steps)
    
    def _save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        from pathlib import Path
        
        suffix = "final" if final else f"step_{self.global_step}"
        path = Path(self.checkpoint_dir) / f"checkpoint_{suffix}.pt"
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
        }, path)
        
        _log.info("Saved checkpoint: %s", path)
    
    def load_checkpoint(self, path: str):
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        
        _log.info(
            "Loaded checkpoint from %s (step %d)",
            path, self.global_step,
        )
```

---

## Part 6: CLI Commands

### 6.1 Hub Commands

**File**: `cadling/cli/hub.py`

```python
"""CLI commands for Hugging Face Hub integration."""
import click


@click.group()
def hub():
    """Hugging Face Hub dataset operations."""
    pass


@hub.command()
@click.argument("source_dir", type=click.Path(exists=True))
@click.option("--repo-id", required=True, help="Hub repo ID (user/dataset)")
@click.option("--split", default="train", help="Dataset split name")
@click.option(
    "--schema",
    type=click.Choice(["command_sequences", "brep_graphs", "text_cad"]),
    default="command_sequences",
)
@click.option("--private/--public", default=False)
def publish(source_dir, repo_id, split, schema, private):
    """Publish local CAD data to Hugging Face Hub.
    
    Example:
        cadling hub publish ./deepcad_json --repo-id myuser/deepcad-seqs
    """
    from cadling.data.hub_publisher import CADDatasetPublisher
    
    publisher = CADDatasetPublisher(
        repo_id=repo_id,
        schema_type=schema,
        private=private,
    )
    publisher.publish_from_directory(source_dir, split)
    click.echo(f"Published to: https://huggingface.co/datasets/{repo_id}")


@hub.command()
@click.option("--repo-id", required=True, help="Hub repo ID to stream from")
@click.option("--split", default="train")
@click.option("--batch-size", default=32, type=int)
@click.option("--max-batches", default=10, type=int, help="Batches to preview")
def preview(repo_id, split, batch_size, max_batches):
    """Preview a streamed dataset from Hub.
    
    Example:
        cadling hub preview --repo-id latticelabs/deepcad-sequences
    """
    from cadling.data.streaming import CADStreamingDataset, CADStreamingConfig
    
    config = CADStreamingConfig(
        dataset_id=repo_id,
        split=split,
        batch_size=batch_size,
        streaming=True,
    )
    dataset = CADStreamingDataset(config)
    
    click.echo(f"Streaming from: {repo_id}/{split}")
    click.echo(f"Batch size: {batch_size}")
    click.echo("-" * 40)
    
    for i, batch in enumerate(dataset):
        if i >= max_batches:
            break
        
        click.echo(f"Batch {i+1}:")
        click.echo(f"  command_types shape: {batch['command_types'].shape}")
        click.echo(f"  parameters shape: {batch['parameters'].shape}")
        click.echo(f"  mask shape: {batch['mask'].shape}")
    
    click.echo("-" * 40)
    click.echo(f"Previewed {min(i+1, max_batches)} batches")
```

---

## Part 7: Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [ ] Implement Parquet schemas in `cadling/data/schemas.py`
- [ ] Create `CADCommandSequenceBuilder` with JSON → Parquet conversion
- [ ] Create `CADStreamingDataset` wrapper
- [ ] Add unit tests for streaming pipeline

### Phase 2: Publishing Pipeline (Week 2-3)

- [ ] Implement `CADDatasetPublisher` with auto-sharding
- [ ] Add dataset card generation
- [ ] Create CLI commands for publish/preview
- [ ] Publish DeepCAD dataset to Hub as reference

### Phase 3: Training Integration (Week 3-4)

- [ ] Update `VAETrainer` to accept `IterableDataset`
- [ ] Update `DiffusionTrainer` for streaming
- [ ] Add checkpointing that works mid-stream
- [ ] Benchmark streaming vs local performance

### Phase 4: Advanced Features (Week 4-5)

- [ ] `BRepGraphBuilder` with PyG batching
- [ ] WebDataset support for STEP files
- [ ] Multi-modal collators (text + geometry)
- [ ] Distributed training validation

### Phase 5: Documentation & Polish (Week 5-6)

- [ ] Tutorial: "Training CAD Models with Streaming"
- [ ] Tutorial: "Publishing CAD Datasets to Hub"
- [ ] API documentation
- [ ] Performance benchmarks

---

## Appendix A: Key HF APIs Used

| API | Purpose | Location |
|-----|---------|----------|
| `load_dataset()` | Load datasets locally or streaming | `cadling/data/streaming.py` |
| `ArrowBasedBuilder` | Custom dataset builder | `cadling/data/hf_builders/*.py` |
| `IterableDataset` | Streaming dataset wrapper | Auto from `streaming=True` |
| `HfApi.upload_file()` | Publish shards to Hub | `cadling/data/hub_publisher.py` |
| `split_dataset_by_node()` | Distributed shard splitting | `cadling/data/streaming.py` |
| `DatasetCard.from_template()` | Generate README.md | `cadling/data/hub_publisher.py` |

## Appendix B: Shard Count Guidelines

| Training Setup | Recommended Shards |
|---------------|-------------------|
| 1 GPU, 4 workers | ≥4 shards |
| 4 GPUs, 4 workers each | ≥16 shards |
| 8 GPUs, 8 workers each | ≥64 shards |
| 64 GPUs (large cluster) | ≥256 shards |

Formula: `num_shards >= world_size × num_workers`

## Appendix C: Bandwidth Estimates

| Data Type | Per-Example Size | 1M Examples | Streaming Rate* |
|-----------|-----------------|-------------|-----------------|
| Command seq (60×16) | ~2 KB | ~2 GB | 100 MB/s |
| B-Rep graph (50 faces) | ~10 KB | ~10 GB | 20 MB/s |
| + UV grids (10×10×7) | ~150 KB | ~150 GB | 1.3 MB/s |
| + Renders (4×128KB) | ~650 KB | ~650 GB | 0.3 MB/s |

*At 100 Mbps connection, batch size 32
