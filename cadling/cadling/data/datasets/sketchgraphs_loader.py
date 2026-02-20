"""SketchGraphs dataset loader.

Loads SketchGraphs' 15M parametric sketches as constraint graphs where
nodes represent sketch primitives and edges represent designer-imposed
geometric constraints.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .base_loader import BaseCADDataset

_log = logging.getLogger(__name__)


class SketchGraphsLoader(BaseCADDataset):
    """Dataset loader for SketchGraphs parametric sketch constraint graphs.

    Each sample is a graph where nodes are sketch primitives (lines, arcs,
    circles, points) and edges are geometric constraints (coincident,
    tangent, perpendicular, parallel, etc.).

    Supports two modes:
    1. Local mode: Load from JSON/JSONL files in root_dir
    2. Hub mode: Stream from HuggingFace Hub (e.g., "latticelabs/sketchgraphs")

    Args:
        root_dir: Path to SketchGraphs dataset directory (optional if hub_id provided).
        split: Dataset split ('train', 'val', 'test').
        max_primitives: Maximum primitives per sketch.
        max_constraints: Maximum constraints per sketch.
        transform: Optional transform for each sample.
        download: Download if not present.
        hub_id: HuggingFace Hub dataset ID (e.g., "latticelabs/sketchgraphs").
        streaming: Whether to stream from Hub (default True for Hub mode).
    """

    # Primitive types
    PRIMITIVE_TYPES = {
        "Point": 0, "Line": 1, "Arc": 2, "Circle": 3,
    }
    NUM_PRIMITIVE_TYPES = 4

    # Constraint types
    CONSTRAINT_TYPES = {
        "Coincident": 0, "Tangent": 1, "Perpendicular": 2,
        "Parallel": 3, "Concentric": 4, "Equal": 5,
        "Distance": 6, "Angle": 7, "Horizontal": 8,
        "Vertical": 9, "Midpoint": 10, "Fix": 11,
    }
    NUM_CONSTRAINT_TYPES = 12

    # Node feature dimension: type one-hot (4) + params (8) = 12
    NODE_FEATURE_DIM = 12
    # Edge feature dimension: constraint type one-hot (12) + value (1) = 13
    EDGE_FEATURE_DIM = 13

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = "train",
        max_primitives: int = 50,
        max_constraints: int = 100,
        transform: Optional[Callable] = None,
        download: bool = False,
        hub_id: Optional[str] = None,
        streaming: bool = True,
    ) -> None:
        self.max_primitives = max_primitives
        self.max_constraints = max_constraints

        self._samples: list[dict[str, Any]] = []

        super().__init__(root_dir, split, transform, download, hub_id, streaming)

        # Local mode only
        if not self._use_hub and self._verify_integrity():
            self._load_samples()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a constraint graph sample.

        Returns:
            Dict with:
                'x': [num_primitives, NODE_FEATURE_DIM] node features.
                'edge_index': [2, num_edges] constraint edges.
                'edge_attr': [num_edges, EDGE_FEATURE_DIM] constraint features.
                'constraint_types': [num_edges] constraint type indices.
                'num_primitives': int.
                'num_constraints': int.
        """
        sample = self._samples[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def download(self) -> None:
        """Download SketchGraphs dataset (15M parametric sketches).

        Attempts automated download in order of preference:

        1. **HuggingFace Hub** via ``huggingface_hub.snapshot_download``.
        2. **S3 pre-processed sequences** from the PrincetonLIPS release.
        3. **Manual fallback** — logs the URL and expected directory layout.

        After download, JSON/JSONL files are placed in ``root_dir/{split}/``.
        """
        if self._verify_integrity():
            _log.info("SketchGraphs dataset already present at %s", self.root_dir)
            return

        self.root_dir.mkdir(parents=True, exist_ok=True)

        # --- Strategy 1: HuggingFace Hub ---
        try:
            from huggingface_hub import snapshot_download

            _log.info("Downloading SketchGraphs from HuggingFace Hub...")
            snapshot_download(
                repo_id="PrincetonLIPS/SketchGraphs",
                repo_type="dataset",
                local_dir=str(self.root_dir),
                allow_patterns=["*.json", "*.jsonl", "*.npy"],
            )

            if self._verify_integrity():
                _log.info("SketchGraphs downloaded via HuggingFace Hub")
                return

        except ImportError:
            _log.debug("huggingface_hub not installed")
        except Exception as exc:
            _log.warning("HuggingFace Hub download failed: %s", exc)

        # --- Strategy 2: Direct S3 download (pre-processed sequences) ---
        try:
            import os
            import tempfile
            import urllib.request

            url = (
                "https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_train.npy"
            )
            _log.info("Downloading SketchGraphs sequences from %s", url)

            (self.root_dir / self.split).mkdir(parents=True, exist_ok=True)
            dest = self.root_dir / self.split / "sg_t16_train.npy"

            urllib.request.urlretrieve(url, str(dest))

            if self._verify_integrity():
                _log.info("SketchGraphs sequences downloaded to %s", dest)
                return

        except Exception as exc:
            _log.warning("Direct S3 download failed: %s", exc)

        # --- Strategy 3: Manual instructions ---
        _log.info(
            "Automated download unsuccessful. Please download from "
            "https://github.com/PrincetonLIPS/SketchGraphs and place "
            "data in %s/",
            self.root_dir,
        )

    def _verify_integrity(self) -> bool:
        """Check if dataset files exist."""
        split_dir = self.root_dir / self.split
        if split_dir.exists():
            return any(split_dir.iterdir())
        # Check for flat files
        return (
            any(self.root_dir.glob("*.json"))
            or any(self.root_dir.glob("*.jsonl"))
            or any(self.root_dir.glob("*.npy"))
        )

    def _load_samples(self) -> None:
        """Load and process sketch constraint graphs."""
        split_dir = self.root_dir / self.split

        if split_dir.exists():
            search_dir = split_dir
        else:
            search_dir = self.root_dir

        # Try JSONL format
        for jsonl_path in sorted(search_dir.glob("*.jsonl")):
            self._load_jsonl(jsonl_path)

        # Try JSON format
        if not self._samples:
            for json_path in sorted(search_dir.glob("*.json")):
                self._load_json(json_path)

        _log.info("Loaded %d SketchGraphs samples", len(self._samples))

    def _load_jsonl(self, path: Path) -> None:
        """Load from JSONL file."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample = self._process_sketch(data)
                    if sample is not None:
                        self._samples.append(sample)
                except (json.JSONDecodeError, KeyError) as e:
                    _log.debug("Skipping line: %s", e)

    def _load_json(self, path: Path) -> None:
        """Load from JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    sample = self._process_sketch(item)
                    if sample is not None:
                        self._samples.append(sample)
        except (json.JSONDecodeError, KeyError) as e:
            _log.debug("Skipping %s: %s", path.name, e)

    def _process_sketch(
        self, data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Process a single sketch into a constraint graph."""
        primitives = data.get("entities", data.get("primitives", []))
        constraints = data.get("constraints", [])

        num_prims = len(primitives)
        num_cons = len(constraints)

        if num_prims < 2 or num_prims > self.max_primitives:
            return None
        if num_cons > self.max_constraints:
            return None

        # Build node features
        node_features = np.zeros((num_prims, self.NODE_FEATURE_DIM), dtype=np.float32)

        for i, prim in enumerate(primitives):
            prim_type = prim.get("type", "Point")
            type_idx = self.PRIMITIVE_TYPES.get(prim_type, 0)

            # One-hot type encoding (first 4 dims)
            node_features[i, type_idx] = 1.0

            # Parameters (dims 4-11): coordinates, radius, angles
            params = prim.get("params", prim.get("parameters", {}))
            if isinstance(params, dict):
                param_values = list(params.values())
            elif isinstance(params, list):
                param_values = params
            else:
                param_values = []

            for j, val in enumerate(param_values[:8]):
                node_features[i, 4 + j] = float(val)

        # Build edge index and features from constraints
        edge_src: list[int] = []
        edge_dst: list[int] = []
        edge_features_list: list[np.ndarray] = []
        constraint_type_list: list[int] = []

        for con in constraints:
            con_type = con.get("type", "Coincident")
            type_idx = self.CONSTRAINT_TYPES.get(con_type, 0)

            # Get referenced primitives
            refs = con.get("references", con.get("entities", []))
            if len(refs) < 2:
                # Self-constraint (like Fix, Horizontal, Vertical)
                if refs:
                    src = refs[0] if isinstance(refs[0], int) else 0
                    dst = src
                else:
                    continue
            else:
                src = refs[0] if isinstance(refs[0], int) else 0
                dst = refs[1] if isinstance(refs[1], int) else 0

            if src >= num_prims or dst >= num_prims:
                continue

            # Bidirectional edges
            for s, d in [(src, dst), (dst, src)]:
                edge_src.append(s)
                edge_dst.append(d)

                # Edge features: constraint type one-hot + value
                feat = np.zeros(self.EDGE_FEATURE_DIM, dtype=np.float32)
                feat[type_idx] = 1.0

                # Constraint value (for Distance/Angle constraints)
                con_value = con.get("value", con.get("param", 0.0))
                feat[-1] = float(con_value) if con_value else 0.0

                edge_features_list.append(feat)
                constraint_type_list.append(type_idx)

        if edge_src:
            edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
            edge_attr = np.stack(edge_features_list)
            constraint_types = np.array(constraint_type_list, dtype=np.int64)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, self.EDGE_FEATURE_DIM), dtype=np.float32)
            constraint_types = np.array([], dtype=np.int64)

        return {
            "x": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "constraint_types": constraint_types,
            "num_primitives": num_prims,
            "num_constraints": num_cons,
        }

    def _process_hub_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Process a sample from Hub format to loader format.

        Hub format has:
        - x: [num_primitives * NODE_FEATURE_DIM] float32 array (flat)
        - edge_index: [2 * num_edges] int64 array (flat)
        - edge_attr: [num_edges * EDGE_FEATURE_DIM] float32 array (flat)
        - constraint_types: [num_edges] int64 array
        - num_primitives: int
        - num_constraints: int
        """
        num_prims = sample.get("num_primitives", 0)
        num_cons = sample.get("num_constraints", 0)

        # Get node features
        x = sample.get("x", sample.get("node_features", []))
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)
        if x.ndim == 1 and num_prims > 0:
            x = x[:num_prims * self.NODE_FEATURE_DIM].reshape(num_prims, self.NODE_FEATURE_DIM)

        # Get edge index
        edge_index = sample.get("edge_index", [])
        if isinstance(edge_index, list):
            edge_index = np.array(edge_index, dtype=np.int64)
        if edge_index.ndim == 1 and edge_index.size > 0:
            edge_index = edge_index.reshape(2, -1)

        # Get edge features
        edge_attr = sample.get("edge_attr", [])
        if isinstance(edge_attr, list):
            edge_attr = np.array(edge_attr, dtype=np.float32)
        num_edges = edge_index.shape[1] if edge_index.ndim == 2 else 0
        if edge_attr.ndim == 1 and num_edges > 0:
            edge_attr = edge_attr[:num_edges * self.EDGE_FEATURE_DIM].reshape(num_edges, self.EDGE_FEATURE_DIM)

        # Get constraint types
        constraint_types = sample.get("constraint_types", [])
        if isinstance(constraint_types, list):
            constraint_types = np.array(constraint_types, dtype=np.int64)

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "constraint_types": constraint_types,
            "num_primitives": num_prims,
            "num_constraints": num_cons,
        }
