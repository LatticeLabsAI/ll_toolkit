"""ABC dataset loader.

Loads ABC dataset STEP files (1M models) and builds face adjacency
graphs suitable for GNN-based generation and segmentation tasks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .base_loader import BaseCADDataset

_log = logging.getLogger(__name__)


class ABCLoader(BaseCADDataset):
    """Dataset loader for the ABC dataset (1M STEP models).

    Parses STEP files, builds face adjacency graphs with geometric
    features, and filters invalid/trivial models.

    Supports two modes:
    1. Local mode: Load from STEP files in root_dir
    2. Hub mode: Stream from HuggingFace Hub (e.g., "latticelabs/abc-graphs")

    Args:
        root_dir: Path to ABC dataset directory (optional if hub_id provided).
        split: Dataset split ('train', 'val', 'test').
        max_faces: Maximum faces per model (filters larger models).
        min_faces: Minimum faces per model (filters trivial models).
        transform: Optional transform for each sample.
        download: Download dataset if not present.
        hub_id: HuggingFace Hub dataset ID (e.g., "latticelabs/abc-graphs").
        streaming: Whether to stream from Hub (default True for Hub mode).
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = "train",
        max_faces: int = 80,
        min_faces: int = 3,
        transform: Optional[Callable] = None,
        download: bool = False,
        hub_id: Optional[str] = None,
        streaming: bool = True,
    ) -> None:
        self.max_faces = max_faces
        self.min_faces = min_faces
        self._file_paths: list[Path] = []
        self._cached_samples: dict[int, dict[str, Any]] = {}

        super().__init__(root_dir, split, transform, download, hub_id, streaming)

        # Local mode only
        if not self._use_hub and self._verify_integrity():
            self._index_files()

    def __len__(self) -> int:
        return len(self._file_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a sample with face adjacency graph.

        Returns:
            Dict with:
                'node_features': [num_faces, 24] face feature array.
                'edge_index': [2, num_edges] adjacency array.
                'edge_attr': [num_edges, 8] edge feature array.
                'num_faces': int.
                'file_path': str.
        """
        if idx in self._cached_samples:
            sample = self._cached_samples[idx]
        else:
            sample = self._load_step_file(self._file_paths[idx])
            if sample is not None:
                self._cached_samples[idx] = sample

        if sample is None:
            # Return placeholder for invalid files
            sample = self._empty_graph()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def download(self) -> None:
        """Download ABC dataset.

        The ABC dataset should be downloaded from:
        https://deep-geometry.github.io/abc-dataset/
        """
        _log.info(
            "ABC dataset download not automated. Please download STEP "
            "files from https://deep-geometry.github.io/abc-dataset/ "
            "and place in %s/{train,val,test}/",
            self.root_dir,
        )

    def _verify_integrity(self) -> bool:
        """Check if STEP files exist in the split directory."""
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            # Try flat directory with split file
            return (self.root_dir / f"{self.split}.txt").exists() or any(
                self.root_dir.glob("*.step")
            )
        return any(split_dir.glob("*.step")) or any(split_dir.glob("*.stp"))

    def _index_files(self) -> None:
        """Build index of STEP files for this split."""
        split_dir = self.root_dir / self.split

        if split_dir.exists():
            self._file_paths = sorted(
                list(split_dir.glob("*.step")) + list(split_dir.glob("*.stp"))
            )
        else:
            # Try using split file for flat directory
            split_file = self.root_dir / f"{self.split}.txt"
            if split_file.exists():
                with open(split_file) as f:
                    names = [line.strip() for line in f if line.strip()]
                for name in names:
                    for ext in (".step", ".stp"):
                        p = self.root_dir / f"{name}{ext}"
                        if p.exists():
                            self._file_paths.append(p)
                            break
            else:
                self._file_paths = sorted(
                    list(self.root_dir.glob("*.step"))
                    + list(self.root_dir.glob("*.stp"))
                )

        _log.info("Indexed %d STEP files for split '%s'", len(self._file_paths), self.split)

    def _load_step_file(self, path: Path) -> Optional[dict[str, Any]]:
        """Load a STEP file and build face adjacency graph."""
        try:
            from cadling.models.segmentation.brep_graph_builder import BRepFaceGraphBuilder

            builder = BRepFaceGraphBuilder()

            # Create a minimal document wrapper
            from cadling.datamodel.base_models import CADlingDocument

            doc = CADlingDocument()
            doc.properties = {"step_text": path.read_text(errors="replace")}

            graph_data = builder.build_face_graph(doc)

            num_faces = graph_data.num_nodes
            if num_faces < self.min_faces or num_faces > self.max_faces:
                return None

            return {
                "node_features": graph_data.x.numpy() if hasattr(graph_data.x, 'numpy') else np.zeros((num_faces, 24)),
                "edge_index": graph_data.edge_index.numpy() if hasattr(graph_data.edge_index, 'numpy') else np.zeros((2, 0), dtype=np.int64),
                "edge_attr": graph_data.edge_attr.numpy() if graph_data.edge_attr is not None and hasattr(graph_data.edge_attr, 'numpy') else np.zeros((0, 8)),
                "num_faces": num_faces,
                "file_path": str(path),
            }
        except Exception as e:
            _log.debug("Failed to load %s: %s", path.name, e)
            return None

    def _empty_graph(self) -> dict[str, Any]:
        """Return a placeholder empty graph for invalid files."""
        return {
            "node_features": np.zeros((1, 24), dtype=np.float32),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_attr": np.zeros((0, 8), dtype=np.float32),
            "num_faces": 0,
            "file_path": "",
        }

    def _process_hub_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Process a sample from Hub format to loader format.

        Hub format is pre-processed with:
        - face_features: [num_faces * feat_dim] float32 array (flat)
        - edge_index: [2 * num_edges] int64 array (flat)
        - edge_features: [num_edges * edge_feat_dim] float32 array (flat)
        - num_faces: int
        - num_edges: int
        - face_feature_dim: int
        - edge_feature_dim: int
        """
        num_faces = sample.get("num_faces", 0)
        num_edges = sample.get("num_edges", 0)
        face_feat_dim = sample.get("face_feature_dim", 24)
        edge_feat_dim = sample.get("edge_feature_dim", 8)

        # Get face features
        face_features = sample.get("face_features", [])
        if isinstance(face_features, list):
            face_features = np.array(face_features, dtype=np.float32)
        if face_features.ndim == 1 and num_faces > 0:
            face_features = face_features[:num_faces * face_feat_dim].reshape(num_faces, face_feat_dim)

        # Get edge index
        edge_index = sample.get("edge_index", [])
        if isinstance(edge_index, list):
            edge_index = np.array(edge_index, dtype=np.int64)
        if edge_index.ndim == 1:
            edge_index = edge_index.reshape(2, -1)

        # Get edge features
        edge_attr = sample.get("edge_features", [])
        if isinstance(edge_attr, list):
            edge_attr = np.array(edge_attr, dtype=np.float32)
        if edge_attr.ndim == 1 and num_edges > 0:
            edge_attr = edge_attr[:num_edges * edge_feat_dim].reshape(num_edges, edge_feat_dim)

        return {
            "node_features": face_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_faces": num_faces,
            "file_path": sample.get("source_path", sample.get("sample_id", "")),
        }
