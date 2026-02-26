from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["ABCDataset", "load_abc"]

_log = logging.getLogger(__name__)

# Lazy imports
_torch = None
_datasets = None
_numpy = None
_pythonocc_available = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_datasets():
    global _datasets
    if _datasets is None:
        import datasets
        _datasets = datasets
    return _datasets


def _get_numpy():
    global _numpy
    if _numpy is None:
        import numpy as np
        _numpy = np
    return _numpy


def _check_pythonocc():
    """Check if pythonocc is available."""
    global _pythonocc_available
    if _pythonocc_available is None:
        try:
            from OCC.Core.TopoDS import TopoDS_Shape
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            _pythonocc_available = True
        except ImportError:
            _pythonocc_available = False
            _log.warning(
                "pythonocc not available. STEP files will be loaded as paths only."
            )
    return _pythonocc_available


def _extract_face_features(step_file: str) -> Dict[str, Any]:
    """Extract face features from a STEP file using pythonocc.

    Args:
        step_file: Path to STEP file.

    Returns:
        Dictionary with node_features and edge_index arrays.
    """
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer, TopExp
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

        np = _get_numpy()

        # Read STEP file
        reader = STEPControl_Reader()
        status = reader.ReadFile(step_file)

        if status != IFSelect_RetDone:
            _log.warning("Failed to read STEP file: %s", step_file)
            return {
                "node_features": None,
                "edge_index": None,
            }

        reader.TransferRoots()
        shape = reader.OneShape()

        # Extract faces
        faces = []
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)

        while face_explorer.More():
            face = face_explorer.Current()
            faces.append(face)
            face_explorer.Next()

        if not faces:
            return {
                "node_features": None,
                "edge_index": None,
            }

        # Extract features for each face
        features = []
        for face in faces:
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)

            # Surface area
            area = props.Mass()

            # Centroid
            centroid = props.CentreOfMass()
            centroid_array = np.array(
                [centroid.X(), centroid.Y(), centroid.Z()],
                dtype=np.float32,
            )

            # Bounding box
            bbox = Bnd_Box()
            brepbndlib.Add(face, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

            bbox_array = np.array(
                [xmin, ymin, zmin, xmax, ymax, zmax],
                dtype=np.float32,
            )

            # Surface type (simplified: plane=0, cylinder=1, sphere=2, other=3)
            surface_type = np.array([3], dtype=np.float32)

            feature_vector = np.concatenate([
                surface_type,
                np.array([area], dtype=np.float32),
                centroid_array,
                bbox_array,
            ])

            features.append(feature_vector)

        node_features = np.stack(features, axis=0)

        # Build edge index: connect faces that share an edge
        edge_map = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_map)

        # Build face-index lookup
        face_index = {}
        for idx, face in enumerate(faces):
            try:
                key = face.HashCode(2**31 - 1)
            except Exception:
                key = hash(face)
            face_index[key] = idx

        adjacency_set = set()
        for edge_idx in range(1, edge_map.Extent() + 1):
            adjacent_faces = edge_map.FindFromIndex(edge_idx)
            face_indices = []
            for fi in range(1, adjacent_faces.Extent() + 1):
                adj_face = adjacent_faces.Value(fi)
                try:
                    key = adj_face.HashCode(2**31 - 1)
                except Exception:
                    key = hash(adj_face)
                if key in face_index:
                    face_indices.append(face_index[key])
            # Connect all faces sharing this edge
            for a in range(len(face_indices)):
                for b in range(a + 1, len(face_indices)):
                    i_idx, j_idx = face_indices[a], face_indices[b]
                    if i_idx != j_idx:
                        adjacency_set.add((i_idx, j_idx))
                        adjacency_set.add((j_idx, i_idx))

        edge_index_list = sorted(adjacency_set)
        if edge_index_list:
            edge_index = np.array(edge_index_list, dtype=np.int64).T
        else:
            edge_index = np.array([], dtype=np.int64).reshape(2, 0)

        return {
            "node_features": node_features,
            "edge_index": edge_index,
        }

    except Exception as e:
        _log.warning("Error extracting features from %s: %s", step_file, e)
        return {
            "node_features": None,
            "edge_index": None,
        }


class ABCDataset:
    """PyTorch Dataset for ABC (Another B-rep Corpus) STEP files.

    This dataset loads STEP files and optionally extracts topological
    and geometric features using pythonocc.

    Attributes:
        data_dir: Directory containing STEP files.
        split: Dataset split ("train", "val", "test").
        max_samples: Maximum number of samples to load.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        """Initialize the ABC dataset.

        Args:
            data_dir: Path to directory containing STEP files.
            split: Dataset split to load.
            max_samples: Limit number of samples loaded.

        Raises:
            FileNotFoundError: If the split directory does not exist.
            ValueError: If no STEP files are found.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        self.pythonocc_available = _check_pythonocc()

        # Scan for STEP files
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Find both .step and .stp files
        step_files = list(split_dir.glob("**/*.step"))
        stp_files = list(split_dir.glob("**/*.stp"))
        self.step_files = sorted(step_files + stp_files)

        if not self.step_files:
            raise ValueError(f"No STEP files found in {split_dir}")

        if max_samples is not None:
            self.step_files = self.step_files[:max_samples]

        _log.info(
            f"Loaded {len(self.step_files)} ABC samples from {split}"
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.step_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
                - step_path: String path to STEP file
                - file_name: Name of the STEP file
                - file_size: Size of the file in bytes
                - node_features: Optional numpy array of face features
                - edge_index: Optional numpy array of face adjacency
                - metadata: Dictionary with file information
        """
        step_file = self.step_files[idx]

        # Get file info
        file_size = os.path.getsize(step_file)
        file_name = step_file.name

        result = {
            "step_path": str(step_file),
            "file_name": file_name,
            "file_size": file_size,
            "metadata": {
                "file_path": str(step_file),
                "split": self.split,
                "pythonocc_available": self.pythonocc_available,
            },
        }

        # Extract features if pythonocc is available
        if self.pythonocc_available:
            features = _extract_face_features(str(step_file))
            result["node_features"] = features.get("node_features")
            result["edge_index"] = features.get("edge_index")
        else:
            result["node_features"] = None
            result["edge_index"] = None

        return result


def _tokenize_abc_sample(
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    """Tokenize a single ABC sample from HuggingFace.

    Args:
        sample: Sample dictionary with STEP file information.

    Returns:
        Dictionary with tokenized output.
    """
    pythonocc_available = _check_pythonocc()

    result = {
        "step_path": sample.get("step_path", ""),
        "file_name": sample.get("file_name", ""),
        "file_size": sample.get("file_size", 0),
    }

    if pythonocc_available and "step_path" in sample:
        try:
            features = _extract_face_features(sample["step_path"])
            result["node_features"] = features.get("node_features")
            result["edge_index"] = features.get("edge_index")
        except Exception as e:
            _log.warning(f"Failed to extract features: {e}")
            result["node_features"] = None
            result["edge_index"] = None
    else:
        result["node_features"] = None
        result["edge_index"] = None

    return result


def load_abc(
    path: str = "latticelabs/abc",
    split: str = "train",
    streaming: bool = True,
    max_samples: Optional[int] = None,
) -> Any:
    """Load the ABC (Another B-rep Corpus) dataset.

    Loads either from a local directory or from the HuggingFace Hub,
    returning a PyTorch Dataset or HuggingFace IterableDataset.

    Args:
        path: Path to local directory or HuggingFace Hub ID.
        split: Dataset split to load ("train", "val", "test").
        streaming: Whether to stream from HuggingFace (if using Hub).
        max_samples: Maximum number of samples to load.

    Returns:
        PyTorch Dataset if path is local directory,
        HuggingFace IterableDataset if path is Hub ID.

    Raises:
        FileNotFoundError: If local path does not exist.
        ValueError: If no STEP files found in specified location.
    """
    # Check if path is a local directory
    if os.path.isdir(path):
        _log.info(f"Loading ABC from local directory: {path}")
        return ABCDataset(
            data_dir=path,
            split=split,
            max_samples=max_samples,
        )
    else:
        # Load from HuggingFace Hub
        _log.info(f"Loading ABC from HuggingFace Hub: {path}")
        datasets = _get_datasets()

        hf_dataset = datasets.load_dataset(
            path, split=split, streaming=streaming
        )

        if max_samples is not None:
            hf_dataset = hf_dataset.take(max_samples)

        tokenized_dataset = hf_dataset.map(
            lambda sample: _tokenize_abc_sample(sample),
            remove_columns=hf_dataset.column_names
            if hasattr(hf_dataset, "column_names")
            else [],
        )

        return tokenized_dataset
