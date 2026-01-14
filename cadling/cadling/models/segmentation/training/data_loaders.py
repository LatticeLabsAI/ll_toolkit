"""HuggingFace dataset loaders with streaming support.

Provides data loaders for CAD segmentation datasets with optional streaming
to avoid downloading entire datasets locally.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset, IterableDataset, Dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logger.warning("HuggingFace datasets not available. Install with: pip install datasets")

try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch torch-geometric")


class BaseDataLoader:
    """Base class for CAD dataset loaders with streaming support.

    Attributes:
        dataset_name: HuggingFace dataset name or local path
        streaming: Whether to use streaming (avoid full download)
        split: Dataset split ('train', 'val', 'test')
        cache_dir: Local cache directory for streamed data
        batch_size: Batch size for iteration
    """

    def __init__(
        self,
        dataset_name: str,
        streaming: bool = True,
        split: str = "train",
        cache_dir: Optional[Path] = None,
        batch_size: int = 1,
        **kwargs,
    ):
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("HuggingFace datasets required. Install: pip install datasets")

        self.dataset_name = dataset_name
        self.streaming = streaming
        self.split = split
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.kwargs = kwargs

        # Load dataset
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> IterableDataset | Dataset:
        """Load dataset from HuggingFace Hub or local path."""
        logger.info(f"Loading dataset: {self.dataset_name} (streaming={self.streaming})")

        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=self.streaming,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            **self.kwargs,
        )

        if self.streaming:
            logger.info("Dataset loaded in streaming mode (no full download)")
        else:
            logger.info(f"Dataset loaded: {len(dataset)} samples")

        return dataset

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over dataset samples."""
        if self.streaming:
            # Streaming dataset - iterate directly
            for sample in self.dataset:
                yield self._preprocess_sample(sample)
        else:
            # Non-streaming - can batch efficiently
            for i in range(0, len(self.dataset), self.batch_size):
                batch_end = min(i + self.batch_size, len(self.dataset))
                for j in range(i, batch_end):
                    yield self._preprocess_sample(self.dataset[j])

    def _preprocess_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Preprocess a single dataset sample.

        Override in subclasses for dataset-specific preprocessing.
        """
        return sample

    def take(self, n: int) -> Iterator[dict[str, Any]]:
        """Take first n samples from dataset."""
        if self.streaming:
            for i, sample in enumerate(self.dataset.take(n)):
                yield self._preprocess_sample(sample)
        else:
            for i in range(min(n, len(self.dataset))):
                yield self._preprocess_sample(self.dataset[i])

    def shuffle(self, seed: int = 42) -> "BaseDataLoader":
        """Shuffle dataset (returns new loader)."""
        if self.streaming:
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=10000)
        else:
            self.dataset = self.dataset.shuffle(seed=seed)
        return self


class MFCADDataLoader(BaseDataLoader):
    """Data loader for MFCAD++ dataset.

    Dataset: 15,488 STEP files with face-level manufacturing feature labels
    Features: 24 manufacturing feature classes
    Size: ~50GB

    Example:
        >>> loader = MFCADDataLoader(
        ...     dataset_name="path/to/mfcad",
        ...     streaming=True,
        ...     split="train"
        ... )
        >>> for sample in loader.take(10):
        ...     step_content = sample["step_content"]
        ...     face_labels = sample["face_labels"]
        ...     process_sample(step_content, face_labels)
    """

    FEATURE_CLASSES = [
        "base", "stock", "boss", "rib", "protrusion", "circular_boss",
        "rectangular_boss", "hex_boss", "pocket", "hole", "slot",
        "chamfer", "fillet", "groove", "through_hole", "blind_hole",
        "countersink", "counterbore", "round_pocket", "rectangular_pocket",
        "thread", "keyway", "dovetail", "t_slot", "o_ring_groove"
    ]

    def _preprocess_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Preprocess MFCAD sample.

        Extracts:
        - step_content: Raw STEP file content
        - entities: Parsed STEP entities dict (for graph construction)
        - face_labels: Per-face feature class indices
        - face_instances: Per-face instance IDs
        - feature_class_names: List of feature class names
        """
        # Parse face annotations
        faces = sample.get("faces", [])

        face_labels = []
        face_instances = []

        for face in faces:
            label_name = face.get("label", "unknown")
            if label_name in self.FEATURE_CLASSES:
                label_idx = self.FEATURE_CLASSES.index(label_name)
            else:
                label_idx = 0  # Default to "base"

            face_labels.append(label_idx)
            face_instances.append(face.get("instance_id", 0))

        # Parse STEP content to extract entities for graph construction
        # PARSER JOB: PARSE, not make up fake data!
        entities = None
        step_content = sample.get("step_content", "")
        if step_content:
            try:
                from cadling.backend.step import STEPTokenizer  # Fix: was non-existent STEPParser
                parser = STEPTokenizer()
                parsed = parser.parse_step_file(step_content)
                entities = parsed.get("entities", None)

                if entities is None or len(entities) == 0:
                    logger.error(
                        f"STEP parsing produced NO ENTITIES for {sample.get('file_name', 'unknown')}. "
                        f"Skipping this sample."
                    )
                    return None  # Skip sample - don't create fake data!

            except Exception as e:
                logger.error(
                    f"STEP parsing FAILED for {sample.get('file_name', 'unknown')}: {e}. "
                    f"Skipping this sample.",
                    exc_info=True
                )
                return None  # Skip sample - don't create fake data!

        # If we get here but still have no entities, skip
        if entities is None:
            logger.warning(f"No STEP content to parse for {sample.get('file_name', 'unknown')}. Skipping.")
            return None

        return {
            "file_name": sample.get("file_name", ""),
            "step_content": step_content,
            "entities": entities,  # Always valid if we reach here
            "face_labels": np.array(face_labels, dtype=np.int32),
            "face_instances": np.array(face_instances, dtype=np.int32),
            "num_faces": len(faces),
            "feature_class_names": self.FEATURE_CLASSES,
        }


class MFInstSegDataLoader(BaseDataLoader):
    """Data loader for MFInstSeg dataset.

    Dataset: 60,000+ STEP files with instance-level segmentation
    Features: Instance segmentation masks + boundaries
    Size: ~200GB

    Example:
        >>> loader = MFInstSegDataLoader(
        ...     dataset_name="path/to/mfinstseg",
        ...     streaming=True
        ... )
        >>> for sample in loader:
        ...     instance_masks = sample["instance_masks"]
        ...     boundary_masks = sample["boundary_masks"]
    """

    def _preprocess_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Preprocess MFInstSeg sample.

        Extracts:
        - step_content: Raw STEP file content
        - entities: Parsed STEP entities dict (for graph construction)
        - instance_masks: Per-face instance IDs
        - boundary_masks: Per-edge boundary flags
        - hierarchical_features: Parent-child feature relationships
        """
        instances = sample.get("instances", [])

        instance_masks = []
        for inst in instances:
            instance_masks.extend([inst.get("instance_id", 0)] * inst.get("num_faces", 1))

        # Parse STEP content to extract entities for graph construction
        # PARSER JOB: PARSE, not make up fake data!
        entities = None
        step_content = sample.get("step_content", "")
        if step_content:
            try:
                from cadling.backend.step import STEPTokenizer  # Fix: was non-existent STEPParser
                parser = STEPTokenizer()
                parsed = parser.parse_step_file(step_content)
                entities = parsed.get("entities", None)

                if entities is None or len(entities) == 0:
                    logger.error(
                        f"STEP parsing produced NO ENTITIES for {sample.get('file_name', 'unknown')}. "
                        f"Skipping this sample."
                    )
                    return None  # Skip sample - don't create fake data!

            except Exception as e:
                logger.error(
                    f"STEP parsing FAILED for {sample.get('file_name', 'unknown')}: {e}. "
                    f"Skipping this sample.",
                    exc_info=True
                )
                return None  # Skip sample - don't create fake data!

        # If we get here but still have no entities, skip
        if entities is None:
            logger.warning(f"No STEP content to parse for {sample.get('file_name', 'unknown')}. Skipping.")
            return None

        return {
            "file_name": sample.get("file_name", ""),
            "step_content": step_content,
            "entities": entities,  # Always valid if we reach here
            "instance_masks": np.array(instance_masks, dtype=np.int32),
            "boundary_masks": np.array(sample.get("boundary_edges", []), dtype=np.bool_),
            "num_instances": len(instances),
            "num_faces": len(instance_masks),  # Add num_faces for consistency
            "hierarchical_features": sample.get("hierarchy", {}),
        }


class ABCDataLoader(BaseDataLoader):
    """Data loader for ABC Dataset.

    Dataset: 1,000,000+ CAD models for mesh segmentation
    Features: Weak supervision from assembly hierarchies
    Size: ~1TB

    Best used with streaming to avoid full download.

    Example:
        >>> loader = ABCDataLoader(
        ...     dataset_name="abc-dataset/abc-meshes",
        ...     streaming=True,
        ...     split="train"
        ... )
        >>> for sample in loader.take(100):
        ...     mesh_data = sample["mesh"]
        ...     assembly_hierarchy = sample["hierarchy"]
    """

    def _preprocess_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Preprocess ABC sample.

        Extracts:
        - mesh_vertices: Vertex coordinates
        - mesh_faces: Face indices
        - assembly_hierarchy: Weak supervision from assembly structure
        """
        mesh_data = sample.get("mesh", {})

        return {
            "file_name": sample.get("file_name", ""),
            "mesh_vertices": np.array(mesh_data.get("vertices", []), dtype=np.float32),
            "mesh_faces": np.array(mesh_data.get("faces", []), dtype=np.int32),
            "assembly_hierarchy": sample.get("hierarchy", {}),
            "num_vertices": len(mesh_data.get("vertices", [])),
            "num_faces": len(mesh_data.get("faces", [])),
        }


class Fusion360DataLoader(BaseDataLoader):
    """Data loader for Fusion 360 Gallery dataset.

    Dataset: 8,625 CAD assemblies with B-Rep topology
    Features: Assembly structure, B-Rep graphs, parametric history
    Size: ~350GB

    Example:
        >>> loader = Fusion360DataLoader(
        ...     dataset_name="fusion360-gallery/assembly-dataset",
        ...     streaming=True
        ... )
        >>> for sample in loader:
        ...     brep_graph = sample["brep_graph"]
        ...     assembly_structure = sample["assembly"]
    """

    def _preprocess_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Preprocess Fusion 360 sample.

        Extracts:
        - brep_graph: B-Rep topology graph (nodes=faces, edges=adjacency)
        - assembly_structure: Assembly hierarchy
        - construction_sequence: Parametric construction history
        """
        return {
            "file_name": sample.get("file_name", ""),
            "brep_graph": sample.get("brep_graph", {}),
            "assembly_structure": sample.get("assembly", {}),
            "construction_sequence": sample.get("construction_history", []),
            "num_parts": len(sample.get("assembly", {}).get("parts", [])),
        }


def get_data_loader(
    dataset_name: str,
    dataset_type: str = "auto",
    streaming: bool = True,
    split: str = "train",
    cache_dir: Optional[Path] = None,
    batch_size: int = 1,
    **kwargs,
) -> BaseDataLoader:
    """Get appropriate data loader for dataset.

    Args:
        dataset_name: HuggingFace dataset name or local path
        dataset_type: Dataset type ('mfcad', 'mfinstseg', 'abc', 'fusion360', 'auto')
        streaming: Whether to use streaming mode
        split: Dataset split ('train', 'val', 'test')
        cache_dir: Local cache directory
        batch_size: Batch size for iteration
        **kwargs: Additional arguments for load_dataset

    Returns:
        Appropriate DataLoader instance

    Example:
        >>> loader = get_data_loader(
        ...     dataset_name="path/to/mfcad",
        ...     dataset_type="mfcad",
        ...     streaming=True
        ... )
        >>> for sample in loader.take(10):
        ...     process(sample)
    """
    # Auto-detect dataset type from name
    if dataset_type == "auto":
        dataset_name_lower = dataset_name.lower()
        if "mfcad" in dataset_name_lower:
            dataset_type = "mfcad"
        elif "mfinstseg" in dataset_name_lower or "inst" in dataset_name_lower:
            dataset_type = "mfinstseg"
        elif "abc" in dataset_name_lower:
            dataset_type = "abc"
        elif "fusion" in dataset_name_lower or "360" in dataset_name_lower:
            dataset_type = "fusion360"
        else:
            logger.warning(f"Could not auto-detect dataset type for '{dataset_name}', using base loader")
            dataset_type = "base"

    # Create appropriate loader
    loader_class = {
        "mfcad": MFCADDataLoader,
        "mfinstseg": MFInstSegDataLoader,
        "abc": ABCDataLoader,
        "fusion360": Fusion360DataLoader,
        "base": BaseDataLoader,
    }.get(dataset_type, BaseDataLoader)

    logger.info(f"Creating {loader_class.__name__} for '{dataset_name}'")

    return loader_class(
        dataset_name=dataset_name,
        streaming=streaming,
        split=split,
        cache_dir=cache_dir,
        batch_size=batch_size,
        **kwargs,
    )


# Example usage
if __name__ == "__main__":
    # Example: Stream MFCAD++ dataset without full download
    loader = MFCADDataLoader(
        dataset_name="path/to/mfcad",
        streaming=True,
        split="train",
    )

    print("Streaming MFCAD++ dataset...")
    for i, sample in enumerate(loader.take(5)):
        print(f"\nSample {i+1}:")
        print(f"  File: {sample['file_name']}")
        print(f"  Faces: {sample['num_faces']}")
        print(f"  Labels: {sample['face_labels'][:10]}...")
        print(f"  Feature classes: {len(sample['feature_class_names'])}")
