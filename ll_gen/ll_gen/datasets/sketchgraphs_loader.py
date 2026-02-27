from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["SketchGraphsDataset", "load_sketchgraphs"]

_log = logging.getLogger(__name__)

# Lazy imports
_torch = None
_datasets = None
_numpy = None


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


# Entity type encodings
ENTITY_TYPES = {
    "Line": 0,
    "Circle": 1,
    "Arc": 2,
    "Point": 3,
}

# Constraint type encodings
CONSTRAINT_TYPES = {
    "Coincident": 0,
    "Parallel": 1,
    "Perpendicular": 2,
    "Tangent": 3,
    "Equal": 4,
    "Distance": 5,
    "Angle": 6,
    "Concentric": 7,
}


class SketchGraphsDataset:
    """PyTorch Dataset for SketchGraphs constraint graphs.

    This dataset loads JSON files containing sketch entities and
    constraints, representing parametric sketches from Onshape.

    Attributes:
        data_dir: Directory containing JSON files organized by split.
        split: Dataset split ("train", "val", "test").
        max_entities: Maximum number of entities in a sketch.
        max_constraints: Maximum number of constraints.
        max_samples: Maximum number of samples to load.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_entities: int = 50,
        max_constraints: int = 100,
        max_samples: Optional[int] = None,
    ):
        """Initialize the SketchGraphs dataset.

        Args:
            data_dir: Path to directory containing SketchGraphs data.
            split: Dataset split to load ("train", "val", "test").
            max_entities: Maximum number of entities per sketch.
            max_constraints: Maximum number of constraints per sketch.
            max_samples: Limit number of samples loaded.

        Raises:
            FileNotFoundError: If split directory does not exist.
            ValueError: If no JSON files are found.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_entities = max_entities
        self.max_constraints = max_constraints
        self.max_samples = max_samples

        # Scan for JSON files
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.json_files = sorted(split_dir.glob("**/*.json"))
        if not self.json_files:
            raise ValueError(f"No JSON files found in {split_dir}")

        if max_samples is not None:
            self.json_files = self.json_files[:max_samples]

        _log.info(
            f"Loaded {len(self.json_files)} SketchGraphs samples from {split}"
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.json_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
                - entity_types: List of entity type IDs
                - entity_params: List of parameter lists (padded)
                - constraint_types: List of constraint type IDs
                - constraint_refs: List of entity reference pairs
                - constraint_values: List of constraint values
                - num_entities: Number of entities
                - num_constraints: Number of constraints
                - metadata: Dictionary with file path and other info
        """
        np = _get_numpy()
        json_file = self.json_files[idx]

        with open(json_file, "r") as f:
            data = json.load(f)

        entities = data.get("entities", [])
        constraints = data.get("constraints", [])

        # Process entities
        entity_types = []
        entity_params = []
        max_params_per_entity = 7

        for i, entity in enumerate(entities):
            if i >= self.max_entities:
                break

            entity_type = entity.get("type", "Point")
            if entity_type in ENTITY_TYPES:
                entity_types.append(ENTITY_TYPES[entity_type])
            else:
                entity_types.append(ENTITY_TYPES.get("Point", 3))

            params = entity.get("params", {})
            param_list = []

            # Extract parameters based on entity type
            if entity_type == "Line":
                start = params.get("start", [0.0, 0.0])
                end = params.get("end", [0.0, 0.0])
                param_list = list(start) + list(end)
            elif entity_type == "Circle":
                center = params.get("center", [0.0, 0.0])
                radius = params.get("radius", 0.0)
                param_list = list(center) + [radius]
            elif entity_type == "Arc":
                center = params.get("center", [0.0, 0.0])
                radius = params.get("radius", 0.0)
                start_angle = params.get("start_angle", 0.0)
                end_angle = params.get("end_angle", 0.0)
                param_list = (
                    list(center)
                    + [radius, start_angle, end_angle]
                )
            elif entity_type == "Point":
                point = params.get("point", [0.0, 0.0])
                param_list = list(point)

            # Pad or truncate to max_params_per_entity
            while len(param_list) < max_params_per_entity:
                param_list.append(0.0)
            param_list = param_list[:max_params_per_entity]

            entity_params.append(param_list)

        # Pad entities
        num_entities = len(entity_types)
        while len(entity_types) < self.max_entities:
            entity_types.append(ENTITY_TYPES["Point"])
            entity_params.append([0.0] * max_params_per_entity)

        entity_types = entity_types[: self.max_entities]
        entity_params = entity_params[: self.max_entities]

        # Process constraints
        constraint_types = []
        constraint_refs = []
        constraint_values = []

        for i, constraint in enumerate(constraints):
            if i >= self.max_constraints:
                break

            constraint_type = constraint.get("type", "Coincident")
            if constraint_type in CONSTRAINT_TYPES:
                constraint_types.append(CONSTRAINT_TYPES[constraint_type])
            else:
                constraint_types.append(CONSTRAINT_TYPES.get("Coincident", 0))

            # Get entity references
            references = constraint.get("references", [0, 0])
            if len(references) >= 2:
                constraint_refs.append(
                    [int(references[0]), int(references[1])]
                )
            else:
                constraint_refs.append([0, 0])

            # Get constraint value if applicable
            value = constraint.get("value", float("nan"))
            if isinstance(value, (int, float)):
                constraint_values.append(float(value))
            else:
                constraint_values.append(float("nan"))

        # Pad constraints
        num_constraints = len(constraint_types)
        while len(constraint_types) < self.max_constraints:
            constraint_types.append(CONSTRAINT_TYPES["Coincident"])
            constraint_refs.append([0, 0])
            constraint_values.append(float("nan"))

        constraint_types = constraint_types[: self.max_constraints]
        constraint_refs = constraint_refs[: self.max_constraints]
        constraint_values = constraint_values[: self.max_constraints]

        # Convert to numpy arrays
        entity_types_array = np.array(entity_types, dtype=np.int32)
        entity_params_array = np.array(entity_params, dtype=np.float32)
        constraint_types_array = np.array(constraint_types, dtype=np.int32)
        constraint_refs_array = np.array(constraint_refs, dtype=np.int32)
        constraint_values_array = np.array(constraint_values, dtype=np.float32)

        return {
            "entity_types": entity_types_array,
            "entity_params": entity_params_array,
            "constraint_types": constraint_types_array,
            "constraint_refs": constraint_refs_array,
            "constraint_values": constraint_values_array,
            "num_entities": num_entities,
            "num_constraints": num_constraints,
            "metadata": {
                "file_path": str(json_file),
                "split": self.split,
            },
        }


def _tokenize_sketchgraphs_sample(
    sample: Dict[str, Any],
    max_entities: int = 50,
    max_constraints: int = 100,
) -> Dict[str, Any]:
    """Tokenize a single SketchGraphs sample from HuggingFace.

    Args:
        sample: Sample dictionary with entities and constraints.
        max_entities: Maximum number of entities.
        max_constraints: Maximum number of constraints.

    Returns:
        Dictionary with tokenized output.
    """
    np = _get_numpy()

    entities = sample.get("entities", [])
    constraints = sample.get("constraints", [])

    # Process entities
    entity_types = []
    entity_params = []
    max_params_per_entity = 7

    for i, entity in enumerate(entities):
        if i >= max_entities:
            break

        entity_type = entity.get("type", "Point")
        if entity_type in ENTITY_TYPES:
            entity_types.append(ENTITY_TYPES[entity_type])
        else:
            entity_types.append(ENTITY_TYPES.get("Point", 3))

        params = entity.get("params", {})
        param_list = []

        if entity_type == "Line":
            start = params.get("start", [0.0, 0.0])
            end = params.get("end", [0.0, 0.0])
            param_list = list(start) + list(end)
        elif entity_type == "Circle":
            center = params.get("center", [0.0, 0.0])
            radius = params.get("radius", 0.0)
            param_list = list(center) + [radius]
        elif entity_type == "Arc":
            center = params.get("center", [0.0, 0.0])
            radius = params.get("radius", 0.0)
            start_angle = params.get("start_angle", 0.0)
            end_angle = params.get("end_angle", 0.0)
            param_list = (
                list(center)
                + [radius, start_angle, end_angle]
            )
        elif entity_type == "Point":
            point = params.get("point", [0.0, 0.0])
            param_list = list(point)

        while len(param_list) < max_params_per_entity:
            param_list.append(0.0)
        param_list = param_list[:max_params_per_entity]

        entity_params.append(param_list)

    num_entities = len(entity_types)
    while len(entity_types) < max_entities:
        entity_types.append(ENTITY_TYPES["Point"])
        entity_params.append([0.0] * max_params_per_entity)

    entity_types = entity_types[:max_entities]
    entity_params = entity_params[:max_entities]

    # Process constraints
    constraint_types = []
    constraint_refs = []
    constraint_values = []

    for i, constraint in enumerate(constraints):
        if i >= max_constraints:
            break

        constraint_type = constraint.get("type", "Coincident")
        if constraint_type in CONSTRAINT_TYPES:
            constraint_types.append(CONSTRAINT_TYPES[constraint_type])
        else:
            constraint_types.append(CONSTRAINT_TYPES.get("Coincident", 0))

        references = constraint.get("references", [0, 0])
        if len(references) >= 2:
            constraint_refs.append(
                [int(references[0]), int(references[1])]
            )
        else:
            constraint_refs.append([0, 0])

        value = constraint.get("value", float("nan"))
        if isinstance(value, (int, float)):
            constraint_values.append(float(value))
        else:
            constraint_values.append(float("nan"))

    num_constraints = len(constraint_types)
    while len(constraint_types) < max_constraints:
        constraint_types.append(CONSTRAINT_TYPES["Coincident"])
        constraint_refs.append([0, 0])
        constraint_values.append(float("nan"))

    constraint_types = constraint_types[:max_constraints]
    constraint_refs = constraint_refs[:max_constraints]
    constraint_values = constraint_values[:max_constraints]

    entity_types_array = np.array(entity_types, dtype=np.int32)
    entity_params_array = np.array(entity_params, dtype=np.float32)
    constraint_types_array = np.array(constraint_types, dtype=np.int32)
    constraint_refs_array = np.array(constraint_refs, dtype=np.int32)
    constraint_values_array = np.array(constraint_values, dtype=np.float32)

    return {
        "entity_types": entity_types_array,
        "entity_params": entity_params_array,
        "constraint_types": constraint_types_array,
        "constraint_refs": constraint_refs_array,
        "constraint_values": constraint_values_array,
        "num_entities": num_entities,
        "num_constraints": num_constraints,
    }


def load_sketchgraphs(
    path: str = "latticelabs/sketchgraphs",
    split: str = "train",
    streaming: bool = True,
    max_entities: int = 50,
    max_constraints: int = 100,
    max_samples: Optional[int] = None,
) -> Any:
    """Load the SketchGraphs dataset.

    Loads either from a local directory or from the HuggingFace Hub,
    returning a PyTorch Dataset or HuggingFace IterableDataset.

    Args:
        path: Path to local directory or HuggingFace Hub ID.
        split: Dataset split to load ("train", "val", "test").
        streaming: Whether to stream from HuggingFace (if using Hub).
        max_entities: Maximum number of entities per sketch.
        max_constraints: Maximum number of constraints per sketch.
        max_samples: Maximum number of samples to load.

    Returns:
        PyTorch Dataset if path is local directory,
        HuggingFace IterableDataset if path is Hub ID.

    Raises:
        FileNotFoundError: If local path does not exist.
        ValueError: If no JSON files found in specified location.
    """
    # Check if path is a local directory
    if os.path.isdir(path):
        _log.info(f"Loading SketchGraphs from local directory: {path}")
        return SketchGraphsDataset(
            data_dir=path,
            split=split,
            max_entities=max_entities,
            max_constraints=max_constraints,
            max_samples=max_samples,
        )
    else:
        # Load from HuggingFace Hub
        _log.info(f"Loading SketchGraphs from HuggingFace Hub: {path}")
        datasets = _get_datasets()

        hf_dataset = datasets.load_dataset(
            path, split=split, streaming=streaming
        )

        if max_samples is not None:
            hf_dataset = hf_dataset.take(max_samples)

        tokenized_dataset = hf_dataset.map(
            lambda sample: _tokenize_sketchgraphs_sample(
                sample,
                max_entities=max_entities,
                max_constraints=max_constraints,
            ),
            remove_columns=hf_dataset.column_names,
        )

        return tokenized_dataset
