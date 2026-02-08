"""Dataset loaders for CAD research datasets.

Provides standardized access to major CAD generation research datasets:
DeepCAD (178K models), ABC (1M STEP files), Text2CAD (660K annotations),
and SketchGraphs (15M sketches).

Also provides HuggingFace Hub integration:
- Streaming datasets for petabyte-scale training
- Collators for efficient batching
- Publishers for uploading to HuggingFace Hub
- PyArrow schemas for CAD data types
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Type

_log = logging.getLogger(__name__)

# Re-export dataset classes for convenient access
from .datasets import (
    ABCLoader,
    BaseCADDataset,
    DeepCADLoader,
    SketchGraphsLoader,
    Text2CADLoader,
)

# Streaming and Hub integration (lazy-loaded)
from .streaming import (
    CADStreamingConfig,
    CADStreamingDataset,
    CADGraphStreamingDataset,
    create_streaming_dataloader,
    create_graph_streaming_dataloader,
)

from .collators import (
    CADCollatorConfig,
    CADCommandCollator,
    CADMultiModalCollator,
    CADGraphCollator,
)

from .hub_publisher import (
    PublishConfig,
    CADDatasetPublisher,
    publish_dataset,
)

from .schemas import (
    get_command_sequence_schema,
    get_brep_graph_schema,
    get_text_cad_schema,
    COMMAND_SEQUENCE_SCHEMA,
    BREP_GRAPH_SCHEMA,
    TEXT_CAD_SCHEMA,
    validate_sample,
    samples_to_table,
)

# WebDataset support for STEP TAR shards
from .webdataset import (
    STEPWebDatasetConfig,
    STEPWebDataset,
    STEPTarSample,
    create_step_tar_shards,
)

__all__ = [
    # Dataset loaders
    "ABCLoader",
    "BaseCADDataset",
    "DeepCADLoader",
    "SketchGraphsLoader",
    "Text2CADLoader",
    "get_dataset",
    # Streaming
    "CADStreamingConfig",
    "CADStreamingDataset",
    "CADGraphStreamingDataset",
    "create_streaming_dataloader",
    "create_graph_streaming_dataloader",
    # Collators
    "CADCollatorConfig",
    "CADCommandCollator",
    "CADMultiModalCollator",
    "CADGraphCollator",
    # Hub publishing
    "PublishConfig",
    "CADDatasetPublisher",
    "publish_dataset",
    # Schemas
    "get_command_sequence_schema",
    "get_brep_graph_schema",
    "get_text_cad_schema",
    "COMMAND_SEQUENCE_SCHEMA",
    "BREP_GRAPH_SCHEMA",
    "TEXT_CAD_SCHEMA",
    "validate_sample",
    "samples_to_table",
    # WebDataset for STEP TAR shards
    "STEPWebDatasetConfig",
    "STEPWebDataset",
    "STEPTarSample",
    "create_step_tar_shards",
]


def get_dataset(name: str, **kwargs: Any) -> Any:
    """Factory function to instantiate a dataset loader by name.

    Args:
        name: Dataset name ('deepcad', 'abc', 'text2cad', 'sketchgraphs').
        **kwargs: Arguments forwarded to the dataset constructor.

    Returns:
        Dataset instance.

    Raises:
        ValueError: If dataset name is unknown.
    """
    DATASETS: Dict[str, Type] = {
        "deepcad": DeepCADLoader,
        "abc": ABCLoader,
        "text2cad": Text2CADLoader,
        "sketchgraphs": SketchGraphsLoader,
    }

    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}"
        )

    _log.info("Loading dataset '%s'", name)
    return DATASETS[name](**kwargs)
