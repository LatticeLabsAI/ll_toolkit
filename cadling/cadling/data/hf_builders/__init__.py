"""HuggingFace Dataset Builders for CAD data.

Provides ArrowBasedBuilder implementations for hosting CAD datasets
on the HuggingFace Hub with efficient Parquet storage.

Builders:
    CADCommandSequenceBuilder: Legacy standalone builder (batch conversion)
    BRepGraphBuilder: Legacy standalone builder (batch conversion)

True ArrowBasedBuilder (inherits from datasets.ArrowBasedBuilder):
    ArrowCADCommandSequenceBuilder: Proper HF ecosystem integration
    ArrowBRepGraphBuilder: Proper HF ecosystem integration

Use the Arrow* variants for:
- Native HF Hub streaming
- Automatic Arrow format handling
- Proper _generate_tables() implementation
"""
from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

# Lazy imports to avoid blocking on datasets library
_CADCommandSequenceBuilder = None
_BRepGraphBuilder = None
_ArrowCADCommandSequenceBuilder = None
_ArrowBRepGraphBuilder = None


def __getattr__(name: str):
    """Lazy load builders to avoid import overhead."""
    global _CADCommandSequenceBuilder, _BRepGraphBuilder
    global _ArrowCADCommandSequenceBuilder, _ArrowBRepGraphBuilder

    if name == "CADCommandSequenceBuilder":
        if _CADCommandSequenceBuilder is None:
            from .cad_dataset_builder import CADCommandSequenceBuilder
            _CADCommandSequenceBuilder = CADCommandSequenceBuilder
        return _CADCommandSequenceBuilder

    if name == "BRepGraphBuilder":
        if _BRepGraphBuilder is None:
            from .brep_graph_builder import BRepGraphBuilder
            _BRepGraphBuilder = BRepGraphBuilder
        return _BRepGraphBuilder

    # True ArrowBasedBuilder implementations
    if name == "ArrowCADCommandSequenceBuilder":
        if _ArrowCADCommandSequenceBuilder is None:
            from .arrow_command_sequence_builder import get_arrow_builder
            _ArrowCADCommandSequenceBuilder = get_arrow_builder
        return _ArrowCADCommandSequenceBuilder

    if name == "ArrowBRepGraphBuilder":
        if _ArrowBRepGraphBuilder is None:
            from .arrow_brep_builder import get_arrow_builder
            _ArrowBRepGraphBuilder = get_arrow_builder
        return _ArrowBRepGraphBuilder

    # Config classes
    if name == "CADCommandSequenceConfig":
        from .arrow_command_sequence_builder import CADCommandSequenceConfig
        return CADCommandSequenceConfig

    if name == "BRepGraphConfig":
        from .arrow_brep_builder import BRepGraphConfig
        return BRepGraphConfig

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Legacy builders (standalone)
    "CADCommandSequenceBuilder",
    "BRepGraphBuilder",
    # True ArrowBasedBuilder implementations
    "ArrowCADCommandSequenceBuilder",
    "ArrowBRepGraphBuilder",
    # Configuration classes
    "CADCommandSequenceConfig",
    "BRepGraphConfig",
]
