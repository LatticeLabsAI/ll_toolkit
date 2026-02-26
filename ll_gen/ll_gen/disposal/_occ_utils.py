"""Shared OCC utility functions for disposal modules.

Provides common helpers used by both ``validator`` and ``introspector``
to avoid code duplication.
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

_OCC_AVAILABLE = False
try:
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    _OCC_AVAILABLE = True
except ImportError:
    _log.debug("pythonocc not available; _occ_utils will not function")


def count_entities(shape: Any, topabs_type: Any) -> int:
    """Count unique entities of a given TopAbs type in a shape.

    Uses TopTools_IndexedMapOfShape to avoid counting duplicates
    (e.g., an edge shared by two faces should only be counted once).

    Args:
        shape: A ``TopoDS_Shape`` to inspect.
        topabs_type: A ``TopAbs_ShapeEnum`` value (e.g. ``TopAbs_FACE``).

    Returns:
        Number of unique entities of the given type.

    Raises:
        ImportError: If pythonocc is not installed.
    """
    if not _OCC_AVAILABLE:
        raise ImportError(
            "pythonocc-core is required for entity counting. "
            "Install with: conda install -c conda-forge pythonocc-core"
        )

    entity_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, topabs_type, entity_map)

    # Handle API differences between pythonocc versions
    if hasattr(entity_map, "Size"):
        return entity_map.Size()
    elif hasattr(entity_map, "Extent"):
        return entity_map.Extent()
    else:
        # Fallback to explorer counting
        count = 0
        explorer = TopExp_Explorer(shape, topabs_type)
        while explorer.More():
            count += 1
            explorer.Next()
        return count
