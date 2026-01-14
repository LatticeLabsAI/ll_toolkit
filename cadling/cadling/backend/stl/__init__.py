"""STL backend components.

This module provides complete STL file processing capabilities:
- Parses both ASCII and binary STL files from scratch
- Extracts mesh data (vertices, normals, facets)
- Computes mesh properties (manifold, watertight, volume, surface area)
- Supports rendering via trimesh
"""

from cadling.backend.stl.stl_backend import STLBackend, STLViewBackend

__all__ = [
    "STLBackend",
    "STLViewBackend",
]
