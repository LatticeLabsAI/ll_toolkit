"""BRep backend components.

This module provides BRep (OpenCASCADE Boundary Representation) file processing:
- Loads BRep files using pythonocc-core
- Extracts topology (solids, shells, faces, edges, vertices)
- Computes geometric properties (volume, surface area, bounding box)
- Supports rendering from multiple views
"""

from cadling.backend.brep.brep_backend import BRepBackend, BRepViewBackend

__all__ = [
    "BRepBackend",
    "BRepViewBackend",
]
