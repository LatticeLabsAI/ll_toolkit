"""CAD file backends.

This package provides format-specific backends for parsing and processing
CAD files.

Available Backends:
    - AbstractCADBackend: Base backend class
    - DeclarativeCADBackend: For text-based parsing
    - RenderableCADBackend: For rendering support
    - CADViewBackend: View-specific rendering
    - DXFBackend: 2D DXF drawing parser
    - PDFBackend: PDF engineering drawing parser (vector + raster)
"""

from cadling.backend.abstract_backend import (
    AbstractCADBackend,
    CADViewBackend,
    DeclarativeCADBackend,
    RenderableCADBackend,
)
try:
    from cadling.backend.dxf_backend import DXFBackend
except ImportError:
    DXFBackend = None

try:
    from cadling.backend.pdf_backend import PDFBackend
except ImportError:
    PDFBackend = None

__all__ = [
    "AbstractCADBackend",
    "DeclarativeCADBackend",
    "RenderableCADBackend",
    "CADViewBackend",
    "DXFBackend",
    "PDFBackend",
]
