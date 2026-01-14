"""CADling - CAD document processing toolkit.

CADling is a docling-inspired toolkit for CAD files that enables optical CAD
recognition, STEP/STL code parsing, chunking, and understanding for LLM/ML/AI
compatibility.

Main Components:
    - DocumentConverter: Main entry point for CAD file conversion
    - InputFormat: Enum of supported CAD formats
    - CADlingDocument: Central data structure for processed CAD files
    - ConversionResult: Wrapper for conversion results

Example:
    from cadling import DocumentConverter, InputFormat

    converter = DocumentConverter(
        allowed_formats=[InputFormat.STEP, InputFormat.STL]
    )

    result = converter.convert("part.step")
    if result.status == ConversionStatus.SUCCESS:
        doc = result.document
        doc.export_to_json("output.json")
"""

__version__ = "0.1.0"

# Main API exports
from cadling.datamodel.base_models import (
    CADlingDocument,
    CADItem,
    ConversionResult,
    ConversionStatus,
    InputFormat,
)
from cadling.backend.document_converter import DocumentConverter, FormatOption

__all__ = [
    # Version
    "__version__",
    # Main converter
    "DocumentConverter",
    "FormatOption",
    # Data models
    "CADlingDocument",
    "CADItem",
    "ConversionResult",
    "ConversionStatus",
    "InputFormat",
]
