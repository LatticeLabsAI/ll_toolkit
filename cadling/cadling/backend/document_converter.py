"""Main document converter for CAD files.

This module provides the primary entry point for converting CAD files to
CADlingDocument, similar to docling's DocumentConverter but adapted for
CAD formats.

Classes:
    FormatOption: Configuration for a specific CAD format.
    DocumentConverter: Main converter class.

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

from __future__ import annotations

import hashlib
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from cadling.backend.abstract_backend import AbstractCADBackend
from cadling.datamodel.base_models import (
    CADInputDocument,
    ConversionResult,
    InputFormat,
)
from cadling.datamodel.backend_options import BackendOptions
from cadling.datamodel.pipeline_options import PipelineOptions
from cadling.models.base_model import EnrichmentModel
from cadling.pipeline.base_pipeline import BaseCADPipeline

_log = logging.getLogger(__name__)


class FormatOption(BaseModel):
    """Configuration for a specific CAD format.

    This defines the backend and pipeline to use for a format, similar to
    docling's FormatOption.

    Attributes:
        backend: Backend class for this format.
        pipeline_cls: Pipeline class to use.
        backend_options: Backend-specific options.
        pipeline_options: Pipeline-specific options.
    """

    backend: Type[AbstractCADBackend]
    pipeline_cls: Type[BaseCADPipeline]
    backend_options: Optional[BackendOptions] = None
    pipeline_options: Optional[PipelineOptions] = None

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def set_optional_field_default(self) -> Self:
        """Set default pipeline options if not provided."""
        if self.pipeline_options is None:
            self.pipeline_options = self.pipeline_cls.get_default_options()
        return self


# Rebuild the models now that forward references are resolved
PipelineOptions.model_rebuild()
FormatOption.model_rebuild()


class DocumentConverter:
    """Main converter for CAD files.

    This is the primary entry point for cadling, providing a unified interface
    for converting various CAD formats to CADlingDocument.

    Similar to docling's DocumentConverter but adapted for CAD files.

    Attributes:
        allowed_formats: List of formats this converter will process.
        format_options: Format-specific configuration.

    Example:
        # Basic usage
        converter = DocumentConverter()
        result = converter.convert("part.step")

        # With format options
        from cadling.backend.step.step_backend import STEPBackend
        from cadling.pipeline.simple_pipeline import SimpleCADPipeline

        converter = DocumentConverter(
            allowed_formats=[InputFormat.STEP],
            format_options={
                InputFormat.STEP: FormatOption(
                    backend=STEPBackend,
                    pipeline_cls=SimpleCADPipeline,
                    backend_options=STEPBackendOptions(
                        enable_ll_stepnet=True
                    )
                )
            }
        )
    """

    def __init__(
        self,
        allowed_formats: Optional[List[InputFormat]] = None,
        format_options: Optional[Dict[InputFormat, FormatOption]] = None,
    ):
        """Initialize document converter.

        Args:
            allowed_formats: Formats to process (None = all formats).
            format_options: Format-specific configuration.
        """
        self.allowed_formats = allowed_formats or list(InputFormat)
        self.format_options = format_options or {}

        # Fill in default format options for allowed formats
        for fmt in self.allowed_formats:
            if fmt not in self.format_options:
                self.format_options[fmt] = self._get_default_format_option(fmt)

        _log.info(
            f"Initialized DocumentConverter with formats: "
            f"{[f.value for f in self.allowed_formats]}"
        )

    def convert(
        self,
        source: Union[Path, str, BytesIO],
        format: Optional[InputFormat] = None,
        pipeline_options: Optional[PipelineOptions] = None,
    ) -> ConversionResult:
        """Convert a CAD file to CADlingDocument.

        Args:
            source: Path to file or byte stream.
            format: Format (auto-detected if not provided).
            pipeline_options: Pipeline options to override format-level defaults.

        Returns:
            ConversionResult with status and document (or errors).

        Example:
            # From file path
            result = converter.convert("part.step")

            # From Path object
            result = converter.convert(Path("part.stl"))

            # With explicit format
            result = converter.convert("data.bin", format=InputFormat.STEP)
        """
        # Normalize source to Path
        if isinstance(source, str):
            source = Path(source)

        # Detect format if not provided
        if format is None:
            format = self._detect_format(source)
            _log.info(f"Detected format: {format.value}")

        # Validate format is allowed
        if format not in self.allowed_formats:
            from cadling.datamodel.base_models import ConversionStatus

            result = ConversionResult(
                input=CADInputDocument(
                    file=source if isinstance(source, Path) else Path("stream"),
                    format=format,
                    document_hash="",
                ),
                status=ConversionStatus.FAILURE,
            )
            result.add_error(
                component="DocumentConverter",
                error_message=f"Format {format.value} not in allowed formats",
            )
            return result

        # Create input document
        input_doc = self._create_input_document(source, format)

        # Get format option
        format_option = self.format_options[format]

        # Initialize backend
        backend = format_option.backend(
            in_doc=input_doc,
            path_or_stream=source,
            options=format_option.backend_options,
        )

        # Validate backend
        if not backend.is_valid():
            from cadling.datamodel.base_models import ConversionStatus

            result = ConversionResult(
                input=input_doc,
                status=ConversionStatus.FAILURE,
            )
            result.add_error(
                component=backend.__class__.__name__,
                error_message=f"File is not a valid {format.value} file",
            )
            return result

        # Attach backend to input doc
        input_doc._backend = backend

        # Initialize pipeline (use caller-provided options if given)
        effective_options = pipeline_options or format_option.pipeline_options
        pipeline = format_option.pipeline_cls(effective_options)

        # Execute conversion
        _log.info(
            f"Converting {input_doc.file.name} using "
            f"{backend.__class__.__name__} and {pipeline.__class__.__name__}"
        )

        result = pipeline.execute(input_doc)

        _log.info(
            f"Conversion completed with status: {result.status.value}"
        )

        return result

    def _detect_format(self, source: Union[Path, BytesIO]) -> InputFormat:
        """Detect CAD file format.

        Args:
            source: File path or byte stream.

        Returns:
            Detected InputFormat.

        Raises:
            ValueError: If format cannot be detected.
        """
        if isinstance(source, Path):
            # Try by extension first
            extension = source.suffix.lower()
            extension_map = {
                ".step": InputFormat.STEP,
                ".stp": InputFormat.STEP,
                ".stl": InputFormat.STL,
                ".brep": InputFormat.BREP,
                ".iges": InputFormat.IGES,
                ".igs": InputFormat.IGES,
                ".dxf": InputFormat.DXF,
                ".pdf": InputFormat.PDF_DRAWING,
            }

            if extension in extension_map:
                return extension_map[extension]

            # Try by content
            with open(source, "rb") as f:
                header = f.read(1024)
                return self._detect_format_by_content(header)

        else:  # BytesIO
            header = source.read(1024)
            source.seek(0)
            return self._detect_format_by_content(header)

    def _detect_format_by_content(self, header: bytes) -> InputFormat:
        """Detect format by file content.

        Args:
            header: First bytes of file.

        Returns:
            Detected InputFormat.

        Raises:
            ValueError: If format cannot be detected.
        """
        header_str = header.decode("utf-8", errors="ignore")

        # STEP files start with "ISO-10303-21"
        if header_str.startswith("ISO-10303-21"):
            return InputFormat.STEP

        # ASCII STL files start with "solid"
        if header_str.strip().startswith("solid"):
            return InputFormat.STL

        # Binary STL has specific header format (80-byte header + 4-byte triangle count)
        # But first rule out other formats that could match the size heuristic
        _known_signatures = (
            b"ISO-10303-21",  # STEP
            b"%PDF",          # PDF
        )
        _known_text_prefixes = ("IGES", "0\nSECTION", "0\r\nSECTION")
        is_known_format = any(header.startswith(sig) for sig in _known_signatures)
        if not is_known_format:
            is_known_format = any(header_str.lstrip().startswith(p) for p in _known_text_prefixes)
        if not is_known_format and len(header) >= 84:
            import struct
            num_triangles = struct.unpack_from('<I', header, 80)[0]
            # Validate: expected file size = 84 + (num_triangles * 50)
            # Only classify as STL if triangle count is reasonable (>0 and <10M)
            if 0 < num_triangles < 10_000_000:
                return InputFormat.STL

        # IGES files start with specific format
        if "IGES" in header_str:
            return InputFormat.IGES

        # DXF files start with section markers
        if header_str.lstrip().startswith("0\nSECTION") or header_str.lstrip().startswith("0\r\nSECTION"):
            return InputFormat.DXF

        # PDF files start with %PDF
        if header_str.startswith("%PDF"):
            return InputFormat.PDF_DRAWING

        raise ValueError(f"Cannot detect format from file content")

    def _create_input_document(
        self,
        source: Union[Path, BytesIO],
        format: InputFormat,
    ) -> CADInputDocument:
        """Create input document descriptor.

        Args:
            source: File path or byte stream.
            format: Detected format.

        Returns:
            CADInputDocument.
        """
        # Compute hash (cache content to avoid redundant re-reads)
        if isinstance(source, Path):
            with open(source, "rb") as f:
                content = f.read()
        else:
            content = source.read()
            source.seek(0)

        doc_hash = hashlib.sha256(content).hexdigest()

        # Create input document
        file_path = source if isinstance(source, Path) else Path("stream")

        input_doc = CADInputDocument(
            file=file_path,
            format=format,
            document_hash=doc_hash,
        )
        input_doc._content_cache = content
        return input_doc

    def _get_default_format_option(self, format: InputFormat) -> FormatOption:
        """Get default format option for a format.

        Args:
            format: CAD format.

        Returns:
            Default FormatOption for this format.

        Raises:
            ValueError: If format is not supported.
        """
        # Import backends and pipelines lazily to avoid circular imports
        from cadling.pipeline.simple_pipeline import SimpleCADPipeline

        if format == InputFormat.STEP:
            from cadling.backend.step.step_backend import STEPBackend

            return FormatOption(
                backend=STEPBackend,
                pipeline_cls=SimpleCADPipeline,
                backend_options=None,
            )

        elif format == InputFormat.STL:
            from cadling.backend.stl.stl_backend import STLBackend

            return FormatOption(
                backend=STLBackend,
                pipeline_cls=SimpleCADPipeline,
                backend_options=None,
            )

        elif format == InputFormat.BREP:
            from cadling.backend.brep.brep_backend import BRepBackend

            return FormatOption(
                backend=BRepBackend,
                pipeline_cls=SimpleCADPipeline,
                backend_options=None,
            )

        elif format == InputFormat.IGES:
            from cadling.backend.iges_backend import IGESBackend

            return FormatOption(
                backend=IGESBackend,
                pipeline_cls=SimpleCADPipeline,
                backend_options=None,
            )

        elif format == InputFormat.CAD_IMAGE:
            from cadling.backend.cadling_parse_backend import CADlingParseBackend

            return FormatOption(
                backend=CADlingParseBackend,
                pipeline_cls=SimpleCADPipeline,
                backend_options=None,
            )

        elif format == InputFormat.DXF:
            from cadling.backend.dxf_backend import DXFBackend
            from cadling.datamodel.backend_options import DXFBackendOptions

            return FormatOption(
                backend=DXFBackend,
                pipeline_cls=SimpleCADPipeline,
                backend_options=DXFBackendOptions(),
            )

        elif format in (InputFormat.PDF_DRAWING, InputFormat.PDF_RASTER):
            from cadling.backend.pdf_backend import PDFBackend
            from cadling.datamodel.backend_options import PDFBackendOptions

            return FormatOption(
                backend=PDFBackend,
                pipeline_cls=SimpleCADPipeline,
                backend_options=PDFBackendOptions(),
            )

        else:
            raise ValueError(f"Unsupported format: {format}")
