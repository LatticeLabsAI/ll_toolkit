"""CADling Parse backend for generic CAD parsing.

This backend provides a generic parsing interface that can handle multiple
CAD formats through a unified parsing API. It's designed as a fallback when
format-specific backends are not available.

Classes:
    CADlingParseBackend: Generic CAD parsing backend
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Set

from cadling.backend.abstract_backend import DeclarativeCADBackend
from cadling.datamodel.base_models import CADlingDocument, InputFormat

_log = logging.getLogger(__name__)


class CADlingParseBackend(DeclarativeCADBackend):
    """Generic CAD parsing backend.

    This backend attempts to parse CAD files using a generic parsing strategy.
    It's primarily used as a fallback when format-specific backends are not
    available or when simple parsing is sufficient.

    Supports:
    - Basic file validation
    - Generic entity extraction
    - Minimal processing

    Note: For production use, prefer format-specific backends (STEPBackend,
    STLBackend, etc.) which provide better feature extraction and validation.
    """

    @classmethod
    def supports_text_parsing(cls) -> bool:
        """Whether backend can parse text representation.

        Returns:
            True - This is a text-based parsing backend
        """
        return True

    @classmethod
    def supports_rendering(cls) -> bool:
        """Whether backend can render to images.

        Returns:
            False - This backend does not support rendering
        """
        return False

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        """Get supported formats.

        Returns:
            Set of all InputFormat values (fallback for any format)
        """
        return set(InputFormat)

    def is_valid(self) -> bool:
        """Check if file is valid for this backend.

        Returns:
            True if file exists and is readable
        """
        try:
            if isinstance(self.path_or_stream, Path):
                return self.path_or_stream.exists() and self.path_or_stream.is_file()
            else:
                # BytesIO - assume valid
                return True
        except Exception as e:
            _log.error(f"Validation error: {e}")
            return False

    def convert(self) -> CADlingDocument:
        """Convert CAD file to CADlingDocument.

        This provides minimal conversion - just creates a document structure
        with basic metadata. For full feature extraction, use format-specific
        backends.

        Returns:
            CADlingDocument with minimal processing

        Raises:
            ValueError: If file cannot be read
        """
        _log.warning(
            f"Using generic CADlingParseBackend for {self.input_format} - "
            f"consider using format-specific backend for better results"
        )

        from cadling.datamodel.base_models import CADDocumentOrigin, CADItem, CADItemLabel

        # Create document
        doc = CADlingDocument(
            name=self.file.name,
            format=self.input_format,
            origin=CADDocumentOrigin(
                filename=self.file.name,
                format=self.input_format,
                binary_hash=self.document_hash,
            ),
            hash=self.document_hash,
        )

        # Read file content
        if isinstance(self.path_or_stream, Path):
            with open(self.path_or_stream, "rb") as f:
                content = f.read()
        else:
            content = self.path_or_stream.read()
            self.path_or_stream.seek(0)

        # Add a single item representing the raw file
        raw_item = CADItem(
            item_type="raw_file",
            label=CADItemLabel(text=f"Raw {self.input_format} file"),
            text=f"File size: {len(content)} bytes",
            properties={
                "file_size_bytes": len(content),
                "format": self.input_format.value,
                "parsed_with": "CADlingParseBackend",
            },
        )
        doc.add_item(raw_item)

        _log.info(
            f"Minimal conversion complete: {self.file.name} "
            f"({len(content)} bytes)"
        )

        return doc

    def _get_default_options(self):
        """Get default backend options.

        Returns:
            Default BackendOptions
        """
        from cadling.datamodel.backend_options import BackendOptions

        return BackendOptions()
