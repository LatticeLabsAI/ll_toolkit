"""Unit tests for DocumentConverter.

Tests the primary public entry point: format detection (by extension and
content), input document creation, default format option resolution,
allowed-format filtering, and error paths.  All backends and pipelines
are mocked so these tests run without CAD files or heavy dependencies.
"""

from __future__ import annotations

import hashlib
import struct
from io import BytesIO
from pathlib import Path
from typing import Optional

import pytest

from cadling.backend.document_converter import DocumentConverter, FormatOption
from cadling.datamodel.base_models import (
    CADInputDocument,
    ConversionResult,
    ConversionStatus,
    InputFormat,
)
from cadling.datamodel.backend_options import BackendOptions
from cadling.datamodel.pipeline_options import PipelineOptions
from cadling.backend.abstract_backend import AbstractCADBackend
from cadling.pipeline.base_pipeline import BaseCADPipeline


# ---------------------------------------------------------------------------
# Stub subclasses (Pydantic validates Type[X] with is_subclass_of)
# ---------------------------------------------------------------------------

_STUB_VALID = True
_STUB_RESULT_STATUS = ConversionStatus.SUCCESS


class _StubBackend(AbstractCADBackend):
    """Minimal concrete backend for unit tests."""

    _last_instance = None

    def __init__(self, in_doc=None, path_or_stream=None, options=None):
        self._in_doc = in_doc
        self._path_or_stream = path_or_stream
        self._options = options
        _StubBackend._last_instance = self

    def is_valid(self) -> bool:
        return _STUB_VALID

    def convert(self):
        pass

    def supported_formats(self):
        return []

    def supports_rendering(self):
        return False

    def supports_text_parsing(self):
        return True


class _StubPipeline(BaseCADPipeline):
    """Minimal concrete pipeline for unit tests."""

    _last_instance = None

    def __init__(self, options=None):
        self._options = options
        _StubPipeline._last_instance = self

    def execute(self, input_doc):
        return ConversionResult(input=input_doc, status=_STUB_RESULT_STATUS)

    def _build_document(self, input_doc):
        return None

    @classmethod
    def get_default_options(cls):
        return PipelineOptions()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_option(*, valid: bool = True, status: ConversionStatus = ConversionStatus.SUCCESS):
    """Build a FormatOption backed by stubs, setting module-level controls."""
    global _STUB_VALID, _STUB_RESULT_STATUS
    _STUB_VALID = valid
    _STUB_RESULT_STATUS = status
    _StubBackend._last_instance = None
    _StubPipeline._last_instance = None
    opt = FormatOption(backend=_StubBackend, pipeline_cls=_StubPipeline)
    return opt


# ---------------------------------------------------------------------------
# FormatOption model tests
# ---------------------------------------------------------------------------


class TestFormatOption:
    """Tests for the FormatOption Pydantic model."""

    def test_default_pipeline_options_set(self):
        """Pipeline options default via ``get_default_options`` when omitted."""
        opt = FormatOption(backend=_StubBackend, pipeline_cls=_StubPipeline)
        assert opt.pipeline_options is not None

    def test_explicit_pipeline_options_preserved(self):
        """Explicit pipeline_options are not overridden by defaults."""
        explicit = PipelineOptions()
        opt = FormatOption(
            backend=_StubBackend,
            pipeline_cls=_StubPipeline,
            pipeline_options=explicit,
        )
        assert opt.pipeline_options is explicit


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestDocumentConverterInit:
    """Tests for DocumentConverter initialisation."""

    def test_default_allows_all_formats(self):
        converter = DocumentConverter()
        assert set(converter.allowed_formats) == set(InputFormat)

    def test_restricted_formats(self):
        converter = DocumentConverter(allowed_formats=[InputFormat.STEP, InputFormat.STL])
        assert set(converter.allowed_formats) == {InputFormat.STEP, InputFormat.STL}

    def test_custom_format_option_used(self):
        opt = _format_option()
        converter = DocumentConverter(
            allowed_formats=[InputFormat.STEP],
            format_options={InputFormat.STEP: opt},
        )
        assert converter.format_options[InputFormat.STEP] is opt

    def test_missing_format_options_filled_with_defaults(self):
        converter = DocumentConverter(allowed_formats=[InputFormat.STEP])
        assert InputFormat.STEP in converter.format_options


# ---------------------------------------------------------------------------
# _detect_format – by extension
# ---------------------------------------------------------------------------


class TestDetectFormatByExtension:
    """Extension-based format detection."""

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("part.step", InputFormat.STEP),
            ("part.stp", InputFormat.STEP),
            ("part.STEP", InputFormat.STEP),
            ("mesh.stl", InputFormat.STL),
            ("model.brep", InputFormat.BREP),
            ("drawing.iges", InputFormat.IGES),
            ("drawing.igs", InputFormat.IGES),
            ("plan.dxf", InputFormat.DXF),
            ("doc.pdf", InputFormat.PDF_DRAWING),
        ],
    )
    def test_extension_detection(self, filename, expected, tmp_path):
        converter = DocumentConverter()
        p = tmp_path / filename
        p.write_bytes(b"\x00" * 100)
        assert converter._detect_format(p) == expected


# ---------------------------------------------------------------------------
# _detect_format – by content
# ---------------------------------------------------------------------------


class TestDetectFormatByContent:
    """Content-based format detection (header sniffing)."""

    def test_step_header(self):
        converter = DocumentConverter()
        buf = BytesIO(b"ISO-10303-21;\nHEADER;")
        assert converter._detect_format(buf) == InputFormat.STEP

    def test_ascii_stl_header(self):
        converter = DocumentConverter()
        buf = BytesIO(b"solid cube\nfacet normal 0 0 1\n")
        assert converter._detect_format(buf) == InputFormat.STL

    def test_binary_stl_header(self):
        converter = DocumentConverter()
        header = b"\x00" * 80 + struct.pack("<I", 12)  # 12 triangles
        buf = BytesIO(header)
        assert converter._detect_format(buf) == InputFormat.STL

    def test_iges_header(self):
        converter = DocumentConverter()
        buf = BytesIO(b"                                IGES file produced by X\n")
        assert converter._detect_format(buf) == InputFormat.IGES

    def test_dxf_header(self):
        converter = DocumentConverter()
        buf = BytesIO(b"0\nSECTION\n2\nHEADER\n")
        assert converter._detect_format(buf) == InputFormat.DXF

    def test_pdf_header(self):
        converter = DocumentConverter()
        buf = BytesIO(b"%PDF-1.4\n")
        assert converter._detect_format(buf) == InputFormat.PDF_DRAWING

    def test_unknown_content_raises(self):
        converter = DocumentConverter()
        buf = BytesIO(b"UNKNOWN FORMAT DATA " * 5)
        with pytest.raises(ValueError, match="Cannot detect format"):
            converter._detect_format(buf)

    def test_bytesio_position_reset_after_detection(self):
        """BytesIO position should be reset to 0 after detection."""
        converter = DocumentConverter()
        buf = BytesIO(b"ISO-10303-21;\nHEADER;")
        converter._detect_format(buf)
        assert buf.tell() == 0


# ---------------------------------------------------------------------------
# _create_input_document
# ---------------------------------------------------------------------------


class TestCreateInputDocument:
    """Tests for input document creation."""

    def test_from_path(self, tmp_path):
        converter = DocumentConverter()
        p = tmp_path / "cube.step"
        content = b"ISO-10303-21;\nHEADER;"
        p.write_bytes(content)

        doc = converter._create_input_document(p, InputFormat.STEP)

        assert doc.file == p
        assert doc.format == InputFormat.STEP
        assert doc.document_hash == hashlib.sha256(content).hexdigest()
        assert doc._content_cache == content

    def test_from_bytesio(self):
        converter = DocumentConverter()
        content = b"solid cube\nfacet normal 0 0 1\n"
        buf = BytesIO(content)

        doc = converter._create_input_document(buf, InputFormat.STL)

        assert doc.file == Path("stream")
        assert doc.format == InputFormat.STL
        assert doc.document_hash == hashlib.sha256(content).hexdigest()
        # BytesIO should be rewound
        assert buf.tell() == 0


# ---------------------------------------------------------------------------
# _get_default_format_option
# ---------------------------------------------------------------------------


class TestGetDefaultFormatOption:
    """Tests for default format option resolution."""

    @pytest.mark.parametrize(
        "fmt",
        [
            InputFormat.STEP,
            InputFormat.STL,
            InputFormat.BREP,
            InputFormat.IGES,
            InputFormat.CAD_IMAGE,
            InputFormat.DXF,
            InputFormat.PDF_DRAWING,
            InputFormat.PDF_RASTER,
        ],
    )
    def test_supported_format_returns_option(self, fmt):
        converter = DocumentConverter.__new__(DocumentConverter)
        converter.allowed_formats = []
        converter.format_options = {}
        opt = converter._get_default_format_option(fmt)
        assert isinstance(opt, FormatOption)
        assert opt.backend is not None
        assert opt.pipeline_cls is not None


# ---------------------------------------------------------------------------
# convert – happy path
# ---------------------------------------------------------------------------


class TestConvertHappyPath:
    """Tests for successful conversion flow."""

    def test_convert_with_path(self, tmp_path):
        opt = _format_option()
        converter = DocumentConverter(
            allowed_formats=[InputFormat.STEP],
            format_options={InputFormat.STEP: opt},
        )

        p = tmp_path / "part.step"
        p.write_bytes(b"ISO-10303-21;\nDATA;")

        result = converter.convert(p)

        assert result.status == ConversionStatus.SUCCESS
        assert _StubBackend._last_instance is not None
        assert _StubPipeline._last_instance is not None

    def test_convert_with_string_path(self, tmp_path):
        opt = _format_option()
        converter = DocumentConverter(
            allowed_formats=[InputFormat.STEP],
            format_options={InputFormat.STEP: opt},
        )

        p = tmp_path / "part.step"
        p.write_bytes(b"ISO-10303-21;\nDATA;")

        result = converter.convert(str(p))
        assert result.status == ConversionStatus.SUCCESS

    def test_convert_with_explicit_format(self, tmp_path):
        opt = _format_option()
        converter = DocumentConverter(
            allowed_formats=[InputFormat.STEP],
            format_options={InputFormat.STEP: opt},
        )

        p = tmp_path / "data.bin"
        p.write_bytes(b"some binary data")

        result = converter.convert(p, format=InputFormat.STEP)
        assert result.status == ConversionStatus.SUCCESS

    def test_pipeline_options_override(self, tmp_path):
        opt = _format_option()
        converter = DocumentConverter(
            allowed_formats=[InputFormat.STEP],
            format_options={InputFormat.STEP: opt},
        )

        p = tmp_path / "part.step"
        p.write_bytes(b"ISO-10303-21;\nDATA;")

        override = PipelineOptions()
        result = converter.convert(p, pipeline_options=override)

        assert result.status == ConversionStatus.SUCCESS
        assert _StubPipeline._last_instance._options is override


# ---------------------------------------------------------------------------
# convert – error paths
# ---------------------------------------------------------------------------


class TestConvertErrorPaths:
    """Tests for conversion error handling."""

    def test_disallowed_format_returns_failure(self, tmp_path):
        converter = DocumentConverter(allowed_formats=[InputFormat.STEP])

        p = tmp_path / "mesh.stl"
        p.write_bytes(b"solid cube\n")

        result = converter.convert(p)
        assert result.status == ConversionStatus.FAILURE
        assert any("not in allowed formats" in e.error_message for e in result.errors)

    def test_invalid_backend_returns_failure(self, tmp_path):
        opt = _format_option(valid=False)
        converter = DocumentConverter(
            allowed_formats=[InputFormat.STEP],
            format_options={InputFormat.STEP: opt},
        )

        p = tmp_path / "part.step"
        p.write_bytes(b"ISO-10303-21;\nDATA;")

        result = converter.convert(p)
        assert result.status == ConversionStatus.FAILURE
        assert any("not a valid" in e.error_message for e in result.errors)

    def test_undetectable_format_raises(self, tmp_path):
        converter = DocumentConverter()

        p = tmp_path / "data.xyz"
        p.write_bytes(b"COMPLETELY UNKNOWN FORMAT " * 5)

        with pytest.raises(ValueError, match="Cannot detect format"):
            converter.convert(p)

    def test_bytesio_disallowed_format(self):
        converter = DocumentConverter(allowed_formats=[InputFormat.STEP])
        buf = BytesIO(b"solid cube\nfacet normal 0 0 1\n")

        result = converter.convert(buf)
        assert result.status == ConversionStatus.FAILURE

    def test_nonexistent_path_raises(self):
        converter = DocumentConverter()
        with pytest.raises(ValueError, match="does not point to an existing file"):
            converter.convert("/nonexistent/path/file.step")

    def test_path_traversal_blocked(self, tmp_path):
        """Traversal paths outside allowed_root are rejected."""
        allowed = tmp_path / "safe"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "secret.step"
        target.write_bytes(b"ISO-10303-21;\nDATA;")

        converter = DocumentConverter()
        with pytest.raises(ValueError, match="Path traversal blocked"):
            converter.convert(str(target), allowed_root=str(allowed))

    def test_path_within_allowed_root_succeeds(self, tmp_path):
        """Paths inside allowed_root pass validation."""
        opt = _format_option()
        converter = DocumentConverter(
            allowed_formats=[InputFormat.STEP],
            format_options={InputFormat.STEP: opt},
        )

        p = tmp_path / "part.step"
        p.write_bytes(b"ISO-10303-21;\nDATA;")

        result = converter.convert(str(p), allowed_root=str(tmp_path))
        assert result.status == ConversionStatus.SUCCESS

    def test_dotdot_traversal_resolved(self, tmp_path):
        """Paths with '../' components are resolved before root check."""
        allowed = tmp_path / "safe"
        allowed.mkdir()
        target = tmp_path / "secret.step"
        target.write_bytes(b"ISO-10303-21;\nDATA;")

        converter = DocumentConverter()
        traversal = str(allowed / ".." / "secret.step")
        with pytest.raises(ValueError, match="Path traversal blocked"):
            converter.convert(traversal, allowed_root=str(allowed))
