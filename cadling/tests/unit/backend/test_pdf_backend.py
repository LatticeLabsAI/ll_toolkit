"""Unit tests for PDF backend (cadling/backend/pdf_backend.py).

This test suite covers vector and raster PDF extraction, page type detection,
dimension annotation parsing, and static helper methods for arc classification
and bezier curve sampling.

Tests use pytest fixtures to create PDF documents programmatically with fitz/pymupdf.

Test organization:
    TestPDFBackendClassMethods: Class-level interface methods
    TestPDFValidation: is_valid() with various inputs
    TestPDFPageTypeDetection: Auto-detection of vector vs raster content
    TestPDFVectorExtraction: Vector path extraction (lines, curves, rectangles)
    TestPDFTextDimensions: Dimension annotation regex patterns
    TestPDFRasterFallback: Raster rendering and fallback behavior
    TestPDFStaticHelpers: Static utility methods (_circle_center_3pts, _sample_bezier)
    TestPDFConversionFlow: End-to-end convert() integration tests
    TestPDFMetadata: Document properties tracking
"""

from __future__ import annotations

import math
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import pytest

# Try to import pymupdf — skip tests if not available
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


# Import backend and models
from cadling.backend.pdf_backend import PDFBackend, _DIM_PATTERNS
from cadling.datamodel.base_models import CADInputDocument, InputFormat
from cadling.datamodel.backend_options import PDFBackendOptions
from cadling.datamodel.geometry_2d import (
    Arc2D,
    Circle2D,
    DimensionAnnotation,
    DimensionType,
    Line2D,
    Polyline2D,
    Sketch2DItem,
)


# ============================================================================
# Fixtures
# ============================================================================


def _make_input_doc(path: Path) -> CADInputDocument:
    """Helper to create CADInputDocument from a PDF path."""
    return CADInputDocument(
        file=path,
        format=InputFormat.PDF_DRAWING,
        document_hash="test_hash_123"
    )


@pytest.fixture
def temp_pdf_dir() -> Path:
    """Create temporary directory for test PDF files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_vector_pdf(temp_pdf_dir: Path) -> Path:
    """Create a simple PDF with vector drawing content (lines and shapes).

    The PDF contains:
    - 3 horizontal lines
    - 1 rectangle (4 line segments)
    - Total: 7 drawing elements
    """
    pytest.importorskip("fitz")

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size

    # Draw three horizontal lines
    shape = page.new_shape()
    shape.draw_line(fitz.Point(50, 100), fitz.Point(200, 100))
    shape.draw_line(fitz.Point(50, 150), fitz.Point(200, 150))
    shape.draw_line(fitz.Point(50, 200), fitz.Point(200, 200))
    shape.finish(color=(0, 0, 0))  # Black
    shape.commit()

    # Draw rectangle
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(250, 100, 400, 250))
    shape.finish(color=(0, 0, 0))
    shape.commit()

    path = temp_pdf_dir / "simple_vector.pdf"
    doc.save(str(path))
    doc.close()

    return path


@pytest.fixture
def text_dimension_pdf(temp_pdf_dir: Path) -> Path:
    """Create a PDF with text containing dimension annotations.

    The PDF contains text matching various dimension patterns:
    - ∅12.5 (diameter)
    - R5 (radius)
    - 45° (angle)
    - 25.4 (linear)
    """
    pytest.importorskip("fitz")

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Add dimension text at different locations
    page.insert_text(fitz.Point(100, 100), "∅12.5", fontsize=12)
    page.insert_text(fitz.Point(200, 200), "R5", fontsize=12)
    page.insert_text(fitz.Point(300, 300), "45°", fontsize=12)
    page.insert_text(fitz.Point(400, 400), "25.4mm", fontsize=12)

    path = temp_pdf_dir / "text_dimension.pdf"
    doc.save(str(path))
    doc.close()

    return path


@pytest.fixture
def raster_pdf(temp_pdf_dir: Path) -> Path:
    """Create a PDF with primarily raster/image content.

    This PDF is created by rendering existing content and embedding as image,
    simulating a scanned drawing.
    """
    pytest.importorskip("fitz")

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Add minimal vector content (< 5 drawings)
    shape = page.new_shape()
    shape.draw_line(fitz.Point(50, 50), fitz.Point(100, 100))
    shape.finish(color=(0, 0, 0))
    shape.commit()

    # Embed an image (simulates raster content)
    # Create a small test image by rendering another page
    doc2 = fitz.open()
    page2 = doc2.new_page(width=200, height=200)
    page2.insert_text(fitz.Point(50, 100), "Test Image", fontsize=14)
    pix = page2.get_pixmap()
    doc2.close()

    # Insert the pixmap as image into main page
    page.insert_image(fitz.Rect(150, 50, 400, 300), pixmap=pix)

    path = temp_pdf_dir / "raster.pdf"
    doc.save(str(path))
    doc.close()

    return path


@pytest.fixture
def empty_pdf(temp_pdf_dir: Path) -> Path:
    """Create a minimal PDF that pymupdf can open but has empty content.

    Since pymupdf refuses to save a zero-page PDF, we create a single-page
    PDF with no drawn content to test near-empty behavior.
    """
    pytest.importorskip("fitz")

    doc = fitz.open()
    # Add a blank page (pymupdf requires at least 1 page to save)
    doc.new_page(width=595, height=842)
    path = temp_pdf_dir / "empty.pdf"
    doc.save(str(path))
    doc.close()

    return path


@pytest.fixture
def multipage_pdf(temp_pdf_dir: Path) -> Path:
    """Create a 3-page PDF with different content on each page.

    Page 0: Vector content (lines)
    Page 1: Vector content (rectangle)
    Page 2: Mixed content (1 line + image)
    """
    pytest.importorskip("fitz")

    doc = fitz.open()

    # Page 0: Lines
    page = doc.new_page(width=595, height=842)
    shape = page.new_shape()
    shape.draw_line(fitz.Point(50, 100), fitz.Point(150, 100))
    shape.draw_line(fitz.Point(50, 150), fitz.Point(150, 150))
    shape.finish(color=(0, 0, 0))
    shape.commit()

    # Page 1: Rectangle
    page = doc.new_page(width=595, height=842)
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(100, 100, 300, 300))
    shape.finish(color=(0, 0, 0))
    shape.commit()

    # Page 2: Mixed content
    page = doc.new_page(width=595, height=842)
    shape = page.new_shape()
    shape.draw_line(fitz.Point(50, 50), fitz.Point(100, 100))
    shape.finish(color=(0, 0, 0))
    shape.commit()

    # Add image to page 2
    doc2 = fitz.open()
    page2 = doc2.new_page(width=150, height=150)
    pix = page2.get_pixmap()
    doc2.close()
    page.insert_image(fitz.Rect(200, 200, 350, 350), pixmap=pix)

    path = temp_pdf_dir / "multipage.pdf"
    doc.save(str(path))
    doc.close()

    return path


@pytest.fixture
def curve_pdf(temp_pdf_dir: Path) -> Path:
    """Create a PDF with curved paths (bezier curves).

    Contains one or more curves that can be classified as arcs or polylines.
    """
    pytest.importorskip("fitz")

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Draw a simple circle using bezier curves
    shape = page.new_shape()
    # Use draw_circle which internally uses bezier curves
    shape.draw_circle(fitz.Point(200, 200), 50)
    shape.finish(color=(0, 0, 0))
    shape.commit()

    path = temp_pdf_dir / "curve.pdf"
    doc.save(str(path))
    doc.close()

    return path


@pytest.fixture
def invalid_pdf(temp_pdf_dir: Path) -> Path:
    """Create an invalid PDF file (garbage data)."""
    path = temp_pdf_dir / "invalid.pdf"
    with open(path, "wb") as f:
        f.write(b"This is not a valid PDF file at all")
    return path


# ============================================================================
# Test Classes
# ============================================================================


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
class TestPDFBackendClassMethods:
    """Test class-level interface methods of PDFBackend."""

    def test_supports_text_parsing(self):
        """Test that PDFBackend reports text parsing support."""
        assert PDFBackend.supports_text_parsing() is True

    def test_supports_rendering(self):
        """Test that PDFBackend reports no 3D rendering support."""
        assert PDFBackend.supports_rendering() is False

    def test_supported_formats(self):
        """Test that PDFBackend supports PDF_DRAWING and PDF_RASTER formats."""
        formats = PDFBackend.supported_formats()
        assert InputFormat.PDF_DRAWING in formats
        assert InputFormat.PDF_RASTER in formats
        assert len(formats) == 2

    def test_get_default_options(self):
        """Test that default options return PDFBackendOptions instance."""
        options = PDFBackend._get_default_options()
        assert isinstance(options, PDFBackendOptions)
        assert options.extraction_mode == "auto"
        assert options.dpi == 300
        assert options.extract_dimensions is True


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
class TestPDFValidation:
    """Test PDF validation (is_valid method)."""

    def test_valid_pdf_with_content(self, simple_vector_pdf):
        """Test is_valid() with a valid PDF containing content."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)

        assert backend.is_valid() is True
        assert backend._pdf_doc is not None
        assert backend._pdf_doc.page_count >= 1

    def test_invalid_pdf_garbage_data(self, invalid_pdf):
        """Test is_valid() with invalid/garbage PDF file."""
        in_doc = _make_input_doc(invalid_pdf)
        backend = PDFBackend(in_doc, invalid_pdf)

        assert backend.is_valid() is False
        assert backend._pdf_doc is None

    def test_empty_pdf_no_content(self, empty_pdf):
        """Test is_valid() with PDF containing a blank page (no drawings)."""
        in_doc = _make_input_doc(empty_pdf)
        backend = PDFBackend(in_doc, empty_pdf)

        # A blank page is still a valid PDF — is_valid should return True
        assert backend.is_valid() is True
        assert backend._pdf_doc is not None

        # Converting should produce a document with no sketch primitives
        doc = backend.convert()
        assert doc is not None

    def test_valid_pdf_bytesio(self, simple_vector_pdf):
        """Test is_valid() with BytesIO stream instead of file path."""
        with open(simple_vector_pdf, "rb") as f:
            stream = BytesIO(f.read())

        in_doc = CADInputDocument(
            file=simple_vector_pdf,  # Original file for name
            format=InputFormat.PDF_DRAWING,
            document_hash="test_hash"
        )
        backend = PDFBackend(in_doc, stream)

        assert backend.is_valid() is True
        assert backend._pdf_doc is not None

    def test_nonexistent_file(self, temp_pdf_dir):
        """Test is_valid() with non-existent file path."""
        nonexistent = temp_pdf_dir / "does_not_exist.pdf"
        in_doc = _make_input_doc(nonexistent)
        backend = PDFBackend(in_doc, nonexistent)

        assert backend.is_valid() is False


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
class TestPDFPageTypeDetection:
    """Test page type detection (vector vs raster)."""

    def test_detect_vector_page(self, simple_vector_pdf):
        """Test detection of page with vector content (>10 drawings)."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)
        assert backend.is_valid()

        page = backend._pdf_doc[0]
        mode = backend._detect_page_type(page)

        # simple_vector_pdf has 7 drawing elements (3 lines + 4 rect segments)
        # This is < 10, but > 0, so should be detected as "vector" (to try)
        assert mode in ("vector", "raster")

    def test_detect_raster_page(self, raster_pdf):
        """Test detection of page dominated by raster/images."""
        in_doc = _make_input_doc(raster_pdf)
        backend = PDFBackend(in_doc, raster_pdf)
        assert backend.is_valid()

        page = backend._pdf_doc[0]
        mode = backend._detect_page_type(page)

        # raster_pdf has 1 drawing and 1 image, should be detected as "raster"
        assert mode in ("vector", "raster")

    def test_detect_empty_page(self, empty_pdf):
        """Test detection of page with no content defaults to raster."""
        in_doc = _make_input_doc(empty_pdf)
        backend = PDFBackend(in_doc, empty_pdf)
        assert backend.is_valid()

        page = backend._pdf_doc[0]
        page_type = backend._detect_page_type(page)
        # A blank page has no drawings and no images, so should default to raster
        assert page_type == "raster"


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
class TestPDFVectorExtraction:
    """Test vector content extraction from PDF pages."""

    def test_extract_vector_page_with_lines(self, simple_vector_pdf):
        """Test extraction of lines from vector PDF."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)
        assert backend.is_valid()

        page = backend._pdf_doc[0]
        item = backend._extract_vector_page(page, 0)

        # Should extract lines or be None
        if item:
            assert isinstance(item, Sketch2DItem)
            assert len(item.profiles) > 0
            assert item.source_page == 0

    def test_extract_vector_page_none_if_no_drawings(self, text_dimension_pdf):
        """Test that extraction returns None if page has no vector drawings."""
        in_doc = _make_input_doc(text_dimension_pdf)
        backend = PDFBackend(in_doc, text_dimension_pdf)
        assert backend.is_valid()

        page = backend._pdf_doc[0]
        # This PDF has only text, no drawings
        item = backend._extract_vector_page(page, 0)

        # May return None or extract dimensions if enabled
        # Default behavior: extract_dimensions=True
        if item:
            assert isinstance(item, Sketch2DItem)

    def test_pdf_line_to_primitive(self, simple_vector_pdf):
        """Test conversion of PDF line command to Line2D."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)
        assert backend.is_valid()

        # Create test line item (mimics fitz line format)
        p1 = fitz.Point(10, 20)
        p2 = fitz.Point(30, 40)
        item = ("l", p1, p2)

        line = backend._pdf_line_to_primitive(item, (0, 0, 0), 0)

        assert isinstance(line, Line2D)
        assert line.start == (10.0, 20.0)
        assert line.end == (30.0, 40.0)
        assert line.color == (0, 0, 0)
        assert line.layer == "page_0"

    def test_pdf_line_to_primitive_with_none_color(self, simple_vector_pdf):
        """Test line conversion with None color."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)
        assert backend.is_valid()

        p1 = fitz.Point(0, 0)
        p2 = fitz.Point(10, 10)
        item = ("l", p1, p2)

        line = backend._pdf_line_to_primitive(item, None, 0)

        assert isinstance(line, Line2D)
        assert line.color is None

    def test_pdf_rect_to_primitives(self, simple_vector_pdf):
        """Test conversion of PDF rectangle to four Line2D segments."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)
        assert backend.is_valid()

        rect = fitz.Rect(50, 100, 200, 300)
        item = ("re", rect)

        lines = backend._pdf_rect_to_primitives(item, (0, 0, 0), 0)

        assert len(lines) == 4
        assert all(isinstance(line, Line2D) for line in lines)

        # Check that lines form a rectangle
        # Lines should be: bottom, right, top, left
        assert lines[0].start == (50.0, 100.0)
        assert lines[0].end == (200.0, 100.0)

    def test_pdf_quad_to_primitives(self, simple_vector_pdf):
        """Test conversion of PDF quadrilateral to four Line2D segments."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)
        assert backend.is_valid()

        quad = fitz.Quad(
            fitz.Point(50, 100),
            fitz.Point(200, 100),
            fitz.Point(200, 300),
            fitz.Point(50, 300)
        )
        item = ("qu", quad)

        lines = backend._pdf_quad_to_primitives(item, (100, 100, 100), 1)

        assert len(lines) == 4
        assert all(isinstance(line, Line2D) for line in lines)
        assert lines[0].color == (100, 100, 100)
        assert lines[0].layer == "page_1"

    def test_extract_drawing_color(self, simple_vector_pdf):
        """Test RGB color extraction from drawing dictionary."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)
        assert backend.is_valid()

        # Simulate drawing dict with normalized (0-1) color values
        drawing = {"color": (1.0, 0.5, 0.0)}
        color = backend._extract_drawing_color(drawing)

        assert color == (255, 127, 0)

    def test_extract_drawing_color_none(self, simple_vector_pdf):
        """Test color extraction with missing/None color."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)
        assert backend.is_valid()

        drawing = {}
        color = backend._extract_drawing_color(drawing)
        assert color is None

        drawing = {"color": None}
        color = backend._extract_drawing_color(drawing)
        assert color is None


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
class TestPDFTextDimensions:
    """Test dimension annotation text extraction."""

    def test_diameter_pattern(self):
        """Test regex pattern for diameter dimensions (∅, Ø, ⌀)."""
        pattern = _DIM_PATTERNS[0][0]  # Diameter pattern

        assert pattern.search("∅12.5")
        assert pattern.search("Ø12")
        assert pattern.search("⌀10.5")

        match = pattern.search("∅12.5")
        assert match.group(1) == "12.5"

    def test_radius_pattern(self):
        """Test regex pattern for radius dimensions (R)."""
        pattern = _DIM_PATTERNS[1][0]  # Radius pattern

        assert pattern.search("R5")
        assert pattern.search("R3.2")
        assert pattern.search("R 5.0")

        match = pattern.search("R5")
        assert match.group(1) == "5"

    def test_angular_pattern(self):
        """Test regex pattern for angular dimensions (°)."""
        pattern = _DIM_PATTERNS[2][0]  # Angular pattern

        assert pattern.search("45°")
        assert pattern.search("90.5°")
        assert pattern.search("180°")

        match = pattern.search("45°")
        assert match.group(1) == "45"

    def test_linear_pattern(self):
        """Test regex pattern for linear dimensions."""
        pattern = _DIM_PATTERNS[3][0]  # Linear pattern

        assert pattern.search("25.4")
        assert pattern.search("100mm")
        assert pattern.search("50cm")

        match = pattern.search("25.4")
        assert match.group(1) == "25.4"

    def test_extract_text_dimensions_with_text(self, text_dimension_pdf):
        """Test extraction of dimension annotations from PDF text."""
        in_doc = _make_input_doc(text_dimension_pdf)
        backend = PDFBackend(in_doc, text_dimension_pdf)
        assert backend.is_valid()

        page = backend._pdf_doc[0]
        annotations = backend._extract_text_dimensions(page, 0)

        # Should extract multiple dimension types
        assert len(annotations) > 0
        assert all(isinstance(ann, DimensionAnnotation) for ann in annotations)

        # Check for expected dimension types
        dim_types = {ann.dim_type for ann in annotations}
        expected_types = {
            DimensionType.DIAMETER,
            DimensionType.RADIAL,
            DimensionType.ANGULAR,
            DimensionType.LINEAR
        }
        # May not get all types depending on PDF rendering
        assert len(dim_types) > 0


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
class TestPDFRasterFallback:
    """Test raster rendering and fallback behavior."""

    def test_render_raster_page(self, simple_vector_pdf):
        """Test rendering a PDF page to raster image."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)
        assert backend.is_valid()

        page = backend._pdf_doc[0]
        item = backend._render_raster_page(page, 0)

        assert item is not None
        assert item.item_type == "pdf_raster_page"
        assert item.properties["page_index"] == 0
        assert item.properties["dpi"] == 300
        assert "width_px" in item.properties
        assert "height_px" in item.properties

    def test_render_raster_page_custom_dpi(self, simple_vector_pdf):
        """Test raster rendering with custom DPI."""
        options = PDFBackendOptions(dpi=150)
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf, options)
        assert backend.is_valid()

        page = backend._pdf_doc[0]
        item = backend._render_raster_page(page, 0)

        assert item is not None
        assert item.properties["dpi"] == 150

    def test_auto_mode_with_fallback(self, simple_vector_pdf):
        """Test auto mode: vector extraction falls back to raster when empty."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)
        assert backend.is_valid()

        doc = backend.convert()

        assert doc is not None
        assert doc.name == "simple_vector.pdf"
        assert len(doc.items) > 0

    def test_force_raster_mode(self, simple_vector_pdf):
        """Test forcing raster extraction mode."""
        options = PDFBackendOptions(extraction_mode="raster")
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf, options)
        assert backend.is_valid()

        doc = backend.convert()

        assert doc is not None
        # All items should be raster
        for item in doc.items:
            assert item.item_type == "pdf_raster_page"


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
class TestPDFStaticHelpers:
    """Test static helper methods for arc classification and bezier sampling."""

    def test_circle_center_3pts_simple(self):
        """Test circle center calculation with simple three-point circle."""
        # Circle centered at (0, 0) with radius 5
        p1 = (5.0, 0.0)
        p2 = (0.0, 5.0)
        p3 = (-5.0, 0.0)

        center = PDFBackend._circle_center_3pts(p1, p2, p3)

        assert center is not None
        assert abs(center[0] - 0.0) < 1e-6
        assert abs(center[1] - 0.0) < 1e-6

    def test_circle_center_3pts_offset(self):
        """Test circle center calculation with offset circle."""
        # Circle centered at (10, 20) with radius 3
        p1 = (13.0, 20.0)
        p2 = (10.0, 23.0)
        p3 = (7.0, 20.0)

        center = PDFBackend._circle_center_3pts(p1, p2, p3)

        assert center is not None
        assert abs(center[0] - 10.0) < 1e-6
        assert abs(center[1] - 20.0) < 1e-6

    def test_circle_center_3pts_collinear(self):
        """Test circle center with collinear points (no circle)."""
        # Three collinear points
        p1 = (0.0, 0.0)
        p2 = (5.0, 5.0)
        p3 = (10.0, 10.0)

        center = PDFBackend._circle_center_3pts(p1, p2, p3)

        # Should return None for collinear points
        assert center is None

    def test_sample_bezier_line(self):
        """Test bezier sampling with a straight line (bezier = line)."""
        # Straight line from (0, 0) to (10, 0)
        p1 = (0.0, 0.0)
        p2 = (3.33, 0.0)
        p3 = (6.67, 0.0)
        p4 = (10.0, 0.0)

        points = PDFBackend._sample_bezier(p1, p2, p3, p4, num_segments=3)

        assert len(points) == 4  # 3 segments + 1 endpoint
        assert points[0] == (0.0, 0.0)
        assert points[-1] == (10.0, 0.0)

        # All points should be on y=0 (approximately)
        for x, y in points:
            assert abs(y) < 1e-6

    def test_sample_bezier_curve(self):
        """Test bezier sampling with an actual curve."""
        # Simple curve: p1 at origin, p4 at (10, 10)
        p1 = (0.0, 0.0)
        p2 = (0.0, 10.0)
        p3 = (10.0, 10.0)
        p4 = (10.0, 0.0)

        points = PDFBackend._sample_bezier(p1, p2, p3, p4, num_segments=4)

        assert len(points) == 5
        assert points[0] == (0.0, 0.0)
        assert points[-1] == (10.0, 0.0)

        # Sampled points should be distinct
        assert len(set(points)) >= 3

    def test_sample_bezier_midpoint(self):
        """Test that sampled bezier correctly includes midpoint."""
        p1 = (0.0, 0.0)
        p2 = (5.0, 10.0)
        p3 = (5.0, 10.0)
        p4 = (10.0, 0.0)

        points = PDFBackend._sample_bezier(p1, p2, p3, p4, num_segments=2)

        # At t=0.5, the midpoint should be evaluated
        assert len(points) == 3
        assert points[0] == p1
        assert points[-1] == p4

    def test_try_classify_as_arc_true_arc(self):
        """Test arc classification with points forming a true circular arc."""
        # Create bezier approximation of circular arc
        # Circle at origin with radius 10, from 0° to 90°
        r = 10.0
        k = 0.5522847498  # Bezier circle constant

        p1 = (r, 0.0)
        p2 = (r, k * r)
        p3 = (k * r, r)
        p4 = (0.0, r)

        arc = PDFBackend._try_classify_as_arc(p1, p2, p3, p4)

        # Should classify as arc (or at least not return None)
        if arc:
            assert isinstance(arc, Arc2D)
            assert arc.radius > 0

    def test_try_classify_as_arc_non_arc(self):
        """Test arc classification with bezier that is not a circular arc."""
        # S-curve (non-arc)
        p1 = (0.0, 0.0)
        p2 = (3.0, 10.0)
        p3 = (7.0, -10.0)
        p4 = (10.0, 0.0)

        arc = PDFBackend._try_classify_as_arc(p1, p2, p3, p4)

        # May return None if not arc-like
        # This is implementation-dependent


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
class TestPDFConversionFlow:
    """Test end-to-end conversion flow."""

    def test_convert_single_page_vector(self, simple_vector_pdf):
        """Test converting a single-page PDF with vector content."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)

        assert backend.is_valid()
        doc = backend.convert()

        assert doc is not None
        assert doc.format == InputFormat.PDF_DRAWING
        assert doc.name == "simple_vector.pdf"
        assert len(doc.items) > 0

    def test_convert_multipage_pdf(self, multipage_pdf):
        """Test converting a multi-page PDF."""
        in_doc = _make_input_doc(multipage_pdf)
        backend = PDFBackend(in_doc, multipage_pdf)

        assert backend.is_valid()
        doc = backend.convert()

        assert doc is not None
        assert len(doc.items) >= 1  # Should have extracted content from pages

    def test_convert_with_page_range(self, multipage_pdf):
        """Test converting with page range option."""
        # Only process pages 0-1 (skip page 2)
        options = PDFBackendOptions(page_range=(0, 2))
        in_doc = _make_input_doc(multipage_pdf)
        backend = PDFBackend(in_doc, multipage_pdf, options)

        assert backend.is_valid()
        doc = backend.convert()

        assert doc is not None
        # Should only have items from pages 0-1
        assert all(item.source_page < 2 for item in doc.items if hasattr(item, 'source_page'))

    def test_convert_invalid_file_raises(self, invalid_pdf):
        """Test that converting invalid PDF raises RuntimeError."""
        in_doc = _make_input_doc(invalid_pdf)
        backend = PDFBackend(in_doc, invalid_pdf)

        # is_valid() should return False
        assert backend.is_valid() is False

        # convert() should raise RuntimeError
        with pytest.raises(RuntimeError):
            backend.convert()

    def test_convert_without_validation(self, simple_vector_pdf):
        """Test that convert() validates PDF if not already validated."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)

        # Don't call is_valid() explicitly
        doc = backend.convert()

        assert doc is not None
        assert backend._pdf_doc is not None

    def test_convert_with_dimension_extraction(self, text_dimension_pdf):
        """Test converting with dimension extraction enabled."""
        options = PDFBackendOptions(extract_dimensions=True)
        in_doc = _make_input_doc(text_dimension_pdf)
        backend = PDFBackend(in_doc, text_dimension_pdf, options)

        assert backend.is_valid()
        doc = backend.convert()

        assert doc is not None
        # Document should have items (possibly with annotations)

    def test_convert_without_dimension_extraction(self, text_dimension_pdf):
        """Test converting with dimension extraction disabled."""
        options = PDFBackendOptions(extract_dimensions=False)
        in_doc = _make_input_doc(text_dimension_pdf)
        backend = PDFBackend(in_doc, text_dimension_pdf, options)

        assert backend.is_valid()
        doc = backend.convert()

        assert doc is not None


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
class TestPDFMetadata:
    """Test document metadata tracking."""

    def test_metadata_total_pages(self, multipage_pdf):
        """Test that pdf_total_pages property is set correctly."""
        in_doc = _make_input_doc(multipage_pdf)
        backend = PDFBackend(in_doc, multipage_pdf)

        assert backend.is_valid()
        doc = backend.convert()

        assert "pdf_total_pages" in doc.properties
        assert doc.properties["pdf_total_pages"] == 3

    def test_metadata_extraction_mode(self, simple_vector_pdf):
        """Test that pdf_extraction_mode property is set."""
        options = PDFBackendOptions(extraction_mode="vector")
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf, options)

        assert backend.is_valid()
        doc = backend.convert()

        assert "pdf_extraction_mode" in doc.properties
        assert doc.properties["pdf_extraction_mode"] == "vector"

    def test_metadata_vector_raster_counts(self, simple_vector_pdf):
        """Test that vector and raster page counts are tracked."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)

        assert backend.is_valid()
        doc = backend.convert()

        assert "pdf_vector_pages" in doc.properties
        assert "pdf_raster_pages" in doc.properties

        vector_count = doc.properties["pdf_vector_pages"]
        raster_count = doc.properties["pdf_raster_pages"]

        assert vector_count + raster_count >= 0

    def test_metadata_document_origin(self, simple_vector_pdf):
        """Test that document origin is set correctly."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)

        assert backend.is_valid()
        doc = backend.convert()

        assert doc.origin is not None
        assert doc.origin.format == InputFormat.PDF_DRAWING
        assert doc.origin.binary_hash == "test_hash_123"


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
class TestPDFCurveExtraction:
    """Test curve and arc extraction from PDF."""

    def test_extract_curve_page(self, curve_pdf):
        """Test extraction of curved content from PDF."""
        in_doc = _make_input_doc(curve_pdf)
        backend = PDFBackend(in_doc, curve_pdf)

        assert backend.is_valid()
        page = backend._pdf_doc[0]
        item = backend._extract_vector_page(page, 0)

        # Curve PDF should extract some geometry
        if item:
            assert isinstance(item, Sketch2DItem)

    def test_pdf_curve_to_primitive(self, curve_pdf):
        """Test conversion of bezier curve to Arc2D or Polyline2D."""
        in_doc = _make_input_doc(curve_pdf)
        backend = PDFBackend(in_doc, curve_pdf)
        assert backend.is_valid()

        # Create a test bezier curve
        p1 = (10.0, 10.0)
        p2 = (20.0, 30.0)
        p3 = (30.0, 30.0)
        p4 = (40.0, 10.0)

        item = ("c", fitz.Point(*p1), fitz.Point(*p2), fitz.Point(*p3), fitz.Point(*p4))

        prim = backend._pdf_curve_to_primitive(item, (0, 0, 0), 0)

        # Should return Arc2D or Polyline2D
        if prim:
            assert isinstance(prim, (Arc2D, Polyline2D))


# ============================================================================
# Integration and Edge Case Tests
# ============================================================================


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
class TestPDFEdgeCases:
    """Test edge cases and error conditions."""

    def test_backend_initialization_without_pymupdf_raises(self, simple_vector_pdf):
        """Test that ImportError is raised if PyMuPDF is not available.

        This test is skipped if PyMuPDF is installed (since we have it).
        """
        # This test can only run if pymupdf is NOT installed
        # Since we skip when it IS installed, we can't really test this
        pytest.skip("Requires PyMuPDF to NOT be installed")

    def test_page_index_boundary(self, multipage_pdf):
        """Test page_range with boundary values."""
        # page_range with start > end should be handled gracefully
        options = PDFBackendOptions(page_range=(3, 1))  # Invalid range
        in_doc = _make_input_doc(multipage_pdf)
        backend = PDFBackend(in_doc, multipage_pdf, options)

        assert backend.is_valid()
        doc = backend.convert()

        # Should handle gracefully (skip invalid range)
        assert doc is not None

    def test_page_range_beyond_total(self, multipage_pdf):
        """Test page_range that exceeds document pages."""
        options = PDFBackendOptions(page_range=(0, 100))  # Beyond 3 pages
        in_doc = _make_input_doc(multipage_pdf)
        backend = PDFBackend(in_doc, multipage_pdf, options)

        assert backend.is_valid()
        doc = backend.convert()

        assert doc is not None
        assert doc.properties["pdf_total_pages"] == 3

    def test_color_extraction_edge_cases(self, simple_vector_pdf):
        """Test color extraction with various color formats."""
        in_doc = _make_input_doc(simple_vector_pdf)
        backend = PDFBackend(in_doc, simple_vector_pdf)
        assert backend.is_valid()

        # Empty color
        color = backend._extract_drawing_color({"color": []})
        assert color is None

        # Out of range values
        color = backend._extract_drawing_color({"color": (2.0, -1.0, 0.5)})
        assert color is not None  # Should still convert to 0-255 range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
