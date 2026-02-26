"""PDF engineering drawing backend.

This module provides a backend for extracting 2D geometry from PDF files,
supporting both vector-based PDFs (where drawing paths are extractable) and
raster/scanned PDFs (which require VLM/OCR fallback).

The backend uses PyMuPDF (fitz) for PDF parsing:
  - **Vector path**: Extracts drawing operators (lines, arcs, curves) from
    each page using ``page.get_drawings()`` and converts them to Primitive2D
    models. Text annotations near geometry are extracted as DimensionAnnotations.
  - **Raster path**: Renders pages to images at the configured DPI and creates
    CADItems suitable for VLM-based geometry description.

Auto-detection chooses the appropriate path per page based on the ratio of
vector content vs. embedded images.

Data flow (vector):
    PDF file → pymupdf page.get_drawings() → path segments
    → classify segments (line/curve/arc) → Primitive2D models
    → SketchProfile per page → Sketch2DItem → CADlingDocument

Data flow (raster):
    PDF file → pymupdf page.get_pixmap() → PIL Image
    → CADItem with rendered image → VLM pipeline (downstream)

Classes:
    PDFBackend: DeclarativeCADBackend for PDF engineering drawings.

Example:
    from cadling.backend.pdf_backend import PDFBackend
    from cadling.datamodel import CADInputDocument, InputFormat, PDFBackendOptions

    in_doc = CADInputDocument(
        file=Path("drawing.pdf"),
        format=InputFormat.PDF_DRAWING,
        document_hash="abc123",
    )
    options = PDFBackendOptions(extraction_mode="auto", dpi=300)
    backend = PDFBackend(in_doc, Path("drawing.pdf"), options)
    if backend.is_valid():
        doc = backend.convert()
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from cadling.backend.abstract_backend import DeclarativeCADBackend
from cadling.datamodel.backend_options import BackendOptions, PDFBackendOptions
from cadling.datamodel.base_models import (
    CADDocumentOrigin,
    CADItem,
    CADItemLabel,
    CADlingDocument,
    InputFormat,
)
from cadling.datamodel.geometry_2d import (
    Arc2D,
    Circle2D,
    DimensionAnnotation,
    DimensionType,
    Line2D,
    Polyline2D,
    Primitive2D,
    Sketch2DItem,
    SketchProfile,
    Spline2D,
)

_log = logging.getLogger(__name__)

# Try to import pymupdf — it's an optional dependency
try:
    import fitz  # PyMuPDF

    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False
    _log.info(
        "PyMuPDF (fitz) not available — PDF backend will not function. "
        "Install with: pip install pymupdf"
    )

# Try to import PIL for raster path
try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# Regex patterns for dimension text extraction
_DIM_PATTERNS = [
    # Diameter: ∅12.5, Ø12, ⌀10
    (re.compile(r"[∅Ø⌀]\s*(\d+\.?\d*)"), DimensionType.DIAMETER),
    # Radius: R5, R3.2
    (re.compile(r"R\s*(\d+\.?\d*)"), DimensionType.RADIAL),
    # Angular: 45°, 90.5°
    (re.compile(r"(\d+\.?\d*)\s*°"), DimensionType.ANGULAR),
    # Linear: plain number with optional unit — 25.4, 100mm, 3.5"
    (re.compile(r"(\d+\.?\d*)\s*(?:mm|cm|in|\")?$"), DimensionType.LINEAR),
]


class PDFBackend(DeclarativeCADBackend):
    """Backend for extracting 2D geometry from PDF engineering drawings.

    Supports dual-path processing:

    **Vector path** (extraction_mode="vector" or auto-detected):
        Extracts line/curve paths from PDF drawing operators using PyMuPDF's
        ``page.get_drawings()`` API. Best for born-digital engineering PDFs
        created from CAD systems.

    **Raster path** (extraction_mode="raster" or auto-detected):
        Renders PDF pages to images for downstream VLM processing. Best for
        scanned engineering drawings or PDFs with embedded images.

    **Auto mode** (extraction_mode="auto", the default):
        Analyzes each page to determine whether it contains extractable
        vector content or is primarily raster. Falls back to raster when
        insufficient vector geometry is found.

    Attributes:
        _pdf_doc: Parsed PyMuPDF document.
        _options: PDFBackendOptions controlling extraction behavior.
    """

    def __init__(
        self,
        in_doc: "CADInputDocument",
        path_or_stream: Union[Path, str, BytesIO],
        options: Optional[BackendOptions] = None,
    ):
        """Initialize PDF backend.

        Args:
            in_doc: Input document descriptor.
            path_or_stream: Path to PDF file or byte stream.
            options: PDFBackendOptions (defaults applied if None).

        Raises:
            ImportError: If PyMuPDF is not installed.
        """
        if not _HAS_PYMUPDF:
            raise ImportError(
                "PyMuPDF is required for PDF support. "
                "Install with: pip install 'cadling[drawings]'"
            )

        super().__init__(in_doc, path_or_stream, options)
        self._options: PDFBackendOptions = (
            options if isinstance(options, PDFBackendOptions)
            else PDFBackendOptions()
        )
        self._pdf_doc = None

    # ------------------------------------------------------------------
    # Class methods (interface contract)
    # ------------------------------------------------------------------

    @classmethod
    def supports_text_parsing(cls) -> bool:
        """PDF backend supports text parsing (vector path)."""
        return True

    @classmethod
    def supports_rendering(cls) -> bool:
        """PDF backend does not support 3D rendering."""
        return False

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        """Return formats handled by this backend."""
        return {InputFormat.PDF_DRAWING, InputFormat.PDF_RASTER}

    @classmethod
    def _get_default_options(cls) -> PDFBackendOptions:
        """Return default PDF backend options."""
        return PDFBackendOptions()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def is_valid(self) -> bool:
        """Validate that the input is a parseable PDF file.

        Attempts to open the file with PyMuPDF. If successful, caches the
        parsed document for use by ``convert()``.

        Returns:
            True if PyMuPDF can open the file, False otherwise.
        """
        try:
            if isinstance(self.path_or_stream, BytesIO):
                self.path_or_stream.seek(0)
                self._pdf_doc = fitz.open(stream=self.path_or_stream, filetype="pdf")
            else:
                self._pdf_doc = fitz.open(str(self.path_or_stream))

            if self._pdf_doc.page_count < 1:
                _log.warning("PDF has no pages: %s", self.file.name)
                return False

            _log.debug(
                "PDF validation passed for %s (%d pages)",
                self.file.name,
                self._pdf_doc.page_count,
            )
            return True
        except (fitz.FileDataError, fitz.FileNotFoundError, fitz.EmptyFileError,
                RuntimeError, ValueError, IOError, OSError) as exc:
            _log.warning("PDF validation failed for %s: %s", self.file.name, exc)
            self._pdf_doc = None
            return False

    # ------------------------------------------------------------------
    # Conversion (main entry point)
    # ------------------------------------------------------------------

    def convert(self) -> CADlingDocument:
        """Parse PDF file and convert to CADlingDocument.

        Processes each page according to the extraction mode:
        - Vector: extracts drawing paths as Primitive2D models
        - Raster: renders page to image for VLM processing
        - Auto: detects per-page and chooses the best path

        Returns:
            CADlingDocument containing Sketch2DItem instances (vector pages)
            and/or CADItem instances (raster pages).

        Raises:
            RuntimeError: If file hasn't been validated or PyMuPDF fails.
        """
        if self._pdf_doc is None:
            if not self.is_valid():
                raise RuntimeError(
                    f"Cannot convert invalid PDF file: {self.file.name}"
                )

        # Create document
        doc = CADlingDocument(
            name=self.file.name,
            format=InputFormat.PDF_DRAWING,
            origin=CADDocumentOrigin(
                filename=self.file.name,
                format=InputFormat.PDF_DRAWING,
                binary_hash=self.document_hash,
            ),
            hash=self.document_hash,
        )

        # Determine page range
        total_pages = self._pdf_doc.page_count
        if self._options.page_range:
            start_page, end_page = self._options.page_range
            start_page = max(0, start_page)
            end_page = min(total_pages, end_page)
        else:
            start_page = 0
            end_page = total_pages

        vector_page_count = 0
        raster_page_count = 0

        for page_idx in range(start_page, end_page):
            page = self._pdf_doc[page_idx]

            # Determine extraction mode for this page
            mode = self._options.extraction_mode
            if mode == "auto":
                mode = self._detect_page_type(page)

            if mode == "vector":
                item = self._extract_vector_page(page, page_idx)
                if item and item.total_primitives > 0:
                    doc.add_item(item)
                    vector_page_count += 1
                else:
                    # Fall back to raster if vector extraction yields nothing
                    _log.debug(
                        "Page %d vector extraction empty, falling back to raster",
                        page_idx,
                    )
                    item = self._render_raster_page(page, page_idx)
                    if item:
                        doc.add_item(item)
                        raster_page_count += 1
            else:
                item = self._render_raster_page(page, page_idx)
                if item:
                    doc.add_item(item)
                    raster_page_count += 1

        # Store extraction metadata
        doc.properties["pdf_total_pages"] = total_pages
        doc.properties["pdf_vector_pages"] = vector_page_count
        doc.properties["pdf_raster_pages"] = raster_page_count
        doc.properties["pdf_extraction_mode"] = self._options.extraction_mode

        _log.info(
            "Converted PDF %s: %d vector pages, %d raster pages (of %d total)",
            self.file.name,
            vector_page_count,
            raster_page_count,
            total_pages,
        )

        return doc

    # ------------------------------------------------------------------
    # Page type detection
    # ------------------------------------------------------------------

    def _detect_page_type(self, page) -> Literal["vector", "raster"]:
        """Detect whether a PDF page contains vector or raster content.

        Heuristic based on the ratio of vector drawing paths to embedded
        images. Pages with significant vector content (>10 drawing paths)
        are treated as vector; pages dominated by images are treated as raster.

        Args:
            page: PyMuPDF Page object.

        Returns:
            "vector" or "raster" based on content analysis.
        """
        try:
            drawings = page.get_drawings()
            images = page.get_images()

            num_drawings = len(drawings) if drawings else 0
            num_images = len(images) if images else 0

            _log.debug(
                "Page type detection: %d drawings, %d images",
                num_drawings,
                num_images,
            )

            # If significant vector content exists, use vector path
            if num_drawings > 10:
                return "vector"

            # If mostly images with little vector content, use raster
            if num_images > 0 and num_drawings < 5:
                return "raster"

            # Few drawings and few images — try vector (may fall back later)
            if num_drawings > 0:
                return "vector"

            return "raster"

        except Exception:
            return "raster"

    # ------------------------------------------------------------------
    # Vector extraction
    # ------------------------------------------------------------------

    def _extract_vector_page(
        self, page, page_idx: int
    ) -> Optional[Sketch2DItem]:
        """Extract vector geometry from a PDF page.

        Uses PyMuPDF's ``page.get_drawings()`` to extract drawing paths,
        then classifies each path segment into Primitive2D models.

        Args:
            page: PyMuPDF Page object.
            page_idx: Zero-based page index.

        Returns:
            Sketch2DItem containing extracted geometry, or None on failure.
        """
        try:
            drawings = page.get_drawings()
            if not drawings:
                return None

            primitives: List[Primitive2D] = []
            annotations: List[DimensionAnnotation] = []

            for drawing in drawings:
                # Each drawing is a dict with 'items' list of path commands
                items = drawing.get("items", [])
                color_rgb = self._extract_drawing_color(drawing)

                for item in items:
                    cmd = item[0]  # Command type: 'l' (line), 'c' (curve), etc.

                    if cmd == "l":
                        # Line: ('l', start_point, end_point)
                        prim = self._pdf_line_to_primitive(item, color_rgb, page_idx)
                        if prim:
                            primitives.append(prim)

                    elif cmd == "c":
                        # Cubic bezier curve: ('c', p1, p2, p3, p4)
                        prim = self._pdf_curve_to_primitive(item, color_rgb, page_idx)
                        if prim:
                            primitives.append(prim)

                    elif cmd == "re":
                        # Rectangle: ('re', rect)
                        rect_prims = self._pdf_rect_to_primitives(
                            item, color_rgb, page_idx
                        )
                        primitives.extend(rect_prims)

                    elif cmd == "qu":
                        # Quad: ('qu', quad)
                        quad_prims = self._pdf_quad_to_primitives(
                            item, color_rgb, page_idx
                        )
                        primitives.extend(quad_prims)

            # Extract dimension annotations from text
            if self._options.extract_dimensions:
                text_annotations = self._extract_text_dimensions(page, page_idx)
                annotations.extend(text_annotations)

            if not primitives and not annotations:
                return None

            # Build profile
            profile = SketchProfile(
                profile_id=f"page_{page_idx}",
                primitives=primitives,
                annotations=annotations,
                closed=False,
            )
            profile.compute_bounds()

            item = Sketch2DItem(
                item_type="sketch_2d",
                label=CADItemLabel(
                    text=f"{self.file.stem} - Page {page_idx + 1}"
                ),
                profiles=[profile],
                source_page=page_idx,
            )

            _log.debug(
                "Extracted %d primitives and %d annotations from page %d",
                len(primitives),
                len(annotations),
                page_idx,
            )

            return item

        except Exception as exc:
            _log.warning(
                "Failed to extract vector content from page %d: %s",
                page_idx,
                exc,
            )
            return None

    def _extract_drawing_color(
        self, drawing: Dict[str, Any]
    ) -> Optional[Tuple[int, int, int]]:
        """Extract RGB color from a PDF drawing dict.

        Args:
            drawing: PyMuPDF drawing dictionary.

        Returns:
            (R, G, B) tuple (0-255) or None.
        """
        try:
            color = drawing.get("color")
            if color and len(color) >= 3:
                return (
                    int(color[0] * 255),
                    int(color[1] * 255),
                    int(color[2] * 255),
                )
        except (IndexError, TypeError, ValueError) as e:
            _log.debug("Failed to extract drawing color: %s", e)
        return None

    def _pdf_line_to_primitive(
        self,
        item: tuple,
        color: Optional[Tuple[int, int, int]],
        page_idx: int,
    ) -> Optional[Line2D]:
        """Convert a PDF line path command to Line2D.

        Args:
            item: ('l', start_point, end_point) tuple.
            color: RGB color tuple.
            page_idx: Page index for layer naming.

        Returns:
            Line2D instance or None.
        """
        try:
            _, p1, p2 = item
            return Line2D(
                start=(float(p1.x), float(p1.y)),
                end=(float(p2.x), float(p2.y)),
                layer=f"page_{page_idx}",
                color=color,
            )
        except (ValueError, TypeError, AttributeError) as e:
            _log.debug("Failed to convert PDF line to primitive (page %d): %s", page_idx, e)
            return None

    def _pdf_curve_to_primitive(
        self,
        item: tuple,
        color: Optional[Tuple[int, int, int]],
        page_idx: int,
    ) -> Optional[Primitive2D]:
        """Convert a PDF cubic bezier curve to a Primitive2D.

        Attempts to classify the curve:
        - If it approximates a circular arc, returns Arc2D
        - Otherwise, samples the curve into a Polyline2D

        Args:
            item: ('c', p1, p2, p3, p4) cubic bezier tuple.
            color: RGB color tuple.
            page_idx: Page index for layer naming.

        Returns:
            Arc2D or Polyline2D instance, or None.
        """
        try:
            _, p1, p2, p3, p4 = item

            # Attempt arc classification
            arc = self._try_classify_as_arc(
                (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)
            )
            if arc:
                arc.layer = f"page_{page_idx}"
                arc.color = color
                return arc

            # Fallback: sample curve into polyline
            points = self._sample_bezier(
                (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y),
                num_segments=8,
            )
            return Polyline2D(
                points=points,
                closed=False,
                layer=f"page_{page_idx}",
                color=color,
                confidence=0.8,  # Sampled approximation
            )
        except (ValueError, TypeError, IndexError) as e:
            _log.debug("Failed to convert PDF curve to primitive (page %d): %s", page_idx, e)
            return None

    def _pdf_rect_to_primitives(
        self,
        item: tuple,
        color: Optional[Tuple[int, int, int]],
        page_idx: int,
    ) -> List[Line2D]:
        """Convert a PDF rectangle to four Line2D segments.

        Args:
            item: ('re', rect) tuple where rect is a fitz.Rect.
            color: RGB color tuple.
            page_idx: Page index for layer naming.

        Returns:
            List of four Line2D segments forming the rectangle.
        """
        try:
            _, rect = item
            x0, y0, x1, y1 = float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)
            layer = f"page_{page_idx}"
            return [
                Line2D(start=(x0, y0), end=(x1, y0), layer=layer, color=color),
                Line2D(start=(x1, y0), end=(x1, y1), layer=layer, color=color),
                Line2D(start=(x1, y1), end=(x0, y1), layer=layer, color=color),
                Line2D(start=(x0, y1), end=(x0, y0), layer=layer, color=color),
            ]
        except (ValueError, TypeError, AttributeError) as e:
            _log.debug("Failed to convert PDF rect to primitives (page %d): %s", page_idx, e)
            return []

    def _pdf_quad_to_primitives(
        self,
        item: tuple,
        color: Optional[Tuple[int, int, int]],
        page_idx: int,
    ) -> List[Line2D]:
        """Convert a PDF quadrilateral to four Line2D segments.

        Args:
            item: ('qu', quad) tuple where quad is a fitz.Quad.
            color: RGB color tuple.
            page_idx: Page index for layer naming.

        Returns:
            List of four Line2D segments forming the quad.
        """
        try:
            _, quad = item
            pts = [
                (float(quad.ul.x), float(quad.ul.y)),
                (float(quad.ur.x), float(quad.ur.y)),
                (float(quad.lr.x), float(quad.lr.y)),
                (float(quad.ll.x), float(quad.ll.y)),
            ]
            layer = f"page_{page_idx}"
            lines = []
            for i in range(4):
                j = (i + 1) % 4
                lines.append(
                    Line2D(start=pts[i], end=pts[j], layer=layer, color=color)
                )
            return lines
        except (ValueError, TypeError, AttributeError) as e:
            _log.debug("Failed to convert PDF quad to primitives (page %d): %s", page_idx, e)
            return []

    # ------------------------------------------------------------------
    # Arc classification from bezier curves
    # ------------------------------------------------------------------

    @staticmethod
    def _try_classify_as_arc(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
        tolerance: float = 0.02,
    ) -> Optional[Arc2D]:
        """Try to classify a cubic bezier as a circular arc.

        A cubic bezier that approximates a circular arc has its control
        points arranged symmetrically around the arc's center. This method
        checks if the four bezier control points are consistent with a
        circular arc within the given tolerance.

        Args:
            p1, p2, p3, p4: Bezier control points as (x, y) tuples.
            tolerance: Relative tolerance for arc classification.

        Returns:
            Arc2D if classification succeeds, None otherwise.
        """
        # Compute candidate center from perpendicular bisectors of chord
        # For a true circular arc, midpoints of p1-p2 and p3-p4 should
        # point toward the center
        mx1 = (p1[0] + p4[0]) / 2
        my1 = (p1[1] + p4[1]) / 2

        # Check if all points are roughly equidistant from a candidate center
        # Use the three-point circle formula with p1, midpoint of curve, p4
        mid_t = 0.5  # Parameter at midpoint
        mid_x = (
            (1 - mid_t) ** 3 * p1[0]
            + 3 * (1 - mid_t) ** 2 * mid_t * p2[0]
            + 3 * (1 - mid_t) * mid_t ** 2 * p3[0]
            + mid_t ** 3 * p4[0]
        )
        mid_y = (
            (1 - mid_t) ** 3 * p1[1]
            + 3 * (1 - mid_t) ** 2 * mid_t * p2[1]
            + 3 * (1 - mid_t) * mid_t ** 2 * p3[1]
            + mid_t ** 3 * p4[1]
        )

        # Three-point circle: p1, (mid_x, mid_y), p4
        center = PDFBackend._circle_center_3pts(
            p1, (mid_x, mid_y), p4
        )
        if center is None:
            return None

        cx, cy = center
        r1 = math.sqrt((p1[0] - cx) ** 2 + (p1[1] - cy) ** 2)
        r_mid = math.sqrt((mid_x - cx) ** 2 + (mid_y - cy) ** 2)
        r4 = math.sqrt((p4[0] - cx) ** 2 + (p4[1] - cy) ** 2)

        if r1 < 1e-6:
            return None

        # Check radii consistency
        if abs(r1 - r_mid) / r1 > tolerance or abs(r1 - r4) / r1 > tolerance:
            return None

        radius = (r1 + r_mid + r4) / 3

        # Compute angles
        start_angle = math.degrees(math.atan2(p1[1] - cy, p1[0] - cx))
        end_angle = math.degrees(math.atan2(p4[1] - cy, p4[0] - cx))

        return Arc2D(
            center=(cx, cy),
            radius=radius,
            start_angle=start_angle % 360,
            end_angle=end_angle % 360,
            confidence=0.9,  # Classification confidence
        )

    @staticmethod
    def _circle_center_3pts(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        """Compute center of circle passing through three points.

        Uses the perpendicular bisector method.

        Args:
            p1, p2, p3: Three points as (x, y) tuples.

        Returns:
            (cx, cy) center point, or None if points are collinear.
        """
        ax, ay = p1
        bx, by = p2
        cx, cy = p3

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            return None  # Collinear

        ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) + (cx ** 2 + cy ** 2) * (ay - by)) / d
        uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) + (cx ** 2 + cy ** 2) * (bx - ax)) / d

        return (ux, uy)

    @staticmethod
    def _sample_bezier(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
        num_segments: int = 8,
    ) -> List[Tuple[float, float]]:
        """Sample a cubic bezier curve into discrete points.

        Args:
            p1, p2, p3, p4: Bezier control points.
            num_segments: Number of segments to sample.

        Returns:
            List of (x, y) sample points including endpoints.
        """
        points = []
        for i in range(num_segments + 1):
            t = i / num_segments
            mt = 1 - t
            x = mt**3 * p1[0] + 3 * mt**2 * t * p2[0] + 3 * mt * t**2 * p3[0] + t**3 * p4[0]
            y = mt**3 * p1[1] + 3 * mt**2 * t * p2[1] + 3 * mt * t**2 * p3[1] + t**3 * p4[1]
            points.append((x, y))
        return points

    # ------------------------------------------------------------------
    # Text / dimension annotation extraction
    # ------------------------------------------------------------------

    def _extract_text_dimensions(
        self, page, page_idx: int
    ) -> List[DimensionAnnotation]:
        """Extract dimension annotations from page text.

        Scans text blocks on the page for dimension-like patterns
        (e.g., "∅12.5", "R5", "45°", bare numbers near geometry).

        Args:
            page: PyMuPDF Page object.
            page_idx: Page index.

        Returns:
            List of DimensionAnnotation instances.
        """
        annotations = []

        try:
            text_blocks = page.get_text("dict")["blocks"]

            for block in text_blocks:
                if block.get("type") != 0:  # Skip non-text blocks
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue

                        # Try to match dimension patterns
                        for pattern, dim_type in _DIM_PATTERNS:
                            match = pattern.search(text)
                            if match:
                                try:
                                    value = float(match.group(1))
                                except (ValueError, IndexError):
                                    continue

                                # Get text position as attachment point
                                bbox = span.get("bbox", (0, 0, 0, 0))
                                center_x = (bbox[0] + bbox[2]) / 2
                                center_y = (bbox[1] + bbox[3]) / 2

                                annotations.append(
                                    DimensionAnnotation(
                                        dim_type=dim_type,
                                        value=value,
                                        text=text,
                                        attachment_points=[(center_x, center_y)],
                                        layer=f"page_{page_idx}",
                                        confidence=0.7,  # Text-extracted
                                    )
                                )
                                break  # First match wins per text span

        except Exception as exc:
            _log.debug("Failed to extract text dimensions from page %d: %s", page_idx, exc)

        return annotations

    # ------------------------------------------------------------------
    # Raster fallback
    # ------------------------------------------------------------------

    def _render_raster_page(
        self, page, page_idx: int
    ) -> Optional[CADItem]:
        """Render a PDF page to a raster image for VLM processing.

        Creates a CADItem with the rendered page image stored in properties,
        suitable for downstream VLM-based geometry extraction.

        Args:
            page: PyMuPDF Page object.
            page_idx: Page index.

        Returns:
            CADItem with rendered image data, or None on failure.
        """
        try:
            # Render page to pixmap at configured DPI
            zoom = self._options.dpi / 72.0  # PDF default is 72 DPI
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix)

            # Convert to PIL Image if available
            image_data = None
            if _HAS_PIL:
                image_data = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )

            # Create CADItem for VLM pipeline
            item = CADItem(
                item_type="pdf_raster_page",
                label=CADItemLabel(
                    text=f"{self.file.stem} - Page {page_idx + 1} (raster)"
                ),
                properties={
                    "page_index": page_idx,
                    "dpi": self._options.dpi,
                    "width_px": pixmap.width,
                    "height_px": pixmap.height,
                    "page_width_pt": page.rect.width,
                    "page_height_pt": page.rect.height,
                    "extraction_mode": "raster",
                    "requires_vlm": True,
                },
            )

            # Store image as bytes in properties for VLM consumption
            if image_data:
                img_buffer = BytesIO()
                image_data.save(img_buffer, format="PNG")
                item.properties["image_png_bytes"] = img_buffer.getvalue()

            _log.debug(
                "Rendered page %d to raster: %dx%d px at %d DPI",
                page_idx,
                pixmap.width,
                pixmap.height,
                self._options.dpi,
            )

            return item

        except Exception as exc:
            _log.warning(
                "Failed to render page %d to raster: %s", page_idx, exc
            )
            return None
