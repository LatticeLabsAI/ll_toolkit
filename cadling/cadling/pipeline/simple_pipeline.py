"""Simple pipeline for declarative CAD backends.

This module provides a simple pipeline for CAD formats that support direct
text-to-document conversion (DeclarativeCADBackend). This is the most basic
pipeline type, similar to docling's SimplePipeline.

Classes:
    SimpleCADPipeline: Pipeline for text-based CAD parsing.

Example:
    from cadling.pipeline.simple_pipeline import SimpleCADPipeline
    from cadling.datamodel.pipeline_options import PipelineOptions

    pipeline = SimpleCADPipeline(PipelineOptions())
    result = pipeline.execute(input_doc)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cadling.pipeline.base_pipeline import BaseCADPipeline

if TYPE_CHECKING:
    from cadling.datamodel.base_models import ConversionResult
    from cadling.backend.abstract_backend import DeclarativeCADBackend

_log = logging.getLogger(__name__)


class SimpleCADPipeline(BaseCADPipeline):
    """Simple pipeline for declarative backends.

    This pipeline is used for CAD formats that have structured text
    representations and can be directly parsed into CADlingDocument
    without requiring view-level processing or rendering.

    Supported backends:
        - STEPBackend (text parsing)
        - STLBackend (ASCII STL)
        - Any backend implementing DeclarativeCADBackend

    The simple pipeline:
        1. Build: Calls backend.convert() to get CADlingDocument
        2. Assemble: No-op (handled by backend)
        3. Enrich: Applies enrichment models if configured

    Example:
        # Convert a STEP file using simple pipeline
        converter = DocumentConverter(
            format_options={
                InputFormat.STEP: STEPFormatOption(
                    pipeline_cls=SimpleCADPipeline
                )
            }
        )
        result = converter.convert("part.step")
    """

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Build document using declarative backend.

        This method delegates directly to the backend's convert() method,
        which returns a fully populated CADlingDocument. If the resulting
        document contains 2D sketch items (from DXF or PDF backends), the
        SketchGeometryExtractor is automatically applied to convert
        primitives into tokenizer-ready command sequences.

        Args:
            conv_res: Conversion result to populate.

        Returns:
            Conversion result with document populated.

        Raises:
            ValueError: If backend is not a DeclarativeCADBackend.
        """
        from cadling.backend.abstract_backend import DeclarativeCADBackend

        backend = conv_res.input._backend

        # Validate backend type
        if not isinstance(backend, DeclarativeCADBackend):
            raise ValueError(
                f"SimpleCADPipeline requires DeclarativeCADBackend, "
                f"got {type(backend).__name__}"
            )

        _log.debug(
            f"Converting {conv_res.input.file.name} "
            f"using {backend.__class__.__name__}"
        )

        # Backend converts directly to CADlingDocument
        conv_res.document = backend.convert()

        _log.debug(
            f"Backend produced document with {len(conv_res.document.items)} items"
        )

        # Auto-apply SketchGeometryExtractor for 2D sketch items
        if conv_res.document:
            self._auto_extract_sketch_geometry(conv_res.document)

        return conv_res

    def _auto_extract_sketch_geometry(self, doc) -> None:
        """Auto-apply SketchGeometryExtractor if document contains 2D sketches.

        Checks whether the document contains any Sketch2DItem instances
        (produced by DXF or PDF backends) and automatically runs the
        SketchGeometryExtractor enrichment model to convert Primitive2D
        geometry into tokenizer-compatible command sequences.

        Args:
            doc: The CADlingDocument to check and enrich.
        """
        from cadling.datamodel.geometry_2d import Sketch2DItem

        sketch_items = [
            item for item in doc.items if isinstance(item, Sketch2DItem)
        ]

        if not sketch_items:
            return

        _log.info(
            "Document contains %d 2D sketch items — "
            "auto-applying SketchGeometryExtractor",
            len(sketch_items),
        )

        try:
            from cadling.models.segmentation.sketch_geometry_extractor import (
                SketchGeometryExtractor,
            )

            extractor = SketchGeometryExtractor()
            extractor(doc, sketch_items)

        except Exception as exc:
            _log.warning(
                "Auto sketch geometry extraction failed: %s", exc
            )
