"""
Hybrid pipeline combining text parsing with vision analysis.

This pipeline processes CAD files using both text-based parsing
(STEP entities, STL mesh) and vision-based annotation extraction (VLM).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from cadling.pipeline.base_pipeline import BaseCADPipeline
from cadling.pipeline._vision_shared import (
    create_vlm_prompt,
    generate_geometry_annotations,
    parse_vlm_response,
    process_views_with_vlm,
    render_views,
)

if TYPE_CHECKING:
    from cadling.datamodel.base_models import ConversionResult
    from cadling.datamodel.pipeline_options import CADVlmPipelineOptions

_log = logging.getLogger(__name__)


class HybridPipeline(BaseCADPipeline):
    """
    Hybrid pipeline combining text parsing with vision analysis.

    This pipeline:
    1. Build: Parses CAD file text AND renders images from multiple views
    2. Assemble: Processes images with VLM and merges with text-based data
    3. Enrich: Applies enrichment models to combined data

    The hybrid approach provides:
    - Complete geometric data from text parsing (exact coordinates, topology)
    - Visual annotations from rendered images (dimensions, tolerances, notes)
    - Cross-validation between text and vision modalities

    Example:
        from cadling.models.vlm_model import ApiVlmModel
        from cadling.datamodel.pipeline_options import CADVlmPipelineOptions

        vlm = ApiVlmModel(provider="openai", model="gpt-4-vision-preview")
        options = CADVlmPipelineOptions(
            vlm_model=vlm,
            views_to_render=["front", "top", "isometric"],
            do_topology_analysis=True
        )
        pipeline = HybridPipeline(options)
        result = pipeline.execute(input_doc)
    """

    def __init__(self, pipeline_options: CADVlmPipelineOptions):
        """Initialize hybrid pipeline."""
        super().__init__(pipeline_options)
        self.vlm_options = pipeline_options.vlm_options
        self.views_to_render = pipeline_options.views_to_render
        self.do_topology = pipeline_options.do_topology_analysis

        _log.debug(
            f"Initialized HybridPipeline with topology={self.do_topology}, "
            f"views={self.views_to_render}"
        )

    @classmethod
    def get_default_options(cls) -> CADVlmPipelineOptions:
        """Get default pipeline options for hybrid."""
        from cadling.datamodel.pipeline_options import CADVlmPipelineOptions, VlmOptions

        return CADVlmPipelineOptions(
            views_to_render=["front", "top", "isometric"],
            vlm_options=VlmOptions(),
            do_topology_analysis=True,
            device="cpu",
        )

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """
        Build: Parse CAD file text AND render images.

        This combines text parsing (like STEPPipeline) with image rendering
        (like VisionPipeline).

        Args:
            conv_res: Conversion result to populate.

        Returns:
            Updated conversion result with both text data and rendered images.
        """
        try:
            backend = conv_res.input._backend

            # Phase 1: Text-based parsing
            _log.info("Phase 1: Text-based parsing")

            # Check if backend is declarative (supports text parsing)
            from cadling.backend.abstract_backend import DeclarativeCADBackend

            if isinstance(backend, DeclarativeCADBackend):
                _log.debug("Parsing CAD file with text-based backend")
                document = backend.convert()
                conv_res.document = document

                _log.info(
                    f"Text parsing completed: {len(document.items)} items, "
                    f"{document.topology.num_nodes if document.topology else 0} topology nodes"
                )
            else:
                # Create empty document
                from cadling.datamodel.base_models import CADlingDocument

                document = CADlingDocument(name=conv_res.input.file.name)
                conv_res.document = document
                _log.warning("Backend doesn't support text parsing")

            # Phase 2: Image rendering using shared utility
            _log.info("Phase 2: Image rendering")

            from cadling.backend.abstract_backend import RenderableCADBackend

            if isinstance(backend, RenderableCADBackend):
                _log.debug(f"Rendering {len(self.views_to_render)} views")

                rendered_views = render_views(
                    backend=backend,
                    views_to_render=self.views_to_render,
                    resolution=(
                        self.vlm_options.image_resolution
                        if self.vlm_options
                        else 1024
                    ),
                    component_name="HybridPipeline",
                    conv_res=conv_res,
                )

                # Store rendered views in document metadata
                if not document.metadata:
                    document.metadata = {}
                document.metadata["rendered_views"] = rendered_views

                _log.info(f"Rendering completed: {len(rendered_views)} views")

            else:
                _log.warning("Backend doesn't support rendering")

            _log.info(
                f"Build completed: {len(document.items)} items, "
                f"{len(document.metadata.get('rendered_views', {}))} views"
            )

            return conv_res

        except Exception as e:
            _log.exception(f"Build stage failed: {e}")
            conv_res.add_error(
                component="HybridPipeline._build_document",
                error_message=f"Failed to build document: {str(e)}",
            )
            raise

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """
        Assemble: Process images with VLM and merge with text data.

        Args:
            conv_res: Conversion result to assemble.

        Returns:
            Updated conversion result with merged data.
        """
        if not conv_res.document:
            return conv_res

        try:
            document = conv_res.document

            # Get rendered views
            rendered_views = document.metadata.get("rendered_views", {})

            if not rendered_views:
                _log.warning("No rendered views found, using text-only analysis fallback")
                return self._text_only_analysis(conv_res)

            # Get VLM model
            vlm_model = self.vlm_options.vlm_model if self.vlm_options else None

            if not vlm_model:
                _log.warning("No VLM model configured, using text-only analysis fallback")
                return self._text_only_analysis(conv_res)

            # Store initial item count (from text parsing)
            initial_item_count = len(document.items)

            # Process views with VLM using shared utility
            custom_prompt = self.vlm_options.prompt if self.vlm_options else None
            vision_items = process_views_with_vlm(
                rendered_views=rendered_views,
                vlm_model=vlm_model,
                document=document,
                custom_prompt=custom_prompt,
                include_extended_types=True,
                component_name="HybridPipeline",
                conv_res=conv_res,
            )

            # Log summary
            _log.info(
                f"Assembly completed: {initial_item_count} text items + "
                f"{vision_items} vision items = {len(document.items)} total"
            )

            # Add summary to metadata
            document.metadata["hybrid_summary"] = {
                "text_items": initial_item_count,
                "vision_items": vision_items,
                "total_items": len(document.items),
                "views_processed": list(rendered_views.keys()),
            }

            return conv_res

        except Exception as e:
            _log.exception(f"Assembly stage failed: {e}")
            conv_res.add_error(
                component="HybridPipeline._assemble_document",
                error_message=f"Assembly failed: {str(e)}",
            )
            return conv_res

    def _create_vlm_prompt(self, view_name: str) -> str:
        """Create prompt for VLM based on view name."""
        custom_prompt = self.vlm_options.prompt if self.vlm_options else None
        return create_vlm_prompt(
            view_name,
            custom_prompt=custom_prompt,
            include_extended_types=True,
        )

    def _parse_vlm_response(self, response: Any, view_name: str) -> list:
        """Parse VLM response and create CAD items."""
        return parse_vlm_response(response, view_name)

    def _text_only_analysis(self, conv_res: ConversionResult) -> ConversionResult:
        """Fallback text-only analysis when VLM/rendered views unavailable.

        Extracts annotations from STEP entity text, geometry analysis results,
        and other text-based sources.

        Args:
            conv_res: Conversion result with document

        Returns:
            Updated conversion result with text-based annotations
        """
        if not conv_res.document:
            return conv_res

        annotations_added = generate_geometry_annotations(
            document=conv_res.document,
            source_view="text_analysis",
        )

        _log.info(
            f"Text-only analysis added {annotations_added} annotations "
            f"from geometry/surface/feature data"
        )

        return conv_res
