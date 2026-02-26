"""
Vision pipeline for optical CAD recognition using VLMs.

This pipeline processes rendered CAD images to extract annotations,
dimensions, tolerances, and other visual information using Vision-Language Models.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from cadling.pipeline.base_pipeline import BaseCADPipeline
from cadling.pipeline._vision_shared import (
    create_vlm_prompt,
    generate_geometry_annotations,
    parse_vlm_response,
    process_views_with_vlm,
    render_views,
)
from cadling.datamodel.base_models import CADItem

if TYPE_CHECKING:
    from cadling.datamodel.base_models import ConversionResult
    from cadling.datamodel.pipeline_options import CADVlmPipelineOptions
    from cadling.models.vlm_model import VlmModel
    from PIL import Image

_log = logging.getLogger(__name__)


class VisionPipeline(BaseCADPipeline):
    """
    Pipeline for optical CAD recognition using VLMs.

    This pipeline:
    1. Build: Renders CAD file to images from multiple views
    2. Assemble: Processes images with VLM to extract annotations
    3. Enrich: Applies enrichment models to extracted annotations

    Example:
        from cadling.models.vlm_model import ApiVlmModel
        from cadling.datamodel.pipeline_options import CADVlmPipelineOptions

        vlm = ApiVlmModel(provider="openai", model="gpt-4-vision-preview")
        options = CADVlmPipelineOptions(
            vlm_model=vlm,
            views_to_render=["front", "top", "isometric"]
        )
        pipeline = VisionPipeline(options)
        result = pipeline.execute(input_doc)
    """

    def __init__(self, pipeline_options: CADVlmPipelineOptions):
        """Initialize vision pipeline."""
        super().__init__(pipeline_options)
        self.vlm_options = pipeline_options.vlm_options
        self.views_to_render = pipeline_options.views_to_render

        if not self.vlm_options or not self.vlm_options.vlm_model:
            raise ValueError("VisionPipeline requires vlm_options with a vlm_model")

        _log.debug(
            f"Initialized VisionPipeline with views: {self.views_to_render}"
        )

    @classmethod
    def get_default_options(cls) -> CADVlmPipelineOptions:
        """Get default pipeline options for vision."""
        from cadling.datamodel.pipeline_options import CADVlmPipelineOptions, VlmOptions

        return CADVlmPipelineOptions(
            views_to_render=["front", "top", "isometric"],
            vlm_options=VlmOptions(),
            device="cpu",
        )

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """
        Build: Render CAD file to images from multiple views.

        Args:
            conv_res: Conversion result to populate.

        Returns:
            Updated conversion result with rendered images.
        """
        try:
            # Get the backend from input
            backend = conv_res.input._backend

            # Check if backend supports rendering
            if not backend.supports_rendering():
                raise ValueError(
                    f"VisionPipeline requires a RenderableCADBackend, "
                    f"got {type(backend)} which doesn't support rendering"
                )

            from cadling.datamodel.base_models import CADlingDocument

            # Create document
            document = CADlingDocument(name=conv_res.input.file.name)

            # Render views using shared utility
            _log.info(f"Rendering {len(self.views_to_render)} views")

            rendered_views = render_views(
                backend=backend,
                views_to_render=self.views_to_render,
                resolution=(
                    self.vlm_options.image_resolution
                    if self.vlm_options
                    else 1024
                ),
                component_name="VisionPipeline",
                conv_res=conv_res,
            )

            if not rendered_views:
                raise ValueError("No views were successfully rendered")

            # Store rendered views in document metadata
            document.metadata = {"rendered_views": rendered_views}

            conv_res.document = document

            _log.info(
                f"Built vision document with {len(rendered_views)} rendered views"
            )

            return conv_res

        except Exception as e:
            _log.error(f"Build stage failed: {e}")
            conv_res.add_error(
                component="VisionPipeline._build_document",
                error_message=f"Failed to build document: {str(e)}",
            )
            return conv_res

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """
        Assemble: Process rendered images with VLM to extract annotations.

        Args:
            conv_res: Conversion result to assemble.

        Returns:
            Updated conversion result with extracted annotations.
        """
        if not conv_res.document:
            return conv_res

        try:
            document = conv_res.document

            # Get rendered views from metadata
            rendered_views = document.metadata.get("rendered_views", {})

            if not rendered_views:
                _log.warning("No rendered views found in document")
                return conv_res

            # Get VLM model
            vlm_model = self.vlm_options.vlm_model if self.vlm_options else None

            if not vlm_model:
                _log.warning("No VLM model configured, using geometry-based annotation fallback")
                return self._generate_geometry_based_annotations(conv_res)

            # Process views with VLM using shared utility
            custom_prompt = self.vlm_options.prompt if self.vlm_options else None
            total = process_views_with_vlm(
                rendered_views=rendered_views,
                vlm_model=vlm_model,
                document=document,
                custom_prompt=custom_prompt,
                include_extended_types=False,
                component_name="VisionPipeline",
                conv_res=conv_res,
            )

            _log.info(f"Assembly completed: {len(document.items)} total annotations")

            return conv_res

        except Exception as e:
            _log.exception(f"Assembly stage failed: {e}")
            conv_res.add_error(
                component="VisionPipeline._assemble_document",
                error_message=f"Assembly failed: {str(e)}",
            )
            return conv_res

    def _create_vlm_prompt(self, view_name: str) -> str:
        """Create prompt for VLM based on view name."""
        custom_prompt = self.vlm_options.prompt if self.vlm_options else None
        return create_vlm_prompt(view_name, custom_prompt=custom_prompt)

    def _parse_vlm_response(
        self, response: Any, view_name: str
    ) -> List[CADItem]:
        """Parse VLM response and create CAD items."""
        return parse_vlm_response(response, view_name)

    def _generate_geometry_based_annotations(
        self, conv_res: ConversionResult
    ) -> ConversionResult:
        """Generate annotations from geometry data when VLM unavailable.

        Uses bounding boxes, surface types, and feature detection to create
        basic annotations without vision model.

        Args:
            conv_res: Conversion result with document

        Returns:
            Updated conversion result with geometry-based annotations
        """
        if not conv_res.document:
            return conv_res

        annotations_added = generate_geometry_annotations(
            document=conv_res.document,
            source_view="geometry_analysis",
        )

        _log.info(
            f"Geometry-based annotation generated {annotations_added} annotations"
        )

        return conv_res
