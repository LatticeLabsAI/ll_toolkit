"""
Vision pipeline for optical CAD recognition using VLMs.

This pipeline processes rendered CAD images to extract annotations,
dimensions, tolerances, and other visual information using Vision-Language Models.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from cadling.pipeline.base_pipeline import BaseCADPipeline
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

            # Render views
            _log.info(f"Rendering {len(self.views_to_render)} views")

            rendered_views = {}
            for view_name in self.views_to_render:
                try:
                    _log.debug(f"Rendering view: {view_name}")
                    image = backend.render_view(
                        view_name,
                        resolution=self.vlm_options.image_resolution
                        if self.vlm_options
                        else 1024,
                    )
                    rendered_views[view_name] = image

                    _log.info(
                        f"Rendered {view_name} view: {image.size[0]}x{image.size[1]}"
                    )

                except Exception as e:
                    _log.error(f"Failed to render view {view_name}: {e}")
                    conv_res.add_error(
                        component="VisionPipeline._build_document",
                        error_message=f"Failed to render view {view_name}: {str(e)}",
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
            _log.exception(f"Build stage failed: {e}")
            conv_res.add_error(
                component="VisionPipeline._build_document",
                error_message=f"Failed to build document: {str(e)}",
            )
            raise

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
                _log.warning("No VLM model configured, skipping annotation extraction")
                return conv_res

            _log.info(
                f"Processing {len(rendered_views)} views with VLM: "
                f"{vlm_model.__class__.__name__}"
            )

            # Process each view
            for view_name, image in rendered_views.items():
                try:
                    _log.debug(f"Processing view: {view_name}")

                    # Create prompt for VLM
                    prompt = self._create_vlm_prompt(view_name)

                    # Process image with VLM
                    response = vlm_model.process_image(image, prompt)

                    # Extract annotations from response
                    annotations = self._parse_vlm_response(response, view_name)

                    # Add annotations as items to document
                    for annotation in annotations:
                        document.add_item(annotation)

                    _log.info(
                        f"Extracted {len(annotations)} annotations from {view_name}"
                    )

                except Exception as e:
                    _log.error(
                        f"Failed to process view {view_name} with VLM: {e}",
                        exc_info=True,
                    )
                    conv_res.add_error(
                        component="VisionPipeline._assemble_document",
                        error_message=f"Failed to process view {view_name}: {str(e)}",
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
        # Use custom prompt if provided
        if self.vlm_options and self.vlm_options.prompt:
            return self.vlm_options.prompt

        # Default prompt for CAD annotation extraction
        return f"""
You are analyzing a {view_name} view of a CAD model. Please extract all visible annotations including:

1. **Dimensions**: Linear, angular, radial dimensions with their values and units
2. **Tolerances**: Geometric dimensioning and tolerancing (GD&T) symbols and values
3. **Notes**: Text notes, callouts, and labels
4. **Part Numbers**: Part identification numbers or labels

For each annotation, provide:
- Type (dimension, tolerance, note, label)
- Text content
- Numeric value (if applicable)
- Unit (if applicable)
- Approximate location in image

Format your response as a JSON array of annotations.
""".strip()

    def _parse_vlm_response(
        self, response: Any, view_name: str
    ) -> List[CADItem]:
        """Parse VLM response and create CAD items."""
        from cadling.datamodel.stl import AnnotationItem

        annotations = []

        # Handle VlmResponse object
        if hasattr(response, "annotations"):
            for vlm_annotation in response.annotations:
                # Create AnnotationItem from VlmAnnotation
                annotation_item = AnnotationItem(
                    label={"text": vlm_annotation.text},
                    text=vlm_annotation.text,
                    annotation_type=vlm_annotation.annotation_type,
                    value=str(vlm_annotation.value) if vlm_annotation.value else None,
                    source_view=view_name,
                    image_bbox=vlm_annotation.bbox,
                )

                annotations.append(annotation_item)

        # Handle raw text response
        elif isinstance(response, str):
            # Try to parse as JSON
            import json

            try:
                parsed = json.loads(response)
                if isinstance(parsed, list):
                    for item in parsed:
                        annotation_item = AnnotationItem(
                            label={"text": item.get("text", "")},
                            text=item.get("text", ""),
                            annotation_type=item.get("type", "note"),
                            value=item.get("value"),
                            source_view=view_name,
                        )
                        annotations.append(annotation_item)
            except json.JSONDecodeError:
                # Treat as single text note
                annotation_item = AnnotationItem(
                    label={"text": "VLM Note"},
                    text=response,
                    annotation_type="note",
                    source_view=view_name,
                )
                annotations.append(annotation_item)

        return annotations
