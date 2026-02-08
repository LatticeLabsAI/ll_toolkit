"""
Hybrid pipeline combining text parsing with vision analysis.

This pipeline processes CAD files using both text-based parsing
(STEP entities, STL mesh) and vision-based annotation extraction (VLM).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cadling.pipeline.base_pipeline import BaseCADPipeline

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

            # Phase 2: Image rendering
            _log.info("Phase 2: Image rendering")

            # Check if backend supports rendering
            from cadling.backend.abstract_backend import RenderableCADBackend

            if isinstance(backend, RenderableCADBackend):
                _log.debug(f"Rendering {len(self.views_to_render)} views")

                rendered_views = {}
                for view_name in self.views_to_render:
                    try:
                        resolution = (
                            self.vlm_options.image_resolution
                            if self.vlm_options
                            else 1024
                        )
                        image = backend.render_view(view_name, resolution=resolution)
                        rendered_views[view_name] = image

                        _log.info(
                            f"Rendered {view_name} view: {image.size[0]}x{image.size[1]}"
                        )

                    except Exception as e:
                        _log.error(f"Failed to render view {view_name}: {e}")
                        conv_res.add_error(
                            component="HybridPipeline._build_document",
                            error_message=f"Failed to render view {view_name}: {str(e)}",
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

            _log.info(
                f"Processing {len(rendered_views)} views with VLM: "
                f"{vlm_model.__class__.__name__}"
            )

            # Store initial item count (from text parsing)
            initial_item_count = len(document.items)

            # Process each view with VLM
            for view_name, image in rendered_views.items():
                try:
                    _log.debug(f"Processing view: {view_name}")

                    # Create prompt
                    prompt = self._create_vlm_prompt(view_name)

                    # Process with VLM
                    response = vlm_model.process_image(image, prompt)

                    # Parse and add annotations
                    annotations = self._parse_vlm_response(response, view_name)

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
                        component="HybridPipeline._assemble_document",
                        error_message=f"Failed to process view {view_name}: {str(e)}",
                    )

            # Log summary
            vision_items = len(document.items) - initial_item_count
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
5. **Surface Finishes**: Surface finish symbols and roughness values
6. **Welding Symbols**: Welding and joining symbols

For each annotation, provide:
- Type (dimension, tolerance, note, label, surface_finish, welding)
- Text content
- Numeric value (if applicable)
- Unit (if applicable)
- Approximate location in image

Format your response as a JSON array of annotations.
""".strip()

    def _parse_vlm_response(self, response: Any, view_name: str) -> list:
        """Parse VLM response and create CAD items."""
        from cadling.datamodel.stl import AnnotationItem

        annotations = []

        # Handle VlmResponse object
        if hasattr(response, "annotations"):
            for vlm_annotation in response.annotations:
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

    def _text_only_analysis(self, conv_res: ConversionResult) -> ConversionResult:
        """Fallback text-only analysis when VLM/rendered views unavailable.

        Extracts annotations from STEP entity text, geometry analysis results,
        and other text-based sources.

        Args:
            conv_res: Conversion result with document

        Returns:
            Updated conversion result with text-based annotations
        """
        from cadling.datamodel.stl import AnnotationItem

        if not conv_res.document:
            return conv_res

        document = conv_res.document
        annotations_added = 0

        for item in document.items:
            # Extract from geometry analysis
            geometry_analysis = item.properties.get("geometry_analysis", {})
            if geometry_analysis:
                # Bounding box annotation
                bbox = geometry_analysis.get("bounding_box", {})
                if bbox:
                    size_text = (
                        f"{bbox.get('size_x', 0):.2f} x "
                        f"{bbox.get('size_y', 0):.2f} x "
                        f"{bbox.get('size_z', 0):.2f}"
                    )
                    annotation = AnnotationItem(
                        label={"text": f"Bounding Box"},
                        text=f"Dimensions: {size_text}",
                        annotation_type="dimension",
                        value=size_text,
                        source_view="text_analysis",
                    )
                    document.add_item(annotation)
                    annotations_added += 1

                # Volume annotation
                volume = geometry_analysis.get("volume")
                if volume and volume > 0:
                    annotation = AnnotationItem(
                        label={"text": "Volume"},
                        text=f"Volume: {volume:.4f}",
                        annotation_type="dimension",
                        value=str(volume),
                        source_view="text_analysis",
                    )
                    document.add_item(annotation)
                    annotations_added += 1

            # Extract from surface analysis
            surface_analysis = item.properties.get("surface_analysis", {})
            if surface_analysis:
                surface_type = surface_analysis.get("surface_type")
                if surface_type and surface_type != "UNKNOWN":
                    annotation = AnnotationItem(
                        label={"text": "Surface Type"},
                        text=f"Surface: {surface_type}",
                        annotation_type="note",
                        value=surface_type,
                        source_view="text_analysis",
                    )
                    document.add_item(annotation)
                    annotations_added += 1

            # Extract from manufacturing features
            mfg_features = item.properties.get("manufacturing_features", [])
            for feature in mfg_features:
                feature_type = feature.get("type", "unknown")
                params = feature.get("parameters", {})
                params_text = ", ".join(f"{k}={v}" for k, v in params.items())
                annotation = AnnotationItem(
                    label={"text": f"Feature: {feature_type}"},
                    text=f"{feature_type}: {params_text}",
                    annotation_type="note",
                    value=params_text,
                    source_view="text_analysis",
                )
                document.add_item(annotation)
                annotations_added += 1

        _log.info(
            f"Text-only analysis added {annotations_added} annotations "
            f"from geometry/surface/feature data"
        )

        return conv_res
