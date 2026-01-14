"""Vision-Language Model pipeline for CAD annotation extraction.

This pipeline uses VLMs to extract annotations (dimensions, tolerances, notes)
from rendered CAD images. It supports both text-based parsing and vision-based
analysis for hybrid understanding.

Classes:
    CADVlmPipeline: Vision pipeline for optical CAD recognition
    CADVlmPipelineOptions: Configuration options for VLM pipeline
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from PIL import Image

from cadling.backend.abstract_backend import RenderableCADBackend
from cadling.datamodel.base_models import (
    CADDocumentOrigin,
    CADItemLabel,
    CADlingDocument,
    ConversionResult,
    InputFormat,
)
from cadling.datamodel.stl import AnnotationItem
from cadling.models.vlm_model import ApiVlmModel, InlineVlmModel, VlmModel, VlmOptions
from cadling.pipeline.base_pipeline import BaseCADPipeline

if TYPE_CHECKING:
    from cadling.datamodel.pipeline_options import PipelineOptions

_log = logging.getLogger(__name__)


class CADVlmPipelineOptions:
    """Options for CAD VLM pipeline.

    Attributes:
        vlm_model: VLM model instance (ApiVlmModel or InlineVlmModel)
        resolution: Rendering resolution for each view
        views: List of views to render (or None for all available)
        include_ocr: Whether to enhance with OCR
        prompt_template: Custom prompt template for VLM
    """

    def __init__(
        self,
        vlm_model: Optional[VlmModel] = None,
        resolution: int = 1024,
        views: Optional[List[str]] = None,
        include_ocr: bool = True,
        prompt_template: Optional[str] = None,
    ):
        """Initialize VLM pipeline options.

        Args:
            vlm_model: VLM model instance
            resolution: Rendering resolution
            views: Views to render (None = all)
            include_ocr: Whether to use OCR enhancement
            prompt_template: Custom prompt template
        """
        self.vlm_model = vlm_model
        self.resolution = resolution
        self.views = views
        self.include_ocr = include_ocr
        self.prompt_template = prompt_template or self._default_prompt()

    @staticmethod
    def _default_prompt() -> str:
        """Get default VLM prompt.

        Returns:
            Default prompt template
        """
        return """Analyze this CAD technical drawing and extract all annotations.

Extract:
1. Dimensions with values and units (e.g., "50mm", "2.5 inch")
2. Tolerances and specifications (e.g., "±0.1mm", "H7")
3. Notes and labels (material specs, part numbers, etc.)
4. Surface finish and GD&T symbols

Return as structured JSON.
"""


class CADVlmPipeline(BaseCADPipeline):
    """Vision pipeline for optical CAD recognition.

    This pipeline renders CAD files to images and uses VLMs to extract
    annotations (dimensions, tolerances, notes). It's designed for:
    - Technical drawings with annotations
    - Legacy CAD files without embedded data
    - Scanned/photographed engineering drawings

    The pipeline renders multiple views (front, top, isometric, etc.) and
    runs VLM inference on each view to extract textual annotations.

    Attributes:
        vlm_options: VLM-specific configuration
        vlm_model: Vision-language model instance

    Example:
        from cadling.models.vlm_model import ApiVlmModel, ApiVlmOptions

        # Setup VLM
        vlm = ApiVlmModel(ApiVlmOptions(
            api_key="sk-...",
            model_name="gpt-4-vision-preview"
        ))

        # Create pipeline
        pipeline = CADVlmPipeline(CADVlmPipelineOptions(vlm_model=vlm))

        # Convert STEP file with vision
        result = converter.convert("part.step", pipeline=pipeline)

        # Access extracted annotations
        for item in result.document.items:
            if item.item_type == "annotation":
                print(f"{item.annotation_type}: {item.value}")
    """

    def __init__(self, vlm_options: CADVlmPipelineOptions):
        """Initialize vision pipeline.

        Args:
            vlm_options: VLM pipeline configuration
        """
        # Convert to PipelineOptions for base class
        from cadling.datamodel.pipeline_options import PipelineOptions

        base_options = PipelineOptions(
            do_table_structure=False,
            do_ocr=vlm_options.include_ocr,
        )

        super().__init__(base_options)

        self.vlm_options = vlm_options
        self.vlm_model = vlm_options.vlm_model

        if not self.vlm_model:
            _log.warning(
                "No VLM model provided. Pipeline will only render images "
                "without annotation extraction."
            )

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Build document by rendering views and extracting annotations.

        Args:
            conv_res: Conversion result with backend

        Returns:
            ConversionResult with document containing annotations
        """
        backend = conv_res.input._backend

        # Validate backend supports rendering
        if not isinstance(backend, RenderableCADBackend):
            raise ValueError(
                f"CADVlmPipeline requires RenderableCADBackend, "
                f"got {type(backend).__name__}"
            )

        # Create document
        doc = CADlingDocument(
            name=backend.file.name,
            origin=CADDocumentOrigin(
                filename=backend.file.name,
                format=backend.input_format,
                binary_hash=backend.document_hash,
            ),
            hash=backend.document_hash,
        )

        # Determine which views to render
        available_views = backend.available_views()
        views_to_render = (
            self.vlm_options.views if self.vlm_options.views else available_views
        )

        _log.info(
            f"Rendering {len(views_to_render)} views for VLM processing: "
            f"{', '.join(views_to_render)}"
        )

        # Process each view
        for view_name in views_to_render:
            if view_name not in available_views:
                _log.warning(f"View '{view_name}' not available, skipping")
                continue

            try:
                # Render view
                _log.debug(f"Rendering view '{view_name}' at {self.vlm_options.resolution}px")
                view_image = backend.render_view(
                    view_name, resolution=self.vlm_options.resolution
                )

                # Extract annotations if VLM available
                if self.vlm_model:
                    annotations = self._extract_annotations_from_image(
                        view_image, view_name
                    )

                    _log.info(
                        f"Extracted {len(annotations)} annotations from view '{view_name}'"
                    )

                    # Add annotations to document
                    for ann in annotations:
                        doc.add_item(ann)

            except Exception as e:
                _log.error(f"Failed to process view '{view_name}': {e}")

        conv_res.document = doc

        _log.info(
            f"Vision pipeline complete: {len(doc.items)} annotations extracted "
            f"from {len(views_to_render)} views"
        )

        return conv_res

    def _extract_annotations_from_image(
        self, image: Image.Image, view_name: str
    ) -> List[AnnotationItem]:
        """Extract annotations from rendered CAD image using VLM.

        Args:
            image: Rendered CAD view
            view_name: Name of the view

        Returns:
            List of AnnotationItem objects
        """
        annotations = []

        if not self.vlm_model:
            return annotations

        try:
            # Run VLM prediction
            _log.debug(f"Running VLM on {view_name} view...")
            vlm_response = self.vlm_model.predict(
                image, self.vlm_options.prompt_template
            )

            # Convert VLM annotations to AnnotationItem
            for vlm_ann in vlm_response.annotations:
                # Parse numeric value if present
                numeric_value = None
                if vlm_ann.value is not None:
                    numeric_value = vlm_ann.value

                # Determine unit from text if not provided
                unit = vlm_ann.unit
                if not unit and vlm_ann.text:
                    # Simple unit extraction (mm, inch, etc.)
                    text_lower = vlm_ann.text.lower()
                    if "mm" in text_lower:
                        unit = "mm"
                    elif "inch" in text_lower or '"' in text_lower:
                        unit = "inch"
                    elif "cm" in text_lower:
                        unit = "cm"

                # Create AnnotationItem
                ann = AnnotationItem(
                    label=CADItemLabel(text=vlm_ann.text),
                    annotation_type=vlm_ann.annotation_type,
                    value=vlm_ann.text,
                    image_bbox=vlm_ann.bbox,
                    source_view=view_name,
                    confidence=vlm_ann.confidence,
                    unit=unit,
                    text=vlm_ann.text,
                )

                # Add numeric value to properties if available
                if numeric_value is not None:
                    ann.properties["numeric_value"] = numeric_value

                # Add provenance
                ann.add_provenance(
                    component_type="vlm_pipeline",
                    component_name=self.__class__.__name__,
                )

                annotations.append(ann)

            _log.debug(
                f"Parsed {len(annotations)} annotations from VLM response "
                f"(raw annotations: {len(vlm_response.annotations)})"
            )

        except Exception as e:
            _log.error(f"Annotation extraction failed for {view_name}: {e}")

        return annotations

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Assemble document with de-duplication and conflict resolution.

        For VLM pipeline, assembly includes:
        - Merging duplicate annotations from different views
        - Resolving conflicts between views (keeping highest confidence)
        - Spatial correlation of annotations

        Args:
            conv_res: Conversion result

        Returns:
            Updated conversion result with de-duplicated annotations
        """
        # De-duplicate annotations from different views
        annotations = [
            item for item in conv_res.document.items if item.item_type == "annotation"
        ]

        if not annotations:
            return conv_res

        _log.info(f"De-duplicating {len(annotations)} annotations")

        # Group similar annotations
        annotation_groups = self._group_similar_annotations(annotations)

        # Keep best annotation from each group
        deduplicated = []
        for group in annotation_groups:
            best = max(group, key=lambda a: a.confidence or 0.0)
            deduplicated.append(best)

        _log.info(
            f"De-duplication: {len(annotations)} → {len(deduplicated)} annotations "
            f"({len(annotations) - len(deduplicated)} duplicates removed)"
        )

        # Replace annotations in document
        conv_res.document.items = [
            item for item in conv_res.document.items if item.item_type != "annotation"
        ] + deduplicated

        return conv_res

    def _group_similar_annotations(
        self, annotations: List[AnnotationItem]
    ) -> List[List[AnnotationItem]]:
        """Group annotations that are likely duplicates.

        Two annotations are considered similar if they have:
        - Same annotation type
        - Similar value (normalized text match)
        - Similar position (if from same view, or overlapping 2D bboxes)

        Args:
            annotations: List of annotations to group

        Returns:
            List of annotation groups (each group contains similar annotations)
        """
        groups: List[List[AnnotationItem]] = []

        for annotation in annotations:
            # Try to find an existing group for this annotation
            found_group = False

            for group in groups:
                if self._are_annotations_similar(annotation, group[0]):
                    group.append(annotation)
                    found_group = True
                    break

            # Create new group if no match found
            if not found_group:
                groups.append([annotation])

        return groups

    def _are_annotations_similar(
        self, ann1: AnnotationItem, ann2: AnnotationItem
    ) -> bool:
        """Check if two annotations are similar (likely duplicates).

        Args:
            ann1: First annotation
            ann2: Second annotation

        Returns:
            True if annotations are similar
        """
        # Must have same type
        if ann1.annotation_type != ann2.annotation_type:
            return False

        # Must have similar values (case-insensitive, normalized)
        val1 = self._normalize_annotation_value(ann1.value or "")
        val2 = self._normalize_annotation_value(ann2.value or "")

        if val1 != val2:
            return False

        # If from different views, consider them similar if values match
        # (same physical annotation visible from multiple views)
        if ann1.source_view != ann2.source_view:
            return True

        # If from same view, check spatial proximity
        if ann1.image_bbox and ann2.image_bbox:
            return self._bboxes_overlap(ann1.image_bbox, ann2.image_bbox)

        # Default: consider similar if types and values match
        return True

    def _normalize_annotation_value(self, value: str) -> str:
        """Normalize annotation value for comparison.

        Args:
            value: Raw annotation value

        Returns:
            Normalized value (lowercase, whitespace trimmed, special chars removed)
        """
        # Lowercase and strip whitespace
        normalized = value.lower().strip()

        # Remove common variation characters
        normalized = normalized.replace(" ", "").replace("-", "").replace("_", "")

        return normalized

    def _bboxes_overlap(self, bbox1: dict, bbox2: dict, overlap_threshold: float = 0.5) -> bool:
        """Check if two 2D bounding boxes overlap significantly.

        Args:
            bbox1: First bounding box {"x_min", "x_max", "y_min", "y_max"}
            bbox2: Second bounding box
            overlap_threshold: Minimum IoU (Intersection over Union) to consider overlap

        Returns:
            True if boxes overlap significantly
        """
        # Calculate intersection
        x_min = max(bbox1.get("x_min", 0), bbox2.get("x_min", 0))
        y_min = max(bbox1.get("y_min", 0), bbox2.get("y_min", 0))
        x_max = min(bbox1.get("x_max", 0), bbox2.get("x_max", 0))
        y_max = min(bbox1.get("y_max", 0), bbox2.get("y_max", 0))

        # Check if there's actual intersection
        if x_max <= x_min or y_max <= y_min:
            return False

        intersection_area = (x_max - x_min) * (y_max - y_min)

        # Calculate areas
        bbox1_area = (bbox1.get("x_max", 0) - bbox1.get("x_min", 0)) * (
            bbox1.get("y_max", 0) - bbox1.get("y_min", 0)
        )
        bbox2_area = (bbox2.get("x_max", 0) - bbox2.get("x_min", 0)) * (
            bbox2.get("y_max", 0) - bbox2.get("y_min", 0)
        )

        # Calculate IoU (Intersection over Union)
        union_area = bbox1_area + bbox2_area - intersection_area

        if union_area == 0:
            return False

        iou = intersection_area / union_area

        return iou >= overlap_threshold

    def _enrich_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Enrich document with additional models.

        Args:
            conv_res: Conversion result

        Returns:
            Enriched conversion result
        """
        # Apply standard enrichment models
        for model in self.enrichment_pipe:
            try:
                model(conv_res.document, conv_res.document.items)
            except Exception as e:
                _log.error(f"Enrichment model {model.__class__.__name__} failed: {e}")

        return conv_res
