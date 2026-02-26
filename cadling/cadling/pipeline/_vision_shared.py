"""Shared utilities for vision-based CAD pipelines.

This module extracts common logic used by VisionPipeline, HybridPipeline,
and CADVlmPipeline to eliminate duplication. It provides:

- View rendering with error handling
- VLM prompt creation
- VLM response parsing into AnnotationItems
- VLM-based annotation extraction from views
- Geometry-based annotation fallback
- Annotation de-duplication
- Closest view matching

Classes:
    VisionPipelineMixin: Mixin providing shared vision pipeline methods.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from PIL import Image

    from cadling.datamodel.base_models import CADItem, ConversionResult
    from cadling.datamodel.stl import AnnotationItem

_log = logging.getLogger(__name__)


def render_views(
    backend: Any,
    views_to_render: List[str],
    resolution: int = 1024,
    component_name: str = "VisionPipeline",
    conv_res: Optional[ConversionResult] = None,
) -> Dict[str, Any]:
    """Render multiple views from a CAD backend.

    Args:
        backend: A RenderableCADBackend instance.
        views_to_render: List of view names to render.
        resolution: Image resolution in pixels.
        component_name: Name of the calling component for error reporting.
        conv_res: Optional ConversionResult for error logging.

    Returns:
        Dictionary mapping view names to rendered PIL Images.
    """
    rendered_views: Dict[str, Any] = {}

    for view_name in views_to_render:
        try:
            _log.debug(f"Rendering view: {view_name}")
            image = backend.render_view(view_name, resolution=resolution)
            rendered_views[view_name] = image

            _log.info(
                f"Rendered {view_name} view: {image.size[0]}x{image.size[1]}"
            )

        except Exception as e:
            _log.error(f"Failed to render view {view_name}: {e}")
            if conv_res is not None:
                conv_res.add_error(
                    component=f"{component_name}._build_document",
                    error_message=f"Failed to render view {view_name}: {str(e)}",
                )

    return rendered_views


def create_vlm_prompt(
    view_name: str,
    custom_prompt: Optional[str] = None,
    include_extended_types: bool = False,
) -> str:
    """Create a VLM prompt for CAD annotation extraction.

    Args:
        view_name: Name of the view being analyzed.
        custom_prompt: Optional custom prompt to use instead of default.
        include_extended_types: If True, include surface finish and welding
            symbol extraction (used by HybridPipeline).

    Returns:
        Prompt string for VLM processing.
    """
    if custom_prompt:
        return custom_prompt

    extended_section = ""
    extended_types = ""
    if include_extended_types:
        extended_section = (
            "5. **Surface Finishes**: Surface finish symbols and roughness values\n"
            "6. **Welding Symbols**: Welding and joining symbols\n"
        )
        extended_types = ", surface_finish, welding"

    return f"""
You are analyzing a {view_name} view of a CAD model. Please extract all visible annotations including:

1. **Dimensions**: Linear, angular, radial dimensions with their values and units
2. **Tolerances**: Geometric dimensioning and tolerancing (GD&T) symbols and values
3. **Notes**: Text notes, callouts, and labels
4. **Part Numbers**: Part identification numbers or labels
{extended_section}
For each annotation, provide:
- Type (dimension, tolerance, note, label{extended_types})
- Text content
- Numeric value (if applicable)
- Unit (if applicable)
- Approximate location in image

Format your response as a JSON array of annotations.
""".strip()


def parse_vlm_response(
    response: Any, view_name: str
) -> List[AnnotationItem]:
    """Parse a VLM response and create AnnotationItem objects.

    Handles both structured VlmResponse objects (with .annotations attribute)
    and raw string responses (JSON or plain text).

    Args:
        response: VLM response, either a VlmResponse object or a string.
        view_name: Name of the source view.

    Returns:
        List of AnnotationItem objects parsed from the response.
    """
    from cadling.datamodel.stl import AnnotationItem

    annotations: List[AnnotationItem] = []

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


def process_views_with_vlm(
    rendered_views: Dict[str, Any],
    vlm_model: Any,
    document: Any,
    custom_prompt: Optional[str] = None,
    include_extended_types: bool = False,
    component_name: str = "VisionPipeline",
    conv_res: Optional[ConversionResult] = None,
) -> int:
    """Process rendered views with a VLM and add annotations to a document.

    Args:
        rendered_views: Dictionary of view_name -> PIL Image.
        vlm_model: VLM model with a process_image(image, prompt) method.
        document: CADlingDocument to add annotations to.
        custom_prompt: Optional custom VLM prompt.
        include_extended_types: Whether to include extended annotation types in prompt.
        component_name: Name of calling component for error reporting.
        conv_res: Optional ConversionResult for error logging.

    Returns:
        Number of annotations extracted.
    """
    total_annotations = 0

    _log.info(
        f"Processing {len(rendered_views)} views with VLM: "
        f"{vlm_model.__class__.__name__}"
    )

    for view_name, image in rendered_views.items():
        try:
            _log.debug(f"Processing view: {view_name}")

            # Create prompt for VLM
            prompt = create_vlm_prompt(
                view_name,
                custom_prompt=custom_prompt,
                include_extended_types=include_extended_types,
            )

            # Process image with VLM
            response = vlm_model.process_image(image, prompt)

            # Extract annotations from response
            annotations = parse_vlm_response(response, view_name)

            # Add annotations as items to document
            for annotation in annotations:
                document.add_item(annotation)

            total_annotations += len(annotations)

            _log.info(
                f"Extracted {len(annotations)} annotations from {view_name}"
            )

        except Exception as e:
            _log.error(
                f"Failed to process view {view_name} with VLM: {e}",
                exc_info=True,
            )
            if conv_res is not None:
                conv_res.add_error(
                    component=f"{component_name}._assemble_document",
                    error_message=f"Failed to process view {view_name}: {str(e)}",
                )

    return total_annotations


def generate_geometry_annotations(
    document: Any,
    source_view: str = "geometry_analysis",
) -> int:
    """Generate annotations from geometry data when VLM is unavailable.

    Extracts bounding box dimensions, surface types, and manufacturing
    features from item properties and creates AnnotationItems.

    Args:
        document: CADlingDocument with items containing geometry properties.
        source_view: Source view label for generated annotations.

    Returns:
        Number of annotations added.
    """
    from cadling.datamodel.stl import AnnotationItem

    annotations_added = 0

    for item in document.items:
        # Extract bounding box annotations
        geometry_analysis = item.properties.get("geometry_analysis", {})
        bbox = geometry_analysis.get("bounding_box", {})

        if bbox:
            dims = {
                "width": bbox.get("size_x", 0),
                "height": bbox.get("size_y", 0),
                "depth": bbox.get("size_z", 0),
            }

            for dim_name, dim_value in dims.items():
                if dim_value and dim_value > 0:
                    annotation = AnnotationItem(
                        label={"text": f"{dim_name.capitalize()}"},
                        text=f"{dim_name.capitalize()}: {dim_value:.2f}",
                        annotation_type="dimension",
                        value=str(dim_value),
                        source_view=source_view,
                    )
                    document.add_item(annotation)
                    annotations_added += 1

        # Volume annotation (for text_analysis source)
        if source_view == "text_analysis":
            volume = geometry_analysis.get("volume")
            if volume and volume > 0:
                annotation = AnnotationItem(
                    label={"text": "Volume"},
                    text=f"Volume: {volume:.4f}",
                    annotation_type="dimension",
                    value=str(volume),
                    source_view=source_view,
                )
                document.add_item(annotation)
                annotations_added += 1

        # Extract surface type annotations
        surface_analysis = item.properties.get("surface_analysis", {})
        surface_type = surface_analysis.get("surface_type")

        if surface_type and surface_type != "UNKNOWN":
            label_text = "Surface Type" if source_view == "text_analysis" else "Surface"
            text = (
                f"Surface: {surface_type}"
                if source_view == "text_analysis"
                else f"Surface type: {surface_type}"
            )
            annotation = AnnotationItem(
                label={"text": label_text},
                text=text,
                annotation_type="note",
                value=surface_type,
                source_view=source_view,
            )
            document.add_item(annotation)
            annotations_added += 1

        # Extract manufacturing feature annotations
        mfg_features = item.properties.get("manufacturing_features", [])
        for feature in mfg_features:
            feature_type = feature.get("type", "unknown")
            params = feature.get("parameters", {})

            if source_view == "text_analysis":
                # HybridPipeline style: key=value format
                params_text = ", ".join(f"{k}={v}" for k, v in params.items())
                label_text = f"Feature: {feature_type}"
                text = f"{feature_type}: {params_text}"
            else:
                # VisionPipeline style: formatted parameters
                param_parts = []
                if "diameter" in params:
                    param_parts.append(f"\u00d8{params['diameter']:.2f}")
                if "depth" in params:
                    param_parts.append(f"depth: {params['depth']:.2f}")
                if "radius" in params:
                    param_parts.append(f"R{params['radius']:.2f}")
                params_text = ", ".join(param_parts) if param_parts else ""
                label_text = feature_type
                text = f"{feature_type.replace('_', ' ').title()}"
                if params_text:
                    text += f": {params_text}"

            annotation = AnnotationItem(
                label={"text": label_text},
                text=text,
                annotation_type="note",
                value=params_text,
                source_view=source_view,
            )
            document.add_item(annotation)
            annotations_added += 1

    return annotations_added


def find_closest_view(
    requested_view: str, available_views: List[str]
) -> Optional[str]:
    """Find the closest available view when the requested view is unavailable.

    Uses view naming conventions to find similar views:
    - "front_left" -> "front" or "left"
    - "top_iso" -> "top" or "iso"
    - "bottom_right" -> "bottom" or "right"

    Args:
        requested_view: Name of the requested view.
        available_views: List of available view names.

    Returns:
        Name of closest available view, or None if no match found.
    """
    if not available_views:
        return None

    requested_lower = requested_view.lower()

    # Define view relationships (ordered by preference)
    view_relationships = {
        "front_left": ["front", "left", "iso", "front_right"],
        "front_right": ["front", "right", "iso", "front_left"],
        "back_left": ["back", "left", "rear", "back_right"],
        "back_right": ["back", "right", "rear", "back_left"],
        "top_front": ["top", "front", "iso"],
        "top_back": ["top", "back", "rear"],
        "bottom_front": ["bottom", "front"],
        "bottom_back": ["bottom", "back"],
        "top_iso": ["top", "iso", "isometric"],
        "front_iso": ["front", "iso", "isometric"],
        "iso": ["isometric", "front", "top"],
        "isometric": ["iso", "front", "top"],
        "rear": ["back", "back_left", "back_right"],
    }

    # Check direct relationships
    if requested_lower in view_relationships:
        for candidate in view_relationships[requested_lower]:
            for available in available_views:
                if candidate in available.lower():
                    return available

    # Try splitting compound view name and matching components
    components = requested_lower.replace("_", " ").replace("-", " ").split()
    for component in components:
        for available in available_views:
            if component in available.lower():
                _log.debug(
                    f"Matched component '{component}' from '{requested_view}' "
                    f"to available view '{available}'"
                )
                return available

    # Last resort: return first available view
    if available_views:
        _log.debug(
            f"No close match for '{requested_view}', using first available: "
            f"'{available_views[0]}'"
        )
        return available_views[0]

    return None


def group_similar_annotations(
    annotations: List[AnnotationItem],
) -> List[List[AnnotationItem]]:
    """Group annotations that are likely duplicates.

    Two annotations are considered similar if they have:
    - Same annotation type
    - Similar value (normalized text match)
    - Similar position (if from same view, or overlapping 2D bboxes)

    Args:
        annotations: List of annotations to group.

    Returns:
        List of annotation groups (each group contains similar annotations).
    """
    groups: List[List[AnnotationItem]] = []

    for annotation in annotations:
        found_group = False

        for group in groups:
            if _are_annotations_similar(annotation, group[0]):
                group.append(annotation)
                found_group = True
                break

        if not found_group:
            groups.append([annotation])

    return groups


def deduplicate_annotations(
    annotations: List[AnnotationItem],
) -> List[AnnotationItem]:
    """De-duplicate annotations by grouping similar ones and keeping the best.

    Args:
        annotations: List of annotations to de-duplicate.

    Returns:
        De-duplicated list of annotations.
    """
    if not annotations:
        return annotations

    _log.info(f"De-duplicating {len(annotations)} annotations")

    annotation_groups = group_similar_annotations(annotations)

    deduplicated = []
    for group in annotation_groups:
        best = max(group, key=lambda a: a.confidence or 0.0)
        deduplicated.append(best)

    _log.info(
        f"De-duplication: {len(annotations)} -> {len(deduplicated)} annotations "
        f"({len(annotations) - len(deduplicated)} duplicates removed)"
    )

    return deduplicated


def _are_annotations_similar(
    ann1: AnnotationItem, ann2: AnnotationItem
) -> bool:
    """Check if two annotations are similar (likely duplicates).

    Args:
        ann1: First annotation.
        ann2: Second annotation.

    Returns:
        True if annotations are similar.
    """
    # Must have same type
    if ann1.annotation_type != ann2.annotation_type:
        return False

    # Must have similar values (case-insensitive, normalized)
    val1 = _normalize_annotation_value(ann1.value or "")
    val2 = _normalize_annotation_value(ann2.value or "")

    if val1 != val2:
        return False

    # If from different views, consider them similar if values match
    if ann1.source_view != ann2.source_view:
        return True

    # If from same view, check spatial proximity
    if ann1.image_bbox and ann2.image_bbox:
        return bboxes_overlap(ann1.image_bbox, ann2.image_bbox)

    # Default: consider similar if types and values match
    return True


def _normalize_annotation_value(value: str) -> str:
    """Normalize annotation value for comparison.

    Args:
        value: Raw annotation value.

    Returns:
        Normalized value (lowercase, whitespace trimmed, special chars removed).
    """
    normalized = value.lower().strip()
    normalized = normalized.replace(" ", "").replace("-", "").replace("_", "")
    return normalized


def bboxes_overlap(
    bbox1: dict, bbox2: dict, overlap_threshold: float = 0.5
) -> bool:
    """Check if two 2D bounding boxes overlap significantly.

    Args:
        bbox1: First bounding box {"x_min", "x_max", "y_min", "y_max"}.
        bbox2: Second bounding box.
        overlap_threshold: Minimum IoU to consider overlap.

    Returns:
        True if boxes overlap significantly.
    """
    # Calculate intersection
    x_min = max(bbox1.get("x_min", 0), bbox2.get("x_min", 0))
    y_min = max(bbox1.get("y_min", 0), bbox2.get("y_min", 0))
    x_max = min(bbox1.get("x_max", 0), bbox2.get("x_max", 0))
    y_max = min(bbox1.get("y_max", 0), bbox2.get("y_max", 0))

    if x_max <= x_min or y_max <= y_min:
        return False

    intersection_area = (x_max - x_min) * (y_max - y_min)

    bbox1_area = (bbox1.get("x_max", 0) - bbox1.get("x_min", 0)) * (
        bbox1.get("y_max", 0) - bbox1.get("y_min", 0)
    )
    bbox2_area = (bbox2.get("x_max", 0) - bbox2.get("x_min", 0)) * (
        bbox2.get("y_max", 0) - bbox2.get("y_min", 0)
    )

    union_area = bbox1_area + bbox2_area - intersection_area

    if union_area == 0:
        return False

    iou = intersection_area / union_area

    return iou >= overlap_threshold
