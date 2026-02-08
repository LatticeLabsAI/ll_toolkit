"""PMI (Product Manufacturing Information) extraction model.

This module provides a VLM-based enrichment model for extracting Product
Manufacturing Information from rendered CAD views, including dimensions,
tolerances, GD&T symbols, surface finish, and material callouts.

Classes:
    PMIExtractionModel: Enrichment model for PMI extraction.

Example:
    from cadling.experimental.models import PMIExtractionModel
    from cadling.experimental.datamodel import CADAnnotationOptions

    options = CADAnnotationOptions(
        annotation_types=["dimension", "tolerance", "gdt"],
        vlm_model="gpt-4-vision"
    )
    model = PMIExtractionModel(options)
    model(doc, item_batch)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from cadling.models.base_model import EnrichmentModel
from cadling.models.vlm_model import (
    ApiVlmModel,
    ApiVlmOptions,
    InlineVlmModel,
    InlineVlmOptions,
    VlmAnnotation,
    VlmModel,
)

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument
    from cadling.experimental.datamodel import CADAnnotationOptions

_log = logging.getLogger(__name__)


class PMIExtractionModel(EnrichmentModel):
    """Enrichment model for extracting PMI using VLM.

    This model extracts Product Manufacturing Information from rendered CAD
    views using vision-language models. It can extract:
    - Dimensions and measurements
    - Tolerances (± and limit)
    - GD&T (Geometric Dimensioning & Tolerancing) symbols
    - Surface finish annotations (Ra, Rz, etc.)
    - Material callouts and specifications
    - Welding symbols and requirements

    The model processes rendered views of the CAD model and uses VLM to
    identify and extract textual annotations with high accuracy.

    Attributes:
        options: Configuration options for PMI extraction
        vlm: Vision-language model for annotation extraction
        pmi_prompts: Specialized prompts for each annotation type

    Example:
        options = CADAnnotationOptions(
            annotation_types=["dimension", "tolerance", "gdt"],
            vlm_model="gpt-4-vision",
            min_confidence=0.8
        )
        model = PMIExtractionModel(options)
        model(doc, [item])

        # Access extracted PMI
        pmi_annotations = item.properties.get("pmi_annotations", [])
        for ann in pmi_annotations:
            print(f"{ann['type']}: {ann['text']}")
    """

    def __init__(self, options: CADAnnotationOptions):
        """Initialize PMI extraction model.

        Args:
            options: Configuration options for PMI extraction

        Raises:
            ValueError: If VLM model not supported
        """
        super().__init__()
        self.options = options
        self.vlm = self._initialize_vlm()
        self.pmi_prompts = self._build_pmi_prompts()

        _log.info(
            f"Initialized PMIExtractionModel with {self.options.vlm_model} "
            f"for types: {self.options.annotation_types}"
        )

    def _initialize_vlm(self) -> VlmModel:
        """Initialize the VLM based on model name.

        Returns:
            VlmModel instance

        Raises:
            ValueError: If model not recognized
        """
        model_lower = self.options.vlm_model.lower()

        # API-based models
        if any(
            name in model_lower for name in ["gpt", "claude", "vision", "opus", "sonnet"]
        ):
            # Get API key from environment if not in options
            import os

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if "claude" in model_lower:
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")

            vlm_options = ApiVlmOptions(
                api_key=api_key,
                model_name=self.options.vlm_model,
                temperature=0.0,  # Deterministic for extraction
                max_tokens=4096,
                use_ocr=True,  # Enhance with OCR
            )
            return ApiVlmModel(vlm_options)

        # Local models
        else:
            vlm_options = InlineVlmOptions(
                model_path=self.options.vlm_model,
                device="cuda" if self.requires_gpu() else "cpu",
                temperature=0.0,
                max_tokens=2048,
                use_ocr=True,
            )
            return InlineVlmModel(vlm_options)

    def _build_pmi_prompts(self) -> Dict[str, str]:
        """Build specialized prompts for each PMI annotation type.

        Returns:
            Dictionary mapping annotation type to specialized prompt
        """
        prompts = {
            "dimension": """
You are analyzing a CAD technical drawing. Extract all dimensional annotations including:
- Linear dimensions (length, width, height) with units
- Angular dimensions in degrees
- Radius (R) and diameter (Ø) callouts
- Arc lengths and chord dimensions

For each dimension found, extract:
- The numeric value
- The unit (mm, in, deg, etc.)
- The full text as shown
- Location if visible
""",
            "tolerance": """
You are analyzing a CAD technical drawing. Extract all tolerance specifications including:
- Plus-minus tolerances (e.g., ±0.1, +0.2/-0.1)
- Limit tolerances (e.g., 10.0/9.8)
- Fit callouts (e.g., H7/g6)
- General tolerances referenced

For each tolerance, extract the complete specification text and associated dimension.
""",
            "gdt": """
You are analyzing a CAD technical drawing. Extract all GD&T (Geometric Dimensioning & Tolerancing) symbols:
- Feature control frames with symbols
- Datum feature symbols (A, B, C, etc.)
- Material condition modifiers (M, L, S)
- Geometric characteristics (flatness, perpendicularity, position, etc.)

Extract the complete GD&T callout including all modifiers and datum references.
""",
            "surface_finish": """
You are analyzing a CAD technical drawing. Extract all surface finish annotations:
- Ra (average roughness) values
- Rz (peak-to-valley) values
- Surface texture symbols
- Machining method indicators
- Lay direction symbols

Extract the complete surface finish specification with numeric values and units (μm, μin).
""",
            "material": """
You are analyzing a CAD technical drawing. Extract all material callouts:
- Material type specifications (e.g., "6061-T6 ALUMINUM")
- Material standards (e.g., "ASTM A36")
- Heat treatment requirements
- Finish specifications (e.g., "ANODIZE TYPE II")

Extract the complete material specification text.
""",
            "welding": """
You are analyzing a CAD technical drawing. Extract all welding symbols:
- Weld type symbols (fillet, groove, plug, etc.)
- Weld size and length specifications
- All-around and field weld symbols
- Finish symbols (G, M, C, etc.)

Extract the complete welding specification.
""",
            "note": """
You are analyzing a CAD technical drawing. Extract all general notes and callouts:
- Manufacturing notes
- Assembly instructions
- Quality requirements
- Reference standards

Extract the complete note text.
""",
        }

        return prompts

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: List[CADItem],
    ) -> None:
        """Extract PMI from CAD items using VLM.

        Args:
            doc: The CADlingDocument being enriched
            item_batch: List of CADItem objects to process

        Note:
            Extracted PMI is added to item.properties["pmi_annotations"]
            Each annotation has: type, text, value, unit, confidence, view
        """
        _log.info(f"Processing {len(item_batch)} items for PMI extraction")

        for item in item_batch:
            try:
                # Check if item has rendered images
                rendered_images = item.properties.get("rendered_images", {})
                if not rendered_images:
                    _log.debug(
                        f"Item {item.self_ref} has no rendered images, using STEP text extraction"
                    )
                    # Use STEP text-based PMI extraction fallback
                    annotations = self._extract_pmi_from_step_text(doc, item)
                    if annotations:
                        item.properties["pmi_annotations"] = annotations
                        item.add_provenance(
                            component_type="enrichment_model",
                            component_name=self.__class__.__name__,
                            notes="step_text_fallback",
                        )
                    continue

                # Extract PMI from each requested view
                all_annotations = []

                for view_name in self.options.views_to_process:
                    if view_name not in rendered_images:
                        _log.debug(f"View {view_name} not found, skipping")
                        continue

                    image = rendered_images[view_name]

                    # Extract each annotation type
                    for ann_type in self.options.annotation_types:
                        if ann_type not in self.pmi_prompts:
                            _log.warning(f"Unknown annotation type: {ann_type}")
                            continue

                        # Get specialized prompt
                        prompt = self.pmi_prompts[ann_type]

                        # Add geometric context if enabled
                        if self.options.include_geometric_context:
                            geom_context = self._build_geometric_context(doc, item)
                            prompt = f"{prompt}\n\nGeometric Context:\n{geom_context}"

                        # Run VLM prediction
                        response = self.vlm.predict(image, prompt)

                        # Filter by confidence and add view info
                        for vlm_ann in response.annotations:
                            if vlm_ann.confidence >= self.options.min_confidence:
                                all_annotations.append(
                                    {
                                        "type": ann_type,
                                        "text": vlm_ann.text,
                                        "value": vlm_ann.value,
                                        "unit": vlm_ann.unit,
                                        "confidence": vlm_ann.confidence,
                                        "view": view_name,
                                        "bbox": vlm_ann.bbox,
                                    }
                                )

                        _log.debug(
                            f"Extracted {len(response.annotations)} {ann_type} "
                            f"annotations from {view_name}"
                        )

                # Cross-view validation if enabled
                if (
                    self.options.enable_cross_view_validation
                    and len(self.options.views_to_process) > 1
                ):
                    all_annotations = self._validate_cross_view(all_annotations)

                # Add to item properties
                item.properties["pmi_annotations"] = all_annotations
                item.properties["pmi_extraction_model"] = self.__class__.__name__
                item.properties["pmi_extraction_vlm"] = self.options.vlm_model

                # Add provenance
                if hasattr(item, "add_provenance"):
                    item.add_provenance(
                        component_type="enrichment_model",
                        component_name=self.__class__.__name__,
                    )

                _log.info(
                    f"Extracted {len(all_annotations)} PMI annotations for item {item.self_ref}"
                )

            except Exception as e:
                _log.error(f"PMI extraction failed for item {item.self_ref}: {e}")
                item.properties["pmi_annotations"] = []
                item.properties["pmi_extraction_error"] = str(e)

    def _build_geometric_context(
        self, doc: CADlingDocument, item: CADItem
    ) -> str:
        """Build geometric context string from extracted features.

        Args:
            doc: The document
            item: The item being processed

        Returns:
            Formatted geometric context string
        """
        context_parts = []

        # Add bounding box info if available
        if "bounding_box" in item.properties:
            bbox = item.properties["bounding_box"]
            context_parts.append(f"Bounding box: {bbox}")

        # Add topology info if available
        if hasattr(doc, "topology") and doc.topology:
            topo = doc.topology
            context_parts.append(
                f"Topology: {topo.get('num_faces', 0)} faces, "
                f"{topo.get('num_edges', 0)} edges"
            )

        # Add detected features if available
        if "detected_features" in item.properties:
            features = item.properties["detected_features"]
            feature_summary = ", ".join([f"{k}: {len(v)}" for k, v in features.items()])
            context_parts.append(f"Detected features: {feature_summary}")

        # Add physical properties if available
        if "volume" in item.properties:
            context_parts.append(f"Volume: {item.properties['volume']}")
        if "mass" in item.properties:
            context_parts.append(f"Mass: {item.properties['mass']}")

        return "\n".join(context_parts) if context_parts else "No geometric context available"

    def _extract_pmi_from_step_text(
        self, doc: CADlingDocument, item: CADItem
    ) -> List[Dict[str, Any]]:
        """Extract PMI from STEP entity text when rendered images unavailable.

        Parses GEOMETRIC_TOLERANCE, DATUM_FEATURE, DIMENSION_CALLOUT and
        other PMI-related STEP entities.

        Args:
            doc: The document
            item: The item to analyze

        Returns:
            List of PMI annotation dictionaries
        """
        import re

        annotations = []

        # Get STEP entity text from various sources
        step_text = ""
        if hasattr(item, "raw_text"):
            step_text = item.raw_text
        elif "step_entity_text" in item.properties:
            step_text = item.properties["step_entity_text"]
        elif hasattr(doc, "_step_content"):
            step_text = doc._step_content

        if not step_text:
            return annotations

        # Pattern for GEOMETRIC_TOLERANCE entities
        # e.g., GEOMETRIC_TOLERANCE('flatness',0.01)
        geom_tolerance_pattern = r"GEOMETRIC_TOLERANCE\s*\(\s*'([^']+)'\s*,\s*([\d.]+)\s*\)"
        for match in re.finditer(geom_tolerance_pattern, step_text, re.IGNORECASE):
            tolerance_type = match.group(1)
            tolerance_value = float(match.group(2))
            annotations.append({
                "type": "geometric_tolerance",
                "text": f"{tolerance_type}: {tolerance_value}",
                "value": tolerance_value,
                "unit": "mm",
                "confidence": 0.7,
                "view": "step_text",
                "subtype": tolerance_type,
            })

        # Pattern for DATUM_FEATURE entities
        # e.g., DATUM_FEATURE('A')
        datum_pattern = r"DATUM_FEATURE\s*\(\s*'([^']+)'\s*\)"
        for match in re.finditer(datum_pattern, step_text, re.IGNORECASE):
            datum_label = match.group(1)
            annotations.append({
                "type": "datum",
                "text": f"Datum {datum_label}",
                "value": datum_label,
                "unit": None,
                "confidence": 0.8,
                "view": "step_text",
                "subtype": "datum_feature",
            })

        # Pattern for dimensional values
        # e.g., LENGTH_MEASURE(25.4) or POSITIVE_LENGTH_MEASURE(10.0)
        dimension_pattern = r"(?:POSITIVE_)?LENGTH_MEASURE\s*\(\s*([\d.]+)\s*\)"
        for match in re.finditer(dimension_pattern, step_text, re.IGNORECASE):
            dim_value = float(match.group(1))
            # Only capture significant dimensions (not very small values)
            if dim_value > 0.1:
                annotations.append({
                    "type": "dimension",
                    "text": f"{dim_value:.2f} mm",
                    "value": dim_value,
                    "unit": "mm",
                    "confidence": 0.6,
                    "view": "step_text",
                    "subtype": "linear",
                })

        # Pattern for angular measures
        # e.g., PLANE_ANGLE_MEASURE(1.5707963) (radians)
        angle_pattern = r"PLANE_ANGLE_MEASURE\s*\(\s*([\d.]+)\s*\)"
        for match in re.finditer(angle_pattern, step_text, re.IGNORECASE):
            import math
            angle_rad = float(match.group(1))
            angle_deg = math.degrees(angle_rad)
            # Filter out common angles that aren't user-specified
            if 0.1 < angle_deg < 179.9:
                annotations.append({
                    "type": "dimension",
                    "text": f"{angle_deg:.1f}°",
                    "value": angle_deg,
                    "unit": "deg",
                    "confidence": 0.55,
                    "view": "step_text",
                    "subtype": "angular",
                })

        # Pattern for surface roughness
        # e.g., SURFACE_ROUGHNESS('Ra', 1.6)
        roughness_pattern = r"SURFACE_ROUGHNESS\s*\(\s*'([^']+)'\s*,\s*([\d.]+)\s*\)"
        for match in re.finditer(roughness_pattern, step_text, re.IGNORECASE):
            roughness_type = match.group(1)
            roughness_value = float(match.group(2))
            annotations.append({
                "type": "surface_finish",
                "text": f"{roughness_type} {roughness_value}",
                "value": roughness_value,
                "unit": "μm",
                "confidence": 0.75,
                "view": "step_text",
                "subtype": roughness_type,
            })

        _log.debug(
            f"Extracted {len(annotations)} PMI annotations from STEP text"
        )

        return annotations

    def _validate_cross_view(
        self, annotations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate annotations across multiple views.

        Args:
            annotations: List of annotations from all views

        Returns:
            Filtered list with validated annotations

        Note:
            Annotations that appear in multiple views with consistent values
            have increased confidence. Conflicting annotations are flagged.
        """
        if not annotations:
            return annotations

        # Group by text content
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for ann in annotations:
            key = ann["text"].strip().lower()
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(ann)

        # Validate and merge
        validated = []
        for key, group in grouped.items():
            if len(group) == 1:
                # Single occurrence - keep as is
                validated.append(group[0])
            else:
                # Multiple occurrences - check consistency
                values = [a.get("value") for a in group if a.get("value") is not None]
                units = [a.get("unit") for a in group if a.get("unit") is not None]

                # Check if values are consistent
                consistent = True
                if values:
                    # Allow small numeric variation (0.1%)
                    avg_value = sum(values) / len(values)
                    for v in values:
                        if abs(v - avg_value) / max(avg_value, 1e-6) > 0.001:
                            consistent = False
                            break

                # Check if units are consistent
                if units and len(set(units)) > 1:
                    consistent = False

                if consistent:
                    # Merge with increased confidence
                    merged = group[0].copy()
                    merged["confidence"] = min(
                        1.0, merged["confidence"] * (1 + 0.1 * (len(group) - 1))
                    )
                    merged["views"] = [a["view"] for a in group]
                    merged["cross_view_validated"] = True
                    validated.append(merged)
                else:
                    # Keep all but flag as conflicting
                    for ann in group:
                        ann_copy = ann.copy()
                        ann_copy["cross_view_conflict"] = True
                        validated.append(ann_copy)

                _log.debug(
                    f"Cross-view validation for '{key}': "
                    f"{len(group)} occurrences, consistent={consistent}"
                )

        return validated

    def supports_batch_processing(self) -> bool:
        """Whether this model supports batch processing.

        Returns:
            True if model can process multiple items in batches
        """
        return False  # VLM calls are sequential per item

    def requires_gpu(self) -> bool:
        """Whether this model requires GPU acceleration.

        Returns:
            True if model requires GPU
        """
        # Only local models might need GPU
        model_lower = self.options.vlm_model.lower()
        return not any(
            name in model_lower for name in ["gpt", "claude", "vision", "opus", "sonnet"]
        )

    def get_model_info(self) -> Dict[str, str]:
        """Get information about this model.

        Returns:
            Dictionary with model metadata
        """
        info = super().get_model_info()
        info.update(
            {
                "vlm_model": self.options.vlm_model,
                "annotation_types": ",".join(self.options.annotation_types),
                "views": ",".join(self.options.views_to_process),
                "min_confidence": str(self.options.min_confidence),
            }
        )
        return info
