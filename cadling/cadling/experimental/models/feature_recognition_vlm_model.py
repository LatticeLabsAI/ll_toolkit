"""VLM-based machining feature recognition model.

This module provides a VLM-based enrichment model for identifying machining
features from rendered CAD views, including holes, pockets, slots, fillets,
chamfers, threads, bosses, and ribs.

Classes:
    FeatureRecognitionVlmModel: Enrichment model for feature recognition.
    MachiningFeature: Data structure for recognized features.

Example:
    from cadling.experimental.models import FeatureRecognitionVlmModel
    from cadling.experimental.datamodel import CADAnnotationOptions

    options = CADAnnotationOptions(
        annotation_types=["hole", "pocket", "fillet"],
        vlm_model="gpt-4-vision"
    )
    model = FeatureRecognitionVlmModel(options)
    model(doc, item_batch)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from cadling.models.base_model import EnrichmentModel
from cadling.models.vlm_model import (
    ApiVlmModel,
    ApiVlmOptions,
    InlineVlmModel,
    InlineVlmOptions,
    VlmModel,
)

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument
    from cadling.experimental.datamodel import CADAnnotationOptions

_log = logging.getLogger(__name__)


class MachiningFeature(BaseModel):
    """Data structure for a recognized machining feature.

    Attributes:
        feature_type: Type of feature (hole, pocket, slot, etc.)
        subtype: More specific type (e.g., "through_hole", "blind_hole")
        parameters: Geometric parameters (diameter, depth, length, etc.)
        location: 3D location or 2D location in view
        confidence: Recognition confidence (0-1)
        view: View name where feature was detected
    """

    feature_type: str
    subtype: Optional[str] = None
    parameters: Dict[str, float] = Field(default_factory=dict)
    location: Optional[List[float]] = None
    confidence: float = 1.0
    view: str = ""
    description: str = ""


class FeatureRecognitionVlmModel(EnrichmentModel):
    """Enrichment model for recognizing machining features using VLM.

    This model uses vision-language models to identify and classify machining
    features from rendered CAD views. It recognizes:

    **Hole Features:**
    - Through holes
    - Blind holes
    - Counterbore holes
    - Countersunk holes
    - Threaded holes

    **Pocket and Slot Features:**
    - Rectangular pockets
    - Circular pockets
    - Through slots
    - Blind slots
    - T-slots

    **Edge Features:**
    - Fillets (concave rounds)
    - Chamfers (angled edges)
    - Rounds (convex)

    **Protrusion Features:**
    - Bosses (raised circular features)
    - Ribs (thin raised walls)
    - Lugs (mounting features)

    The model processes multiple views and uses cross-view validation to
    increase recognition accuracy.

    Attributes:
        options: Configuration options for feature recognition
        vlm: Vision-language model for feature detection
        feature_prompts: Specialized prompts for each feature type

    Example:
        options = CADAnnotationOptions(
            annotation_types=["hole", "pocket", "fillet"],
            vlm_model="gpt-4-vision",
            views_to_process=["front", "top", "right"]
        )
        model = FeatureRecognitionVlmModel(options)
        model(doc, [item])

        # Access detected features
        features = item.properties.get("machining_features", [])
        for feature in features:
            print(f"{feature['feature_type']}: {feature['parameters']}")
    """

    def __init__(self, options: CADAnnotationOptions):
        """Initialize feature recognition model.

        Args:
            options: Configuration options for feature recognition

        Raises:
            ValueError: If VLM model not supported
        """
        super().__init__()
        self.options = options
        self.vlm = self._initialize_vlm()
        self.feature_prompts = self._build_feature_prompts()

        _log.info(
            f"Initialized FeatureRecognitionVlmModel with {self.options.vlm_model}"
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
            import os

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if "claude" in model_lower:
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")

            vlm_options = ApiVlmOptions(
                api_key=api_key,
                model_name=self.options.vlm_model,
                temperature=0.0,
                max_tokens=4096,
                use_ocr=False,  # OCR not needed for feature recognition
            )
            return ApiVlmModel(vlm_options)

        # Local models
        else:
            vlm_options = InlineVlmOptions(
                model_path=self.options.vlm_model,
                device="cuda" if self.requires_gpu() else "cpu",
                temperature=0.0,
                max_tokens=2048,
                use_ocr=False,
            )
            return InlineVlmModel(vlm_options)

    def _build_feature_prompts(self) -> Dict[str, str]:
        """Build specialized prompts for each feature type.

        Returns:
            Dictionary mapping feature type to specialized prompt
        """
        prompts = {
            "hole": """
You are analyzing a CAD model to identify hole features. Identify all holes including:
- Through holes (completely through the part)
- Blind holes (do not go completely through)
- Counterbore holes (enlarged opening for bolt heads)
- Countersunk holes (angled opening for flat-head screws)
- Threaded holes (with threads visible)

For each hole, extract:
- Type (through, blind, counterbore, countersunk, threaded)
- Diameter (approximate from view)
- Depth (if visible)
- Location in the view
- Any associated dimensions

Return as JSON array:
[{"feature_type": "hole", "subtype": "through_hole", "parameters": {"diameter": 10.0, "depth": null}, "location": [x, y]}]
""",
            "pocket": """
You are analyzing a CAD model to identify pocket and slot features. Identify:
- Rectangular pockets (recessed rectangular areas)
- Circular pockets (recessed circular areas)
- Through slots (completely through)
- Blind slots (do not go through)
- T-slots (for mounting)

For each pocket/slot, extract:
- Type (rectangular_pocket, circular_pocket, through_slot, blind_slot, t_slot)
- Dimensions (length, width, diameter, depth)
- Location in the view

Return as JSON array:
[{"feature_type": "pocket", "subtype": "rectangular_pocket", "parameters": {"length": 50.0, "width": 20.0, "depth": 10.0}, "location": [x, y]}]
""",
            "fillet": """
You are analyzing a CAD model to identify fillet and chamfer features. Identify:
- Fillets (rounded concave edges)
- Chamfers (angled edges, typically 45°)
- Rounds (rounded convex edges)

For each edge feature, extract:
- Type (fillet, chamfer, round)
- Radius (for fillets/rounds) or distance (for chamfers)
- Approximate location
- Edge type (internal, external)

Return as JSON array:
[{"feature_type": "fillet", "subtype": "internal_fillet", "parameters": {"radius": 5.0}, "location": [x, y]}]
""",
            "chamfer": """
You are analyzing a CAD model to identify chamfer features. Identify:
- Edge chamfers (angled edges)
- Hole chamfers (on hole entrances)
- Part chamfers (on part edges)

For each chamfer, extract:
- Distance or angle
- Location
- Associated feature (if any)

Return as JSON array:
[{"feature_type": "chamfer", "subtype": "edge_chamfer", "parameters": {"distance": 2.0, "angle": 45.0}, "location": [x, y]}]
""",
            "boss": """
You are analyzing a CAD model to identify boss and rib features. Identify:
- Bosses (raised circular features, often for mounting holes)
- Ribs (thin raised walls for structural support)
- Lugs (mounting tabs or ears)

For each protrusion, extract:
- Type (boss, rib, lug)
- Dimensions (diameter/width, height, thickness)
- Location

Return as JSON array:
[{"feature_type": "boss", "subtype": "circular_boss", "parameters": {"diameter": 20.0, "height": 5.0}, "location": [x, y]}]
""",
            "thread": """
You are analyzing a CAD model to identify threaded features. Identify:
- External threads (on shafts, screws)
- Internal threads (in holes)
- Thread callouts if visible

For each thread, extract:
- Type (external, internal)
- Nominal diameter
- Thread pitch if visible
- Thread standard if labeled (M, UNC, UNF, etc.)

Return as JSON array:
[{"feature_type": "thread", "subtype": "internal_thread", "parameters": {"diameter": 10.0, "pitch": 1.5}, "description": "M10x1.5"}]
""",
        }

        return prompts

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: List[CADItem],
    ) -> None:
        """Recognize machining features from CAD items using VLM.

        Args:
            doc: The CADlingDocument being enriched
            item_batch: List of CADItem objects to process

        Note:
            Recognized features are added to item.properties["machining_features"]
        """
        _log.info(f"Processing {len(item_batch)} items for feature recognition")

        for item in item_batch:
            try:
                # Check if item has rendered images
                rendered_images = item.properties.get("rendered_images", {})
                if not rendered_images:
                    _log.warning(
                        f"Item {item.self_ref} has no rendered images, skipping"
                    )
                    continue

                # Detect features from each view
                all_features = []

                for view_name in self.options.views_to_process:
                    if view_name not in rendered_images:
                        _log.debug(f"View {view_name} not found, skipping")
                        continue

                    image = rendered_images[view_name]

                    # Detect each feature type
                    for feature_type in self.options.annotation_types:
                        if feature_type not in self.feature_prompts:
                            _log.warning(f"Unknown feature type: {feature_type}")
                            continue

                        # Get specialized prompt
                        prompt = self.feature_prompts[feature_type]

                        # Add geometric context if enabled
                        if self.options.include_geometric_context:
                            geom_context = self._build_geometric_context(doc, item)
                            prompt = f"{prompt}\n\nGeometric Context:\n{geom_context}"

                        # Run VLM prediction
                        response = self.vlm.predict(image, prompt)

                        # Parse features from response
                        features = self._parse_features_from_response(
                            response.raw_text, view_name
                        )

                        # Filter by confidence
                        for feature in features:
                            if feature["confidence"] >= self.options.min_confidence:
                                all_features.append(feature)

                        _log.debug(
                            f"Detected {len(features)} {feature_type} features "
                            f"from {view_name}"
                        )

                # Cross-view validation if enabled
                if (
                    self.options.enable_cross_view_validation
                    and len(self.options.views_to_process) > 1
                ):
                    all_features = self._validate_cross_view_features(all_features)

                # Add to item properties
                item.properties["machining_features"] = all_features
                item.properties["feature_recognition_model"] = self.__class__.__name__
                item.properties["feature_recognition_vlm"] = self.options.vlm_model

                # Add provenance
                if hasattr(item, "add_provenance"):
                    item.add_provenance(
                        component_type="enrichment_model",
                        component_name=self.__class__.__name__,
                    )

                _log.info(
                    f"Recognized {len(all_features)} machining features "
                    f"for item {item.self_ref}"
                )

            except Exception as e:
                _log.error(f"Feature recognition failed for item {item.self_ref}: {e}")
                item.properties["machining_features"] = []
                item.properties["feature_recognition_error"] = str(e)

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

        # Add bounding box info
        if "bounding_box" in item.properties:
            bbox = item.properties["bounding_box"]
            context_parts.append(f"Part dimensions (bbox): {bbox}")

        # Add topology info
        if hasattr(doc, "topology") and doc.topology:
            topo = doc.topology
            context_parts.append(
                f"Topology: {topo.get('num_faces', 0)} faces, "
                f"{topo.get('num_edges', 0)} edges, "
                f"{topo.get('num_vertices', 0)} vertices"
            )

        # Add material info if available
        if "material" in item.properties:
            context_parts.append(f"Material: {item.properties['material']}")

        return "\n".join(context_parts) if context_parts else "No geometric context available"

    def _parse_features_from_response(
        self, response_text: str, view_name: str
    ) -> List[Dict[str, Any]]:
        """Parse feature data from VLM response.

        Args:
            response_text: Raw text response from VLM
            view_name: Name of the view

        Returns:
            List of feature dictionaries
        """
        import json

        features = []

        try:
            # Try to find JSON array in response
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)

                for item in parsed:
                    # Ensure required fields
                    feature = {
                        "feature_type": item.get("feature_type", "unknown"),
                        "subtype": item.get("subtype"),
                        "parameters": item.get("parameters", {}),
                        "location": item.get("location"),
                        "confidence": item.get("confidence", 0.8),
                        "view": view_name,
                        "description": item.get("description", ""),
                    }
                    features.append(feature)

            _log.debug(f"Parsed {len(features)} features from response")

        except Exception as e:
            _log.error(f"Failed to parse features from response: {e}")

        return features

    def _validate_cross_view_features(
        self, features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate and merge features detected across multiple views.

        Args:
            features: List of features from all views

        Returns:
            Validated and merged feature list

        Note:
            Features detected in multiple views are merged with increased
            confidence. This helps reduce false positives.
        """
        if not features:
            return features

        # Group by feature type and approximate location/parameters
        validated = []
        processed = set()

        for i, feat1 in enumerate(features):
            if i in processed:
                continue

            # Find similar features in other views
            similar = [feat1]
            for j, feat2 in enumerate(features[i + 1 :], start=i + 1):
                if j in processed:
                    continue

                # Check if features are similar
                if self._features_similar(feat1, feat2):
                    similar.append(feat2)
                    processed.add(j)

            # Merge similar features
            if len(similar) == 1:
                # Single occurrence - keep as is
                validated.append(similar[0])
            else:
                # Multiple occurrences - merge with increased confidence
                merged = self._merge_features(similar)
                validated.append(merged)

            processed.add(i)

        _log.debug(
            f"Cross-view validation: {len(features)} -> {len(validated)} features"
        )

        return validated

    def _features_similar(
        self, feat1: Dict[str, Any], feat2: Dict[str, Any]
    ) -> bool:
        """Check if two features are similar enough to be the same.

        Args:
            feat1: First feature
            feat2: Second feature

        Returns:
            True if features are likely the same
        """
        # Must be same type
        if feat1["feature_type"] != feat2["feature_type"]:
            return False

        if feat1.get("subtype") and feat2.get("subtype"):
            if feat1["subtype"] != feat2["subtype"]:
                return False

        # Check parameter similarity
        params1 = feat1.get("parameters", {})
        params2 = feat2.get("parameters", {})

        # Compare common parameters
        for key in params1:
            if key in params2:
                val1, val2 = params1[key], params2[key]
                if val1 and val2:
                    # Allow 10% tolerance
                    if abs(val1 - val2) / max(val1, val2, 1e-6) > 0.1:
                        return False

        return True

    def _merge_features(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple similar features into one.

        Args:
            features: List of similar features

        Returns:
            Merged feature with averaged parameters and increased confidence
        """
        merged = features[0].copy()

        # Average numeric parameters
        params = {}
        for key in merged.get("parameters", {}):
            values = [
                f["parameters"][key]
                for f in features
                if key in f.get("parameters", {}) and f["parameters"][key] is not None
            ]
            if values:
                params[key] = sum(values) / len(values)

        merged["parameters"] = params

        # Increase confidence
        merged["confidence"] = min(
            1.0, merged["confidence"] * (1 + 0.15 * (len(features) - 1))
        )

        # Track views
        merged["views"] = [f["view"] for f in features]
        merged["cross_view_validated"] = True

        return merged

    def supports_batch_processing(self) -> bool:
        """Whether this model supports batch processing."""
        return False

    def requires_gpu(self) -> bool:
        """Whether this model requires GPU acceleration."""
        model_lower = self.options.vlm_model.lower()
        return not any(
            name in model_lower for name in ["gpt", "claude", "vision", "opus", "sonnet"]
        )

    def get_model_info(self) -> Dict[str, str]:
        """Get information about this model."""
        info = super().get_model_info()
        info.update(
            {
                "vlm_model": self.options.vlm_model,
                "feature_types": ",".join(self.options.annotation_types),
                "views": ",".join(self.options.views_to_process),
                "min_confidence": str(self.options.min_confidence),
            }
        )
        return info
