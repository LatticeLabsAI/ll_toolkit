"""Design intent inference model using geometric analysis and VLM.

This module provides an enrichment model for inferring the design intent
and functional purpose of CAD parts by analyzing geometric features,
topology, and visual characteristics.

Classes:
    DesignIntent: Data structure for inferred design intent
    DesignIntentInferenceModel: Enrichment model for intent inference
    IntentCategory: Enumeration of intent categories

Example:
    from cadling.experimental.models import DesignIntentInferenceModel
    from cadling.experimental.datamodel import CADAnnotationOptions

    options = CADAnnotationOptions(vlm_model="gpt-4-vision")
    model = DesignIntentInferenceModel(options)
    model(doc, item_batch)
"""

from __future__ import annotations

import logging
from enum import Enum
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


class IntentCategory(str, Enum):
    """Categories of design intent."""

    STRUCTURAL = "structural"  # Load-bearing, strength-critical
    COSMETIC = "cosmetic"  # Aesthetic, appearance-focused
    MOUNTING = "mounting"  # Attachment, fastening
    ALIGNMENT = "alignment"  # Positioning, locating
    SEALING = "sealing"  # Fluid/gas containment
    THERMAL = "thermal"  # Heat dissipation/insulation
    ELECTRICAL = "electrical"  # Electrical routing/insulation
    MOTION = "motion"  # Moving parts, articulation
    FLUID_FLOW = "fluid_flow"  # Fluid/gas passage
    PROTECTION = "protection"  # Shielding, guarding
    ERGONOMIC = "ergonomic"  # User interaction, comfort
    UNKNOWN = "unknown"  # Cannot determine intent


class LoadType(str, Enum):
    """Types of mechanical loads."""

    TENSION = "tension"
    COMPRESSION = "compression"
    SHEAR = "shear"
    BENDING = "bending"
    TORSION = "torsion"
    COMBINED = "combined"
    NONE = "none"


class DesignIntent(BaseModel):
    """Data structure for inferred design intent.

    Attributes:
        primary_intent: Primary design intent category
        secondary_intents: Additional intent categories
        confidence: Confidence in intent classification (0-1)
        is_load_bearing: Whether part/feature is load-bearing
        expected_loads: Types of loads part is designed for
        functional_description: Natural language description of function
        design_rationale: Inferred reasoning behind design choices
        critical_features: Features essential to the design intent
        constraints: Inferred design constraints
    """

    primary_intent: IntentCategory
    secondary_intents: List[IntentCategory] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    is_load_bearing: bool = False
    expected_loads: List[LoadType] = Field(default_factory=list)
    functional_description: str = ""
    design_rationale: str = ""
    critical_features: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)


class DesignIntentInferenceModel(EnrichmentModel):
    """Enrichment model for inferring design intent from CAD models.

    This model analyzes CAD parts to understand their functional purpose
    and design intent by combining:

    1. **Geometric Analysis**: Part shape, features, topology
    2. **Feature Pattern Recognition**: Common design patterns
    3. **VLM-Based Inference**: Visual understanding of form and function
    4. **Material and Property Analysis**: How materials relate to function

    The model can identify:
    - **Structural Intent**: Load-bearing members, reinforcements, ribs
    - **Mounting Intent**: Brackets, bosses, mounting holes
    - **Alignment Intent**: Locating pins, key features, datum surfaces
    - **Sealing Intent**: O-ring grooves, gasket surfaces
    - **Thermal Intent**: Heat sinks, cooling fins, thermal breaks
    - **Motion Intent**: Bearing surfaces, pivot points, sliding features
    - **Cosmetic Intent**: Decorative features, brand elements

    Attributes:
        options: Configuration options
        vlm: Vision-language model for visual analysis
        intent_patterns: Geometric patterns indicating specific intents

    Example:
        options = CADAnnotationOptions(vlm_model="gpt-4-vision")
        model = DesignIntentInferenceModel(options)
        model(doc, [item])

        # Access inferred intent
        intent = item.properties.get("design_intent")
        print(f"Primary intent: {intent['primary_intent']}")
        print(f"Function: {intent['functional_description']}")
    """

    def __init__(self, options: CADAnnotationOptions):
        """Initialize design intent inference model.

        Args:
            options: Configuration options

        Raises:
            ValueError: If VLM model not supported
        """
        super().__init__()
        self.options = options
        self.vlm = self._initialize_vlm()
        self.intent_patterns = self._define_intent_patterns()

        _log.info("Initialized DesignIntentInferenceModel")

    def _initialize_vlm(self) -> VlmModel:
        """Initialize the VLM based on model name."""
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
                temperature=0.3,  # Allow some creativity for inference
                max_tokens=4096,
                use_ocr=False,
            )
            return ApiVlmModel(vlm_options)

        # Local models
        else:
            vlm_options = InlineVlmOptions(
                model_path=self.options.vlm_model,
                device="cuda" if self.requires_gpu() else "cpu",
                temperature=0.3,
                max_tokens=2048,
                use_ocr=False,
            )
            return InlineVlmModel(vlm_options)

    def _define_intent_patterns(self) -> Dict[IntentCategory, Dict[str, Any]]:
        """Define geometric patterns that indicate specific design intents.

        Returns:
            Dictionary mapping intent categories to pattern definitions
        """
        return {
            IntentCategory.STRUCTURAL: {
                "features": ["rib", "boss", "thick_section"],
                "topology": "high_connectivity",
                "material_hint": ["steel", "aluminum", "titanium"],
            },
            IntentCategory.MOUNTING: {
                "features": ["hole", "boss", "slot", "tab"],
                "patterns": ["bolt_circle", "mounting_holes"],
            },
            IntentCategory.ALIGNMENT: {
                "features": ["pin", "key", "datum_surface"],
                "patterns": ["locating_feature", "registration"],
            },
            IntentCategory.SEALING: {
                "features": ["groove", "flat_surface"],
                "patterns": ["o_ring_groove", "gasket_surface"],
            },
            IntentCategory.THERMAL: {
                "features": ["fin", "rib", "large_surface_area"],
                "patterns": ["heat_sink", "cooling_fins"],
            },
            IntentCategory.MOTION: {
                "features": ["bearing_surface", "pivot", "sliding_surface"],
                "patterns": ["clearance", "running_fit"],
            },
        }

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: List[CADItem],
    ) -> None:
        """Infer design intent from CAD items.

        Args:
            doc: The CADlingDocument being enriched
            item_batch: List of CADItem objects to process

        Note:
            Inferred intent is added to item.properties["design_intent"]
        """
        _log.info(f"Processing {len(item_batch)} items for design intent inference")

        for item in item_batch:
            try:
                # Step 1: Geometric pattern matching
                geometric_hints = self._analyze_geometric_patterns(doc, item)

                # Step 2: VLM-based visual inference
                vlm_intent = self._vlm_intent_inference(doc, item)

                # Step 3: Combine evidence and determine intent
                design_intent = self._determine_intent(
                    doc, item, geometric_hints, vlm_intent
                )

                # Add to item properties
                item.properties["design_intent"] = design_intent.model_dump()
                item.properties["design_intent_model"] = self.__class__.__name__

                # Add provenance
                if hasattr(item, "add_provenance"):
                    item.add_provenance(
                        component_type="enrichment_model",
                        component_name=self.__class__.__name__,
                    )

                _log.info(
                    f"Inferred design intent for item {item.self_ref}: "
                    f"{design_intent.primary_intent} (confidence={design_intent.confidence:.2f})"
                )

            except Exception as e:
                _log.error(f"Design intent inference failed for item {item.self_ref}: {e}")
                item.properties["design_intent"] = None
                item.properties["design_intent_error"] = str(e)

    def _analyze_geometric_patterns(
        self, doc: CADlingDocument, item: CADItem
    ) -> Dict[IntentCategory, float]:
        """Analyze geometric features to identify intent patterns.

        Args:
            doc: The document
            item: The item to analyze

        Returns:
            Dictionary mapping intent categories to confidence scores
        """
        hints: Dict[IntentCategory, float] = {}

        # Get detected features
        features = item.properties.get("machining_features", [])
        feature_types = set(f.get("feature_type") for f in features)

        # Check each intent pattern
        for intent, pattern_def in self.intent_patterns.items():
            score = 0.0
            matches = 0
            total_checks = 0

            # Check feature matches
            if "features" in pattern_def:
                for required_feature in pattern_def["features"]:
                    total_checks += 1
                    if required_feature in feature_types:
                        matches += 1

            # Check for specific patterns (e.g., bolt circle)
            if "patterns" in pattern_def:
                for pattern_name in pattern_def["patterns"]:
                    total_checks += 1
                    if self._detect_pattern(features, pattern_name):
                        matches += 1

            # Calculate score
            if total_checks > 0:
                score = matches / total_checks

            if score > 0:
                hints[intent] = score

        return hints

    def _detect_pattern(self, features: List[Dict[str, Any]], pattern_name: str) -> bool:
        """Detect specific geometric patterns in features.

        Args:
            features: List of detected features
            pattern_name: Name of pattern to detect

        Returns:
            True if pattern detected
        """
        if pattern_name == "bolt_circle":
            # Check for circular pattern of holes
            holes = [f for f in features if f.get("feature_type") == "hole"]
            return len(holes) >= 3  # Simplified check

        elif pattern_name == "mounting_holes":
            # Check for holes with bosses
            holes = [f for f in features if f.get("feature_type") == "hole"]
            bosses = [f for f in features if f.get("feature_type") == "boss"]
            return len(holes) > 0 and len(bosses) > 0

        elif pattern_name == "o_ring_groove":
            # Check for groove with specific proportions
            grooves = [f for f in features if f.get("feature_type") == "pocket" and
                      f.get("subtype") == "circular_pocket"]
            return len(grooves) > 0

        return False

    def _vlm_intent_inference(
        self, doc: CADlingDocument, item: CADItem
    ) -> Dict[str, Any]:
        """Use VLM to infer design intent from visual analysis.

        Args:
            doc: The document
            item: The item to analyze

        Returns:
            Dictionary with VLM-inferred intent information
        """
        rendered_images = item.properties.get("rendered_images", {})
        if not rendered_images:
            return {}

        prompt = """
You are a mechanical design engineer analyzing a CAD part. Infer the design intent and functional purpose of this part.

Analyze:
1. **Overall Form**: What does the shape suggest about its function?
2. **Key Features**: Which features are critical to its purpose?
3. **Load Path**: Where would forces be applied and transmitted?
4. **Functional Purpose**: What is this part designed to do?
5. **Design Constraints**: What constraints influenced this design?

Consider these intent categories:
- Structural: Load-bearing, strength-critical components
- Mounting: Brackets, attachment features
- Alignment: Locating pins, datum surfaces
- Sealing: O-ring grooves, gasket surfaces
- Thermal: Heat sinks, cooling features
- Motion: Bearings, pivots, sliding surfaces
- Cosmetic: Aesthetic, branding

Provide your analysis as JSON:
{
  "primary_intent": "...",
  "secondary_intents": ["..."],
  "confidence": 0.0-1.0,
  "is_load_bearing": true/false,
  "expected_loads": ["tension", "compression", "bending", ...],
  "functional_description": "This part functions as...",
  "design_rationale": "The design suggests...",
  "critical_features": ["...", "..."],
  "constraints": ["...", "..."]
}
"""

        try:
            # Use isometric view if available
            view_name = "isometric" if "isometric" in rendered_images else list(rendered_images.keys())[0]
            image = rendered_images[view_name]

            # Run VLM inference
            response = self.vlm.predict(image, prompt)

            # Parse response
            import json

            json_start = response.raw_text.find("{")
            json_end = response.raw_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response.raw_text[json_start:json_end]
                parsed = json.loads(json_str)
                return parsed

        except Exception as e:
            _log.error(f"VLM intent inference failed: {e}")

        return {}

    def _determine_intent(
        self,
        doc: CADlingDocument,
        item: CADItem,
        geometric_hints: Dict[IntentCategory, float],
        vlm_intent: Dict[str, Any],
    ) -> DesignIntent:
        """Combine geometric and VLM evidence to determine final intent.

        Args:
            doc: The document
            item: The item
            geometric_hints: Scores from geometric analysis
            vlm_intent: Intent from VLM analysis

        Returns:
            DesignIntent object
        """
        # Parse VLM intent
        vlm_primary = vlm_intent.get("primary_intent", "unknown")
        vlm_confidence = vlm_intent.get("confidence", 0.5)

        # Try to match VLM intent to category
        try:
            primary_intent = IntentCategory(vlm_primary.lower())
        except ValueError:
            # Use highest scoring geometric hint
            if geometric_hints:
                primary_intent = max(geometric_hints, key=geometric_hints.get)
            else:
                primary_intent = IntentCategory.UNKNOWN

        # Combine confidences (weighted average)
        geometric_conf = geometric_hints.get(primary_intent, 0.0)
        combined_confidence = 0.6 * vlm_confidence + 0.4 * geometric_conf

        # Parse secondary intents
        secondary_intents = []
        for intent_str in vlm_intent.get("secondary_intents", []):
            try:
                intent = IntentCategory(intent_str.lower())
                if intent != primary_intent:
                    secondary_intents.append(intent)
            except ValueError:
                pass

        # Parse expected loads
        expected_loads = []
        for load_str in vlm_intent.get("expected_loads", []):
            try:
                load = LoadType(load_str.lower())
                expected_loads.append(load)
            except ValueError:
                pass

        return DesignIntent(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=combined_confidence,
            is_load_bearing=vlm_intent.get("is_load_bearing", False),
            expected_loads=expected_loads,
            functional_description=vlm_intent.get("functional_description", ""),
            design_rationale=vlm_intent.get("design_rationale", ""),
            critical_features=vlm_intent.get("critical_features", []),
            constraints=vlm_intent.get("constraints", []),
        )

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
                "intent_categories": str(len(IntentCategory)),
            }
        )
        return info
