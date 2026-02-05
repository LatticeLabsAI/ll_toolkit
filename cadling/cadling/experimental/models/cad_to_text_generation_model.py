"""CAD-to-text generation model for creating natural language descriptions.

This module provides an enrichment model for generating comprehensive natural
language descriptions of CAD parts, including geometric descriptions, feature
summaries, manufacturing notes, and assembly instructions.

Classes:
    CADToTextGenerationModel: Enrichment model for text generation
    CADDescription: Structured CAD part description

Example:
    from cadling.experimental.models import CADToTextGenerationModel
    from cadling.experimental.datamodel import CADAnnotationOptions

    options = CADAnnotationOptions(vlm_model="gpt-4-vision")
    model = CADToTextGenerationModel(options)
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


class CADDescription(BaseModel):
    """Structured description of a CAD part.

    Attributes:
        summary: Brief one-sentence summary of the part
        detailed_description: Comprehensive description of the part
        key_features: List of notable geometric features
        dimensions_summary: Summary of key dimensions
        material_notes: Material and finish information
        manufacturing_notes: Manufacturing process recommendations
        assembly_instructions: Assembly or installation instructions
        technical_specifications: Technical specs and requirements
        context_aware_notes: Additional context-specific information
    """

    summary: str = ""
    detailed_description: str = ""
    key_features: List[str] = Field(default_factory=list)
    dimensions_summary: str = ""
    material_notes: str = ""
    manufacturing_notes: str = ""
    assembly_instructions: str = ""
    technical_specifications: List[str] = Field(default_factory=list)
    context_aware_notes: str = ""


class CADToTextGenerationModel(EnrichmentModel):
    """Enrichment model for generating natural language descriptions of CAD parts.

    This model creates comprehensive textual descriptions of CAD models by
    combining:

    1. **Template-Based Generation**: Structured descriptions from extracted data
    2. **VLM-Based Captioning**: Natural language understanding of visual form
    3. **Feature Summarization**: Aggregation of detected features
    4. **Context Integration**: Incorporation of design intent, manufacturability

    The model can generate:
    - **Part Summaries**: Brief descriptions for catalogs/BOMs
    - **Detailed Descriptions**: Comprehensive technical documentation
    - **Manufacturing Instructions**: Process recommendations and notes
    - **Assembly Instructions**: How to install/use the part
    - **Technical Specs**: Key dimensions and requirements

    Useful for:
    - Automated documentation generation
    - Part database population
    - RAG (Retrieval-Augmented Generation) applications
    - Technical writing assistance
    - Assembly manuals

    Attributes:
        options: Configuration options
        vlm: Vision-language model for captioning
        templates: Description templates for different contexts

    Example:
        options = CADAnnotationOptions(vlm_model="gpt-4-vision")
        model = CADToTextGenerationModel(options)
        model(doc, [item])

        # Access generated description
        desc = item.properties.get("text_description")
        print(desc["summary"])
        print(desc["detailed_description"])
    """

    def __init__(self, options: CADAnnotationOptions):
        """Initialize CAD-to-text generation model.

        Args:
            options: Configuration options

        Raises:
            ValueError: If VLM model not supported
        """
        super().__init__()
        self.options = options
        self.vlm = self._initialize_vlm()
        self.templates = self._load_templates()

        _log.info("Initialized CADToTextGenerationModel")

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
                temperature=0.7,  # More creative for natural language
                max_tokens=4096,
                use_ocr=False,
            )
            return ApiVlmModel(vlm_options)

        # Local models
        else:
            vlm_options = InlineVlmOptions(
                model_path=self.options.vlm_model,
                device="cuda" if self.requires_gpu() else "cpu",
                temperature=0.7,
                max_tokens=2048,
                use_ocr=False,
            )
            return InlineVlmModel(vlm_options)

    def _load_templates(self) -> Dict[str, str]:
        """Load description templates.

        Returns:
            Dictionary of template strings
        """
        return {
            "summary": "A {part_type} featuring {key_features}",
            "dimensions": "Overall dimensions: {bbox_description}",
            "features_list": "Key features include: {features}",
        }

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: List[CADItem],
    ) -> None:
        """Generate text descriptions for CAD items.

        Args:
            doc: The CADlingDocument being enriched
            item_batch: List of CADItem objects to process

        Note:
            Generated description is added to item.properties["text_description"]
        """
        _log.info(f"Processing {len(item_batch)} items for text generation")

        for item in item_batch:
            try:
                # Step 1: Generate template-based description
                template_desc = self._generate_template_description(doc, item)

                # Step 2: Generate VLM-based caption
                vlm_desc = self._generate_vlm_description(doc, item)

                # Step 3: Combine and structure
                final_desc = self._combine_descriptions(
                    doc, item, template_desc, vlm_desc
                )

                # Add to item properties
                item.properties["text_description"] = final_desc.model_dump()
                item.properties["text_generation_model"] = self.__class__.__name__

                # Add provenance
                if hasattr(item, "add_provenance"):
                    item.add_provenance(
                        component_type="enrichment_model",
                        component_name=self.__class__.__name__,
                    )

                _log.info(
                    f"Generated description for item {item.self_ref} "
                    f"({len(final_desc.summary)} chars summary)"
                )

            except Exception as e:
                _log.error(f"Text generation failed for item {item.self_ref}: {e}")
                item.properties["text_description"] = None
                item.properties["text_generation_error"] = str(e)

    def _generate_template_description(
        self, doc: CADlingDocument, item: CADItem
    ) -> Dict[str, Any]:
        """Generate description using templates and extracted data.

        Args:
            doc: The document
            item: The item to describe

        Returns:
            Dictionary with template-generated content
        """
        desc = {}

        # Extract bounding box for dimensions
        bbox = item.properties.get("bounding_box", {})
        if bbox:
            desc["dimensions"] = (
                f"Dimensions: {bbox.get('x', 0):.1f} x "
                f"{bbox.get('y', 0):.1f} x {bbox.get('z', 0):.1f} mm"
            )

        # Summarize detected features
        features = item.properties.get("machining_features", [])
        if features:
            feature_counts = {}
            for feat in features:
                ftype = feat.get("feature_type", "unknown")
                feature_counts[ftype] = feature_counts.get(ftype, 0) + 1

            feature_list = [
                f"{count} {ftype}{'s' if count > 1 else ''}"
                for ftype, count in feature_counts.items()
            ]
            desc["features"] = ", ".join(feature_list)

        # Include PMI annotations if available
        pmi = item.properties.get("pmi_annotations", [])
        if pmi:
            dimensions = [a for a in pmi if a.get("type") == "dimension"]
            tolerances = [a for a in pmi if a.get("type") == "tolerance"]
            desc["pmi_summary"] = (
                f"{len(dimensions)} dimensions, {len(tolerances)} tolerances specified"
            )

        # Include material if available
        if "material" in item.properties:
            desc["material"] = item.properties["material"]

        # Include design intent if available
        intent = item.properties.get("design_intent")
        if intent:
            desc["intent"] = intent.get("functional_description", "")

        # Include manufacturability notes
        mfg_report = item.properties.get("manufacturability_report")
        if mfg_report:
            desc["manufacturability"] = (
                f"Manufacturability score: {mfg_report.get('overall_score', 0):.1f}/100. "
                f"{mfg_report.get('estimated_difficulty', '')}"
            )

        return desc

    def _generate_vlm_description(
        self, doc: CADlingDocument, item: CADItem
    ) -> Dict[str, Any]:
        """Generate description using VLM caption.

        Args:
            doc: The document
            item: The item to describe

        Returns:
            Dictionary with VLM-generated content
        """
        rendered_images = item.properties.get("rendered_images", {})
        if not rendered_images:
            return {}

        prompt = """
You are a technical writer creating documentation for a CAD part. Provide a comprehensive description including:

1. **Summary** (1 sentence): What is this part and what is its primary purpose?

2. **Detailed Description** (2-3 paragraphs):
   - Overall form and geometry
   - Key features and their purposes
   - Notable design characteristics
   - How the part might be used

3. **Key Features** (bullet list):
   - List 3-5 most important geometric features
   - Be specific about their characteristics

4. **Manufacturing Considerations**:
   - Suggested manufacturing processes
   - Any notable manufacturing challenges
   - Material recommendations if apparent

5. **Assembly/Installation Notes**:
   - How this part would be installed or assembled
   - Connection points or interfaces
   - Orientation requirements

Return as JSON:
{
  "summary": "...",
  "detailed_description": "...",
  "key_features": ["...", "..."],
  "manufacturing_notes": "...",
  "assembly_instructions": "...",
  "technical_specifications": ["...", "..."]
}
"""

        try:
            # Use isometric view if available
            view_name = "isometric" if "isometric" in rendered_images else list(rendered_images.keys())[0]
            image = rendered_images[view_name]

            # Run VLM caption generation
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
            _log.error(f"VLM description generation failed: {e}")

        return {}

    def _combine_descriptions(
        self,
        doc: CADlingDocument,
        item: CADItem,
        template_desc: Dict[str, Any],
        vlm_desc: Dict[str, Any],
    ) -> CADDescription:
        """Combine template and VLM descriptions into final output.

        Args:
            doc: The document
            item: The item
            template_desc: Template-generated content
            vlm_desc: VLM-generated content

        Returns:
            CADDescription object
        """
        # Use VLM summary or generate from template
        summary = vlm_desc.get("summary", "")
        if not summary and "features" in template_desc:
            summary = f"CAD part with {template_desc['features']}"

        # Use VLM detailed description or build from template
        detailed = vlm_desc.get("detailed_description", "")
        if not detailed:
            parts = []
            if "dimensions" in template_desc:
                parts.append(template_desc["dimensions"])
            if "features" in template_desc:
                parts.append(f"Features: {template_desc['features']}")
            if "intent" in template_desc:
                parts.append(template_desc["intent"])
            detailed = ". ".join(parts) + "."

        # Combine key features from both sources
        key_features = vlm_desc.get("key_features", [])
        if "features" in template_desc and not key_features:
            key_features = [template_desc["features"]]

        # Build dimensions summary
        dimensions_summary = template_desc.get("dimensions", "")
        if "pmi_summary" in template_desc:
            dimensions_summary += f" ({template_desc['pmi_summary']})"

        # Material notes
        material_notes = template_desc.get("material", "")

        # Manufacturing notes
        manufacturing_notes = vlm_desc.get("manufacturing_notes", "")
        if "manufacturability" in template_desc:
            manufacturing_notes = (
                f"{template_desc['manufacturability']}. {manufacturing_notes}"
            )

        # Assembly instructions
        assembly_instructions = vlm_desc.get("assembly_instructions", "")

        # Technical specifications
        tech_specs = vlm_desc.get("technical_specifications", [])
        if dimensions_summary and dimensions_summary not in tech_specs:
            tech_specs.insert(0, dimensions_summary)

        # Context-aware notes
        context_notes = ""
        if "intent" in template_desc:
            context_notes = f"Design Intent: {template_desc['intent']}"

        return CADDescription(
            summary=summary,
            detailed_description=detailed,
            key_features=key_features,
            dimensions_summary=dimensions_summary,
            material_notes=material_notes,
            manufacturing_notes=manufacturing_notes,
            assembly_instructions=assembly_instructions,
            technical_specifications=tech_specs,
            context_aware_notes=context_notes,
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
                "supports_templates": "true",
            }
        )
        return info
