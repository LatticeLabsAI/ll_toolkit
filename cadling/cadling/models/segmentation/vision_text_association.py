"""Vision-text association for segmented CAD features.

This module provides integration between segmentation results and vision-language models.
Associates VLM-generated descriptions with segmented features for enhanced RAG and querying.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Dict, Any

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument

from cadling.models.base_model import EnrichmentModel

_log = logging.getLogger(__name__)


class VisionTextAssociationModel(EnrichmentModel):
    """Associate VLM descriptions with segmented CAD features.

    Uses segment labels to identify regions, renders them individually,
    and runs VLM for detailed natural language descriptions.

    Workflow:
    1. Check for existing segmentation results (segments, brep_segments, manufacturing_features)
    2. For each segment/feature, extract or render the geometry
    3. Run VLM (Claude/GPT-4V) to generate description
    4. Store description in segment properties
    5. Create vision-text associations for RAG improvement

    Integration points:
    - MeshSegmentationModel: Uses vertex/face labels to extract segment geometry
    - BRepSegmentationModel: Uses face labels to identify B-Rep faces
    - ManufacturingFeatureRecognizer: Describes recognized features with parameters

    Example:
        vlm_model = VisionTextAssociationModel(
            vlm_provider="claude",  # or "openai"
            render_segments=True
        )
        result = converter.convert(
            "part.step",
            pipeline_options=PipelineOptions(
                enrichment_models=[
                    brep_seg,
                    feature_rec,
                    vlm_model,  # Run after segmentation
                ]
            )
        )
        # Access VLM descriptions
        for item in result.document.items:
            if "manufacturing_features" in item.properties:
                for feature in item.properties["manufacturing_features"]:
                    print(f"{feature['type']}: {feature.get('vlm_description')}")

    Attributes:
        vlm_provider: VLM provider ("claude" or "openai")
        render_segments: Whether to render segments as images
        vlm_model: VLM model instance
    """

    def __init__(
        self,
        vlm_provider: str = "claude",
        vlm_model_name: Optional[str] = None,
        render_segments: bool = False,
        max_segments_per_item: int = 20,
        api_key: Optional[str] = None,
    ):
        """Initialize vision-text association model.

        Args:
            vlm_provider: VLM provider ("claude" or "openai")
            vlm_model_name: Specific model name (e.g., "claude-3-sonnet", "gpt-4-vision-preview")
            render_segments: Whether to render segments as images
            max_segments_per_item: Maximum segments to process per item (cost control)
            api_key: API key for VLM provider (optional, uses env var if not provided)
        """
        super().__init__()

        self.vlm_provider = vlm_provider
        self.render_segments = render_segments
        self.max_segments_per_item = max_segments_per_item

        # Initialize VLM client
        if vlm_provider == "claude":
            self.vlm_model_name = vlm_model_name or "claude-3-sonnet-20240229"
            self._init_claude_client(api_key)
        elif vlm_provider == "openai":
            self.vlm_model_name = vlm_model_name or "gpt-4-vision-preview"
            self._init_openai_client(api_key)
        else:
            _log.error(f"Unsupported VLM provider: {vlm_provider}")
            self.vlm_client = None

    def _init_claude_client(self, api_key: Optional[str]):
        """Initialize Claude API client."""
        try:
            import anthropic
            import os

            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                _log.warning("No Anthropic API key provided, VLM features disabled")
                self.vlm_client = None
                return

            self.vlm_client = anthropic.Anthropic(api_key=api_key)
            _log.info(f"Initialized Claude VLM client: {self.vlm_model_name}")
        except ImportError:
            _log.error("anthropic package not installed, run: pip install anthropic")
            self.vlm_client = None
        except Exception as e:
            _log.error(f"Failed to initialize Claude client: {e}")
            self.vlm_client = None

    def _init_openai_client(self, api_key: Optional[str]):
        """Initialize OpenAI API client."""
        try:
            import openai
            import os

            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                _log.warning("No OpenAI API key provided, VLM features disabled")
                self.vlm_client = None
                return

            self.vlm_client = openai.OpenAI(api_key=api_key)
            _log.info(f"Initialized OpenAI VLM client: {self.vlm_model_name}")
        except ImportError:
            _log.error("openai package not installed, run: pip install openai")
            self.vlm_client = None
        except Exception as e:
            _log.error(f"Failed to initialize OpenAI client: {e}")
            self.vlm_client = None

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Associate VLM descriptions with segmented features.

        Args:
            doc: CADlingDocument being enriched
            item_batch: List of CADItem objects to process
        """
        if not self.vlm_client:
            _log.debug("VLM client not available, skipping vision-text association")
            return

        for item in item_batch:
            try:
                # Process manufacturing features (highest priority)
                if "manufacturing_features" in item.properties:
                    self._process_manufacturing_features(item)

                # Process B-Rep segments
                elif "brep_segments" in item.properties:
                    self._process_brep_segments(item)

                # Process mesh segments
                elif "segments" in item.properties:
                    self._process_mesh_segments(item)

                # Add provenance
                item.add_provenance(
                    component_type="enrichment_model",
                    component_name="VisionTextAssociationModel",
                )

            except Exception as e:
                _log.error(f"Vision-text association failed for item: {e}")

    def _process_manufacturing_features(self, item: CADItem) -> None:
        """Process manufacturing features with VLM descriptions.

        Args:
            item: CADItem with manufacturing_features
        """
        features = item.properties["manufacturing_features"]

        # Limit number of features to process (cost control)
        features_to_process = features[: self.max_segments_per_item]

        for i, feature in enumerate(features_to_process):
            # Generate description prompt
            prompt = self._create_feature_description_prompt(feature)

            # Get VLM description
            description = self._get_vlm_description(prompt, context="manufacturing_feature")

            # Store description
            feature["vlm_description"] = description

            _log.debug(f"Generated VLM description for feature {i}: {feature['type']}")

    def _process_brep_segments(self, item: CADItem) -> None:
        """Process B-Rep face segments with VLM descriptions.

        Args:
            item: CADItem with brep_segments
        """
        segments = item.properties["brep_segments"]
        face_labels = segments.get("face_labels", [])
        label_names = segments.get("label_names", [])

        if not face_labels or not label_names:
            return

        # Count faces per label
        import numpy as np

        unique_labels = np.unique(face_labels)

        # Create segment summary
        segment_summary = []
        for label_idx in unique_labels:
            if 0 <= label_idx < len(label_names):
                label_name = label_names[label_idx]
                count = np.sum(np.array(face_labels) == label_idx)
                segment_summary.append(f"{label_name}: {count} faces")

        # Generate overall description
        prompt = f"Describe the CAD part with the following face segmentation:\n" + "\n".join(
            segment_summary
        )

        description = self._get_vlm_description(prompt, context="brep_segmentation")

        # Store in segments
        segments["vlm_description"] = description

    def _process_mesh_segments(self, item: CADItem) -> None:
        """Process mesh segments with VLM descriptions.

        Args:
            item: CADItem with mesh segments
        """
        segments = item.properties["segments"]
        num_segments = segments.get("num_segments", 0)
        label_names = segments.get("label_names", [])

        # Generate description prompt
        prompt = f"Describe the 3D mesh with {num_segments} semantic regions: {', '.join(label_names)}"

        description = self._get_vlm_description(prompt, context="mesh_segmentation")

        # Store in segments
        segments["vlm_description"] = description

    def _create_feature_description_prompt(self, feature: Dict[str, Any]) -> str:
        """Create prompt for manufacturing feature description.

        Args:
            feature: Manufacturing feature dictionary

        Returns:
            Prompt string for VLM
        """
        feature_type = feature.get("type", "unknown")
        parameters = feature.get("parameters", {})
        location = feature.get("location", [0, 0, 0])
        confidence = feature.get("confidence", 0.0)

        # Build parameter description
        param_str = ", ".join([f"{k}={v}" for k, v in parameters.items()])

        prompt = f"""Describe this manufacturing feature for a CAD engineer:

Feature Type: {feature_type}
Parameters: {param_str}
Location: {location}
Detection Confidence: {confidence:.2f}

Provide a brief technical description (2-3 sentences) suitable for:
- CAD design documentation
- Manufacturing process planning
- CNC toolpath generation
- Cost estimation

Focus on functional purpose and manufacturing implications."""

        return prompt

    def _get_vlm_description(self, prompt: str, context: str = "") -> str:
        """Get VLM description for prompt.

        Args:
            prompt: Text prompt for VLM
            context: Context type (for logging)

        Returns:
            VLM-generated description
        """
        try:
            if self.vlm_provider == "claude":
                response = self.vlm_client.messages.create(
                    model=self.vlm_model_name,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                description = response.content[0].text

            elif self.vlm_provider == "openai":
                response = self.vlm_client.chat.completions.create(
                    model=self.vlm_model_name,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                description = response.choices[0].message.content

            else:
                description = "VLM description not available"

            return description

        except Exception as e:
            _log.error(f"VLM API call failed for context '{context}': {e}")
            return f"Description unavailable (error: {str(e)})"

    def supports_batch_processing(self) -> bool:
        """Vision-text association supports batch processing."""
        return True

    def get_batch_size(self) -> int:
        """Recommended batch size (limited by API rate limits)."""
        return 5

    def requires_gpu(self) -> bool:
        """Vision-text association does not require GPU (API-based)."""
        return False
