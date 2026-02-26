"""CAD part classification model.

This module provides classification models for CAD parts using ll_stepnet.

Classes:
    CADPartClassifier: Classify CAD parts into categories
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from cadling.models.base_model import EnrichmentModel

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument
    from cadling.datamodel.step import STEPEntityItem

_log = logging.getLogger(__name__)

try:
    import torch
    from stepnet import (
        STEPForClassification,
        STEPTokenizer,
        STEPFeatureExtractor,
        STEPTopologyBuilder,
    )
    _has_torch = True
except ImportError:
    torch = None
    STEPForClassification = None
    STEPTokenizer = None
    STEPFeatureExtractor = None
    STEPTopologyBuilder = None
    _has_torch = False


class CADPartClassifier(EnrichmentModel):
    """CAD part classification model.

    Classifies CAD parts into categories using ll_stepnet's STEPForClassification
    model. Categories may include:
    - Mechanical parts: bracket, housing, shaft, gear, bearing, etc.
    - Structural parts: beam, plate, frame, etc.
    - Fasteners: bolt, nut, screw, washer, etc.

    The model uses STEP entities and topology to make predictions.

    Attributes:
        model: STEPForClassification model instance
        class_names: List of class names
        artifacts_path: Path to model artifacts

    Example:
        classifier = CADPartClassifier(Path("models/classifier.pt"))
        result = converter.convert(
            "part.step",
            pipeline_options=PipelineOptions(
                enrichment_models=[classifier]
            )
        )
        for item in result.document.items:
            if "predicted_class" in item.properties:
                print(f"Class: {item.properties['predicted_class']}")
    """

    def __init__(
        self,
        artifacts_path: Optional[Path] = None,
        class_names: Optional[list[str]] = None,
        vocab_size: int = 50000,
        output_dim: int = 1024,
    ):
        """Initialize CAD part classifier.

        Args:
            artifacts_path: Path to model checkpoint (.pt file)
            class_names: List of class names (optional)
            vocab_size: Vocabulary size for tokenizer
            output_dim: Model output dimension
        """
        super().__init__()

        self.artifacts_path = artifacts_path
        self.class_names = class_names or [
            "bracket",
            "housing",
            "shaft",
            "gear",
            "bearing",
            "plate",
            "frame",
            "fastener",
            "unknown",
        ]
        self.vocab_size = vocab_size
        self.output_dim = output_dim

        # Initialize tokenizer and extractors
        if _has_torch and STEPTokenizer is not None:
            self.tokenizer = STEPTokenizer(vocab_size=vocab_size)
            self.feature_extractor = STEPFeatureExtractor()
            self.topology_builder = STEPTopologyBuilder()
        else:
            _log.warning("torch/stepnet not available; tokenizer and extractors disabled")
            self.tokenizer = None
            self.feature_extractor = None
            self.topology_builder = None

        # Load model if artifacts_path provided
        if artifacts_path:
            try:
                # Create model instance
                self.model = STEPForClassification(
                    vocab_size=vocab_size,
                    num_classes=len(self.class_names),
                    output_dim=output_dim,
                )

                # Load checkpoint
                checkpoint = torch.load(
                    str(artifacts_path), map_location="cpu", weights_only=False
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()

                _log.info(f"Loaded classification model from {artifacts_path}")
            except Exception as e:
                _log.error(f"Failed to load classification model: {e}")
                self.model = None
        else:
            self.model = None
            _log.info("Classification model path not provided")

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Classify CAD items.

        Args:
            doc: CADlingDocument being enriched
            item_batch: List of CADItem objects to classify
        """
        from cadling.datamodel.step import STEPEntityItem

        if not self.model:
            _log.debug("Classification skipped: model not available")
            return

        for item in item_batch:
            # Only classify STEP entities
            if not isinstance(item, STEPEntityItem):
                continue

            try:
                # Prepare input
                entity_text = item.text or ""

                # Tokenize entity text
                token_ids = self.tokenizer.encode(entity_text)

                # Truncate or pad to max length
                max_length = 512
                if len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                else:
                    token_ids.extend([0] * (max_length - len(token_ids)))

                # Convert to tensor
                token_tensor = torch.tensor([token_ids], dtype=torch.long)

                # Extract features and build topology
                features = self.feature_extractor.extract_entity_info(entity_text)
                topology_data = self.topology_builder.build_complete_topology(
                    [features]
                )

                # Run model inference
                with torch.no_grad():
                    logits = self.model(token_tensor, topology_data=topology_data)
                    probabilities = torch.softmax(logits, dim=1)
                    predicted_idx = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, predicted_idx].item()

                predicted_class = self.class_names[predicted_idx]

                # Add predictions to item properties
                item.properties["predicted_class"] = predicted_class
                item.properties["class_confidence"] = confidence
                item.properties["class_names"] = self.class_names

                # Add top-k predictions
                top_k = min(3, len(self.class_names))
                top_probs, top_indices = torch.topk(probabilities[0], k=top_k)
                item.properties["top_classes"] = [
                    {
                        "class": self.class_names[idx.item()],
                        "confidence": prob.item(),
                    }
                    for prob, idx in zip(top_probs, top_indices)
                ]

                # Add provenance
                item.add_provenance(
                    component_type="enrichment_model",
                    component_name="CADPartClassifier",
                )

                _log.debug(
                    f"Classified entity #{item.entity_id} as '{predicted_class}' "
                    f"(confidence: {confidence:.2f})"
                )

            except Exception as e:
                _log.error(f"Classification failed for item {item.label.text}: {e}")

    def supports_batch_processing(self) -> bool:
        """Classification supports batch processing."""
        return True

    def get_batch_size(self) -> int:
        """Recommended batch size for classification."""
        return 32

    def requires_gpu(self) -> bool:
        """Classification benefits from GPU."""
        return True
