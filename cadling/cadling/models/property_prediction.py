"""CAD property prediction model.

This module provides models for predicting physical properties of CAD parts
using ll_stepnet.

Classes:
    CADPropertyPredictor: Predict physical properties (volume, mass, etc.)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from cadling.datamodel.base_models import CADItem, CADlingDocument
from cadling.models.base_model import EnrichmentModel

if TYPE_CHECKING:
    from cadling.datamodel.step import STEPEntityItem

_log = logging.getLogger(__name__)

import torch
from stepnet import (
    STEPForPropertyPrediction,
    STEPTokenizer,
    STEPFeatureExtractor,
    STEPTopologyBuilder,
)


class CADPropertyPredictor(EnrichmentModel):
    """CAD property prediction model.

    Predicts physical properties of CAD parts using ll_stepnet's
    STEPForPropertyPrediction model. Properties include:
    - Volume (cubic units)
    - Surface area (square units)
    - Mass (with material density)
    - Bounding box dimensions
    - Center of mass

    The model uses STEP entities, topology, and geometric features.

    Attributes:
        model: STEPForPropertyPrediction model instance
        artifacts_path: Path to model artifacts
        material_density: Default material density (kg/m³)

    Example:
        predictor = CADPropertyPredictor(Path("models/properties.pt"))
        result = converter.convert(
            "part.step",
            pipeline_options=PipelineOptions(
                enrichment_models=[predictor]
            )
        )
        for item in result.document.items:
            if "predicted_volume" in item.properties:
                print(f"Volume: {item.properties['predicted_volume']} m³")
    """

    def __init__(
        self,
        artifacts_path: Optional[Path] = None,
        material_density: float = 7850.0,  # Steel density in kg/m³
        property_names: Optional[list[str]] = None,
        vocab_size: int = 50000,
        output_dim: int = 1024,
    ):
        """Initialize CAD property predictor.

        Args:
            artifacts_path: Path to model checkpoint (.pt file)
            material_density: Material density in kg/m³ (default: steel)
            property_names: List of property names to predict
            vocab_size: Vocabulary size for tokenizer
            output_dim: Model output dimension
        """
        super().__init__()

        self.artifacts_path = artifacts_path
        self.material_density = material_density
        self.property_names = property_names or [
            "volume_mm3",
            "surface_area_mm2",
            "mass_g",
            "bbox_x_mm",
            "bbox_y_mm",
            "bbox_z_mm",
        ]
        self.vocab_size = vocab_size
        self.output_dim = output_dim

        # Initialize tokenizer and extractors
        self.tokenizer = STEPTokenizer(vocab_size=vocab_size)
        self.feature_extractor = STEPFeatureExtractor()
        self.topology_builder = STEPTopologyBuilder()

        # Load model if artifacts_path provided
        if artifacts_path:
            try:
                # Create model instance
                self.model = STEPForPropertyPrediction(
                    vocab_size=vocab_size,
                    num_properties=len(self.property_names),
                    output_dim=output_dim,
                )

                # Load checkpoint
                checkpoint = torch.load(
                    str(artifacts_path), map_location="cpu", weights_only=False
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()

                _log.info(f"Loaded property prediction model from {artifacts_path}")
            except Exception as e:
                _log.error(f"Failed to load property prediction model: {e}")
                self.model = None
        else:
            self.model = None
            _log.info("Property prediction model path not provided")

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Predict properties for CAD items.

        Args:
            doc: CADlingDocument being enriched
            item_batch: List of CADItem objects to process
        """
        from cadling.datamodel.step import STEPEntityItem

        if not self.model:
            _log.debug("Property prediction skipped: model not available")
            return

        for item in item_batch:
            # Only predict for STEP entities with geometric meaning
            if not isinstance(item, STEPEntityItem):
                continue

            try:
                # Predict properties
                predictions = self._predict_properties(item, doc)

                # Add predictions to item properties
                for prop_name, prop_value in predictions.items():
                    item.properties[prop_name] = prop_value

                # Add provenance
                item.add_provenance(
                    component_type="enrichment_model",
                    component_name="CADPropertyPredictor",
                )

                _log.debug(
                    f"Predicted properties for entity #{item.entity_id}: "
                    f"{list(predictions.keys())}"
                )

            except Exception as e:
                _log.error(
                    f"Property prediction failed for item {item.label.text}: {e}"
                )

    def _predict_properties(
        self, item: STEPEntityItem, doc: CADlingDocument
    ) -> dict[str, float]:
        """Predict properties for a STEP entity.

        Args:
            item: STEP entity item
            doc: Parent document

        Returns:
            Dictionary of predicted properties
        """
        predictions = {}

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
        topology_data = self.topology_builder.build_complete_topology([features])

        # Run model inference
        with torch.no_grad():
            property_values = self.model(token_tensor, topology_data=topology_data)

        # Convert predictions to dictionary
        for i, prop_name in enumerate(self.property_names):
            if i < property_values.shape[1]:
                predictions[f"predicted_{prop_name}"] = property_values[0, i].item()

        # Add derived properties
        if "predicted_volume_mm3" in predictions:
            # Convert volume to mass (assuming material density)
            volume_m3 = predictions["predicted_volume_mm3"] / 1e9  # mm³ to m³
            predictions["predicted_mass_kg"] = volume_m3 * self.material_density

        _log.debug(f"Predicted {len(predictions)} properties using model")

        return predictions

    def supports_batch_processing(self) -> bool:
        """Property prediction supports batch processing."""
        return True

    def get_batch_size(self) -> int:
        """Recommended batch size for property prediction."""
        return 16

    def requires_gpu(self) -> bool:
        """Property prediction benefits from GPU."""
        return True
