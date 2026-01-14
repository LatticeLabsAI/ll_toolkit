"""Base classes for enrichment models.

This module provides the foundational interfaces for enrichment models that
add predictions, embeddings, and other computed properties to CADlingDocument
items during the pipeline enrichment stage.

Classes:
    EnrichmentModel: Base class for all enrichment models
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument

_log = logging.getLogger(__name__)


class EnrichmentModel(ABC):
    """Base class for enrichment models.

    Enrichment models add predictions, embeddings, or other computed properties
    to CAD items during the pipeline enrichment stage. This is similar to
    docling's enrichment models but adapted for CAD-specific tasks.

    Examples of enrichment models:
    - CADPartClassifier: Classify parts (bracket, housing, shaft, etc.)
    - CADPropertyPredictor: Predict physical properties (volume, mass)
    - CADSimilarityEmbedder: Generate embeddings for RAG
    - CADCaptioner: Generate text descriptions of parts

    Subclasses must implement:
    - __call__(doc, item_batch): Process items and add predictions

    Example:
        class MyEnrichmentModel(EnrichmentModel):
            def __init__(self, model_path):
                self.model = load_model(model_path)

            def __call__(self, doc, item_batch):
                for item in item_batch:
                    prediction = self.model(item)
                    item.properties["my_prediction"] = prediction
    """

    def __init__(self):
        """Initialize enrichment model."""
        self._initialized = True
        _log.debug(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Enrich items with predictions or computed properties.

        This method is called by the pipeline enrichment stage. It should:
        1. Process each item in the batch
        2. Add predictions to item.properties
        3. Optionally update document-level data (embeddings, topology)

        Args:
            doc: The CADlingDocument being enriched
            item_batch: List of CADItem objects to process

        Example:
            def __call__(self, doc, item_batch):
                for item in item_batch:
                    # Run inference
                    prediction = self.model(item.text)

                    # Add to properties
                    item.properties["prediction"] = prediction
                    item.properties["confidence"] = prediction.confidence

                    # Add provenance
                    item.add_provenance(
                        component_type="enrichment_model",
                        component_name=self.__class__.__name__
                    )
        """
        pass

    def supports_batch_processing(self) -> bool:
        """Whether this model supports batch processing.

        Returns:
            True if model can process multiple items efficiently in batches
        """
        return False

    def get_batch_size(self) -> int:
        """Get recommended batch size for this model.

        Returns:
            Recommended batch size (0 = process all at once)
        """
        return 1

    def requires_gpu(self) -> bool:
        """Whether this model requires GPU acceleration.

        Returns:
            True if model requires GPU, False otherwise
        """
        return False

    def get_model_info(self) -> dict[str, str]:
        """Get information about this model.

        Returns:
            Dictionary with model metadata (name, version, etc.)
        """
        return {
            "model_class": self.__class__.__name__,
            "supports_batch": str(self.supports_batch_processing()),
            "batch_size": str(self.get_batch_size()),
            "requires_gpu": str(self.requires_gpu()),
        }
