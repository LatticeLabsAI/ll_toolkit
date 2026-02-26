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
from typing import TYPE_CHECKING, Optional

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

    def _get_backend_resource(self, doc: CADlingDocument, resource_name: str):
        """Get a resource from the document's backend using multiple attribute patterns.

        Tries the following patterns in order:
        1. backend.{resource_name} (e.g., backend.shape)
        2. backend._{resource_name} (e.g., backend._shape)
        3. backend.load_{resource_name}() (e.g., backend.load_shape())
        4. backend.get_{resource_name}() (e.g., backend.get_shape())

        Args:
            doc: Document with backend
            resource_name: Base name of the resource (e.g., "shape", "mesh")

        Returns:
            The resource if found, None otherwise
        """
        if not hasattr(doc, '_backend') or doc._backend is None:
            _log.debug("No backend available for %s loading", resource_name)
            return None

        backend = doc._backend

        attr_patterns = [
            (resource_name, False),
            (f"_{resource_name}", False),
            (f"load_{resource_name}", True),
            (f"get_{resource_name}", True),
        ]

        try:
            for attr_name, is_method in attr_patterns:
                if hasattr(backend, attr_name):
                    attr = getattr(backend, attr_name)
                    if is_method:
                        if callable(attr):
                            result = attr()
                            if result is not None:
                                _log.debug("Loaded %s from backend.%s()", resource_name, attr_name)
                                return result
                    else:
                        if attr is not None:
                            _log.debug("Loaded %s from backend.%s", resource_name, attr_name)
                            return attr

            _log.debug(
                "Backend %s does not provide %s (tried: %s)",
                type(backend).__name__, resource_name,
                ", ".join(p[0] for p in attr_patterns),
            )
            return None

        except Exception as e:
            _log.error("Failed to load %s from backend: %s", resource_name, e)
            return None

    def _get_step_text(self, doc: CADlingDocument, item: CADItem) -> "Optional[str]":
        """Get STEP text from document or item.

        Shared utility for enrichment models that need to parse STEP text
        as a fallback when OCC shapes are unavailable.

        Args:
            doc: Document containing the item
            item: Item to get text for

        Returns:
            STEP text or None
        """
        # Try item text
        if hasattr(item, 'text') and item.text:
            return item.text

        # Try document raw content
        if hasattr(doc, 'raw_content') and doc.raw_content:
            return doc.raw_content

        # Try backend content
        if hasattr(doc, '_backend') and doc._backend:
            if hasattr(doc._backend, 'content'):
                return doc._backend.content

        return None

    def _get_shape_for_item(
        self, doc: CADlingDocument, item: CADItem
    ) -> "Optional[any]":
        """Get shape object for analysis or validation.

        Shared utility for enrichment models that need to access the
        underlying OCC shape or trimesh object.

        Args:
            doc: Document containing the item
            item: Item to get shape for

        Returns:
            Shape object (OCC shape or trimesh), or None
        """
        # Check if item has shape stored
        if hasattr(item, "_shape") and item._shape is not None:
            return item._shape

        # Try to load from backend based on format
        format_str = str(doc.format).lower()

        has_pythonocc = getattr(self, 'has_pythonocc', False)
        has_trimesh = getattr(self, 'has_trimesh', False)

        if format_str in ["step", "iges", "brep"] and has_pythonocc:
            return self._get_backend_resource(doc, "shape")
        elif format_str == "stl" and has_trimesh:
            return self._get_backend_resource(doc, "mesh")

        return None

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
