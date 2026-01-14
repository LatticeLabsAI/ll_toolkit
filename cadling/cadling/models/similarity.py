"""CAD similarity and embedding model.

This module provides models for generating embeddings of CAD parts
for similarity search and RAG applications using ll_stepnet.

Classes:
    CADSimilarityEmbedder: Generate embeddings for CAD parts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from cadling.models.base_model import EnrichmentModel

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument
    from cadling.datamodel.step import STEPEntityItem

_log = logging.getLogger(__name__)

import torch
from stepnet import (
    STEPForSimilarity,
    STEPTokenizer,
    STEPFeatureExtractor,
    STEPTopologyBuilder,
)


class CADSimilarityEmbedder(EnrichmentModel):
    """CAD similarity embedding model.

    Generates dense vector embeddings for CAD parts using ll_stepnet's
    STEPForSimilarity model. These embeddings can be used for:
    - Similarity search (find similar parts)
    - RAG (Retrieval-Augmented Generation)
    - Clustering and analysis
    - Part recommendations

    The model uses STEP entities, topology, and geometric features to
    create embeddings that capture geometric and structural similarity.

    Attributes:
        model: STEPForSimilarity model instance
        embedding_dim: Dimension of output embeddings
        artifacts_path: Path to model artifacts

    Example:
        embedder = CADSimilarityEmbedder(Path("models/similarity.pt"))
        result = converter.convert(
            "part.step",
            pipeline_options=PipelineOptions(
                enrichment_models=[embedder]
            )
        )

        # Access embeddings
        for item in result.document.items:
            if "embedding" in item.properties:
                embedding = item.properties["embedding"]
                # Use for similarity search or RAG
    """

    def __init__(
        self,
        artifacts_path: Optional[Path] = None,
        embedding_dim: int = 512,
        vocab_size: int = 50000,
    ):
        """Initialize CAD similarity embedder.

        Args:
            artifacts_path: Path to model checkpoint (.pt file)
            embedding_dim: Dimension of output embeddings
            vocab_size: Vocabulary size for tokenizer
        """
        super().__init__()

        self.artifacts_path = artifacts_path
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # Initialize tokenizer and extractors
        self.tokenizer = STEPTokenizer(vocab_size=vocab_size)
        self.feature_extractor = STEPFeatureExtractor()
        self.topology_builder = STEPTopologyBuilder()

        # Load model if artifacts_path provided
        if artifacts_path:
            try:
                # Create model instance
                self.model = STEPForSimilarity(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                )

                # Load checkpoint
                checkpoint = torch.load(
                    str(artifacts_path), map_location="cpu", weights_only=False
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()

                _log.info(f"Loaded similarity model from {artifacts_path}")
            except Exception as e:
                _log.error(f"Failed to load similarity model: {e}")
                self.model = None
        else:
            self.model = None
            _log.info("Similarity model path not provided")

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Generate embeddings for CAD items.

        Args:
            doc: CADlingDocument being enriched
            item_batch: List of CADItem objects to embed
        """
        from cadling.datamodel.step import STEPEntityItem

        if not self.model:
            _log.debug("Embedding generation skipped: model not available")
            return

        embeddings_list = []

        for item in item_batch:
            # Only embed STEP entities
            if not isinstance(item, STEPEntityItem):
                continue

            try:
                # Generate embedding
                embedding = self._generate_embedding(item, doc)

                if embedding is not None:
                    # Add embedding to item properties
                    item.properties["embedding"] = embedding.tolist()
                    item.properties["embedding_dim"] = len(embedding)
                    embeddings_list.append(embedding)

                    # Add provenance
                    item.add_provenance(
                        component_type="enrichment_model",
                        component_name="CADSimilarityEmbedder",
                    )

                    _log.debug(
                        f"Generated embedding for entity #{item.entity_id} "
                        f"(dim: {len(embedding)})"
                    )

            except Exception as e:
                _log.error(f"Embedding generation failed for item {item.label.text}: {e}")

        # Store document-level embeddings
        if embeddings_list:
            # Convert to list of lists for JSON serialization
            doc.embeddings = [emb.tolist() for emb in embeddings_list]
            _log.info(
                f"Generated {len(embeddings_list)} embeddings for document "
                f"(dim: {self.embedding_dim})"
            )

    def _generate_embedding(
        self, item: STEPEntityItem, doc: CADlingDocument
    ) -> Optional[np.ndarray]:
        """Generate embedding for a STEP entity.

        Args:
            item: STEP entity item
            doc: Parent document

        Returns:
            Embedding vector as numpy array, or None if generation fails
        """
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
            embedding_tensor = self.model(token_tensor, topology_data=topology_data)

        # Convert to numpy array (already L2 normalized by model)
        embedding = embedding_tensor[0].cpu().numpy()

        _log.debug(f"Generated embedding with dimension {len(embedding)}")

        return embedding

    def supports_batch_processing(self) -> bool:
        """Embedding generation supports batch processing."""
        return True

    def get_batch_size(self) -> int:
        """Recommended batch size for embedding generation."""
        return 64

    def requires_gpu(self) -> bool:
        """Embedding generation benefits from GPU."""
        return True

    def cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
