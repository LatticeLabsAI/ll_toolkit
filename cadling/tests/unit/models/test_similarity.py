"""Unit tests for CAD similarity embedding model."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np


class TestCADSimilarityEmbedder:
    """Test CADSimilarityEmbedder model initialization and configuration."""

    def test_model_initialization_without_artifacts(self):
        """Test model initializes without artifacts path."""
        from cadling.models.similarity import CADSimilarityEmbedder

        model = CADSimilarityEmbedder()

        assert model.model is None
        assert model.artifacts_path is None
        assert model.vocab_size == 50000
        assert model.embedding_dim == 512

    def test_model_initialization_with_custom_dim(self):
        """Test model initializes with custom embedding dimension."""
        from cadling.models.similarity import CADSimilarityEmbedder

        model = CADSimilarityEmbedder(embedding_dim=256)

        assert model.embedding_dim == 256

    def test_model_initialization_with_custom_vocab(self):
        """Test model initializes with custom vocab size."""
        from cadling.models.similarity import CADSimilarityEmbedder

        model = CADSimilarityEmbedder(vocab_size=30000, embedding_dim=1024)

        assert model.vocab_size == 30000
        assert model.embedding_dim == 1024

    def test_model_has_required_attributes(self):
        """Test model has all required attributes."""
        from cadling.models.similarity import CADSimilarityEmbedder

        model = CADSimilarityEmbedder()

        assert hasattr(model, "tokenizer")
        assert hasattr(model, "feature_extractor")
        assert hasattr(model, "topology_builder")
        assert hasattr(model, "embedding_dim")
        assert hasattr(model, "vocab_size")

    def test_model_has_required_methods(self):
        """Test model has all required methods."""
        from cadling.models.similarity import CADSimilarityEmbedder

        model = CADSimilarityEmbedder()

        assert callable(model.__call__)
        assert callable(model._generate_embedding)
        assert callable(model.supports_batch_processing)
        assert callable(model.get_batch_size)
        assert callable(model.requires_gpu)
        assert callable(model.cosine_similarity)


class TestCADSimilarityEmbedderMethods:
    """Test CADSimilarityEmbedder helper methods."""

    def test_supports_batch_processing(self):
        """Test that model supports batch processing."""
        from cadling.models.similarity import CADSimilarityEmbedder

        model = CADSimilarityEmbedder()

        assert model.supports_batch_processing() is True

    def test_get_batch_size(self):
        """Test that model returns correct batch size."""
        from cadling.models.similarity import CADSimilarityEmbedder

        model = CADSimilarityEmbedder()

        assert model.get_batch_size() == 64

    def test_requires_gpu(self):
        """Test that model indicates GPU benefit."""
        from cadling.models.similarity import CADSimilarityEmbedder

        model = CADSimilarityEmbedder()

        assert model.requires_gpu() is True

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity for identical vectors."""
        from cadling.models.similarity import CADSimilarityEmbedder

        model = CADSimilarityEmbedder()

        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])

        similarity = model.cosine_similarity(vec1, vec2)

        assert abs(similarity - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity for orthogonal vectors."""
        from cadling.models.similarity import CADSimilarityEmbedder

        model = CADSimilarityEmbedder()

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        similarity = model.cosine_similarity(vec1, vec2)

        assert abs(similarity - 0.0) < 1e-6

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity for opposite vectors."""
        from cadling.models.similarity import CADSimilarityEmbedder

        model = CADSimilarityEmbedder()

        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])

        similarity = model.cosine_similarity(vec1, vec2)

        assert abs(similarity - (-1.0)) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity handles zero vectors."""
        from cadling.models.similarity import CADSimilarityEmbedder

        model = CADSimilarityEmbedder()

        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])

        similarity = model.cosine_similarity(vec1, vec2)

        assert similarity == 0.0


class TestCADSimilarityEmbedderCall:
    """Test CADSimilarityEmbedder __call__ method."""

    def test_call_without_model_loaded(self):
        """Test that __call__ skips when model not loaded."""
        from cadling.models.similarity import CADSimilarityEmbedder
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = CADSimilarityEmbedder()  # No artifacts_path

        # Create mock document and items
        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="CYLINDRICAL_SURFACE",
            text="#1=CYLINDRICAL_SURFACE('',#2,0.5);",
        )

        # Call should return without error and without adding embeddings
        model(doc, [item])

        assert "embedding" not in item.properties
        assert doc.embeddings is None

    def test_call_with_non_step_entity(self):
        """Test that __call__ skips non-STEP entities."""
        from cadling.models.similarity import CADSimilarityEmbedder
        from cadling.datamodel.step import STEPDocument
        from cadling.datamodel.base_models import CADItem, CADItemLabel, CADDocumentOrigin, InputFormat

        model = CADSimilarityEmbedder()

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        # Create non-STEP item
        item = CADItem(
            item_type="mesh",
            label=CADItemLabel(text="Mesh Item"),
        )

        # Mock the model to be loaded
        model.model = Mock()

        # Call should skip this item
        model(doc, [item])

        # Model should not have been called
        model.model.assert_not_called()

    @patch("cadling.models.similarity.torch")
    def test_call_with_mocked_model(self, mock_torch):
        """Test embedding generation with mocked model inference."""
        from cadling.models.similarity import CADSimilarityEmbedder
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = CADSimilarityEmbedder(embedding_dim=512)

        # Mock the STEPForSimilarity model
        mock_model = Mock()
        # Return 512-dimensional embedding
        mock_embedding = torch.randn(1, 512)
        mock_model.return_value = mock_embedding
        model.model = mock_model

        # Mock torch operations
        mock_torch.tensor.return_value = Mock()
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()

        # Create document and item
        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="CYLINDRICAL_SURFACE",
            text="#1=CYLINDRICAL_SURFACE('',#2,0.5);",
        )

        # Run embedding generation
        with torch.no_grad():
            model(doc, [item])

        # Check that embedding was added
        assert "embedding" in item.properties
        assert "embedding_dim" in item.properties
        assert item.properties["embedding_dim"] == 512
        assert isinstance(item.properties["embedding"], list)
        assert len(item.properties["embedding"]) == 512

    def test_document_level_embeddings(self):
        """Test that document-level embeddings are stored."""
        from cadling.models.similarity import CADSimilarityEmbedder
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = CADSimilarityEmbedder(embedding_dim=256)

        # Mock the model
        mock_model = Mock()
        mock_embedding = torch.randn(1, 256)
        mock_model.return_value = mock_embedding
        model.model = mock_model

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        items = [
            STEPEntityItem(
                label=CADItemLabel(text=f"Entity {i}"),
                entity_id=i,
                entity_type="CARTESIAN_POINT",
                text=f"#{i}=CARTESIAN_POINT('',({i},{i},{i}));",
            )
            for i in range(3)
        ]

        with patch("cadling.models.similarity.torch") as mock_torch:
            mock_torch.tensor.return_value = Mock()
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()

            model(doc, items)

        # Check document-level embeddings
        assert doc.embeddings is not None
        assert len(doc.embeddings) == 3
        for emb in doc.embeddings:
            assert isinstance(emb, list)
            assert len(emb) == 256

    def test_provenance_tracking(self):
        """Test that provenance is added to embedded items."""
        from cadling.models.similarity import CADSimilarityEmbedder
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = CADSimilarityEmbedder()

        # Mock the model
        mock_model = Mock()
        mock_model.return_value = torch.randn(1, 512)
        model.model = mock_model

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="PLANE",
            text="#1=PLANE('',#2);",
        )

        with patch("cadling.models.similarity.torch") as mock_torch:
            mock_torch.tensor.return_value = Mock()
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()

            model(doc, [item])

        # Verify provenance was added
        assert len(item.prov) > 0
        assert any(
            prov.component_type == "enrichment_model" and
            prov.component_name == "CADSimilarityEmbedder"
            for prov in item.prov
        )


class TestCADSimilarityEmbedderBatchProcessing:
    """Test CADSimilarityEmbedder batch processing."""

    def test_batch_processing_multiple_items(self):
        """Test generating embeddings for multiple items in a batch."""
        from cadling.models.similarity import CADSimilarityEmbedder
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = CADSimilarityEmbedder()

        # Mock the model
        mock_model = Mock()
        mock_model.return_value = torch.randn(1, 512)
        model.model = mock_model

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        # Create multiple items
        items = [
            STEPEntityItem(
                label=CADItemLabel(text=f"Entity {i}"),
                entity_id=i,
                entity_type="CARTESIAN_POINT",
                text=f"#{i}=CARTESIAN_POINT('',({i},{i},{i}));",
            )
            for i in range(5)
        ]

        with patch("cadling.models.similarity.torch") as mock_torch:
            mock_torch.tensor.return_value = Mock()
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()

            model(doc, items)

        # All items should have embeddings
        for item in items:
            assert "embedding" in item.properties
            assert "embedding_dim" in item.properties


class TestCADSimilarityEmbedderEmbeddings:
    """Test embedding generation and properties."""

    def test_embedding_dimension_matches_config(self):
        """Test that generated embeddings match configured dimension."""
        from cadling.models.similarity import CADSimilarityEmbedder
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        embedding_dims = [128, 256, 512, 1024]

        for dim in embedding_dims:
            model = CADSimilarityEmbedder(embedding_dim=dim)

            # Mock the model
            mock_model = Mock()
            mock_model.return_value = torch.randn(1, dim)
            model.model = mock_model

            doc = STEPDocument(
                name="test.step",
                origin=CADDocumentOrigin(
                    filename="test.step",
                    format=InputFormat.STEP,
                    binary_hash="test_hash",
                ),
                hash="test_hash",
            )

            item = STEPEntityItem(
                label=CADItemLabel(text="Test Entity"),
                entity_id=1,
                entity_type="CYLINDRICAL_SURFACE",
                text="#1=CYLINDRICAL_SURFACE('',#2,0.5);",
            )

            with patch("cadling.models.similarity.torch") as mock_torch:
                mock_torch.tensor.return_value = Mock()
                mock_torch.no_grad.return_value.__enter__ = Mock()
                mock_torch.no_grad.return_value.__exit__ = Mock()

                model(doc, [item])

            assert item.properties["embedding_dim"] == dim
            assert len(item.properties["embedding"]) == dim

    def test_embedding_is_list_for_json_serialization(self):
        """Test that embeddings are stored as lists for JSON compatibility."""
        from cadling.models.similarity import CADSimilarityEmbedder
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = CADSimilarityEmbedder()

        # Mock the model
        mock_model = Mock()
        mock_model.return_value = torch.randn(1, 512)
        model.model = mock_model

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="PLANE",
            text="#1=PLANE('',#2);",
        )

        with patch("cadling.models.similarity.torch") as mock_torch:
            mock_torch.tensor.return_value = Mock()
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()

            model(doc, [item])

        # Embedding should be a list of floats
        assert isinstance(item.properties["embedding"], list)
        for val in item.properties["embedding"]:
            assert isinstance(val, (int, float))


class TestIntegration:
    """Integration tests for CADSimilarityEmbedder."""

    def test_similarity_computation_workflow(self):
        """Test end-to-end similarity computation workflow."""
        from cadling.models.similarity import CADSimilarityEmbedder

        model = CADSimilarityEmbedder()

        # Create two similar embeddings
        emb1 = np.array([0.5, 0.5, 0.5, 0.5])
        emb2 = np.array([0.51, 0.49, 0.5, 0.5])

        # Should have high similarity
        sim = model.cosine_similarity(emb1, emb2)
        assert sim > 0.95

        # Create orthogonal embeddings
        emb3 = np.array([1.0, 0.0, 0.0, 0.0])
        emb4 = np.array([0.0, 1.0, 0.0, 0.0])

        # Should have zero similarity
        sim2 = model.cosine_similarity(emb3, emb4)
        assert abs(sim2) < 0.01

    def test_embedding_different_dimensions(self):
        """Test embedder with various embedding dimensions."""
        from cadling.models.similarity import CADSimilarityEmbedder

        # Test common embedding dimensions
        dims = [64, 128, 256, 512, 768, 1024]

        for dim in dims:
            model = CADSimilarityEmbedder(embedding_dim=dim)
            assert model.embedding_dim == dim
