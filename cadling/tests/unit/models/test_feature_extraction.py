"""Unit tests for feature extraction enrichment model.

Tests the FeatureExtractionModel that auto-populates geometric features
on CADItems during pipeline enrichment.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np


class TestFeatureExtractionModelImports:
    """Test module imports."""

    def test_module_imports(self):
        """Test that the module imports successfully."""
        from cadling.models.feature_extraction import FeatureExtractionModel
        from cadling.models.base_model import EnrichmentModel

        assert FeatureExtractionModel is not None
        assert issubclass(FeatureExtractionModel, EnrichmentModel)


class TestFeatureExtractionModelInit:
    """Test FeatureExtractionModel initialization."""

    def test_default_init(self):
        """Test default initialization."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel()

        assert model.extract_face_features is True
        assert model.extract_edge_features is True
        assert model.extract_uv_grids is False
        assert model.extract_adjacency is True
        assert model.use_cache is True
        assert model.uv_grid_size == 10

    def test_custom_init(self):
        """Test custom initialization."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel(
            extract_face_features=False,
            extract_edge_features=False,
            extract_uv_grids=True,
            extract_adjacency=False,
            use_cache=False,
            uv_grid_size=5,
        )

        assert model.extract_face_features is False
        assert model.extract_edge_features is False
        assert model.extract_uv_grids is True
        assert model.extract_adjacency is False
        assert model.use_cache is False
        assert model.uv_grid_size == 5

    def test_cache_created_when_enabled(self):
        """Test that cache is created when enabled."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel(use_cache=True)
        assert model._cache is not None

    def test_cache_not_created_when_disabled(self):
        """Test that cache is not created when disabled."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel(use_cache=False)
        assert model._cache is None


class TestFeatureExtractionModelMethods:
    """Test FeatureExtractionModel methods."""

    def test_supports_batch_processing(self):
        """Test supports_batch_processing returns True."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel()
        assert model.supports_batch_processing() is True

    def test_get_batch_size(self):
        """Test get_batch_size returns 0 (all at once)."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel()
        assert model.get_batch_size() == 0

    def test_requires_gpu(self):
        """Test requires_gpu returns False."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel()
        assert model.requires_gpu() is False

    def test_get_model_info(self):
        """Test get_model_info returns complete info."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel()
        info = model.get_model_info()

        assert "model_class" in info
        assert info["model_class"] == "FeatureExtractionModel"
        assert "extract_face_features" in info
        assert "extract_edge_features" in info
        assert "extract_uv_grids" in info
        assert "use_cache" in info


class TestFeatureExtractionModelCall:
    """Test FeatureExtractionModel __call__ method."""

    def test_call_no_shape(self):
        """Test __call__ with document without OCC shape."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel(use_cache=False)

        mock_doc = MagicMock()
        mock_doc._occ_shape = None
        mock_doc._backend = None
        mock_items = []

        # Should not raise
        model(mock_doc, mock_items)

    def test_get_occ_shape_from_doc(self):
        """Test _get_occ_shape retrieves shape from document."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel(use_cache=False)

        mock_doc = MagicMock()
        mock_shape = MagicMock()
        mock_doc._occ_shape = mock_shape

        result = model._get_occ_shape(mock_doc)
        assert result is mock_shape

    def test_get_occ_shape_from_backend(self):
        """Test _get_occ_shape retrieves shape from backend."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel(use_cache=False)

        mock_doc = MagicMock()
        mock_doc._occ_shape = None
        mock_shape = MagicMock()
        mock_doc._backend._occ_shape = mock_shape

        result = model._get_occ_shape(mock_doc)
        assert result is mock_shape

    def test_get_occ_shape_from_backend_method(self):
        """Test _get_occ_shape retrieves shape via get_shape()."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel(use_cache=False)

        mock_doc = MagicMock()
        mock_doc._occ_shape = None
        mock_shape = MagicMock()
        mock_doc._backend._occ_shape = None
        mock_doc._backend.get_shape.return_value = mock_shape

        result = model._get_occ_shape(mock_doc)
        assert result is mock_shape

    def test_get_occ_shape_none(self):
        """Test _get_occ_shape returns None when not available."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel(use_cache=False)

        mock_doc = MagicMock()
        mock_doc._occ_shape = None
        mock_doc._backend = None

        result = model._get_occ_shape(mock_doc)
        assert result is None


class TestFeatureExtractionModelExtraction:
    """Test feature extraction methods."""

    def test_extract_features_empty(self):
        """Test _extract_features with empty wrapper results."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel(
            extract_face_features=False,
            extract_edge_features=False,
            extract_uv_grids=False,
            extract_adjacency=False,
            use_cache=False,
        )

        mock_shape = MagicMock()

        with patch("cadling.lib.occ_wrapper.OCCShape") as MockOCCShape:
            mock_wrapper = MagicMock()
            mock_wrapper.num_faces.return_value = 0
            mock_wrapper.num_edges.return_value = 0
            mock_wrapper.num_vertices.return_value = 0
            mock_wrapper.volume.return_value = 0.0
            mock_wrapper.surface_area.return_value = 0.0
            mock_wrapper.bbox.return_value = ([0, 0, 0], [1, 1, 1])
            MockOCCShape.return_value = mock_wrapper

            features = model._extract_features(mock_shape)

        assert features["num_faces"] == 0
        assert features["num_edges"] == 0
        assert features["volume"] == 0.0

    def test_apply_features_to_items(self):
        """Test _apply_features_to_items updates document properties."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel(use_cache=False)

        features = {
            "num_faces": 6,
            "num_edges": 12,
            "num_vertices": 8,
            "volume": 1.0,
            "surface_area": 6.0,
            "bbox_min": [0, 0, 0],
            "bbox_max": [1, 1, 1],
            "bbox_dimensions": [1, 1, 1],
            "adjacency": {0: [1, 2]},
        }

        mock_doc = MagicMock()
        mock_doc.properties = {}
        mock_items = []

        model._apply_features_to_items(features, mock_doc, mock_items)

        assert "geometry_features" in mock_doc.properties
        assert mock_doc.properties["geometry_features"]["num_faces"] == 6
        assert mock_doc.properties["geometry_features"]["volume"] == 1.0

    def test_apply_features_cached_format(self):
        """Test _apply_features_to_items handles cached format."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel(use_cache=False)

        # Cached format has features nested
        features = {
            "features": {
                "num_faces": 6,
                "num_edges": 12,
                "num_vertices": 8,
                "volume": 1.0,
                "surface_area": 6.0,
                "bbox_min": [0, 0, 0],
                "bbox_max": [1, 1, 1],
                "bbox_dimensions": [1, 1, 1],
            }
        }

        mock_doc = MagicMock()
        mock_doc.properties = {}
        mock_items = []

        model._apply_features_to_items(features, mock_doc, mock_items)

        assert "geometry_features" in mock_doc.properties
        assert mock_doc.properties["geometry_features"]["num_faces"] == 6


class TestFeatureExtractionModelCaching:
    """Test feature caching functionality."""

    def test_compute_cache_key(self):
        """Test _compute_cache_key generates valid key."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel(use_cache=True)

        mock_doc = MagicMock()
        mock_doc.name = "test_model.step"
        mock_doc.input = None

        key = model._compute_cache_key(mock_doc)

        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex digest

    def test_caching_flow(self):
        """Test that caching is used when enabled."""
        from cadling.models.feature_extraction import FeatureExtractionModel

        model = FeatureExtractionModel(use_cache=True)

        # Mock the cache
        mock_cache = MagicMock()
        mock_cache.compute_key.return_value = "test_key"
        mock_cache.get.return_value = {
            "features": {
                "num_faces": 6,
                "num_edges": 12,
                "num_vertices": 8,
                "volume": 1.0,
                "surface_area": 6.0,
                "bbox_min": [0, 0, 0],
                "bbox_max": [1, 1, 1],
                "bbox_dimensions": [1, 1, 1],
            }
        }
        model._cache = mock_cache

        mock_doc = MagicMock()
        mock_doc._occ_shape = MagicMock()
        mock_doc.name = "test.step"
        mock_doc.input = None
        mock_doc.properties = {}

        mock_items = []

        model(mock_doc, mock_items)

        # Should have used cache
        mock_cache.get.assert_called_once()
        # Should have applied features
        assert "geometry_features" in mock_doc.properties
