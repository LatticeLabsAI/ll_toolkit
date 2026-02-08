"""Unit tests for geotoken integration bridge.

Tests the GeoTokenIntegration class that bridges cadling data models
to geotoken tokenization, including the new tokenize_with_embeddings()
method and return_metadata pattern.
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

from cadling.backend.geotoken_integration import (
    GeoTokenIntegration,
    GeoTokenResult,
)


class TestGeoTokenIntegrationAvailability:
    """Test availability and initialization."""

    def test_available_property_is_bool(self):
        """Test available property returns a boolean."""
        bridge = GeoTokenIntegration()
        # Will be True or False depending on environment
        assert isinstance(bridge.available, bool)

    @patch("cadling.backend.geotoken_integration._try_import_tokenizers")
    def test_unavailable_when_import_fails(self, mock_import):
        """Test bridge reports unavailable when geotoken not installed."""
        mock_import.return_value = (None,) * 5
        bridge = GeoTokenIntegration()
        assert bridge.available is False

    def test_config_stored(self):
        """Test that config is stored on initialization."""
        config = {"source_format": "deepcad", "include_constraints": True}
        bridge = GeoTokenIntegration(config=config)
        assert bridge._config == config

    def test_default_config_empty(self):
        """Test that default config is empty dict."""
        bridge = GeoTokenIntegration()
        assert bridge._config == {}


class TestTokenizeMeshReturnMetadata:
    """Test tokenize_mesh with return_metadata pattern."""

    def test_returns_none_when_unavailable(self):
        """Test tokenize_mesh returns None when geotoken unavailable."""
        bridge = GeoTokenIntegration()
        with patch.object(
            GeoTokenIntegration, "available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False
            result = bridge.tokenize_mesh(MagicMock())
            assert result is None

    def test_metadata_returned_when_unavailable(self):
        """Test return_metadata returns degraded=True when unavailable."""
        bridge = GeoTokenIntegration()
        with patch.object(
            GeoTokenIntegration, "available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False
            result, metadata = bridge.tokenize_mesh(
                MagicMock(), return_metadata=True
            )
            assert result is None
            assert metadata["degraded"] is True
            assert metadata["method"] == "geotoken"
            assert "not available" in metadata["warning"]

    def test_metadata_returned_when_no_mesh_data(self):
        """Test return_metadata when mesh data extraction fails."""
        bridge = GeoTokenIntegration()
        with patch.object(
            GeoTokenIntegration, "available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = True
            mock_item = MagicMock()
            mock_item.to_numpy.side_effect = Exception("no mesh")

            result, metadata = bridge.tokenize_mesh(
                mock_item, return_metadata=True
            )
            assert result is None
            assert metadata["degraded"] is True

    def test_without_return_metadata_returns_result_only(self):
        """Test that without return_metadata, only result is returned."""
        bridge = GeoTokenIntegration()
        with patch.object(
            GeoTokenIntegration, "available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False
            result = bridge.tokenize_mesh(MagicMock(), return_metadata=False)
            assert result is None
            # Should not be a tuple
            assert not isinstance(result, tuple)


class TestTokenizeTopologyReturnMetadata:
    """Test tokenize_topology with return_metadata pattern."""

    def test_returns_none_when_unavailable(self):
        """Test tokenize_topology returns None when unavailable."""
        bridge = GeoTokenIntegration()
        with patch.object(
            GeoTokenIntegration, "available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False
            result = bridge.tokenize_topology(MagicMock())
            assert result is None

    def test_metadata_returned_when_unavailable(self):
        """Test return_metadata returns degraded=True when unavailable."""
        bridge = GeoTokenIntegration()
        with patch.object(
            GeoTokenIntegration, "available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False
            result, metadata = bridge.tokenize_topology(
                MagicMock(), return_metadata=True
            )
            assert result is None
            assert metadata["degraded"] is True
            assert metadata["method"] == "geotoken"


class TestTokenizeSketchReturnMetadata:
    """Test tokenize_sketch with return_metadata pattern."""

    def test_returns_none_when_unavailable(self):
        """Test tokenize_sketch returns None when unavailable."""
        bridge = GeoTokenIntegration()
        with patch.object(
            GeoTokenIntegration, "available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False
            result = bridge.tokenize_sketch(MagicMock())
            assert result is None

    def test_metadata_returned_when_unavailable(self):
        """Test return_metadata returns degraded=True when unavailable."""
        bridge = GeoTokenIntegration()
        with patch.object(
            GeoTokenIntegration, "available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False
            result, metadata = bridge.tokenize_sketch(
                MagicMock(), return_metadata=True
            )
            assert result is None
            assert metadata["degraded"] is True
            assert metadata["method"] == "geotoken"

    def test_metadata_when_no_commands(self):
        """Test return_metadata when command extraction fails."""
        bridge = GeoTokenIntegration()
        with patch.object(
            GeoTokenIntegration, "available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = True
            # Mock item without commands
            mock_item = MagicMock(spec=[])
            mock_item.properties = {}

            result, metadata = bridge.tokenize_sketch(
                mock_item, return_metadata=True
            )
            assert result is None
            assert metadata["degraded"] is True
            assert "no command" in metadata["warning"]


class TestTokenizeWithEmbeddings:
    """Test embedding-aware tokenization."""

    def test_returns_none_when_unavailable(self):
        """Test returns None when geotoken unavailable."""
        bridge = GeoTokenIntegration()
        with patch.object(
            GeoTokenIntegration, "available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False
            result = bridge.tokenize_with_embeddings(MagicMock())
            assert result is None

    def test_metadata_returned_when_unavailable(self):
        """Test return_metadata returns degraded=True when unavailable."""
        bridge = GeoTokenIntegration()
        with patch.object(
            GeoTokenIntegration, "available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False
            result, metadata = bridge.tokenize_with_embeddings(
                MagicMock(), return_metadata=True
            )
            assert result is None
            assert metadata["degraded"] is True
            assert metadata["embeddings_included"] is False

    def test_metadata_when_no_mesh_data(self):
        """Test return_metadata when mesh data extraction fails."""
        bridge = GeoTokenIntegration()
        with patch.object(
            GeoTokenIntegration, "available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = True
            mock_item = MagicMock(spec=[])
            mock_item.properties = {}

            result, metadata = bridge.tokenize_with_embeddings(
                mock_item, return_metadata=True
            )
            assert result is None
            assert metadata["degraded"] is True
            assert "no mesh" in metadata["warning"]

    def test_embeddings_param_accepted(self):
        """Test that embeddings parameter is accepted."""
        bridge = GeoTokenIntegration()
        with patch.object(
            GeoTokenIntegration, "available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False
            embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

            # Should not raise even when unavailable
            result = bridge.tokenize_with_embeddings(
                MagicMock(), embeddings=embeddings
            )
            assert result is None


class TestReturnMetadataConsistency:
    """Test that return_metadata pattern is consistent across methods."""

    def test_all_methods_have_return_metadata(self):
        """Verify all tokenize methods support return_metadata."""
        bridge = GeoTokenIntegration()

        # Check method signatures
        import inspect

        for method_name in [
            "tokenize_mesh",
            "tokenize_topology",
            "tokenize_sketch",
            "tokenize_with_embeddings",
        ]:
            method = getattr(bridge, method_name)
            sig = inspect.signature(method)
            assert "return_metadata" in sig.parameters, (
                f"{method_name} missing return_metadata parameter"
            )

    def test_metadata_format_consistent(self):
        """Test that metadata dict format is consistent across methods."""
        bridge = GeoTokenIntegration()
        with patch.object(
            GeoTokenIntegration, "available", new_callable=PropertyMock
        ) as mock_available:
            mock_available.return_value = False

            _, mesh_meta = bridge.tokenize_mesh(MagicMock(), return_metadata=True)
            _, topo_meta = bridge.tokenize_topology(
                MagicMock(), return_metadata=True
            )
            _, sketch_meta = bridge.tokenize_sketch(
                MagicMock(), return_metadata=True
            )
            _, emb_meta = bridge.tokenize_with_embeddings(
                MagicMock(), return_metadata=True
            )

            # All should have these keys
            for meta in [mesh_meta, topo_meta, sketch_meta, emb_meta]:
                assert "degraded" in meta
                assert "method" in meta
                assert "warning" in meta


class TestGeoTokenResult:
    """Test GeoTokenResult dataclass."""

    def test_default_values(self):
        """Test GeoTokenResult has sensible defaults."""
        result = GeoTokenResult()
        assert result.token_sequences == {}
        assert result.vocabulary is None
        assert result.metadata == {}
        assert result.errors == {}

    def test_with_values(self):
        """Test GeoTokenResult with provided values."""
        result = GeoTokenResult(
            token_sequences={"item1": "tokens"},
            vocabulary="vocab",
            metadata={"duration_ms": 100},
            errors={"item2": "failed"},
        )
        assert result.token_sequences == {"item1": "tokens"}
        assert result.vocabulary == "vocab"
        assert result.metadata == {"duration_ms": 100}
        assert result.errors == {"item2": "failed"}


class TestGeometryExtraction:
    """Test internal geometry extraction helpers."""

    def test_extract_mesh_data_from_to_numpy(self):
        """Test mesh extraction via to_numpy method."""
        mock_item = MagicMock()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mock_item.to_numpy.return_value = (vertices, faces)

        result = GeoTokenIntegration._extract_mesh_data(mock_item)

        assert result is not None
        np.testing.assert_array_equal(result[0], vertices)
        np.testing.assert_array_equal(result[1], faces)

    def test_extract_mesh_data_from_properties(self):
        """Test mesh extraction from properties dict."""
        mock_item = MagicMock(spec=[])  # No to_numpy
        mock_item.properties = {
            "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "faces": [[0, 1, 2]],
        }

        result = GeoTokenIntegration._extract_mesh_data(mock_item)

        assert result is not None
        assert result[0].shape == (3, 3)
        assert result[1].shape == (1, 3)

    def test_extract_mesh_data_returns_none_when_missing(self):
        """Test mesh extraction returns None when no mesh data."""
        mock_item = MagicMock(spec=[])
        mock_item.properties = {}

        result = GeoTokenIntegration._extract_mesh_data(mock_item)

        assert result is None

    def test_extract_commands_from_method(self):
        """Test command extraction via to_geotoken_commands method."""
        mock_item = MagicMock()
        commands = [{"type": "line", "start": [0, 0], "end": [1, 1]}]
        mock_item.to_geotoken_commands.return_value = commands

        result = GeoTokenIntegration._extract_commands(mock_item)

        assert result == commands

    def test_extract_commands_from_properties(self):
        """Test command extraction from properties dict."""
        mock_item = MagicMock(spec=[])
        commands = [{"type": "arc", "center": [0, 0], "radius": 1}]
        mock_item.properties = {"command_sequence": commands}

        result = GeoTokenIntegration._extract_commands(mock_item)

        assert result == commands

    def test_extract_constraints_from_method(self):
        """Test constraint extraction via to_geotoken_constraints method."""
        mock_item = MagicMock()
        constraints = [{"type": "perpendicular", "entities": [0, 1]}]
        mock_item.to_geotoken_constraints.return_value = constraints

        result = GeoTokenIntegration._extract_constraints(mock_item)

        assert result == constraints

    def test_extract_constraints_from_properties(self):
        """Test constraint extraction from properties dict."""
        mock_item = MagicMock(spec=[])
        constraints = [{"type": "parallel", "entities": [2, 3]}]
        mock_item.properties = {"constraints": constraints}

        result = GeoTokenIntegration._extract_constraints(mock_item)

        assert result == constraints


@pytest.mark.skipif(
    not GeoTokenIntegration().available,
    reason="geotoken not installed",
)
class TestWithGeotoken:
    """Integration tests that require geotoken to be installed."""

    def test_tokenize_mesh_succeeds(self):
        """Test mesh tokenization with real geotoken."""
        from cadling.datamodel.stl import MeshItem, CADItemLabel

        mesh = MeshItem(
            label=CADItemLabel(text="test"),
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            facets=[[0, 1, 2]],
        )

        bridge = GeoTokenIntegration()
        result = bridge.tokenize_mesh(mesh)

        assert result is not None

    def test_tokenize_mesh_with_metadata(self):
        """Test mesh tokenization returns valid metadata."""
        from cadling.datamodel.stl import MeshItem, CADItemLabel

        mesh = MeshItem(
            label=CADItemLabel(text="test"),
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            facets=[[0, 1, 2]],
        )

        bridge = GeoTokenIntegration()
        result, metadata = bridge.tokenize_mesh(mesh, return_metadata=True)

        assert result is not None
        assert metadata["degraded"] is False
        assert metadata["method"] == "geotoken"

    def test_tokenize_with_embeddings_stores_embedding_data(self):
        """Test embeddings are quantized and stored in metadata."""
        from cadling.datamodel.stl import MeshItem, CADItemLabel

        mesh = MeshItem(
            label=CADItemLabel(text="test"),
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            facets=[[0, 1, 2]],
        )
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

        bridge = GeoTokenIntegration()
        result, metadata = bridge.tokenize_with_embeddings(
            mesh, embeddings=embeddings, return_metadata=True
        )

        assert result is not None
        assert metadata["embeddings_included"] is True
        assert "embeddings" in result.metadata
        assert "embedding_params" in result.metadata

    def test_vocabulary_created(self):
        """Test vocabulary can be created."""
        bridge = GeoTokenIntegration()
        vocab = bridge._create_vocabulary()

        assert vocab is not None
