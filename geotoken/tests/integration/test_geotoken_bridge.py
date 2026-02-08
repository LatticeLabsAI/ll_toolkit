"""Integration tests for the GeoTokenIntegration bridge module.

Tests the centralised bridge (``cadling.backend.geotoken_integration``)
which is the single entry-point for all cadling → geotoken tokenization.

The tests cover:
- Bridge instantiation / graceful degradation
- Mesh tokenization via the bridge
- Topology graph tokenization
- Sketch / command tokenization
- Encode → decode roundtrip
- Roundtrip quality validation
- SDG vocabulary alignment
- UV-grid quantizer
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bridge():
    """Create a GeoTokenIntegration instance."""
    from cadling.backend.geotoken_integration import GeoTokenIntegration

    return GeoTokenIntegration()


@pytest.fixture
def mock_mesh_item(cube_mesh):
    """Build a minimal mock CADItem with mesh data."""
    vertices, faces = cube_mesh

    item = MagicMock()
    item.item_id = "test_mesh_001"
    item.to_numpy.return_value = (
        vertices.astype(np.float32),
        faces.astype(np.int64),
    )
    # No command / constraint methods
    del item.to_geotoken_commands
    del item.to_geotoken_constraints
    return item


@pytest.fixture
def mock_sketch_item():
    """Build a minimal mock CADItem with sketch commands."""
    item = MagicMock()
    item.item_id = "test_sketch_001"

    # No mesh data
    del item.to_numpy

    # Command sequence (line + arc)
    item.to_geotoken_commands.return_value = [
        {"type": "Line", "start": [0, 0], "end": [10, 0]},
        {"type": "Arc", "start": [10, 0], "end": [10, 10], "mid": [15, 5]},
        {"type": "Line", "start": [10, 10], "end": [0, 10]},
        {"type": "Line", "start": [0, 10], "end": [0, 0]},
    ]
    item.to_geotoken_constraints.return_value = [
        {"type": "perpendicular", "entity_a": 0, "entity_b": 1},
    ]
    return item


@pytest.fixture
def mock_document(mock_mesh_item, mock_sketch_item):
    """Build a minimal mock CADlingDocument."""
    doc = MagicMock()
    doc.items = [mock_mesh_item, mock_sketch_item]
    doc.topology = None  # No topology graph
    doc.segments = []
    doc.name = "test_doc"
    doc.hash = "abc123"
    return doc


@pytest.fixture
def mock_topology():
    """Build a minimal mock TopologyGraph."""
    topo = MagicMock()

    # 4 nodes, 48-dim features each (enough for UV extraction)
    node_feats = np.random.rand(4, 48).astype(np.float32)
    # Set UV bounds at known indices [30:34] and xyz stats at [34:40]
    for i in range(4):
        node_feats[i, 30] = 0.0   # u_min
        node_feats[i, 31] = 1.0   # u_max
        node_feats[i, 32] = 0.0   # v_min
        node_feats[i, 33] = 1.0   # v_max
        node_feats[i, 34:37] = [0.5, 0.5, 0.5]  # xyz mean
        node_feats[i, 37:40] = [0.1, 0.1, 0.1]  # xyz std

    topo.to_numpy_node_features.return_value = node_feats

    # 6 edges (complete graph on 4 nodes minus self-loops minus duplicates)
    edge_index = np.array([
        [0, 0, 0, 1, 1, 2],
        [1, 2, 3, 2, 3, 3],
    ], dtype=np.int64)
    topo.to_edge_index.return_value = edge_index

    # 16-dim edge features
    edge_feats = np.random.rand(6, 16).astype(np.float32)
    topo.to_numpy_edge_features.return_value = edge_feats

    return topo


# ---------------------------------------------------------------------------
# Tests: Bridge instantiation
# ---------------------------------------------------------------------------


class TestBridgeInstantiation:
    """Tests for creating GeoTokenIntegration instances."""

    def test_bridge_available(self, bridge):
        """Bridge should be available when geotoken is importable."""
        # geotoken is in the same monorepo, so it should be available
        assert bridge.available is True

    def test_bridge_with_config(self):
        """Bridge accepts an optional config dict."""
        from cadling.backend.geotoken_integration import GeoTokenIntegration

        bridge = GeoTokenIntegration(config={
            "source_format": "deepcad",
            "include_constraints": True,
        })
        assert bridge.available is True

    def test_bridge_graceful_degradation(self):
        """Bridge should degrade gracefully if geotoken is missing."""
        from cadling.backend.geotoken_integration import GeoTokenIntegration

        # Simulate geotoken not being installed
        with patch.dict(sys.modules, {
            "geotoken": None,
            "geotoken.tokenizer": None,
            "geotoken.tokenizer.geo_tokenizer": None,
            "geotoken.tokenizer.graph_tokenizer": None,
            "geotoken.tokenizer.command_tokenizer": None,
            "geotoken.tokenizer.vocabulary": None,
            "geotoken.tokenizer.token_types": None,
        }):
            # Need to re-import to test with patched modules
            # Use a fresh instance that probes availability
            bridge = GeoTokenIntegration.__new__(GeoTokenIntegration)
            bridge._config = {}
            bridge._GeoTokenizer = None
            bridge._GraphTokenizer = None
            bridge._CommandSequenceTokenizer = None
            bridge._CADVocabulary = None
            bridge._TokenSequence = None

            assert bridge.available is False

            # Methods should return None / empty
            result = bridge.tokenize_mesh(MagicMock())
            assert result is None

            result = bridge.tokenize_topology(MagicMock())
            assert result is None

            result = bridge.tokenize_sketch(MagicMock())
            assert result is None

            val = bridge.validate_roundtrip(MagicMock())
            assert val == {}


# ---------------------------------------------------------------------------
# Tests: Mesh tokenization
# ---------------------------------------------------------------------------


class TestMeshTokenization:
    """Tests for mesh tokenization via the bridge."""

    def test_tokenize_mesh_item(self, bridge, mock_mesh_item):
        """Single mesh item should produce a TokenSequence."""
        result = bridge.tokenize_mesh(mock_mesh_item)
        assert result is not None
        # TokenSequence should have coordinate and geometry tokens
        assert hasattr(result, "coordinate_tokens") or hasattr(result, "command_tokens")
        assert len(result.coordinate_tokens) > 0

    def test_tokenize_document_mesh(self, bridge, mock_document):
        """Document tokenization should pick up mesh items."""
        result = bridge.tokenize_document(
            mock_document,
            include_mesh=True,
            include_graph=False,
            include_commands=False,
        )

        assert result.token_sequences
        # The mesh item should have been tokenized
        mesh_keys = [
            k for k in result.token_sequences
            if not k.endswith("__cmd") and k != "__graph__"
        ]
        assert len(mesh_keys) >= 1
        assert result.metadata.get("mesh_tokenized", 0) >= 1


# ---------------------------------------------------------------------------
# Tests: Topology graph tokenization
# ---------------------------------------------------------------------------


class TestTopologyTokenization:
    """Tests for topology graph tokenization via the bridge."""

    def test_tokenize_topology(self, bridge, mock_topology):
        """Topology graph should produce a TokenSequence."""
        result = bridge.tokenize_topology(mock_topology)
        assert result is not None
        # Graph tokens live in graph_node_tokens / graph_edge_tokens
        assert hasattr(result, "graph_node_tokens") or hasattr(result, "graph_edge_tokens")

    def test_tokenize_document_topology(
        self, bridge, mock_document, mock_topology
    ):
        """Document with topology should produce graph tokens."""
        mock_document.topology = mock_topology

        result = bridge.tokenize_document(
            mock_document,
            include_mesh=False,
            include_graph=True,
            include_commands=False,
        )

        assert "__graph__" in result.token_sequences
        assert result.metadata.get("graph_tokenized", 0) == 1


# ---------------------------------------------------------------------------
# Tests: Sketch / command tokenization
# ---------------------------------------------------------------------------


class TestSketchTokenization:
    """Tests for sketch command tokenization via the bridge."""

    def test_tokenize_sketch(self, bridge, mock_sketch_item):
        """Sketch item should produce command tokens."""
        result = bridge.tokenize_sketch(mock_sketch_item)
        assert result is not None
        assert hasattr(result, "command_tokens")
        assert len(result.command_tokens) > 0

    def test_tokenize_sketch_with_constraints(self, bridge, mock_sketch_item):
        """Sketch with constraints enabled should still work."""
        result = bridge.tokenize_sketch(
            mock_sketch_item, include_constraints=True,
        )
        assert result is not None

    def test_tokenize_document_commands(self, bridge, mock_document):
        """Document tokenization should pick up command items."""
        result = bridge.tokenize_document(
            mock_document,
            include_mesh=False,
            include_graph=False,
            include_commands=True,
        )

        cmd_keys = [k for k in result.token_sequences if k.endswith("__cmd")]
        assert len(cmd_keys) >= 1
        assert result.metadata.get("command_tokenized", 0) >= 1


# ---------------------------------------------------------------------------
# Tests: Encode / decode roundtrip
# ---------------------------------------------------------------------------


class TestEncodeDecodeRoundtrip:
    """Tests for encoding token sequences to IDs and back."""

    def test_encode_sequences(self, bridge, mock_mesh_item):
        """Encoding should produce integer ID lists."""
        ts = bridge.tokenize_mesh(mock_mesh_item)
        assert ts is not None

        encoded = bridge.encode_sequences({"mesh": ts})
        assert "mesh" in encoded
        assert all(isinstance(i, int) for i in encoded["mesh"])
        assert len(encoded["mesh"]) > 0

    def test_decode_token_ids(self, bridge, mock_mesh_item):
        """Decoding IDs should produce a TokenSequence."""
        ts = bridge.tokenize_mesh(mock_mesh_item)
        assert ts is not None

        encoded = bridge.encode_sequences({"mesh": ts})
        ids = encoded["mesh"]

        decoded = bridge.decode_token_ids(ids)
        # decoded should be a TokenSequence or equivalent
        assert decoded is not None

    def test_encode_decode_roundtrip_produces_output(
        self, bridge, mock_sketch_item,
    ):
        """Encoded → decoded should produce a valid TokenSequence.

        Uses sketch/command tokens for the roundtrip since
        ``vocab.decode()`` is command-oriented. Mesh token IDs
        occupy a different region of the vocabulary and don't
        round-trip through command decoding.
        """
        ts = bridge.tokenize_sketch(mock_sketch_item)
        assert ts is not None

        encoded = bridge.encode_sequences({"sketch": ts})
        ids = encoded["sketch"]
        assert len(ids) > 0

        decoded = bridge.decode_token_ids(ids)
        assert decoded is not None

        # Decoded should contain command_tokens (from vocab.decode)
        if hasattr(decoded, "command_tokens"):
            # Command tokens should be recovered from the IDs
            assert isinstance(decoded.command_tokens, list)


# ---------------------------------------------------------------------------
# Tests: Roundtrip quality validation
# ---------------------------------------------------------------------------


class TestRoundtripValidation:
    """Tests for the validate_roundtrip quality check."""

    def test_validate_roundtrip_returns_metrics(self, bridge, mock_mesh_item):
        """validate_roundtrip should return quality metrics dict."""
        report = bridge.validate_roundtrip(mock_mesh_item)
        # Should be a dict (may be empty if impact module not available)
        assert isinstance(report, dict)

    def test_validate_roundtrip_no_mesh(self, bridge):
        """validate_roundtrip on a non-mesh item returns empty dict."""
        non_mesh = MagicMock()
        del non_mesh.to_numpy
        non_mesh.properties = {}

        report = bridge.validate_roundtrip(non_mesh)
        assert report == {}


# ---------------------------------------------------------------------------
# Tests: SDG vocabulary alignment
# ---------------------------------------------------------------------------


class TestSDGVocabularyAlignment:
    """Tests that SDG sequence annotator uses geotoken vocabulary."""

    def test_sequence_annotator_uses_geotoken(self):
        """SequenceAnnotator should prefer geotoken when available."""
        from cadling.sdg.qa.sequence_annotator import SequenceAnnotator

        annotator = SequenceAnnotator()
        assert annotator._use_geotoken is True
        assert annotator._geotoken_tokenizer is not None
        assert annotator._geotoken_vocab is not None

    def test_construction_history_via_geotoken(self):
        """Construction history should tokenize via geotoken vocab."""
        from cadling.sdg.qa.sequence_annotator import SequenceAnnotator

        annotator = SequenceAnnotator()
        if not annotator._use_geotoken:
            pytest.skip("geotoken not available")

        history = {
            "operations": [
                {"type": "sketch", "plane": "XY"},
                {"type": "extrude", "distance": 50.0, "direction": "positive"},
            ]
        }

        tokens = annotator._tokenize_construction_history(history)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_deepcad_json_via_geotoken(self):
        """DeepCAD JSON should tokenize via geotoken vocab."""
        from cadling.sdg.qa.sequence_annotator import SequenceAnnotator

        annotator = SequenceAnnotator()
        if not annotator._use_geotoken:
            pytest.skip("geotoken not available")

        data = {
            "sequence": [
                {"type": "SKETCH", "profiles": []},
                {"type": "EXT", "extent_one": 0.5, "boolean": "new_body"},
            ]
        }

        tokens = annotator._tokenize_deepcad_json(data)
        assert isinstance(tokens, list)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# Tests: UV-grid quantizer
# ---------------------------------------------------------------------------


class TestUVGridQuantizer:
    """Tests for the UVGridQuantizer module."""

    def test_quantize_surface_samples(self):
        """Basic UV-grid quantization should work."""
        from geotoken.quantization.uv_grid_quantizer import UVGridQuantizer

        quantizer = UVGridQuantizer(grid_resolution=(5, 5), bits=8)

        uv = np.random.rand(25, 2).astype(np.float32)
        xyz = np.random.rand(25, 3).astype(np.float32)

        result = quantizer.quantize_surface_samples(uv, xyz)

        assert result.quantized_grid.shape == (25, 3)
        assert result.quantized_grid.dtype in (np.int32, np.int64)
        assert result.bits == 8
        assert result.params is not None

    def test_quantize_dequantize_roundtrip(self):
        """Dequantized values should approximate the originals."""
        from geotoken.quantization.uv_grid_quantizer import UVGridQuantizer

        quantizer = UVGridQuantizer(grid_resolution=(3, 3), bits=10)

        uv = np.random.rand(9, 2).astype(np.float32)
        xyz = np.random.rand(9, 3).astype(np.float32)

        tokens = quantizer.quantize_surface_samples(uv, xyz)
        reconstructed = quantizer.dequantize(tokens)

        assert reconstructed.shape == xyz.shape
        # Error should be bounded by quantization step size
        max_err = np.max(np.abs(xyz - reconstructed))
        expected_step = 1.0 / (2**10)
        # Allow some margin for edge effects
        assert max_err < expected_step * 5

    def test_to_flat_tokens(self):
        """Flat token list should have length N * 3."""
        from geotoken.quantization.uv_grid_quantizer import UVGridQuantizer

        quantizer = UVGridQuantizer(grid_resolution=(4, 4), bits=8)

        uv = np.random.rand(16, 2).astype(np.float32)
        xyz = np.random.rand(16, 3).astype(np.float32)

        tokens = quantizer.quantize_surface_samples(uv, xyz)
        flat = quantizer.to_flat_tokens(tokens)

        assert isinstance(flat, list)
        assert len(flat) == 16 * 3
        assert all(isinstance(t, int) for t in flat)

    def test_quantize_from_topology(self, mock_topology):
        """UV grid quantization from topology should process faces."""
        from geotoken.quantization.uv_grid_quantizer import UVGridQuantizer

        quantizer = UVGridQuantizer(grid_resolution=(3, 3), bits=8)
        results = quantizer.quantize_from_topology(mock_topology)

        assert isinstance(results, dict)
        # Should have quantized at least some faces
        assert len(results) > 0

        for face_idx, tokens in results.items():
            assert tokens.quantized_grid.shape[1] == 3
            assert tokens.face_index == face_idx

    def test_shape_validation(self):
        """Invalid shapes should raise ValueError."""
        from geotoken.quantization.uv_grid_quantizer import UVGridQuantizer

        quantizer = UVGridQuantizer()

        with pytest.raises(ValueError, match="uv_samples must be"):
            quantizer.quantize_surface_samples(
                np.zeros((10, 3)),  # wrong: 3 cols instead of 2
                np.zeros((10, 3)),
            )

        with pytest.raises(ValueError, match="xyz_samples must be"):
            quantizer.quantize_surface_samples(
                np.zeros((10, 2)),
                np.zeros((10, 2)),  # wrong: 2 cols instead of 3
            )

        with pytest.raises(ValueError, match="Sample count mismatch"):
            quantizer.quantize_surface_samples(
                np.zeros((10, 2)),
                np.zeros((5, 3)),  # wrong: different N
            )

    def test_importable_from_package(self):
        """UVGridQuantizer should be importable from geotoken root."""
        from geotoken import UVGridQuantizer, UVGridTokens

        assert UVGridQuantizer is not None
        assert UVGridTokens is not None


# ---------------------------------------------------------------------------
# Tests: Full document tokenization
# ---------------------------------------------------------------------------


class TestFullDocumentTokenization:
    """End-to-end test covering all token channels."""

    def test_tokenize_all_channels(
        self, bridge, mock_document, mock_topology,
    ):
        """Document with mesh, topology, and commands should tokenize all."""
        mock_document.topology = mock_topology

        result = bridge.tokenize_document(
            mock_document,
            include_mesh=True,
            include_graph=True,
            include_commands=True,
            include_constraints=False,
        )

        # Should have mesh tokens
        assert result.metadata.get("mesh_tokenized", 0) >= 1

        # Should have graph tokens
        assert "__graph__" in result.token_sequences
        assert result.metadata.get("graph_tokenized", 0) == 1

        # Should have command tokens
        cmd_keys = [k for k in result.token_sequences if k.endswith("__cmd")]
        assert len(cmd_keys) >= 1

        # Vocabulary should be present
        assert result.vocabulary is not None

        # No errors expected
        assert len(result.errors) == 0

    def test_encode_all_sequences(
        self, bridge, mock_document, mock_topology,
    ):
        """All token sequences should be encodable to integer IDs."""
        mock_document.topology = mock_topology

        result = bridge.tokenize_document(mock_document)
        encoded = bridge.encode_sequences(result.token_sequences)

        assert len(encoded) > 0
        for key, ids in encoded.items():
            assert isinstance(ids, list)
            assert all(isinstance(i, int) for i in ids)
            assert len(ids) > 0
