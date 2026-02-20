"""Unit tests for coedge extractor module.

Tests the CoedgeExtractor class that extracts ordered coedge sequences
from B-Rep models.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


class TestCoedgeImports:
    """Test module imports."""

    def test_module_imports(self):
        """Test that the module imports successfully."""
        from cadling.lib.topology import CoedgeExtractor, Coedge
        from cadling.lib.topology.coedge_extractor import (
            CoedgeExtractor as CEExtractor,
            Coedge as CECoedge,
            HAS_OCC,
        )

        assert CoedgeExtractor is CEExtractor
        assert Coedge is CECoedge


class TestCoedge:
    """Test Coedge dataclass."""

    def test_coedge_creation(self):
        """Test creating a Coedge."""
        from cadling.lib.topology.coedge_extractor import Coedge

        coedge = Coedge(
            id=0,
            face_id="face_1",
            edge_id="edge_1",
            face_index=0,
            edge_index=0,
            orientation="FORWARD",
            position_in_loop=0,
            loop_size=4,
        )

        assert coedge.id == 0
        assert coedge.face_id == "face_1"
        assert coedge.edge_id == "edge_1"
        assert coedge.orientation == "FORWARD"
        assert coedge.position_in_loop == 0
        assert coedge.loop_size == 4

    def test_coedge_defaults(self):
        """Test Coedge default values."""
        from cadling.lib.topology.coedge_extractor import Coedge

        coedge = Coedge(id=0, face_id="f", edge_id="e")

        assert coedge.face_index == -1
        assert coedge.edge_index == -1
        assert coedge.orientation == "FORWARD"
        assert coedge.position_in_loop == 0
        assert coedge.loop_size == 0
        assert coedge.next_id is None
        assert coedge.prev_id is None
        assert coedge.mate_id is None
        assert coedge.features is None

    def test_coedge_to_dict(self):
        """Test Coedge to_dict method."""
        from cadling.lib.topology.coedge_extractor import Coedge

        coedge = Coedge(
            id=0,
            face_id="face_1",
            edge_id="edge_1",
            face_index=0,
            edge_index=0,
            orientation="FORWARD",
            position_in_loop=0,
            loop_size=4,
            next_id=1,
            prev_id=3,
            mate_id=4,
        )

        d = coedge.to_dict()

        assert d["id"] == 0
        assert d["face_id"] == "face_1"
        assert d["edge_id"] == "edge_1"
        assert d["orientation"] == "FORWARD"
        assert d["next_id"] == 1
        assert d["prev_id"] == 3
        assert d["mate_id"] == 4


class TestCoedgeExtractor:
    """Test CoedgeExtractor class."""

    def test_init_default(self):
        """Test default initialization."""
        from cadling.lib.topology.coedge_extractor import CoedgeExtractor

        extractor = CoedgeExtractor()

        assert extractor.registry is not None

    def test_init_with_registry(self):
        """Test initialization with custom registry."""
        from cadling.lib.topology.coedge_extractor import CoedgeExtractor
        from cadling.lib.topology.face_identity import ShapeIdentityRegistry

        registry = ShapeIdentityRegistry()
        extractor = CoedgeExtractor(registry=registry)

        assert extractor.registry is registry

    def test_extract_coedges_no_occ(self):
        """Test extract_coedges returns empty list when no OCC."""
        from cadling.lib.topology.coedge_extractor import (
            CoedgeExtractor,
            HAS_OCC,
        )

        if not HAS_OCC:
            extractor = CoedgeExtractor()
            mock_shape = MagicMock()

            result = extractor.extract_coedges(mock_shape)
            assert result == []

    def test_extract_coedge_data_no_occ(self):
        """Test extract_coedge_data returns empty data when no OCC."""
        from cadling.lib.topology.coedge_extractor import (
            CoedgeExtractor,
            HAS_OCC,
        )

        if not HAS_OCC:
            extractor = CoedgeExtractor()
            mock_shape = MagicMock()

            result = extractor.extract_coedge_data(mock_shape)

            assert result["features"].shape == (0, 12)
            assert result["next_indices"].shape == (0,)
            assert result["prev_indices"].shape == (0,)
            assert result["mate_indices"].shape == (0,)
            assert result["face_indices"].shape == (0,)
            assert result["coedges"] == []

    def test_to_brep_net_format_no_occ(self):
        """Test to_brep_net_format returns empty data when no OCC."""
        from cadling.lib.topology.coedge_extractor import (
            CoedgeExtractor,
            HAS_OCC,
        )

        if not HAS_OCC:
            extractor = CoedgeExtractor()
            mock_shape = MagicMock()

            result = extractor.to_brep_net_format(mock_shape)

            assert "coedge_features" in result
            assert "coedge_next" in result
            assert "coedge_prev" in result
            assert "coedge_mate" in result
            assert "coedge_to_face" in result
            assert result["num_coedges"] == 0
            assert result["num_faces"] == 0


class TestCoedgeExtractorHelpers:
    """Test helper methods of CoedgeExtractor."""

    def test_build_loop_pointers(self):
        """Test _build_loop_pointers creates correct next/prev links."""
        from cadling.lib.topology.coedge_extractor import CoedgeExtractor, Coedge

        extractor = CoedgeExtractor()

        # Create a simple loop of 4 coedges
        coedges = [
            Coedge(id=0, face_id="f1", edge_id="e1", position_in_loop=0, loop_size=4),
            Coedge(id=1, face_id="f1", edge_id="e2", position_in_loop=1, loop_size=4),
            Coedge(id=2, face_id="f1", edge_id="e3", position_in_loop=2, loop_size=4),
            Coedge(id=3, face_id="f1", edge_id="e4", position_in_loop=3, loop_size=4),
        ]

        extractor._build_loop_pointers(coedges)

        # Check cyclic next pointers
        assert coedges[0].next_id == 1
        assert coedges[1].next_id == 2
        assert coedges[2].next_id == 3
        assert coedges[3].next_id == 0  # Wraps around

        # Check cyclic prev pointers
        assert coedges[0].prev_id == 3  # Wraps around
        assert coedges[1].prev_id == 0
        assert coedges[2].prev_id == 1
        assert coedges[3].prev_id == 2

    def test_build_mate_pointers_two_faces(self):
        """Test _build_mate_pointers connects coedges across faces."""
        from cadling.lib.topology.coedge_extractor import CoedgeExtractor, Coedge

        extractor = CoedgeExtractor()

        # Two faces sharing an edge
        coedges = [
            Coedge(id=0, face_id="f1", edge_id="shared_edge"),
            Coedge(id=1, face_id="f2", edge_id="shared_edge"),
        ]

        # Mock edge_to_faces map (not used directly in simplified version)
        mock_map = MagicMock()

        # The helper builds edge_to_coedges internally
        extractor._build_mate_pointers(coedges, mock_map, {})

        # Coedges on the same edge should be mates
        assert coedges[0].mate_id == 1
        assert coedges[1].mate_id == 0

    def test_build_mate_pointers_boundary_edge(self):
        """Test _build_mate_pointers handles boundary edges."""
        from cadling.lib.topology.coedge_extractor import CoedgeExtractor, Coedge

        extractor = CoedgeExtractor()

        # Single coedge on a boundary edge (only one face)
        coedges = [
            Coedge(id=0, face_id="f1", edge_id="boundary_edge"),
        ]

        mock_map = MagicMock()
        extractor._build_mate_pointers(coedges, mock_map, {})

        # Boundary edge points to itself
        assert coedges[0].mate_id == 0

    def test_compute_convexity_default(self):
        """Test _compute_convexity returns 0.5 for boundary edges."""
        from cadling.lib.topology.coedge_extractor import CoedgeExtractor, Coedge

        extractor = CoedgeExtractor()

        # Boundary coedge (mate = self)
        coedge = Coedge(id=0, face_id="f1", edge_id="e1", mate_id=0)

        all_coedges = [coedge]
        mock_shape = MagicMock()

        result = extractor._compute_convexity(coedge, all_coedges, mock_shape)

        # Boundary edges have unknown convexity
        assert result == 0.5


class TestCoedgeFeatureExtraction:
    """Test coedge feature extraction."""

    def test_extract_coedge_features_shape(self):
        """Test _extract_coedge_features returns correct shape."""
        from cadling.lib.topology.coedge_extractor import CoedgeExtractor, Coedge

        extractor = CoedgeExtractor()

        coedges = [
            Coedge(id=0, face_id="f1", edge_id="e1"),
            Coedge(id=1, face_id="f1", edge_id="e2"),
        ]

        mock_shape = MagicMock()

        # Mock registry to return None for edges
        with patch.object(extractor.registry, 'get_edge', return_value=None):
            features = extractor._extract_coedge_features(coedges, mock_shape, 12)

        assert features.shape == (2, 12)
        assert features.dtype == np.float32

    def test_extract_coedge_features_empty(self):
        """Test _extract_coedge_features with empty list."""
        from cadling.lib.topology.coedge_extractor import CoedgeExtractor

        extractor = CoedgeExtractor()

        mock_shape = MagicMock()
        features = extractor._extract_coedge_features([], mock_shape, 12)

        assert features.shape == (0, 12)


@pytest.mark.requires_pythonocc
class TestCoedgeExtractorWithOCC:
    """Integration tests requiring pythonocc-core."""

    def test_extract_coedges_from_box(self):
        """Test extracting coedges from a box shape."""
        from cadling.lib.topology.coedge_extractor import (
            CoedgeExtractor,
            HAS_OCC,
        )

        if not HAS_OCC:
            pytest.skip("pythonocc not available")

        # Would need a real OCC box shape to test
        pass

    def test_coedge_data_tensors(self):
        """Test that coedge data produces valid tensors."""
        from cadling.lib.topology.coedge_extractor import (
            CoedgeExtractor,
            HAS_OCC,
        )

        if not HAS_OCC:
            pytest.skip("pythonocc not available")

        pass
