"""Unit tests for OCC wrapper module.

Tests the OCCShape, OCCFace, and OCCEdge wrapper classes that provide
AI-friendly access to pythonocc-core functionality.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


class TestOCCWrapperImports:
    """Test module imports and availability flags."""

    def test_module_imports(self):
        """Test that the module imports successfully."""
        from cadling.lib import occ_wrapper

        assert hasattr(occ_wrapper, "HAS_OCC")
        assert hasattr(occ_wrapper, "HAS_OCCWL")
        assert hasattr(occ_wrapper, "OCCShape")
        assert hasattr(occ_wrapper, "OCCFace")
        assert hasattr(occ_wrapper, "OCCEdge")

    def test_surface_type_map(self):
        """Test surface type mapping is complete."""
        from cadling.lib.occ_wrapper import SURFACE_TYPE_MAP

        expected_types = [
            "PLANE", "CYLINDER", "CONE", "SPHERE", "TORUS",
            "BEZIER", "BSPLINE", "REVOLUTION", "EXTRUSION", "OTHER"
        ]

        for i, name in enumerate(expected_types):
            assert SURFACE_TYPE_MAP[name] == i

    def test_curve_type_map(self):
        """Test curve type mapping is complete."""
        from cadling.lib.occ_wrapper import CURVE_TYPE_MAP

        expected_types = [
            "LINE", "CIRCLE", "ELLIPSE", "HYPERBOLA",
            "PARABOLA", "BEZIER", "BSPLINE", "OTHER"
        ]

        for i, name in enumerate(expected_types):
            assert CURVE_TYPE_MAP[name] == i


class TestOCCEdge:
    """Test OCCEdge wrapper class."""

    def test_init_with_mock(self):
        """Test OCCEdge initialization with mock edge."""
        from cadling.lib.occ_wrapper import OCCEdge

        mock_edge = MagicMock()
        wrapper = OCCEdge(mock_edge)

        assert wrapper._occ_edge is mock_edge
        assert wrapper.occ_edge is mock_edge

    def test_curve_type_fallback(self):
        """Test curve type returns OTHER when OCC unavailable."""
        from cadling.lib.occ_wrapper import OCCEdge, HAS_OCC

        mock_edge = MagicMock()
        wrapper = OCCEdge(mock_edge)

        # When OCC is not available or fails, should return OTHER
        with patch.object(wrapper, '_curve_type_pythonocc', return_value="OTHER"):
            result = wrapper.curve_type()
            assert result == "OTHER"

    def test_length_fallback(self):
        """Test length returns 0 when OCC unavailable."""
        from cadling.lib.occ_wrapper import OCCEdge

        mock_edge = MagicMock()
        wrapper = OCCEdge(mock_edge)

        with patch.object(wrapper, '_length_pythonocc', return_value=0.0):
            result = wrapper.length()
            assert result == 0.0

    def test_tangent_at_default(self):
        """Test tangent_at returns default when extraction fails."""
        from cadling.lib.occ_wrapper import OCCEdge, HAS_OCC

        if not HAS_OCC:
            mock_edge = MagicMock()
            wrapper = OCCEdge(mock_edge)
            result = wrapper.tangent_at(0.5)
            assert result == [0.0, 0.0, 1.0]

    def test_curvature_at_default(self):
        """Test curvature_at returns 0 when extraction fails."""
        from cadling.lib.occ_wrapper import OCCEdge, HAS_OCC

        if not HAS_OCC:
            mock_edge = MagicMock()
            wrapper = OCCEdge(mock_edge)
            result = wrapper.curvature_at(0.5)
            assert result == 0.0

    def test_endpoints_default(self):
        """Test endpoints returns zeros when OCC unavailable."""
        from cadling.lib.occ_wrapper import OCCEdge, HAS_OCC

        if not HAS_OCC:
            mock_edge = MagicMock()
            wrapper = OCCEdge(mock_edge)
            start, end = wrapper.endpoints()
            assert start == [0.0, 0.0, 0.0]
            assert end == [0.0, 0.0, 0.0]

    def test_extract_features(self):
        """Test extract_features returns complete dict."""
        from cadling.lib.occ_wrapper import OCCEdge

        mock_edge = MagicMock()
        wrapper = OCCEdge(mock_edge)

        with patch.object(wrapper, 'curve_type', return_value="LINE"):
            with patch.object(wrapper, 'length', return_value=5.0):
                with patch.object(wrapper, 'tangent_at', return_value=[1.0, 0.0, 0.0]):
                    with patch.object(wrapper, 'curvature_at', return_value=0.0):
                        with patch.object(wrapper, 'endpoints', return_value=([0, 0, 0], [5, 0, 0])):
                            features = wrapper.extract_features()

        assert features["curve_type"] == "LINE"
        assert features["curve_type_idx"] == 0
        assert features["length"] == 5.0
        assert features["tangent"] == [1.0, 0.0, 0.0]
        assert features["curvature"] == 0.0


class TestOCCFace:
    """Test OCCFace wrapper class."""

    def test_init_with_mock(self):
        """Test OCCFace initialization with mock face."""
        from cadling.lib.occ_wrapper import OCCFace

        mock_face = MagicMock()
        wrapper = OCCFace(mock_face)

        assert wrapper._occ_face is mock_face
        assert wrapper.occ_face is mock_face

    def test_surface_type_fallback(self):
        """Test surface type returns OTHER when OCC unavailable."""
        from cadling.lib.occ_wrapper import OCCFace

        mock_face = MagicMock()
        wrapper = OCCFace(mock_face)

        with patch.object(wrapper, '_surface_type_pythonocc', return_value="OTHER"):
            result = wrapper.surface_type()
            assert result == "OTHER"

    def test_area_fallback(self):
        """Test area returns 0 when OCC unavailable."""
        from cadling.lib.occ_wrapper import OCCFace

        mock_face = MagicMock()
        wrapper = OCCFace(mock_face)

        with patch.object(wrapper, '_area_pythonocc', return_value=0.0):
            result = wrapper.area()
            assert result == 0.0

    def test_normal_at_default(self):
        """Test normal_at returns default when extraction fails."""
        from cadling.lib.occ_wrapper import OCCFace, HAS_OCC

        if not HAS_OCC:
            mock_face = MagicMock()
            wrapper = OCCFace(mock_face)
            result = wrapper.normal_at()
            assert result == [0.0, 0.0, 1.0]

    def test_centroid_default(self):
        """Test centroid returns zeros when OCC unavailable."""
        from cadling.lib.occ_wrapper import OCCFace, HAS_OCC

        if not HAS_OCC:
            mock_face = MagicMock()
            wrapper = OCCFace(mock_face)
            result = wrapper.centroid()
            assert result == [0.0, 0.0, 0.0]

    def test_curvature_at_default(self):
        """Test curvature_at returns (0, 0) when OCC unavailable."""
        from cadling.lib.occ_wrapper import OCCFace, HAS_OCC

        if not HAS_OCC:
            mock_face = MagicMock()
            wrapper = OCCFace(mock_face)
            gauss, mean = wrapper.curvature_at()
            assert gauss == 0.0
            assert mean == 0.0

    def test_bbox_default(self):
        """Test bbox returns default when OCC unavailable."""
        from cadling.lib.occ_wrapper import OCCFace, HAS_OCC

        if not HAS_OCC:
            mock_face = MagicMock()
            wrapper = OCCFace(mock_face)
            bbox_min, bbox_max = wrapper.bbox()
            assert bbox_min == [0.0, 0.0, 0.0]
            assert bbox_max == [1.0, 1.0, 1.0]

    def test_extract_features(self):
        """Test extract_features returns complete dict."""
        from cadling.lib.occ_wrapper import OCCFace

        mock_face = MagicMock()
        wrapper = OCCFace(mock_face)

        with patch.object(wrapper, 'surface_type', return_value="PLANE"):
            with patch.object(wrapper, 'area', return_value=10.0):
                with patch.object(wrapper, 'normal_at', return_value=[0, 0, 1]):
                    with patch.object(wrapper, 'centroid', return_value=[0, 0, 0]):
                        with patch.object(wrapper, 'curvature_at', return_value=(0.0, 0.0)):
                            with patch.object(wrapper, 'bbox', return_value=([0, 0, 0], [1, 1, 0])):
                                features = wrapper.extract_features()

        assert features["surface_type"] == "PLANE"
        assert features["surface_type_idx"] == 0
        assert features["area"] == 10.0
        assert features["normal"] == [0, 0, 1]
        assert features["gaussian_curvature"] == 0.0
        assert features["mean_curvature"] == 0.0


class TestOCCShape:
    """Test OCCShape wrapper class."""

    def test_init_with_mock(self):
        """Test OCCShape initialization with mock shape."""
        from cadling.lib.occ_wrapper import OCCShape

        mock_shape = MagicMock()
        wrapper = OCCShape(mock_shape)

        assert wrapper._occ_shape is mock_shape
        assert wrapper.occ_shape is mock_shape

    def test_get_shape_id_with_hashcode(self):
        """Test get_shape_id uses HashCode when available."""
        from cadling.lib.occ_wrapper import OCCShape

        mock_shape = MagicMock()
        mock_shape.HashCode.return_value = 12345

        wrapper = OCCShape(mock_shape)
        shape_id = wrapper.get_shape_id(mock_shape)

        assert shape_id == "12345"
        mock_shape.HashCode.assert_called_once_with(2**31 - 1)

    def test_get_shape_id_fallback(self):
        """Test get_shape_id falls back to hash() when HashCode unavailable."""
        from cadling.lib.occ_wrapper import OCCShape

        mock_shape = MagicMock()
        del mock_shape.HashCode  # Remove HashCode attribute
        mock_shape.__hash__ = lambda self: 67890

        wrapper = OCCShape(mock_shape)
        shape_id = wrapper.get_shape_id(mock_shape)

        # Should use Python hash
        assert isinstance(shape_id, str)

    def test_num_faces_empty(self):
        """Test num_faces returns 0 when no OCC."""
        from cadling.lib.occ_wrapper import OCCShape, HAS_OCC

        if not HAS_OCC:
            mock_shape = MagicMock()
            wrapper = OCCShape(mock_shape)
            assert wrapper.num_faces() == 0

    def test_num_edges_empty(self):
        """Test num_edges returns 0 when no OCC."""
        from cadling.lib.occ_wrapper import OCCShape, HAS_OCC

        if not HAS_OCC:
            mock_shape = MagicMock()
            wrapper = OCCShape(mock_shape)
            assert wrapper.num_edges() == 0

    def test_num_vertices_empty(self):
        """Test num_vertices returns 0 when no OCC."""
        from cadling.lib.occ_wrapper import OCCShape, HAS_OCC

        if not HAS_OCC:
            mock_shape = MagicMock()
            wrapper = OCCShape(mock_shape)
            assert wrapper.num_vertices() == 0

    def test_face_adjacency_graph_empty(self):
        """Test face_adjacency_graph returns empty dict when no OCC."""
        from cadling.lib.occ_wrapper import OCCShape, HAS_OCC

        if not HAS_OCC:
            mock_shape = MagicMock()
            wrapper = OCCShape(mock_shape)
            adjacency = wrapper.face_adjacency_graph()
            assert adjacency == {}

    def test_register_all_faces_empty(self):
        """Test register_all_faces returns empty dict when no OCC."""
        from cadling.lib.occ_wrapper import OCCShape, HAS_OCC

        mock_shape = MagicMock()
        wrapper = OCCShape(mock_shape)
        wrapper._face_registry = {}

        result = wrapper.register_all_faces()
        assert isinstance(result, dict)

    def test_bbox_default(self):
        """Test bbox returns default when OCC unavailable."""
        from cadling.lib.occ_wrapper import OCCShape, HAS_OCC

        if not HAS_OCC:
            mock_shape = MagicMock()
            wrapper = OCCShape(mock_shape)
            bbox_min, bbox_max = wrapper.bbox()
            assert bbox_min == [0.0, 0.0, 0.0]
            assert bbox_max == [1.0, 1.0, 1.0]

    def test_volume_zero_when_no_occ(self):
        """Test volume returns 0 when OCC unavailable."""
        from cadling.lib.occ_wrapper import OCCShape, HAS_OCC

        if not HAS_OCC:
            mock_shape = MagicMock()
            wrapper = OCCShape(mock_shape)
            assert wrapper.volume() == 0.0

    def test_surface_area_zero_when_no_occ(self):
        """Test surface_area returns 0 when OCC unavailable."""
        from cadling.lib.occ_wrapper import OCCShape, HAS_OCC

        if not HAS_OCC:
            mock_shape = MagicMock()
            wrapper = OCCShape(mock_shape)
            assert wrapper.surface_area() == 0.0

    def test_extract_face_feature_matrix_empty(self):
        """Test extract_face_feature_matrix with no faces."""
        from cadling.lib.occ_wrapper import OCCShape

        mock_shape = MagicMock()
        wrapper = OCCShape(mock_shape)

        with patch.object(wrapper, 'face_list', return_value=[]):
            matrix = wrapper.extract_face_feature_matrix()

        assert matrix.shape == (0, 24)
        assert matrix.dtype == np.float32

    def test_extract_edge_index_empty(self):
        """Test extract_edge_index with no adjacency."""
        from cadling.lib.occ_wrapper import OCCShape

        mock_shape = MagicMock()
        wrapper = OCCShape(mock_shape)

        with patch.object(wrapper, 'face_adjacency_graph', return_value={}):
            edge_index = wrapper.extract_edge_index()

        assert edge_index.shape == (2, 0)
        assert edge_index.dtype == np.int64


@pytest.mark.requires_pythonocc
class TestOCCWrapperWithOCC:
    """Integration tests requiring pythonocc-core."""

    def test_faces_iteration(self):
        """Test iterating over faces of a real OCC shape."""
        from cadling.lib.occ_wrapper import OCCShape, HAS_OCC

        if not HAS_OCC:
            pytest.skip("pythonocc not available")

        # Would need a real OCC shape to test
        # This is a placeholder for when OCC is available
        pass

    def test_edges_iteration(self):
        """Test iterating over edges of a real OCC shape."""
        from cadling.lib.occ_wrapper import OCCShape, HAS_OCC

        if not HAS_OCC:
            pytest.skip("pythonocc not available")

        pass

    def test_extract_all_features(self):
        """Test extracting all features from a real OCC shape."""
        from cadling.lib.occ_wrapper import OCCShape, HAS_OCC

        if not HAS_OCC:
            pytest.skip("pythonocc not available")

        pass
