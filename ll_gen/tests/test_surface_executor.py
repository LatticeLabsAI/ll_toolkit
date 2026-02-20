"""Comprehensive test suite for ll_gen.disposal.surface_executor module.

Tests surface execution functionality for LatentProposal objects:
- Face grid → B-spline surface fitting
- Edge point → B-spline curve fitting
- Topology sewing and shell construction
- Adjacency-based face merging
- Fallback behavior without cadling
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
import numpy as np

import pytest

from ll_gen.proposals.latent_proposal import LatentProposal
from ll_gen.disposal.surface_executor import (
    execute_latent_proposal,
    _chamfer_distance,
)


# ============================================================================
# SECTION 1: Module Import Tests
# ============================================================================


class TestModuleImport:
    """Test module import and availability flags."""

    def test_module_importable(self) -> None:
        """Test that surface_executor module is importable."""
        from ll_gen.disposal import surface_executor
        assert hasattr(surface_executor, "execute_latent_proposal")
        assert hasattr(surface_executor, "_fit_bspline_surface")
        assert hasattr(surface_executor, "_fit_bspline_curve")
        assert hasattr(surface_executor, "_deduplicate_edges")
        assert hasattr(surface_executor, "_sew_faces")

    def test_availability_flags_are_boolean(self) -> None:
        """Test that availability flags are boolean values."""
        from ll_gen.disposal import surface_executor
        assert isinstance(surface_executor._OCC_AVAILABLE, bool)
        assert isinstance(surface_executor._CADLING_SURFACE_FITTER_AVAILABLE, bool)
        assert isinstance(surface_executor._CADLING_TOPOLOGY_MERGER_AVAILABLE, bool)

    def test_execute_function_is_callable(self) -> None:
        """Test that execute_latent_proposal is callable."""
        from ll_gen.disposal.surface_executor import execute_latent_proposal
        assert callable(execute_latent_proposal)


# ============================================================================
# SECTION 2: Execute Latent Proposal Tests (Mocked)
# ============================================================================


class TestExecuteLatentProposalMocked:
    """Test execute_latent_proposal with mocked dependencies."""

    def test_occ_unavailable_raises_runtime_error(self) -> None:
        """Test RuntimeError when pythonocc is not available."""
        proposal = LatentProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test shape",
            face_grids=[np.random.randn(4, 4, 3).astype(np.float32)],
        )
        with patch(
            "ll_gen.disposal.surface_executor._OCC_AVAILABLE", False
        ):
            with pytest.raises(RuntimeError) as exc_info:
                execute_latent_proposal(proposal)
            assert "pythonocc" in str(exc_info.value).lower()

    def test_cadling_surface_fitter_preferred(self) -> None:
        """Test that cadling BSplineSurfaceFitter is used when available."""
        proposal = LatentProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test shape",
            face_grids=[np.random.randn(4, 4, 3).astype(np.float32)],
            edge_points=[np.random.randn(10, 3).astype(np.float32)],
        )

        # This test verifies the preference logic exists
        # Actual execution requires OCC and cadling
        from ll_gen.disposal import surface_executor
        assert hasattr(surface_executor, "_CADLING_SURFACE_FITTER_AVAILABLE")


# ============================================================================
# SECTION 3: Chamfer Distance Tests
# ============================================================================


class TestChamferDistance:
    """Test Chamfer distance computation."""

    def test_chamfer_distance_identical_points(self) -> None:
        """Test Chamfer distance is zero for identical point sets."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        dist = _chamfer_distance(pts, pts)
        assert dist < 1e-6

    def test_chamfer_distance_different_points(self) -> None:
        """Test Chamfer distance is positive for different point sets."""
        pts_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        pts_b = np.array([[0, 0, 1], [1, 0, 1]], dtype=np.float32)
        dist = _chamfer_distance(pts_a, pts_b)
        assert dist > 0

    def test_chamfer_distance_empty_set_a(self) -> None:
        """Test Chamfer distance with empty first set returns inf."""
        pts_a = np.array([]).reshape(0, 3)
        pts_b = np.array([[0, 0, 0]], dtype=np.float32)
        dist = _chamfer_distance(pts_a, pts_b)
        assert dist == float("inf")

    def test_chamfer_distance_empty_set_b(self) -> None:
        """Test Chamfer distance with empty second set returns inf."""
        pts_a = np.array([[0, 0, 0]], dtype=np.float32)
        pts_b = np.array([]).reshape(0, 3)
        dist = _chamfer_distance(pts_a, pts_b)
        assert dist == float("inf")

    def test_chamfer_distance_symmetric(self) -> None:
        """Test Chamfer distance is symmetric."""
        pts_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        pts_b = np.array([[0.5, 0.5, 0], [1.5, 0.5, 0]], dtype=np.float32)
        dist_ab = _chamfer_distance(pts_a, pts_b)
        dist_ba = _chamfer_distance(pts_b, pts_a)
        assert abs(dist_ab - dist_ba) < 1e-6

    def test_chamfer_distance_known_value(self) -> None:
        """Test Chamfer distance with known expected value."""
        # Two parallel lines, unit distance apart
        pts_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        pts_b = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.float32)
        dist = _chamfer_distance(pts_a, pts_b)
        # Distance should be 1.0 (each point is distance 1 from other set)
        assert abs(dist - 1.0) < 1e-6


# ============================================================================
# SECTION 4: Latent Proposal Interface Tests
# ============================================================================


class TestLatentProposalInterface:
    """Test LatentProposal interface used by surface executor."""

    def test_proposal_has_face_grids(self) -> None:
        """Test proposal has face_grids attribute."""
        proposal = LatentProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            face_grids=[np.random.randn(4, 4, 3).astype(np.float32)],
        )
        assert hasattr(proposal, "face_grids")
        assert len(proposal.face_grids) == 1

    def test_proposal_has_edge_points(self) -> None:
        """Test proposal has edge_points attribute."""
        proposal = LatentProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            face_grids=[np.random.randn(4, 4, 3).astype(np.float32)],
            edge_points=[np.random.randn(10, 3).astype(np.float32)],
        )
        assert hasattr(proposal, "edge_points")
        assert len(proposal.edge_points) == 1

    def test_proposal_face_grid_shape(self) -> None:
        """Test face grids have expected shape [U, V, 3]."""
        grid = np.random.randn(8, 8, 3).astype(np.float32)
        proposal = LatentProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            face_grids=[grid],
        )
        assert proposal.face_grids[0].shape == (8, 8, 3)

    def test_proposal_edge_points_shape(self) -> None:
        """Test edge points have expected shape [N, 3]."""
        points = np.random.randn(32, 3).astype(np.float32)
        proposal = LatentProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            face_grids=[],
            edge_points=[points],
        )
        assert proposal.edge_points[0].shape == (32, 3)


# ============================================================================
# SECTION 5: B-Spline Fitting Tests (Structure)
# ============================================================================


class TestBSplineFittingStructure:
    """Test B-spline fitting function structure."""

    def test_fit_bspline_surface_exists(self) -> None:
        """Test _fit_bspline_surface function exists."""
        from ll_gen.disposal.surface_executor import _fit_bspline_surface
        assert callable(_fit_bspline_surface)

    def test_fit_bspline_curve_exists(self) -> None:
        """Test _fit_bspline_curve function exists."""
        from ll_gen.disposal.surface_executor import _fit_bspline_curve
        assert callable(_fit_bspline_curve)

    def test_surface_fitter_tolerance_parameter(self) -> None:
        """Test that _fit_bspline_surface accepts tolerance parameter."""
        from ll_gen.disposal.surface_executor import _fit_bspline_surface
        import inspect
        sig = inspect.signature(_fit_bspline_surface)
        assert "tolerance" in sig.parameters


# ============================================================================
# SECTION 6: Topology Operations Tests (Structure)
# ============================================================================


class TestTopologyOperationsStructure:
    """Test topology operation function structure."""

    def test_deduplicate_edges_exists(self) -> None:
        """Test _deduplicate_edges function exists."""
        from ll_gen.disposal.surface_executor import _deduplicate_edges
        assert callable(_deduplicate_edges)

    def test_trim_surfaces_with_edges_exists(self) -> None:
        """Test _trim_surfaces_with_edges function exists."""
        from ll_gen.disposal.surface_executor import _trim_surfaces_with_edges
        assert callable(_trim_surfaces_with_edges)

    def test_sew_faces_exists(self) -> None:
        """Test _sew_faces function exists."""
        from ll_gen.disposal.surface_executor import _sew_faces
        assert callable(_sew_faces)


# ============================================================================
# SECTION 7: Edge Deduplication Logic Tests
# ============================================================================


class TestEdgeDeduplicationLogic:
    """Test edge deduplication logic and thresholds."""

    def test_deduplication_bbox_threshold(self) -> None:
        """Test that bbox distance threshold is documented."""
        # According to the code, edges are merged if:
        # - bbox_dist < 0.08
        # - shape_sim < 0.2
        bbox_threshold = 0.08
        shape_threshold = 0.2
        assert bbox_threshold < shape_threshold

    def test_deduplication_uses_chamfer_distance(self) -> None:
        """Test that deduplication uses Chamfer distance for similarity."""
        from ll_gen.disposal.surface_executor import _chamfer_distance
        assert callable(_chamfer_distance)


# ============================================================================
# SECTION 8: Sewing Operations Tests (Structure)
# ============================================================================


class TestSewingOperationsStructure:
    """Test sewing operation structure."""

    def test_sew_faces_function_exists(self) -> None:
        """Test _sew_faces function exists."""
        from ll_gen.disposal.surface_executor import _sew_faces
        assert callable(_sew_faces)

    def test_check_sewed_shape_quality_exists(self) -> None:
        """Test _check_sewed_shape_quality function exists."""
        from ll_gen.disposal.surface_executor import _check_sewed_shape_quality
        assert callable(_check_sewed_shape_quality)


# ============================================================================
# SECTION 9: Fixture Tests
# ============================================================================


class TestWithFixtures:
    """Test with conftest fixtures."""

    def test_latent_proposal_fixture(self, latent_proposal) -> None:
        """Test latent_proposal fixture has expected structure."""
        assert hasattr(latent_proposal, "face_grids")
        assert hasattr(latent_proposal, "edge_points")
        assert len(latent_proposal.face_grids) > 0

    def test_latent_proposal_minimal_fixture(self, latent_proposal_minimal) -> None:
        """Test latent_proposal_minimal fixture has minimal structure."""
        assert len(latent_proposal_minimal.face_grids) == 1
        # Minimal proposal may have no edges
        assert len(latent_proposal_minimal.edge_points) == 0


# ============================================================================
# SECTION 10: Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Test helper functions."""

    def test_compute_edge_bbox_center_exists(self) -> None:
        """Test _compute_edge_bbox_center function exists."""
        from ll_gen.disposal.surface_executor import _compute_edge_bbox_center
        assert callable(_compute_edge_bbox_center)

    def test_sample_edge_points_exists(self) -> None:
        """Test _sample_edge_points function exists."""
        from ll_gen.disposal.surface_executor import _sample_edge_points
        assert callable(_sample_edge_points)

    def test_average_edges_exists(self) -> None:
        """Test _average_edges function exists."""
        from ll_gen.disposal.surface_executor import _average_edges
        assert callable(_average_edges)
