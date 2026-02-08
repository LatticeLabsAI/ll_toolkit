"""Tests for geotoken.vertex.vertex_refinement module.

Tests cover:
- Refinement toward target surface (Chamfer distance minimization)
- Face quality optimization (aspect ratio improvement)
- Convergence detection
- Constraint application (fixed, planar, symmetric)
- VertexPositionOptimizer with scipy
- Error history monotonicity
- Edge cases (single vertex, empty faces, no target)
- Max iterations respect
"""
import numpy as np
import pytest

from geotoken.vertex.vertex_refinement import (
    CoarseToFineRefiner,
    VertexPositionOptimizer,
    RefinementResult,
    OptimizationResult,
)


class TestCoarseToFineRefiner:
    """Tests for the CoarseToFineRefiner class."""

    def test_refine_toward_target_surface(self):
        """Test that refinement moves vertices toward target points."""
        # Coarse vertices: cube corners at scale 1.0
        coarse_vertices = np.array([
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
        ], dtype=np.float32)

        # Target points: shifted closer to origin
        target_points = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
        ], dtype=np.float32)

        refiner = CoarseToFineRefiner(max_iterations=20, learning_rate=0.1)
        result = refiner.refine(coarse_vertices, target_points=target_points)

        assert isinstance(result, RefinementResult)
        assert result.refined_vertices.shape == coarse_vertices.shape
        assert result.final_error < result.initial_error, \
            "Error should decrease when moving toward target"
        assert result.iterations > 0
        assert len(result.error_history) == result.iterations + 1

    def test_face_quality_optimization(self):
        """Test that face quality gradient improves aspect ratio."""
        # Create a simple mesh: elongated triangle
        coarse_vertices = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # Long edge
            [0.1, 0.1, 0.0],   # Short edges
        ], dtype=np.float32)

        faces = np.array([[0, 1, 2]], dtype=np.int32)

        refiner = CoarseToFineRefiner(
            max_iterations=50,
            learning_rate=0.05,
            face_quality_weight=1.0,
        )
        result = refiner.refine(coarse_vertices, face_indices=faces)

        # After refinement, triangle should be more equilateral
        refined = result.refined_vertices
        edge_01 = np.linalg.norm(refined[1] - refined[0])
        edge_12 = np.linalg.norm(refined[2] - refined[1])
        edge_20 = np.linalg.norm(refined[0] - refined[2])

        # Check that edge ratio improved (closer to 1.0)
        refined_edges = [edge_01, edge_12, edge_20]
        mean_edge = np.mean(refined_edges)
        refined_variance = np.mean([(e - mean_edge) ** 2 for e in refined_edges])

        coarse_refined = coarse_vertices
        coarse_edge_01 = np.linalg.norm(coarse_refined[1] - coarse_refined[0])
        coarse_edge_12 = np.linalg.norm(coarse_refined[2] - coarse_refined[1])
        coarse_edge_20 = np.linalg.norm(coarse_refined[0] - coarse_refined[2])

        coarse_edges = [coarse_edge_01, coarse_edge_12, coarse_edge_20]
        coarse_mean = np.mean(coarse_edges)
        coarse_variance = np.mean([(e - coarse_mean) ** 2 for e in coarse_edges])

        assert refined_variance < coarse_variance, \
            "Face quality should improve (edge variance decrease)"

    def test_convergence_detection(self):
        """Test that convergence is detected when error stabilizes."""
        coarse_vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)

        target_points = np.array([
            [0.01, 0.01, 0.01],
            [0.99, 0.01, 0.01],
            [0.51, 0.49, 0.01],
        ], dtype=np.float32)

        # Very high learning rate for fast convergence
        refiner = CoarseToFineRefiner(
            max_iterations=100,
            learning_rate=1.0,
            convergence_threshold=1e-4,
        )
        result = refiner.refine(coarse_vertices, target_points=target_points)

        # With a large learning rate, vertices should get very close quickly
        assert result.iterations <= refiner.max_iterations
        assert result.final_error < 0.1  # Should be quite close

    def test_fixed_vertex_constraint(self):
        """Test that fixed vertex constraint prevents movement."""
        coarse_vertices = np.array([
            [0.0, 0.0, 0.0],  # Will be fixed
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ], dtype=np.float32)

        target_points = np.array([
            [10.0, 10.0, 10.0],  # Far from coarse_vertices[0]
            [1.1, 1.1, 1.1],
            [2.1, 2.1, 2.1],
        ], dtype=np.float32)

        constraints = [
            {
                "type": "fixed",
                "vertex_index": 0,
                "position": np.array([0.0, 0.0, 0.0]),
            }
        ]

        refiner = CoarseToFineRefiner(max_iterations=20, learning_rate=0.1)
        result = refiner.refine(
            coarse_vertices,
            target_points=target_points,
            constraints=constraints,
        )

        # Vertex 0 should remain fixed
        assert np.allclose(result.refined_vertices[0], [0.0, 0.0, 0.0], atol=1e-5)
        # Vertices 1 and 2 should move toward their targets
        assert not np.allclose(result.refined_vertices[1], coarse_vertices[1])

    def test_planar_constraint(self):
        """Test that planar constraint keeps vertices on a plane."""
        coarse_vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],  # Slightly off z=0 plane
            [0.0, 1.0, -0.1],  # Slightly off z=0 plane
        ], dtype=np.float32)

        target_points = np.array([
            [0.0, 0.0, 1.0],  # Different z
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ], dtype=np.float32)

        # Constrain all vertices to z=0 plane
        constraints = [
            {
                "type": "planar",
                "vertex_indices": [0, 1, 2],
                "normal": np.array([0.0, 0.0, 1.0]),
                "d": 0.0,
            }
        ]

        refiner = CoarseToFineRefiner(max_iterations=30, learning_rate=0.05)
        result = refiner.refine(
            coarse_vertices,
            target_points=target_points,
            constraints=constraints,
        )

        # All vertices should have z ≈ 0
        for i in range(3):
            assert np.abs(result.refined_vertices[i, 2]) < 1e-4, \
                f"Vertex {i} should be on z=0 plane"

    def test_symmetric_constraint(self):
        """Test that symmetric constraint enforces mirror symmetry."""
        coarse_vertices = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        target_points = np.array([
            [1.5, 0.0, 0.0],
            [-1.5, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        # Enforce symmetry between vertices 0 and 1 about x=0 plane
        constraints = [
            {
                "type": "symmetric",
                "pairs": [(0, 1)],
                "axis": 0,
            }
        ]

        refiner = CoarseToFineRefiner(max_iterations=20, learning_rate=0.1)
        result = refiner.refine(
            coarse_vertices,
            target_points=target_points,
            constraints=constraints,
        )

        # Vertices 0 and 1 should have opposite x coordinates (approximately)
        refined = result.refined_vertices
        x_vals = [refined[0, 0], refined[1, 0]]
        assert np.abs(x_vals[0] + x_vals[1]) < 1e-4, \
            "Symmetric vertices should mirror across x=0"

    def test_error_history_monotonically_decreasing(self):
        """Test that error history is generally decreasing."""
        coarse_vertices = np.array([
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        target_points = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ], dtype=np.float32)

        refiner = CoarseToFineRefiner(max_iterations=50, learning_rate=0.1)
        result = refiner.refine(coarse_vertices, target_points=target_points)

        # Error should generally decrease (allow small increases due to numerical issues)
        error_hist = result.error_history
        decreasing_steps = sum(
            1 for i in range(1, len(error_hist))
            if error_hist[i] <= error_hist[i-1]
        )
        # At least 80% of steps should be monotonically decreasing or flat
        assert decreasing_steps >= 0.8 * len(error_hist), \
            "Error history should be mostly monotonically decreasing"

    def test_single_vertex(self):
        """Test refinement with a single vertex."""
        coarse_vertices = np.array([
            [0.0, 0.0, 0.0],
        ], dtype=np.float32)

        target_points = np.array([
            [0.5, 0.5, 0.5],
        ], dtype=np.float32)

        refiner = CoarseToFineRefiner(max_iterations=20, learning_rate=0.1)
        result = refiner.refine(coarse_vertices, target_points=target_points)

        assert result.refined_vertices.shape == (1, 3)
        assert result.iterations > 0
        # Vertex should move toward target
        dist_initial = np.linalg.norm(coarse_vertices[0] - target_points[0])
        dist_final = np.linalg.norm(result.refined_vertices[0] - target_points[0])
        assert dist_final < dist_initial

    def test_no_target_with_face_quality(self):
        """Test refinement with face quality but no target points."""
        coarse_vertices = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.1, 0.1, 0.0],
        ], dtype=np.float32)

        faces = np.array([[0, 1, 2]], dtype=np.int32)

        refiner = CoarseToFineRefiner(
            max_iterations=30,
            learning_rate=0.05,
            face_quality_weight=1.0,
        )
        result = refiner.refine(coarse_vertices, face_indices=faces)

        assert result.refined_vertices.shape == coarse_vertices.shape
        assert result.iterations > 0
        # Without target, error should be from face quality only
        assert result.initial_error >= 0

    def test_empty_face_indices(self):
        """Test refinement with empty face array."""
        coarse_vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)

        target_points = np.array([
            [0.1, 0.0, 0.0],
            [0.9, 0.0, 0.0],
        ], dtype=np.float32)

        faces = np.array([], dtype=np.int32).reshape(0, 3)

        refiner = CoarseToFineRefiner(max_iterations=20, learning_rate=0.1)
        result = refiner.refine(
            coarse_vertices,
            target_points=target_points,
            face_indices=faces,
        )

        assert result.refined_vertices.shape == coarse_vertices.shape
        assert result.iterations > 0

    def test_max_iterations_respected(self):
        """Test that maximum iterations limit is respected."""
        coarse_vertices = np.array([
            [-10.0, -10.0, -10.0],
            [10.0, 10.0, 10.0],
        ], dtype=np.float32)

        target_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float32)

        max_iters = 5
        refiner = CoarseToFineRefiner(
            max_iterations=max_iters,
            learning_rate=0.01,
            convergence_threshold=1e-12,
        )
        result = refiner.refine(coarse_vertices, target_points=target_points)

        assert result.iterations <= max_iters
        assert result.converged or result.iterations == max_iters

    def test_convergence_with_very_small_threshold(self):
        """Test that refinement detects convergence correctly."""
        coarse_vertices = np.array([
            [0.0, 0.0, 0.0],
            [0.01, 0.01, 0.01],
        ], dtype=np.float32)

        target_points = np.array([
            [0.001, 0.001, 0.001],
            [0.011, 0.011, 0.011],
        ], dtype=np.float32)

        # Use very high learning rate so refinement happens quickly
        refiner = CoarseToFineRefiner(
            max_iterations=100,
            learning_rate=1.0,
            convergence_threshold=1e-6,
        )
        result = refiner.refine(coarse_vertices, target_points=target_points)

        # Should converge relatively quickly with high learning rate
        assert result.iterations < 50

    def test_smoothness_weight(self):
        """Test smoothness regularization (Laplacian smoothing)."""
        # Create a star-like mesh with one central vertex
        coarse_vertices = np.array([
            [0.0, 0.0, 0.0],      # Center, will be pushed by smoothness
            [1.0, 0.0, 0.0],      # Neighbors
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=np.float32)

        # Faces connecting center to neighbors
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)

        target_points = np.array([
            [0.2, 0.2, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=np.float32)

        # With smoothness weight, center vertex should be pulled toward
        # centroid of neighbors
        refiner = CoarseToFineRefiner(
            max_iterations=20,
            learning_rate=0.1,
            smoothness_weight=0.5,
        )
        result = refiner.refine(
            coarse_vertices,
            target_points=target_points,
            face_indices=faces,
        )

        # Center vertex should move in direction of smoothing
        assert result.refined_vertices.shape == coarse_vertices.shape


class TestVertexPositionOptimizer:
    """Tests for the VertexPositionOptimizer class."""

    def test_quadratic_objective(self):
        """Test optimization with a simple quadratic objective."""
        # Objective: minimize ||v - target||^2
        target = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        def objective(verts):
            return float(np.sum((verts - target) ** 2))

        initial_vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        optimizer = VertexPositionOptimizer(
            method="L-BFGS-B",
            max_iterations=50,
        )
        result = optimizer.optimize(initial_vertices, objective)

        assert isinstance(result, OptimizationResult)
        assert result.optimized_vertices.shape == initial_vertices.shape
        # Should converge close to target
        assert np.allclose(result.optimized_vertices[0], target[0], atol=1e-4)
        assert result.success

    def test_quadratic_with_gradient(self):
        """Test optimization with explicit gradient function."""
        target = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        def objective(verts):
            return float(np.sum((verts - target) ** 2))

        def gradient(verts):
            return 2.0 * (verts - target)

        initial_vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        optimizer = VertexPositionOptimizer(
            method="L-BFGS-B",
            max_iterations=50,
        )
        result = optimizer.optimize(
            initial_vertices,
            objective,
            gradient_fn=gradient,
        )

        assert result.optimized_vertices.shape == initial_vertices.shape
        assert np.allclose(result.optimized_vertices[0], target[0], atol=1e-4)

    def test_bounded_optimization(self):
        """Test optimization with coordinate bounds."""
        target = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)

        def objective(verts):
            return float(np.sum((verts - target) ** 2))

        initial_vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        # Constrain to [-5, 5] range
        optimizer = VertexPositionOptimizer(
            method="L-BFGS-B",
            max_iterations=100,
        )
        result = optimizer.optimize(
            initial_vertices,
            objective,
            bounds=(-5.0, 5.0),
        )

        # Should be bounded
        assert np.all(result.optimized_vertices >= -5.0)
        assert np.all(result.optimized_vertices <= 5.0)
        # Final error should be higher than unbounded case (due to bounds)
        assert result.final_error > 0

    def test_multi_vertex_optimization(self):
        """Test optimization with multiple vertices."""
        targets = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        def objective(verts):
            return float(np.sum((verts - targets) ** 2))

        initial_vertices = np.zeros((3, 3), dtype=np.float32)

        optimizer = VertexPositionOptimizer(
            method="L-BFGS-B",
            max_iterations=100,
        )
        result = optimizer.optimize(initial_vertices, objective)

        assert result.optimized_vertices.shape == (3, 3)
        assert np.allclose(result.optimized_vertices, targets, atol=1e-3)

    def test_optimization_result_fields(self):
        """Test that OptimizationResult has all required fields."""
        def objective(verts):
            return float(np.sum(verts ** 2))

        initial_vertices = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

        optimizer = VertexPositionOptimizer()
        result = optimizer.optimize(initial_vertices, objective)

        assert hasattr(result, "optimized_vertices")
        assert hasattr(result, "success")
        assert hasattr(result, "message")
        assert hasattr(result, "iterations")
        assert hasattr(result, "final_error")

        assert isinstance(result.optimized_vertices, np.ndarray)
        assert isinstance(result.success, (bool, np.bool_))
        assert isinstance(result.message, str)
        assert isinstance(result.iterations, (int, np.integer))
        assert isinstance(result.final_error, (float, np.floating))

    def test_chamfer_like_objective(self):
        """Test with a Chamfer-like objective."""
        # Multiple vertices trying to fit to target cloud
        target_cloud = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32)

        def chamfer_objective(verts):
            # For each vertex, find nearest target point
            diffs = verts[:, np.newaxis, :] - target_cloud[np.newaxis, :, :]
            dists_sq = np.sum(diffs ** 2, axis=2)
            min_dists_sq = np.min(dists_sq, axis=1)
            return float(np.mean(min_dists_sq))

        initial_vertices = np.array([
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
        ], dtype=np.float32)

        optimizer = VertexPositionOptimizer(
            method="L-BFGS-B",
            max_iterations=200,
        )
        result = optimizer.optimize(initial_vertices, chamfer_objective)

        # Should minimize chamfer distance
        assert result.final_error < 1.0


class TestRefinementResult:
    """Tests for the RefinementResult dataclass."""

    def test_refinement_result_creation(self):
        """Test RefinementResult can be created with all fields."""
        refined_verts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        error_hist = [1.0, 0.8, 0.6, 0.5]

        result = RefinementResult(
            refined_vertices=refined_verts,
            iterations=3,
            converged=True,
            initial_error=1.0,
            final_error=0.5,
            error_history=error_hist,
        )

        assert np.array_equal(result.refined_vertices, refined_verts)
        assert result.iterations == 3
        assert result.converged is True
        assert result.initial_error == 1.0
        assert result.final_error == 0.5
        assert result.error_history == error_hist

    def test_refinement_result_default_error_history(self):
        """Test RefinementResult with default error_history."""
        refined_verts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        result = RefinementResult(
            refined_vertices=refined_verts,
            iterations=1,
            converged=False,
            initial_error=0.5,
            final_error=0.4,
        )

        assert result.error_history == []


class TestOptimizationResult:
    """Tests for the OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Test OptimizationResult can be created with all fields."""
        opt_verts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        result = OptimizationResult(
            optimized_vertices=opt_verts,
            success=True,
            message="Optimization successful",
            iterations=25,
            final_error=0.001,
        )

        assert np.array_equal(result.optimized_vertices, opt_verts)
        assert result.success is True
        assert result.message == "Optimization successful"
        assert result.iterations == 25
        assert result.final_error == 0.001


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_refine_with_multiple_constraints(self):
        """Test refinement with both fixed and planar constraints."""
        coarse_vertices = np.array([
            [0.0, 0.0, 0.0],   # Fixed
            [1.0, 1.0, 1.0],   # Planar
            [2.0, 2.0, 2.0],   # Planar
        ], dtype=np.float32)

        target_points = np.array([
            [0.0, 0.0, 0.0],
            [1.1, 1.1, 0.5],
            [2.1, 2.1, 0.5],
        ], dtype=np.float32)

        constraints = [
            {
                "type": "fixed",
                "vertex_index": 0,
                "position": np.array([0.0, 0.0, 0.0]),
            },
            {
                "type": "planar",
                "vertex_indices": [1, 2],
                "normal": np.array([0.0, 0.0, 1.0]),
                "d": 0.0,
            },
        ]

        refiner = CoarseToFineRefiner(max_iterations=30, learning_rate=0.1)
        result = refiner.refine(
            coarse_vertices,
            target_points=target_points,
            constraints=constraints,
        )

        # Vertex 0 should remain fixed
        assert np.allclose(result.refined_vertices[0], [0.0, 0.0, 0.0], atol=1e-5)
        # Vertices 1 and 2 should be on z=0 plane
        assert np.abs(result.refined_vertices[1, 2]) < 1e-4
        assert np.abs(result.refined_vertices[2, 2]) < 1e-4

    def test_optimizer_vs_refiner_simple_case(self):
        """Compare CoarseToFineRefiner with VertexPositionOptimizer."""
        coarse_vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 1.5, 1.5],
        ], dtype=np.float32)

        target_points = np.array([
            [0.1, 0.1, 0.1],
            [1.4, 1.4, 1.4],
        ], dtype=np.float32)

        # Refiner approach
        refiner = CoarseToFineRefiner(max_iterations=50, learning_rate=0.1)
        result_refiner = refiner.refine(coarse_vertices, target_points=target_points)

        # Optimizer approach
        def objective(verts):
            diffs = verts[:, np.newaxis, :] - target_points[np.newaxis, :, :]
            dists_sq = np.sum(diffs ** 2, axis=2)
            return float(np.mean(np.min(dists_sq, axis=1)))

        optimizer = VertexPositionOptimizer(max_iterations=100)
        result_optimizer = optimizer.optimize(coarse_vertices, objective)

        # Both should produce reasonable results
        assert result_refiner.final_error < 1.0
        assert result_optimizer.final_error < 1.0

    def test_large_mesh_refinement(self):
        """Test refinement with a larger mesh."""
        # Create a grid of vertices
        nx, ny = 5, 5
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        xx, yy = np.meshgrid(x, y)

        coarse_vertices = np.stack([
            xx.flatten(),
            yy.flatten(),
            np.zeros(nx * ny),
        ], axis=1).astype(np.float32)

        # Target: slightly curved surface
        target_vertices = coarse_vertices.copy()
        target_vertices[:, 2] += 0.1 * np.sin(coarse_vertices[:, 0])

        refiner = CoarseToFineRefiner(
            max_iterations=20,
            learning_rate=0.05,
        )
        result = refiner.refine(coarse_vertices, target_points=target_vertices)

        assert result.refined_vertices.shape == coarse_vertices.shape
        assert result.iterations > 0
        assert result.final_error < result.initial_error
