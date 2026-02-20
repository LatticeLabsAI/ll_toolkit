"""Tests for vertex prediction neural network modules.

Covers:
- VertexPredictionHead forward pass and output shapes
- VertexRefinementHead iterative refinement
- decode_vertices threshold behavior
- Integration with CompositeHead
- Gradient flow through the vertex heads
"""
import sys
import os
import pytest

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch")
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stepnet.vertex_prediction import (
    VertexPredictionHead,
    VertexRefinementHead,
    VertexPredictionOutput,
)
from stepnet.output_heads import CompositeHead


# ===========================================================================
# VertexPredictionHead Tests
# ===========================================================================


class TestVertexPredictionHeadForwardPass:
    """Test VertexPredictionHead forward pass with default parameters."""

    def test_forward_output_shapes(self):
        """Verify output shapes for default configuration."""
        batch_size = 4
        seq_len = 32
        embed_dim = 256
        max_vertices = 512

        head = VertexPredictionHead(
            embed_dim=embed_dim,
            max_vertices=max_vertices,
            hidden_dim=256,
            num_refinement_steps=3,
        )

        hidden_states = torch.randn(batch_size, seq_len, embed_dim)
        output = head(hidden_states)

        assert isinstance(output, VertexPredictionOutput)
        assert output.vertex_presence.shape == (batch_size, max_vertices)
        assert output.coarse_positions.shape == (batch_size, max_vertices, 3)
        assert output.refined_positions.shape == (batch_size, max_vertices, 3)

    def test_forward_single_batch(self):
        """Test with batch size 1."""
        head = VertexPredictionHead(embed_dim=256, max_vertices=256)

        hidden_states = torch.randn(1, 16, 256)
        output = head(hidden_states)

        assert output.vertex_presence.shape == (1, 256)
        assert output.coarse_positions.shape == (1, 256, 3)
        assert output.refined_positions.shape == (1, 256, 3)

    def test_forward_large_batch(self):
        """Test with large batch size."""
        batch_size = 32
        head = VertexPredictionHead(embed_dim=128, max_vertices=128)

        hidden_states = torch.randn(batch_size, 20, 128)
        output = head(hidden_states)

        assert output.vertex_presence.shape == (batch_size, 128)
        assert output.coarse_positions.shape == (batch_size, 128, 3)
        assert output.refined_positions.shape == (batch_size, 128, 3)

    def test_coarse_positions_are_tanh_clamped(self):
        """Verify coarse positions are clamped to [-1, 1]."""
        head = VertexPredictionHead(embed_dim=256, max_vertices=512)

        hidden_states = torch.randn(4, 32, 256)
        output = head(hidden_states)

        # All coarse positions should be in [-1, 1] due to tanh
        assert torch.all(output.coarse_positions >= -1.0)
        assert torch.all(output.coarse_positions <= 1.0)

    def test_vertex_presence_logits_unbounded(self):
        """Verify vertex presence logits are unbounded (not squashed)."""
        head = VertexPredictionHead(embed_dim=256, max_vertices=512)

        hidden_states = torch.randn(4, 32, 256)
        output = head(hidden_states)

        # Logits should be unbounded (could be any real value)
        # Just verify they're not all in [0, 1] range
        presence_min = output.vertex_presence.min().item()
        presence_max = output.vertex_presence.max().item()
        # Very unlikely that random logits are all constrained to [0, 1]
        assert not (presence_min >= 0.0 and presence_max <= 1.0)


class TestVertexPredictionHeadParameterVariations:
    """Test VertexPredictionHead with different configurations."""

    def test_different_embed_dim(self):
        """Test with different embedding dimensions."""
        for embed_dim in [64, 128, 512]:
            head = VertexPredictionHead(
                embed_dim=embed_dim,
                max_vertices=256,
                hidden_dim=256,
            )
            hidden_states = torch.randn(2, 16, embed_dim)
            output = head(hidden_states)

            assert output.vertex_presence.shape == (2, 256)
            assert output.coarse_positions.shape == (2, 256, 3)

    def test_different_max_vertices(self):
        """Test with different maximum vertex counts."""
        for max_vertices in [64, 256, 1024]:
            head = VertexPredictionHead(
                embed_dim=256,
                max_vertices=max_vertices,
                hidden_dim=256,
            )
            hidden_states = torch.randn(2, 16, 256)
            output = head(hidden_states)

            assert output.vertex_presence.shape == (2, max_vertices)
            assert output.coarse_positions.shape == (2, max_vertices, 3)
            assert output.refined_positions.shape == (2, max_vertices, 3)

    def test_different_hidden_dim(self):
        """Test with different hidden dimensions."""
        for hidden_dim in [64, 128, 512]:
            head = VertexPredictionHead(
                embed_dim=256,
                max_vertices=256,
                hidden_dim=hidden_dim,
            )
            hidden_states = torch.randn(2, 16, 256)
            output = head(hidden_states)

            assert output.vertex_presence.shape == (2, 256)
            assert output.coarse_positions.shape == (2, 256, 3)

    def test_all_parameters_different(self):
        """Test with all parameters at non-default values."""
        head = VertexPredictionHead(
            embed_dim=384,
            max_vertices=768,
            hidden_dim=384,
            num_refinement_steps=5,
        )
        hidden_states = torch.randn(8, 24, 384)
        output = head(hidden_states)

        assert output.vertex_presence.shape == (8, 768)
        assert output.coarse_positions.shape == (8, 768, 3)
        assert output.refined_positions.shape == (8, 768, 3)


class TestVertexPredictionHeadNoRefinement:
    """Test VertexPredictionHead with num_refinement_steps=0."""

    def test_no_refinement_coarse_equals_refined(self):
        """When num_refinement_steps=0, coarse should equal refined."""
        head = VertexPredictionHead(
            embed_dim=256,
            max_vertices=256,
            hidden_dim=256,
            num_refinement_steps=0,
        )

        hidden_states = torch.randn(4, 16, 256)
        output = head(hidden_states)

        # Should be exactly equal (same tensor)
        assert torch.equal(output.coarse_positions, output.refined_positions)

    def test_no_refinement_no_refiner_module(self):
        """Verify refiner is None when num_refinement_steps=0."""
        head = VertexPredictionHead(
            embed_dim=256,
            max_vertices=256,
            num_refinement_steps=0,
        )

        assert head.refiner is None

    def test_refinement_has_refiner_module(self):
        """Verify refiner exists when num_refinement_steps > 0."""
        head = VertexPredictionHead(
            embed_dim=256,
            max_vertices=256,
            num_refinement_steps=3,
        )

        assert head.refiner is not None
        assert isinstance(head.refiner, VertexRefinementHead)


# ===========================================================================
# VertexRefinementHead Tests
# ===========================================================================


class TestVertexRefinementHeadForwardPass:
    """Test VertexRefinementHead forward pass."""

    def test_forward_output_shape(self):
        """Verify output shape matches input shape."""
        batch_size = 4
        num_vertices = 128
        vertex_dim = 3

        refiner = VertexRefinementHead(
            vertex_dim=vertex_dim,
            context_dim=256,
            num_iterations=3,
            hidden_dim=128,
        )

        coarse_positions = torch.randn(batch_size, num_vertices, vertex_dim)
        context = torch.randn(batch_size, 256)

        refined = refiner(coarse_positions, context=context)

        assert refined.shape == coarse_positions.shape
        assert refined.shape == (batch_size, num_vertices, vertex_dim)

    def test_forward_without_context(self):
        """Test refinement with context=None (uses zero context)."""
        batch_size = 4
        num_vertices = 128
        vertex_dim = 3

        refiner = VertexRefinementHead(
            vertex_dim=vertex_dim,
            context_dim=256,
            num_iterations=3,
            hidden_dim=128,
        )

        coarse_positions = torch.randn(batch_size, num_vertices, vertex_dim)

        # Should work without context
        refined = refiner(coarse_positions, context=None)

        assert refined.shape == (batch_size, num_vertices, vertex_dim)

    def test_forward_with_context_none_vs_zeros(self):
        """Verify None context behaves like zero context."""
        batch_size = 2
        num_vertices = 64
        vertex_dim = 3
        context_dim = 256

        refiner = VertexRefinementHead(
            vertex_dim=vertex_dim,
            context_dim=context_dim,
            num_iterations=3,
            hidden_dim=128,
        )

        coarse_positions = torch.randn(batch_size, num_vertices, vertex_dim)

        # Set seed for reproducibility
        torch.manual_seed(42)
        refined_no_context = refiner(coarse_positions, context=None)

        # Reset seed to get same random initialization
        torch.manual_seed(42)
        refiner_copy = VertexRefinementHead(
            vertex_dim=vertex_dim,
            context_dim=context_dim,
            num_iterations=3,
            hidden_dim=128,
        )
        # Copy weights
        for p1, p2 in zip(refiner.parameters(), refiner_copy.parameters()):
            p2.data.copy_(p1.data)

        zero_context = torch.zeros(batch_size, context_dim)
        refined_zero_context = refiner_copy(coarse_positions, context=zero_context)

        # Results should be identical
        assert torch.allclose(refined_no_context, refined_zero_context, atol=1e-6)

    def test_refinement_different_iterations(self):
        """Test with different numbers of iterations."""
        for num_iterations in [1, 3, 5, 10]:
            refiner = VertexRefinementHead(
                vertex_dim=3,
                context_dim=256,
                num_iterations=num_iterations,
                hidden_dim=128,
            )

            coarse_positions = torch.randn(4, 128, 3)
            context = torch.randn(4, 256)

            refined = refiner(coarse_positions, context=context)
            assert refined.shape == (4, 128, 3)

    def test_refinement_improves_over_iterations(self):
        """Verify that refinement layers are actually being applied.

        After refinement, positions should differ from coarse positions.
        """
        refiner = VertexRefinementHead(
            vertex_dim=3,
            context_dim=256,
            num_iterations=3,
            hidden_dim=128,
        )

        coarse_positions = torch.randn(4, 128, 3)
        context = torch.randn(4, 256)

        refined = refiner(coarse_positions, context=context)

        # Refined positions should be different from coarse
        # (extremely unlikely to be identical after 3 refinement steps)
        assert not torch.allclose(coarse_positions, refined, atol=1e-6)


class TestVertexRefinementHeadDifferentConfigurations:
    """Test VertexRefinementHead with different configurations."""

    def test_different_context_dim(self):
        """Test with different context dimensions."""
        for context_dim in [64, 256, 512]:
            refiner = VertexRefinementHead(
                vertex_dim=3,
                context_dim=context_dim,
                num_iterations=3,
                hidden_dim=128,
            )

            coarse_positions = torch.randn(4, 128, 3)
            context = torch.randn(4, context_dim)

            refined = refiner(coarse_positions, context=context)
            assert refined.shape == (4, 128, 3)

    def test_different_hidden_dim(self):
        """Test with different hidden dimensions."""
        for hidden_dim in [64, 256, 512]:
            refiner = VertexRefinementHead(
                vertex_dim=3,
                context_dim=256,
                num_iterations=3,
                hidden_dim=hidden_dim,
            )

            coarse_positions = torch.randn(4, 128, 3)
            context = torch.randn(4, 256)

            refined = refiner(coarse_positions, context=context)
            assert refined.shape == (4, 128, 3)


# ===========================================================================
# decode_vertices Tests
# ===========================================================================


class TestDecodeVerticesThreshold:
    """Test decode_vertices method and threshold behavior."""

    def test_decode_vertices_high_threshold(self):
        """High threshold results in fewer active vertices."""
        head = VertexPredictionHead(embed_dim=256, max_vertices=256)

        hidden_states = torch.randn(2, 16, 256)
        output = head(hidden_states)

        # Decode with high threshold (e.g., 0.9)
        vertices_high, mask_high = head.decode_vertices(output, threshold=0.9)
        # Decode with low threshold (e.g., 0.1)
        vertices_low, mask_low = head.decode_vertices(output, threshold=0.1)

        # High threshold should give fewer or equal vertices
        num_high = len(vertices_high)
        num_low = len(vertices_low)

        assert num_high <= num_low

    def test_decode_vertices_output_shapes(self):
        """Verify decode_vertices returns correct shapes."""
        head = VertexPredictionHead(embed_dim=256, max_vertices=256)

        hidden_states = torch.randn(4, 16, 256)
        output = head(hidden_states)

        # decode_vertices operates on first batch element
        vertices, mask = head.decode_vertices(output, threshold=0.5)

        # vertices should be (K, 3) where K <= max_vertices
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert vertices.shape[0] <= 256
        assert isinstance(vertices, np.ndarray)
        assert vertices.dtype == np.float32

        # mask should be (max_vertices,) bool
        assert mask.ndim == 1
        assert mask.shape[0] == 256
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_decode_vertices_threshold_zero(self):
        """With threshold=0, all vertices with positive logit are included."""
        head = VertexPredictionHead(embed_dim=256, max_vertices=256)

        hidden_states = torch.randn(2, 16, 256)
        output = head(hidden_states)

        vertices, mask = head.decode_vertices(output, threshold=0.0)

        # Should include all vertices with sigmoid > 0 (roughly half)
        # But at least some should be included
        num_vertices = len(vertices)
        assert num_vertices > 0

    def test_decode_vertices_threshold_one(self):
        """With threshold=1.0, almost no vertices are included."""
        head = VertexPredictionHead(embed_dim=256, max_vertices=256)

        hidden_states = torch.randn(2, 16, 256)
        output = head(hidden_states)

        vertices, mask = head.decode_vertices(output, threshold=1.0)

        # Very few or no vertices should be included
        num_vertices = len(vertices)
        assert num_vertices <= 3  # Allow tiny numerical errors

    def test_decode_vertices_uses_refined_positions(self):
        """Verify decode_vertices uses refined_positions, not coarse."""
        head = VertexPredictionHead(
            embed_dim=256, max_vertices=256, num_refinement_steps=3
        )

        hidden_states = torch.randn(2, 16, 256)
        output = head(hidden_states)

        vertices, mask = head.decode_vertices(output, threshold=0.5)

        # Filter refined positions by mask
        expected_vertices = output.refined_positions[0, mask].detach().cpu().numpy()

        # Should match what decode_vertices returns
        assert np.allclose(vertices, expected_vertices, atol=1e-6)

    def test_decode_vertices_first_batch_only(self):
        """Verify decode_vertices only operates on first batch element."""
        head = VertexPredictionHead(embed_dim=256, max_vertices=256)

        hidden_states = torch.randn(4, 16, 256)
        output = head(hidden_states)

        vertices, mask = head.decode_vertices(output, threshold=0.5)

        # Result should only use first batch element
        expected = output.refined_positions[0, mask].detach().cpu().numpy()
        assert np.allclose(vertices, expected, atol=1e-6)


class TestDecodeVerticesNoGradient:
    """Test that decode_vertices is no_grad."""

    def test_decode_vertices_no_grad_context(self):
        """decode_vertices should not create computational graph."""
        head = VertexPredictionHead(embed_dim=256, max_vertices=256)

        hidden_states = torch.randn(2, 16, 256)
        output = head(hidden_states)

        # decode_vertices is marked with @torch.no_grad()
        vertices, mask = head.decode_vertices(output, threshold=0.5)

        # Returned arrays should not have gradients
        assert isinstance(vertices, np.ndarray)
        assert isinstance(mask, np.ndarray)


# ===========================================================================
# Integration with CompositeHead Tests
# ===========================================================================


class TestVertexPredictionIntegration:
    """Test VertexPredictionHead integration with CompositeHead."""

    def test_composite_head_without_vertex_head(self):
        """CompositeHead without vertex_head should not have vertex_prediction key."""
        composite = CompositeHead(
            embed_dim=256,
            num_command_types=6,
            num_param_slots=16,
            num_levels=256,
            include_vertex_head=False,
        )

        hidden_states = torch.randn(4, 32, 256)
        result = composite(hidden_states)

        assert "vertex_prediction" not in result
        assert "command_type_logits" in result
        assert "parameter_logits" in result

    def test_composite_head_with_vertex_head(self):
        """CompositeHead with include_vertex_head=True should have vertex_prediction key."""
        composite = CompositeHead(
            embed_dim=256,
            num_command_types=6,
            num_param_slots=16,
            num_levels=256,
            include_vertex_head=True,
            max_vertices=512,
            num_refinement_steps=3,
        )

        hidden_states = torch.randn(4, 32, 256)
        result = composite(hidden_states)

        assert "vertex_prediction" in result
        assert isinstance(result["vertex_prediction"], VertexPredictionOutput)
        assert result["vertex_prediction"].vertex_presence.shape == (4, 512)
        assert result["vertex_prediction"].coarse_positions.shape == (4, 512, 3)
        assert result["vertex_prediction"].refined_positions.shape == (4, 512, 3)

    def test_composite_head_vertex_head_is_vertex_prediction_head(self):
        """Verify CompositeHead.vertex_head is a VertexPredictionHead instance."""
        composite = CompositeHead(
            embed_dim=256,
            include_vertex_head=True,
            max_vertices=512,
            num_refinement_steps=3,
        )

        assert composite.vertex_head is not None
        assert isinstance(composite.vertex_head, VertexPredictionHead)

    def test_composite_head_vertex_parameters_passed_correctly(self):
        """Verify vertex head is initialized with correct parameters."""
        max_vertices = 768
        num_refinement_steps = 5

        composite = CompositeHead(
            embed_dim=256,
            include_vertex_head=True,
            max_vertices=max_vertices,
            num_refinement_steps=num_refinement_steps,
        )

        assert composite.vertex_head.max_vertices == max_vertices
        assert composite.vertex_head.refiner is not None

    def test_composite_head_command_and_vertex_outputs_independent(self):
        """Verify command predictions are independent of vertex predictions."""
        composite = CompositeHead(
            embed_dim=256,
            num_command_types=6,
            num_param_slots=16,
            num_levels=256,
            include_vertex_head=True,
            max_vertices=256,
        )

        hidden_states = torch.randn(4, 32, 256)
        result = composite(hidden_states)

        # All keys should be present
        assert "command_type_logits" in result
        assert "parameter_logits" in result
        assert "vertex_prediction" in result
        assert "parameter_mask" in result

        # Command logits should have correct shape (independent of vertex)
        assert result["command_type_logits"].shape == (4, 32, 6)


# ===========================================================================
# Gradient Flow Tests
# ===========================================================================


class TestGradientFlow:
    """Test that gradients propagate through vertex prediction modules."""

    def test_vertex_prediction_head_gradients(self):
        """Verify gradients flow through VertexPredictionHead."""
        head = VertexPredictionHead(
            embed_dim=256, max_vertices=256, num_refinement_steps=3
        )

        hidden_states = torch.randn(2, 16, 256, requires_grad=True)
        output = head(hidden_states)

        # Create a simple loss
        loss = output.refined_positions.mean()
        loss.backward()

        # Check gradients exist
        assert hidden_states.grad is not None
        assert hidden_states.grad.shape == hidden_states.shape
        assert not torch.isnan(hidden_states.grad).any()

    def test_vertex_refinement_head_gradients(self):
        """Verify gradients flow through VertexRefinementHead."""
        refiner = VertexRefinementHead(
            vertex_dim=3,
            context_dim=256,
            num_iterations=3,
            hidden_dim=128,
        )

        coarse_positions = torch.randn(2, 128, 3, requires_grad=True)
        context = torch.randn(2, 256, requires_grad=True)

        refined = refiner(coarse_positions, context=context)

        # Create a simple loss
        loss = refined.mean()
        loss.backward()

        # Check gradients exist
        assert coarse_positions.grad is not None
        assert context.grad is not None
        assert not torch.isnan(coarse_positions.grad).any()
        assert not torch.isnan(context.grad).any()

    def test_composite_head_with_vertex_gradients(self):
        """Verify gradients flow through CompositeHead with vertex_head."""
        composite = CompositeHead(
            embed_dim=256,
            include_vertex_head=True,
            max_vertices=256,
            num_refinement_steps=3,
        )

        hidden_states = torch.randn(2, 16, 256, requires_grad=True)
        result = composite(hidden_states)

        # Create a combined loss from both heads
        cmd_loss = result["command_type_logits"].mean()
        vertex_loss = result["vertex_prediction"].refined_positions.mean()
        total_loss = cmd_loss + vertex_loss

        total_loss.backward()

        # Gradients should flow to input
        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()

    def test_vertex_head_gradients_independent_of_command_head(self):
        """Verify vertex head gradients don't interfere with command head gradients."""
        composite = CompositeHead(
            embed_dim=256,
            include_vertex_head=True,
            max_vertices=256,
        )

        hidden_states = torch.randn(2, 16, 256, requires_grad=True)
        result = composite(hidden_states)

        # Loss from vertex head only
        vertex_loss = result["vertex_prediction"].refined_positions.sum()
        vertex_loss.backward()

        vertex_grad = hidden_states.grad.clone()

        # Reset gradients
        hidden_states.grad = None

        # Loss from command head only
        cmd_loss = result["command_type_logits"].sum()
        cmd_loss.backward()

        cmd_grad = hidden_states.grad.clone()

        # Gradients should be different (non-zero for both)
        assert not torch.allclose(vertex_grad, cmd_grad)
        assert not torch.allclose(vertex_grad, torch.zeros_like(vertex_grad))
        assert not torch.allclose(cmd_grad, torch.zeros_like(cmd_grad))


# ===========================================================================
# Edge Cases and Robustness Tests
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_very_small_hidden_states(self):
        """Test with minimal sequence length."""
        head = VertexPredictionHead(embed_dim=256, max_vertices=256)

        hidden_states = torch.randn(1, 1, 256)
        output = head(hidden_states)

        assert output.refined_positions.shape == (1, 256, 3)

    def test_very_large_sequence(self):
        """Test with very long sequence."""
        head = VertexPredictionHead(embed_dim=256, max_vertices=256)

        hidden_states = torch.randn(2, 1024, 256)
        output = head(hidden_states)

        assert output.refined_positions.shape == (2, 256, 3)

    def test_zero_input_values(self):
        """Test with zero-valued hidden states."""
        head = VertexPredictionHead(embed_dim=256, max_vertices=256)

        hidden_states = torch.zeros(2, 16, 256)
        output = head(hidden_states)

        # Should still produce valid output
        assert output.refined_positions.shape == (2, 256, 3)
        assert not torch.isnan(output.refined_positions).any()

    def test_large_input_values(self):
        """Test with large-valued hidden states."""
        head = VertexPredictionHead(embed_dim=256, max_vertices=256)

        hidden_states = torch.randn(2, 16, 256) * 100
        output = head(hidden_states)

        # Should still produce valid output (tanh clamps to [-1, 1])
        assert torch.all(output.coarse_positions >= -1.0)
        assert torch.all(output.coarse_positions <= 1.0)
        assert not torch.isnan(output.refined_positions).any()

    def test_batch_consistency(self):
        """Verify that batch dimension is handled correctly."""
        head = VertexPredictionHead(embed_dim=256, max_vertices=256)

        hidden_states_single = torch.randn(1, 16, 256)
        output_single = head(hidden_states_single)

        # Process same data in batch of 4 (repeated)
        hidden_states_batch = hidden_states_single.repeat(4, 1, 1)
        output_batch = head(hidden_states_batch)

        # All batch elements should be identical
        for i in range(4):
            assert torch.allclose(
                output_single.refined_positions[0],
                output_batch.refined_positions[i],
                atol=1e-6,
            )

    def test_deterministic_with_seed(self):
        """Verify outputs are deterministic with fixed seed."""
        torch.manual_seed(42)
        head1 = VertexPredictionHead(embed_dim=256, max_vertices=256)
        hidden_states = torch.randn(2, 16, 256)
        output1 = head1(hidden_states)

        torch.manual_seed(42)
        head2 = VertexPredictionHead(embed_dim=256, max_vertices=256)
        output2 = head2(hidden_states)

        assert torch.allclose(
            output1.refined_positions, output2.refined_positions, atol=1e-6
        )


# ===========================================================================
# Device Compatibility Tests
# ===========================================================================


class TestDeviceCompatibility:
    """Test device compatibility (CPU and CUDA if available)."""

    def test_cpu_device(self):
        """Test on CPU device."""
        device = torch.device("cpu")

        head = VertexPredictionHead(embed_dim=256, max_vertices=256)
        head = head.to(device)

        hidden_states = torch.randn(2, 16, 256, device=device)
        output = head(hidden_states)

        assert output.refined_positions.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test on CUDA device if available."""
        device = torch.device("cuda")

        head = VertexPredictionHead(embed_dim=256, max_vertices=256)
        head = head.to(device)

        hidden_states = torch.randn(2, 16, 256, device=device)
        output = head(hidden_states)

        assert output.refined_positions.device.type == "cuda"

    def test_refinement_head_device_handling(self):
        """Test VertexRefinementHead device handling."""
        device = torch.device("cpu")

        refiner = VertexRefinementHead(
            vertex_dim=3, context_dim=256, num_iterations=3, hidden_dim=128
        )
        refiner = refiner.to(device)

        coarse_positions = torch.randn(2, 128, 3, device=device)
        context = torch.randn(2, 256, device=device)

        refined = refiner(coarse_positions, context=context)

        assert refined.device.type == device.type


# ===========================================================================
# Dataclass Tests
# ===========================================================================


class TestVertexPredictionOutput:
    """Test VertexPredictionOutput dataclass."""

    def test_output_dataclass_creation(self):
        """Verify VertexPredictionOutput can be created and accessed."""
        batch_size = 4
        max_vertices = 256

        vertex_presence = torch.randn(batch_size, max_vertices)
        coarse_positions = torch.randn(batch_size, max_vertices, 3)
        refined_positions = torch.randn(batch_size, max_vertices, 3)

        output = VertexPredictionOutput(
            vertex_presence=vertex_presence,
            coarse_positions=coarse_positions,
            refined_positions=refined_positions,
        )

        assert torch.equal(output.vertex_presence, vertex_presence)
        assert torch.equal(output.coarse_positions, coarse_positions)
        assert torch.equal(output.refined_positions, refined_positions)

    def test_output_dataclass_attributes(self):
        """Verify all required attributes are present."""
        output = VertexPredictionOutput(
            vertex_presence=torch.randn(4, 256),
            coarse_positions=torch.randn(4, 256, 3),
            refined_positions=torch.randn(4, 256, 3),
        )

        assert hasattr(output, "vertex_presence")
        assert hasattr(output, "coarse_positions")
        assert hasattr(output, "refined_positions")
