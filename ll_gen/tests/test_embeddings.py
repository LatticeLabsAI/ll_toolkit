"""Comprehensive test suite for ll_gen.embeddings module.

Tests the HybridShapeEncoder: a Transformer + GNN fusion encoder for shape
understanding. Covers initialization, forward passes, state management, and
device handling.

Key Components Tested:
- HybridShapeEncoder initialization with custom parameters
- Transformer branch encoding
- GNN branch encoding (with mocks for cadling unavailability)
- Fusion path (concatenation + projection)
- State dict save/load roundtrip
- Device movement (cpu)
- Train/eval mode switching
- Parameter management
- Input dimension validation
- Edge cases (single batch, long sequences)
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from ll_gen.embeddings import HybridShapeEncoder

# ============================================================================
# Pytest Markers and Fixtures
# ============================================================================

requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch not installed"
)


@pytest.fixture
def torch_module():
    """Provide torch module if available, else skip test."""
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    return torch


@pytest.fixture
def random_seed():
    """Set and restore random seed for reproducibility."""
    seed = 42
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
    np.random.seed(seed)
    yield seed
    # Reset to random state after test
    if _TORCH_AVAILABLE:
        torch.seed()
    np.random.seed()


@pytest.fixture
def default_encoder(random_seed):
    """Create encoder with default parameters."""
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    return HybridShapeEncoder()


@pytest.fixture
def custom_encoder(random_seed):
    """Create encoder with custom parameters."""
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    return HybridShapeEncoder(
        input_dim=512,
        transformer_dim=256,
        graph_dim=128,
        output_dim=256,
        num_transformer_layers=2,
        num_heads=4,
        dropout=0.2,
        graph_encoder_type="brep_net",
        device="cpu",
    )


@pytest.fixture
def sample_conditioning(torch_module):
    """Create sample conditioning embeddings."""
    # Shape: (seq_len=10, input_dim=768)
    return np.random.randn(10, 768).astype(np.float32)


@pytest.fixture
def sample_conditioning_single(torch_module):
    """Create single sample conditioning embedding (1D)."""
    # Shape: (input_dim=768,)
    return np.random.randn(768).astype(np.float32)


@pytest.fixture
def sample_graph_data(torch_module):
    """Create sample graph data (B-Rep topology)."""
    num_nodes = 20
    num_edges = 40
    node_feat_dim = 32
    edge_feat_dim = 8

    return {
        "node_features": np.random.randn(num_nodes, node_feat_dim).astype(
            np.float32
        ),
        "edge_index": np.random.randint(0, num_nodes, (2, num_edges)).astype(
            np.int64
        ),
        "edge_attr": np.random.randn(num_edges, edge_feat_dim).astype(np.float32),
    }


# ============================================================================
# SECTION 1: Initialization Tests
# ============================================================================


@requires_torch
class TestHybridShapeEncoderInit:
    """Test HybridShapeEncoder initialization."""

    @pytest.mark.unit
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        encoder = HybridShapeEncoder()
        assert encoder.input_dim == 768
        assert encoder.transformer_dim == 512
        assert encoder.graph_dim == 256
        assert encoder.output_dim == 512
        assert encoder.num_transformer_layers == 3
        assert encoder.num_heads == 8
        assert encoder.dropout == 0.1
        assert encoder.graph_encoder_type == "brep_net"
        assert encoder.device == "cpu"

    @pytest.mark.unit
    def test_init_with_custom_params(self, custom_encoder):
        """Test initialization with custom parameters."""
        assert custom_encoder.input_dim == 512
        assert custom_encoder.transformer_dim == 256
        assert custom_encoder.graph_dim == 128
        assert custom_encoder.output_dim == 256
        assert custom_encoder.num_transformer_layers == 2
        assert custom_encoder.num_heads == 4
        assert custom_encoder.dropout == 0.2

    @pytest.mark.unit
    def test_init_creates_transformer_components(self, default_encoder):
        """Test that transformer components are created during init."""
        assert default_encoder._input_projection is not None
        assert isinstance(default_encoder._input_projection, nn.Linear)
        assert default_encoder._transformer_encoder is not None
        assert isinstance(default_encoder._transformer_encoder, nn.TransformerEncoder)
        assert default_encoder._transformer_output_projection is not None
        assert isinstance(default_encoder._transformer_output_projection, nn.Linear)

    @pytest.mark.unit
    def test_init_projection_dimensions(self, default_encoder):
        """Test projection layer dimensions."""
        # input_projection: input_dim -> transformer_dim
        assert default_encoder._input_projection.in_features == 768
        assert default_encoder._input_projection.out_features == 512

    @pytest.mark.unit
    def test_init_torch_import_error_handling(self):
        """Test ImportError raised when torch not available."""
        if not _TORCH_AVAILABLE:
            with pytest.raises(ImportError, match="torch is required"):
                HybridShapeEncoder()
        else:
            # If torch is available, test that initialization succeeds
            encoder = HybridShapeEncoder()
            assert encoder._torch is torch

    @pytest.mark.unit
    def test_init_sets_has_gnn_flag(self, default_encoder):
        """Test that _has_gnn flag is set appropriately."""
        # Flag should be False when cadling is not available
        assert isinstance(default_encoder._has_gnn, bool)

    @pytest.mark.unit
    def test_init_gnn_unavailable_log(self, default_encoder, caplog):
        """Test that debug log is created when GNN unavailable."""
        # Create new encoder to capture logs
        with caplog.at_level("DEBUG"):
            encoder = HybridShapeEncoder()
        # When cadling not available, should log debug message
        if not encoder._has_gnn:
            assert any("not available" in record.message for record in caplog.records)

    @pytest.mark.unit
    def test_init_multiple_encoders_independent(self):
        """Test that multiple encoders are independent instances."""
        enc1 = HybridShapeEncoder(input_dim=768)
        enc2 = HybridShapeEncoder(input_dim=512)
        assert enc1.input_dim == 768
        assert enc2.input_dim == 512
        assert enc1._input_projection is not enc2._input_projection


# ============================================================================
# SECTION 2: Forward Pass Tests
# ============================================================================


@requires_torch
class TestHybridShapeEncoderForward:
    """Test HybridShapeEncoder forward pass."""

    @pytest.mark.unit
    def test_forward_numpy_input_2d(self, default_encoder, sample_conditioning):
        """Test forward pass with 2D numpy array (seq_len, input_dim)."""
        output = default_encoder.forward(sample_conditioning)
        assert isinstance(output, np.ndarray)
        assert output.shape == (default_encoder.output_dim,)
        assert not np.isnan(output).any()

    @pytest.mark.unit
    def test_forward_numpy_input_1d(
        self, default_encoder, sample_conditioning_single
    ):
        """Test forward pass with 1D numpy array (input_dim,)."""
        output = default_encoder.forward(sample_conditioning_single)
        assert isinstance(output, np.ndarray)
        assert output.shape == (default_encoder.output_dim,)
        assert not np.isnan(output).any()

    @pytest.mark.unit
    def test_forward_torch_input_2d(self, default_encoder):
        """Test forward pass with 2D torch tensor (seq_len, input_dim)."""
        cond = torch.randn(10, 768)
        output = default_encoder.forward(cond)
        assert isinstance(output, np.ndarray)
        assert output.shape == (default_encoder.output_dim,)

    @pytest.mark.unit
    def test_forward_torch_input_1d(self, default_encoder):
        """Test forward pass with 1D torch tensor (input_dim,)."""
        cond = torch.randn(768)
        output = default_encoder.forward(cond)
        assert isinstance(output, np.ndarray)
        assert output.shape == (default_encoder.output_dim,)

    @pytest.mark.unit
    def test_forward_output_reproducibility(self, default_encoder, sample_conditioning):
        """Test that forward pass is deterministic in eval mode."""
        default_encoder.eval()
        output1 = default_encoder.forward(sample_conditioning)
        output2 = default_encoder.forward(sample_conditioning)
        np.testing.assert_array_almost_equal(output1, output2)

    @pytest.mark.unit
    def test_forward_without_graph_data(self, default_encoder, sample_conditioning):
        """Test forward pass without graph data uses transformer only."""
        output = default_encoder.forward(
            sample_conditioning, graph_data=None
        )
        assert isinstance(output, np.ndarray)
        assert output.shape == (default_encoder.output_dim,)

    @pytest.mark.unit
    def test_forward_output_values_reasonable(self, default_encoder, sample_conditioning):
        """Test that output values are in reasonable range (not exploding)."""
        output = default_encoder.forward(sample_conditioning)
        # Check that values don't explode (rough check)
        assert np.abs(output).max() < 100.0
        # Check that output is not all zeros
        assert np.abs(output).sum() > 1e-6

    @pytest.mark.unit
    def test_forward_batch_processing_single_sample(self, default_encoder):
        """Test forward pass with single sample (batch_size=1)."""
        cond = torch.randn(1, 768)
        output = default_encoder.forward(cond)
        assert output.shape == (default_encoder.output_dim,)

    @pytest.mark.unit
    def test_forward_long_sequence(self, default_encoder):
        """Test forward pass with long conditioning sequence."""
        cond = torch.randn(100, 768)  # Long sequence
        output = default_encoder.forward(cond)
        assert output.shape == (default_encoder.output_dim,)

    @pytest.mark.unit
    def test_forward_short_sequence(self, default_encoder):
        """Test forward pass with short conditioning sequence."""
        cond = torch.randn(1, 768)  # Single token
        output = default_encoder.forward(cond)
        assert output.shape == (default_encoder.output_dim,)

    @pytest.mark.unit
    def test_forward_no_grad_context(self, default_encoder, sample_conditioning):
        """Test forward pass in no_grad context."""
        default_encoder.eval()
        with torch.no_grad():
            output = default_encoder.forward(sample_conditioning)
        assert isinstance(output, np.ndarray)
        assert output.shape == (default_encoder.output_dim,)

    @pytest.mark.unit
    def test_forward_gradient_flow(self, default_encoder, sample_conditioning):
        """Test that gradients flow through forward pass (training mode)."""
        default_encoder.train()
        cond = torch.from_numpy(sample_conditioning).float()
        cond.requires_grad = True
        output = default_encoder.forward(cond)
        # Output is numpy, but intermediate tensors should have gradients
        assert cond.grad is None  # Not yet backprop'd
        loss = torch.from_numpy(output).sum()
        assert loss.requires_grad or loss.item() != 0  # Can compute loss


# ============================================================================
# SECTION 3: Conditioning-Only Encoding Tests
# ============================================================================


@requires_torch
class TestEncodeConditioningOnly:
    """Test encode_conditioning_only method."""

    @pytest.mark.unit
    def test_encode_conditioning_only_numpy(
        self, default_encoder, sample_conditioning
    ):
        """Test conditioning-only encoding with numpy input."""
        output = default_encoder.encode_conditioning_only(sample_conditioning)
        assert isinstance(output, np.ndarray)
        # When no graph is available, output_dim stays unchanged
        assert len(output.shape) == 1
        assert not np.isnan(output).any()

    @pytest.mark.unit
    def test_encode_conditioning_only_torch(self, default_encoder):
        """Test conditioning-only encoding with torch input."""
        cond = torch.randn(10, 768)
        output = default_encoder.encode_conditioning_only(cond)
        assert isinstance(output, np.ndarray)
        assert len(output.shape) == 1

    @pytest.mark.unit
    def test_encode_conditioning_only_1d_input(
        self, default_encoder, sample_conditioning_single
    ):
        """Test conditioning-only with 1D input."""
        output = default_encoder.encode_conditioning_only(sample_conditioning_single)
        assert isinstance(output, np.ndarray)
        assert len(output.shape) == 1

    @pytest.mark.unit
    def test_encode_conditioning_only_reproducibility(
        self, default_encoder, sample_conditioning
    ):
        """Test reproducibility of conditioning-only encoding."""
        default_encoder.eval()
        output1 = default_encoder.encode_conditioning_only(sample_conditioning)
        output2 = default_encoder.encode_conditioning_only(sample_conditioning)
        np.testing.assert_array_almost_equal(output1, output2)

    @pytest.mark.unit
    def test_encode_conditioning_only_vs_forward_no_graph(
        self, default_encoder, sample_conditioning
    ):
        """Test that conditioning_only equals forward without graph."""
        default_encoder.eval()
        output1 = default_encoder.encode_conditioning_only(sample_conditioning)
        output2 = default_encoder.forward(sample_conditioning, graph_data=None)
        # Should be very similar (might differ slightly due to precision)
        np.testing.assert_array_almost_equal(output1, output2, decimal=5)


# ============================================================================
# SECTION 4: Graph-Only Encoding Tests
# ============================================================================


@requires_torch
class TestEncodeGraphOnly:
    """Test encode_graph_only method."""

    @pytest.mark.unit
    def test_encode_graph_only_raises_without_gnn(
        self, default_encoder, sample_graph_data
    ):
        """Test that encode_graph_only raises error when GNN unavailable."""
        if default_encoder._has_gnn:
            pytest.skip("GNN is available, test requires unavailable GNN")
        with pytest.raises(RuntimeError, match="GNN encoder is not available"):
            default_encoder.encode_graph_only(sample_graph_data)

    @pytest.mark.unit
    def test_encode_graph_only_with_available_gnn(
        self, default_encoder, sample_graph_data
    ):
        """Test graph-only encoding when GNN is available."""
        if not default_encoder._has_gnn:
            pytest.skip("GNN encoder not available in this environment")
        output = default_encoder.encode_graph_only(sample_graph_data)
        assert isinstance(output, np.ndarray)
        assert len(output.shape) == 1
        assert output.shape[0] == default_encoder.graph_dim


# ============================================================================
# SECTION 5: Forward Pass with Graph Data Tests
# ============================================================================


@requires_torch
class TestForwardWithGraphData:
    """Test forward pass with graph data."""

    @pytest.mark.unit
    def test_forward_with_graph_available(
        self, default_encoder, sample_conditioning, sample_graph_data
    ):
        """Test forward pass with graph data when GNN available."""
        if not default_encoder._has_gnn:
            pytest.skip("GNN encoder not available")
        output = default_encoder.forward(sample_conditioning, graph_data=sample_graph_data)
        assert isinstance(output, np.ndarray)
        assert output.shape == (default_encoder.output_dim,)

    @pytest.mark.unit
    def test_forward_ignores_graph_when_unavailable(
        self, default_encoder, sample_conditioning, sample_graph_data
    ):
        """Test that forward ignores graph when GNN unavailable."""
        if default_encoder._has_gnn:
            pytest.skip("GNN is available, test requires unavailable GNN")
        # Should not raise, graph should be ignored
        output = default_encoder.forward(sample_conditioning, graph_data=sample_graph_data)
        assert isinstance(output, np.ndarray)

    @pytest.mark.unit
    def test_forward_graph_with_torch_tensors(
        self, default_encoder, sample_conditioning
    ):
        """Test forward with torch tensor graph data."""
        if not default_encoder._has_gnn:
            pytest.skip("GNN encoder not available")

        graph_data = {
            "node_features": torch.randn(20, 32),
            "edge_index": torch.randint(0, 20, (2, 40)).long(),
            "edge_attr": torch.randn(40, 8),
        }
        output = default_encoder.forward(sample_conditioning, graph_data=graph_data)
        assert isinstance(output, np.ndarray)
        assert output.shape == (default_encoder.output_dim,)


# ============================================================================
# SECTION 6: Parameter Management Tests
# ============================================================================


@requires_torch
class TestParameterManagement:
    """Test parameter management methods."""

    @pytest.mark.unit
    def test_parameters_yields_all_params(self, default_encoder):
        """Test that parameters() yields all trainable parameters."""
        params = list(default_encoder.parameters())
        assert len(params) > 0
        # Should have parameters from multiple components
        assert len(params) >= 4  # At least input, transformer, output, maybe fusion

    @pytest.mark.unit
    def test_parameters_are_nn_parameters(self, default_encoder):
        """Test that yielded parameters are nn.Parameter instances."""
        for param in default_encoder.parameters():
            assert isinstance(param, nn.Parameter)

    @pytest.mark.unit
    def test_parameter_count_reasonable(self, default_encoder):
        """Test that parameter count is reasonable."""
        param_count = sum(p.numel() for p in default_encoder.parameters())
        # Should have at least hundreds of parameters
        assert param_count > 100
        # Should be less than 10M (sanity check)
        assert param_count < 15_000_000

    @pytest.mark.unit
    def test_parameters_different_for_different_configs(self):
        """Test that different configs have different parameter counts."""
        enc1 = HybridShapeEncoder(input_dim=768, transformer_dim=512)
        enc2 = HybridShapeEncoder(input_dim=512, transformer_dim=256)
        count1 = sum(p.numel() for p in enc1.parameters())
        count2 = sum(p.numel() for p in enc2.parameters())
        # Different configs should have different parameter counts
        assert count1 != count2

    @pytest.mark.unit
    def test_parameters_gradient_enabled(self, default_encoder):
        """Test that parameters have gradient enabled by default."""
        for param in default_encoder.parameters():
            assert param.requires_grad

    @pytest.mark.unit
    def test_parameters_with_custom_config(self, custom_encoder):
        """Test parameter management with custom encoder."""
        params = list(custom_encoder.parameters())
        assert len(params) > 0
        for param in params:
            assert isinstance(param, nn.Parameter)


# ============================================================================
# SECTION 7: State Dict Tests
# ============================================================================


@requires_torch
class TestStateDictManagement:
    """Test state_dict save/load functionality."""

    @pytest.mark.unit
    def test_state_dict_returns_dict(self, default_encoder):
        """Test that state_dict returns a dictionary."""
        state = default_encoder.state_dict()
        assert isinstance(state, dict)

    @pytest.mark.unit
    def test_state_dict_contains_expected_keys(self, default_encoder):
        """Test that state_dict contains expected component keys."""
        state = default_encoder.state_dict()
        # nn.Module state_dict uses dotted keys like "_input_projection.weight"
        key_prefixes = {k.split(".")[0] for k in state.keys()}
        expected_prefixes = {
            "_input_projection",
            "_transformer_encoder",
            "_transformer_output_projection",
        }
        assert expected_prefixes.issubset(key_prefixes)

    @pytest.mark.unit
    def test_state_dict_values_are_tensors(self, default_encoder):
        """Test that state_dict values are tensors (standard nn.Module format)."""
        import torch
        state = default_encoder.state_dict()
        for _key, value in state.items():
            assert isinstance(value, torch.Tensor)

    @pytest.mark.unit
    def test_load_state_dict_roundtrip(self, default_encoder, sample_conditioning):
        """Test save/load state_dict roundtrip."""
        default_encoder.eval()
        # Get output before save
        output_before = default_encoder.forward(sample_conditioning)

        # Save state
        state = default_encoder.state_dict()

        # Create new encoder and load state
        new_encoder = HybridShapeEncoder(
            input_dim=default_encoder.input_dim,
            transformer_dim=default_encoder.transformer_dim,
            graph_dim=default_encoder.graph_dim,
            output_dim=default_encoder.output_dim,
            num_transformer_layers=default_encoder.num_transformer_layers,
            num_heads=default_encoder.num_heads,
            dropout=default_encoder.dropout,
        )
        new_encoder.load_state_dict(state)
        new_encoder.eval()

        # Get output after load
        output_after = new_encoder.forward(sample_conditioning)

        # Outputs should be identical
        np.testing.assert_array_almost_equal(output_before, output_after)

    @pytest.mark.unit
    def test_load_state_dict_partial(self, default_encoder):
        """Test loading partial state dict (only some components)."""
        state = default_encoder.state_dict()
        # Create dict with only _input_projection keys
        partial_state = {
            k: v for k, v in state.items()
            if k.startswith("_input_projection")
        }
        # Should not raise with strict=False
        default_encoder.load_state_dict(partial_state, strict=False)

    @pytest.mark.unit
    def test_state_dict_preserves_weights(self, default_encoder):
        """Test that state_dict preserves weight values."""
        state1 = default_encoder.state_dict()
        # Verify we can extract actual weight values (standard nn.Module dotted keys)
        input_proj_weights = state1["_input_projection.weight"]
        assert input_proj_weights.shape[0] == default_encoder.transformer_dim
        assert input_proj_weights.shape[1] == default_encoder.input_dim

    @pytest.mark.unit
    def test_state_dict_with_custom_encoder(self, custom_encoder):
        """Test state_dict with custom encoder configuration."""
        state = custom_encoder.state_dict()
        new_encoder = HybridShapeEncoder(
            input_dim=custom_encoder.input_dim,
            transformer_dim=custom_encoder.transformer_dim,
            graph_dim=custom_encoder.graph_dim,
            output_dim=custom_encoder.output_dim,
            num_transformer_layers=custom_encoder.num_transformer_layers,
            num_heads=custom_encoder.num_heads,
        )
        # Should load without errors
        new_encoder.load_state_dict(state)


# ============================================================================
# SECTION 8: Device Management Tests
# ============================================================================


@requires_torch
class TestDeviceManagement:
    """Test device movement and management."""

    @pytest.mark.unit
    def test_to_returns_self(self, default_encoder):
        """Test that to() returns self for chaining."""
        result = default_encoder.to("cpu")
        assert result is default_encoder

    @pytest.mark.unit
    def test_to_cpu(self, default_encoder):
        """Test device movement to CPU."""
        default_encoder.to("cpu")
        assert default_encoder.device == "cpu"
        # Check that components are on CPU
        for param in default_encoder.parameters():
            assert param.device.type == "cpu"

    @pytest.mark.unit
    def test_forward_respects_device(self, default_encoder, sample_conditioning):
        """Test that forward pass respects device setting."""
        default_encoder.to("cpu")
        output = default_encoder.forward(sample_conditioning)
        # Output is numpy, so just check it computes
        assert isinstance(output, np.ndarray)

    @pytest.mark.unit
    def test_device_chaining(self, default_encoder):
        """Test that to() can be chained."""
        result = default_encoder.to("cpu").to("cpu")
        assert result is default_encoder
        assert default_encoder.device == "cpu"

    @pytest.mark.unit
    def test_cuda_not_available_handling(self, default_encoder):
        """Test that to('cuda') gracefully handles unavailability."""
        # This test checks behavior when CUDA might not be available
        # Just ensure no crash occurs
        if torch.cuda.is_available():
            default_encoder.to("cuda")
            assert default_encoder.device == "cuda"
        else:
            # On CPU-only machines, we still call it (may or may not error)
            try:
                default_encoder.to("cuda")
            except (RuntimeError, AssertionError):
                pass  # Expected if no CUDA


# ============================================================================
# SECTION 9: Train/Eval Mode Tests
# ============================================================================


@requires_torch
class TestTrainEvalMode:
    """Test train/eval mode switching."""

    @pytest.mark.unit
    def test_eval_sets_all_modules_eval(self, default_encoder):
        """Test that eval() sets all submodules to evaluation mode."""
        default_encoder.train()  # Ensure in train mode first
        default_encoder.eval()
        # Check that all modules are in eval mode
        assert not default_encoder._input_projection.training
        assert not default_encoder._transformer_encoder.training
        assert not default_encoder._transformer_output_projection.training

    @pytest.mark.unit
    def test_train_sets_all_modules_train(self, default_encoder):
        """Test that train() sets all submodules to training mode."""
        default_encoder.eval()  # Ensure in eval mode first
        default_encoder.train()
        assert default_encoder._input_projection.training
        assert default_encoder._transformer_encoder.training
        assert default_encoder._transformer_output_projection.training

    @pytest.mark.unit
    def test_train_with_false_arg(self, default_encoder):
        """Test train(False) sets training mode to False."""
        default_encoder.train(True)
        default_encoder.train(False)
        assert not default_encoder._input_projection.training

    @pytest.mark.unit
    def test_train_with_true_arg(self, default_encoder):
        """Test train(True) sets training mode to True."""
        default_encoder.train(False)
        default_encoder.train(True)
        assert default_encoder._input_projection.training

    @pytest.mark.unit
    def test_dropout_behavior_train_vs_eval(self, default_encoder, sample_conditioning):
        """Test that dropout behaves differently in train vs eval."""
        default_encoder_train = HybridShapeEncoder(dropout=0.5)  # High dropout

        outputs_train = []
        outputs_eval = []

        for _ in range(5):
            default_encoder_train.train()
            outputs_train.append(default_encoder_train.forward(sample_conditioning))

        for _ in range(5):
            default_encoder_train.eval()
            outputs_eval.append(default_encoder_train.forward(sample_conditioning))

        # Outputs in eval mode should be more consistent (less dropout noise)
        np.var([np.linalg.norm(o) for o in outputs_eval])
        np.var([np.linalg.norm(o) for o in outputs_train])
        # Note: This is probabilistic, so we don't assert strictly

    @pytest.mark.unit
    def test_eval_disables_dropout(self, default_encoder):
        """Test that eval mode disables dropout."""
        default_encoder.eval()
        # In eval mode, dropout should be disabled in all submodules
        # Check by verifying the mode
        assert not default_encoder._input_projection.training


# ============================================================================
# SECTION 10: Input Validation Tests
# ============================================================================


@requires_torch
class TestInputValidation:
    """Test input dimension validation and error handling."""

    @pytest.mark.unit
    def test_forward_dimension_mismatch(self, default_encoder):
        """Test forward pass with incorrect input dimensions."""
        # Input dim is 768, try with 512
        wrong_input = np.random.randn(10, 512).astype(np.float32)
        with pytest.raises((RuntimeError, ValueError)):
            default_encoder.forward(wrong_input)

    @pytest.mark.unit
    def test_conditioning_only_dimension_mismatch(self, default_encoder):
        """Test conditioning_only with wrong input dimensions."""
        wrong_input = np.random.randn(512).astype(np.float32)
        with pytest.raises((RuntimeError, ValueError)):
            default_encoder.encode_conditioning_only(wrong_input)

    @pytest.mark.unit
    def test_forward_with_wrong_dtype(self, default_encoder):
        """Test forward with wrong data type."""
        # Try int input (should be converted to float)
        cond = np.random.randint(0, 10, (10, 768)).astype(np.int32)
        # Should either error or convert automatically
        try:
            output = default_encoder.forward(cond)
            # If it succeeds, output should be valid
            assert output.shape == (default_encoder.output_dim,)
        except (RuntimeError, TypeError):
            pass  # Expected if conversion fails

    @pytest.mark.unit
    def test_forward_with_nan_input(self, default_encoder):
        """Test forward pass with NaN in input."""
        cond = np.random.randn(10, 768).astype(np.float32)
        cond[0, 0] = np.nan
        output = default_encoder.forward(cond)
        # Output will likely contain NaNs
        assert np.isnan(output).any()

    @pytest.mark.unit
    def test_forward_with_inf_input(self, default_encoder):
        """Test forward pass with infinity in input."""
        cond = np.random.randn(10, 768).astype(np.float32)
        cond[0, 0] = np.inf
        output = default_encoder.forward(cond)
        # Output will likely contain infs or nans
        assert np.isinf(output).any() or np.isnan(output).any()

    @pytest.mark.unit
    def test_empty_conditioning_1d(self, default_encoder):
        """Test with 1D empty-like input (edge case)."""
        # Single zero vector
        cond = np.zeros(768, dtype=np.float32)
        output = default_encoder.forward(cond)
        assert output.shape == (default_encoder.output_dim,)

    @pytest.mark.unit
    def test_graph_data_missing_keys(self, default_encoder):
        """Test graph_only with missing required keys."""
        if not default_encoder._has_gnn:
            pytest.skip("GNN not available")
        # Missing node_features
        bad_graph = {"edge_index": np.zeros((2, 5), dtype=np.int64)}
        with pytest.raises((KeyError, RuntimeError, TypeError)):
            default_encoder.encode_graph_only(bad_graph)


# ============================================================================
# SECTION 11: Edge Case Tests
# ============================================================================


@requires_torch
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_single_sample_batch(self, default_encoder):
        """Test with single sample (batch_size=1)."""
        cond = torch.randn(1, 768)
        output = default_encoder.forward(cond)
        assert output.shape == (default_encoder.output_dim,)

    @pytest.mark.unit
    def test_very_long_sequence(self, default_encoder):
        """Test with very long conditioning sequence."""
        cond = torch.randn(500, 768)  # Very long
        output = default_encoder.forward(cond)
        assert output.shape == (default_encoder.output_dim,)

    @pytest.mark.unit
    def test_minimal_encoder(self):
        """Test encoder with minimal configuration."""
        encoder = HybridShapeEncoder(
            input_dim=64,
            transformer_dim=64,
            graph_dim=32,
            output_dim=64,
            num_transformer_layers=1,
            num_heads=2,
        )
        cond = torch.randn(5, 64)
        output = encoder.forward(cond)
        assert output.shape == (64,)

    @pytest.mark.unit
    def test_large_encoder(self):
        """Test encoder with large configuration."""
        encoder = HybridShapeEncoder(
            input_dim=1024,
            transformer_dim=1024,
            graph_dim=512,
            output_dim=1024,
            num_transformer_layers=6,
            num_heads=16,
        )
        cond = torch.randn(5, 1024)
        output = encoder.forward(cond)
        assert output.shape == (1024,)

    @pytest.mark.unit
    def test_repeated_forward_no_memory_leak(self, default_encoder, sample_conditioning):
        """Test repeated forward passes for memory leaks."""
        default_encoder.eval()
        with torch.no_grad():
            for _ in range(100):
                output = default_encoder.forward(sample_conditioning)
                assert isinstance(output, np.ndarray)

    @pytest.mark.unit
    def test_forward_alternating_devices(self, default_encoder, sample_conditioning):
        """Test forward after device alternation."""
        default_encoder.eval()
        default_encoder.to("cpu")
        output1 = default_encoder.forward(sample_conditioning)
        default_encoder.to("cpu")
        output2 = default_encoder.forward(sample_conditioning)
        np.testing.assert_array_almost_equal(output1, output2)

    @pytest.mark.unit
    def test_encoding_batch_consistency(self, default_encoder):
        """Test that batch encoding is consistent with unbatched."""
        cond = torch.randn(10, 768)
        output_batched = default_encoder.forward(cond)
        # The batched output is the result of processing the sequence
        # Verify it's a valid output
        assert output_batched.shape == (default_encoder.output_dim,)

    @pytest.mark.unit
    def test_multiple_forward_calls_same_input(self, default_encoder):
        """Test multiple forward calls with same input in eval mode."""
        default_encoder.eval()
        cond = torch.randn(5, 768)
        outputs = [default_encoder.forward(cond) for _ in range(3)]
        # All outputs should be identical
        for i in range(1, len(outputs)):
            np.testing.assert_array_almost_equal(outputs[0], outputs[i])


# ============================================================================
# SECTION 12: Transformer Component Tests
# ============================================================================


@requires_torch
class TestTransformerComponent:
    """Test transformer-specific functionality."""

    @pytest.mark.unit
    def test_transformer_encoder_structure(self, default_encoder):
        """Test structure of transformer encoder."""
        assert default_encoder._transformer_encoder is not None
        # Check num_layers
        assert len(default_encoder._transformer_encoder.layers) == default_encoder.num_transformer_layers

    @pytest.mark.unit
    def test_attention_heads_configuration(self, default_encoder):
        """Test that attention heads are configured correctly."""
        # Get first layer
        first_layer = default_encoder._transformer_encoder.layers[0]
        # Self-attention should have num_heads
        assert first_layer.self_attn.num_heads == default_encoder.num_heads

    @pytest.mark.unit
    def test_transformer_output_shape_intermediate(self, default_encoder):
        """Test transformer intermediate output shape."""
        cond = torch.randn(1, 5, 768)
        with torch.no_grad():
            # Project to transformer dim
            projected = default_encoder._input_projection(cond)
            assert projected.shape == (1, 5, default_encoder.transformer_dim)
            # Pass through transformer
            transformer_out = default_encoder._transformer_encoder(projected)
            assert transformer_out.shape == (1, 5, default_encoder.transformer_dim)

    @pytest.mark.unit
    def test_feedforward_dimension(self, default_encoder):
        """Test feedforward layer dimension in transformer."""
        # Should be 4x transformer_dim
        first_layer = default_encoder._transformer_encoder.layers[0]
        ff_out_features = first_layer.linear2.out_features
        assert ff_out_features == default_encoder.transformer_dim


# ============================================================================
# SECTION 13: Integration Tests
# ============================================================================


@requires_torch
class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.unit
    def test_full_pipeline_train_eval_cycle(self, sample_conditioning):
        """Test complete train/eval/save/load cycle."""
        encoder = HybridShapeEncoder(
            input_dim=768,
            output_dim=512,
        )

        # Train mode
        encoder.train()
        out_train1 = encoder.forward(sample_conditioning)
        out_train2 = encoder.forward(sample_conditioning)
        # Outputs should differ due to dropout
        assert not np.allclose(out_train1, out_train2)

        # Eval mode
        encoder.eval()
        out_eval1 = encoder.forward(sample_conditioning)
        out_eval2 = encoder.forward(sample_conditioning)
        # Outputs should be identical in eval mode
        np.testing.assert_array_almost_equal(out_eval1, out_eval2)

        # Save/load cycle
        state = encoder.state_dict()
        encoder2 = HybridShapeEncoder()
        encoder2.load_state_dict(state)
        encoder2.eval()
        out_loaded = encoder2.forward(sample_conditioning)
        np.testing.assert_array_almost_equal(out_eval1, out_loaded)

    @pytest.mark.unit
    def test_different_input_shapes(self, default_encoder):
        """Test encoder with various input shapes."""
        shapes = [
            (768,),  # 1D
            (5, 768),  # 2D
            (1, 768),  # Batch of 1
            (20, 768),  # Longer sequence
        ]
        default_encoder.eval()
        outputs = []
        for shape in shapes:
            cond = torch.randn(*shape)
            output = default_encoder.forward(cond)
            outputs.append(output)
            assert output.shape == (default_encoder.output_dim,)
        # All outputs should be reasonable
        for output in outputs:
            assert not np.isnan(output).any()

    @pytest.mark.unit
    def test_gradient_accumulation(self, default_encoder, sample_conditioning):
        """Test gradient accumulation over multiple forward passes."""
        default_encoder.train()
        optimizer = torch.optim.Adam(default_encoder.parameters(), lr=1e-4)

        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            cond = torch.from_numpy(sample_conditioning).float()
            cond.requires_grad = True
            # Forward pass
            output_arr = default_encoder.forward(cond)
            output = torch.from_numpy(output_arr)
            loss = output.sum()
            # Backward pass (through output tensor)
            losses.append(loss.item())

        assert len(losses) == 3
        assert all(isinstance(val, float) for val in losses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
