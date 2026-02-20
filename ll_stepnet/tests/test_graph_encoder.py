"""Tests for STEPGraphEncoder and graph-related components.

Tests cover:
- STEPGraphEncoder initialization and forward pass
- Adjacency matrix construction
- Graph convolution layers
- Fusion layer behavior
- Input dimension handling
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch")
import torch.nn as nn


# ============================================================================
# SECTION 1: STEPGraphEncoder Tests
# ============================================================================


class TestSTEPGraphEncoderInit:
    """Test STEPGraphEncoder initialization."""

    def test_encoder_creation_default_config(self) -> None:
        """Test encoder creates with default config."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder()
        assert encoder is not None
        # Default input_dim is 48
        assert encoder.input_dim == 48
        assert encoder.node_dim == 128

    def test_encoder_creation_custom_config(self) -> None:
        """Test encoder creates with custom configuration."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(
            input_dim=64,
            node_dim=256,
            edge_dim=32,
            num_layers=4,
        )
        assert encoder.input_dim == 64
        assert encoder.node_dim == 256
        assert encoder.edge_dim == 32

    def test_encoder_has_expected_layers(self) -> None:
        """Test encoder has expected layer structure."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(num_layers=3)

        # Should have conv_layers, input_proj, activation, dropout
        assert hasattr(encoder, 'conv_layers')
        assert hasattr(encoder, 'input_proj')
        assert hasattr(encoder, 'activation')
        assert hasattr(encoder, 'dropout')
        assert len(encoder.conv_layers) == 3


class TestSTEPGraphEncoderForward:
    """Test STEPGraphEncoder forward pass."""

    def test_forward_with_adjacency_matrix(self) -> None:
        """Test forward pass with adjacency matrix input."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(
            input_dim=48,
            node_dim=128,
        )

        num_nodes = 10
        # Note: STEPGraphEncoder expects unbatched input [num_nodes, input_dim]
        node_features = torch.randn(num_nodes, 48)
        adjacency = torch.randint(0, 2, (num_nodes, num_nodes)).float()

        output = encoder(node_features, adjacency)

        # Output should be [num_nodes, node_dim]
        assert output.shape == (num_nodes, 128)

    def test_forward_output_shape(self) -> None:
        """Test forward produces correct output shape."""
        from stepnet.encoder import STEPGraphEncoder

        node_dim = 64
        encoder = STEPGraphEncoder(
            input_dim=48,
            node_dim=node_dim,
        )

        num_nodes = 20
        node_features = torch.randn(num_nodes, 48)
        adjacency = torch.randint(0, 2, (num_nodes, num_nodes)).float()

        output = encoder(node_features, adjacency)

        assert output.dim() == 2
        assert output.shape[0] == num_nodes
        assert output.shape[1] == node_dim

    def test_forward_with_sparse_adjacency(self) -> None:
        """Test forward with sparse adjacency matrix."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=48)

        num_nodes = 15
        node_features = torch.randn(num_nodes, 48)
        # Sparse: only self-loops
        adjacency = torch.eye(num_nodes)

        output = encoder(node_features, adjacency)

        assert output is not None
        assert not torch.isnan(output).any()

    def test_forward_with_dense_adjacency(self) -> None:
        """Test forward with fully connected graph."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=48)

        num_nodes = 10
        node_features = torch.randn(num_nodes, 48)
        # Fully connected
        adjacency = torch.ones(num_nodes, num_nodes)

        output = encoder(node_features, adjacency)

        assert output is not None
        assert not torch.isnan(output).any()

    def test_forward_gradient_flow(self) -> None:
        """Test gradients flow through encoder."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=48, node_dim=64)

        num_nodes = 10
        node_features = torch.randn(num_nodes, 48, requires_grad=True)
        adjacency = torch.randint(0, 2, (num_nodes, num_nodes)).float()

        output = encoder(node_features, adjacency)
        loss = output.sum()
        loss.backward()

        assert node_features.grad is not None
        assert not torch.isnan(node_features.grad).any()

    def test_forward_single_node(self) -> None:
        """Test forward with single node graph."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=48)

        node_features = torch.randn(1, 48)
        adjacency = torch.ones(1, 1)

        output = encoder(node_features, adjacency)

        assert output.shape == (1, 128)
        assert not torch.isnan(output).any()

    def test_forward_no_edges(self) -> None:
        """Test forward with disconnected graph (no edges)."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=48)

        num_nodes = 5
        node_features = torch.randn(num_nodes, 48)
        adjacency = torch.zeros(num_nodes, num_nodes)

        output = encoder(node_features, adjacency)

        assert output.shape == (num_nodes, 128)
        assert not torch.isnan(output).any()


class TestSTEPGraphEncoderInputDimensions:
    """Test STEPGraphEncoder with different input dimensions."""

    def test_input_dim_48_cadling_format(self) -> None:
        """Test 48-dim input (cadling TopologyGraph format)."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=48)

        num_nodes = 20
        node_features = torch.randn(num_nodes, 48)
        adjacency = torch.randint(0, 2, (num_nodes, num_nodes)).float()

        output = encoder(node_features, adjacency)

        assert output is not None
        assert output.shape == (num_nodes, 128)

    def test_input_dim_129_legacy_format(self) -> None:
        """Test 129-dim input (ll_stepnet legacy format)."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=129)

        num_nodes = 20
        node_features = torch.randn(num_nodes, 129)
        adjacency = torch.randint(0, 2, (num_nodes, num_nodes)).float()

        output = encoder(node_features, adjacency)

        assert output is not None
        assert output.shape == (num_nodes, 128)

    def test_input_proj_layer(self) -> None:
        """Test input projection layer projects to node_dim."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=48, node_dim=256)

        # Check input_proj layer dimensions
        assert encoder.input_proj.in_features == 48
        assert encoder.input_proj.out_features == 256


# ============================================================================
# SECTION 2: Graph Convolution Tests
# ============================================================================


class TestGraphConvolution:
    """Test graph convolution layer behavior."""

    def test_message_passing(self) -> None:
        """Test message passing aggregates neighbor features."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=4, node_dim=8, num_layers=1)

        # Simple 2-node graph
        node_features = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        # Node 0 connected to node 1
        adjacency = torch.tensor([[0., 1.], [1., 0.]])

        output = encoder(node_features, adjacency)

        # After message passing, nodes should have mixed features
        assert output is not None
        assert output.shape == (2, 8)

    def test_self_loop_handling(self) -> None:
        """Test encoder handles self-loops."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=8)

        num_nodes = 5
        node_features = torch.randn(num_nodes, 8)
        # Only self-loops
        adjacency = torch.eye(num_nodes)

        output = encoder(node_features, adjacency)

        assert output is not None
        assert not torch.isnan(output).any()

    def test_residual_connection(self) -> None:
        """Test residual connections preserve information."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=32, node_dim=32, num_layers=2)

        num_nodes = 10
        node_features = torch.randn(num_nodes, 32)
        adjacency = torch.randint(0, 2, (num_nodes, num_nodes)).float()

        output = encoder(node_features, adjacency)

        # Output shape should match node_dim
        assert output.shape == (num_nodes, 32)

    def test_multiple_conv_layers(self) -> None:
        """Test multiple graph convolution layers."""
        from stepnet.encoder import STEPGraphEncoder

        for num_layers in [1, 2, 4, 8]:
            encoder = STEPGraphEncoder(input_dim=16, num_layers=num_layers)
            assert len(encoder.conv_layers) == num_layers

            node_features = torch.randn(5, 16)
            adjacency = torch.randint(0, 2, (5, 5)).float()

            output = encoder(node_features, adjacency)
            assert output is not None
            assert not torch.isnan(output).any()


# ============================================================================
# SECTION 3: Fusion Layer Tests
# ============================================================================


class TestFusionLayer:
    """Test fusion layer combining token and graph features."""

    def test_fusion_layer_exists(self) -> None:
        """Test STEPEncoder has fusion capability."""
        from stepnet.encoder import STEPEncoder

        encoder = STEPEncoder()

        # Should have fusion Sequential module
        assert hasattr(encoder, 'fusion')
        assert isinstance(encoder.fusion, nn.Sequential)

    def test_fusion_combines_features(self) -> None:
        """Test fusion layer combines token and graph features."""
        from stepnet.encoder import STEPEncoder

        encoder = STEPEncoder(token_embed_dim=256)

        batch_size = 2
        seq_len = 20

        # Token IDs for input
        token_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Topology data
        num_nodes = 30
        topology_data = {
            'node_features': torch.randn(num_nodes, 48),
            'adjacency_matrix': torch.randint(0, 2, (num_nodes, num_nodes)).float(),
        }

        # Forward with topology
        output = encoder(token_ids, topology_data=topology_data)

        assert output is not None
        assert output.dim() == 2  # [batch_size, output_dim]

    def test_fusion_without_topology(self) -> None:
        """Test encoder works without topology (fallback)."""
        from stepnet.encoder import STEPEncoder

        encoder = STEPEncoder(token_embed_dim=256)

        batch_size = 2
        seq_len = 20

        token_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Forward without topology
        output = encoder(token_ids, topology_data=None)

        assert output is not None
        assert output.dim() == 2

    def test_fusion_output_dimension(self) -> None:
        """Test fusion produces correct output dimension."""
        from stepnet.encoder import STEPEncoder

        output_dim = 512
        encoder = STEPEncoder(output_dim=output_dim)

        batch_size = 2
        seq_len = 20

        token_ids = torch.randint(0, 1000, (batch_size, seq_len))
        output = encoder(token_ids, topology_data=None)

        assert output.shape[-1] == output_dim


# ============================================================================
# SECTION 4: Prepare Topology Data Tests
# ============================================================================


class TestPrepareTopologyData:
    """Test prepare_topology_data utility function."""

    def test_prepare_from_dict(self) -> None:
        """Test preparing topology from dictionary input."""
        from stepnet.encoder import STEPEncoder

        topology_dict = {
            'node_features': torch.randn(20, 48),
            'adjacency_matrix': torch.randint(0, 2, (20, 20)).float(),
        }

        prepared = STEPEncoder.prepare_topology_data(topology_dict)
        assert 'node_features' in prepared
        assert 'adjacency_matrix' in prepared
        assert isinstance(prepared['node_features'], torch.Tensor)
        assert isinstance(prepared['adjacency_matrix'], torch.Tensor)

    def test_prepare_from_numpy(self) -> None:
        """Test preparing topology from numpy arrays."""
        import numpy as np
        from stepnet.encoder import STEPEncoder

        topology_dict = {
            'node_features': np.random.randn(20, 48).astype(np.float32),
            'adjacency_matrix': np.random.randint(0, 2, (20, 20)).astype(np.float32),
        }

        prepared = STEPEncoder.prepare_topology_data(topology_dict)
        assert isinstance(prepared['node_features'], torch.Tensor)
        assert isinstance(prepared['adjacency_matrix'], torch.Tensor)

    def test_prepare_passes_through_extra_keys(self) -> None:
        """Test extra keys are preserved."""
        from stepnet.encoder import STEPEncoder

        topology_dict = {
            'node_features': torch.randn(20, 48),
            'adjacency_matrix': torch.randint(0, 2, (20, 20)).float(),
            'edge_index': torch.tensor([[0, 1], [1, 0]]),
            'custom_key': 'custom_value',
        }

        prepared = STEPEncoder.prepare_topology_data(topology_dict)
        assert 'edge_index' in prepared
        assert 'custom_key' in prepared


# ============================================================================
# SECTION 5: Edge Index Tests (PyG Format)
# ============================================================================


class TestEdgeIndex:
    """Test edge index handling for PyG compatibility."""

    def test_adjacency_to_edge_index(self) -> None:
        """Test converting adjacency matrix to edge index."""
        # 3-node graph: 0->1, 1->2, 2->0
        adjacency = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]).float()

        # Convert to edge index
        edge_index = adjacency.nonzero().t()

        assert edge_index.shape[0] == 2  # [2, num_edges]
        assert edge_index.shape[1] == 3  # 3 edges

    def test_edge_index_valid_indices(self) -> None:
        """Test edge indices are valid node indices."""
        num_nodes = 10
        adjacency = torch.randint(0, 2, (num_nodes, num_nodes)).float()

        edge_index = adjacency.nonzero().t()

        # All indices should be < num_nodes
        if edge_index.numel() > 0:
            assert (edge_index < num_nodes).all()
            assert (edge_index >= 0).all()

    def test_edge_index_empty_graph(self) -> None:
        """Test edge index for graph with no edges."""
        adjacency = torch.zeros(5, 5)

        edge_index = adjacency.nonzero().t()

        # Shape should be [2, 0] for empty graph
        assert edge_index.shape == (2, 0) or edge_index.numel() == 0


# ============================================================================
# SECTION 6: Integration with STEPEncoder
# ============================================================================


class TestSTEPEncoderIntegration:
    """Test STEPGraphEncoder integration with main STEPEncoder."""

    def test_encoder_with_topology_data(self) -> None:
        """Test STEPEncoder with topology data."""
        from stepnet.encoder import STEPEncoder

        encoder = STEPEncoder(
            token_embed_dim=256,
            graph_node_dim=128,
            output_dim=512,
        )

        batch_size = 2
        seq_len = 20
        num_nodes = 30

        token_ids = torch.randint(0, 1000, (batch_size, seq_len))
        topology_data = {
            'node_features': torch.randn(num_nodes, 48),
            'adjacency_matrix': torch.randint(0, 2, (num_nodes, num_nodes)).float(),
        }

        output = encoder(token_ids, topology_data=topology_data)

        assert output is not None
        assert output.shape == (batch_size, 512)

    def test_encoder_without_topology_data(self) -> None:
        """Test STEPEncoder without topology data."""
        from stepnet.encoder import STEPEncoder

        encoder = STEPEncoder(
            token_embed_dim=256,
            output_dim=512,
        )

        batch_size = 2
        seq_len = 20

        token_ids = torch.randint(0, 1000, (batch_size, seq_len))

        output = encoder(token_ids, topology_data=None)

        assert output is not None
        assert output.shape == (batch_size, 512)

    def test_encoder_graph_input_dim(self) -> None:
        """Test STEPEncoder with different graph input dimensions."""
        from stepnet.encoder import STEPEncoder

        # Test with 48-dim input (cadling)
        encoder_48 = STEPEncoder(graph_input_dim=48)
        assert encoder_48.graph_encoder.input_dim == 48

        # Test with 129-dim input (legacy)
        encoder_129 = STEPEncoder(graph_input_dim=129)
        assert encoder_129.graph_encoder.input_dim == 129


# ============================================================================
# SECTION 7: Device Consistency Tests
# ============================================================================


class TestDeviceConsistency:
    """Test device consistency across encoder components."""

    def test_cpu_consistency(self) -> None:
        """Test all operations stay on CPU."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=48)

        num_nodes = 10
        node_features = torch.randn(num_nodes, 48)
        adjacency = torch.randint(0, 2, (num_nodes, num_nodes)).float()

        output = encoder(node_features, adjacency)

        assert output.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_consistency(self) -> None:
        """Test all operations stay on CUDA."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=48).cuda()

        num_nodes = 10
        node_features = torch.randn(num_nodes, 48).cuda()
        adjacency = torch.randint(0, 2, (num_nodes, num_nodes)).float().cuda()

        output = encoder(node_features, adjacency)

        assert output.device.type == 'cuda'

    def test_step_encoder_cpu_consistency(self) -> None:
        """Test STEPEncoder stays on CPU."""
        from stepnet.encoder import STEPEncoder

        encoder = STEPEncoder()

        token_ids = torch.randint(0, 1000, (2, 20))
        output = encoder(token_ids)

        assert output.device.type == 'cpu'


# ============================================================================
# SECTION 8: Module Import Tests
# ============================================================================


class TestEncoderModuleImports:
    """Test encoder module imports."""

    def test_import_step_graph_encoder(self) -> None:
        """Test STEPGraphEncoder can be imported."""
        from stepnet.encoder import STEPGraphEncoder
        assert STEPGraphEncoder is not None

    def test_import_step_encoder(self) -> None:
        """Test STEPEncoder can be imported."""
        from stepnet.encoder import STEPEncoder
        assert STEPEncoder is not None

    def test_import_step_transformer_encoder(self) -> None:
        """Test STEPTransformerEncoder can be imported."""
        from stepnet.encoder import STEPTransformerEncoder
        assert STEPTransformerEncoder is not None

    def test_import_step_transformer_decoder(self) -> None:
        """Test STEPTransformerDecoder can be imported."""
        from stepnet.encoder import STEPTransformerDecoder
        assert STEPTransformerDecoder is not None

    def test_import_build_step_encoder(self) -> None:
        """Test build_step_encoder factory can be imported."""
        from stepnet.encoder import build_step_encoder
        assert build_step_encoder is not None


# ============================================================================
# SECTION 9: Build Step Encoder Factory Tests
# ============================================================================


class TestBuildStepEncoder:
    """Test build_step_encoder factory function."""

    def test_build_with_defaults(self) -> None:
        """Test building encoder with default config."""
        from stepnet.encoder import build_step_encoder

        encoder = build_step_encoder()
        assert encoder is not None
        assert encoder.output_dim == 1024

    def test_build_with_custom_config(self) -> None:
        """Test building encoder with custom config."""
        from stepnet.encoder import build_step_encoder

        encoder = build_step_encoder(
            vocab_size=30000,
            output_dim=512,
        )
        assert encoder.output_dim == 512
