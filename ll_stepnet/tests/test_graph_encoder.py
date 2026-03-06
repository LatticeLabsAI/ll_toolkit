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
# SECTION 6b: Per-Sample Topology Regression Tests
# ============================================================================


class TestPerSampleTopology:
    """Verify that batched topology data produces per-sample graph encodings."""

    def test_different_graphs_produce_different_encodings(self) -> None:
        """Regression: each batch item should get its own topology encoding.

        Previously, a single graph was mean-pooled and broadcast to all items,
        so every sample received identical topology regardless of its source
        STEP file.
        """
        from stepnet.encoder import STEPEncoder

        encoder = STEPEncoder(token_embed_dim=256, output_dim=512)

        batch_size = 3
        seq_len = 20
        token_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Create distinct per-sample topology dicts
        torch.manual_seed(0)
        topo_a = {
            'node_features': torch.randn(10, 48),
            'adjacency_matrix': torch.randint(0, 2, (10, 10)).float(),
        }
        torch.manual_seed(42)
        topo_b = {
            'node_features': torch.randn(15, 48),
            'adjacency_matrix': torch.randint(0, 2, (15, 15)).float(),
        }
        torch.manual_seed(99)
        topo_c = {
            'node_features': torch.randn(8, 48),
            'adjacency_matrix': torch.randint(0, 2, (8, 8)).float(),
        }

        output = encoder(token_ids, topology_data=[topo_a, topo_b, topo_c])

        assert output.shape == (batch_size, 512)
        # Each sample should get a unique encoding — pairwise differences
        # should be non-zero (they share token_ids but have different graphs)
        assert not torch.allclose(output[0], output[1], atol=1e-5)
        assert not torch.allclose(output[1], output[2], atol=1e-5)

    def test_single_dict_backward_compat(self) -> None:
        """A single topology dict still works (batch_size=1 use case)."""
        from stepnet.encoder import STEPEncoder

        encoder = STEPEncoder(token_embed_dim=256, output_dim=512)

        token_ids = torch.randint(0, 1000, (1, 20))
        topo = {
            'node_features': torch.randn(10, 48),
            'adjacency_matrix': torch.randint(0, 2, (10, 10)).float(),
        }

        output = encoder(token_ids, topology_data=topo)
        assert output.shape == (1, 512)

    def test_list_with_none_entries(self) -> None:
        """Batch items without topology get zero graph features."""
        from stepnet.encoder import STEPEncoder

        encoder = STEPEncoder(token_embed_dim=256, output_dim=512)

        token_ids = torch.randint(0, 1000, (2, 20))
        topo = {
            'node_features': torch.randn(10, 48),
            'adjacency_matrix': torch.randint(0, 2, (10, 10)).float(),
        }

        output = encoder(token_ids, topology_data=[topo, None])
        assert output.shape == (2, 512)


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


# ============================================================================
# SECTION 10: Sparse Adjacency Matrix Tests
# ============================================================================


class TestSparseAdjacency:
    """Test sparse adjacency matrix support (OOM fix for large B-Rep graphs)."""

    def test_forward_with_sparse_coo_input(self) -> None:
        """Test forward pass accepts sparse COO adjacency directly."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=48)
        num_nodes = 20

        node_features = torch.randn(num_nodes, 48)
        # Build sparse adjacency: chain graph 0->1->2->...->19
        src = torch.arange(num_nodes - 1)
        dst = torch.arange(1, num_nodes)
        indices = torch.stack([src, dst])
        values = torch.ones(num_nodes - 1)
        adj_sparse = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

        output = encoder(node_features, adj_sparse)
        assert output.shape == (num_nodes, 128)
        assert not torch.isnan(output).any()

    def test_sparse_and_dense_produce_same_output(self) -> None:
        """Sparse and dense adjacency must produce identical GCN output."""
        from stepnet.encoder import STEPGraphEncoder

        torch.manual_seed(42)
        encoder = STEPGraphEncoder(input_dim=16, node_dim=32, num_layers=2)
        encoder.eval()

        num_nodes = 15
        node_features = torch.randn(num_nodes, 16)

        # Dense adjacency
        adj_dense = torch.zeros(num_nodes, num_nodes)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6)]
        for s, d in edges:
            adj_dense[s, d] = 1.0

        # Sparse adjacency from same edges
        src = torch.tensor([s for s, d in edges])
        dst = torch.tensor([d for s, d in edges])
        adj_sparse = torch.sparse_coo_tensor(
            torch.stack([src, dst]),
            torch.ones(len(edges)),
            (num_nodes, num_nodes),
        )

        with torch.no_grad():
            out_dense = encoder(node_features, adj_dense)
            out_sparse = encoder(node_features, adj_sparse)

        assert torch.allclose(out_dense, out_sparse, atol=1e-6)

    def test_large_sparse_graph_no_oom(self) -> None:
        """Regression: large sparse graph should not OOM.

        A 10K-node graph with ~5 edges/node uses ~0.8 MB sparse vs ~400 MB dense.
        """
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=16, node_dim=32, num_layers=2)
        encoder.eval()

        num_nodes = 10000
        avg_degree = 5
        num_edges = num_nodes * avg_degree

        node_features = torch.randn(num_nodes, 16)
        src = torch.randint(0, num_nodes, (num_edges,))
        dst = torch.randint(0, num_nodes, (num_edges,))
        adj_sparse = torch.sparse_coo_tensor(
            torch.stack([src, dst]),
            torch.ones(num_edges),
            (num_nodes, num_nodes),
        ).coalesce()

        with torch.no_grad():
            output = encoder(node_features, adj_sparse)

        assert output.shape == (num_nodes, 32)
        assert not torch.isnan(output).any()

    def test_gradient_flow_with_sparse(self) -> None:
        """Gradients flow through sparse GCN path."""
        from stepnet.encoder import STEPGraphEncoder

        encoder = STEPGraphEncoder(input_dim=16, node_dim=32)

        num_nodes = 10
        node_features = torch.randn(num_nodes, 16, requires_grad=True)
        adj_sparse = torch.sparse_coo_tensor(
            torch.tensor([[0, 1, 2], [1, 2, 0]]),
            torch.ones(3),
            (num_nodes, num_nodes),
        )

        output = encoder(node_features, adj_sparse)
        loss = output.sum()
        loss.backward()

        assert node_features.grad is not None
        assert not torch.isnan(node_features.grad).any()

    def test_prepare_topology_data_returns_sparse(self) -> None:
        """prepare_topology_data from duck-typed object should return sparse adj."""
        import numpy as np
        from unittest.mock import MagicMock
        from stepnet.encoder import STEPEncoder

        mock_topo = MagicMock()
        mock_topo.to_numpy_node_features.return_value = np.random.randn(100, 48).astype(np.float32)
        mock_topo.to_edge_index.return_value = np.array(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64,
        )

        result = STEPEncoder.prepare_topology_data(mock_topo)
        adj = result["adjacency_matrix"]

        assert adj.is_sparse
        assert adj.shape == (100, 100)
        # Only 4 edges stored, not 10000 values
        assert adj._nnz() == 4

    def test_topology_builder_returns_sparse(self) -> None:
        """STEPTopologyBuilder.build_adjacency_matrix should return sparse."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()
        entities = [
            {"entity_id": 1, "entity_type": "A", "references": [2, 3]},
            {"entity_id": 2, "entity_type": "B", "references": [3]},
            {"entity_id": 3, "entity_type": "C", "references": []},
        ]
        graph = builder.build_reference_graph(entities)
        adj = builder.build_adjacency_matrix(graph)

        assert adj.is_sparse
        assert adj.shape == (3, 3)
        assert adj._nnz() == 3  # edges: 1->2, 1->3, 2->3


class TestFeatureProjectionRegistration:
    """Regression tests: lazy projection layers must be optimizer-visible."""

    def test_expected_feature_dims_pre_registers(self):
        """Projections passed via expected_feature_dims exist before forward()."""
        from stepnet.encoder import STEPEncoder

        encoder = STEPEncoder(expected_feature_dims=[16, 32])
        param_ids_before = {id(p) for p in encoder.parameters()}
        # The projection layers for dims 16 and 32 should already be registered
        assert "16" in encoder._feature_projs
        assert "32" in encoder._feature_projs
        # Their parameters are part of model.parameters()
        for proj in encoder._feature_projs.values():
            for p in proj.parameters():
                assert id(p) in param_ids_before

    def test_register_feature_projection_explicit(self):
        """register_feature_projection() adds params visible to optimizer."""
        from stepnet.encoder import STEPEncoder

        encoder = STEPEncoder()
        assert len(encoder._feature_projs) == 0

        encoder.register_feature_projection(64)
        assert "64" in encoder._feature_projs
        # Params are now in model.parameters()
        all_param_ids = {id(p) for p in encoder.parameters()}
        for p in encoder._feature_projs["64"].parameters():
            assert id(p) in all_param_ids

    def test_lazy_proj_updates_with_sync(self):
        """Trainer._sync_optimizer_params picks up lazily-created projections."""
        from stepnet.encoder import STEPEncoder

        encoder = STEPEncoder(graph_input_dim=48)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        initial_param_count = sum(1 for _ in encoder.parameters())

        # Simulate lazy creation during forward
        encoder.register_feature_projection(16)
        new_param_count = sum(1 for _ in encoder.parameters())
        assert new_param_count > initial_param_count

        # Simulate what _sync_optimizer_params does
        tracked_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
        new_params = [p for p in encoder.parameters()
                      if id(p) not in tracked_ids and p.requires_grad]
        assert len(new_params) > 0, "New projection params should be detected"
        optimizer.add_param_group({'params': new_params})

        # Verify all params are now tracked
        all_optim_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
        for p in encoder.parameters():
            assert id(p) in all_optim_ids
