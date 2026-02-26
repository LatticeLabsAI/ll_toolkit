"""Unit tests for GraphTokenizer."""
from __future__ import annotations

import numpy as np
import pytest

from geotoken.tokenizer.graph_tokenizer import GraphTokenizer


@pytest.fixture
def tokenizer():
    return GraphTokenizer()


def _make_simple_graph():
    """4 nodes, 4 edges, 8-dim features."""
    np.random.seed(42)
    node_features = np.random.randn(4, 8).astype(np.float32)
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)
    edge_features = np.random.randn(4, 4).astype(np.float32)
    return node_features, edge_index, edge_features


class TestGraphTokenizer:
    def test_tokenize_simple_graph(self, tokenizer):
        node_feats, edge_idx, edge_feats = _make_simple_graph()
        result = tokenizer.tokenize(node_feats, edge_idx, edge_feats)
        assert len(result.graph_node_tokens) == 4
        assert len(result.graph_edge_tokens) == 4
        assert len(result.graph_structure_tokens) > 0

    def test_fit_then_tokenize(self, tokenizer):
        node_feats, edge_idx, edge_feats = _make_simple_graph()
        tokenizer.fit(node_feats, edge_feats)
        r1 = tokenizer.tokenize(node_feats, edge_idx, edge_feats)
        r2 = tokenizer.tokenize(node_feats, edge_idx, edge_feats)
        # Same input with fitted params should give same tokens
        for t1, t2 in zip(r1.graph_node_tokens, r2.graph_node_tokens):
            assert t1.feature_tokens == t2.feature_tokens

    def test_detokenize_roundtrip(self, tokenizer):
        node_feats, edge_idx, edge_feats = _make_simple_graph()
        tokenizer.fit(node_feats, edge_feats)
        tokens = tokenizer.tokenize(node_feats, edge_idx, edge_feats)
        result = tokenizer.detokenize(tokens)
        reconstructed = result["node_features"]
        assert reconstructed.shape == node_feats.shape
        # Quantization error should be bounded
        error = np.max(np.abs(reconstructed - node_feats))
        assert error < 1.0, f"Max reconstruction error {error} too large"

    def test_invalid_edge_index_shape(self, tokenizer):
        node_feats = np.random.randn(4, 8).astype(np.float32)
        bad_edge = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)  # (3, 2) not (2, M)
        with pytest.raises(ValueError, match="edge_index must be \\(2, M\\)"):
            tokenizer.tokenize(node_feats, bad_edge)

    def test_empty_graph(self, tokenizer):
        node_feats = np.zeros((0, 8), dtype=np.float32)
        edge_idx = np.zeros((2, 0), dtype=np.int64)
        # Empty graph triggers min/max on empty array in quantizer;
        # pre-fit with dummy data so quantizer has valid params
        tokenizer.fit(np.random.randn(2, 8).astype(np.float32))
        result = tokenizer.tokenize(node_feats, edge_idx)
        assert len(result.graph_node_tokens) == 0
        assert len(result.graph_edge_tokens) == 0
