"""Graph tokenizer for B-Rep topology data.

Serializes enriched B-Rep topology graphs (with node features, edge
features, and adjacency structure from cadling's enhanced_features
module) into flat token sequences for transformer consumption.

The serialization order is designed for autoregressive generation:
    1. GRAPH_START token (num_nodes, num_edges metadata)
    2. For each node:
       a. NODE_START token (node_index, node_type)
       b. Quantized feature tokens (one per feature dimension)
       c. Adjacency tokens (neighbor indices)
       d. NODE_END token
    3. For each edge:
       a. EDGE token (source, target)
       b. Quantized edge feature tokens
    4. GRAPH_END token

This linearization allows a transformer to learn the mapping from
partial token sequences to complete graph structures.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from ..config import GraphTokenizationConfig
from ..quantization.feature_quantizer import FeatureVectorQuantizer, FeatureQuantizationParams
from .token_types import (
    GraphEdgeToken,
    GraphNodeToken,
    GraphStructureToken,
    TokenSequence,
)

_log = logging.getLogger(__name__)


class GraphTokenizer:
    """Tokenize B-Rep topology graphs with node/edge features.

    Converts a topology graph with dense feature vectors on nodes and
    edges into a TokenSequence containing quantized graph tokens. The
    resulting flat token sequence can be encoded into integer IDs via
    CADVocabulary for transformer input.

    The tokenizer includes FeatureVectorQuantizer instances for both
    node and edge features. These quantizers can be pre-fitted on
    training data, or will auto-fit on the first call to tokenize().

    Args:
        config: Graph tokenization configuration.

    Example:
        config = GraphTokenizationConfig(node_bits=8, edge_bits=8)
        tokenizer = GraphTokenizer(config)

        # From cadling's TopologyGraph:
        node_feats = topology_graph.to_numpy_node_features()  # (N, 48)
        edge_idx = topology_graph.to_edge_index()              # (2, M)
        edge_feats = topology_graph.to_numpy_edge_features()   # (M, 16)

        token_seq = tokenizer.tokenize(node_feats, edge_idx, edge_feats)
    """

    def __init__(self, config: Optional[GraphTokenizationConfig] = None) -> None:
        self.config = config or GraphTokenizationConfig()
        self.node_quantizer = FeatureVectorQuantizer(
            bits=self.config.node_bits,
            strategy="per_dimension",
        )
        self.edge_quantizer = FeatureVectorQuantizer(
            bits=self.config.edge_bits,
            strategy="per_dimension",
        )
        self._node_params: Optional[FeatureQuantizationParams] = None
        self._edge_params: Optional[FeatureQuantizationParams] = None
        self._explicitly_fitted: bool = False

    def fit(
        self,
        node_features: np.ndarray,
        edge_features: Optional[np.ndarray] = None,
    ) -> None:
        """Pre-fit quantization parameters on training data.

        Call this before tokenize() to use consistent normalization
        across multiple graphs. If not called, tokenize() will auto-fit
        on each graph independently.

        Args:
            node_features: Training node features, shape (N_total, D_node).
            edge_features: Training edge features, shape (M_total, D_edge).
        """
        self._node_params = self.node_quantizer.fit(node_features)
        if edge_features is not None:
            self._edge_params = self.edge_quantizer.fit(edge_features)
        self._explicitly_fitted = True
        _log.info(
            "GraphTokenizer fitted: node_dim=%d, edge_dim=%s",
            node_features.shape[1],
            edge_features.shape[1] if edge_features is not None else "N/A",
        )

    def tokenize(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_features: Optional[np.ndarray] = None,
        node_types: Optional[list[str]] = None,
    ) -> TokenSequence:
        """Convert topology graph to token sequence.

        Args:
            node_features: Node feature array, shape (N, D_node) float32.
                Typically 48-dim for cadling face features.
            edge_index: Edge index array, shape (2, M) int64.
                Row 0 = source indices, row 1 = target indices.
            edge_features: Optional edge feature array, shape (M, D_edge) float32.
                Typically 16-dim for cadling edge features.
            node_types: Optional per-node type labels (e.g., ["face", "face", ...]).

        Returns:
            TokenSequence with graph_node_tokens, graph_edge_tokens,
            and graph_structure_tokens populated.

        Raises:
            ValueError: If array shapes are inconsistent.
        """
        # Validate inputs
        self._validate_inputs(node_features, edge_index, edge_features)

        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1] if edge_index.ndim == 2 else 0

        # Truncate if needed
        if num_nodes > self.config.max_nodes:
            _log.warning(
                "Truncating graph from %d to %d nodes",
                num_nodes, self.config.max_nodes,
            )
            node_features = node_features[:self.config.max_nodes]
            # Filter edges to only include nodes within range
            mask = (edge_index[0] < self.config.max_nodes) & (edge_index[1] < self.config.max_nodes)
            edge_index = edge_index[:, mask]
            if edge_features is not None:
                edge_features = edge_features[mask]
            num_nodes = self.config.max_nodes
            num_edges = edge_index.shape[1]

        if num_edges > self.config.max_edges:
            _log.warning(
                "Truncating edges from %d to %d",
                num_edges, self.config.max_edges,
            )
            edge_index = edge_index[:, :self.config.max_edges]
            if edge_features is not None:
                edge_features = edge_features[:self.config.max_edges]
            num_edges = self.config.max_edges

        # Fit quantizers: use cached params only when explicitly fitted
        if self._explicitly_fitted:
            node_params = self._node_params
            edge_params = self._edge_params
        else:
            # Auto-fit fresh params per call — don't cache so graph B
            # doesn't reuse graph A's normalization parameters
            node_params = self.node_quantizer.fit(node_features)
            edge_params = None

        if edge_features is not None and edge_params is None:
            edge_params = self.edge_quantizer.fit(edge_features)

        # Quantize features
        q_node_feats = self.node_quantizer.quantize(node_features, node_params)
        q_edge_feats = None
        if edge_features is not None and edge_params is not None:
            q_edge_feats = self.edge_quantizer.quantize(edge_features, edge_params)

        # Build adjacency map
        adjacency: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
        for e in range(num_edges):
            src = int(edge_index[0, e])
            tgt = int(edge_index[1, e])
            if src < num_nodes:
                adjacency[src].append(tgt)

        # Serialize to tokens
        structure_tokens: list[GraphStructureToken] = []
        node_tokens: list[GraphNodeToken] = []
        edge_tokens: list[GraphEdgeToken] = []

        # 1. GRAPH_START
        structure_tokens.append(GraphStructureToken(
            token_type="graph_start",
            value=num_nodes,
        ))

        # 2. Nodes
        for node_idx in range(num_nodes):
            ntype = node_types[node_idx] if node_types and node_idx < len(node_types) else "face"

            # NODE_START
            structure_tokens.append(GraphStructureToken(
                token_type="node_start",
                value=node_idx,
            ))

            # Node feature tokens
            feat_tokens = q_node_feats[node_idx].tolist()
            node_tokens.append(GraphNodeToken(
                node_index=node_idx,
                feature_tokens=feat_tokens,
                node_type=ntype,
                bits=self.config.node_bits,
            ))

            # Adjacency tokens
            if self.config.adjacency_encoding == "explicit":
                for neighbor in adjacency.get(node_idx, []):
                    structure_tokens.append(GraphStructureToken(
                        token_type="adjacency",
                        value=neighbor,
                    ))

            # NODE_END
            structure_tokens.append(GraphStructureToken(
                token_type="node_end",
                value=node_idx,
            ))

        # 3. Edges with features
        for e in range(num_edges):
            src = int(edge_index[0, e])
            tgt = int(edge_index[1, e])

            if src > 65535 or tgt > 65535:
                raise ValueError(
                    f"Edge node indices ({src}, {tgt}) exceed 16-bit packing "
                    f"limit of 65535. Reduce graph size or max_nodes."
                )

            structure_tokens.append(GraphStructureToken(
                token_type="edge",
                value=(src << 16) | (tgt & 0xFFFF),
            ))

            if q_edge_feats is not None:
                feat_tokens = q_edge_feats[e].tolist()
                edge_tokens.append(GraphEdgeToken(
                    source_index=src,
                    target_index=tgt,
                    feature_tokens=feat_tokens,
                    bits=self.config.edge_bits,
                ))

        # 4. GRAPH_END
        structure_tokens.append(GraphStructureToken(
            token_type="graph_end",
            value=0,
        ))

        seq = TokenSequence(
            graph_node_tokens=node_tokens,
            graph_edge_tokens=edge_tokens,
            graph_structure_tokens=structure_tokens,
            metadata={
                "tokenizer": "graph",
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "node_feature_dim": node_features.shape[1],
                "edge_feature_dim": edge_features.shape[1] if edge_features is not None else 0,
                "node_bits": self.config.node_bits,
                "edge_bits": self.config.edge_bits,
                "adjacency_encoding": self.config.adjacency_encoding,
            },
        )

        _log.debug(
            "Tokenized graph: %d nodes, %d edges → %d node tokens, "
            "%d edge tokens, %d structure tokens",
            num_nodes, num_edges, len(node_tokens),
            len(edge_tokens), len(structure_tokens),
        )
        return seq

    def detokenize(self, token_sequence: TokenSequence) -> dict[str, Any]:
        """Reconstruct graph structure from token sequence.

        Inverse of tokenize(). Recovers approximate node features, edge
        features, and adjacency structure from the token sequence.

        Args:
            token_sequence: TokenSequence with graph tokens.

        Returns:
            Dict with keys:
                - node_features: Reconstructed (N, D) float32 array
                - edge_index: (2, M) int64 array
                - edge_features: Reconstructed (M, D) float32 array or None
                - num_nodes: int
                - num_edges: int
                - node_types: list[str]
                - adjacency: dict[int, list[int]]
        """
        node_tokens = token_sequence.graph_node_tokens
        edge_tokens = token_sequence.graph_edge_tokens
        structure_tokens = token_sequence.graph_structure_tokens

        # Reconstruct node features
        num_nodes = len(node_tokens)
        node_types: list[str] = []
        adjacency: dict[int, list[int]] = {}

        if num_nodes > 0:
            feat_dim = node_tokens[0].num_features
            q_node_feats = np.zeros((num_nodes, feat_dim), dtype=np.int64)
            for i, nt in enumerate(node_tokens):
                q_node_feats[i] = nt.feature_tokens[:feat_dim]
                node_types.append(nt.node_type)

            # Dequantize
            node_params = self._node_params or self.node_quantizer.fitted_params
            if node_params is None:
                raise ValueError(
                    "Cannot dequantize node features: no quantization params "
                    "available. Call tokenize() or fit() first so that "
                    "fitted_params are populated."
                )
            node_features = self.node_quantizer.dequantize(q_node_feats, node_params)
        else:
            node_features = np.zeros((0, self.config.node_feature_dim), dtype=np.float32)

        # Reconstruct adjacency from explicit adjacency structure tokens.
        # During tokenization, adjacency tokens are emitted between
        # node_start and node_end markers with value = neighbor_index.
        # We track the current node via node_start/node_end boundaries.
        current_node: Optional[int] = None
        for st in structure_tokens:
            if st.token_type == "node_start":
                current_node = st.value
            elif st.token_type == "node_end":
                current_node = None
            elif st.token_type == "adjacency" and current_node is not None:
                if current_node not in adjacency:
                    adjacency[current_node] = []
                adjacency[current_node].append(st.value)

        # Reconstruct edge features and edge index
        num_edges = len(edge_tokens)
        if num_edges > 0:
            feat_dim = edge_tokens[0].num_features
            edge_index = np.zeros((2, num_edges), dtype=np.int64)
            q_edge_feats = np.zeros((num_edges, feat_dim), dtype=np.int64)

            for i, et in enumerate(edge_tokens):
                edge_index[0, i] = et.source_index
                edge_index[1, i] = et.target_index
                q_edge_feats[i] = et.feature_tokens[:feat_dim]

            edge_params = self._edge_params or self.edge_quantizer.fitted_params
            if edge_params is None:
                raise ValueError(
                    "Cannot dequantize edge features: no quantization params "
                    "available. Call tokenize() or fit() first so that "
                    "fitted_params are populated."
                )
            edge_features = self.edge_quantizer.dequantize(q_edge_feats, edge_params)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = None

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "node_types": node_types,
            "adjacency": adjacency,
        }

    def _validate_inputs(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_features: Optional[np.ndarray],
    ) -> None:
        """Validate input array shapes and types."""
        if node_features.ndim != 2:
            raise ValueError(
                f"node_features must be 2D (N, D), got shape {node_features.shape}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must be (2, M), got shape {edge_index.shape}"
            )
        if edge_features is not None:
            if edge_features.ndim != 2:
                raise ValueError(
                    f"edge_features must be 2D (M, D), got shape {edge_features.shape}"
                )
            if edge_features.shape[0] != edge_index.shape[1]:
                raise ValueError(
                    f"edge_features rows ({edge_features.shape[0]}) != "
                    f"edge_index columns ({edge_index.shape[1]})"
                )
