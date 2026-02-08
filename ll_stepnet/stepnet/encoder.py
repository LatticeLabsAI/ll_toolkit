"""
STEP Encoder Module
Neural network that processes tokenized STEP data.
Uses tokenizer, features, and topology separately.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .tokenizer import STEPTokenizer
from .features import STEPFeatureExtractor
from .topology import STEPTopologyBuilder


class STEPTransformerEncoder(nn.Module):
    """
    Transformer encoder for STEP token sequences.
    Standard transformer architecture.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, 5000, embed_dim))

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            token_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] optional

        Returns:
            encoded: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = token_ids.shape

        # Embed tokens
        x = self.token_embedding(token_ids)  # [batch, seq_len, embed_dim]

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]

        x = self.dropout(x)

        # Create attention mask if provided
        if attention_mask is not None:
            # Convert to transformer mask format (True = ignore)
            mask = (attention_mask == 0)
        else:
            mask = None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        x = self.layer_norm(x)

        return x


class STEPTransformerDecoder(nn.Module):
    """
    Transformer decoder for STEP token sequences with causal attention.
    Used for autoregressive generation (GPT-style).

    Supports optional conditioning via a conditioner module (TextConditioner,
    ImageConditioner, or MultiModalConditioner) that applies cross-attention
    to inject text/image features. Following Text2CAD, the first N blocks
    can skip cross-attention via the conditioner's skip_cross_attention_blocks.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, 5000, embed_dim))

        # Transformer decoder layers (causal attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_memory: Optional[torch.Tensor] = None,
        conditioner: Optional[nn.Module] = None,
        conditioning_inputs: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] optional
            cross_attention_memory: [batch_size, mem_len, embed_dim] optional
                When provided, used as the memory (key/value) for the
                decoder cross-attention layers. When None, the decoder
                uses self-attention only (existing behaviour).
            conditioner: Optional conditioning module (TextConditioner,
                ImageConditioner, or MultiModalConditioner). When provided,
                applies cross-attention conditioning after each decoder block.
            conditioning_inputs: Dict with conditioning data for the conditioner:
                - text_input_ids: [B, L] text token ids
                - text_attention_mask: [B, L] text padding mask
                - pixel_values: [B, C, H, W] image tensors

        Returns:
            encoded: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = token_ids.shape

        # Embed tokens
        x = self.token_embedding(token_ids)  # [batch, seq_len, embed_dim]

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]

        x = self.dropout(x)

        # Create causal mask (can only attend to previous tokens)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len,
            device=token_ids.device
        )

        # Create attention mask if provided
        if attention_mask is not None:
            # Convert to transformer mask format (True = ignore)
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None

        # Determine memory for the decoder cross-attention
        # When cross_attention_memory is provided, use it as external memory;
        # otherwise fall back to self-attention (memory = x).
        if cross_attention_memory is not None:
            memory = cross_attention_memory
        else:
            memory = x

        # Process through decoder layers with optional per-layer conditioning
        if conditioner is not None and conditioning_inputs is not None:
            # Apply conditioning with block_index for skip logic (Text2CAD)
            x = self._forward_with_conditioning(
                x, memory, causal_mask, padding_mask,
                conditioner, conditioning_inputs,
            )
        else:
            # Standard decoder forward
            x = self.transformer(
                x, memory=memory, tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask,
            )

        x = self.layer_norm(x)
        return x

    def _forward_with_conditioning(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        conditioner: nn.Module,
        conditioning_inputs: Dict,
    ) -> torch.Tensor:
        """Forward pass through decoder layers with per-layer conditioning.

        Applies the conditioner after each decoder layer, passing the
        block_index so the conditioner can skip cross-attention for
        early blocks (Text2CAD design).

        Args:
            x: Hidden states [B, S, D].
            memory: Memory for cross-attention [B, M, D].
            causal_mask: Causal attention mask [S, S].
            padding_mask: Padding mask [B, S].
            conditioner: Conditioner module.
            conditioning_inputs: Dict with text_input_ids, text_attention_mask,
                and/or pixel_values.

        Returns:
            Updated hidden states [B, S, D].
        """
        # Access individual decoder layers
        for block_index, layer in enumerate(self.transformer.layers):
            # Standard decoder layer forward (self-attn + cross-attn + FFN)
            x = layer(
                x, memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask,
            )

            # Apply conditioner with block_index for skip logic
            # The conditioner will skip cross-attention if block_index < skip_blocks
            x = conditioner(
                hidden_states=x,
                block_index=block_index,
                **conditioning_inputs,
            )

        return x


class STEPGraphEncoder(nn.Module):
    """
    Graph neural network for STEP/B-Rep topology.

    Processes entity reference graphs from either:
    - ll_stepnet's STEPTopologyBuilder (129-dim features: 128 numeric + 1 hash)
    - cadling's TopologyGraph (48-dim features, native format)

    The input_dim parameter controls which format is accepted.  When
    working with cadling data, set ``input_dim=48`` to accept cadling's
    native topology features directly with no conversion.
    """

    def __init__(self, input_dim: int = 48, node_dim: int = 128, edge_dim: int = 64, num_layers: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Input projection layer (from feature dim to node dim)
        self.input_proj = nn.Linear(input_dim, node_dim)

        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            nn.Linear(node_dim, node_dim) for _ in range(num_layers)
        ])

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, input_dim]
            adjacency_matrix: [num_nodes, num_nodes]

        Returns:
            updated_features: [num_nodes, node_dim]
        """
        # Project input features to node_dim
        x = self.input_proj(node_features)
        x = self.activation(x)

        for conv in self.conv_layers:
            # Message passing: aggregate neighbor features
            messages = torch.matmul(adjacency_matrix, x)  # [num_nodes, node_dim]

            # Update with MLP
            x_new = conv(messages)
            x_new = self.activation(x_new)
            x_new = self.dropout(x_new)

            # Residual connection
            x = x + x_new

        return x


class STEPEncoder(nn.Module):
    """
    Complete STEP encoder combining all components.

    Architecture:
        1. Tokenizer: Text → Token IDs (external)
        2. Transformer: Token IDs → Sequence features
        3. Feature Extractor: Entities → Geometric features (external)
        4. Graph Network: Topology → Structural features
        5. Fusion: Combine all representations

    The graph encoder accepts node features from either source natively:
    - cadling's TopologyGraph: 48-dim features (default)
    - ll_stepnet's STEPTopologyBuilder: 129-dim features (set graph_input_dim=129)

    No adapters or conversion needed — pass topology_data directly from
    cadling's TopologyGraph or ll_stepnet's topology builder.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        token_embed_dim: int = 256,
        graph_node_dim: int = 128,
        graph_input_dim: int = 48,
        output_dim: int = 1024,
        num_transformer_layers: int = 6,
        num_graph_layers: int = 3
    ):
        super().__init__()

        self.output_dim = output_dim

        # Transformer for token sequences
        self.transformer_encoder = STEPTransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=token_embed_dim,
            num_layers=num_transformer_layers
        )

        # Graph network for topology — accepts cadling 48-dim features natively
        self.graph_encoder = STEPGraphEncoder(
            input_dim=graph_input_dim,
            node_dim=graph_node_dim,
            num_layers=num_graph_layers
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(token_embed_dim + graph_node_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, token_ids: torch.Tensor, topology_data: Optional[Dict] = None) -> torch.Tensor:
        """
        Args:
            token_ids: [batch_size, seq_len] from STEPTokenizer
            topology_data: Dict with topology from either cadling or ll_stepnet:
                - adjacency_matrix: [num_nodes, num_nodes]
                - node_features: [num_nodes, feature_dim]
                  (48-dim from cadling TopologyGraph, or 129-dim from STEPTopologyBuilder)

        Returns:
            encoded: [batch_size, output_dim] final encoding
        """
        # Encode token sequence
        token_features = self.transformer_encoder(token_ids)  # [batch, seq_len, embed_dim]

        # Pool token features (mean pooling)
        token_pooled = token_features.mean(dim=1)  # [batch, embed_dim]

        # Encode topology if provided
        if topology_data is not None and 'adjacency_matrix' in topology_data:
            adj_matrix = topology_data['adjacency_matrix']
            node_features = topology_data.get('node_features')

            if node_features is None:
                # Create default node features matching the graph encoder input_dim
                num_nodes = adj_matrix.shape[0]
                node_features = torch.zeros(num_nodes, self.graph_encoder.input_dim)

            # Auto-project if feature dim doesn't match graph encoder input_dim
            if node_features.shape[-1] != self.graph_encoder.input_dim:
                # Project to expected dim (handles cadling 48→129 or vice versa)
                proj = nn.Linear(
                    node_features.shape[-1],
                    self.graph_encoder.input_dim,
                    device=node_features.device,
                )
                node_features = proj(node_features)

            # Graph encoding
            graph_features = self.graph_encoder(node_features, adj_matrix)  # [num_nodes, node_dim]

            # Pool graph features
            graph_pooled = graph_features.mean(dim=0)  # [node_dim]

            # Expand to batch
            graph_pooled = graph_pooled.unsqueeze(0).expand(token_pooled.shape[0], -1)

            # Concatenate token and graph features
            combined = torch.cat([token_pooled, graph_pooled], dim=-1)
        else:
            # No topology - just use token features with zero padding
            zero_graph = torch.zeros(
                token_pooled.shape[0],
                self.graph_encoder.node_dim,
                device=token_pooled.device
            )
            combined = torch.cat([token_pooled, zero_graph], dim=-1)

        # Fuse features
        output = self.fusion(combined)  # [batch, output_dim]

        return output


    @staticmethod
    def prepare_topology_data(topology_obj) -> Dict[str, torch.Tensor]:
        """Convert a cadling TopologyGraph or raw dict to forward()-ready format.

        Accepts either:
        - A dict already in forward() format (with ``adjacency_matrix`` and
          ``node_features`` tensor values) — returned as-is after ensuring
          values are tensors.
        - A cadling ``TopologyGraph`` object (or any object with
          ``to_numpy_node_features()`` and ``to_edge_index()`` methods) —
          extracts numpy arrays, builds the adjacency matrix, and converts
          to tensors.

        This lets callers pass a cadling TopologyGraph directly without
        writing glue code::

            topo = cadling_item.topology_graph  # cadling TopologyGraph
            out = encoder(token_ids, STEPEncoder.prepare_topology_data(topo))

        Args:
            topology_obj: Either a dict with ``adjacency_matrix`` and
                ``node_features`` keys, or an object with
                ``to_numpy_node_features()`` / ``to_edge_index()`` methods
                (e.g. cadling's ``TopologyGraph``).

        Returns:
            Dict with ``adjacency_matrix`` ``[N, N]`` and ``node_features``
            ``[N, D]`` float tensors ready for :meth:`forward`.
        """
        import numpy as np

        # --- Already a dict: ensure tensor values and return ----------
        if isinstance(topology_obj, dict):
            result = {}
            for key in ('adjacency_matrix', 'node_features'):
                val = topology_obj.get(key)
                if val is not None and not isinstance(val, torch.Tensor):
                    val = torch.tensor(np.asarray(val), dtype=torch.float32)
                result[key] = val
            # Pass through any extra keys (edge_index, etc.)
            for key, val in topology_obj.items():
                if key not in result:
                    result[key] = val
            return result

        # --- Duck-type: cadling TopologyGraph or similar --------------
        if not hasattr(topology_obj, 'to_numpy_node_features'):
            raise TypeError(
                f"Expected dict or object with to_numpy_node_features(), "
                f"got {type(topology_obj).__name__}"
            )

        node_feats_np = topology_obj.to_numpy_node_features()   # (N, D) float32
        num_nodes = node_feats_np.shape[0]

        # Build dense adjacency matrix from edge_index (2, M) int64
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        if hasattr(topology_obj, 'to_edge_index'):
            edge_index = topology_obj.to_edge_index()           # (2, M) int64
            if edge_index.shape[1] > 0:
                src = edge_index[0]
                dst = edge_index[1]
                # Clip to valid range (defensive)
                valid = (src < num_nodes) & (dst < num_nodes)
                adj_matrix[src[valid], dst[valid]] = 1.0

        return {
            'adjacency_matrix': torch.tensor(adj_matrix, dtype=torch.float32),
            'node_features': torch.tensor(node_feats_np, dtype=torch.float32),
        }


def build_step_encoder(
    vocab_size: int = 50000,
    output_dim: int = 1024,
    **kwargs
) -> STEPEncoder:
    """Build STEP encoder with default config."""
    return STEPEncoder(vocab_size=vocab_size, output_dim=output_dim, **kwargs)
