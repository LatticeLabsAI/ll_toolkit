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

        # Transformer decoding (self-attention with causal mask)
        # Note: TransformerDecoder expects memory, we use self-attention only
        x = self.transformer(x, memory=x, tgt_mask=causal_mask, tgt_key_padding_mask=padding_mask)

        x = self.layer_norm(x)

        return x


class STEPGraphEncoder(nn.Module):
    """
    Graph neural network for STEP topology.
    Processes entity reference graph.
    """

    def __init__(self, input_dim: int = 129, node_dim: int = 128, edge_dim: int = 64, num_layers: int = 3):
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
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        token_embed_dim: int = 256,
        graph_node_dim: int = 128,
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

        # Graph network for topology
        self.graph_encoder = STEPGraphEncoder(
            node_dim=graph_node_dim,
            num_layers=num_graph_layers
        )

        # Projection for geometric features
        self.geom_projection = nn.Linear(64, graph_node_dim)  # Max 64 params per entity

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
            topology_data: Optional dict with:
                - adjacency_matrix: [num_nodes, num_nodes]
                - node_features: [num_nodes, feature_dim]

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
                # Create default node features
                num_nodes = adj_matrix.shape[0]
                node_features = torch.randn(num_nodes, self.graph_encoder.node_dim)

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


def build_step_encoder(
    vocab_size: int = 50000,
    output_dim: int = 1024,
    **kwargs
) -> STEPEncoder:
    """Build STEP encoder with default config."""
    return STEPEncoder(vocab_size=vocab_size, output_dim=output_dim, **kwargs)
