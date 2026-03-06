"""
STEP Encoder Module
Neural network that processes tokenized STEP data.
Uses tokenizer, features, and topology separately.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .tokenizer import STEPTokenizer
from .features import STEPFeatureExtractor
from .topology import STEPTopologyBuilder

_log = logging.getLogger(__name__)


def _sinusoidal_positional_encoding(max_len: int, embed_dim: int) -> torch.Tensor:
    """Generate sinusoidal positional encoding (Vaswani et al., 2017).

    Returns:
        Tensor of shape [1, max_len, embed_dim] with sin/cos positional signals.
    """
    pe = torch.zeros(max_len, embed_dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, max_len, embed_dim]


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

        # Positional encoding (sinusoidal init for faster convergence)
        self.pos_embedding = nn.Parameter(
            _sinusoidal_positional_encoding(5000, embed_dim)
        )

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
        self._num_heads = num_heads
        self._ff_dim = ff_dim
        self._dropout_rate = dropout

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding (sinusoidal init for faster convergence)
        self.pos_embedding = nn.Parameter(
            _sinusoidal_positional_encoding(5000, embed_dim)
        )

        # Lazy-built stacks: only one is instantiated per use-case to halve
        # decoder memory.  Both are registered as proper submodules so that
        # state_dict / load_state_dict still work transparently.
        self._causal_encoder: Optional[nn.TransformerEncoder] = None
        self._transformer: Optional[nn.TransformerDecoder] = None

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    @property
    def causal_encoder(self) -> nn.TransformerEncoder:
        """Lazily build the GPT-style causal encoder on first access."""
        if self._causal_encoder is None:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self._num_heads,
                dim_feedforward=self._ff_dim,
                dropout=self._dropout_rate,
                batch_first=True,
            )
            self._causal_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.num_layers,
            )
            # Move to same device/dtype as token embedding
            self._causal_encoder = self._causal_encoder.to(
                device=self.token_embedding.weight.device,
                dtype=self.token_embedding.weight.dtype,
            )
            # Register as submodule so state_dict sees it
            self.add_module('_causal_encoder', self._causal_encoder)
        return self._causal_encoder

    @property
    def transformer(self) -> nn.TransformerDecoder:
        """Lazily build the cross-attention decoder on first access."""
        if self._transformer is None:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=self._num_heads,
                dim_feedforward=self._ff_dim,
                dropout=self._dropout_rate,
                batch_first=True,
            )
            self._transformer = nn.TransformerDecoder(
                decoder_layer, num_layers=self.num_layers,
            )
            self._transformer = self._transformer.to(
                device=self.token_embedding.weight.device,
                dtype=self.token_embedding.weight.dtype,
            )
            self.add_module('_transformer', self._transformer)
        return self._transformer

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Backward-compatible loading: remap old causal_encoder.*/transformer.* keys."""
        remaps = {}
        for key in list(state_dict.keys()):
            if key.startswith(prefix + 'causal_encoder.'):
                new_key = key.replace(prefix + 'causal_encoder.', prefix + '_causal_encoder.', 1)
                remaps[key] = new_key
            elif key.startswith(prefix + 'transformer.'):
                new_key = key.replace(prefix + 'transformer.', prefix + '_transformer.', 1)
                remaps[key] = new_key
        for old_key, new_key in remaps.items():
            state_dict[new_key] = state_dict.pop(old_key)

        # Eagerly build any stack whose weights appear in the state_dict
        if any(k.startswith(prefix + '_causal_encoder.') for k in state_dict):
            _ = self.causal_encoder  # trigger lazy init
        if any(k.startswith(prefix + '_transformer.') for k in state_dict):
            _ = self.transformer  # trigger lazy init

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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

        # Route: cross-attention decoder vs GPT-style causal encoder
        has_memory = cross_attention_memory is not None
        has_conditioner = conditioner is not None and conditioning_inputs is not None

        if has_conditioner:
            # Conditioning path always uses the decoder (cross-attention needed)
            memory = cross_attention_memory if has_memory else x
            x = self._forward_with_conditioning(
                x, memory, causal_mask, padding_mask,
                conditioner, conditioning_inputs,
            )
        elif has_memory:
            # External memory provided — use decoder cross-attention
            x = self.transformer(
                x, memory=cross_attention_memory, tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask,
            )
        else:
            # GPT-style: causal self-attention only (no cross-attention)
            x = self.causal_encoder(
                x, mask=causal_mask,
                src_key_padding_mask=padding_mask,
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

    @staticmethod
    def _to_sparse(adj: torch.Tensor) -> torch.Tensor:
        """Convert adjacency to sparse COO format if not already sparse.

        Accepts dense [N, N], sparse COO, or sparse CSR tensors.
        Returns a coalesced sparse COO tensor.
        """
        if adj.is_sparse or adj.layout == torch.sparse_csr:
            return adj.to_sparse_coo().coalesce()
        # Dense -> sparse COO
        indices = adj.nonzero(as_tuple=False).t().contiguous()  # [2, nnz]
        values = adj[indices[0], indices[1]]
        return torch.sparse_coo_tensor(
            indices, values, adj.shape, device=adj.device, dtype=adj.dtype,
        ).coalesce()

    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, input_dim]
            adjacency_matrix: [num_nodes, num_nodes] — dense, sparse COO, or sparse CSR.
                Sparse inputs avoid O(N^2) memory for large B-Rep graphs.

        Returns:
            updated_features: [num_nodes, node_dim]
        """
        num_nodes = node_features.shape[0]

        # Project input features to node_dim
        x = self.input_proj(node_features)
        x = self.activation(x)

        # Convert to sparse COO for memory-efficient GCN
        adj_sp = self._to_sparse(adjacency_matrix)

        # Add self-loops: A_hat = A + I (sparse)
        self_loop_indices = torch.arange(num_nodes, device=adj_sp.device)
        self_loop_indices = torch.stack([self_loop_indices, self_loop_indices])  # [2, N]
        self_loop_values = torch.ones(num_nodes, device=adj_sp.device, dtype=adj_sp.dtype)
        identity_sp = torch.sparse_coo_tensor(
            self_loop_indices, self_loop_values, (num_nodes, num_nodes),
            device=adj_sp.device, dtype=adj_sp.dtype,
        )
        adj_hat = (adj_sp + identity_sp).coalesce()

        # Symmetric normalization: D^{-1/2} A_hat D^{-1/2} (standard GCN)
        # Compute degree from sparse tensor
        deg = torch.sparse.sum(adj_hat, dim=1).to_dense().clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)

        # Scale values: adj_norm[i,j] = deg_inv_sqrt[i] * adj_hat[i,j] * deg_inv_sqrt[j]
        row, col = adj_hat.indices()
        vals = adj_hat.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj_norm = torch.sparse_coo_tensor(
            adj_hat.indices(), vals, adj_hat.shape,
            device=adj_hat.device, dtype=adj_hat.dtype,
        ).coalesce()

        for conv in self.conv_layers:
            # Message passing: sparse matmul — O(nnz * node_dim) instead of O(N^2 * node_dim)
            messages = torch.sparse.mm(adj_norm, x)  # [num_nodes, node_dim]

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
        num_graph_layers: int = 3,
        expected_feature_dims: Optional[List[int]] = None,
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

        # Projection layers for mismatched feature dims, keyed by input dim.
        # Registered as nn.ModuleDict so weights are part of state_dict/optimizer.
        self._feature_projs = nn.ModuleDict()

        # Pre-register projections for known feature dims so they are captured
        # by the optimizer at construction time.
        if expected_feature_dims:
            for dim in expected_feature_dims:
                if dim != graph_input_dim:
                    self.register_feature_projection(dim)

        # Track whether lazy projections were created after init (for warnings)
        self._lazy_proj_warning_issued: set = set()

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(token_embed_dim + graph_node_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def register_feature_projection(self, in_dim: int) -> nn.Linear:
        """Pre-register a projection layer for a given input feature dimension.

        Call this before constructing the optimizer to ensure the projection
        parameters are included in ``model.parameters()``.

        Args:
            in_dim: Input feature dimension to project from.

        Returns:
            The ``nn.Linear`` projection layer (also stored in
            ``self._feature_projs``).
        """
        key = str(in_dim)
        if key not in self._feature_projs:
            self._feature_projs[key] = nn.Linear(
                in_dim, self.graph_encoder.input_dim
            )
        return self._feature_projs[key]

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Pre-create lazy projection layers found in checkpoint before loading.

        ``_feature_projs`` entries are created lazily during ``forward()`` so a
        freshly constructed model has an empty ``ModuleDict``.  Without this
        override, ``load_state_dict`` would either error (strict=True) or
        silently drop the learned projection weights (strict=False).
        """
        prefix = "_feature_projs."
        proj_keys: Dict[str, int] = {}
        for key in state_dict:
            if key.startswith(prefix):
                # key format: "_feature_projs.<dim>.weight" or ".bias"
                parts = key[len(prefix):].split(".")
                dim_key = parts[0]
                if dim_key not in proj_keys and key.endswith(".weight"):
                    # infer in_features from weight shape [out, in]
                    proj_keys[dim_key] = state_dict[key].shape[1]

        for dim_key, in_dim in proj_keys.items():
            if dim_key not in self._feature_projs:
                self._feature_projs[dim_key] = nn.Linear(
                    in_dim, self.graph_encoder.input_dim
                )

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, token_ids: torch.Tensor, topology_data: Optional[Dict] = None) -> torch.Tensor:
        """
        Args:
            token_ids: [batch_size, seq_len] from STEPTokenizer
            topology_data: Dict with topology from either cadling or ll_stepnet:
                - adjacency_matrix: [num_nodes, num_nodes] (dense, sparse COO, or sparse CSR)
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
        # Normalize topology_data: accept a single dict or a list of dicts
        topo_list: Optional[List[Optional[Dict]]] = None
        if topology_data is not None:
            if isinstance(topology_data, list):
                topo_list = topology_data
            elif 'adjacency_matrix' in topology_data:
                topo_list = [topology_data]

        if topo_list is not None:
            batch_size = token_pooled.shape[0]
            # Pad list to batch_size (trailing items get zero graph features)
            while len(topo_list) < batch_size:
                topo_list.append(None)

            per_sample_pooled = []
            for topo in topo_list[:batch_size]:
                if topo is not None and 'adjacency_matrix' in topo:
                    adj_matrix = topo['adjacency_matrix']
                    node_features = topo.get('node_features')

                    if node_features is None:
                        num_nodes = adj_matrix.shape[0]
                        node_features = torch.zeros(
                            num_nodes, self.graph_encoder.input_dim,
                            device=token_pooled.device,
                        )

                    # Auto-project if feature dim doesn't match graph encoder input_dim
                    if node_features.shape[-1] != self.graph_encoder.input_dim:
                        in_dim = node_features.shape[-1]
                        key = str(in_dim)
                        if key not in self._feature_projs:
                            if self.training and key not in self._lazy_proj_warning_issued:
                                _log.warning(
                                    "Creating feature projection for dim=%d during "
                                    "forward(). These parameters are NOT in the "
                                    "optimizer. Call register_feature_projection(%d) "
                                    "before constructing the optimizer, or pass "
                                    "expected_feature_dims=[%d] to __init__.",
                                    in_dim, in_dim, in_dim,
                                )
                                self._lazy_proj_warning_issued.add(key)
                            proj = self.register_feature_projection(in_dim)
                            proj.to(node_features.device)
                        node_features = self._feature_projs[key](node_features)

                    graph_features = self.graph_encoder(node_features, adj_matrix)
                    per_sample_pooled.append(graph_features.mean(dim=0))
                else:
                    per_sample_pooled.append(
                        torch.zeros(self.graph_encoder.node_dim, device=token_pooled.device)
                    )

            graph_pooled = torch.stack(per_sample_pooled, dim=0)  # [batch, node_dim]

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
          extracts numpy arrays, builds a **sparse** adjacency matrix, and
          converts to tensors.

        The adjacency matrix is returned as a sparse COO tensor to avoid
        O(N^2) memory on large B-Rep graphs.

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
            Dict with ``adjacency_matrix`` (sparse COO ``[N, N]``) and
            ``node_features`` ``[N, D]`` float tensors ready for
            :meth:`forward`.
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

        # Build sparse adjacency matrix from edge_index (2, M) int64
        if hasattr(topology_obj, 'to_edge_index'):
            edge_index = topology_obj.to_edge_index()           # (2, M) int64
            if edge_index.shape[1] > 0:
                src = edge_index[0]
                dst = edge_index[1]
                # Clip to valid range (defensive)
                valid = (src < num_nodes) & (dst < num_nodes)
                indices = torch.tensor(
                    np.stack([src[valid], dst[valid]]), dtype=torch.long,
                )
                values = torch.ones(indices.shape[1], dtype=torch.float32)
                adj_matrix = torch.sparse_coo_tensor(
                    indices, values, (num_nodes, num_nodes),
                ).coalesce()
            else:
                adj_matrix = torch.sparse_coo_tensor(
                    torch.zeros(2, 0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.float32),
                    (num_nodes, num_nodes),
                )
        else:
            adj_matrix = torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                (num_nodes, num_nodes),
            )

        return {
            'adjacency_matrix': adj_matrix,
            'node_features': torch.tensor(node_feats_np, dtype=torch.float32),
        }


def build_step_encoder(
    vocab_size: int = 50000,
    output_dim: int = 1024,
    **kwargs
) -> STEPEncoder:
    """Build STEP encoder with default config."""
    return STEPEncoder(vocab_size=vocab_size, output_dim=output_dim, **kwargs)
