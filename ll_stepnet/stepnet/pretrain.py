"""
Pre-training models for unsupervised learning on STEP files.
Token prediction tasks (autoregressive and masked) for self-supervised learning.

Uses STEP-aware architecture combining:
- Token sequence modeling (STEPTransformerEncoder)
- Topology/geometry understanding (STEPGraphEncoder)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .encoder import STEPTransformerEncoder, STEPTransformerDecoder, STEPGraphEncoder


class STEPForCausalLM(nn.Module):
    """
    Autoregressive (GPT-style) token prediction for STEP files.
    Predicts next token given previous tokens.

    STEP-aware architecture:
    - Token sequence modeling with causal attention
    - Topology/geometry understanding via graph encoder
    - Fusion of both modalities

    Train on raw STEP files with NO LABELS!
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        max_length: int = 4096,
        dropout: float = 0.1,
        graph_node_dim: int = 128,
        num_graph_layers: int = 3
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.graph_node_dim = graph_node_dim

        # Use STEP-aware transformer decoder (causal attention)
        self.transformer_decoder = STEPTransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # Graph encoder for topology (STEP-specific!)
        self.graph_encoder = STEPGraphEncoder(
            node_dim=graph_node_dim,
            num_layers=num_graph_layers
        )

        # Fusion layer (combine token + graph features)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + graph_node_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output head - predict next token
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        topology_data: Optional[Dict] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len] - tokenized STEP content
            attention_mask: [batch_size, seq_len] - mask for padding
            topology_data: Optional dict with:
                - adjacency_matrix: [num_nodes, num_nodes]
                - node_features: [num_nodes, feature_dim]
            labels: [batch_size, seq_len] - next token targets (optional, for training)

        Returns:
            Dictionary with:
                - logits: [batch_size, seq_len, vocab_size]
                - loss: scalar (if labels provided)
        """
        batch_size, seq_len = input_ids.shape

        # Use STEP-aware transformer decoder (handles embeddings, causal mask, etc.)
        hidden_states = self.transformer_decoder(input_ids, attention_mask)

        # Encode topology if provided (STEP-specific!)
        if topology_data is not None and 'adjacency_matrix' in topology_data:
            adj_matrix = topology_data['adjacency_matrix']
            node_features = topology_data.get('node_features')

            if node_features is None:
                # Create default node features
                num_nodes = adj_matrix.shape[0]
                node_features = torch.randn(num_nodes, self.graph_node_dim, device=input_ids.device)

            # Graph encoding
            graph_features = self.graph_encoder(node_features, adj_matrix)  # [num_nodes, node_dim]

            # Pool graph features
            graph_pooled = graph_features.mean(dim=0)  # [node_dim]

            # Expand to match sequence length
            graph_pooled = graph_pooled.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

            # Fuse token and graph features
            combined = torch.cat([hidden_states, graph_pooled], dim=-1)
            hidden_states = self.fusion(combined)

        # Predict next token
        logits = self.lm_head(hidden_states)

        output = {'logits': logits}

        # Compute loss if labels provided
        if labels is not None:
            # Shift so we predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
            output['loss'] = loss

        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50
    ) -> torch.Tensor:
        """
        Generate STEP tokens autoregressively.

        Args:
            input_ids: [batch_size, seq_len] - prompt tokens
            max_new_tokens: How many tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Only sample from top K tokens

        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Get predictions for next token
            with torch.no_grad():
                outputs = self(input_ids)
                logits = outputs['logits']

                # Get logits for last position
                next_token_logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


class STEPForMaskedLM(nn.Module):
    """
    Masked language modeling (BERT-style) for STEP files.
    Predict masked tokens from context.

    STEP-aware architecture:
    - Token sequence modeling with bidirectional attention (can use STEPTransformerEncoder!)
    - Topology/geometry understanding via graph encoder
    - Fusion of both modalities

    Train on raw STEP files with NO LABELS!
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        max_length: int = 4096,
        dropout: float = 0.1,
        graph_node_dim: int = 128,
        num_graph_layers: int = 3
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.graph_node_dim = graph_node_dim

        # Special tokens
        self.mask_token_id = vocab_size - 1

        # Use STEP-aware transformer encoder (bidirectional)
        self.transformer_encoder = STEPTransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # Graph encoder for topology (STEP-specific!)
        self.graph_encoder = STEPGraphEncoder(
            node_dim=graph_node_dim,
            num_layers=num_graph_layers
        )

        # Fusion layer (combine token + graph features)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + graph_node_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Prediction head
        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        topology_data: Optional[Dict] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len] - tokenized STEP with [MASK] tokens
            attention_mask: [batch_size, seq_len] - mask for padding
            topology_data: Optional dict with:
                - adjacency_matrix: [num_nodes, num_nodes]
                - node_features: [num_nodes, feature_dim]
            labels: [batch_size, seq_len] - original tokens (before masking)

        Returns:
            Dictionary with:
                - logits: [batch_size, seq_len, vocab_size]
                - loss: scalar (if labels provided)
        """
        batch_size, seq_len = input_ids.shape

        # Use STEP-aware transformer encoder (bidirectional)
        hidden_states = self.transformer_encoder(input_ids, attention_mask)  # [batch, seq_len, embed_dim]

        # Encode topology if provided (STEP-specific!)
        if topology_data is not None and 'adjacency_matrix' in topology_data:
            adj_matrix = topology_data['adjacency_matrix']
            node_features = topology_data.get('node_features')

            if node_features is None:
                # Create default node features
                num_nodes = adj_matrix.shape[0]
                node_features = torch.randn(num_nodes, self.graph_node_dim, device=input_ids.device)

            # Graph encoding
            graph_features = self.graph_encoder(node_features, adj_matrix)  # [num_nodes, node_dim]

            # Pool graph features
            graph_pooled = graph_features.mean(dim=0)  # [node_dim]

            # Expand to match sequence length
            graph_pooled = graph_pooled.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

            # Fuse token and graph features
            combined = torch.cat([hidden_states, graph_pooled], dim=-1)
            hidden_states = self.fusion(combined)

        # Predict tokens
        logits = self.mlm_head(hidden_states)

        output = {'logits': logits}

        # Compute loss only on masked tokens
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # -100 = not masked
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
            output['loss'] = loss

        return output


class STEPForHybridLM(nn.Module):
    """
    Hybrid model combining causal and masked prediction.
    Best of both worlds for pre-training.

    Both models use STEP-aware architecture with topology understanding!
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        max_length: int = 4096,
        dropout: float = 0.1,
        graph_node_dim: int = 128,
        num_graph_layers: int = 3
    ):
        super().__init__()

        self.causal_lm = STEPForCausalLM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=max_length,
            dropout=dropout,
            graph_node_dim=graph_node_dim,
            num_graph_layers=num_graph_layers
        )

        self.masked_lm = STEPForMaskedLM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=max_length,
            dropout=dropout,
            graph_node_dim=graph_node_dim,
            num_graph_layers=num_graph_layers
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        topology_data: Optional[Dict] = None,
        labels: Optional[torch.Tensor] = None,
        masked_input_ids: Optional[torch.Tensor] = None,
        masked_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Train both objectives simultaneously with STEP topology awareness.

        Args:
            input_ids: For causal LM
            attention_mask: Attention mask
            topology_data: STEP topology (adjacency + node features)
            labels: Next token labels for causal LM
            masked_input_ids: For masked LM
            masked_labels: Original tokens for masked LM
        """
        outputs = {}
        total_loss = 0.0

        # Causal LM loss
        if input_ids is not None:
            causal_outputs = self.causal_lm(input_ids, attention_mask, topology_data, labels)
            outputs['causal_logits'] = causal_outputs['logits']
            if 'loss' in causal_outputs:
                outputs['causal_loss'] = causal_outputs['loss']
                total_loss += causal_outputs['loss']

        # Masked LM loss
        if masked_input_ids is not None:
            masked_outputs = self.masked_lm(masked_input_ids, attention_mask, topology_data, masked_labels)
            outputs['masked_logits'] = masked_outputs['logits']
            if 'loss' in masked_outputs:
                outputs['masked_loss'] = masked_outputs['loss']
                total_loss += masked_outputs['loss']

        if total_loss > 0:
            outputs['loss'] = total_loss

        return outputs


def mask_tokens(
    input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
    replace_prob: float = 0.1,
    random_prob: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create masked input for BERT-style training.

    Args:
        input_ids: [batch_size, seq_len] - original tokens
        mask_token_id: ID for [MASK] token
        vocab_size: Vocabulary size
        mask_prob: Probability of masking a token
        replace_prob: Probability of replacing with random token instead of [MASK]
        random_prob: Probability of keeping original token

    Returns:
        masked_input: [batch_size, seq_len] - input with masks
        labels: [batch_size, seq_len] - targets (-100 for non-masked)
    """
    labels = input_ids.clone()

    # Create mask probability matrix
    probability_matrix = torch.full(labels.shape, mask_prob)

    # Don't mask padding (token 0) or special tokens
    special_tokens_mask = input_ids <= 3
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Select tokens to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # 80% of the time, replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 10% of the time, replace with random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=input_ids.device)
    input_ids[indices_random] = random_words[indices_random]

    # 10% of the time, keep original token (do nothing)

    return input_ids, labels
