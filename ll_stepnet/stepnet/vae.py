"""
Variational Autoencoder for STEP CAD sequences.

Architecture following DeepCAD: encodes tokenized STEP sequences into a
continuous latent space via reparameterization, then decodes back to
command/parameter predictions. Supports beta-VAE with linear KL warmup.

Encoder path:
    STEPTransformerEncoder -> mean pool -> mu_head / sigma_head -> z
Decoder path:
    z -> project -> positional expand -> STEPTransformerDecoder -> output heads
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import STEPTransformerEncoder, STEPTransformerDecoder

_log = logging.getLogger(__name__)


class STEPVAE(nn.Module):
    """Variational Autoencoder wrapping existing STEP encoder/decoder.

    Follows the DeepCAD architecture: sequences of CAD command tokens are
    encoded into a Gaussian latent, then decoded autoregressively back to
    command-type and parameter predictions.

    Args:
        encoder_config: Configuration object with vocab_size, token_embed_dim,
            num_transformer_layers, dropout, etc.
        latent_dim: Dimensionality of the latent vector z.
        kl_weight: Maximum weight applied to the KL divergence term.
        num_command_types: Number of distinct CAD command types.
        num_param_levels: Number of quantisation levels per parameter.
        max_seq_len: Maximum sequence length the decoder can produce.
    """

    def __init__(
        self,
        encoder_config,
        latent_dim: int = 256,
        kl_weight: float = 1.0,
        num_command_types: int = 6,
        num_param_levels: int = 256,
        max_seq_len: int = 60,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.num_command_types = num_command_types
        self.num_param_levels = num_param_levels
        self.max_seq_len = max_seq_len

        embed_dim = getattr(encoder_config, "token_embed_dim", 256)
        vocab_size = getattr(encoder_config, "vocab_size", 50000)
        # Default to 4 transformer layers per DeepCAD architecture.
        # The encoder_config can override this to any value.
        num_enc_layers = getattr(encoder_config, "num_transformer_layers", 4)
        dropout = getattr(encoder_config, "dropout", 0.1)

        self.embed_dim = embed_dim

        self.encoder = STEPTransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_enc_layers,
            dropout=dropout,
        )

        # Gaussian heads
        self.mu_head = nn.Linear(embed_dim, latent_dim)
        self.log_var_head = nn.Linear(embed_dim, latent_dim)

        # DeepCAD uses matched encoder/decoder depth
        num_dec_layers = getattr(encoder_config, "num_transformer_layers", 4)

        self.decoder = STEPTransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_dec_layers,
            dropout=dropout,
        )

        # Project latent z -> initial decoder hidden
        self.latent_project = nn.Linear(latent_dim, embed_dim)

        # Learned positional embeddings for the decoder sequence
        self.dec_pos_embedding = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim)
        )
        nn.init.normal_(self.dec_pos_embedding, std=0.02)

        # Output heads
        self.command_head = nn.Linear(embed_dim, num_command_types)
        # 16 parameter slots, each quantised into num_param_levels bins
        self.param_heads = nn.ModuleList(
            [nn.Linear(embed_dim, num_param_levels) for _ in range(16)]
        )

        # KL warmup tracking
        self._current_epoch: int = 0
        self._kl_warmup_epochs: int = 10

        _log.info(
            "STEPVAE initialised: latent_dim=%d, embed_dim=%d, "
            "max_seq_len=%d, num_command_types=%d, num_param_levels=%d",
            latent_dim, embed_dim, max_seq_len, num_command_types, num_param_levels,
        )

    @property
    def beta(self) -> float:
        """Current KL weight with linear warmup from 0 to kl_weight."""
        if self._kl_warmup_epochs <= 0:
            return self.kl_weight
        progress = min(self._current_epoch / self._kl_warmup_epochs, 1.0)
        return self.kl_weight * progress

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for KL warmup scheduling.

        Args:
            epoch: Current training epoch (0-indexed).
        """
        self._current_epoch = epoch

    def set_kl_warmup_epochs(self, warmup_epochs: int) -> None:
        """Set the number of epochs over which beta warms up.

        Args:
            warmup_epochs: Number of warmup epochs.
        """
        self._kl_warmup_epochs = warmup_epochs

    def encode(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode token sequence to Gaussian parameters.

        Args:
            token_ids: [batch_size, seq_len] token indices.
            attention_mask: [batch_size, seq_len] 1=real, 0=pad.

        Returns:
            Tuple of (mu, log_var) each [batch_size, latent_dim].
        """
        hidden = self.encoder(token_ids, attention_mask)  # [B, S, D]

        # Mean-pool over sequence (respecting mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(
                dim=1
            ).clamp(min=1e-9)
        else:
            pooled = hidden.mean(dim=1)  # [B, D]

        mu = self.mu_head(pooled)
        log_var = self.log_var_head(pooled)

        return mu, log_var

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * exp(0.5 * log_var).

        Args:
            mu: Mean of the posterior, [batch_size, latent_dim].
            log_var: Log-variance of the posterior, [batch_size, latent_dim].

        Returns:
            Sampled latent z of shape [batch_size, latent_dim].
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(
        self,
        z: torch.Tensor,
        seq_len: Optional[int] = None,
        conditioner: Optional[nn.Module] = None,
        conditioning_inputs: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Decode a latent vector to hidden states.

        Optionally applies text/image conditioning via a conditioner module
        that injects cross-attention. Following Text2CAD, the first N decoder
        blocks can skip cross-attention (controlled by the conditioner's
        skip_cross_attention_blocks parameter).

        Args:
            z: Latent vector [batch_size, latent_dim].
            seq_len: Length of output sequence. Defaults to max_seq_len.
            conditioner: Optional conditioning module (TextConditioner,
                ImageConditioner, or MultiModalConditioner).
            conditioning_inputs: Dict with conditioning data for the conditioner:
                - text_input_ids: [B, L] text token ids
                - text_attention_mask: [B, L] text padding mask
                - pixel_values: [B, C, H, W] image tensors

        Returns:
            Hidden states [batch_size, seq_len, embed_dim].
        """
        if seq_len is None:
            seq_len = self.max_seq_len

        batch_size = z.shape[0]

        # Project z -> embed_dim and expand to full sequence
        z_proj = self.latent_project(z)  # [B, embed_dim]
        z_expanded = z_proj.unsqueeze(1).expand(
            batch_size, seq_len, -1
        )  # [B, S, D]

        # Add learned positional embeddings
        hidden = z_expanded + self.dec_pos_embedding[:, :seq_len, :]

        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=z.device
        )
        hidden = self.decoder.dropout(hidden)

        # Process through decoder layers with optional per-layer conditioning
        if conditioner is not None and conditioning_inputs is not None:
            # Apply conditioning with block_index for skip logic (Text2CAD)
            hidden = self._decode_with_conditioning(
                hidden, causal_mask, conditioner, conditioning_inputs
            )
        else:
            # Standard decoder forward (self-attention only)
            hidden = self.decoder.transformer(
                hidden, memory=hidden, tgt_mask=causal_mask
            )

        hidden = self.decoder.layer_norm(hidden)
        return hidden  # [B, S, embed_dim]

    def _decode_with_conditioning(
        self,
        hidden: torch.Tensor,
        causal_mask: torch.Tensor,
        conditioner: nn.Module,
        conditioning_inputs: Dict,
    ) -> torch.Tensor:
        """Decode with per-layer conditioning injection.

        Applies the conditioner after each decoder layer, passing the
        block_index so the conditioner can skip cross-attention for
        early blocks (Text2CAD: first 2 blocks skip cross-attention).

        Args:
            hidden: Hidden states [B, S, D].
            causal_mask: Causal attention mask [S, S].
            conditioner: Conditioner module.
            conditioning_inputs: Dict with text_input_ids, text_attention_mask,
                and/or pixel_values.

        Returns:
            Updated hidden states [B, S, D].
        """
        # Access individual decoder layers
        for block_index, layer in enumerate(self.decoder.transformer.layers):
            # Standard decoder layer forward (self-attn + cross-attn + FFN)
            hidden = layer(hidden, memory=hidden, tgt_mask=causal_mask)

            # Apply conditioner with block_index for skip logic
            # The conditioner will skip cross-attention if block_index < skip_blocks
            hidden = conditioner(
                hidden_states=hidden,
                block_index=block_index,
                **conditioning_inputs,
            )

        return hidden

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        command_targets: Optional[torch.Tensor] = None,
        param_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode, reparameterize, decode, compute losses.

        Args:
            token_ids: [batch_size, seq_len] input token ids.
            attention_mask: [batch_size, seq_len] padding mask.
            command_targets: [batch_size, seq_len] ground-truth command types.
            param_targets: [batch_size, seq_len, 16] ground-truth params.

        Returns:
            Dictionary with z, mu, log_var, command_logits, param_logits,
            kl_loss, and optionally recon_loss and loss.
        """
        seq_len = token_ids.shape[1]

        # Encode
        mu, log_var = self.encode(token_ids, attention_mask)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        hidden = self.decode(z, seq_len=seq_len)

        # Output heads
        command_logits = self.command_head(hidden)  # [B, S, C]
        param_logits = [head(hidden) for head in self.param_heads]  # 16 x [B, S, P]

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        )

        outputs: Dict[str, torch.Tensor] = {
            "z": z,
            "mu": mu,
            "log_var": log_var,
            "command_logits": command_logits,
            "param_logits": param_logits,
            "kl_loss": kl_loss,
        }

        # Reconstruction loss (if targets provided)
        if command_targets is not None:
            cmd_loss = F.cross_entropy(
                command_logits.reshape(-1, self.num_command_types),
                command_targets.reshape(-1),
                ignore_index=-1,
            )
            recon_loss = cmd_loss

            if param_targets is not None:
                param_loss = torch.tensor(0.0, device=token_ids.device)
                for i, head_logits in enumerate(param_logits):
                    p_target = param_targets[..., i].reshape(-1)
                    p_logits = head_logits.reshape(-1, self.num_param_levels)
                    param_loss = param_loss + F.cross_entropy(
                        p_logits, p_target, ignore_index=-1
                    )
                recon_loss = recon_loss + param_loss / len(param_logits)

            outputs["recon_loss"] = recon_loss
            outputs["loss"] = recon_loss + self.beta * kl_loss

        return outputs

    def sample(
        self,
        num_samples: int = 1,
        seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        conditioner: Optional[nn.Module] = None,
        conditioning_inputs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample new CAD sequences from the prior N(0, I).

        Optionally applies text/image conditioning via a conditioner module.
        Following Text2CAD, the first N decoder blocks skip cross-attention
        to allow initial CAD structure formation before conditioning kicks in.

        Args:
            num_samples: Number of sequences to generate.
            seq_len: Output sequence length. Defaults to max_seq_len.
            device: Target device for the generated tensors.
            conditioner: Optional conditioning module (TextConditioner,
                ImageConditioner, or MultiModalConditioner).
            conditioning_inputs: Dict with conditioning data:
                - text_input_ids: [B, L] text token ids
                - text_attention_mask: [B, L] text padding mask
                - pixel_values: [B, C, H, W] image tensors

        Returns:
            Dictionary with command_preds [N, S] and param_preds [N, S, 16].
        """
        if device is None:
            device = next(self.parameters()).device
        if seq_len is None:
            seq_len = self.max_seq_len

        z = torch.randn(num_samples, self.latent_dim, device=device)
        hidden = self.decode(
            z,
            seq_len=seq_len,
            conditioner=conditioner,
            conditioning_inputs=conditioning_inputs,
        )

        command_logits = self.command_head(hidden)
        command_preds = command_logits.argmax(dim=-1)  # [N, S]

        param_preds = torch.stack(
            [head(hidden).argmax(dim=-1) for head in self.param_heads], dim=-1
        )  # [N, S, 16]

        return {"command_preds": command_preds, "param_preds": param_preds}
