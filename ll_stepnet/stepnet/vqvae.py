"""
VQ-VAE and Codebook Module for CAD Generation.

Implements SkexGen's disentangled codebook approach for CAD generation,
separating topology (curve types), geometry (2D positions), and extrusion
(3D operations) into three independent codebook streams.

Architecture follows:
    Input features -> Encoder MLP -> 3 codebook streams -> VQ -> Decoder MLP -> Reconstruction

Each codebook uses exponential moving average (EMA) updates with a warmup
period where quantization is bypassed to stabilize early training.

References:
    - SkexGen: Autoregressive Generation of CAD Construction Sequences
      with Disentangled Codebooks (Xu et al., 2022)
    - Neural Discrete Representation Learning (van den Oord et al., 2017)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)


class VectorQuantizer(nn.Module):
    """Core vector quantization layer with EMA codebook updates.

    Maps continuous latent vectors to the nearest entry in a learned
    codebook embedding table.  During training the codebook is updated
    via exponential moving average (EMA) rather than straight gradient
    descent, which is more stable for VQ-VAE training.

    A warmup period (SkexGen stabilisation trick) bypasses quantization
    for the first ``warmup_epochs`` epochs so the encoder can learn a
    reasonable latent distribution before the codebook locks in.

    Args:
        num_embeddings: Number of codebook entries (K).
        embedding_dim: Dimensionality of each codebook vector.
        commitment_cost: Weight beta for the commitment loss term.
        decay: EMA decay factor for codebook updates.
        warmup_epochs: Number of initial training epochs to skip
            quantization (pass-through mode).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        warmup_epochs: int = 25,
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.warmup_epochs = warmup_epochs

        # Codebook embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

        # EMA tracking buffers (not model parameters)
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer(
            "_ema_embedding_sum", self.embedding.weight.data.clone()
        )

        # Epoch counter for warmup gating
        self.register_buffer(
            "_training_epoch", torch.tensor(0, dtype=torch.long)
        )

        _log.debug(
            "VectorQuantizer initialised: K=%d, D=%d, beta=%.3f, "
            "decay=%.3f, warmup=%d",
            num_embeddings,
            embedding_dim,
            commitment_cost,
            decay,
            warmup_epochs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def training_epoch(self) -> int:
        """Current training epoch as tracked by ``set_epoch``."""
        return int(self._training_epoch.item())

    def set_epoch(self, epoch: int) -> None:
        """Update the internal epoch counter (used for warmup gating).

        Args:
            epoch: Current training epoch (0-indexed).
        """
        self._training_epoch.fill_(epoch)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize continuous input vectors to nearest codebook entries.

        During warmup (``epoch < warmup_epochs``) quantization is skipped
        and the inputs are returned as-is with zero commitment loss so
        the encoder can pre-train freely.

        Args:
            inputs: Continuous latent vectors of shape
                ``(batch, *, embedding_dim)`` where ``*`` is any number
                of intermediate dimensions (typically sequence length).

        Returns:
            A tuple of ``(quantized, commitment_loss, encoding_indices)``
            where:

            - **quantized** has the same shape as *inputs* and contains
              the selected codebook vectors (with straight-through
              gradient).
            - **commitment_loss** is a scalar
              ``beta * ||z_e - sg(z_q)||^2``.
            - **encoding_indices** is a ``LongTensor`` of shape
              ``(batch * seq_len,)`` giving the selected codebook index
              for each input vector.
        """
        # Flatten all leading dims except embedding_dim
        input_shape = inputs.shape
        flat_inputs = inputs.reshape(-1, self.embedding_dim)  # (N, D)

        # ----------------------------------------------------------
        # Warmup bypass: skip quantization entirely
        # ----------------------------------------------------------
        if self.training and self.training_epoch < self.warmup_epochs:
            # Return inputs unchanged; no commitment loss; dummy indices
            zeros_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
            dummy_indices = torch.zeros(
                flat_inputs.shape[0], dtype=torch.long, device=inputs.device
            )
            _log.debug(
                "Warmup epoch %d/%d -- quantization bypassed",
                self.training_epoch,
                self.warmup_epochs,
            )
            return inputs, zeros_loss, dummy_indices

        # ----------------------------------------------------------
        # Compute distances: ||z_e - e_j||^2
        # ----------------------------------------------------------
        # Expand: (N, 1, D) - (1, K, D) -> (N, K)
        distances = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2.0 * torch.matmul(flat_inputs, self.embedding.weight.t())
        )

        # Nearest codebook entry for each input vector
        encoding_indices = torch.argmin(distances, dim=1)  # (N,)

        # Look up the quantized vectors
        quantized_flat = self.embedding(encoding_indices)  # (N, D)
        quantized = quantized_flat.reshape(input_shape)

        # ----------------------------------------------------------
        # EMA codebook update (training only)
        # ----------------------------------------------------------
        if self.training:
            self._update_codebook_ema(flat_inputs, encoding_indices)

        # ----------------------------------------------------------
        # Losses
        # ----------------------------------------------------------
        # Commitment loss: encourages encoder outputs to stay close
        # to the codebook entries (stop gradient on quantized).
        e_latent_loss = F.mse_loss(inputs, quantized.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        # ----------------------------------------------------------
        # Straight-through estimator: copy gradients from quantized
        # to inputs, since argmin is non-differentiable.
        # ----------------------------------------------------------
        quantized = inputs + (quantized - inputs).detach()

        return quantized, commitment_loss, encoding_indices

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_codebook_ema(
        self,
        flat_inputs: torch.Tensor,
        encoding_indices: torch.Tensor,
    ) -> None:
        """Update codebook embeddings via exponential moving average.

        Args:
            flat_inputs: Flattened encoder outputs ``(N, D)``.
            encoding_indices: Selected codebook indices ``(N,)``.
        """
        # One-hot assignments: (N, K)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Cluster sizes: how many inputs map to each codebook entry
        new_cluster_size = encodings.sum(dim=0)  # (K,)

        # Sum of inputs assigned to each entry
        new_embedding_sum = encodings.t() @ flat_inputs  # (K, D)

        # EMA update
        self._ema_cluster_size.mul_(self.decay).add_(
            new_cluster_size, alpha=1.0 - self.decay
        )
        self._ema_embedding_sum.mul_(self.decay).add_(
            new_embedding_sum, alpha=1.0 - self.decay
        )

        # Laplace smoothing to avoid zero-count entries
        n = self._ema_cluster_size.sum()
        cluster_size_smoothed = (
            (self._ema_cluster_size + 1e-5)
            / (n + self.num_embeddings * 1e-5)
            * n
        )

        # Normalise to get updated embeddings
        self.embedding.weight.data.copy_(
            self._ema_embedding_sum / cluster_size_smoothed.unsqueeze(1)
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def codebook_utilization(self) -> float:
        """Fraction of codebook entries actively used (non-zero EMA count).

        Returns:
            A float in ``[0.0, 1.0]`` representing the proportion of
            codebook entries that have received at least one assignment.
        """
        active = (self._ema_cluster_size > 1e-7).sum().item()
        return active / self.num_embeddings


class DisentangledCodebooks(nn.Module):
    """SkexGen's three-codebook disentangled quantization system.

    Maintains separate codebooks for three aspects of a CAD model:

    - **Topology codebook**: encodes curve type sequences (e.g. line,
      arc, spline ordering in a sketch profile).
    - **Geometry codebook**: encodes 2D point positions on a
      64x64 quantized grid.
    - **Extrusion codebook**: encodes 3D extrusion operations
      (direction, depth, taper, boolean type).

    Each input feature stream is projected to ``code_dim``, quantized
    through its respective codebook, then decoded back.  The model
    produces 10 total codes per CAD model split across the three
    codebooks (3 topology + 4 geometry + 3 extrusion by default).

    Args:
        topology_codes: Number of entries in the topology codebook.
        geometry_codes: Number of entries in the geometry codebook.
        extrusion_codes: Number of entries in the extrusion codebook.
        code_dim: Dimensionality shared by all codebook vectors.
    """

    # Default split: how many codes each stream produces per model
    TOPOLOGY_NUM_CODES = 3
    GEOMETRY_NUM_CODES = 4
    EXTRUSION_NUM_CODES = 3
    TOTAL_CODES = TOPOLOGY_NUM_CODES + GEOMETRY_NUM_CODES + EXTRUSION_NUM_CODES

    def __init__(
        self,
        topology_codes: int = 500,
        geometry_codes: int = 1000,
        extrusion_codes: int = 1000,
        code_dim: int = 256,
    ) -> None:
        super().__init__()

        self.code_dim = code_dim

        # Three independent codebooks
        self.topology_codebook = VectorQuantizer(
            num_embeddings=topology_codes,
            embedding_dim=code_dim,
        )
        self.geometry_codebook = VectorQuantizer(
            num_embeddings=geometry_codes,
            embedding_dim=code_dim,
        )
        self.extrusion_codebook = VectorQuantizer(
            num_embeddings=extrusion_codes,
            embedding_dim=code_dim,
        )

        # Input projections: map raw features to code_dim sequences
        self.topology_proj = nn.Sequential(
            nn.Linear(code_dim, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, code_dim * self.TOPOLOGY_NUM_CODES),
        )
        self.geometry_proj = nn.Sequential(
            nn.Linear(code_dim, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, code_dim * self.GEOMETRY_NUM_CODES),
        )
        self.extrusion_proj = nn.Sequential(
            nn.Linear(code_dim, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, code_dim * self.EXTRUSION_NUM_CODES),
        )

        # Decode projections: map quantized codes back to feature space
        self.topology_decode = nn.Sequential(
            nn.Linear(code_dim * self.TOPOLOGY_NUM_CODES, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, code_dim),
        )
        self.geometry_decode = nn.Sequential(
            nn.Linear(code_dim * self.GEOMETRY_NUM_CODES, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, code_dim),
        )
        self.extrusion_decode = nn.Sequential(
            nn.Linear(code_dim * self.EXTRUSION_NUM_CODES, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, code_dim),
        )

        # Accumulated losses from the most recent forward pass
        self._last_commitment_losses: Dict[str, torch.Tensor] = {}

        _log.debug(
            "DisentangledCodebooks initialised: topology=%d, geometry=%d, "
            "extrusion=%d, code_dim=%d",
            topology_codes,
            geometry_codes,
            extrusion_codes,
            code_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch counter to all child codebooks.

        Args:
            epoch: Current training epoch (0-indexed).
        """
        self.topology_codebook.set_epoch(epoch)
        self.geometry_codebook.set_epoch(epoch)
        self.extrusion_codebook.set_epoch(epoch)

    @property
    def total_commitment_loss(self) -> torch.Tensor:
        """Sum of commitment losses from the most recent ``encode`` call.

        Returns:
            Scalar tensor aggregating topology + geometry + extrusion
            commitment losses.  Returns ``0.0`` if ``encode`` has not
            been called yet.
        """
        if not self._last_commitment_losses:
            return torch.tensor(0.0)
        return sum(self._last_commitment_losses.values())

    def encode(
        self,
        sketch_features: torch.Tensor,
        geometry_features: torch.Tensor,
        extrusion_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode feature streams into discrete codebook indices.

        Each feature tensor is projected, reshaped into a short code
        sequence, then quantized through the corresponding codebook.

        Args:
            sketch_features: Topology/sketch features of shape
                ``(batch, code_dim)``.
            geometry_features: Geometry features of shape
                ``(batch, code_dim)``.
            extrusion_features: Extrusion features of shape
                ``(batch, code_dim)``.

        Returns:
            A tuple of ``(topology_codes, geometry_codes, extrusion_codes)``
            where each is a ``LongTensor`` of codebook indices with shape
            ``(batch, num_codes_for_that_stream)``.
        """
        batch_size = sketch_features.shape[0]

        # --- Topology stream ---
        topo_proj = self.topology_proj(sketch_features)  # (B, D*3)
        topo_seq = topo_proj.reshape(
            batch_size, self.TOPOLOGY_NUM_CODES, self.code_dim
        )
        topo_quantized, topo_loss, topo_indices = self.topology_codebook(topo_seq)
        topo_indices = topo_indices.reshape(batch_size, self.TOPOLOGY_NUM_CODES)

        # --- Geometry stream ---
        geom_proj = self.geometry_proj(geometry_features)  # (B, D*4)
        geom_seq = geom_proj.reshape(
            batch_size, self.GEOMETRY_NUM_CODES, self.code_dim
        )
        geom_quantized, geom_loss, geom_indices = self.geometry_codebook(geom_seq)
        geom_indices = geom_indices.reshape(batch_size, self.GEOMETRY_NUM_CODES)

        # --- Extrusion stream ---
        extr_proj = self.extrusion_proj(extrusion_features)  # (B, D*3)
        extr_seq = extr_proj.reshape(
            batch_size, self.EXTRUSION_NUM_CODES, self.code_dim
        )
        extr_quantized, extr_loss, extr_indices = self.extrusion_codebook(extr_seq)
        extr_indices = extr_indices.reshape(batch_size, self.EXTRUSION_NUM_CODES)

        # Cache quantized outputs for decode and commitment losses
        self._last_topo_quantized = topo_quantized
        self._last_geom_quantized = geom_quantized
        self._last_extr_quantized = extr_quantized

        self._last_commitment_losses = {
            "topology": topo_loss,
            "geometry": geom_loss,
            "extrusion": extr_loss,
        }

        return topo_indices, geom_indices, extr_indices

    def decode(
        self,
        topology_codes: torch.Tensor,
        geometry_codes: torch.Tensor,
        extrusion_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode codebook indices back to reconstructed feature vectors.

        Args:
            topology_codes: ``(batch, TOPOLOGY_NUM_CODES)`` LongTensor
                of topology codebook indices.
            geometry_codes: ``(batch, GEOMETRY_NUM_CODES)`` LongTensor
                of geometry codebook indices.
            extrusion_codes: ``(batch, EXTRUSION_NUM_CODES)`` LongTensor
                of extrusion codebook indices.

        Returns:
            A tuple of ``(topo_features, geom_features, extr_features)``
            each of shape ``(batch, code_dim)``.
        """
        batch_size = topology_codes.shape[0]

        # Look up codebook embeddings
        topo_emb = self.topology_codebook.embedding(topology_codes)  # (B, 3, D)
        geom_emb = self.geometry_codebook.embedding(geometry_codes)  # (B, 4, D)
        extr_emb = self.extrusion_codebook.embedding(extrusion_codes)  # (B, 3, D)

        # Flatten code sequences and decode
        topo_flat = topo_emb.reshape(batch_size, -1)  # (B, 3*D)
        geom_flat = geom_emb.reshape(batch_size, -1)  # (B, 4*D)
        extr_flat = extr_emb.reshape(batch_size, -1)  # (B, 3*D)

        topo_features = self.topology_decode(topo_flat)    # (B, D)
        geom_features = self.geometry_decode(geom_flat)    # (B, D)
        extr_features = self.extrusion_decode(extr_flat)   # (B, D)

        return topo_features, geom_features, extr_features

    def decode_quantized(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode using the cached quantized outputs from the last ``encode``.

        This uses the straight-through quantized tensors (with gradients)
        from the most recent ``encode`` call, which is needed during
        training to allow gradient flow through the VQ bottleneck.

        Returns:
            A tuple of ``(topo_features, geom_features, extr_features)``
            each of shape ``(batch, code_dim)``.

        Raises:
            RuntimeError: If called before ``encode``.
        """
        if not hasattr(self, "_last_topo_quantized"):
            raise RuntimeError(
                "decode_quantized() called before encode(). "
                "Call encode() first to cache quantized outputs."
            )

        batch_size = self._last_topo_quantized.shape[0]

        topo_flat = self._last_topo_quantized.reshape(batch_size, -1)
        geom_flat = self._last_geom_quantized.reshape(batch_size, -1)
        extr_flat = self._last_extr_quantized.reshape(batch_size, -1)

        topo_features = self.topology_decode(topo_flat)
        geom_features = self.geometry_decode(geom_flat)
        extr_features = self.extrusion_decode(extr_flat)

        return topo_features, geom_features, extr_features


class CodebookDecoder(nn.Module):
    """Autoregressive transformer decoder for generating codebook indices.

    Given a sequence of codebook indices, this module predicts the next
    index autoregressively.  One ``CodebookDecoder`` is instantiated per
    codebook stream (topology, geometry, extrusion) so that each stream
    can be generated independently.

    The architecture is a standard GPT-style transformer decoder with
    causal masking, learned positional embeddings, and a final linear
    head projecting to the codebook vocabulary.

    Args:
        code_dim: Hidden dimension of the transformer.
        num_layers: Number of transformer decoder layers.
        num_heads: Number of attention heads.
        vocab_size: Number of codebook entries this decoder predicts
            over (must match the corresponding ``VectorQuantizer``
            ``num_embeddings``).
        max_codes: Maximum sequence length of codes to generate.
        dropout: Dropout rate applied throughout the transformer.
    """

    def __init__(
        self,
        code_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        vocab_size: int = 500,
        max_codes: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.code_dim = code_dim
        self.vocab_size = vocab_size
        self.max_codes = max_codes

        # Embedding for codebook indices + BOS token
        # vocab_size + 1 to include a special BOS token at index vocab_size
        self.bos_token_id = vocab_size
        self.token_embedding = nn.Embedding(vocab_size + 1, code_dim)

        # Learned positional embeddings (max_codes + 1 for BOS prefix)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_codes + 1, code_dim)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Transformer decoder layers (self-attention only, causal mask)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=code_dim,
            nhead=num_heads,
            dim_feedforward=code_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(code_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # Output head: project to codebook vocabulary logits
        self.output_head = nn.Linear(code_dim, vocab_size)

        _log.debug(
            "CodebookDecoder initialised: code_dim=%d, layers=%d, "
            "heads=%d, vocab=%d, max_codes=%d",
            code_dim,
            num_layers,
            num_heads,
            vocab_size,
            max_codes,
        )

    def forward(
        self,
        codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute next-code logits for a sequence of codebook indices.

        Args:
            codes: ``(batch, seq_len)`` LongTensor of codebook indices.
                Should be prepended with BOS token for teacher-forced
                training.
            mask: Optional ``(batch, seq_len)`` padding mask where
                ``True``/``1`` indicates valid positions and
                ``False``/``0`` indicates padding.

        Returns:
            Logits tensor of shape ``(batch, seq_len, vocab_size)``
            giving the predicted distribution over the next codebook
            index at each position.
        """
        batch_size, seq_len = codes.shape

        # Embed tokens
        x = self.token_embedding(codes)  # (B, S, D)

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout_layer(x)

        # Causal mask: prevent attending to future tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=codes.device
        )

        # Padding mask (transformer expects True = ignore)
        padding_mask = None
        if mask is not None:
            padding_mask = (mask == 0)

        # Transformer forward (self-attention with causal mask)
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        x = self.layer_norm(x)

        # Project to vocab logits
        logits = self.output_head(x)  # (B, S, vocab_size)

        return logits

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        max_codes: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate code sequences autoregressively.

        Starts from a BOS token and samples one code at a time,
        feeding each sampled code back as input for the next step.

        Args:
            num_samples: Batch size of sequences to generate in
                parallel.
            max_codes: Maximum number of codes to generate per
                sequence.  Defaults to ``self.max_codes``.
            temperature: Sampling temperature.  Higher values produce
                more diverse outputs.
            top_k: If set, only sample from the top-k highest
                probability entries at each step.

        Returns:
            ``(num_samples, max_codes)`` LongTensor of generated
            codebook indices.
        """
        if max_codes is None:
            max_codes = self.max_codes

        device = self.token_embedding.weight.device
        self.eval()

        # Start with BOS token
        generated = torch.full(
            (num_samples, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=device,
        )

        for step in range(max_codes):
            # Forward pass on current sequence
            logits = self.forward(generated)  # (B, current_len, vocab)

            # Get logits for the last position
            next_logits = logits[:, -1, :] / temperature  # (B, vocab)

            # Top-k filtering
            if top_k is not None and top_k > 0:
                top_k_clamped = min(top_k, self.vocab_size)
                topk_values, _ = torch.topk(next_logits, top_k_clamped, dim=-1)
                threshold = topk_values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(
                    next_logits < threshold, float("-inf")
                )

            # Sample from distribution
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            generated = torch.cat([generated, next_token], dim=1)

        # Remove BOS token, keep only generated codes
        generated = generated[:, 1:]  # (B, max_codes)

        return generated


class VQVAEModel(nn.Module):
    """Complete VQ-VAE model for CAD generation.

    Combines an encoder MLP, the :class:`DisentangledCodebooks` vector
    quantization layer, and a decoder MLP into an end-to-end model that
    can:

    1. **Encode** continuous CAD feature vectors into compact discrete
       code sequences (10 codes per model, split 3/4/3 across topology,
       geometry, and extrusion codebooks).
    2. **Decode** discrete codes back to reconstructed feature vectors.
    3. **Train** the full pipeline with reconstruction loss +
       commitment loss.

    The encoder splits its output into three equal-sized chunks that are
    fed to the topology, geometry, and extrusion codebook streams
    respectively.  The decoder concatenates the three decoded streams
    and projects back to the original input dimensionality.

    Args:
        input_dim: Dimensionality of the input feature vector (e.g.
            flattened STEP entity features).
        code_dim: Internal dimensionality for codebook vectors.
        topology_codes: Number of topology codebook entries.
        geometry_codes: Number of geometry codebook entries.
        extrusion_codes: Number of extrusion codebook entries.
        encoder_hidden_dim: Hidden dimension of the encoder MLP.
        decoder_hidden_dim: Hidden dimension of the decoder MLP.
    """

    def __init__(
        self,
        input_dim: int,
        code_dim: int = 256,
        topology_codes: int = 500,
        geometry_codes: int = 1000,
        extrusion_codes: int = 1000,
        encoder_hidden_dim: int = 512,
        decoder_hidden_dim: int = 512,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.code_dim = code_dim

        # Encoder: input_dim -> 512 -> code_dim * 3
        # Output is split into three streams for the three codebooks
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(encoder_hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(encoder_hidden_dim, code_dim * 3),
        )

        # Disentangled codebooks
        self.codebooks = DisentangledCodebooks(
            topology_codes=topology_codes,
            geometry_codes=geometry_codes,
            extrusion_codes=extrusion_codes,
            code_dim=code_dim,
        )

        # Decoder: code_dim * 3 -> 512 -> input_dim
        # Reconstructs original features from concatenated decoded streams
        self.decoder = nn.Sequential(
            nn.Linear(code_dim * 3, decoder_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(decoder_hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(decoder_hidden_dim, input_dim),
        )

        # Autoregressive decoders for each codebook stream (generation)
        self.topology_ar_decoder = CodebookDecoder(
            code_dim=code_dim,
            num_layers=4,
            num_heads=8,
            vocab_size=topology_codes,
            max_codes=DisentangledCodebooks.TOPOLOGY_NUM_CODES,
        )
        self.geometry_ar_decoder = CodebookDecoder(
            code_dim=code_dim,
            num_layers=4,
            num_heads=8,
            vocab_size=geometry_codes,
            max_codes=DisentangledCodebooks.GEOMETRY_NUM_CODES,
        )
        self.extrusion_ar_decoder = CodebookDecoder(
            code_dim=code_dim,
            num_layers=4,
            num_heads=8,
            vocab_size=extrusion_codes,
            max_codes=DisentangledCodebooks.EXTRUSION_NUM_CODES,
        )

        _log.info(
            "VQVAEModel initialised: input_dim=%d, code_dim=%d, "
            "topology_K=%d, geometry_K=%d, extrusion_K=%d",
            input_dim,
            code_dim,
            topology_codes,
            geometry_codes,
            extrusion_codes,
        )

    # ------------------------------------------------------------------
    # Epoch management
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to all codebooks for warmup tracking.

        Args:
            epoch: Current training epoch (0-indexed).
        """
        self.codebooks.set_epoch(epoch)

    # ------------------------------------------------------------------
    # Forward pass (training)
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode -> quantize -> decode.

        Args:
            x: Input features of shape ``(batch, input_dim)``.

        Returns:
            Dictionary containing:

            - ``"reconstructed"``: Reconstructed features
              ``(batch, input_dim)``.
            - ``"commitment_loss"``: Scalar commitment loss.
            - ``"codes"``: Dictionary with ``"topology"``,
              ``"geometry"``, and ``"extrusion"`` index tensors.
            - ``"reconstruction_loss"``: MSE reconstruction loss.
        """
        batch_size = x.shape[0]

        # Encode to latent space
        encoded = self.encoder(x)  # (B, code_dim * 3)

        # Split into three streams
        sketch_features = encoded[:, : self.code_dim]
        geometry_features = encoded[:, self.code_dim : self.code_dim * 2]
        extrusion_features = encoded[:, self.code_dim * 2 :]

        # Quantize through disentangled codebooks
        topo_indices, geom_indices, extr_indices = self.codebooks.encode(
            sketch_features, geometry_features, extrusion_features
        )

        # Decode using straight-through quantized tensors (gradient flow)
        topo_dec, geom_dec, extr_dec = self.codebooks.decode_quantized()

        # Concatenate decoded streams
        combined = torch.cat([topo_dec, geom_dec, extr_dec], dim=-1)  # (B, D*3)

        # Reconstruct original features
        reconstructed = self.decoder(combined)  # (B, input_dim)

        # Compute reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed, x)

        # Get commitment loss from codebooks
        commitment_loss = self.codebooks.total_commitment_loss

        return {
            "reconstructed": reconstructed,
            "commitment_loss": commitment_loss,
            "reconstruction_loss": reconstruction_loss,
            "codes": {
                "topology": topo_indices,
                "geometry": geom_indices,
                "extrusion": extr_indices,
            },
        }

    # ------------------------------------------------------------------
    # Inference: encode to codes / decode from codes
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_to_codes(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Encode input features to compact discrete codes.

        Produces 10 codes per model (3 topology + 4 geometry + 3
        extrusion), which serve as a compact discrete representation
        of the CAD model.

        Args:
            x: Input features of shape ``(batch, input_dim)``.

        Returns:
            Dictionary with ``"topology"``, ``"geometry"``, and
            ``"extrusion"`` keys mapping to ``LongTensor`` codebook
            indices.
        """
        self.eval()

        encoded = self.encoder(x)

        sketch_features = encoded[:, : self.code_dim]
        geometry_features = encoded[:, self.code_dim : self.code_dim * 2]
        extrusion_features = encoded[:, self.code_dim * 2 :]

        topo_indices, geom_indices, extr_indices = self.codebooks.encode(
            sketch_features, geometry_features, extrusion_features
        )

        return {
            "topology": topo_indices,
            "geometry": geom_indices,
            "extrusion": extr_indices,
        }

    @torch.no_grad()
    def decode_from_codes(
        self,
        topology_codes: torch.Tensor,
        geometry_codes: torch.Tensor,
        extrusion_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Decode discrete codebook indices back to reconstructed features.

        Args:
            topology_codes: ``(batch, 3)`` topology codebook indices.
            geometry_codes: ``(batch, 4)`` geometry codebook indices.
            extrusion_codes: ``(batch, 3)`` extrusion codebook indices.

        Returns:
            Reconstructed features of shape ``(batch, input_dim)``.
        """
        self.eval()

        # Decode through codebooks
        topo_dec, geom_dec, extr_dec = self.codebooks.decode(
            topology_codes, geometry_codes, extrusion_codes
        )

        # Concatenate and reconstruct
        combined = torch.cat([topo_dec, geom_dec, extr_dec], dim=-1)
        reconstructed = self.decoder(combined)

        return reconstructed

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate new CAD models by sampling codes autoregressively.

        Uses the three independent ``CodebookDecoder`` instances to
        generate topology, geometry, and extrusion code sequences in
        parallel, then decodes them back to feature space.

        Args:
            num_samples: Number of CAD models to generate.
            temperature: Sampling temperature for all three decoders.
            top_k: Top-k filtering for sampling.

        Returns:
            Dictionary containing:

            - ``"reconstructed"``: Generated features
              ``(num_samples, input_dim)``.
            - ``"codes"``: Dictionary with ``"topology"``,
              ``"geometry"``, and ``"extrusion"`` index tensors.
        """
        self.eval()

        # Generate codes from each autoregressive decoder
        topo_codes = self.topology_ar_decoder.sample(
            num_samples=num_samples,
            max_codes=DisentangledCodebooks.TOPOLOGY_NUM_CODES,
            temperature=temperature,
            top_k=top_k,
        )
        geom_codes = self.geometry_ar_decoder.sample(
            num_samples=num_samples,
            max_codes=DisentangledCodebooks.GEOMETRY_NUM_CODES,
            temperature=temperature,
            top_k=top_k,
        )
        extr_codes = self.extrusion_ar_decoder.sample(
            num_samples=num_samples,
            max_codes=DisentangledCodebooks.EXTRUSION_NUM_CODES,
            temperature=temperature,
            top_k=top_k,
        )

        # Clamp to valid codebook range
        topo_codes = topo_codes.clamp(
            0, self.codebooks.topology_codebook.num_embeddings - 1
        )
        geom_codes = geom_codes.clamp(
            0, self.codebooks.geometry_codebook.num_embeddings - 1
        )
        extr_codes = extr_codes.clamp(
            0, self.codebooks.extrusion_codebook.num_embeddings - 1
        )

        # Decode to features
        reconstructed = self.decode_from_codes(topo_codes, geom_codes, extr_codes)

        return {
            "reconstructed": reconstructed,
            "codes": {
                "topology": topo_codes,
                "geometry": geom_codes,
                "extrusion": extr_codes,
            },
        }

    # ------------------------------------------------------------------
    # Autoregressive training
    # ------------------------------------------------------------------

    def compute_ar_loss(
        self,
        topology_codes: torch.Tensor,
        geometry_codes: torch.Tensor,
        extrusion_codes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute autoregressive next-code prediction losses.

        Used for training the ``CodebookDecoder`` modules.  Takes
        ground-truth code sequences (from ``encode_to_codes``) and
        computes cross-entropy loss for each decoder.

        Args:
            topology_codes: ``(batch, 3)`` ground-truth topology indices.
            geometry_codes: ``(batch, 4)`` ground-truth geometry indices.
            extrusion_codes: ``(batch, 3)`` ground-truth extrusion indices.

        Returns:
            Dictionary with ``"topology_ar_loss"``,
            ``"geometry_ar_loss"``, ``"extrusion_ar_loss"``, and
            ``"total_ar_loss"`` scalar tensors.
        """
        losses = {}

        # For each stream: prepend BOS, predict next token
        for name, decoder, codes in [
            ("topology", self.topology_ar_decoder, topology_codes),
            ("geometry", self.geometry_ar_decoder, geometry_codes),
            ("extrusion", self.extrusion_ar_decoder, extrusion_codes),
        ]:
            batch_size, seq_len = codes.shape

            # Prepend BOS token
            bos = torch.full(
                (batch_size, 1),
                decoder.bos_token_id,
                dtype=torch.long,
                device=codes.device,
            )
            input_codes = torch.cat([bos, codes[:, :-1]], dim=1)  # (B, S)

            # Forward through decoder
            logits = decoder(input_codes)  # (B, S, vocab_size)

            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, decoder.vocab_size),
                codes.reshape(-1),
            )
            losses[f"{name}_ar_loss"] = loss

        losses["total_ar_loss"] = sum(
            v for k, v in losses.items() if k != "total_ar_loss"
        )
        return losses

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def codebook_utilization(self) -> Dict[str, float]:
        """Report utilization (fraction of active entries) per codebook.

        Returns:
            Dictionary mapping codebook name to utilization float
            in ``[0.0, 1.0]``.
        """
        return {
            "topology": self.codebooks.topology_codebook.codebook_utilization(),
            "geometry": self.codebooks.geometry_codebook.codebook_utilization(),
            "extrusion": self.codebooks.extrusion_codebook.codebook_utilization(),
        }

    def total_codes_per_model(self) -> int:
        """Number of discrete codes produced per CAD model.

        Returns:
            Total number of codes (10 by default: 3 + 4 + 3).
        """
        return DisentangledCodebooks.TOTAL_CODES

    def num_parameters(self) -> Dict[str, int]:
        """Count trainable parameters by component.

        Returns:
            Dictionary mapping component names to parameter counts.
        """
        encoder_params = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        codebook_params = sum(
            p.numel() for p in self.codebooks.parameters() if p.requires_grad
        )
        decoder_params = sum(
            p.numel() for p in self.decoder.parameters() if p.requires_grad
        )
        ar_params = sum(
            p.numel()
            for module in [
                self.topology_ar_decoder,
                self.geometry_ar_decoder,
                self.extrusion_ar_decoder,
            ]
            for p in module.parameters()
            if p.requires_grad
        )
        total = encoder_params + codebook_params + decoder_params + ar_params

        return {
            "encoder": encoder_params,
            "codebooks": codebook_params,
            "decoder": decoder_params,
            "ar_decoders": ar_params,
            "total": total,
        }
