"""
Diffusion-based denoising for structured CAD generation.

Following BrepGen, this module implements:
    - DDPMScheduler: linear beta schedule with forward/reverse processes
      and an optional PNDM sampler for accelerated inference.
    - CADDenoiser: self-attention transformer that predicts noise given
      noisy latents and sinusoidal timestep embeddings.
    - StructuredDiffusion: orchestrates 4-stage sequential denoising
      (face positions, face geometry, edge positions, edge-vertex geometry),
      each with its own CADDenoiser.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT_DENOISER_HEADS, DEFAULT_DENOISER_HIDDEN_DIM

_log = logging.getLogger(__name__)


class DDPMScheduler:
    """Linear-beta DDPM noise scheduler.

    Precomputes alpha, alpha_bar, and sigma schedules for T timesteps
    and supports:
        - add_noise (forward process)
        - step (single reverse step, DDPM)
        - pndm_step (accelerated PNDM/PLMS reverse step)

    Args:
        num_timesteps: Total number of diffusion steps.
        beta_start: Starting value of the linear beta schedule.
        beta_end: Ending value of the linear beta schedule.
        inference_steps: Number of evenly-spaced steps for PNDM sampling.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        inference_steps: int = 200,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.inference_steps = inference_steps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)

        # Precompute quantities for q(x_t | x_0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

        # Posterior variance for reverse process
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )

        # PNDM timestep schedule (evenly spaced subset)
        self._pndm_timesteps = torch.linspace(
            num_timesteps - 1, 0, inference_steps, dtype=torch.long
        )

        # Buffer for PNDM multi-step noise predictions
        self._pndm_ets: List[torch.Tensor] = []

        _log.info(
            "DDPMScheduler initialised: T=%d, beta=[%.1e, %.1e], "
            "inference_steps=%d",
            num_timesteps, beta_start, beta_end, inference_steps,
        )

    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to clean data.

        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x_start: Clean data [B, ...] (e.g. [B, D] or [B, S, D]).
            noise: Gaussian noise, same shape as ``x_start``.
            timesteps: Integer timestep indices [B].

        Returns:
            Noisy data, same shape as ``x_start``, at the given timesteps.
        """
        device = x_start.device
        # Reshape the per-batch coefficients to [B, 1, ..., 1] so they broadcast
        # over every feature dim (works for [B, D] and [B, S, D] alike).
        coeff_shape = [x_start.shape[0]] + [1] * (x_start.ndim - 1)
        sqrt_ab = self.sqrt_alpha_bar.to(device)[timesteps].reshape(coeff_shape)
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar.to(device)[
            timesteps
        ].reshape(coeff_shape)

        return sqrt_ab * x_start + sqrt_one_minus_ab * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Single DDPM reverse step: predict x_{t-1} from x_t.

        Args:
            model_output: Predicted noise [B, D].
            timestep: Current integer timestep.
            sample: Current noisy sample [B, D].

        Returns:
            Denoised sample at t-1, [B, D].
        """
        device = sample.device
        beta_t = self.betas[timestep].to(device)
        alpha_t = self.alphas[timestep].to(device)
        alpha_bar_t = self.alpha_bar[timestep].to(device)
        alpha_bar_prev_t = self.alpha_bar_prev[timestep].to(device)

        # Predict x_0
        pred_x0 = (
            sample - torch.sqrt(1.0 - alpha_bar_t) * model_output
        ) / torch.sqrt(alpha_bar_t)

        # Posterior mean
        coeff1 = beta_t * torch.sqrt(alpha_bar_prev_t) / (1.0 - alpha_bar_t)
        coeff2 = (
            (1.0 - alpha_bar_prev_t) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)
        )
        mean = coeff1 * pred_x0 + coeff2 * sample

        # Add noise (except at t=0)
        if timestep > 0:
            noise = torch.randn_like(sample)
            variance = self.posterior_variance[timestep].to(device)
            return mean + torch.sqrt(variance) * noise
        return mean

    def ddim_step_with_log_prob(
        self,
        model_output: torch.Tensor,
        timestep: int,
        timestep_prev: int,
        sample: torch.Tensor,
        eta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stochastic DDIM reverse step returning its Gaussian log-probability.

        Implements the per-step transition used by diffusion policy-gradient
        training (DDPO; Black et al., 2023). The math is the DDIM ``eta``
        sampler shared verbatim by the reference implementations
        (Make-a-Shape ``gaussian_diffusion.ddim_sample`` and the identical
        ``brepdiff``/``diff3d`` variants)::

            x0    = (x_t - sqrt(1 - ab_t) * eps) / sqrt(ab_t)
            sigma = eta * sqrt((1 - ab_prev)/(1 - ab_t)) * sqrt(1 - ab_t/ab_prev)
            mean  = sqrt(ab_prev) * x0 + sqrt(max(1 - ab_prev - sigma^2, 0)) * eps
            x_prev = mean + sigma * noise            # noise ~ N(0, I)

        where ``eps`` is the model's predicted noise. The returned log-prob is
        ``log N(x_prev; mean, sigma^2 I)`` summed over the feature dimension
        (one scalar per batch element). Because ``mean`` is a function of
        ``model_output`` (the denoiser's epsilon prediction), gradients flow
        into the model parameters — this is what makes the RL signal real
        rather than a detached stand-in.

        Args:
            model_output: Predicted noise eps_theta(x_t, t), shape [B, D].
            timestep: Current integer timestep ``t``.
            timestep_prev: Next (smaller) timestep ``t'``; pass ``-1`` for the
                final step landing on x_0 (``ab_prev = 1.0``).
            sample: Current noisy sample x_t, shape [B, D].
            eta: DDIM stochasticity. Must be > 0 for a non-degenerate policy
                gradient; ``1.0`` recovers ancestral-DDPM-like noise.

        Returns:
            Tuple ``(x_prev, log_prob, entropy)`` where ``log_prob`` and
            ``entropy`` are shape ``[B]`` tensors. ``log_prob`` carries the
            gradient to ``model_output`` (and thus the model parameters).
        """
        device = sample.device
        alpha_bar = self.alpha_bar.to(device)
        ab_t = alpha_bar[timestep]
        if timestep_prev < 0:
            ab_prev = torch.ones((), device=device, dtype=ab_t.dtype)
        else:
            ab_prev = alpha_bar[timestep_prev]

        # Reconstruct x0 from the epsilon prediction (model_output).
        x0 = (sample - torch.sqrt(1.0 - ab_t) * model_output) / torch.sqrt(ab_t)

        # DDIM stochasticity. At the final step ab_prev == 1, so the second
        # factor is sqrt(1 - ab_t) * 0 == 0 -> sigma == 0 (deterministic).
        sigma = (
            eta
            * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t))
            * torch.sqrt(torch.clamp(1.0 - ab_t / ab_prev, min=0.0))
        )

        # Deterministic direction term (radicand clamped >= 0 for safety).
        dir_coeff = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma**2, min=0.0))
        mean = torch.sqrt(ab_prev) * x0 + dir_coeff * model_output

        batch_size = sample.shape[0]
        # Reduce log-prob/entropy over every non-batch dimension, so this works
        # for both a single latent [B, D] and a primitive-set latent [B, N, D].
        reduce_dims = tuple(range(1, sample.ndim))
        per_sample_dim = 1
        for d in sample.shape[1:]:
            per_sample_dim *= d

        if float(sigma) > 0.0:
            noise = torch.randn_like(sample)
            x_prev = mean + sigma * noise
            var = torch.clamp(sigma**2, min=1e-12)
            # DDPO/REINFORCE score function: the sampled action must be treated
            # as FIXED so that grad log N(a; mean, var) flows through `mean`
            # (the policy) only. Using the live x_prev would make
            # (x_prev - mean) == sigma*noise a constant w.r.t. mean -> zero
            # gradient. Detach the action before scoring it.
            action = x_prev.detach()
            # log N(action; mean, var * I) summed over all feature dims -> [B].
            log_prob = (
                -0.5 * ((action - mean) ** 2) / var
                - 0.5 * torch.log(2.0 * math.pi * var)
            ).sum(dim=reduce_dims)
            # Differential entropy of the isotropic Gaussian (per sample) -> [B].
            entropy = (
                0.5 * per_sample_dim * (1.0 + math.log(2.0 * math.pi))
                + 0.5 * per_sample_dim * torch.log(var)
            ).expand(batch_size)
        else:
            # Deterministic final step: no stochastic action, no log-prob term.
            x_prev = mean
            log_prob = torch.zeros(batch_size, device=device, dtype=mean.dtype)
            entropy = torch.zeros(batch_size, device=device, dtype=mean.dtype)

        return x_prev, log_prob, entropy

    @property
    def pndm_timesteps(self) -> torch.Tensor:
        """Return the PNDM timestep schedule."""
        return self._pndm_timesteps

    def reset_pndm(self) -> None:
        """Reset the PNDM multi-step buffer before a new sampling run."""
        self._pndm_ets = []

    def pndm_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Pseudo Numerical Diffusion Model (PNDM/PLMS) reverse step.

        Uses a linear multi-step method (up to 4th order) for faster
        inference with fewer function evaluations.

        Args:
            model_output: Predicted noise [B, D].
            timestep: Current timestep index.
            sample: Current noisy sample [B, D].

        Returns:
            Denoised sample, [B, D].
        """
        self._pndm_ets.append(model_output)

        if len(self._pndm_ets) == 1:
            # First step: use Euler
            et = model_output
        elif len(self._pndm_ets) == 2:
            # 2nd-order linear multistep
            et = (3.0 * self._pndm_ets[-1] - self._pndm_ets[-2]) / 2.0
        elif len(self._pndm_ets) == 3:
            # 3rd-order
            et = (
                23.0 * self._pndm_ets[-1]
                - 16.0 * self._pndm_ets[-2]
                + 5.0 * self._pndm_ets[-3]
            ) / 12.0
        else:
            # 4th-order Adams-Bashforth
            et = (
                55.0 * self._pndm_ets[-1]
                - 59.0 * self._pndm_ets[-2]
                + 37.0 * self._pndm_ets[-3]
                - 9.0 * self._pndm_ets[-4]
            ) / 24.0

        # Apply the step using the multi-step corrected noise estimate
        return self.step(et, timestep, sample)


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.

    Maps integer timesteps to dense vectors using sin/cos functions,
    then projects through a 2-layer MLP.

    Args:
        embed_dim: Output embedding dimension.
    """

    def __init__(self, embed_dim: int = 1024) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute timestep embeddings.

        Args:
            timesteps: Integer timestep indices [B].

        Returns:
            Embeddings [B, embed_dim].
        """
        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            * -emb_scale
        )
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, embed_dim]

        # If embed_dim is odd, pad
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.projection(emb)


class CADDenoiser(nn.Module):
    """Self-attention denoiser that predicts noise from noisy latents.

    Architecture: sinusoidal timestep embedding + self-attention transformer
    with num_layers layers and num_heads heads.

    Args:
        latent_dim: Dimension of the input noisy latent.
        hidden_dim: Transformer hidden dimension.
        num_layers: Number of self-attention layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = DEFAULT_DENOISER_HIDDEN_DIM,
        num_layers: int = 12,
        num_heads: int = DEFAULT_DENOISER_HEADS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(hidden_dim)

        # Project noisy latent to hidden_dim
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Project back to latent_dim (predict noise)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        _log.info(
            "CADDenoiser initialised: latent_dim=%d, hidden_dim=%d, "
            "layers=%d, heads=%d",
            latent_dim, hidden_dim, num_layers, num_heads,
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from a noisy latent and timestep.

        Args:
            noisy_latent: [B, D] or [B, S, D] noisy data.
            timesteps: [B] integer timestep indices.

        Returns:
            Predicted noise, same shape as noisy_latent.
        """
        is_2d = noisy_latent.dim() == 2
        if is_2d:
            # Treat as single-element sequence for the transformer
            noisy_latent = noisy_latent.unsqueeze(1)  # [B, 1, D]

        batch_size, seq_len, _ = noisy_latent.shape

        # Project to hidden dim
        h = self.input_proj(noisy_latent)  # [B, S, H]

        # Add timestep embedding (broadcast across sequence)
        t_emb = self.time_embed(timesteps)  # [B, H]
        h = h + t_emb.unsqueeze(1)  # [B, S, H]

        # Self-attention
        h = self.transformer(h)  # [B, S, H]
        h = self.layer_norm(h)

        # Project back to latent_dim
        noise_pred = self.output_proj(h)  # [B, S, D]

        if is_2d:
            noise_pred = noise_pred.squeeze(1)  # [B, D]

        return noise_pred


class GeometryCodec(nn.Module):
    """Autoencoder between per-primitive diffusion latents and B-Rep geometry.

    A face is a ``U x V`` grid of 3D points; an edge is an ``M``-point polyline
    (the BrepGen / BrepDiff representation — ``brepdiff`` ``uvgrid.py``). Each
    primitive (face or edge) has its **own** latent token, so encode/decode is a
    clean per-primitive mapping with no set-bottleneck:

        encode_faces: [B, N_faces, U, V, 3] -> [B, N_faces, latent_dim]
        decode_faces: [B, N_faces, latent_dim] -> [B, N_faces, U, V, 3]
        encode_edges: [B, N_edges, M, 3]     -> [B, N_edges, latent_dim]
        decode_edges: [B, N_edges, latent_dim] -> [B, N_edges, M, 3]

    The encoders are UV-Net-style convolutional encoders (Conv2d over the face
    grid, Conv1d over the edge polyline; cad-feature-detection ``encoders.py``).
    No learned UV-grid decoder exists in the references (BrepDiff's detokenizer
    is an identity reshape), so the decoders mirror the encoders as MLP heads
    back to the grid/polyline. Trained with a masked MSE reconstruction loss
    (``brepdiff.py`` loss). This gives the diffusion latent space a real,
    differentiable mapping to geometry that the surface executor can fit and
    sew into solids.
    """

    def __init__(
        self,
        latent_dim: int,
        uv_grid_size: int,
        edge_num_points: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.uv = uv_grid_size
        self.edge_pts = edge_num_points

        # Face encoder (UV-Net surface style): [B, 3, U, V] -> [B, latent_dim].
        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
        )
        # Edge encoder (UV-Net curve style): [B, 3, M] -> [B, latent_dim].
        self.edge_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
        )
        # Decoders mirror the encoders (per-primitive MLP heads).
        self.face_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, uv_grid_size * uv_grid_size * 3),
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, edge_num_points * 3),
        )

    def encode_faces(self, face_grids: torch.Tensor) -> torch.Tensor:
        """[B, N, U, V, 3] -> [B, N, latent_dim]."""
        b, n = face_grids.shape[0], face_grids.shape[1]
        # .contiguous() after permute: Conv2d's backward calls .view() on its
        # input, which fails on the permuted (non-contiguous) tensor.
        x = face_grids.reshape(b * n, self.uv, self.uv, 3).permute(0, 3, 1, 2).contiguous()
        z = self.face_encoder(x)
        return z.reshape(b, n, self.latent_dim)

    def decode_faces(self, latent: torch.Tensor) -> torch.Tensor:
        """[B, N, latent_dim] -> [B, N, U, V, 3]."""
        b, n = latent.shape[0], latent.shape[1]
        out = self.face_decoder(latent.reshape(b * n, self.latent_dim))
        return out.reshape(b, n, self.uv, self.uv, 3)

    def encode_edges(self, edge_points: torch.Tensor) -> torch.Tensor:
        """[B, N, M, 3] -> [B, N, latent_dim]."""
        b, n = edge_points.shape[0], edge_points.shape[1]
        # .contiguous() after permute (see encode_faces): Conv1d backward .view().
        x = edge_points.reshape(b * n, self.edge_pts, 3).permute(0, 2, 1).contiguous()
        z = self.edge_encoder(x)
        return z.reshape(b, n, self.latent_dim)

    def decode_edges(self, latent: torch.Tensor) -> torch.Tensor:
        """[B, N, latent_dim] -> [B, N, M, 3]."""
        b, n = latent.shape[0], latent.shape[1]
        out = self.edge_decoder(latent.reshape(b * n, self.latent_dim))
        return out.reshape(b, n, self.edge_pts, 3)

    @staticmethod
    def _masked_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        empty_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mean squared error per primitive, averaged over non-empty primitives.

        ``empty_mask`` is ``[B, N]`` with True marking padded/empty primitives
        (BrepDiff ``empty_mask``); those are excluded from the loss.
        """
        per_primitive = (pred - target).pow(2).flatten(2).mean(dim=2)  # [B, N]
        if empty_mask is None:
            return per_primitive.mean()
        valid = (~empty_mask).to(per_primitive.dtype)
        denom = valid.sum().clamp(min=1.0)
        return (per_primitive * valid).sum() / denom

    def reconstruction_loss(
        self,
        face_grids: torch.Tensor,
        edge_points: torch.Tensor,
        face_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode then decode the geometry and return masked-MSE recon losses."""
        face_rec = self.decode_faces(self.encode_faces(face_grids))
        edge_rec = self.decode_edges(self.encode_edges(edge_points))
        face_loss = self._masked_mse(face_rec, face_grids, face_mask)
        edge_loss = self._masked_mse(edge_rec, edge_points, edge_mask)
        return {
            "face_recon_loss": face_loss,
            "edge_recon_loss": edge_loss,
            "total_recon_loss": face_loss + edge_loss,
        }


class StructuredDiffusion(nn.Module):
    """Four-stage sequential diffusion following BrepGen.

    Stages (each with its own CADDenoiser):
        1. Face positions
        2. Face geometry
        3. Edge positions
        4. Edge-vertex geometry

    Each stage is conditioned on the denoised output of the preceding
    stage via concatenation.

    Args:
        config: DiffusionConfig with architectural hyperparameters.
    """

    STAGE_NAMES: Tuple[str, ...] = (
        "face_positions",
        "face_geometry",
        "edge_positions",
        "edge_vertex_geometry",
    )

    def __init__(self, config: Optional[object] = None) -> None:
        super().__init__()

        if config is None:
            from .config import DiffusionConfig
            config = DiffusionConfig()
        self.config = config

        latent_dim = getattr(config, "latent_dim", 256)
        denoiser_hidden_dim = getattr(
            config, "denoiser_hidden_dim", DEFAULT_DENOISER_HIDDEN_DIM
        )
        denoiser_layers = getattr(config, "denoiser_layers", 12)
        denoiser_heads = getattr(config, "denoiser_heads", DEFAULT_DENOISER_HEADS)
        num_timesteps = getattr(config, "num_timesteps", 1000)
        beta_start = getattr(config, "beta_start", 1e-4)
        beta_end = getattr(config, "beta_end", 0.02)
        inference_steps = getattr(config, "inference_steps", 200)

        # One denoiser per stage
        self.denoisers = nn.ModuleDict()
        for stage_name in self.STAGE_NAMES:
            self.denoisers[stage_name] = CADDenoiser(
                latent_dim=latent_dim,
                hidden_dim=denoiser_hidden_dim,
                num_layers=denoiser_layers,
                num_heads=denoiser_heads,
            )

        # Conditioning projections: map denoised output of previous stage
        # into the latent dim of the next stage.
        self.cond_projections = nn.ModuleDict()
        for i in range(1, len(self.STAGE_NAMES)):
            curr_stage = self.STAGE_NAMES[i]
            # Previous denoised + current noisy -> current latent_dim
            self.cond_projections[curr_stage] = nn.Linear(
                latent_dim * 2, latent_dim
            )

        # Shared scheduler
        self.scheduler = DDPMScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            inference_steps=inference_steps,
        )

        # Per-primitive set sizes: face stages produce `num_faces` tokens and
        # edge stages produce `num_edges` tokens (each token is one face/edge).
        num_faces = getattr(config, "num_faces", 8)
        num_edges = getattr(config, "num_edges", 12)
        uv_grid_size = getattr(config, "uv_grid_size", 8)
        edge_num_points = getattr(config, "edge_num_points", 12)
        codec_hidden_dim = getattr(config, "codec_hidden_dim", 256)

        self._stage_tokens: Dict[str, int] = {
            "face_positions": num_faces,
            "face_geometry": num_faces,
            "edge_positions": num_edges,
            "edge_vertex_geometry": num_edges,
        }
        self._num_faces = num_faces
        self._num_edges = num_edges

        # Latent <-> geometry autoencoder. The final face-geometry tokens decode
        # to U×V×3 face grids; the final edge tokens decode to M×3 polylines.
        self.geometry_codec = GeometryCodec(
            latent_dim=latent_dim,
            uv_grid_size=uv_grid_size,
            edge_num_points=edge_num_points,
            hidden_dim=codec_hidden_dim,
        )

        self._latent_dim = latent_dim
        self._num_timesteps = num_timesteps

        _log.info(
            "StructuredDiffusion initialised: stages=%s, T=%d, "
            "num_faces=%d, num_edges=%d, uv_grid=%d, edge_pts=%d",
            self.STAGE_NAMES, num_timesteps, num_faces, num_edges,
            uv_grid_size, edge_num_points,
        )

    def _decode_geometry(
        self, stage_latents: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Decode the final face/edge stage tokens into B-Rep geometry tensors.

        Returns ``face_grids`` [B, N_faces, U, V, 3] and ``edge_points``
        [B, N_edges, M, 3] via the geometry codec.
        """
        face_tokens = stage_latents["face_geometry"]
        edge_tokens = stage_latents["edge_vertex_geometry"]
        return {
            "face_grids": self.geometry_codec.decode_faces(face_tokens),
            "edge_points": self.geometry_codec.decode_edges(edge_tokens),
        }

    def _condition_on_prev(
        self,
        stage_name: str,
        x: torch.Tensor,
        prev_denoised: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Condition the current stage's tokens on a pooled summary of the
        previous stage. Pooling makes conditioning robust to differing token
        counts across stages (e.g. faces -> edges)."""
        if prev_denoised is None or stage_name not in self.cond_projections:
            return x
        prev_summary = prev_denoised.mean(dim=1, keepdim=True)  # [B, 1, D]
        prev_broadcast = prev_summary.expand(-1, x.shape[1], -1)  # [B, S, D]
        combined = torch.cat([x, prev_broadcast], dim=-1)  # [B, S, 2D]
        return self.cond_projections[stage_name](combined)

    def forward_train(
        self,
        stage_data: Optional[Dict[str, torch.Tensor]] = None,
        geometry: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward: denoising loss per stage + codec reconstruction.

        Each stage independently samples a random timestep, adds noise, and
        predicts the noise (teacher-forced on a pooled summary of the previous
        stage's clean tokens). When ``geometry`` is supplied, the clean
        per-stage token latents are the codec's *encoding* of the real geometry,
        so the diffusion learns to denoise in the codec's latent space, and a
        masked-MSE reconstruction term trains the codec itself — making the
        latent<->geometry mapping coherent end to end.

        Args:
            stage_data: Optional explicit clean stage latents, each [B, S, D]
                (or [B, D], which is promoted to a single token). Overrides the
                geometry-derived targets per stage when both are given.
            geometry: Optional dict with ``face_grids`` [B, N_faces, U, V, 3],
                ``edge_points`` [B, N_edges, M, 3] and optional ``face_mask`` /
                ``edge_mask`` [B, N] (True = padded/empty primitive).

        Returns:
            Dictionary with ``{stage_name}_loss`` denoising terms, optional
            ``face_recon_loss`` / ``edge_recon_loss``, and ``total_loss``.
        """
        if not stage_data and not geometry:
            raise ValueError(
                "forward_train requires stage_data and/or geometry."
            )

        # Build clean per-stage token targets.
        stage_targets: Dict[str, torch.Tensor] = {}
        if geometry is not None:
            face_z = self.geometry_codec.encode_faces(geometry["face_grids"])
            edge_z = self.geometry_codec.encode_edges(geometry["edge_points"])
            stage_targets["face_positions"] = face_z
            stage_targets["face_geometry"] = face_z
            stage_targets["edge_positions"] = edge_z
            stage_targets["edge_vertex_geometry"] = edge_z
        if stage_data:
            stage_targets.update(stage_data)

        losses: Dict[str, torch.Tensor] = {}
        device = next(iter(stage_targets.values())).device
        batch_size = next(iter(stage_targets.values())).shape[0]

        prev_clean: Optional[torch.Tensor] = None

        for stage_name in self.STAGE_NAMES:
            if stage_name not in stage_targets:
                continue

            clean = stage_targets[stage_name]
            if clean.dim() == 2:  # promote [B, D] -> [B, 1, D]
                clean = clean.unsqueeze(1)

            # Random timesteps for this stage
            t = torch.randint(
                0, self._num_timesteps, (batch_size,), device=device
            )
            noise = torch.randn_like(clean)
            noisy = self.scheduler.add_noise(clean, noise, t)

            # Condition on previous stage (teacher forcing with clean tokens).
            noisy = self._condition_on_prev(stage_name, noisy, prev_clean)

            # Predict noise
            noise_pred = self.denoisers[stage_name](noisy, t)
            stage_loss = F.mse_loss(noise_pred, noise)
            losses[f"{stage_name}_loss"] = stage_loss

            prev_clean = clean

        total_loss = sum(losses.values())

        # Codec reconstruction loss (trains the latent<->geometry autoencoder).
        if geometry is not None:
            recon = self.geometry_codec.reconstruction_loss(
                geometry["face_grids"],
                geometry["edge_points"],
                geometry.get("face_mask"),
                geometry.get("edge_mask"),
            )
            losses["face_recon_loss"] = recon["face_recon_loss"]
            losses["edge_recon_loss"] = recon["edge_recon_loss"]
            total_loss = total_loss + recon["total_recon_loss"]

        losses["total_loss"] = total_loss

        return losses

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        use_pndm: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Generate new structured CAD data via sequential denoising.

        Args:
            batch_size: Number of samples.
            device: Target device.
            use_pndm: Whether to use PNDM accelerated sampling.

        Returns:
            Dictionary mapping each stage name to its denoised token latents
            ([B, N_faces or N_edges, D]) plus decoded geometry tensors
            ``face_grids`` [B, N_faces, U, V, 3] and ``edge_points``
            [B, N_edges, M, 3].
        """
        if device is None:
            device = next(self.parameters()).device

        results: Dict[str, torch.Tensor] = {}
        prev_denoised: Optional[torch.Tensor] = None

        for stage_name in self.STAGE_NAMES:
            denoiser = self.denoisers[stage_name]
            denoiser.eval()

            # Reset PNDM buffer before each stage to prevent stale
            # noise predictions from prior stages leaking through
            self.scheduler.reset_pndm()

            # Start from pure noise — one token per primitive (face or edge).
            n_tokens = self._stage_tokens[stage_name]
            x = torch.randn(
                batch_size, n_tokens, self._latent_dim, device=device
            )

            if use_pndm:
                timesteps = self.scheduler.pndm_timesteps.to(device)
            else:
                timesteps = torch.arange(
                    self._num_timesteps - 1, -1, -1, device=device
                )

            for t_val in timesteps:
                t = t_val.long()
                t_batch = t.expand(batch_size)

                # Condition on a pooled summary of the previous stage result.
                denoiser_input = self._condition_on_prev(
                    stage_name, x, prev_denoised
                )

                noise_pred = denoiser(denoiser_input, t_batch)

                if use_pndm:
                    x = self.scheduler.pndm_step(noise_pred, t.item(), x)
                else:
                    x = self.scheduler.step(noise_pred, t.item(), x)

            results[stage_name] = x
            prev_denoised = x

        # Decode the final stage tokens into B-Rep geometry.
        results.update(self._decode_geometry(results))
        return results

    def sample_with_log_prob(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        num_inference_steps: Optional[int] = None,
        eta: float = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """DDPO sampling: geometry plus a *differentiable* trajectory log-prob.

        Runs stochastic DDIM reverse diffusion (``eta > 0``) over all four
        stages **without** ``torch.no_grad`` and accumulates the per-step
        Gaussian log-probabilities from
        :meth:`DDPMScheduler.ddim_step_with_log_prob`. Each step's transition
        mean depends on its denoiser's epsilon prediction, so the summed
        log-prob backpropagates into every denoiser (and the stage conditioning
        projections) — enabling real diffusion policy-gradient (REINFORCE /
        DDPO) reinforcement learning. This is the path that makes the RL signal
        train the actual model parameters, replacing the previously decoupled
        noise-prior stand-in.

        Trajectory states are detached between steps (each ``x_t`` is treated as
        a fixed state in the action history, as in DDPO), which bounds memory
        while preserving the gradient inside each transition's log-prob.

        Args:
            batch_size: Number of samples to draw.
            device: Target device (defaults to the model's device).
            num_inference_steps: Reverse steps per stage (defaults to the
                scheduler's ``inference_steps``).
            eta: DDIM stochasticity. Coerced to ``1.0`` if ``<= 0`` because a
                deterministic trajectory has a degenerate (delta) policy that
                cannot provide a usable policy gradient.

        Returns:
            Tuple ``(results, total_log_prob, total_entropy)``:
                * ``results``: ``{stage_name: token latent [B, N, D]}`` plus
                  decoded ``face_grids`` / ``edge_points`` (all detached).
                * ``total_log_prob``: ``[B]`` sum of per-step log-probs across all
                  stages, connected to the model parameters.
                * ``total_entropy``: ``[B]`` sum of per-step Gaussian entropies.
        """
        if device is None:
            device = next(self.parameters()).device
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.inference_steps
        if eta <= 0.0:
            eta = 1.0

        # Evenly-spaced decreasing timestep schedule ending at 0.
        timesteps = (
            torch.linspace(
                self._num_timesteps - 1, 0, num_inference_steps, dtype=torch.long
            )
            .tolist()
        )

        results: Dict[str, torch.Tensor] = {}
        prev_denoised: Optional[torch.Tensor] = None
        total_log_prob = torch.zeros(batch_size, device=device)
        total_entropy = torch.zeros(batch_size, device=device)

        for stage_name in self.STAGE_NAMES:
            denoiser = self.denoisers[stage_name]

            # One token per primitive (face or edge) for this stage.
            n_tokens = self._stage_tokens[stage_name]
            x = torch.randn(
                batch_size, n_tokens, self._latent_dim, device=device
            )

            for i, t_val in enumerate(timesteps):
                t = int(t_val)
                t_prev = int(timesteps[i + 1]) if i + 1 < len(timesteps) else -1
                t_batch = torch.full(
                    (batch_size,), t, device=device, dtype=torch.long
                )

                # Condition on a pooled summary of the previous stage (detached).
                denoiser_input = self._condition_on_prev(
                    stage_name, x, prev_denoised
                )

                noise_pred = denoiser(denoiser_input, t_batch)

                x_prev, log_prob, entropy = (
                    self.scheduler.ddim_step_with_log_prob(
                        noise_pred, t, t_prev, x, eta=eta
                    )
                )
                total_log_prob = total_log_prob + log_prob
                total_entropy = total_entropy + entropy

                # Detach the trajectory state for the next step (DDPO treats
                # states as fixed); the gradient lives inside each log_prob.
                x = x_prev.detach()

            results[stage_name] = x
            prev_denoised = x  # already detached

        # Decode geometry from the final (detached) tokens. The geometry decode
        # is not part of the policy gradient — the RL signal trains the denoisers
        # via total_log_prob; the codec is trained separately via forward_train.
        results.update(self._decode_geometry(results))
        return results, total_log_prob, total_entropy
