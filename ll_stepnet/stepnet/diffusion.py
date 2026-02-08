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
            x_start: Clean data [B, D].
            noise: Gaussian noise [B, D].
            timesteps: Integer timestep indices [B].

        Returns:
            Noisy data [B, D] at the given timesteps.
        """
        device = x_start.device
        sqrt_ab = self.sqrt_alpha_bar.to(device)[timesteps].unsqueeze(-1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar.to(device)[
            timesteps
        ].unsqueeze(-1)

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
        hidden_dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 12,
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
        denoiser_hidden_dim = getattr(config, "denoiser_hidden_dim", 1024)
        denoiser_layers = getattr(config, "denoiser_layers", 12)
        denoiser_heads = getattr(config, "denoiser_heads", 12)
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

        self._latent_dim = latent_dim
        self._num_timesteps = num_timesteps

        _log.info(
            "StructuredDiffusion initialised: stages=%s, T=%d",
            self.STAGE_NAMES, num_timesteps,
        )

    def forward_train(
        self,
        stage_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Training forward: compute denoising loss for each stage.

        Each stage independently samples a random timestep, adds noise,
        and predicts the noise. Later stages receive the clean
        ground-truth of previous stages as conditioning (teacher forcing).

        Args:
            stage_data: Mapping from stage name to clean latent tensors [B, D].

        Returns:
            Dictionary with {stage_name}_loss scalar tensors and a total_loss.
        """
        losses: Dict[str, torch.Tensor] = {}
        device = next(iter(stage_data.values())).device
        batch_size = next(iter(stage_data.values())).shape[0]

        prev_clean: Optional[torch.Tensor] = None

        for stage_name in self.STAGE_NAMES:
            if stage_name not in stage_data:
                continue

            clean = stage_data[stage_name]

            # Random timesteps for this stage
            t = torch.randint(
                0, self._num_timesteps, (batch_size,), device=device
            )
            noise = torch.randn_like(clean)
            noisy = self.scheduler.add_noise(clean, noise, t)

            # Condition on previous stage (teacher forcing with clean data)
            if prev_clean is not None and stage_name in self.cond_projections:
                combined = torch.cat([noisy, prev_clean], dim=-1)
                noisy = self.cond_projections[stage_name](combined)

            # Predict noise
            noise_pred = self.denoisers[stage_name](noisy, t)
            stage_loss = F.mse_loss(noise_pred, noise)
            losses[f"{stage_name}_loss"] = stage_loss

            prev_clean = clean

        total_loss = sum(losses.values())
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
            Dictionary mapping stage names to denoised latents [B, D].
        """
        if device is None:
            device = next(self.parameters()).device

        results: Dict[str, torch.Tensor] = {}
        prev_denoised: Optional[torch.Tensor] = None

        for stage_name in self.STAGE_NAMES:
            denoiser = self.denoisers[stage_name]
            denoiser.eval()

            # Start from pure noise
            x = torch.randn(
                batch_size, self._latent_dim, device=device
            )

            if use_pndm:
                self.scheduler.reset_pndm()
                timesteps = self.scheduler.pndm_timesteps.to(device)
            else:
                timesteps = torch.arange(
                    self._num_timesteps - 1, -1, -1, device=device
                )

            for t_val in timesteps:
                t = t_val.long()
                t_batch = t.expand(batch_size)

                # Condition on previous stage result
                denoiser_input = x
                if (
                    prev_denoised is not None
                    and stage_name in self.cond_projections
                ):
                    combined = torch.cat([x, prev_denoised], dim=-1)
                    denoiser_input = self.cond_projections[stage_name](combined)

                noise_pred = denoiser(denoiser_input, t_batch)

                if use_pndm:
                    x = self.scheduler.pndm_step(noise_pred, t.item(), x)
                else:
                    x = self.scheduler.step(noise_pred, t.item(), x)

            results[stage_name] = x
            prev_denoised = x

        return results
