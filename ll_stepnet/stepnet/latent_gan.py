"""
Wasserstein GAN with Gradient Penalty (WGAN-GP) for the VAE latent space.

Following DeepCAD, a WGAN-GP is trained to match the prior distribution
in the VAE latent space. After training, the generator can produce
latent vectors that, when decoded by the VAE decoder, yield novel CAD
command sequences.

Architecture:
    - LatentGenerator: noise -> MLP -> z_fake
    - LatentDiscriminator: z -> MLP -> scalar score
    - LatentGAN: orchestrates training with WGAN-GP loss
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

_log = logging.getLogger(__name__)


class LatentGenerator(nn.Module):
    """Maps noise vectors to fake latent codes.

    Args:
        latent_dim: Dimension of both input noise and output latent.
        hidden_dims: Sizes of hidden layers in the MLP.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]

        self.latent_dim = latent_dim

        layers: List[nn.Module] = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(h_dim),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, latent_dim))

        self.net = nn.Sequential(*layers)

        _log.info(
            "LatentGenerator initialised: latent_dim=%d, hidden_dims=%s",
            latent_dim, hidden_dims,
        )

    def forward(self, z_noise: torch.Tensor) -> torch.Tensor:
        """Generate fake latent vectors from noise.

        Args:
            z_noise: [batch_size, latent_dim] samples from N(0,I).

        Returns:
            Fake latent vectors [batch_size, latent_dim].
        """
        return self.net(z_noise)


class LatentDiscriminator(nn.Module):
    """Wasserstein critic that scores latent vectors.

    Args:
        latent_dim: Dimensionality of the input latent vector.
        hidden_dims: Sizes of hidden layers in the MLP.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]

        self.latent_dim = latent_dim

        layers: List[nn.Module] = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

        _log.info(
            "LatentDiscriminator initialised: latent_dim=%d, hidden_dims=%s",
            latent_dim, hidden_dims,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Score a latent vector (higher = more real).

        Args:
            z: [batch_size, latent_dim].

        Returns:
            Scalar scores [batch_size, 1].
        """
        return self.net(z)


class LatentGAN:
    """WGAN-GP training loop for the VAE latent space.

    Manages the generator, discriminator, their optimizers, and the
    gradient-penalty computation.

    Args:
        latent_dim: Dimensionality of the latent space.
        gen_hidden_dims: Generator MLP hidden sizes.
        disc_hidden_dims: Discriminator MLP hidden sizes.
        gp_lambda: Gradient penalty coefficient.
        n_critic: Number of discriminator updates per generator update.
        lr_gen: Generator learning rate.
        lr_disc: Discriminator learning rate.
        device: Torch device.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        gen_hidden_dims: Optional[List[int]] = None,
        disc_hidden_dims: Optional[List[int]] = None,
        gp_lambda: float = 10.0,
        n_critic: int = 5,
        lr_gen: float = 1e-4,
        lr_disc: float = 1e-4,
        device: Optional[torch.device] = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.gp_lambda = gp_lambda
        self.n_critic = n_critic

        if gen_hidden_dims is None:
            gen_hidden_dims = [512, 512]
        if disc_hidden_dims is None:
            disc_hidden_dims = [512, 512]

        self.device = device or torch.device("cpu")

        self.generator = LatentGenerator(
            latent_dim=latent_dim, hidden_dims=gen_hidden_dims
        ).to(self.device)

        self.discriminator = LatentDiscriminator(
            latent_dim=latent_dim, hidden_dims=disc_hidden_dims
        ).to(self.device)

        self.optim_gen = torch.optim.Adam(
            self.generator.parameters(), lr=lr_gen, betas=(0.5, 0.999)
        )
        self.optim_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999)
        )

        self._step_count: int = 0

        _log.info(
            "LatentGAN initialised: latent_dim=%d, gp_lambda=%.1f, "
            "n_critic=%d, lr_gen=%.1e, lr_disc=%.1e",
            latent_dim, gp_lambda, n_critic, lr_gen, lr_disc,
        )

    def _gradient_penalty(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP.

        Args:
            real: Real latent vectors [B, D].
            fake: Fake latent vectors [B, D].

        Returns:
            Scalar gradient penalty.
        """
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        scores = self.discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1.0) ** 2).mean()

        return penalty

    def train_step(
        self, real_latents: torch.Tensor
    ) -> Dict[str, float]:
        """One training step: update critic n_critic times, then generator once.

        Args:
            real_latents: [batch_size, latent_dim] latent vectors
                produced by the VAE encoder on real data.

        Returns:
            Dictionary of loss values for logging:
                - disc_loss: latest discriminator loss
                - gen_loss: generator loss (0 if not updated this step)
                - gp: gradient penalty
                - wasserstein_distance: estimated Wasserstein distance
        """
        batch_size = real_latents.size(0)
        real_latents = real_latents.to(self.device).detach()

        metrics: Dict[str, float] = {
            "disc_loss": 0.0,
            "gen_loss": 0.0,
            "gp": 0.0,
            "wasserstein_distance": 0.0,
        }

        # Discriminator / Critic update
        self.discriminator.train()
        self.generator.train()

        for _ in range(self.n_critic):
            self.optim_disc.zero_grad()

            # Real
            score_real = self.discriminator(real_latents).mean()

            # Fake
            z_noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_latents = self.generator(z_noise).detach()
            score_fake = self.discriminator(fake_latents).mean()

            # Gradient penalty
            gp = self._gradient_penalty(real_latents, fake_latents)

            disc_loss = score_fake - score_real + self.gp_lambda * gp
            disc_loss.backward()
            self.optim_disc.step()

            metrics["disc_loss"] = disc_loss.item()
            metrics["gp"] = gp.item()
            metrics["wasserstein_distance"] = (score_real - score_fake).item()

        # Generator update
        self.optim_gen.zero_grad()

        z_noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_latents = self.generator(z_noise)
        score_fake = self.discriminator(fake_latents).mean()

        gen_loss = -score_fake
        gen_loss.backward()
        self.optim_gen.step()

        metrics["gen_loss"] = gen_loss.item()
        self._step_count += 1

        return metrics

    def sample(
        self,
        num_samples: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Sample latent vectors from the trained generator.

        Args:
            num_samples: Number of samples to generate.
            device: Target device (defaults to self.device).

        Returns:
            Generated latent vectors [num_samples, latent_dim].
        """
        target_device = device or self.device
        self.generator.eval()
        with torch.no_grad():
            z_noise = torch.randn(
                num_samples, self.latent_dim, device=target_device
            )
            z_fake = self.generator(z_noise)
        return z_fake

    def to(self, device: torch.device) -> LatentGAN:
        """Move all models and state to the given device.

        Args:
            device: Target torch device.

        Returns:
            Self for chaining.
        """
        self.device = device
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        return self
