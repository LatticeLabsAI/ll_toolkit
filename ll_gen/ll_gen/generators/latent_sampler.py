"""Latent space sampler — diverse exploration utilities for VAE latent spaces.

Provides interpolation, neighborhood sampling, prior sampling, and
GAN-based sampling methods for exploring the learned latent geometry
and generating diverse shape variants.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ll_gen.proposals.command_proposal import CommandSequenceProposal

if TYPE_CHECKING:
    from ll_gen.generators.neural_vae import NeuralVAEGenerator

_log = logging.getLogger(__name__)


class LatentSampler:
    """Utilities for exploring VAE latent space.

    Provides interpolation, neighborhood sampling, prior sampling, and
    GAN-based latent vector generation. Optionally integrates with a
    NeuralVAEGenerator for decoding latent vectors back to proposals.

    Attributes:
        vae_generator: Optional NeuralVAEGenerator for decoding latents.
        latent_dim: Latent vector dimension.
        device: Target device ("cpu" or "cuda").
    """

    def __init__(
        self,
        vae_generator: NeuralVAEGenerator | None = None,
        latent_dim: int = 256,
        device: str = "cpu",
    ) -> None:
        """Initialize the latent sampler.

        Args:
            vae_generator: Optional VAE generator for decoding latents.
            latent_dim: Dimensionality of latent vectors.
            device: Target device ("cpu" or "cuda").
        """
        self.vae_generator = vae_generator
        self.latent_dim = latent_dim
        self.device = device

    def interpolate(
        self,
        latent1: np.ndarray,
        latent2: np.ndarray,
        steps: int = 5,
        seed: int | None = None,
    ) -> list[np.ndarray]:
        """Interpolate between two latent vectors via spherical linear interpolation.

        Uses SLERP to follow the geodesic on the hypersphere, providing
        smooth interpolation in latent space.

        Args:
            latent1: First latent vector (shape: (latent_dim,)).
            latent2: Second latent vector (shape: (latent_dim,)).
            steps: Number of interpolation steps (including endpoints).
            seed: Optional random seed for reproducibility.

        Returns:
            List of interpolated latent vectors, from latent1 to latent2.
        """
        latent1 = np.array(latent1, dtype=np.float32)
        latent2 = np.array(latent2, dtype=np.float32)

        if latent1.shape != latent2.shape:
            raise ValueError(
                f"Latent shapes must match: {latent1.shape} vs {latent2.shape}"
            )

        # Guard against steps <= 1
        if steps <= 1:
            return [latent1.copy()]

        # Normalize for SLERP
        l1_norm = latent1 / (np.linalg.norm(latent1) + 1e-8)
        l2_norm = latent2 / (np.linalg.norm(latent2) + 1e-8)

        # Compute angle
        dot_product = np.clip(np.dot(l1_norm, l2_norm), -1.0, 1.0)
        omega = np.arccos(dot_product)

        if np.abs(omega) < 1e-6:
            # Vectors are nearly parallel
            _log.debug("Latent vectors are nearly parallel; using linear interpolation")
            return [
                latent1 + (latent2 - latent1) * (t / (steps - 1))
                for t in range(steps)
            ]

        results: list[np.ndarray] = []
        sin_omega = np.sin(omega)

        for i in range(steps):
            t = i / (steps - 1)
            weight1 = np.sin((1 - t) * omega) / sin_omega
            weight2 = np.sin(t * omega) / sin_omega

            interpolated = weight1 * latent1 + weight2 * latent2
            results.append(interpolated.astype(np.float32))

        return results

    def explore_neighborhood(
        self,
        seed_latent: np.ndarray,
        radius: float = 0.3,
        num_samples: int = 5,
        seed: int | None = None,
    ) -> list[np.ndarray]:
        """Sample points in a hypersphere around a seed latent vector.

        Generates random directions on the unit hypersphere, scales by
        the specified radius, and adds to the seed.

        Args:
            seed_latent: Center of the neighborhood (shape: (latent_dim,)).
            radius: Radius of the hypersphere neighborhood.
            num_samples: Number of points to sample.
            seed: Optional random seed for reproducibility.

        Returns:
            List of latent vectors sampled in the neighborhood.
        """
        rng = np.random.default_rng(seed)
        seed_latent = np.array(seed_latent, dtype=np.float32)
        results: list[np.ndarray] = []

        for _ in range(num_samples):
            # Sample random direction on unit hypersphere
            direction = rng.standard_normal(self.latent_dim).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            # Scale by radius and add random magnitude component
            magnitude = rng.uniform(0.5, 1.0)
            perturbation = direction * radius * magnitude

            sampled = seed_latent + perturbation
            results.append(sampled.astype(np.float32))

        return results

    def sample_from_prior(
        self,
        num_samples: int = 3,
        seed: int | None = None,
    ) -> list[np.ndarray]:
        """Sample latent vectors from the prior N(0, I).

        Args:
            num_samples: Number of vectors to sample.
            seed: Optional random seed for reproducibility.

        Returns:
            List of latent vectors sampled from the standard normal prior.
        """
        rng = np.random.default_rng(seed)
        samples: list[np.ndarray] = []

        for _ in range(num_samples):
            sample = rng.standard_normal(self.latent_dim).astype(np.float32)
            samples.append(sample)

        _log.debug(f"Sampled {num_samples} latent vectors from prior N(0, I)")
        return samples

    def sample_from_gan(
        self,
        num_samples: int = 3,
        seed: int | None = None,
    ) -> list[np.ndarray]:
        """Sample latent vectors from a learned GAN generator (if available).

        Falls back to sample_from_prior if GAN is not available.

        Args:
            num_samples: Number of vectors to sample.
            seed: Optional random seed for reproducibility.

        Returns:
            List of latent vectors from GAN or prior.
        """
        try:
            from ll_stepnet.stepnet.latent_gan import LatentGAN
        except ImportError:
            _log.warning(
                "ll_stepnet.stepnet.latent_gan not available; "
                "falling back to prior sampling"
            )
            return self.sample_from_prior(num_samples, seed=seed)

        try:
            rng = np.random.default_rng(seed)
            gan = LatentGAN(latent_dim=self.latent_dim, device=self.device)
            samples: list[np.ndarray] = []

            for _ in range(num_samples):
                noise = rng.standard_normal((1, self.latent_dim)).astype(np.float32)

                try:
                    import torch

                    noise_tensor = torch.from_numpy(noise).float()
                    if self.device.startswith("cuda"):
                        noise_tensor = noise_tensor.to(self.device)

                    with torch.no_grad():
                        generated = gan.generator(noise_tensor)

                    if isinstance(generated, torch.Tensor):
                        sample = generated.detach().cpu().numpy().squeeze()
                    else:
                        sample = np.array(generated).squeeze()

                    samples.append(sample.astype(np.float32))
                except (ImportError, RuntimeError) as e:
                    _log.warning(f"GAN sampling failed: {e}; falling back to prior")
                    return self.sample_from_prior(num_samples, seed=seed)

            _log.debug(f"Sampled {num_samples} latent vectors from GAN")
            return samples

        except Exception as e:
            _log.warning(f"GAN initialization failed: {e}; falling back to prior")
            return self.sample_from_prior(num_samples, seed=seed)

    def decode_latents(
        self,
        latents: list[np.ndarray],
        prompt: str = "",
    ) -> list[CommandSequenceProposal]:
        """Decode latent vectors into command sequence proposals.

        Requires self.vae_generator to be set.

        Args:
            latents: List of latent vectors to decode.
            prompt: Optional prompt for the proposals.

        Returns:
            List of CommandSequenceProposal objects.

        Raises:
            RuntimeError: If vae_generator is not set.
        """
        if self.vae_generator is None:
            raise RuntimeError(
                "vae_generator is required for decoding; "
                "provide one in __init__"
            )

        proposals: list[CommandSequenceProposal] = []

        for latent in latents:
            latent = np.array(latent, dtype=np.float32)

            # Create a proposal with this latent vector
            # The generator can use this latent for decoding
            try:
                import torch

                # Set model's latent vector if possible
                if hasattr(self.vae_generator._model, "set_latent"):
                    self.vae_generator._model.set_latent(
                        torch.from_numpy(latent).float().to(self.device)
                    )

                    # Trigger decoding
                    with torch.no_grad():
                        latent_tensor = torch.from_numpy(latent).float().to(self.device)
                        output = self.vae_generator._model.decode(latent_tensor)

                    command_logits = output.get("command_logits")
                    param_logits = output.get("param_logits")

                    token_ids = []
                    if command_logits is not None and param_logits is not None:
                        token_ids = self.vae_generator._logits_to_token_ids(
                            command_logits, param_logits
                        )

                    confidence = self.vae_generator._compute_confidence(
                        command_logits, param_logits
                    )

                    proposal = CommandSequenceProposal(
                        token_ids=token_ids,
                        source_prompt=prompt,
                        confidence=confidence,
                        generation_metadata=self.vae_generator._build_metadata(
                            "STEPVAE",
                            from_latent_sampler=True,
                        ),
                        latent_vector=latent,
                    )
                    proposals.append(proposal)

                else:
                    # Model doesn't support direct latent setting
                    _log.warning(
                        "Model does not support set_latent(); skipping this latent"
                    )

            except ImportError:
                _log.warning("torch not available; cannot decode latents")
                break

        _log.debug(
            f"Decoded {len(proposals)} latent vectors into proposals"
        )
        return proposals

    @property
    def is_vae_ready(self) -> bool:
        """Check if VAE generator is available and initialized.

        Returns:
            True if vae_generator is set and its model is initialized.
        """
        if self.vae_generator is None:
            return False
        return self.vae_generator._model is not None
