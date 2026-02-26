"""GAN Trainer for latent space generation of CAD models.

Implements Wasserstein GAN with Gradient Penalty (WGAN-GP) for training
a generator/discriminator pair in the latent space of a pre-trained VAE.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    _has_torch = True
except ImportError:
    _has_torch = False


class GANTrainer:
    """Trainer for Wasserstein GAN with gradient penalty in the latent space.

    Trains a generator and discriminator (critic) to produce realistic latent
    vectors that can be decoded by a pre-trained VAE decoder into valid CAD
    token sequences.

    The WGAN-GP formulation provides:
    - Wasserstein distance as a meaningful training signal
    - Gradient penalty for stable training without mode collapse
    - Alternating critic/generator updates with configurable ratio

    Args:
        generator: Generator network mapping noise -> latent vectors.
        discriminator: Discriminator (critic) network scoring latent vectors.
        train_dataloader: DataLoader providing real latent vectors for training.
        device: Device string. 'auto' selects CUDA if available, else CPU.
        checkpoint_dir: Directory path for saving checkpoints.
        gp_lambda: Gradient penalty coefficient (default 10.0 per WGAN-GP paper).
        n_critic: Number of critic updates per generator update (default 5).
        lr_gen: Learning rate for the generator optimizer.
        lr_disc: Learning rate for the discriminator optimizer.
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        train_dataloader: DataLoader,
        device: str = "auto",
        checkpoint_dir: Optional[str] = None,
        gp_lambda: float = 10.0,
        n_critic: int = 5,
        lr_gen: float = 1e-4,
        lr_disc: float = 1e-4,
    ) -> None:
        if not _has_torch:
            raise ImportError(
                "PyTorch is required for GANTrainer. "
                "Install via conda: conda install pytorch -c conda-forge"
            )

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.train_dataloader = train_dataloader
        self.gp_lambda = gp_lambda
        self.n_critic = n_critic

        # Separate optimizers for G and D.
        # WGAN-GP paper (Gulrajani et al. 2017) uses Adam with
        # beta1=0.5, beta2=0.999 for both generator and discriminator.
        self.optimizer_gen = AdamW(
            generator.parameters(), lr=lr_gen, betas=(0.5, 0.999)
        )
        self.optimizer_disc = AdamW(
            discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999)
        )

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_wasserstein_dist = float("inf")
        self.history: Dict[str, List[float]] = {
            "d_loss": [],
            "g_loss": [],
            "gp_loss": [],
            "wasserstein_dist": [],
        }

        # Infer latent dimension from generator
        self._latent_dim = self._infer_latent_dim()

        _log.info(
            "GANTrainer initialized: device=%s, gp_lambda=%.1f, "
            "n_critic=%d, latent_dim=%d",
            self.device,
            self.gp_lambda,
            self.n_critic,
            self._latent_dim,
        )

    def _infer_latent_dim(self) -> int:
        """Infer the latent/noise dimension from the generator architecture.

        Returns:
            Latent dimension size, defaulting to 128 if cannot be inferred.
        """
        # Try common attribute names for the input dimension
        for attr in ("latent_dim", "noise_dim", "z_dim", "input_dim"):
            if hasattr(self.generator, attr):
                return getattr(self.generator, attr)

        # Try inspecting first linear layer
        for module in self.generator.modules():
            if isinstance(module, nn.Linear):
                return module.in_features

        _log.warning("Could not infer latent dim; defaulting to 128.")
        return 128

    def _compute_gradient_penalty(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP.

        Interpolates between real and fake samples and penalizes the critic's
        gradient norm deviating from 1.

        Args:
            real: Real latent vectors, shape (batch, latent_dim).
            fake: Generated latent vectors, shape (batch, latent_dim).

        Returns:
            Scalar gradient penalty tensor.
        """
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real)

        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        disc_interpolated = self.discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1.0) ** 2).mean()

        return gradient_penalty

    def train_discriminator_step(
        self, real_latents: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one discriminator (critic) training step.

        Computes the WGAN loss with gradient penalty:
        D_loss = D(fake) - D(real) + lambda * GP

        Args:
            real_latents: Real latent vectors, shape (batch, latent_dim).

        Returns:
            Dictionary with keys: 'd_loss', 'gp_loss', 'wasserstein_dist'.
        """
        batch_size = real_latents.size(0)
        self.discriminator.train()
        self.optimizer_disc.zero_grad()

        # Generate fake latents
        noise = torch.randn(batch_size, self._latent_dim, device=self.device)
        fake_latents = self.generator(noise).detach()

        # Critic scores
        real_score = self.discriminator(real_latents)
        fake_score = self.discriminator(fake_latents)

        # WGAN loss: maximize D(real) - D(fake) => minimize D(fake) - D(real)
        wasserstein_dist = real_score.mean() - fake_score.mean()
        d_loss = -wasserstein_dist

        # Gradient penalty
        gp = self._compute_gradient_penalty(real_latents, fake_latents)
        total_d_loss = d_loss + self.gp_lambda * gp

        total_d_loss.backward()
        self.optimizer_disc.step()

        return {
            "d_loss": total_d_loss.item(),
            "gp_loss": gp.item(),
            "wasserstein_dist": wasserstein_dist.item(),
        }

    def train_generator_step(self, batch_size: int) -> Dict[str, float]:
        """Perform one generator training step.

        Generates fake latents and optimizes the generator to fool the critic:
        G_loss = -D(G(z))

        Args:
            batch_size: Number of samples to generate.

        Returns:
            Dictionary with key: 'g_loss'.
        """
        self.generator.train()
        self.optimizer_gen.zero_grad()

        # Generate fake latents
        noise = torch.randn(batch_size, self._latent_dim, device=self.device)
        fake_latents = self.generator(noise)

        # Generator wants to maximize D(fake) => minimize -D(fake)
        fake_score = self.discriminator(fake_latents)
        g_loss = -fake_score.mean()

        g_loss.backward()
        self.optimizer_gen.step()

        return {"g_loss": g_loss.item()}

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with alternating critic/generator updates.

        Performs n_critic discriminator updates for every 1 generator update,
        following the WGAN-GP training protocol.

        Returns:
            Dictionary with keys: 'd_loss', 'g_loss', 'gp_loss',
            'wasserstein_dist'.
        """
        d_loss_sum = 0.0
        g_loss_sum = 0.0
        gp_loss_sum = 0.0
        wd_sum = 0.0
        num_d_steps = 0
        num_g_steps = 0

        pbar = tqdm(self.train_dataloader, desc=f"GAN Epoch {self.epoch}")
        critic_step = 0

        for batch in pbar:
            # Extract real latents from batch
            if isinstance(batch, torch.Tensor):
                real_latents = batch.to(self.device)
            elif isinstance(batch, dict):
                real_latents = batch.get(
                    "latents", batch.get("token_ids", None)
                )
                if real_latents is None:
                    raise KeyError(
                        "Batch must contain 'latents' or 'token_ids' key."
                    )
                real_latents = real_latents.to(self.device).float()
            else:
                raise TypeError(
                    f"Unsupported batch type: {type(batch)}. "
                    "Expected Tensor or dict."
                )

            batch_size = real_latents.size(0)

            # Train discriminator
            d_metrics = self.train_discriminator_step(real_latents)
            d_loss_sum += d_metrics["d_loss"]
            gp_loss_sum += d_metrics["gp_loss"]
            wd_sum += d_metrics["wasserstein_dist"]
            num_d_steps += 1
            critic_step += 1

            # Train generator every n_critic steps
            if critic_step >= self.n_critic:
                g_metrics = self.train_generator_step(batch_size)
                g_loss_sum += g_metrics["g_loss"]
                num_g_steps += 1
                critic_step = 0
                self.global_step += 1

            pbar.set_postfix(
                {
                    "d_loss": d_metrics["d_loss"],
                    "wd": d_metrics["wasserstein_dist"],
                }
            )

        metrics = {
            "d_loss": d_loss_sum / max(num_d_steps, 1),
            "g_loss": g_loss_sum / max(num_g_steps, 1),
            "gp_loss": gp_loss_sum / max(num_d_steps, 1),
            "wasserstein_dist": wd_sum / max(num_d_steps, 1),
        }

        _log.info(
            "Epoch %d: d_loss=%.4f g_loss=%.4f wd=%.4f gp=%.4f",
            self.epoch,
            metrics["d_loss"],
            metrics["g_loss"],
            metrics["wasserstein_dist"],
            metrics["gp_loss"],
        )

        # Track best wasserstein distance (lower = generator closer to real)
        epoch_wd = metrics["wasserstein_dist"]
        if epoch_wd < self.best_wasserstein_dist:
            self.best_wasserstein_dist = epoch_wd

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Compute validation metrics for the GAN.

        Evaluates generation quality using FID-style metrics:
        - Mean and std difference between generated and real latent distributions
        - Approximate FID score from distribution moments
        - Discriminator accuracy on real vs fake

        Returns:
            Dictionary with validation metrics including 'fid_approx',
            'mean_diff', 'std_diff', 'disc_accuracy'.
        """
        self.generator.eval()
        self.discriminator.eval()

        # Collect real latents from validation data (limit for efficiency)
        val_loader = getattr(self, 'val_dataloader', None)
        if val_loader is None:
            _log.warning("No val_dataloader set; skipping GAN validation.")
            return {}

        all_real = []
        for batch in val_loader:
            if isinstance(batch, torch.Tensor):
                real = batch.to(self.device)
            elif isinstance(batch, dict):
                real = batch.get(
                    "latents", batch.get("token_ids")
                ).to(self.device).float()
            else:
                continue
            all_real.append(real)
            if len(all_real) >= 10:
                break

        all_real = torch.cat(all_real, dim=0)
        num_samples = all_real.size(0)

        # Generate fake latents
        noise = torch.randn(
            num_samples, self._latent_dim, device=self.device
        )
        all_fake = self.generator(noise)

        # Distribution statistics
        real_mean = all_real.mean(dim=0)
        real_std = all_real.std(dim=0)
        fake_mean = all_fake.mean(dim=0)
        fake_std = all_fake.std(dim=0)

        mean_diff = (real_mean - fake_mean).norm().item()
        std_diff = (real_std - fake_std).norm().item()

        # FID approximation: ||mu_r - mu_f||^2 + ||sigma_r - sigma_f||_F^2
        fid_approx = mean_diff**2 + std_diff**2

        # Discriminator accuracy
        real_scores = self.discriminator(all_real)
        fake_scores = self.discriminator(all_fake)
        real_correct = (real_scores > 0).float().mean().item()
        fake_correct = (fake_scores < 0).float().mean().item()
        disc_accuracy = (real_correct + fake_correct) / 2.0

        metrics = {
            "fid_approx": fid_approx,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "disc_accuracy": disc_accuracy,
        }

        _log.info(
            "Validation: fid_approx=%.4f mean_diff=%.4f "
            "std_diff=%.4f disc_acc=%.4f",
            fid_approx,
            mean_diff,
            std_diff,
            disc_accuracy,
        )

        return metrics

    @torch.no_grad()
    def sample(self, num_samples: int) -> torch.Tensor:
        """Generate latent vectors using the trained generator.

        Args:
            num_samples: Number of latent vectors to generate.

        Returns:
            Tensor of generated latent vectors, shape (num_samples, latent_dim).
        """
        self.generator.eval()
        noise = torch.randn(
            num_samples, self._latent_dim, device=self.device
        )
        generated = self.generator(noise)
        _log.info(
            "Generated %d latent samples of dim %d",
            num_samples,
            generated.shape[-1],
        )
        return generated

    def train(self, num_epochs: int, save_every: int = 1) -> None:
        """Train for multiple epochs.

        Orchestrates the full GAN training loop with checkpointing
        and periodic validation.

        Args:
            num_epochs: Total number of epochs to train.
            save_every: Save a checkpoint every N epochs.
        """
        _log.info("Starting GAN training for %d epochs", num_epochs)

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train one epoch
            epoch_metrics = self.train_epoch()
            self.history["d_loss"].append(epoch_metrics["d_loss"])
            self.history["g_loss"].append(epoch_metrics["g_loss"])
            self.history["gp_loss"].append(epoch_metrics["gp_loss"])
            self.history["wasserstein_dist"].append(
                epoch_metrics["wasserstein_dist"]
            )

            print(
                f"\nEpoch {epoch}: D Loss = {epoch_metrics['d_loss']:.4f}, "
                f"G Loss = {epoch_metrics['g_loss']:.4f}, "
                f"WD = {epoch_metrics['wasserstein_dist']:.4f}"
            )

            # Validate periodically
            if (epoch + 1) % save_every == 0:
                val_metrics = self.validate()
                print(
                    f"Val: FID_approx = {val_metrics['fid_approx']:.4f}, "
                    f"Disc Acc = {val_metrics['disc_accuracy']:.4f}"
                )

                # Update best wasserstein distance and save checkpoint if improved
                epoch_wd = epoch_metrics["wasserstein_dist"]
                if epoch_wd < self.best_wasserstein_dist:
                    self.best_wasserstein_dist = epoch_wd
                    if self.checkpoint_dir:
                        self.save_checkpoint("best_model.pt")
                        _log.info(
                            "Saved best model at epoch %d (wasserstein_dist=%.6f)",
                            epoch,
                            epoch_wd,
                        )

            # Save periodic checkpoint
            if self.checkpoint_dir and (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

        # Save final model and history
        if self.checkpoint_dir:
            self.save_checkpoint("final_model.pt")
            self.save_history()

        _log.info("GAN training complete.")

    def save_checkpoint(self, filename: str) -> None:
        """Save generator and discriminator checkpoint to disk.

        Args:
            filename: Name of the checkpoint file.
        """
        if self.checkpoint_dir is None:
            _log.warning("No checkpoint_dir set; skipping save.")
            return

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_gen_state_dict": self.optimizer_gen.state_dict(),
            "optimizer_disc_state_dict": self.optimizer_disc.state_dict(),
            "best_wasserstein_dist": self.best_wasserstein_dist,
            "history": self.history,
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        _log.info("Checkpoint saved to %s", save_path)

    def load_checkpoint(self, filename: str) -> None:
        """Load generator and discriminator checkpoint from disk.

        Args:
            filename: Name of the checkpoint file to load.
        """
        if self.checkpoint_dir is None:
            raise ValueError(
                "No checkpoint_dir set; cannot load checkpoint."
            )

        load_path = self.checkpoint_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(
            checkpoint["discriminator_state_dict"]
        )
        self.optimizer_gen.load_state_dict(
            checkpoint["optimizer_gen_state_dict"]
        )
        self.optimizer_disc.load_state_dict(
            checkpoint["optimizer_disc_state_dict"]
        )
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_wasserstein_dist = checkpoint.get(
            "best_wasserstein_dist", self.best_wasserstein_dist
        )
        self.history = checkpoint["history"]

        _log.info(
            "Loaded checkpoint from epoch %d (%s)", self.epoch, load_path
        )

    def save_history(self) -> None:
        """Save training history to a JSON file in the checkpoint directory."""
        if self.checkpoint_dir is None:
            _log.warning("No checkpoint_dir set; skipping history save.")
            return

        history_path = self.checkpoint_dir / "gan_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        _log.info("Training history saved to %s", history_path)
