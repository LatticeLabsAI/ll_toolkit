"""Streaming GAN Trainer for HuggingFace IterableDataset.

Extends GANTrainer with support for streaming datasets, step-based
scheduling, and mid-stream checkpointing for petabyte-scale training.
Combines the streaming patterns from StreamingVAETrainer with the
WGAN-GP training from GANTrainer.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from stepnet.training.streaming_utils import build_dataset_from_config

_log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from tqdm import tqdm

    _has_torch = True
except ImportError:
    _has_torch = False


class StreamingGANTrainer:
    """Trainer for WGAN-GP models with streaming dataset support.

    Combines the step-based training loop from StreamingVAETrainer with
    the Wasserstein GAN with gradient penalty from GANTrainer. Key features:

    - Step-based scheduling: Uses total_steps instead of epochs
    - Alternating critic/generator updates: n_critic critic steps per generator step
    - Gradient penalty: WGAN-GP for stable training
    - Mid-stream checkpointing: Save/resume within a streaming epoch
    - Epoch-based shuffle seed: set_epoch() for reproducible shuffling

    Usage:
        >>> from cadling.data.streaming import CADStreamingDataset, CADStreamingConfig
        >>>
        >>> config = CADStreamingConfig(
        ...     dataset_id="latticelabs/deepcad-latents",
        ...     batch_size=8,
        ... )
        >>> dataset = CADStreamingDataset(config)
        >>>
        >>> trainer = StreamingGANTrainer(
        ...     generator=generator_model,
        ...     critic=critic_model,
        ...     dataset=dataset,
        ...     total_steps=100000,
        ...     n_critic=5,
        ...     lambda_gp=10.0,
        ... )
        >>> trainer.train()

    Args:
        generator: Generator network mapping noise -> latent vectors.
        critic: Critic (discriminator) network scoring latent vectors.
        dataset: Streaming dataset with __iter__() method.
        total_steps: Total training steps (generator updates).
        warmup_steps: Steps for learning rate warmup.
        n_critic: Number of critic updates per generator update (default 5).
        lambda_gp: Gradient penalty coefficient (default 10.0 per WGAN-GP paper).
        optimizer_gen: Optional generator optimizer.
        optimizer_critic: Optional critic optimizer.
        device: Device string ('auto' selects CUDA if available).
        checkpoint_dir: Directory for saving checkpoints.
        log_every: Log metrics every N steps.
        eval_every: Run validation every N steps.
        save_every: Save checkpoint every N steps.
        sample_every: Generate samples every N steps.
        max_grad_norm: Maximum gradient norm for clipping.
        lr_gen: Learning rate for generator.
        lr_critic: Learning rate for critic.
    """

    def __init__(
        self,
        generator: "nn.Module",
        critic: "nn.Module",
        dataset: Any = None,
        total_steps: int = 100000,
        warmup_steps: int = 1000,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        optimizer_gen: Optional[Any] = None,
        optimizer_critic: Optional[Any] = None,
        device: str = "auto",
        checkpoint_dir: Optional[str] = None,
        log_every: int = 100,
        eval_every: int = 5000,
        save_every: int = 10000,
        sample_every: int = 10000,
        max_grad_norm: float = 1.0,
        lr_gen: float = 1e-4,
        lr_critic: float = 1e-4,
        dataset_config: Optional[Any] = None,
    ) -> None:
        if not _has_torch:
            raise ImportError(
                "PyTorch is required for StreamingGANTrainer. "
                "Install via: conda install pytorch -c conda-forge"
            )

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)

        # Build dataset from StreamingCadlingConfig if provided
        if dataset_config is not None and dataset is None:
            dataset = self._build_dataset_from_config(dataset_config)
        elif dataset is None:
            raise ValueError(
                "Either 'dataset' or 'dataset_config' must be provided."
            )

        self.dataset = dataset
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.sample_every = sample_every
        self.max_grad_norm = max_grad_norm

        # WGAN-GP uses specific optimizer settings
        if optimizer_gen is None:
            self.optimizer_gen = AdamW(
                generator.parameters(), lr=lr_gen, betas=(0.0, 0.9)
            )
        else:
            self.optimizer_gen = optimizer_gen

        if optimizer_critic is None:
            self.optimizer_critic = AdamW(
                critic.parameters(), lr=lr_critic, betas=(0.0, 0.9)
            )
        else:
            self.optimizer_critic = optimizer_critic

        # Create schedulers with warmup
        self.scheduler_gen = self._create_scheduler(self.optimizer_gen)
        self.scheduler_critic = self._create_scheduler(self.optimizer_critic)

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0  # Generator update steps
        self.critic_step = 0  # Critic update counter within each generator step
        self.epoch = 0
        self.best_wasserstein_dist = float("-inf")  # Higher is better for Wasserstein
        self.history: Dict[str, List[float]] = {
            "d_loss": [],
            "g_loss": [],
            "gp_loss": [],
            "wasserstein_dist": [],
            "learning_rate": [],
        }

        # Running averages for logging
        self._running_d_loss = 0.0
        self._running_g_loss = 0.0
        self._running_gp = 0.0
        self._running_wd = 0.0
        self._running_d_count = 0
        self._running_g_count = 0

        # Infer latent dimension from generator
        self._latent_dim = self._infer_latent_dim()

        _log.info(
            "StreamingGANTrainer initialized: device=%s, total_steps=%d, "
            "n_critic=%d, lambda_gp=%.1f, latent_dim=%d",
            self.device,
            self.total_steps,
            self.n_critic,
            self.lambda_gp,
            self._latent_dim,
        )

    def _create_scheduler(self, optimizer: Any) -> "LambdaLR":
        """Create a learning rate scheduler with linear warmup and constant decay."""

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                # Linear warmup
                return step / max(self.warmup_steps, 1)
            else:
                # Constant after warmup (WGAN-GP typically uses constant LR)
                return 1.0

        return LambdaLR(optimizer, lr_lambda)

    def _infer_latent_dim(self) -> int:
        """Infer the latent/noise dimension from the generator architecture."""
        for attr in ("latent_dim", "noise_dim", "z_dim", "input_dim"):
            if hasattr(self.generator, attr):
                return getattr(self.generator, attr)

        # Try inspecting first linear layer
        for module in self.generator.modules():
            if isinstance(module, nn.Linear):
                return module.in_features

        _log.warning("Could not infer latent dim; defaulting to 128.")
        return 128

    @staticmethod
    def _build_dataset_from_config(dataset_config) -> Any:
        """Build a streaming dataset from a StreamingCadlingConfig.

        Delegates to :func:`streaming_utils.build_dataset_from_config`.

        Args:
            dataset_config: A ``StreamingCadlingConfig`` instance.

        Returns:
            A ``CADStreamingDataset`` ready for iteration.
        """
        return build_dataset_from_config(dataset_config)

    def _prepare_batch(
        self, batch: Dict[str, Any]
    ) -> "torch.Tensor":
        """Prepare a batch for training.

        Args:
            batch: Dictionary with 'latents' or 'token_ids'.

        Returns:
            Real latent tensor.
        """
        if "latents" in batch:
            real = batch["latents"]
        elif "token_ids" in batch:
            real = batch["token_ids"]
        elif "command_types" in batch:
            real = batch["command_types"]
        else:
            raise ValueError(
                "Batch must contain 'latents', 'token_ids', or 'command_types'"
            )

        if not isinstance(real, torch.Tensor):
            real = torch.tensor(real)

        return real.to(self.device).float()

    def _compute_gradient_penalty(
        self, real: "torch.Tensor", fake: "torch.Tensor"
    ) -> "torch.Tensor":
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
        critic_interpolated = self.critic(interpolated)

        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1.0) ** 2).mean()

        return gradient_penalty

    def train_critic_step(
        self, real_latents: "torch.Tensor"
    ) -> Dict[str, float]:
        """Perform one critic training step.

        Args:
            real_latents: Real latent vectors.

        Returns:
            Dictionary with critic metrics.
        """
        batch_size = real_latents.size(0)
        self.critic.train()
        self.optimizer_critic.zero_grad()

        # Generate fake latents
        noise = torch.randn(batch_size, self._latent_dim, device=self.device)
        fake_latents = self.generator(noise).detach()

        # Critic scores
        real_score = self.critic(real_latents)
        fake_score = self.critic(fake_latents)

        # WGAN loss: maximize D(real) - D(fake) => minimize D(fake) - D(real)
        wasserstein_dist = real_score.mean() - fake_score.mean()
        d_loss = -wasserstein_dist

        # Gradient penalty
        gp = self._compute_gradient_penalty(real_latents, fake_latents)
        total_d_loss = d_loss + self.lambda_gp * gp

        total_d_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), max_norm=self.max_grad_norm
        )
        self.optimizer_critic.step()
        self.scheduler_critic.step()

        return {
            "d_loss": total_d_loss.item(),
            "gp_loss": gp.item(),
            "wasserstein_dist": wasserstein_dist.item(),
        }

    def train_generator_step(self, batch_size: int) -> Dict[str, float]:
        """Perform one generator training step.

        Args:
            batch_size: Number of samples to generate.

        Returns:
            Dictionary with generator metrics.
        """
        self.generator.train()
        self.optimizer_gen.zero_grad()

        # Generate fake latents
        noise = torch.randn(batch_size, self._latent_dim, device=self.device)
        fake_latents = self.generator(noise)

        # Generator wants to maximize D(fake) => minimize -D(fake)
        fake_score = self.critic(fake_latents)
        g_loss = -fake_score.mean()

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(), max_norm=self.max_grad_norm
        )
        self.optimizer_gen.step()
        self.scheduler_gen.step()

        self.global_step += 1

        return {
            "g_loss": g_loss.item(),
            "lr": self.scheduler_gen.get_last_lr()[0],
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Compute validation metrics for the GAN.

        Returns:
            Dictionary with validation metrics.
        """
        self.generator.eval()
        self.critic.eval()

        # Collect real latents from training data
        all_real = []
        data_iter = iter(self.dataset)

        for batch in data_iter:
            real = self._prepare_batch(batch)
            all_real.append(real)
            if len(all_real) >= 10:
                break

        if not all_real:
            return {}

        all_real = torch.cat(all_real, dim=0)
        num_samples = all_real.size(0)

        # Generate fake latents
        noise = torch.randn(num_samples, self._latent_dim, device=self.device)
        all_fake = self.generator(noise)

        # Distribution statistics
        real_mean = all_real.mean(dim=0)
        real_std = all_real.std(dim=0)
        fake_mean = all_fake.mean(dim=0)
        fake_std = all_fake.std(dim=0)

        mean_diff = (real_mean - fake_mean).norm().item()
        std_diff = (real_std - fake_std).norm().item()

        # FID approximation
        fid_approx = mean_diff**2 + std_diff**2

        # Discriminator accuracy
        real_scores = self.critic(all_real)
        fake_scores = self.critic(all_fake)
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
            "Step %d validation: fid_approx=%.4f disc_acc=%.4f",
            self.global_step,
            fid_approx,
            disc_accuracy,
        )

        return metrics

    @torch.no_grad()
    def sample(self, num_samples: int = 16) -> "torch.Tensor":
        """Generate latent vectors using the trained generator.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Generated latent vectors.
        """
        self.generator.eval()
        noise = torch.randn(num_samples, self._latent_dim, device=self.device)
        generated = self.generator(noise)
        return generated

    def train(self) -> None:
        """Run the full training loop.

        Iterates through the streaming dataset, training until total_steps
        is reached. Uses n_critic critic updates per generator update.
        """
        _log.info("Starting streaming GAN training for %d steps", self.total_steps)

        pbar = tqdm(total=self.total_steps, desc="GAN Training", initial=self.global_step)
        critic_updates = 0

        while self.global_step < self.total_steps:
            # Set epoch for reproducible shuffling
            if hasattr(self.dataset, "set_epoch"):
                self.dataset.set_epoch(self.epoch)

            # Iterate through the dataset
            data_iter = iter(self.dataset)

            for batch in data_iter:
                if self.global_step >= self.total_steps:
                    break

                real_latents = self._prepare_batch(batch)
                batch_size = real_latents.size(0)

                # Train critic
                d_metrics = self.train_critic_step(real_latents)
                self._running_d_loss += d_metrics["d_loss"]
                self._running_gp += d_metrics["gp_loss"]
                self._running_wd += d_metrics["wasserstein_dist"]
                self._running_d_count += 1
                critic_updates += 1

                # Train generator every n_critic steps
                if critic_updates >= self.n_critic:
                    g_metrics = self.train_generator_step(batch_size)
                    self._running_g_loss += g_metrics["g_loss"]
                    self._running_g_count += 1
                    critic_updates = 0

                    # Logging
                    if self.global_step % self.log_every == 0:
                        avg_d = self._running_d_loss / max(self._running_d_count, 1)
                        avg_g = self._running_g_loss / max(self._running_g_count, 1)
                        avg_gp = self._running_gp / max(self._running_d_count, 1)
                        avg_wd = self._running_wd / max(self._running_d_count, 1)

                        self.history["d_loss"].append(avg_d)
                        self.history["g_loss"].append(avg_g)
                        self.history["gp_loss"].append(avg_gp)
                        self.history["wasserstein_dist"].append(avg_wd)
                        self.history["learning_rate"].append(g_metrics["lr"])

                        pbar.set_postfix({
                            "d_loss": f"{avg_d:.4f}",
                            "g_loss": f"{avg_g:.4f}",
                            "wd": f"{avg_wd:.4f}",
                        })

                        # Reset running averages
                        self._running_d_loss = 0.0
                        self._running_g_loss = 0.0
                        self._running_gp = 0.0
                        self._running_wd = 0.0
                        self._running_d_count = 0
                        self._running_g_count = 0

                    # Validation
                    if self.global_step % self.eval_every == 0:
                        val_metrics = self.validate()
                        # Track best Wasserstein distance
                        wd = val_metrics.get("wasserstein_dist", self._running_wd)
                        if wd > self.best_wasserstein_dist:
                            self.best_wasserstein_dist = wd
                            if self.checkpoint_dir:
                                self.save_checkpoint("best_model.pt")

                    # Sample generation
                    if self.global_step % self.sample_every == 0 and self.checkpoint_dir:
                        samples = self.sample(16)
                        samples_path = self.checkpoint_dir / f"samples_step_{self.global_step}.pt"
                        torch.save(samples.cpu(), samples_path)
                        _log.info("Saved samples to %s", samples_path)

                    # Checkpointing
                    if self.global_step % self.save_every == 0 and self.checkpoint_dir:
                        self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

                    pbar.update(1)

            # End of epoch
            self.epoch += 1
            _log.info("Completed epoch %d at step %d", self.epoch, self.global_step)

        pbar.close()

        # Final save
        if self.checkpoint_dir:
            self.save_checkpoint("final_model.pt")
            self.save_history()

        _log.info("GAN training complete.")

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        if self.checkpoint_dir is None:
            _log.warning("No checkpoint_dir set; skipping save.")
            return

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "generator_state_dict": self.generator.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_gen_state_dict": self.optimizer_gen.state_dict(),
            "optimizer_critic_state_dict": self.optimizer_critic.state_dict(),
            "scheduler_gen_state_dict": self.scheduler_gen.state_dict(),
            "scheduler_critic_state_dict": self.scheduler_critic.state_dict(),
            "best_wasserstein_dist": self.best_wasserstein_dist,
            "n_critic": self.n_critic,
            "lambda_gp": self.lambda_gp,
            "total_steps": self.total_steps,
            "history": self.history,
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        _log.info("Saved checkpoint to %s", save_path)

    def load_checkpoint(self, filename: str) -> None:
        """Load training checkpoint."""
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint_dir set; cannot load checkpoint.")

        load_path = self.checkpoint_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer_gen.load_state_dict(checkpoint["optimizer_gen_state_dict"])
        self.optimizer_critic.load_state_dict(checkpoint["optimizer_critic_state_dict"])
        self.scheduler_gen.load_state_dict(checkpoint["scheduler_gen_state_dict"])
        self.scheduler_critic.load_state_dict(checkpoint["scheduler_critic_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_wasserstein_dist = checkpoint.get("best_wasserstein_dist", float("-inf"))
        self.history = checkpoint["history"]

        _log.info(
            "Loaded checkpoint from step %d (%s)",
            self.global_step,
            load_path,
        )

    def save_history(self) -> None:
        """Save training history to JSON."""
        if self.checkpoint_dir is None:
            return

        history_path = self.checkpoint_dir / "streaming_gan_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        _log.info("Saved history to %s", history_path)


__all__ = ["StreamingGANTrainer"]
