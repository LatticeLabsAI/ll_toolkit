"""Diffusion Trainer for denoising diffusion models on CAD latent sequences.

Implements the training loop for denoising diffusion probabilistic models (DDPM)
with exponential moving average (EMA) of model weights for stable generation.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    _has_torch = True
except ImportError:
    _has_torch = False


class DiffusionTrainer:
    """Trainer for denoising diffusion probabilistic models on CAD data.

    Implements the DDPM training procedure:
    1. Sample random timesteps for each item in the batch
    2. Add noise according to the noise schedule at those timesteps
    3. Train the model to predict the added noise
    4. Maintain an EMA copy of the model for generation

    Args:
        model: Denoising model that takes (noisy_input, timestep) and predicts noise.
        scheduler: Noise scheduler with add_noise() and step() methods, providing
            the beta schedule and noise levels for each timestep.
        train_dataloader: DataLoader for training data.
        val_dataloader: Optional DataLoader for validation data.
        device: Device string. 'auto' selects CUDA if available, else CPU.
        checkpoint_dir: Directory path for saving checkpoints and samples.
        ema_decay: Decay rate for exponential moving average (default 0.9999).
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: Any,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: str = "auto",
        checkpoint_dir: Optional[str] = None,
        ema_decay: float = 0.9999,
    ) -> None:
        if not _has_torch:
            raise ImportError(
                "PyTorch is required for DiffusionTrainer. "
                "Install via conda: conda install pytorch -c conda-forge"
            )

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.ema_decay = ema_decay

        # Default optimizer
        self.optimizer = AdamW(model.parameters(), lr=1e-4)

        # Track optimizer param ids for detecting lazily-added parameters
        self._optimizer_param_ids: set = {
            id(p) for group in self.optimizer.param_groups for p in group['params']
        }

        # EMA model: deep copy of model weights for stable generation
        self.ema_model = copy.deepcopy(model).to(self.device)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        # Infer number of diffusion timesteps from scheduler
        self._num_timesteps = self._get_num_timesteps()

        _log.info(
            "DiffusionTrainer initialized: device=%s, ema_decay=%.6f, "
            "timesteps=%d",
            self.device,
            self.ema_decay,
            self._num_timesteps,
        )

    def _get_num_timesteps(self) -> int:
        """Get the total number of diffusion timesteps from the scheduler.

        Returns:
            Number of timesteps, defaulting to 1000 if not inferrable.
        """
        for attr in (
            "num_timesteps",
            "T",
            "num_train_timesteps",
            "timesteps",
        ):
            if hasattr(self.scheduler, attr):
                val = getattr(self.scheduler, attr)
                if isinstance(val, int):
                    return val
                if hasattr(val, "__len__"):
                    return len(val)

        _log.warning(
            "Could not infer num_timesteps from scheduler; defaulting to 1000."
        )
        return 1000

    def _sync_optimizer_params(self) -> None:
        """Add any lazily-created model parameters to the optimizer."""
        new_params = [
            p for p in self.model.parameters()
            if id(p) not in self._optimizer_param_ids and p.requires_grad
        ]
        if new_params:
            self.optimizer.add_param_group({'params': new_params})
            self._optimizer_param_ids.update(id(p) for p in new_params)
            _log.info(
                "Added %d lazily-created parameters to optimizer", len(new_params)
            )

    def _update_ema(self) -> None:
        """Update the EMA model weights.

        Applies exponential moving average:
        ema_param = decay * ema_param + (1 - decay) * model_param

        This provides a smoothed version of the model for more stable generation.
        """
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    model_param.data, alpha=1.0 - self.ema_decay
                )

    def _build_cosine_schedule(self) -> torch.Tensor:
        """Build the DDPM cosine noise schedule (Nichol & Dhariwal, 2021).

        The cosine schedule produces a smoother noise progression than
        a linear schedule, preserving more signal at early timesteps and
        destroying information more gradually throughout the process.

        Returns:
            1-D tensor of cumulative alpha-bar values, length
            ``self._num_timesteps``.
        """
        import math

        steps = self._num_timesteps
        s = 0.008  # offset to prevent beta_t being too small at t=0
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((t / steps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]  # normalise so alpha_bar[0] = 1
        return alpha_bar[1:].float()  # drop the extra leading entry

    def _add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to clean data according to the scheduler's noise schedule.

        Delegates to the scheduler's ``add_noise`` method if available,
        otherwise uses a **DDPM cosine schedule** (Nichol & Dhariwal 2021)
        which produces smoother noise transitions than a linear fallback.

        Args:
            clean: Clean input data, shape (batch, ...).
            noise: Gaussian noise with same shape as clean.
            timesteps: Integer timesteps for each batch element, shape (batch,).

        Returns:
            Noisy version of the input data.
        """
        if hasattr(self.scheduler, "add_noise"):
            return self.scheduler.add_noise(clean, noise, timesteps)

        # Fallback: DDPM cosine schedule (cached on first call)
        if not hasattr(self, "_alpha_bar_cache"):
            self._alpha_bar_cache = self._build_cosine_schedule().to(
                clean.device
            )

        alpha_bar = self._alpha_bar_cache.to(clean.device)
        t_clamped = timesteps.clamp(0, len(alpha_bar) - 1).long()
        alpha_t = alpha_bar[t_clamped]  # (batch,)

        # Reshape for broadcasting
        while alpha_t.dim() < clean.dim():
            alpha_t = alpha_t.unsqueeze(-1)

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        noisy = alpha_t.sqrt() * clean + (1.0 - alpha_t).sqrt() * noise
        return noisy

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch of denoising diffusion.

        For each batch:
        1. Sample random timesteps uniformly
        2. Sample Gaussian noise
        3. Create noisy versions of the input
        4. Predict the noise with the model
        5. Compute MSE between predicted and actual noise
        6. Update EMA model

        Returns:
            Dictionary with keys: 'loss', 'noise_mse'.
        """
        self.model.train()
        total_loss = 0.0
        total_noise_mse = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_dataloader,
            desc=f"Diffusion Epoch {self.epoch}",
        )

        for batch in pbar:
            # Extract input data
            if isinstance(batch, torch.Tensor):
                clean = batch.to(self.device)
            elif isinstance(batch, dict):
                clean = batch.get(
                    "latents", batch.get("token_ids")
                ).to(self.device).float()
            else:
                raise TypeError(
                    f"Unsupported batch type: {type(batch)}. "
                    "Expected Tensor or dict."
                )

            batch_size = clean.size(0)

            # Sample random timesteps for each element in the batch
            timesteps = torch.randint(
                0, self._num_timesteps, (batch_size,), device=self.device
            ).long()

            # Sample noise
            noise = torch.randn_like(clean)

            # Create noisy input
            noisy = self._add_noise(clean, noise, timesteps)

            # Predict noise
            noise_pred = self.model(noisy, timesteps)

            # Compute loss: MSE between predicted and actual noise (mean over all elements)
            loss = F.mse_loss(noise_pred, noise)

            # Per-sample noise MSE: average of per-sample MSE values
            with torch.no_grad():
                per_sample_mse = F.mse_loss(
                    noise_pred, noise, reduction="none"
                )
                # Average over all dims except batch, then mean over batch
                while per_sample_mse.dim() > 1:
                    per_sample_mse = per_sample_mse.mean(dim=-1)
                noise_mse = per_sample_mse.mean().item()

            # Sync lazily-created parameters before backward pass
            self._sync_optimizer_params()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()

            # Update EMA model
            self._update_ema()

            # Accumulate metrics
            total_loss += loss.item()
            total_noise_mse += noise_mse
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({"loss": loss.item(), "noise_mse": noise_mse})

        n = max(num_batches, 1)
        metrics = {
            "loss": total_loss / n,
            "noise_mse": total_noise_mse / n,
        }

        _log.info(
            "Epoch %d train: loss=%.6f",
            self.epoch,
            metrics["loss"],
        )

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation on the diffusion model.

        Computes noise prediction MSE on the validation set using both
        the training model and the EMA model.

        Returns:
            Dictionary with keys: 'val_loss', 'val_noise_mse', 'ema_val_loss'.
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_ema_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Diffusion Validation"):
            if isinstance(batch, torch.Tensor):
                clean = batch.to(self.device)
            elif isinstance(batch, dict):
                clean = batch.get(
                    "latents", batch.get("token_ids")
                ).to(self.device).float()
            else:
                continue

            batch_size = clean.size(0)
            timesteps = torch.randint(
                0, self._num_timesteps, (batch_size,), device=self.device
            ).long()
            noise = torch.randn_like(clean)
            noisy = self._add_noise(clean, noise, timesteps)

            # Model prediction
            noise_pred = self.model(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()

            # EMA model prediction
            ema_noise_pred = self.ema_model(noisy, timesteps)
            ema_loss = F.mse_loss(ema_noise_pred, noise)
            total_ema_loss += ema_loss.item()

            num_batches += 1

        n = max(num_batches, 1)
        metrics = {
            "val_loss": total_loss / n,
            "ema_val_loss": total_ema_loss / n,
        }

        _log.info(
            "Epoch %d val: loss=%.6f ema_loss=%.6f",
            self.epoch,
            metrics["val_loss"],
            metrics["ema_val_loss"],
        )

        return metrics

    @torch.no_grad()
    def sample_and_visualize(self, num_samples: int, epoch: int) -> None:
        """Generate samples using the EMA model and save visualizations.

        Runs the full reverse diffusion process starting from pure noise
        and saves the resulting samples and a visualization to the checkpoint
        directory.

        Args:
            num_samples: Number of samples to generate.
            epoch: Current epoch number for labeling the output.
        """
        if self.checkpoint_dir is None:
            _log.warning("No checkpoint_dir; skipping sample visualization.")
            return

        self.ema_model.eval()

        # Infer sample shape from training data
        sample_batch = next(iter(self.train_dataloader))
        if isinstance(sample_batch, torch.Tensor):
            sample_shape = sample_batch.shape[1:]
        elif isinstance(sample_batch, dict):
            ref = sample_batch.get("latents", sample_batch.get("token_ids"))
            sample_shape = ref.shape[1:]
        else:
            _log.warning("Cannot infer sample shape; skipping visualization.")
            return

        # Start from pure noise
        x = torch.randn(num_samples, *sample_shape, device=self.device)

        # Reverse diffusion: iterate from T-1 to 0
        for t in reversed(range(self._num_timesteps)):
            t_batch = torch.full(
                (num_samples,), t, device=self.device, dtype=torch.long
            )
            noise_pred = self.ema_model(x, t_batch)

            # Use scheduler step if available
            if hasattr(self.scheduler, "step"):
                scheduler_output = self.scheduler.step(noise_pred, t, x)
                if hasattr(scheduler_output, "prev_sample"):
                    x = scheduler_output.prev_sample
                elif isinstance(scheduler_output, torch.Tensor):
                    x = scheduler_output
                else:
                    x = scheduler_output
            else:
                # Fallback: simple denoising step
                alpha_t = 1.0 - t / self._num_timesteps
                alpha_t = max(alpha_t, 1e-6)
                x = (x - (1.0 - alpha_t) * noise_pred) / (alpha_t**0.5)

                # Add noise for all steps except t=0
                if t > 0:
                    beta_t = 1.0 - alpha_t
                    noise = torch.randn_like(x)
                    x = x + (beta_t**0.5) * noise

        # Save generated samples as tensor
        samples_path = self.checkpoint_dir / f"samples_epoch_{epoch}.pt"
        torch.save(x.cpu(), samples_path)
        _log.info(
            "Saved %d generated samples to %s", num_samples, samples_path
        )

        # Visualize: plot first few dimensions of generated samples
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            num_plots = min(4, num_samples)
            fig, axes = plt.subplots(1, num_plots, figsize=(16, 4))
            if num_plots == 1:
                axes = [axes]

            samples_np = x.cpu().numpy()
            for i, ax in enumerate(axes):
                if i >= num_samples:
                    break
                sample = samples_np[i]
                if sample.ndim == 1:
                    ax.plot(sample)
                    ax.set_title(f"Sample {i}")
                else:
                    # For 2D+ data, show first dim flattened
                    ax.plot(sample.flatten()[:256])
                    ax.set_title(f"Sample {i} (first 256 values)")
                ax.set_xlabel("Index")
                ax.set_ylabel("Value")

            fig.suptitle(f"Generated Samples - Epoch {epoch}")
            fig.tight_layout()

            viz_path = (
                self.checkpoint_dir / f"samples_viz_epoch_{epoch}.png"
            )
            fig.savefig(viz_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            _log.info("Sample visualization saved to %s", viz_path)
        except ImportError:
            _log.warning(
                "matplotlib not available; skipping sample visualization plot."
            )

    def train(self, num_epochs: int, save_every: int = 1) -> None:
        """Train for multiple epochs with EMA updates and periodic sampling.

        Orchestrates the full diffusion training loop:
        - Per-epoch noise prediction training
        - EMA model updates each step
        - Validation and sample generation
        - Checkpointing

        Args:
            num_epochs: Total number of epochs to train.
            save_every: Save a checkpoint and generate samples every N epochs.
        """
        _log.info("Starting diffusion training for %d epochs", num_epochs)

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train one epoch
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["loss"])

            print(
                f"\nEpoch {epoch}: Loss = {train_metrics['loss']:.6f}"
            )

            # Validate
            if self.val_dataloader:
                val_metrics = self.validate()
                self.history["val_loss"].append(
                    val_metrics.get("val_loss", 0.0)
                )

                print(
                    f"Val Loss = {val_metrics['val_loss']:.6f}, "
                    f"EMA Val Loss = "
                    f"{val_metrics.get('ema_val_loss', 0.0):.6f}"
                )

                # Save best model based on EMA validation loss
                val_loss = val_metrics.get(
                    "ema_val_loss", val_metrics["val_loss"]
                )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.checkpoint_dir:
                        self.save_checkpoint("best_model.pt")
                        print(
                            f"Saved best model (val_loss={val_loss:.6f})"
                        )

            # Periodic checkpoint and sample generation
            if (epoch + 1) % save_every == 0:
                if self.checkpoint_dir:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
                self.sample_and_visualize(num_samples=4, epoch=epoch)

        # Save final model and history
        if self.checkpoint_dir:
            self.save_checkpoint("final_model.pt")
            self.save_history()

        _log.info(
            "Diffusion training complete. Best val loss: %.6f",
            self.best_val_loss,
        )

    def save_checkpoint(self, filename: str) -> None:
        """Save model and EMA model checkpoint to disk.

        Args:
            filename: Name of the checkpoint file.
        """
        if self.checkpoint_dir is None:
            _log.warning("No checkpoint_dir set; skipping save.")
            return

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "ema_decay": self.ema_decay,
            "history": self.history,
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        _log.info("Checkpoint saved to %s", save_path)

    def load_checkpoint(self, filename: str) -> None:
        """Load model and EMA model checkpoint from disk.

        Args:
            filename: Name of the checkpoint file to load.
        """
        if self.checkpoint_dir is None:
            raise ValueError(
                "No checkpoint_dir set; cannot load checkpoint."
            )

        load_path = self.checkpoint_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.ema_decay = checkpoint.get("ema_decay", self.ema_decay)
        self.history = checkpoint["history"]

        _log.info(
            "Loaded checkpoint from epoch %d (%s)", self.epoch, load_path
        )

    def save_history(self) -> None:
        """Save training history to a JSON file in the checkpoint directory."""
        if self.checkpoint_dir is None:
            _log.warning("No checkpoint_dir set; skipping history save.")
            return

        history_path = self.checkpoint_dir / "diffusion_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        _log.info("Training history saved to %s", history_path)
