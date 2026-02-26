"""Streaming Diffusion Trainer for HuggingFace IterableDataset.

Extends DiffusionTrainer with support for streaming datasets, step-based
scheduling, and mid-stream checkpointing for petabyte-scale training.
Combines the streaming patterns from StreamingVAETrainer with the EMA
and noise scheduling from DiffusionTrainer.
"""
from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from stepnet.training.streaming_utils import build_dataset_from_config, create_cosine_scheduler

_log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from tqdm import tqdm

    _has_torch = True
except ImportError:
    _has_torch = False


class StreamingDiffusionTrainer:
    """Trainer for diffusion models with streaming dataset support.

    Combines the step-based training loop from StreamingVAETrainer with
    the denoising diffusion training from DiffusionTrainer. Key features:

    - Step-based scheduling: Uses total_steps instead of epochs
    - EMA model maintenance: Updates EMA weights per step for stable generation
    - Noise scheduling: Configurable noise schedule with step-based warmup
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
        >>> trainer = StreamingDiffusionTrainer(
        ...     model=diffusion_model,
        ...     scheduler=noise_scheduler,
        ...     dataset=dataset,
        ...     total_steps=100000,
        ...     warmup_steps=1000,
        ...     ema_decay=0.9999,
        ... )
        >>> trainer.train()

    Args:
        model: Denoising model that takes (noisy_input, timestep) and predicts noise.
        scheduler: Noise scheduler with add_noise() method, providing the beta
            schedule and noise levels for each timestep.
        dataset: Streaming dataset with __iter__() method.
        val_dataset: Optional validation dataset.
        total_steps: Total training steps.
        warmup_steps: Steps for learning rate warmup.
        ema_decay: Decay rate for exponential moving average (default 0.9999).
        optimizer: Optional optimizer (creates AdamW if None).
        lr_scheduler: Optional LR scheduler.
        device: Device string ('auto' selects CUDA if available).
        checkpoint_dir: Directory for saving checkpoints.
        log_every: Log metrics every N steps.
        eval_every: Run validation every N steps.
        save_every: Save checkpoint every N steps.
        sample_every: Generate samples every N steps.
        gradient_accumulation_steps: Accumulate gradients over N steps.
        max_grad_norm: Maximum gradient norm for clipping.
        learning_rate: Learning rate for optimizer.
    """

    def __init__(
        self,
        model: "nn.Module",
        scheduler: Any,
        dataset: Any = None,
        val_dataset: Optional[Any] = None,
        total_steps: int = 100000,
        warmup_steps: int = 1000,
        ema_decay: float = 0.9999,
        optimizer: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
        device: str = "auto",
        checkpoint_dir: Optional[str] = None,
        log_every: int = 100,
        eval_every: int = 5000,
        save_every: int = 10000,
        sample_every: int = 10000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        learning_rate: float = 1e-4,
        dataset_config: Optional[Any] = None,
    ) -> None:
        if not _has_torch:
            raise ImportError(
                "PyTorch is required for StreamingDiffusionTrainer. "
                "Install via: conda install pytorch -c conda-forge"
            )

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.scheduler = scheduler

        # Build dataset from StreamingCadlingConfig if provided
        if dataset_config is not None and dataset is None:
            dataset = self._build_dataset_from_config(dataset_config)
        elif dataset is None:
            raise ValueError(
                "Either 'dataset' or 'dataset_config' must be provided."
            )

        self.dataset = dataset
        self.val_dataset = val_dataset
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.ema_decay = ema_decay
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.sample_every = sample_every
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Default optimizer
        if optimizer is None:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        # Default scheduler: cosine decay with warmup
        if lr_scheduler is None:
            self.lr_scheduler = self._create_scheduler()
        else:
            self.lr_scheduler = lr_scheduler

        # EMA model: deep copy for stable generation
        self.ema_model = copy.deepcopy(model).to(self.device)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "noise_mse": [],
            "val_loss": [],
            "val_noise_mse": [],
            "ema_val_loss": [],
            "learning_rate": [],
        }

        # Running averages for logging
        self._running_loss = 0.0
        self._running_count = 0

        # Infer number of timesteps from scheduler
        self._num_timesteps = self._get_num_timesteps()

        _log.info(
            "StreamingDiffusionTrainer initialized: device=%s, total_steps=%d, "
            "warmup_steps=%d, ema_decay=%.6f, timesteps=%d",
            self.device,
            self.total_steps,
            self.warmup_steps,
            self.ema_decay,
            self._num_timesteps,
        )

    def _create_scheduler(self) -> "LambdaLR":
        """Create a learning rate scheduler with linear warmup and cosine decay."""
        return create_cosine_scheduler(self.optimizer, self.warmup_steps, self.total_steps)

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

    def _get_num_timesteps(self) -> int:
        """Get the total number of diffusion timesteps from the scheduler."""
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

    def _update_ema(self) -> None:
        """Update the EMA model weights.

        Applies exponential moving average:
        ema_param = decay * ema_param + (1 - decay) * model_param
        """
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    model_param.data, alpha=1.0 - self.ema_decay
                )

    def _add_noise(
        self,
        clean: "torch.Tensor",
        noise: "torch.Tensor",
        timesteps: "torch.Tensor",
    ) -> "torch.Tensor":
        """Add noise to clean data according to the scheduler's noise schedule."""
        if hasattr(self.scheduler, "add_noise"):
            return self.scheduler.add_noise(clean, noise, timesteps)

        # Fallback: DDPM cosine schedule (cached, matches non-streaming DiffusionTrainer)
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

    def _build_cosine_schedule(self) -> "torch.Tensor":
        """Build the DDPM cosine noise schedule (Nichol & Dhariwal, 2021).

        Returns:
            1-D tensor of cumulative alpha-bar values, length
            ``self._num_timesteps``.
        """
        steps = self._num_timesteps
        s = 0.008  # offset to prevent beta_t being too small at t=0
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((t / steps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]  # normalise so alpha_bar[0] = 1
        return alpha_bar[1:].float()  # drop the extra leading entry

    def _prepare_batch(
        self, batch: Dict[str, Any]
    ) -> "torch.Tensor":
        """Prepare a batch for training.

        Args:
            batch: Dictionary with 'latents' or 'token_ids'.

        Returns:
            Clean tensor for denoising.
        """
        # Get input data
        if "latents" in batch:
            clean = batch["latents"]
        elif "token_ids" in batch:
            clean = batch["token_ids"]
        elif "command_types" in batch:
            clean = batch["command_types"]
        else:
            raise ValueError(
                "Batch must contain 'latents', 'token_ids', or 'command_types'"
            )

        # Ensure tensor
        if not isinstance(clean, torch.Tensor):
            clean = torch.tensor(clean)

        return clean.to(self.device).float()

    def train_step(
        self, batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Training batch.

        Returns:
            Dictionary with step metrics.
        """
        self.model.train()

        clean = self._prepare_batch(batch)
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

        # Compute loss: MSE between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)

        # Scale loss for gradient accumulation
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        # Update weights every gradient_accumulation_steps
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        # Update EMA model every step
        self._update_ema()

        self.global_step += 1

        return {
            "loss": loss.item(),
            "noise_mse": loss.item(),
            "lr": self.lr_scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation on the validation dataset.

        Returns:
            Dictionary with validation metrics.
        """
        if self.val_dataset is None:
            return {}

        self.model.eval()
        self.ema_model.eval()

        total_loss = 0.0
        total_ema_loss = 0.0
        num_batches = 0

        # Iterate through validation dataset
        val_iter = iter(self.val_dataset)

        for batch in val_iter:
            clean = self._prepare_batch(batch)
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

            # Limit validation batches for streaming
            if num_batches >= 100:
                break

        n = max(num_batches, 1)
        metrics = {
            "val_loss": total_loss / n,
            "val_noise_mse": total_loss / n,
            "ema_val_loss": total_ema_loss / n,
        }

        _log.info(
            "Step %d validation: loss=%.6f ema_loss=%.6f",
            self.global_step,
            metrics["val_loss"],
            metrics["ema_val_loss"],
        )

        return metrics

    @torch.no_grad()
    def sample_and_visualize(self, num_samples: int = 4) -> None:
        """Generate samples using the EMA model and save visualizations."""
        if self.checkpoint_dir is None:
            _log.warning("No checkpoint_dir; skipping sample visualization.")
            return

        self.ema_model.eval()

        # Infer sample shape from a training batch
        try:
            sample_batch = next(iter(self.dataset))
            clean = self._prepare_batch(sample_batch)
            sample_shape = clean.shape[1:]
        except StopIteration:
            _log.warning("Cannot get sample shape from dataset.")
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

        # Save generated samples
        samples_path = self.checkpoint_dir / f"samples_step_{self.global_step}.pt"
        torch.save(x.cpu(), samples_path)
        _log.info("Saved %d generated samples to %s", num_samples, samples_path)

    def train(self) -> None:
        """Run the full training loop.

        Iterates through the streaming dataset, training until total_steps
        is reached. Supports resuming from checkpoints and handles
        epoch boundaries for streaming datasets.
        """
        _log.info("Starting streaming diffusion training for %d steps", self.total_steps)

        self.optimizer.zero_grad()
        pbar = tqdm(total=self.total_steps, desc="Diffusion Training", initial=self.global_step)

        while self.global_step < self.total_steps:
            # Set epoch for reproducible shuffling
            if hasattr(self.dataset, "set_epoch"):
                self.dataset.set_epoch(self.epoch)

            # Iterate through the dataset
            data_iter = iter(self.dataset)

            for batch in data_iter:
                if self.global_step >= self.total_steps:
                    break

                # Training step
                metrics = self.train_step(batch)

                # Update running averages
                self._running_loss += metrics["loss"]
                self._running_count += 1

                # Logging
                if self.global_step % self.log_every == 0:
                    avg_loss = self._running_loss / max(self._running_count, 1)

                    self.history["train_loss"].append(avg_loss)
                    self.history["noise_mse"].append(avg_loss)
                    self.history["learning_rate"].append(metrics["lr"])

                    pbar.set_postfix({
                        "loss": f"{avg_loss:.6f}",
                        "lr": f"{metrics['lr']:.2e}",
                    })

                    # Reset running averages
                    self._running_loss = 0.0
                    self._running_count = 0

                # Validation
                if self.global_step % self.eval_every == 0 and self.val_dataset:
                    val_metrics = self.validate()
                    self.history["val_loss"].append(val_metrics.get("val_loss", 0.0))
                    self.history["val_noise_mse"].append(
                        val_metrics.get("val_noise_mse", 0.0)
                    )
                    self.history["ema_val_loss"].append(
                        val_metrics.get("ema_val_loss", 0.0)
                    )

                    # Save best model based on EMA validation loss
                    val_loss = val_metrics.get("ema_val_loss", val_metrics.get("val_loss", float("inf")))
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        if self.checkpoint_dir:
                            self.save_checkpoint("best_model.pt")
                            _log.info(
                                "Saved best model at step %d (val_loss=%.6f)",
                                self.global_step,
                                val_loss,
                            )

                # Sample generation
                if self.global_step % self.sample_every == 0:
                    self.sample_and_visualize()

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

        _log.info(
            "Training complete. Best val loss: %.6f",
            self.best_val_loss,
        )

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        if self.checkpoint_dir is None:
            _log.warning("No checkpoint_dir set; skipping save.")
            return

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "ema_decay": self.ema_decay,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "history": self.history,
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        _log.info("Saved checkpoint to %s", save_path)

    def load_checkpoint(self, filename: str) -> None:
        """Load training checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint_dir set; cannot load checkpoint.")

        load_path = self.checkpoint_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.ema_decay = checkpoint.get("ema_decay", self.ema_decay)
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

        history_path = self.checkpoint_dir / "streaming_diffusion_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        _log.info("Saved history to %s", history_path)


__all__ = ["StreamingDiffusionTrainer"]
