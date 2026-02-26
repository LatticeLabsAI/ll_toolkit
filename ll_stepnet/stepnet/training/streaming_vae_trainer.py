"""Streaming VAE Trainer for HuggingFace IterableDataset.

Extends VAETrainer with support for streaming datasets, step-based
scheduling, and mid-stream checkpointing for petabyte-scale training.
"""
from __future__ import annotations

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
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
    from tqdm import tqdm

    _has_torch = True
except ImportError:
    _has_torch = False


class StreamingVAETrainer:
    """Trainer for VAE models with streaming dataset support.

    Extends the standard VAETrainer to work with HuggingFace IterableDatasets
    and CADStreamingDataset. Key differences from epoch-based training:

    - Step-based scheduling: Uses total_steps instead of epochs
    - KL warmup on global_step: Linear warmup over warmup_steps
    - Mid-stream checkpointing: Save/resume within a streaming epoch
    - Epoch-based shuffle seed: set_epoch() for reproducible shuffling

    Usage:
        >>> from cadling.data.streaming import CADStreamingDataset, CADStreamingConfig
        >>>
        >>> config = CADStreamingConfig(
        ...     dataset_id="latticelabs/deepcad-sequences",
        ...     batch_size=8,
        ... )
        >>> dataset = CADStreamingDataset(config)
        >>>
        >>> trainer = StreamingVAETrainer(
        ...     model=vae_model,
        ...     dataset=dataset,
        ...     total_steps=100000,
        ...     warmup_steps=5000,
        ... )
        >>> trainer.train()

    Args:
        model: VAE model with encode(), decode(), and forward() returning
            (reconstructed, mu, log_var).
        dataset: Streaming dataset with __iter__() method.
        val_dataset: Optional validation dataset.
        total_steps: Total training steps.
        warmup_steps: Steps for KL divergence warmup.
        optimizer: Optional optimizer (creates AdamW if None).
        scheduler: Optional LR scheduler.
        device: Device string ('auto' selects CUDA if available).
        checkpoint_dir: Directory for saving checkpoints.
        log_every: Log metrics every N steps.
        eval_every: Run validation every N steps.
        save_every: Save checkpoint every N steps.
        gradient_accumulation_steps: Accumulate gradients over N steps.
        max_grad_norm: Maximum gradient norm for clipping.
    """

    def __init__(
        self,
        model: "nn.Module",
        dataset: Any = None,
        val_dataset: Optional[Any] = None,
        total_steps: int = 100000,
        warmup_steps: int = 5000,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        device: str = "auto",
        checkpoint_dir: Optional[str] = None,
        log_every: int = 100,
        eval_every: int = 5000,
        save_every: int = 10000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        learning_rate: float = 1e-4,
        dataset_config: Optional[Any] = None,
    ) -> None:
        if not _has_torch:
            raise ImportError(
                "PyTorch is required for StreamingVAETrainer. "
                "Install via: conda install pytorch -c conda-forge"
            )

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)

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
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Default optimizer
        if optimizer is None:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        # Default scheduler: cosine decay with warmup
        if scheduler is None:
            self.scheduler = self._create_scheduler()
        else:
            self.scheduler = scheduler

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
            "recon_loss": [],
            "kl_loss": [],
            "beta": [],
            "val_loss": [],
            "val_recon_loss": [],
            "val_kl_loss": [],
            "learning_rate": [],
        }

        # Running averages for logging
        self._running_loss = 0.0
        self._running_recon = 0.0
        self._running_kl = 0.0
        self._running_count = 0

        _log.info(
            "StreamingVAETrainer initialized: device=%s, total_steps=%d, "
            "warmup_steps=%d",
            self.device,
            self.total_steps,
            self.warmup_steps,
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

    def _compute_beta(self) -> float:
        """Compute KL weight (beta) based on global step.

        Returns:
            Beta value between 0.0 and 1.0, linearly increasing over warmup_steps.
        """
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, self.global_step / self.warmup_steps)

    def _reconstruction_loss(
        self, reconstructed: "torch.Tensor", target: "torch.Tensor"
    ) -> "torch.Tensor":
        """Compute reconstruction loss as cross-entropy.

        Args:
            reconstructed: Logits [batch, seq_len, vocab_size].
            target: Target token IDs [batch, seq_len].

        Returns:
            Scalar cross-entropy loss.
        """
        batch_size, seq_len = target.shape
        vocab_size = reconstructed.shape[-1]
        logits_flat = reconstructed.reshape(batch_size * seq_len, vocab_size)
        target_flat = target.reshape(batch_size * seq_len)
        return F.cross_entropy(logits_flat, target_flat, ignore_index=0)

    def _kl_divergence(
        self, mu: "torch.Tensor", log_var: "torch.Tensor"
    ) -> "torch.Tensor":
        """Compute KL divergence from standard normal.

        Args:
            mu: Latent mean [batch, latent_dim].
            log_var: Latent log variance [batch, latent_dim].

        Returns:
            Scalar KL divergence loss.
        """
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        return kl.mean()

    def _unpack_model_output(self, output):
        """Auto-detect dict vs tuple model output."""
        if isinstance(output, dict):
            return output.get('reconstructed') or output.get('command_logits'), output['mu'], output['log_var']
        return output[0], output[1], output[2]

    def _prepare_batch(
        self, batch: Dict[str, Any]
    ) -> tuple:
        """Prepare a batch for training.

        Args:
            batch: Dictionary with 'command_types' or 'token_ids'.

        Returns:
            Tuple of (input_ids, target_ids, attention_mask).
        """
        # Get input tokens
        if "token_ids" in batch:
            input_ids = batch["token_ids"]
        elif "command_types" in batch:
            input_ids = batch["command_types"]
        else:
            raise ValueError("Batch must contain 'token_ids' or 'command_types'")

        # Ensure tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)

        input_ids = input_ids.to(self.device)

        # Target is same as input for autoencoding
        target_ids = batch.get("target_ids", input_ids).to(self.device)

        # Attention mask
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask)
            attention_mask = attention_mask.to(self.device)

        return input_ids, target_ids, attention_mask

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

        input_ids, target_ids, attention_mask = self._prepare_batch(batch)

        # Forward pass
        output = self.model(input_ids)
        reconstructed, mu, log_var = self._unpack_model_output(output)

        # Compute losses
        recon_loss = self._reconstruction_loss(reconstructed, target_ids)
        kl_loss = self._kl_divergence(mu, log_var)

        beta = self._compute_beta()
        total_loss = recon_loss + beta * kl_loss

        # Scale loss for gradient accumulation
        scaled_loss = total_loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        # Update weights every gradient_accumulation_steps
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1

        return {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "beta": beta,
            "lr": self.scheduler.get_last_lr()[0],
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

        total_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_batches = 0

        beta = self._compute_beta()

        # Iterate through validation dataset
        val_iter = iter(self.val_dataset)

        for batch in val_iter:
            input_ids, target_ids, attention_mask = self._prepare_batch(batch)

            output = self.model(input_ids)
            reconstructed, mu, log_var = self._unpack_model_output(output)

            recon_loss = self._reconstruction_loss(reconstructed, target_ids)
            kl_loss = self._kl_divergence(mu, log_var)
            loss = recon_loss + beta * kl_loss

            total_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1

            # Limit validation batches for streaming
            if num_batches >= 100:
                break

        n = max(num_batches, 1)
        metrics = {
            "val_loss": total_loss / n,
            "val_recon_loss": recon_loss_sum / n,
            "val_kl_loss": kl_loss_sum / n,
        }

        _log.info(
            "Step %d validation: loss=%.4f recon=%.4f kl=%.4f",
            self.global_step,
            metrics["val_loss"],
            metrics["val_recon_loss"],
            metrics["val_kl_loss"],
        )

        return metrics

    def train(self) -> None:
        """Run the full training loop.

        Iterates through the streaming dataset, training until total_steps
        is reached. Supports resuming from checkpoints and handles
        epoch boundaries for streaming datasets.
        """
        _log.info("Starting streaming training for %d steps", self.total_steps)

        self.optimizer.zero_grad()
        pbar = tqdm(total=self.total_steps, desc="Training", initial=self.global_step)

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
                self._running_recon += metrics["recon_loss"]
                self._running_kl += metrics["kl_loss"]
                self._running_count += 1

                # Logging
                if self.global_step % self.log_every == 0:
                    avg_loss = self._running_loss / max(self._running_count, 1)
                    avg_recon = self._running_recon / max(self._running_count, 1)
                    avg_kl = self._running_kl / max(self._running_count, 1)

                    self.history["train_loss"].append(avg_loss)
                    self.history["recon_loss"].append(avg_recon)
                    self.history["kl_loss"].append(avg_kl)
                    self.history["beta"].append(metrics["beta"])
                    self.history["learning_rate"].append(metrics["lr"])

                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "recon": f"{avg_recon:.4f}",
                        "kl": f"{avg_kl:.4f}",
                        "beta": f"{metrics['beta']:.3f}",
                        "lr": f"{metrics['lr']:.2e}",
                    })

                    # Reset running averages
                    self._running_loss = 0.0
                    self._running_recon = 0.0
                    self._running_kl = 0.0
                    self._running_count = 0

                # Validation
                if self.global_step % self.eval_every == 0 and self.val_dataset:
                    val_metrics = self.validate()
                    self.history["val_loss"].append(val_metrics.get("val_loss", 0.0))
                    self.history["val_recon_loss"].append(
                        val_metrics.get("val_recon_loss", 0.0)
                    )
                    self.history["val_kl_loss"].append(
                        val_metrics.get("val_kl_loss", 0.0)
                    )

                    # Save best model
                    val_loss = val_metrics.get("val_loss", float("inf"))
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        if self.checkpoint_dir:
                            self.save_checkpoint("best_model.pt")
                            _log.info(
                                "Saved best model at step %d (val_loss=%.4f)",
                                self.global_step,
                                val_loss,
                            )

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
            "Training complete. Best val loss: %.4f at step %d",
            self.best_val_loss,
            self.global_step,
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
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
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
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
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

        history_path = self.checkpoint_dir / "streaming_vae_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        _log.info("Saved history to %s", history_path)


__all__ = ["StreamingVAETrainer"]
