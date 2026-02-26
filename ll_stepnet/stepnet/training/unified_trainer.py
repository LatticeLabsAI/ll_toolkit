"""Unified trainer for STEP neural network models.

Provides a high-level interface for training different generative architectures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from .logger import TensorBoardLogger, WandbLogger
from .vae_trainer import VAETrainer
from .gan_trainer import GANTrainer
from .diffusion_trainer import DiffusionTrainer
from .streaming_vae_trainer import StreamingVAETrainer
from .streaming_diffusion_trainer import StreamingDiffusionTrainer
from .streaming_gan_trainer import StreamingGANTrainer

_log = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Type of generative model."""

    VAE = "vae"
    GAN = "gan"
    DIFFUSION = "diffusion"
    VQVAE = "vqvae"


@dataclass
class TrainingConfig:
    """Configuration for unified training.

    Args:
        model_type: Type of model being trained.
        epochs: Total training epochs.
        batch_size: Batch size for training.
        learning_rate: Base learning rate.
        weight_decay: Weight decay for optimizer.
        grad_clip: Gradient clipping norm (None to disable).
        checkpoint_dir: Directory for saving checkpoints.
        checkpoint_freq: Save checkpoint every N epochs.
        log_freq: Log metrics every N steps.
        eval_freq: Evaluate every N epochs.
        use_wandb: Whether to use W&B logging.
        use_tensorboard: Whether to use TensorBoard logging.
        wandb_project: W&B project name.
        device: Device to train on ('cuda', 'cpu', 'mps').
    """

    model_type: ModelType = ModelType.VAE
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    checkpoint_dir: str = "checkpoints"
    checkpoint_freq: int = 10
    log_freq: int = 100
    eval_freq: int = 5
    use_wandb: bool = False
    use_tensorboard: bool = True
    wandb_project: str = "stepnet"
    device: str = "cuda"
    streaming: bool = False
    total_steps: int = 100000
    warmup_steps: int = 5000
    dataset_config: Optional[Any] = None


@dataclass
class STEPNetTrainer:
    """Unified trainer for STEP neural network models.

    Provides a high-level interface that delegates to specialized trainers
    based on the model type (VAE, GAN, Diffusion).

    Args:
        model: The model to train (STEPVAE, LatentGAN, StructuredDiffusion, etc.)
        config: Training configuration.
        train_loader: Training data loader.
        val_loader: Validation data loader (optional).
        optimizer: Custom optimizer (auto-created if None).
        scheduler: Learning rate scheduler (optional).
        logger: Custom logger (auto-created based on config if None).

    Example:
        >>> from stepnet.vae import STEPVAE
        >>> from stepnet.training import STEPNetTrainer, TrainingConfig
        >>>
        >>> model = STEPVAE()
        >>> config = TrainingConfig(model_type=ModelType.VAE, epochs=50)
        >>> trainer = STEPNetTrainer(model, config, train_loader)
        >>> trainer.train()
    """

    model: nn.Module
    config: TrainingConfig
    train_loader: DataLoader
    val_loader: Optional[DataLoader] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    logger: Optional[Union[WandbLogger, TensorBoardLogger]] = None

    _inner_trainer: Any = field(default=None, init=False, repr=False)
    _global_step: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize trainer components."""
        self._setup_device()
        self._setup_optimizer()
        self._setup_logger()
        self._setup_inner_trainer()

    def _setup_device(self) -> None:
        """Move model to target device."""
        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            _log.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            _log.warning("MPS not available, falling back to CPU")
            device = "cpu"

        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        _log.info("Using device: %s", self.device)

    def _setup_optimizer(self) -> None:
        """Create optimizer if not provided."""
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

    def _setup_logger(self) -> None:
        """Create logger based on config."""
        if self.logger is not None:
            return

        if self.config.use_wandb:
            self.logger = WandbLogger(
                project=self.config.wandb_project,
                config={
                    "model_type": self.config.model_type.value,
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                },
            )
            self.logger.init()
        elif self.config.use_tensorboard:
            log_dir = Path(self.config.checkpoint_dir) / "logs"
            self.logger = TensorBoardLogger(log_dir=str(log_dir))
            self.logger.init()

    def _setup_inner_trainer(self) -> None:
        """Create specialized trainer based on model type and streaming flag.

        When ``config.streaming`` is True, delegates to the corresponding
        streaming trainer (step-based, IterableDataset). When False, uses
        the standard epoch-based trainer.
        """
        model_type = self.config.model_type
        streaming = self.config.streaming

        if streaming:
            self._setup_streaming_trainer(model_type)
        else:
            self._setup_epoch_trainer(model_type)

        mode = "streaming" if streaming else "epoch-based"
        _log.info("Using %s %s trainer", mode, model_type.value)

    def _setup_epoch_trainer(self, model_type: ModelType) -> None:
        """Create an epoch-based specialized trainer."""
        if model_type == ModelType.VAE or model_type == ModelType.VQVAE:
            self._inner_trainer = VAETrainer(
                model=self.model,
                train_dataloader=self.train_loader,
                val_dataloader=self.val_loader,
                optimizer=self.optimizer,
                device=str(self.device),
                checkpoint_dir=self.config.checkpoint_dir,
            )
        elif model_type == ModelType.GAN:
            # GAN needs generator/discriminator — assume model exposes them
            gen = getattr(self.model, "generator", self.model)
            disc = getattr(self.model, "discriminator", self.model)
            self._inner_trainer = GANTrainer(
                generator=gen,
                discriminator=disc,
                train_dataloader=self.train_loader,
                device=str(self.device),
                checkpoint_dir=self.config.checkpoint_dir,
            )
            # GANTrainer doesn't accept val_dataloader in __init__; set it
            # as an attribute so validate() can find it via getattr().
            if self.val_loader is not None:
                self._inner_trainer.val_dataloader = self.val_loader
        elif model_type == ModelType.DIFFUSION:
            # Diffusion needs a noise scheduler — check model
            scheduler = getattr(self.model, "scheduler", None)
            if scheduler is None:
                from ..diffusion import DDPMScheduler
                scheduler = DDPMScheduler()
            self._inner_trainer = DiffusionTrainer(
                model=self.model,
                scheduler=scheduler,
                train_dataloader=self.train_loader,
                val_dataloader=self.val_loader,
                device=str(self.device),
                checkpoint_dir=self.config.checkpoint_dir,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _setup_streaming_trainer(self, model_type: ModelType) -> None:
        """Create a step-based streaming trainer.

        The streaming dataset is resolved in this order:
        1. ``config.dataset_config`` (a ``StreamingCadlingConfig``) — builds
           the dataset automatically via cadling.
        2. ``train_loader`` — used directly as a streaming dataset if no
           dataset_config is provided.
        """
        ds_cfg = self.config.dataset_config
        # If no dataset_config, pass train_loader as the dataset directly
        dataset = None if ds_cfg is not None else self.train_loader

        common_kwargs: Dict[str, Any] = {
            "total_steps": self.config.total_steps,
            "warmup_steps": self.config.warmup_steps,
            "device": str(self.device),
            "checkpoint_dir": self.config.checkpoint_dir,
            "learning_rate": self.config.learning_rate,
        }

        if model_type in (ModelType.VAE, ModelType.VQVAE):
            self._inner_trainer = StreamingVAETrainer(
                model=self.model,
                dataset=dataset,
                dataset_config=ds_cfg,
                **common_kwargs,
            )
        elif model_type == ModelType.GAN:
            # GAN needs generator/critic — assume model exposes them
            gen = getattr(self.model, "generator", self.model)
            critic = getattr(self.model, "discriminator", self.model)
            self._inner_trainer = StreamingGANTrainer(
                generator=gen,
                critic=critic,
                dataset=dataset,
                dataset_config=ds_cfg,
                total_steps=self.config.total_steps,
                warmup_steps=self.config.warmup_steps,
                device=str(self.device),
                checkpoint_dir=self.config.checkpoint_dir,
            )
        elif model_type == ModelType.DIFFUSION:
            # Diffusion needs a noise scheduler — check model
            scheduler = getattr(self.model, "scheduler", None)
            if scheduler is None:
                from ..diffusion import DDPMScheduler
                scheduler = DDPMScheduler()
            self._inner_trainer = StreamingDiffusionTrainer(
                model=self.model,
                scheduler=scheduler,
                dataset=dataset,
                dataset_config=ds_cfg,
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self) -> Dict[str, List[float]]:
        """Run full training loop.

        When ``config.streaming`` is True, delegates entirely to the
        streaming trainer's built-in ``train()`` loop (step-based with
        checkpointing, logging, and validation baked in).

        When False, runs the standard epoch-based loop.

        Returns:
            Dictionary of metric histories.
        """
        if self.config.streaming:
            return self._train_streaming()

        return self._train_epoch_based()

    def _train_streaming(self) -> Dict[str, List[float]]:
        """Delegate to the streaming trainer's train() loop."""
        _log.info(
            "Starting streaming training for %d steps",
            self.config.total_steps,
        )
        self._inner_trainer.train()

        # Return history from the streaming trainer
        history = getattr(self._inner_trainer, "history", {})
        _log.info("Streaming training complete")
        return history

    def _train_epoch_based(self) -> Dict[str, List[float]]:
        """Run standard epoch-based training loop."""
        _log.info("Starting training for %d epochs", self.config.epochs)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(self.config.epochs):
            # Training epoch
            train_metrics = self._train_epoch(epoch)
            # VAETrainer returns 'total_loss'; GAN/Diffusion return 'loss'
            train_loss = train_metrics.get("loss", train_metrics.get("total_loss", 0.0))
            history["train_loss"].append(train_loss)

            # Validation
            if self.val_loader is not None and (epoch + 1) % self.config.eval_freq == 0:
                val_metrics = self._validate(epoch)
                history["val_loss"].append(val_metrics.get("loss", 0.0))

            # Checkpointing
            if (epoch + 1) % self.config.checkpoint_freq == 0:
                self._save_checkpoint(epoch)

            # Logging
            if self.logger is not None:
                self.logger.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        **{f"train_{k}": v for k, v in train_metrics.items() if k not in ("loss", "total_loss")},
                    },
                    step=epoch,
                )

        _log.info("Training complete")
        return history

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run a single training epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of epoch metrics.
        """
        self.model.train()
        metrics = self._inner_trainer.train_epoch()
        return metrics

    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run validation.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        with torch.no_grad():
            metrics = self._inner_trainer.validate()
        return metrics

    def _save_checkpoint(self, epoch: int) -> None:
        """Save a training checkpoint.

        Args:
            epoch: Current epoch number.
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        self._inner_trainer.save_checkpoint(str(checkpoint_path))
        _log.info("Saved checkpoint: %s", checkpoint_path)

    def load_checkpoint(self, path: str) -> int:
        """Load a training checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Epoch number from checkpoint.
        """
        self._inner_trainer.load_checkpoint(path)
        _log.info("Loaded checkpoint from %s", path)
        return 0

    def finish(self) -> None:
        """Clean up resources."""
        if self.logger is not None:
            self.logger.finish()
