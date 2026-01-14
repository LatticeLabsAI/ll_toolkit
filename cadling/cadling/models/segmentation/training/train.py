"""Training loops for CAD segmentation models.

Provides training utilities for:
- Mesh segmentation (EdgeConv GNN)
- B-Rep segmentation (GAT + Transformer)
- Manufacturing feature recognition (AAGNet-style)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")

try:
    from torch_geometric.data import Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Training configuration.

    Attributes:
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        batch_size: Batch size
        gradient_clip: Gradient clipping threshold (None = no clipping)
        lr_scheduler: Learning rate scheduler type ('plateau', 'cosine', None)
        early_stopping_patience: Epochs without improvement before stopping (None = no early stopping)
        checkpoint_dir: Directory to save model checkpoints
        checkpoint_frequency: Save checkpoint every N epochs
        log_frequency: Log metrics every N batches
        device: Training device ('cuda', 'cpu', 'auto')
        mixed_precision: Whether to use mixed precision training (AMP)
    """

    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    gradient_clip: Optional[float] = 1.0
    lr_scheduler: Optional[str] = "plateau"
    early_stopping_patience: Optional[int] = 10
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_frequency: int = 5
    log_frequency: int = 10
    device: str = "auto"
    mixed_precision: bool = False


class SegmentationTrainer:
    """Trainer for CAD segmentation models.

    Supports:
    - Semantic segmentation (face-level classification)
    - Instance segmentation (with discriminative loss)
    - Streaming datasets (avoid full download)
    - Mixed precision training
    - Checkpointing and early stopping

    Example:
        >>> from cadling.models.segmentation.architectures import HybridGATTransformer
        >>> from cadling.models.segmentation.training import (
        ...     SegmentationTrainer, TrainingConfig, create_streaming_pipeline
        ... )
        >>>
        >>> # Create model
        >>> model = HybridGATTransformer(in_dim=24, num_classes=24)
        >>>
        >>> # Create streaming pipeline
        >>> train_pipeline = create_streaming_pipeline(
        ...     dataset_name="path/to/mfcad",
        ...     graph_builder=build_brep_graph,
        ...     split="train",
        ...     batch_size=16
        ... )
        >>> val_pipeline = create_streaming_pipeline(
        ...     dataset_name="path/to/mfcad",
        ...     graph_builder=build_brep_graph,
        ...     split="val",
        ...     batch_size=16
        ... )
        >>>
        >>> # Train
        >>> config = TrainingConfig(num_epochs=50, learning_rate=1e-3)
        >>> trainer = SegmentationTrainer(model, config)
        >>> trainer.train(train_pipeline, val_pipeline)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
    ):
        """Initialize trainer.

        Args:
            model: Segmentation model (must have forward(x, edge_index, edge_attr, batch))
            config: Training configuration
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: Adam)
        """
        if not TORCH_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch and PyTorch Geometric required. "
                "Install: pip install torch torch-geometric"
            )

        self.model = model
        self.config = config

        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        logger.info(f"Training on device: {self.device}")
        self.model = self.model.to(self.device)

        # Setup loss function
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Setup learning rate scheduler
        self.scheduler = None
        if config.lr_scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=0.5
            )
        elif config.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.num_epochs
            )

        # Setup mixed precision
        self.scaler = None
        if config.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Using mixed precision training (AMP)")

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_pipeline: Any) -> dict[str, float]:
        """Train for one epoch.

        Args:
            train_pipeline: StreamingDataPipeline instance

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_count = 0

        start_time = time.time()

        for batch_idx, batch in enumerate(train_pipeline):
            # Move batch to device
            batch = batch.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass (with mixed precision if enabled)
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(
                        batch.x, batch.edge_index, batch.edge_attr, batch.batch
                    )
                    loss = self.criterion(logits, batch.y)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Regular forward pass
                logits = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = self.criterion(logits, batch.y)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )

                # Optimizer step
                self.optimizer.step()

            # Metrics
            total_loss += loss.item() * batch.num_graphs
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == batch.y).sum().item()
            total_samples += batch.num_nodes
            batch_count += 1

            # Log
            if (batch_idx + 1) % self.config.log_frequency == 0:
                avg_loss = total_loss / max(total_samples, 1)
                accuracy = total_correct / max(total_samples, 1)
                logger.info(
                    f"Epoch {self.current_epoch} - Batch {batch_idx+1}: "
                    f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}"
                )

        epoch_time = time.time() - start_time

        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
            "time": epoch_time,
        }

    @torch.no_grad()
    def validate(self, val_pipeline: Any) -> dict[str, float]:
        """Validate on validation set.

        Args:
            val_pipeline: StreamingDataPipeline instance

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in val_pipeline:
            batch = batch.to(self.device)

            # Forward pass
            logits = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = self.criterion(logits, batch.y)

            # Metrics
            total_loss += loss.item() * batch.num_graphs
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == batch.y).sum().item()
            total_samples += batch.num_nodes

        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
        }

    def train(
        self,
        train_pipeline: Any,
        val_pipeline: Optional[Any] = None,
    ) -> dict[str, list[float]]:
        """Complete training loop.

        Args:
            train_pipeline: StreamingDataPipeline for training
            val_pipeline: Optional StreamingDataPipeline for validation

        Returns:
            Dictionary with training history
        """
        logger.info(
            f"Starting training: {self.config.num_epochs} epochs, "
            f"lr={self.config.learning_rate}, batch_size={self.config.batch_size}"
        )

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1

            # Train
            train_metrics = self.train_epoch(train_pipeline)
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])

            logger.info(
                f"Epoch {self.current_epoch}/{self.config.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Time: {train_metrics['time']:.1f}s"
            )

            # Validate
            if val_pipeline is not None:
                val_metrics = self.validate(val_pipeline)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])

                logger.info(
                    f"Epoch {self.current_epoch}/{self.config.num_epochs} - "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

                # Learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics["loss"])
                    else:
                        self.scheduler.step()

                # Early stopping
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.epochs_without_improvement = 0

                    # Save best model
                    self.save_checkpoint(
                        self.config.checkpoint_dir / "best_model.pt", is_best=True
                    )
                else:
                    self.epochs_without_improvement += 1

                    if (
                        self.config.early_stopping_patience is not None
                        and self.epochs_without_improvement >= self.config.early_stopping_patience
                    ):
                        logger.info(
                            f"Early stopping triggered after {self.current_epoch} epochs "
                            f"({self.epochs_without_improvement} epochs without improvement)"
                        )
                        break

            # Checkpoint
            if self.current_epoch % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(
                    self.config.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
                )

        logger.info("Training complete!")
        return history

    def save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            path: Checkpoint file path
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}" + (" (BEST)" if is_best else ""))

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint.

        Args:
            path: Checkpoint file path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Checkpoint loaded: {path} (epoch {self.current_epoch})")


# Example usage
if __name__ == "__main__":
    from ..architectures.gat_net import HybridGATTransformer
    from .streaming_pipeline import create_streaming_pipeline, build_brep_graph

    # Create model
    model = HybridGATTransformer(
        in_dim=24,
        num_classes=24,
        gat_hidden_dim=256,
        gat_num_heads=8,
        gat_num_layers=3,
        transformer_hidden_dim=512,
        transformer_num_layers=4,
    )

    # Create streaming pipelines
    train_pipeline = create_streaming_pipeline(
        dataset_name="path/to/mfcad",
        graph_builder=build_brep_graph,
        dataset_type="mfcad",
        split="train",
        batch_size=16,
        streaming=True,
    )

    val_pipeline = create_streaming_pipeline(
        dataset_name="path/to/mfcad",
        graph_builder=build_brep_graph,
        dataset_type="mfcad",
        split="val",
        batch_size=16,
        streaming=True,
    )

    # Training config
    config = TrainingConfig(
        num_epochs=50,
        learning_rate=1e-3,
        batch_size=16,
        checkpoint_dir=Path("checkpoints/brep_seg"),
        early_stopping_patience=10,
    )

    # Train
    trainer = SegmentationTrainer(model, config)
    history = trainer.train(train_pipeline, val_pipeline)

    print(f"Training complete! Best val loss: {trainer.best_val_loss:.4f}")
