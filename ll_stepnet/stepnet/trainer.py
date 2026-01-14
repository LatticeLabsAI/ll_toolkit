"""
Training utilities for STEP models.
Handles training loops, evaluation, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Callable
from tqdm import tqdm
import json


class STEPTrainer:
    """
    Trainer for STEP models.
    Handles training loop, validation, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: Optional[str] = None
    ):
        """
        Args:
            model: STEP model to train
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            optimizer: Optimizer (creates AdamW if None)
            loss_fn: Loss function (creates based on task if None)
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device

        # Default optimizer
        if optimizer is None:
            self.optimizer = AdamW(model.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer

        # Default loss function based on model type
        if loss_fn is None:
            self.loss_fn = self._get_default_loss_fn()
        else:
            self.loss_fn = loss_fn

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def _get_default_loss_fn(self) -> Callable:
        """Get default loss function based on model type."""
        model_name = self.model.__class__.__name__

        if 'Classification' in model_name:
            return nn.CrossEntropyLoss()
        elif 'Property' in model_name or 'Similarity' in model_name:
            return nn.MSELoss()
        elif 'Captioning' in model_name or 'QA' in model_name:
            return nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        else:
            return nn.MSELoss()

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")

        for batch in pbar:
            # Move batch to device
            token_ids = batch['token_ids'].to(self.device)
            topology_data = batch.get('topology_data', None)

            # Forward pass
            if 'labels' in batch:
                labels = batch['labels'].to(self.device)

                # Handle different model types
                model_name = self.model.__class__.__name__

                if 'Captioning' in model_name or 'QA' in model_name:
                    # Sequence generation models
                    outputs = self.model(token_ids, caption_ids=labels[:, :-1], topology_data=topology_data)
                    # Reshape for cross entropy: [batch * seq_len, vocab_size]
                    loss = self.loss_fn(
                        outputs.reshape(-1, outputs.size(-1)),
                        labels[:, 1:].reshape(-1)
                    )
                else:
                    # Classification/regression models
                    outputs = self.model(token_ids, topology_data=topology_data)
                    loss = self.loss_fn(outputs, labels)
            else:
                # Unsupervised (similarity learning, etc.)
                outputs = self.model(token_ids, topology_data=topology_data)
                # Default to contrastive or reconstruction loss
                loss = outputs.mean()  # Placeholder

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary with validation metrics
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(self.val_dataloader, desc="Validation"):
            # Move batch to device
            token_ids = batch['token_ids'].to(self.device)
            topology_data = batch.get('topology_data', None)

            if 'labels' in batch:
                labels = batch['labels'].to(self.device)

                # Forward pass
                model_name = self.model.__class__.__name__

                if 'Captioning' in model_name or 'QA' in model_name:
                    outputs = self.model(token_ids, caption_ids=labels[:, :-1], topology_data=topology_data)
                    loss = self.loss_fn(
                        outputs.reshape(-1, outputs.size(-1)),
                        labels[:, 1:].reshape(-1)
                    )
                else:
                    outputs = self.model(token_ids, topology_data=topology_data)
                    loss = self.loss_fn(outputs, labels)

                    # Calculate accuracy for classification
                    if 'Classification' in model_name:
                        preds = torch.argmax(outputs, dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                total_loss += loss.item()

        metrics = {
            'val_loss': total_loss / len(self.val_dataloader)
        }

        if total_samples > 0:
            metrics['val_accuracy'] = total_correct / total_samples

        return metrics

    def train(self, num_epochs: int, save_every: int = 1):
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)

            print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}")

            # Validate
            if self.val_dataloader:
                val_metrics = self.validate()
                self.history['val_loss'].append(val_metrics.get('val_loss', 0))

                if 'val_accuracy' in val_metrics:
                    self.history['val_accuracy'].append(val_metrics['val_accuracy'])
                    print(f"Val Loss = {val_metrics['val_loss']:.4f}, "
                          f"Val Acc = {val_metrics['val_accuracy']:.4f}")
                else:
                    print(f"Val Loss = {val_metrics['val_loss']:.4f}")

                # Save best model
                val_loss = val_metrics['val_loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.checkpoint_dir:
                        self.save_checkpoint('best_model.pt')
                        print(f"Saved best model (val_loss={val_loss:.4f})")

            # Save periodic checkpoint
            if self.checkpoint_dir and (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        # Save final model
        if self.checkpoint_dir:
            self.save_checkpoint('final_model.pt')
            self.save_history()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        load_path = self.checkpoint_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        print(f"Loaded checkpoint from epoch {self.epoch}")

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
