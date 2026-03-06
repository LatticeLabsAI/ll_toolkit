"""
Training utilities for STEP models.
Handles training loops, evaluation, and checkpointing.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    _has_torch = True
except ImportError:
    _has_torch = False

import logging
from pathlib import Path
from typing import Optional, Dict, Callable
import json

_log = logging.getLogger(__name__)


if not _has_torch:

    class STEPTrainer:  # type: ignore[no-redef]
        """Stub raised when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "STEPTrainer requires PyTorch. Install via conda-forge: "
                "conda install -c conda-forge pytorch"
            )

else:

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
            device: str = "auto",
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
            if not _has_torch:
                raise ImportError(
                    "STEPTrainer requires PyTorch. Install via conda-forge: "
                    "conda install -c conda-forge pytorch"
                )

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.model = model.to(device)
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader

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

            # Track optimizer param ids for detecting lazily-added parameters
            self._optimizer_param_ids: set = {
                id(p) for group in self.optimizer.param_groups for p in group['params']
            }

            _log.info(
                "STEPTrainer initialized: model=%s, device=%s",
                model.__class__.__name__, self.device,
            )

        def _sync_optimizer_params(self) -> None:
            """Add any lazily-created model parameters to the optimizer.

            Some modules (e.g. ``STEPEncoder._feature_projs``) create
            ``nn.Linear`` layers during ``forward()`` which are not present
            when the optimizer is first constructed.  This method detects new
            parameters and adds them to the optimizer so they receive gradient
            updates.
            """
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

        def _compute_forward(self, batch: Dict) -> Dict:
            """
            Shared forward-pass dispatch for all model types.

            Routes the batch through the correct forward path based on model class name,
            returning loss, raw outputs, and labels.

            Args:
                batch: Data batch from a DataLoader.

            Returns:
                Dictionary with 'loss' (Tensor), 'outputs' (Tensor or dict), and 'labels' (Tensor or None).
            """
            token_ids = batch['token_ids'].to(self.device)
            topology_data = batch.get('topology_data', None)
            model_name = self.model.__class__.__name__
            labels = batch['labels'].to(self.device) if 'labels' in batch else None

            if 'CausalLM' in model_name or 'MaskedLM' in model_name or 'HybridLM' in model_name:
                forward_kwargs = dict(
                    attention_mask=batch.get('attention_mask', torch.ones_like(token_ids)).to(self.device),
                    topology_data=topology_data,
                    labels=labels,
                )
                if 'HybridLM' in model_name:
                    if 'masked_input_ids' in batch:
                        forward_kwargs['masked_input_ids'] = batch['masked_input_ids'].to(self.device)
                    if 'masked_labels' in batch:
                        forward_kwargs['masked_labels'] = batch['masked_labels'].to(self.device)

                outputs = self.model(token_ids, **forward_kwargs)
                if 'loss' not in outputs:
                    raise ValueError(
                        f"{model_name} returned no loss. Ensure labels are provided "
                        "in the batch for pretrain models."
                    )
                return {'loss': outputs['loss'], 'outputs': outputs, 'labels': labels}

            elif labels is not None:
                if 'Captioning' in model_name or 'QA' in model_name:
                    outputs = self.model(token_ids, caption_ids=labels[:, :-1], topology_data=topology_data)
                    loss = self.loss_fn(
                        outputs.reshape(-1, outputs.size(-1)),
                        labels[:, 1:].reshape(-1)
                    )
                else:
                    outputs = self.model(token_ids, topology_data=topology_data)
                    loss = self.loss_fn(outputs, labels)
                return {'loss': loss, 'outputs': outputs, 'labels': labels}

            elif 'Similarity' in model_name:
                embeddings = self.model(token_ids, topology_data=topology_data)
                sim_matrix = torch.mm(embeddings, embeddings.t())
                temperature = 0.07
                sim_matrix = sim_matrix / temperature
                info_nce_labels = torch.arange(embeddings.size(0), device=self.device)
                loss = nn.CrossEntropyLoss()(sim_matrix, info_nce_labels)
                return {'loss': loss, 'outputs': embeddings, 'labels': None}

            else:
                raise ValueError(
                    f"No labels in batch and no unsupervised loss defined for {model_name}. "
                    "Provide labels or use a pretrain/similarity model."
                )

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
                result = self._compute_forward(batch)
                loss = result['loss']

                # Detect lazily-created parameters not yet tracked by optimizer
                self._sync_optimizer_params()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1

                pbar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / max(num_batches, 1)
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
            model_name = self.model.__class__.__name__

            for batch in tqdm(self.val_dataloader, desc="Validation"):
                result = self._compute_forward(batch)
                total_loss += result['loss'].item()

                # Calculate accuracy for classification
                if 'Classification' in model_name and result['labels'] is not None:
                    preds = torch.argmax(result['outputs'], dim=1)
                    total_correct += (preds == result['labels']).sum().item()
                    total_samples += result['labels'].size(0)

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

                _log.info("Epoch %d: Train Loss = %.4f", epoch, train_loss)

                # Validate
                if self.val_dataloader:
                    val_metrics = self.validate()
                    self.history['val_loss'].append(val_metrics.get('val_loss', 0))

                    if 'val_accuracy' in val_metrics:
                        self.history['val_accuracy'].append(val_metrics['val_accuracy'])
                        _log.info(
                            "Val Loss = %.4f, Val Acc = %.4f",
                            val_metrics['val_loss'], val_metrics['val_accuracy'],
                        )
                    else:
                        _log.info("Val Loss = %.4f", val_metrics['val_loss'])

                    # Save best model
                    val_loss = val_metrics['val_loss']
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        if self.checkpoint_dir:
                            self.save_checkpoint('best_model.pt')
                            _log.info("Saved best model (val_loss=%.4f)", val_loss)

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
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']
            self.history = checkpoint['history']

            _log.info("Loaded checkpoint from epoch %d", self.epoch)

        def save_history(self):
            """Save training history to JSON."""
            history_path = self.checkpoint_dir / 'history.json'
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
