"""VAE Trainer for generative CAD models.

Implements training loop for Variational Autoencoders with:
- Reconstruction loss (cross-entropy on command tokens)
- KL divergence with beta-warmup scheduling
- Latent space visualization via t-SNE/UMAP
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
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    _has_torch = True
except ImportError:
    _has_torch = False


class VAETrainer:
    """Trainer for Variational Autoencoder models on CAD token sequences.

    Extends the STEPTrainer concept with VAE-specific training:
    - Beta-VAE warmup: linearly ramps KL weight from 0 to 1 over warmup epochs
    - Reconstruction loss via cross-entropy on command tokens
    - KL divergence regularization on the latent distribution
    - Latent space visualization at each epoch

    Supports two model output conventions:

    - **Dict output** (STEPVAE): ``forward()`` returns a dict with keys
      ``command_logits``, ``param_logits``, ``mu``, ``log_var``, ``kl_loss``,
      and optionally ``recon_loss`` and ``loss``.
    - **Tuple output** (legacy): ``forward()`` returns
      ``(reconstructed, mu, log_var)``.

    The trainer auto-detects which convention is used on the first batch
    and adapts accordingly.

    Args:
        model: VAE model with encode(), decode(), and reparameterize() methods.
        train_dataloader: Training data loader.
        val_dataloader: Optional validation data loader.
        optimizer: Optimizer instance. Creates AdamW with lr=1e-4 if None.
        device: Device string. 'auto' selects CUDA if available, else CPU.
        checkpoint_dir: Directory path for saving checkpoints and visualizations.
        kl_warmup_epochs: Number of epochs to linearly ramp beta from 0 to 1.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Any] = None,
        device: str = "auto",
        checkpoint_dir: Optional[str] = None,
        kl_warmup_epochs: int = 10,
    ) -> None:
        if not _has_torch:
            raise ImportError(
                "PyTorch is required for VAETrainer. "
                "Install via conda: conda install pytorch -c conda-forge"
            )

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.kl_warmup_epochs = kl_warmup_epochs

        # Default optimizer
        if optimizer is None:
            self.optimizer = AdamW(model.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self._dict_output: Optional[bool] = None  # Auto-detect on first batch
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "beta": [],
            "val_loss": [],
            "val_recon_loss": [],
            "val_kl_loss": [],
            "command_accuracy": [],
            "param_mse": [],
        }

        _log.info(
            "VAETrainer initialized: device=%s, kl_warmup_epochs=%d",
            self.device,
            self.kl_warmup_epochs,
        )

    def _compute_beta(self) -> float:
        """Compute the current KL divergence weight (beta) based on warmup schedule.

        Returns:
            Beta value between 0.0 and 1.0, linearly increasing over warmup epochs.
        """
        if self.kl_warmup_epochs <= 0:
            return 1.0
        return min(1.0, self.epoch / self.kl_warmup_epochs)

    def _reconstruction_loss(
        self, reconstructed: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss as cross-entropy over command tokens.

        Args:
            reconstructed: Model output logits of shape (batch, seq_len, vocab_size).
            target: Target token IDs of shape (batch, seq_len).

        Returns:
            Scalar cross-entropy loss tensor.
        """
        batch_size, seq_len = target.shape
        vocab_size = reconstructed.shape[-1]
        logits_flat = reconstructed.reshape(batch_size * seq_len, vocab_size)
        target_flat = target.reshape(batch_size * seq_len)
        return F.cross_entropy(logits_flat, target_flat, ignore_index=0)

    def _kl_divergence(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between learned distribution and standard normal.

        Uses the closed-form KL(q(z|x) || p(z)) for diagonal Gaussian:
        KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

        Args:
            mu: Mean of the latent distribution, shape (batch, latent_dim).
            log_var: Log variance of the latent distribution, shape (batch, latent_dim).

        Returns:
            Scalar KL divergence loss tensor (mean over batch).
        """
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        return kl.mean()

    def _unpack_model_output(
        self,
        output: Any,
        target_ids: torch.Tensor,
    ) -> tuple:
        """Unpack model forward output, handling both dict and tuple formats.

        STEPVAE returns a dict with ``command_logits``, ``param_logits``,
        ``mu``, ``log_var``, ``kl_loss``, and optionally ``recon_loss`` /
        ``loss``.  Legacy models return ``(reconstructed, mu, log_var)``
        as a plain tuple.

        This method normalises both into a consistent set of values that
        the training loop can consume, so callers don't need to branch.

        Args:
            output: Raw model forward output (dict or tuple).
            target_ids: Target token IDs for loss computation.

        Returns:
            Tuple of (reconstructed, mu, log_var, recon_loss, kl_loss) where
            *reconstructed* is the logits tensor ``[B, S, V]`` used for
            accuracy and MSE metrics, and *recon_loss* / *kl_loss* are
            scalar tensors.  For dict outputs that already include precomputed
            losses we reuse them; for tuple outputs we compute them here.
        """
        if isinstance(output, dict):
            # --- STEPVAE-style dict output ---
            if self._dict_output is None:
                self._dict_output = True
                _log.info(
                    "Detected dict model output (STEPVAE convention). "
                    "Losses will be extracted from the model output dict."
                )

            mu = output["mu"]
            log_var = output["log_var"]
            command_logits = output["command_logits"]  # [B, S, C]

            # Use pre-computed losses when available
            if "loss" in output and "recon_loss" in output:
                recon_loss = output["recon_loss"]
                kl_loss = output["kl_loss"]
            else:
                # Compute losses ourselves
                recon_loss = self._reconstruction_loss(command_logits, target_ids)
                kl_loss = self._kl_divergence(mu, log_var)

            # Use command_logits as the "reconstructed" tensor for metrics
            reconstructed = command_logits

            return reconstructed, mu, log_var, recon_loss, kl_loss

        else:
            # --- Legacy tuple output: (reconstructed, mu, log_var) ---
            if self._dict_output is None:
                self._dict_output = False
                _log.info(
                    "Detected tuple model output (legacy convention). "
                    "Losses will be computed by the trainer."
                )

            reconstructed, mu, log_var = output

            recon_loss = self._reconstruction_loss(reconstructed, target_ids)
            kl_loss = self._kl_divergence(mu, log_var)

            return reconstructed, mu, log_var, recon_loss, kl_loss

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with beta-VAE warmup.

        Computes:
        - Reconstruction loss (cross-entropy on command tokens)
        - KL divergence with current beta weight
        - Total loss = recon_loss + beta * kl_loss

        Returns:
            Dictionary with keys: 'total_loss', 'recon_loss', 'kl_loss', 'beta'.
        """
        self.model.train()
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_batches = 0

        beta = self._compute_beta()

        pbar = tqdm(
            self.train_dataloader,
            desc=f"VAE Epoch {self.epoch} (beta={beta:.3f})",
        )

        for batch in pbar:
            # Move batch to device
            token_ids = batch["token_ids"].to(self.device)
            target_ids = batch.get("target_ids", token_ids).to(self.device)

            # Forward pass: handles both dict (STEPVAE) and tuple (legacy) output
            output = self.model(token_ids)
            reconstructed, mu, log_var, recon_loss, kl_loss = (
                self._unpack_model_output(output, target_ids)
            )
            total_loss = recon_loss + beta * kl_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate metrics
            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix(
                {
                    "loss": total_loss.item(),
                    "recon": recon_loss.item(),
                    "kl": kl_loss.item(),
                }
            )

        metrics = {
            "total_loss": total_loss_sum / max(num_batches, 1),
            "recon_loss": recon_loss_sum / max(num_batches, 1),
            "kl_loss": kl_loss_sum / max(num_batches, 1),
            "beta": beta,
        }

        _log.info(
            "Epoch %d train: total=%.4f recon=%.4f kl=%.4f beta=%.3f",
            self.epoch,
            metrics["total_loss"],
            metrics["recon_loss"],
            metrics["kl_loss"],
            metrics["beta"],
        )

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute reconstruction quality metrics.

        Computes:
        - Validation loss (recon + beta * KL)
        - Command accuracy: exact match rate of predicted vs target command tokens
        - Parameter MSE: mean squared error of continuous parameter predictions

        Returns:
            Dictionary with keys: 'val_loss', 'recon_loss', 'kl_loss',
            'command_accuracy', 'param_mse'.
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        total_correct = 0
        total_tokens = 0
        param_mse_sum = 0.0
        param_count = 0
        num_batches = 0

        beta = self._compute_beta()

        for batch in tqdm(self.val_dataloader, desc="VAE Validation"):
            token_ids = batch["token_ids"].to(self.device)
            target_ids = batch.get("target_ids", token_ids).to(self.device)

            # Forward pass: handles both dict (STEPVAE) and tuple (legacy) output
            output = self.model(token_ids)
            reconstructed, mu, log_var, recon_loss, kl_loss = (
                self._unpack_model_output(output, target_ids)
            )
            total_loss = recon_loss + beta * kl_loss

            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1

            # Command accuracy: argmax predictions vs target
            pred_tokens = reconstructed.argmax(dim=-1)  # (batch, seq_len)
            mask = target_ids != 0  # Ignore padding tokens
            total_correct += (pred_tokens[mask] == target_ids[mask]).sum().item()
            total_tokens += mask.sum().item()

            # Parameter MSE: compare softmax probabilities to one-hot targets
            # This captures how confident the model is on correct tokens
            probs = F.softmax(reconstructed, dim=-1)
            target_one_hot = F.one_hot(
                target_ids, num_classes=reconstructed.shape[-1]
            ).float()
            batch_mse = F.mse_loss(probs[mask], target_one_hot[mask], reduction="sum")
            param_mse_sum += batch_mse.item()
            param_count += mask.sum().item()

        n = max(num_batches, 1)
        metrics = {
            "val_loss": total_loss_sum / n,
            "recon_loss": recon_loss_sum / n,
            "kl_loss": kl_loss_sum / n,
            "command_accuracy": total_correct / max(total_tokens, 1),
            "param_mse": param_mse_sum / max(param_count, 1),
        }

        _log.info(
            "Epoch %d val: loss=%.4f cmd_acc=%.4f param_mse=%.6f",
            self.epoch,
            metrics["val_loss"],
            metrics["command_accuracy"],
            metrics["param_mse"],
        )

        return metrics

    @torch.no_grad()
    def visualize_latent_space(self, epoch: int, max_samples: int = 1000) -> None:
        """Encode validation set and visualize latent space in 2D.

        Uses t-SNE (or UMAP if available) to reduce latent representations to
        2D and saves the scatter plot to the checkpoint directory.

        Args:
            epoch: Current epoch number, used for the filename.
            max_samples: Maximum number of samples to process (default 1000).
        """
        if self.val_dataloader is None:
            _log.warning("No validation dataloader; skipping latent visualization.")
            return

        if self.checkpoint_dir is None:
            _log.warning("No checkpoint_dir; skipping latent visualization.")
            return

        self.model.eval()
        all_mu = []
        all_labels = []
        sample_count = 0

        for batch in self.val_dataloader:
            if sample_count >= max_samples:
                break
            token_ids = batch["token_ids"].to(self.device)

            # Encode to get latent means
            if hasattr(self.model, "encode"):
                mu, _ = self.model.encode(token_ids)
            else:
                # Fallback: run full forward and extract mu
                output = self.model(token_ids)
                if isinstance(output, dict):
                    mu = output["mu"]
                else:
                    _, mu, _ = output

            all_mu.append(mu.cpu())
            sample_count += mu.shape[0]
            if "labels" in batch:
                all_labels.append(batch["labels"])

        all_mu_cat = torch.cat(all_mu, dim=0).numpy()

        if all_labels:
            all_labels_cat = torch.cat(all_labels, dim=0).numpy()
        else:
            all_labels_cat = None

        # Lazy import for dimensionality reduction and plotting
        try:
            from sklearn.manifold import TSNE

            perplexity = min(30, max(1, len(all_mu_cat) - 1))
            reducer = TSNE(
                n_components=2, random_state=42, perplexity=perplexity
            )
            reducer_name = "t-SNE"
        except ImportError:
            _log.warning("sklearn not available for latent visualization.")
            return

        _log.info(
            "Running %s on %d latent vectors...", reducer_name, len(all_mu_cat)
        )
        coords = reducer.fit_transform(all_mu_cat)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=all_labels_cat if all_labels_cat is not None else None,
                cmap="tab10" if all_labels_cat is not None else None,
                alpha=0.6,
                s=10,
            )
            if all_labels_cat is not None:
                plt.colorbar(scatter, ax=ax, label="Class")
            ax.set_title(f"VAE Latent Space ({reducer_name}) - Epoch {epoch}")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")

            save_path = self.checkpoint_dir / f"latent_space_epoch_{epoch}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            _log.info("Latent space visualization saved to %s", save_path)
        except ImportError:
            _log.warning(
                "matplotlib not available; skipping latent visualization plot."
            )

    def train(self, num_epochs: int, save_every: int = 1) -> None:
        """Train for multiple epochs with beta-VAE scheduling.

        Orchestrates the full training loop with:
        - Per-epoch training with beta warmup
        - Validation after each epoch
        - Latent space visualization
        - Checkpointing

        Args:
            num_epochs: Total number of epochs to train.
            save_every: Save a checkpoint every N epochs.
        """
        _log.info("Starting VAE training for %d epochs", num_epochs)

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train one epoch
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["total_loss"])
            self.history["recon_loss"].append(train_metrics["recon_loss"])
            self.history["kl_loss"].append(train_metrics["kl_loss"])
            self.history["beta"].append(train_metrics["beta"])

            print(
                f"\nEpoch {epoch}: Train Loss = {train_metrics['total_loss']:.4f} "
                f"(recon={train_metrics['recon_loss']:.4f}, "
                f"kl={train_metrics['kl_loss']:.4f}, "
                f"beta={train_metrics['beta']:.3f})"
            )

            # Validate
            if self.val_dataloader:
                val_metrics = self.validate()
                self.history["val_loss"].append(val_metrics.get("val_loss", 0.0))
                self.history["val_recon_loss"].append(
                    val_metrics.get("recon_loss", 0.0)
                )
                self.history["val_kl_loss"].append(
                    val_metrics.get("kl_loss", 0.0)
                )
                self.history["command_accuracy"].append(
                    val_metrics.get("command_accuracy", 0.0)
                )
                self.history["param_mse"].append(
                    val_metrics.get("param_mse", 0.0)
                )

                print(
                    f"Val Loss = {val_metrics['val_loss']:.4f}, "
                    f"Cmd Acc = {val_metrics['command_accuracy']:.4f}, "
                    f"Param MSE = {val_metrics['param_mse']:.6f}"
                )

                # Save best model
                val_loss = val_metrics["val_loss"]
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.checkpoint_dir:
                        self.save_checkpoint("best_model.pt")
                        print(f"Saved best model (val_loss={val_loss:.4f})")

                # Visualize latent space
                self.visualize_latent_space(epoch)

            # Save periodic checkpoint
            if self.checkpoint_dir and (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

        # Save final model and history
        if self.checkpoint_dir:
            self.save_checkpoint("final_model.pt")
            self.save_history()

        _log.info(
            "VAE training complete. Best val loss: %.4f", self.best_val_loss
        )

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint to disk.

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
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "kl_warmup_epochs": self.kl_warmup_epochs,
            "history": self.history,
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        _log.info("Checkpoint saved to %s", save_path)

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint from disk.

        Args:
            filename: Name of the checkpoint file to load.
        """
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint_dir set; cannot load checkpoint.")

        load_path = self.checkpoint_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.kl_warmup_epochs = checkpoint.get(
            "kl_warmup_epochs", self.kl_warmup_epochs
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

        history_path = self.checkpoint_dir / "vae_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        _log.info("Training history saved to %s", history_path)
