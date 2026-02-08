"""Training infrastructure for generative CAD models.

Provides specialized trainers for different generative architectures:

- VAETrainer: Variational Autoencoder with beta-warmup scheduling
- GANTrainer: Wasserstein GAN with gradient penalty for latent space generation
- DiffusionTrainer: Denoising diffusion with EMA model averaging
- StreamingVAETrainer: VAE trainer for HuggingFace IterableDatasets
- STEPNetTrainer: Unified high-level trainer supporting all architectures

Also provides logging utilities:
- WandbLogger: Weights & Biases experiment tracking
- TensorBoardLogger: Local TensorBoard logging
"""

from __future__ import annotations

from .vae_trainer import VAETrainer
from .gan_trainer import GANTrainer
from .diffusion_trainer import DiffusionTrainer
from .streaming_vae_trainer import StreamingVAETrainer
from .streaming_diffusion_trainer import StreamingDiffusionTrainer
from .streaming_gan_trainer import StreamingGANTrainer
from .logger import WandbLogger, TensorBoardLogger
from .unified_trainer import STEPNetTrainer, TrainingConfig, ModelType

__all__ = [
    "DiffusionTrainer",
    "GANTrainer",
    "ModelType",
    "STEPNetTrainer",
    "StreamingDiffusionTrainer",
    "StreamingGANTrainer",
    "StreamingVAETrainer",
    "TensorBoardLogger",
    "TrainingConfig",
    "VAETrainer",
    "WandbLogger",
]
